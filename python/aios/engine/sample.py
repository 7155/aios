from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..core import SamplingParams


def _splitmix64_uniform(seed: int, step: int) -> float:
    """Return one deterministic U(0, 1) value for a candidate/token step."""

    mask = (1 << 64) - 1
    value = (int(seed) + (int(step) + 1) * 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    value ^= value >> 31
    return (((value >> 11) & ((1 << 53) - 1)) + 0.5) / float(1 << 53)


def stateless_uniforms(
    candidate_seeds: Sequence[int],
    steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Precompute independent random streams with shape ``(rows, steps)``.

    Candidate randomness is keyed by ``(candidate_seed, token_step)`` instead
    of the current decode-row order. If an earlier candidate reaches EOS and
    active rows compact, every surviving candidate therefore keeps the same
    continuation it would have produced in the full batch.
    """

    if steps < 1:
        raise ValueError("steps must be positive")
    values = [
        [_splitmix64_uniform(seed, step) for step in range(steps)]
        for seed in candidate_seeds
    ]
    host = torch.tensor(values, dtype=torch.float32)
    if device.type == "cuda":
        host = host.pin_memory()
    return host.to(device, non_blocking=device.type == "cuda")


@dataclass
class Sampler:
    """Independent sampler module (aligned with mini-sglang's engine/sample.py)."""

    sampling_params: SamplingParams

    def sample_with_logprobs(
        self,
        logits: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        uniforms: torch.Tensor | None = None,
        forbidden_token_ids: Sequence[int] = (),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample token ids and return their raw model log probabilities.

        Sampling uses temperature/top-k/top-p, while the returned score is
        gathered from the unmodified model distribution. This keeps candidate
        ranking independent from the exploration policy. ``uniforms`` enables
        request-local stateless sampling; ``forbidden_token_ids`` only changes
        the sampling copy, never the returned raw model log probability.
        """
        if generator is not None and uniforms is not None:
            raise ValueError("generator and uniforms are mutually exclusive")

        raw_log_probs = F.log_softmax(logits.float(), dim=-1)
        sampling_logits = logits
        if forbidden_token_ids:
            blocked = sorted({int(token_id) for token_id in forbidden_token_ids})
            invalid = [
                token_id
                for token_id in blocked
                if token_id < 0 or token_id >= logits.size(-1)
            ]
            if invalid:
                raise ValueError(
                    f"forbidden token ids are outside the vocabulary: {invalid}"
                )
            sampling_logits = logits.clone()
            sampling_logits[..., blocked] = -torch.inf

        if self.sampling_params.is_greedy:
            token_ids = sampling_logits.argmax(dim=-1, keepdim=True)
            return token_ids, raw_log_probs.gather(-1, token_ids)

        scaled = sampling_logits.float() / self.sampling_params.temperature
        if self.sampling_params.top_k > 0:
            candidate_logits, candidate_ids = torch.topk(
                scaled,
                min(self.sampling_params.top_k, scaled.size(-1)),
            )
        elif self.sampling_params.top_p >= 1.0:
            candidate_logits = scaled
            candidate_ids = torch.arange(
                scaled.size(-1), device=scaled.device
            ).expand_as(scaled)
        else:
            candidate_logits, candidate_ids = torch.sort(scaled, descending=True)

        probabilities = F.softmax(candidate_logits, dim=-1)
        if self.sampling_params.top_p < 1.0:
            cumulative = probabilities.cumsum(dim=-1)
            remove = cumulative > self.sampling_params.top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            candidate_logits = candidate_logits.masked_fill(remove, float("-inf"))
            probabilities = F.softmax(candidate_logits, dim=-1)

        if uniforms is None:
            sampled_index = torch.multinomial(
                probabilities,
                num_samples=1,
                generator=generator,
            )
        else:
            if uniforms.shape != probabilities.shape[:-1]:
                raise ValueError(
                    "uniforms must match the sampling batch dimensions: "
                    f"expected {probabilities.shape[:-1]}, got {uniforms.shape}"
                )
            uniforms = uniforms.to(
                device=probabilities.device,
                dtype=probabilities.dtype,
            ).unsqueeze(-1)
            sampled_index = (
                probabilities.cumsum(dim=-1) < uniforms
            ).sum(dim=-1, keepdim=True)
            sampled_index.clamp_max_(probabilities.size(-1) - 1)
        token_ids = candidate_ids.gather(-1, sampled_index)
        return token_ids, raw_log_probs.gather(-1, token_ids)

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample next token from logits. logits shape: (batch, vocab_size)"""
        return self.sample_with_logprobs(logits)[0]
