from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch

from .core import Batch, Req, SamplingParams
from .engine.sample import Sampler, stateless_uniforms
from .llm import LLM


ASSISTANT_PATTERN = re.compile(r"以下是|首先|其次|综上所述|作为(?:一个)?AI|我是AI|我来回答")
REPEAT_PATTERN = re.compile(r"(.)\1{3,}")
FUNCTION_WORD_ENDINGS = (
    "的", "地", "得", "和", "与", "或", "但", "而", "在", "把", "被", "给",
    "对", "从", "向", "为", "再", "就", "会", "能", "如果", "因为", "所以",
    "但是", "然后",
)
NATURAL_REDUPLICATIONS = {
    "看看", "想想", "试试", "说说", "问问", "等等", "慢慢", "好好", "聊聊",
    "找找", "改改", "查查", "走走", "听听", "读读", "写写", "帮帮", "常常",
    "刚刚", "轻轻", "渐渐", "偷偷", "偏偏", "仅仅", "人人", "天天",
}


@dataclass(frozen=True)
class ImeGenerationConfig:
    display_candidates: int = 3
    sampling_attempts: int = 8
    max_sampling_attempts: int = 12
    refill_batch_size: int = 4
    max_new_tokens: int = 12
    min_new_tokens: int = 2
    max_candidate_chars: int = 16
    temperature: float = 0.35
    top_k: int = 50
    top_p: float = 0.9
    refill_temperature: float = 0.55
    refill_top_k: int = 80
    diversity_lambda: float = 0.35
    seed: int = 20260814

    def validate(self) -> None:
        if self.display_candidates < 1:
            raise ValueError("display_candidates must be positive")
        if self.sampling_attempts < self.display_candidates:
            raise ValueError("sampling_attempts must cover display_candidates")
        if self.max_sampling_attempts < self.sampling_attempts:
            raise ValueError("max_sampling_attempts must cover the initial attempts")
        if self.refill_batch_size < 1:
            raise ValueError("refill_batch_size must be positive")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if not 0 <= self.min_new_tokens <= self.max_new_tokens:
            raise ValueError("min_new_tokens must be within max_new_tokens")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.temperature <= 0.0 or self.refill_temperature <= 0.0:
            raise ValueError("IME candidate sampling temperatures must be positive")
        if self.top_k < 1 or self.refill_top_k < 1:
            raise ValueError("IME candidate top_k values must be positive")


@dataclass(frozen=True)
class ImeCandidate:
    text: str
    token_count: int
    average_logprob: float
    base_score: float
    selection_score: float
    stop_reason: str
    invalid_reasons: tuple[str, ...]


@dataclass(frozen=True)
class ImeCompletionResult:
    prefix: str
    generation_id: int
    cancelled: bool
    candidates: tuple[ImeCandidate, ...]
    raw_candidates: tuple[ImeCandidate, ...]
    prefix_tokens: int
    sampling_attempts: int
    generated_tokens: int
    latency_ms: float
    gpu_latency_ms: float
    unique_kv_pages: int
    reused_prefix_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ImeCandidateScore:
    text: str
    token_count: int
    average_logprob: float
    sum_logprob: float


@dataclass(frozen=True)
class ImeScoringResult:
    prefix: str
    candidates: tuple[ImeCandidateScore, ...]
    latency_ms: float
    gpu_latency_ms: float
    unique_kv_pages: int
    reused_prefix_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def token_longest_common_prefix(left: list[int], right: list[int]) -> int:
    length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        length += 1
    return length


class CancellationToken:
    def __init__(self, generation_id: int) -> None:
        self.generation_id = generation_id
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


def normalize_candidate(text: str) -> str:
    return re.sub(r"\s+", "", text.replace("\ufffd", "")).strip()


def candidate_key(text: str) -> str:
    return normalize_candidate(text).rstrip("，。！？；：,.!?;:").casefold()


def repeated_ngram(text: str) -> bool:
    compact = normalize_candidate(text)
    if REPEAT_PATTERN.search(compact):
        return True
    for width in range(2, 5):
        for start in range(max(0, len(compact) - width * 3 + 1)):
            unit = compact[start : start + width]
            if unit and compact[start : start + width * 3] == unit * 3:
                return True
    return False


def invalid_reasons(
    text: str,
    config: ImeGenerationConfig,
    prefix: str = "",
) -> tuple[str, ...]:
    reasons: list[str] = []
    compact = normalize_candidate(text)
    if not compact:
        return ("empty",)
    if len(compact) < 2:
        reasons.append("too_short")
    if config.max_candidate_chars > 0 and len(compact) > config.max_candidate_chars:
        reasons.append("too_long")
    if ASSISTANT_PATTERN.search(compact):
        reasons.append("assistant_template")
    if repeated_ngram(compact):
        reasons.append("repeated_ngram")
    if compact.rstrip("，。！？；：,.!?;:").endswith(FUNCTION_WORD_ENDINGS):
        reasons.append("unfinished_fragment")
    if (
        prefix
        and compact
        and prefix[-1] == compact[0]
        and "\u3400" <= compact[0] <= "\u9fff"
    ):
        reasons.append("boundary_repeat")
    return tuple(reasons)


def soft_penalty(text: str) -> float:
    compact = normalize_candidate(text)
    penalty = 0.0
    for match in re.finditer(r"([\u3400-\u9fff])\1", compact):
        if match.group(0) not in NATURAL_REDUPLICATIONS:
            penalty += 0.45
    if len(compact) <= 3:
        penalty += 0.08
    return penalty


def ngram_similarity(left: str, right: str, width: int = 2) -> float:
    def grams(text: str) -> set[str]:
        compact = candidate_key(text)
        if len(compact) < width:
            return {compact} if compact else set()
        return {compact[index : index + width] for index in range(len(compact) - width + 1)}

    left_grams = grams(left)
    right_grams = grams(right)
    if not left_grams or not right_grams:
        return 0.0
    return len(left_grams & right_grams) / len(left_grams | right_grams)


def select_top_candidates(
    candidates: list[ImeCandidate],
    count: int,
    diversity_lambda: float,
) -> list[ImeCandidate]:
    best_by_key: dict[str, ImeCandidate] = {}
    for candidate in candidates:
        if candidate.invalid_reasons or not math.isfinite(candidate.base_score):
            continue
        key = candidate_key(candidate.text)
        previous = best_by_key.get(key)
        if previous is None or candidate.base_score > previous.base_score:
            best_by_key[key] = candidate

    remaining = list(best_by_key.values())
    selected: list[ImeCandidate] = []
    while remaining and len(selected) < count:
        def selection_score(candidate: ImeCandidate) -> float:
            similarity = max(
                (ngram_similarity(candidate.text, item.text) for item in selected),
                default=0.0,
            )
            return candidate.base_score - diversity_lambda * similarity

        chosen = max(remaining, key=selection_score)
        chosen_selection_score = selection_score(chosen)
        remaining.remove(chosen)
        selected.append(ImeCandidate(
            text=chosen.text,
            token_count=chosen.token_count,
            average_logprob=chosen.average_logprob,
            base_score=chosen.base_score,
            selection_score=chosen_selection_score,
            stop_reason=chosen.stop_reason,
            invalid_reasons=chosen.invalid_reasons,
        ))
    return selected


class ImeCompletionEngine:
    """Single-user, same-prefix candidate-group runtime for MiniMind-IME."""

    def __init__(self, llm: LLM) -> None:
        self.llm = llm
        self._generation_id = 0
        self._active_token: CancellationToken | None = None
        self._generation_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._prefix_token_ids: list[int] = []
        self._prefix_pages: torch.Tensor | None = None
        self._prefix_logits: torch.Tensor | None = None
        punctuation = "。！？；!?;"
        self._terminal_token_ids = {
            ids[0]
            for symbol in punctuation
            if len(ids := llm.tokenizer.encode(symbol, add_special_tokens=False)) == 1
        }
        self._terminal_token_tensor = torch.tensor(
            sorted(self._terminal_token_ids), dtype=torch.long, device=llm.device
        )

    def new_generation(self) -> CancellationToken:
        with self._generation_lock:
            if self._active_token is not None:
                self._active_token.cancel()
            self._generation_id += 1
            self._active_token = CancellationToken(self._generation_id)
            return self._active_token

    def reset_prefix_cache(self) -> None:
        """Release the persistent single-user prefix KV pages."""
        with self._run_lock:
            self._reset_prefix_cache_unlocked()

    def _reset_prefix_cache_unlocked(self) -> None:
        if self._prefix_pages is not None and len(self._prefix_pages):
            self.llm.cache_manager._free(self._prefix_pages)
        self._prefix_token_ids = []
        self._prefix_pages = None
        self._prefix_logits = None

    @torch.inference_mode()
    def complete(
        self,
        prefix: str,
        config: ImeGenerationConfig | None = None,
        cancellation: CancellationToken | None = None,
    ) -> ImeCompletionResult:
        cancellation = cancellation or self.new_generation()
        with self._run_lock:
            return self._complete_locked(prefix, config, cancellation)

    def _complete_locked(
        self,
        prefix: str,
        config: ImeGenerationConfig | None,
        cancellation: CancellationToken,
    ) -> ImeCompletionResult:
        wall_started = time.perf_counter()
        config = config or ImeGenerationConfig()
        config.validate()
        token_ids = self.llm.tokenizer.encode(prefix, add_special_tokens=False)
        if self.llm.tokenizer.bos_token_id is not None:
            token_ids = [self.llm.tokenizer.bos_token_id, *token_ids]
        if not token_ids:
            raise ValueError("prefix produced no input tokens")
        if cancellation.cancelled:
            return ImeCompletionResult(
                prefix=prefix,
                generation_id=cancellation.generation_id,
                cancelled=True,
                candidates=(),
                raw_candidates=(),
                prefix_tokens=len(token_ids),
                sampling_attempts=0,
                generated_tokens=0,
                latency_ms=(time.perf_counter() - wall_started) * 1000.0,
                gpu_latency_ms=0.0,
                unique_kv_pages=0,
                reused_prefix_tokens=0,
            )
        max_total_len = len(token_ids) + config.max_new_tokens
        if max_total_len > self.llm.config.max_position_embeddings:
            raise ValueError(
                f"prefix + output exceeds context limit: {max_total_len} > "
                f"{self.llm.config.max_position_embeddings}"
            )
        required_pages = len(token_ids) + config.sampling_attempts * max(
            0, config.max_new_tokens - 1
        )
        if required_pages > self.llm.num_pages:
            raise ValueError(
                "IME CandidateGroup KV budget is too small: "
                f"need up to {required_pages} pages, configured {self.llm.num_pages}; "
                "increase kv_cache_max_tokens or shorten the context/output"
            )

        device = self.llm.device
        max_attempts = config.max_sampling_attempts
        page_table = torch.zeros(
            (max_attempts, max_total_len), dtype=torch.int32, device=device
        )
        self.llm.ctx.page_table = page_table
        allocated_pages: list[torch.Tensor] = []
        started = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)
        started.record()
        cancelled = False
        peak_unique_pages = 0

        try:
            prefix_logits, prefix_pages, reused_prefix_tokens = self._prepare_prefix(
                token_ids, page_table
            )
            peak_unique_pages = len(prefix_pages)
            raw_candidates: list[ImeCandidate] = []
            actual_attempts = 0
            generated_tokens = 0
            while actual_attempts < max_attempts:
                attempts = (
                    config.sampling_attempts
                    if actual_attempts == 0
                    else min(config.refill_batch_size, max_attempts - actual_attempts)
                )
                candidates, pages, batch_generated, batch_cancelled = self._generate_branch_batch(
                    prefix=prefix,
                    prefix_logits=prefix_logits,
                    page_table=page_table,
                    row_offset=actual_attempts,
                    attempts=attempts,
                    prefix_len=len(token_ids),
                    config=config,
                    seed=config.seed + actual_attempts * 100_003,
                    temperature=(
                        config.temperature
                        if actual_attempts == 0
                        else config.refill_temperature
                    ),
                    top_k=(
                        config.top_k
                        if actual_attempts == 0
                        else config.refill_top_k
                    ),
                    cancellation=cancellation,
                )
                raw_candidates.extend(candidates)
                allocated_pages.extend(pages)
                peak_unique_pages = max(
                    peak_unique_pages,
                    len(prefix_pages) + sum(len(item) for item in allocated_pages),
                )
                actual_attempts += attempts
                generated_tokens += batch_generated
                cancelled |= batch_cancelled
                # Candidate text and scores are now materialized; branch KV is
                # not needed by a later adaptive refill batch.
                if allocated_pages:
                    self.llm.cache_manager._free(torch.cat(allocated_pages))
                    allocated_pages.clear()
                selected = select_top_candidates(
                    raw_candidates,
                    config.display_candidates,
                    config.diversity_lambda,
                )
                if cancelled or len(selected) >= config.display_candidates:
                    break

            ended.record()
            ended.synchronize()
            gpu_latency_ms = started.elapsed_time(ended)
            selected = select_top_candidates(
                raw_candidates,
                config.display_candidates,
                config.diversity_lambda,
            )
            latency_ms = (time.perf_counter() - wall_started) * 1000.0
            return ImeCompletionResult(
                prefix=prefix,
                generation_id=cancellation.generation_id,
                cancelled=cancelled,
                candidates=tuple(selected),
                raw_candidates=tuple(raw_candidates),
                prefix_tokens=len(token_ids),
                sampling_attempts=actual_attempts,
                generated_tokens=generated_tokens,
                latency_ms=latency_ms,
                gpu_latency_ms=gpu_latency_ms,
                unique_kv_pages=peak_unique_pages,
                reused_prefix_tokens=reused_prefix_tokens,
            )
        finally:
            if allocated_pages:
                self.llm.cache_manager._free(torch.cat(allocated_pages))

    @torch.inference_mode()
    def score_candidates(
        self,
        prefix: str,
        candidates: list[str],
        mode: Literal["stable", "shared_decode"] = "stable",
    ) -> ImeScoringResult:
        """Teacher-force and rank a candidate group by raw average LM score.

        ``stable`` evaluates complete sequences in one varlen prefill. It costs
        more KV work but avoids BF16 near-tie flips between prefill and decode
        kernels, so it is the default for final same-pinyin/context reranking.
        ``shared_decode`` reuses one prefix KV and is useful for diagnostics.
        """
        if not candidates:
            raise ValueError("candidates must not be empty")
        with self._run_lock:
            if mode == "stable":
                return self._score_candidates_stable_locked(prefix, candidates)
            if mode == "shared_decode":
                return self._score_candidates_locked(prefix, candidates)
            raise ValueError(f"Unsupported scoring mode: {mode}")

    def _score_candidates_stable_locked(
        self,
        prefix: str,
        candidates: list[str],
    ) -> ImeScoringResult:
        wall_started = time.perf_counter()
        prefix_ids = self.llm.tokenizer.encode(prefix, add_special_tokens=False)
        if self.llm.tokenizer.bos_token_id is not None:
            prefix_ids = [self.llm.tokenizer.bos_token_id, *prefix_ids]
        candidate_ids = [
            self.llm.tokenizer.encode(text, add_special_tokens=False)
            for text in candidates
        ]
        sequences = [[*prefix_ids, *ids] for ids in candidate_ids]
        max_total_len = max(map(len, sequences))
        if max_total_len > self.llm.config.max_position_embeddings:
            raise ValueError(
                f"prefix + candidate exceeds context limit: {max_total_len} > "
                f"{self.llm.config.max_position_embeddings}"
            )
        self._reset_prefix_cache_unlocked()
        required_pages = sum(map(len, sequences))
        if required_pages > self.llm.num_pages:
            raise ValueError(
                "Stable candidate scoring KV budget is too small: "
                f"need {required_pages} pages, configured {self.llm.num_pages}"
            )

        device = self.llm.device
        page_table = torch.zeros(
            (len(candidates), max_total_len), dtype=torch.int32, device=device
        )
        self.llm.ctx.page_table = page_table
        allocated_pages: list[torch.Tensor] = []
        reqs: list[Req] = []
        started = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)
        started.record()
        try:
            for row, sequence in enumerate(sequences):
                pages = self.llm.cache_manager.allocate(len(sequence))
                allocated_pages.append(pages)
                page_table[row, : len(sequence)] = pages
                reqs.append(Req(
                    input_ids=torch.empty(len(sequence), dtype=torch.long),
                    cached_len=0,
                    output_len=1,
                    uid=row,
                    sampling_params=SamplingParams(max_tokens=1),
                    table_idx=row,
                ))

            batch = Batch(reqs=reqs, phase="prefill", return_all_logits=True)
            batch.input_ids = torch.tensor(
                [token for sequence in sequences for token in sequence],
                dtype=torch.long,
                device=device,
            )
            batch.positions = torch.cat([
                torch.arange(len(sequence), dtype=torch.int32, device=device)
                for sequence in sequences
            ])
            batch.out_loc = torch.cat(allocated_pages)
            self.llm.model.attn_backend.prepare_metadata(batch)
            with self.llm.ctx.forward_batch(batch):
                logits = self.llm.model.forward()
            log_probs = torch.log_softmax(logits.float(), dim=-1)

            scored: list[ImeCandidateScore] = []
            offset = 0
            for text, sequence, ids in zip(candidates, sequences, candidate_ids):
                if ids:
                    positions = torch.arange(
                        offset + len(prefix_ids) - 1,
                        offset + len(prefix_ids) - 1 + len(ids),
                        dtype=torch.long,
                        device=device,
                    )
                    targets = torch.tensor(ids, dtype=torch.long, device=device)
                    token_log_probs = log_probs[positions, targets]
                    score_sum = float(token_log_probs.sum())
                    average = float(token_log_probs.mean())
                else:
                    score_sum = float("-inf")
                    average = float("-inf")
                scored.append(ImeCandidateScore(
                    text=text,
                    token_count=len(ids),
                    average_logprob=average,
                    sum_logprob=score_sum,
                ))
                offset += len(sequence)

            ended.record()
            ended.synchronize()
            scored.sort(key=lambda item: item.average_logprob, reverse=True)
            return ImeScoringResult(
                prefix=prefix,
                candidates=tuple(scored),
                latency_ms=(time.perf_counter() - wall_started) * 1000.0,
                gpu_latency_ms=started.elapsed_time(ended),
                unique_kv_pages=sum(len(pages) for pages in allocated_pages),
                reused_prefix_tokens=0,
            )
        finally:
            if allocated_pages:
                self.llm.cache_manager._free(torch.cat(allocated_pages))

    def _score_candidates_locked(
        self,
        prefix: str,
        candidates: list[str],
    ) -> ImeScoringResult:
        wall_started = time.perf_counter()
        prefix_ids = self.llm.tokenizer.encode(prefix, add_special_tokens=False)
        if self.llm.tokenizer.bos_token_id is not None:
            prefix_ids = [self.llm.tokenizer.bos_token_id, *prefix_ids]
        candidate_ids = [
            self.llm.tokenizer.encode(text, add_special_tokens=False)
            for text in candidates
        ]
        max_candidate_tokens = max((len(ids) for ids in candidate_ids), default=0)
        max_total_len = len(prefix_ids) + max_candidate_tokens
        if max_total_len > self.llm.config.max_position_embeddings:
            raise ValueError(
                f"prefix + candidate exceeds context limit: {max_total_len} > "
                f"{self.llm.config.max_position_embeddings}"
            )
        required_pages = len(prefix_ids) + len(candidates) * max(
            0, max_candidate_tokens - 1
        )
        if required_pages > self.llm.num_pages:
            raise ValueError(
                "Shared-decode scoring KV budget is too small: "
                f"need up to {required_pages} pages, configured {self.llm.num_pages}"
            )

        device = self.llm.device
        page_table = torch.zeros(
            (len(candidates), max_total_len), dtype=torch.int32, device=device
        )
        self.llm.ctx.page_table = page_table
        allocated_pages: list[torch.Tensor] = []
        started = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)
        started.record()
        try:
            prefix_logits, prefix_pages, reused_prefix_tokens = self._prepare_prefix(
                prefix_ids, page_table
            )
            score_sums = torch.zeros(len(candidates), dtype=torch.float32, device=device)
            counts = torch.tensor(
                [len(ids) for ids in candidate_ids], dtype=torch.long, device=device
            )
            first_rows = [index for index, ids in enumerate(candidate_ids) if ids]
            if first_rows:
                row_tensor = torch.tensor(first_rows, dtype=torch.long, device=device)
                targets = torch.tensor(
                    [candidate_ids[index][0] for index in first_rows],
                    dtype=torch.long,
                    device=device,
                )
                first_log_probs = torch.log_softmax(
                    prefix_logits.float(), dim=-1
                ).expand(len(first_rows), -1)
                score_sums[row_tensor] += first_log_probs.gather(
                    1, targets.unsqueeze(1)
                ).squeeze(1)

            for step in range(1, max_candidate_tokens):
                active_rows = [
                    index for index, ids in enumerate(candidate_ids) if len(ids) > step
                ]
                if not active_rows:
                    break
                row_tensor = torch.tensor(active_rows, dtype=torch.long, device=device)
                previous_tokens = torch.tensor(
                    [candidate_ids[index][step - 1] for index in active_rows],
                    dtype=torch.long,
                    device=device,
                )
                targets = torch.tensor(
                    [candidate_ids[index][step] for index in active_rows],
                    dtype=torch.long,
                    device=device,
                )
                logits, pages = self._decode_step(
                    previous_tokens,
                    page_table,
                    cached_len=len(prefix_ids) + step - 1,
                    row_indices=row_tensor,
                )
                allocated_pages.append(pages)
                token_log_probs = torch.log_softmax(logits.float(), dim=-1).gather(
                    1, targets.unsqueeze(1)
                ).squeeze(1)
                score_sums[row_tensor] += token_log_probs

            ended.record()
            ended.synchronize()
            sums = score_sums.cpu().tolist()
            count_values = counts.cpu().tolist()
            scored = [
                ImeCandidateScore(
                    text=text,
                    token_count=count,
                    average_logprob=(total / count if count else float("-inf")),
                    sum_logprob=(total if count else float("-inf")),
                )
                for text, count, total in zip(candidates, count_values, sums)
            ]
            scored.sort(key=lambda item: item.average_logprob, reverse=True)
            return ImeScoringResult(
                prefix=prefix,
                candidates=tuple(scored),
                latency_ms=(time.perf_counter() - wall_started) * 1000.0,
                gpu_latency_ms=started.elapsed_time(ended),
                unique_kv_pages=len(prefix_pages) + sum(
                    len(pages) for pages in allocated_pages
                ),
                reused_prefix_tokens=reused_prefix_tokens,
            )
        finally:
            if allocated_pages:
                self.llm.cache_manager._free(torch.cat(allocated_pages))

    def _generate_branch_batch(
        self,
        *,
        prefix: str,
        prefix_logits: torch.Tensor,
        page_table: torch.Tensor,
        row_offset: int,
        attempts: int,
        prefix_len: int,
        config: ImeGenerationConfig,
        seed: int,
        temperature: float,
        top_k: int,
        cancellation: CancellationToken,
    ) -> tuple[list[ImeCandidate], list[torch.Tensor], int, bool]:
        device = self.llm.device
        logits = prefix_logits.expand(attempts, -1)
        output_ids = torch.full(
            (attempts, config.max_new_tokens),
            self.llm.tokenizer.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        counts = torch.zeros(attempts, dtype=torch.long, device=device)
        logprob_sums = torch.zeros(attempts, dtype=torch.float32, device=device)
        stop_codes = torch.zeros(attempts, dtype=torch.int8, device=device)
        candidate_uniforms = stateless_uniforms(
            [seed + index for index in range(attempts)],
            config.max_new_tokens,
            device,
        )
        sampler = Sampler(SamplingParams(
            temperature=temperature,
            top_k=top_k,
            top_p=config.top_p,
            max_tokens=config.max_new_tokens,
        ))
        row_indices = torch.arange(
            row_offset, row_offset + attempts, dtype=torch.long, device=device
        )
        allocated_pages: list[torch.Tensor] = []
        cancelled = False
        active_local = torch.arange(attempts, dtype=torch.long, device=device)
        minimum_length_blocked_ids = tuple(sorted({
            *self._terminal_token_ids,
            *(
                (self.llm.tokenizer.eos_token_id,)
                if self.llm.tokenizer.eos_token_id is not None
                else ()
            ),
        }))

        for step in range(config.max_new_tokens):
            if cancellation.cancelled:
                cancelled = True
                break
            token, raw_logprob = sampler.sample_with_logprobs(
                logits,
                uniforms=candidate_uniforms[active_local, step],
                forbidden_token_ids=(
                    minimum_length_blocked_ids
                    if step < config.min_new_tokens
                    else ()
                ),
            )
            token = token.squeeze(-1)
            raw_logprob = raw_logprob.squeeze(-1)
            is_eos = token.eq(self.llm.tokenizer.eos_token_id)
            emitted = ~is_eos
            output_ids[active_local, step] = torch.where(
                emitted,
                token,
                torch.full_like(token, self.llm.tokenizer.pad_token_id),
            )
            counts[active_local] += emitted.long()
            logprob_sums[active_local] += torch.where(
                emitted, raw_logprob, torch.zeros_like(raw_logprob)
            )

            terminal = torch.zeros_like(emitted)
            if self._terminal_token_tensor.numel():
                terminal = emitted & torch.isin(token, self._terminal_token_tensor)
            stop_codes[active_local] = torch.where(
                is_eos,
                torch.ones_like(stop_codes[active_local]),
                stop_codes[active_local],
            )
            stop_codes[active_local] = torch.where(
                terminal,
                torch.full_like(stop_codes[active_local], 2),
                stop_codes[active_local],
            )
            branch_finished = is_eos | terminal
            if step + 1 == config.max_new_tokens:
                break

            survivors = ~branch_finished
            active_local = active_local[survivors]
            if active_local.numel() == 0:
                break
            decode_tokens = token[survivors]
            logits, pages = self._decode_step(
                decode_tokens,
                page_table,
                cached_len=prefix_len + step,
                row_indices=row_indices[active_local],
            )
            allocated_pages.append(pages)

        output_cpu = output_ids.cpu()
        counts_cpu = counts.cpu().tolist()
        score_sums = logprob_sums.cpu().tolist()
        stop_cpu = stop_codes.cpu().tolist()
        decoded = self.llm.tokenizer.batch_decode(
            [output_cpu[row, : counts_cpu[row]].tolist() for row in range(attempts)],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        candidates: list[ImeCandidate] = []
        for row, text in enumerate(decoded):
            normalized = normalize_candidate(text)
            reasons = invalid_reasons(normalized, config, prefix)
            average = score_sums[row] / counts_cpu[row] if counts_cpu[row] else float("-inf")
            score = average - soft_penalty(normalized) if not reasons else float("-inf")
            stop_reason = {1: "eos", 2: "terminal_punctuation"}.get(
                stop_cpu[row], "max_new_tokens"
            )
            candidates.append(ImeCandidate(
                text=normalized,
                token_count=counts_cpu[row],
                average_logprob=average,
                base_score=score,
                selection_score=score,
                stop_reason=stop_reason,
                invalid_reasons=reasons,
            ))
        return candidates, allocated_pages, sum(counts_cpu), cancelled

    def _prepare_prefix(
        self,
        token_ids: list[int],
        page_table: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Build or incrementally extend the persistent latest-prefix KV."""
        old_ids = self._prefix_token_ids
        old_pages = self._prefix_pages
        reused = token_longest_common_prefix(old_ids, token_ids)

        if (
            old_pages is not None
            and self._prefix_logits is not None
            and reused == len(old_ids) == len(token_ids)
        ):
            page_table[:, : len(token_ids)] = old_pages.unsqueeze(0)
            return self._prefix_logits, old_pages, reused

        # If the user only backspaced, the old cache does not retain the logits
        # at the now-final token. Re-prefill that short prefix for exact scores.
        if old_ids and reused == len(token_ids) < len(old_ids):
            reused = 0

        if old_pages is not None:
            if reused:
                kept_pages = old_pages[:reused]
                released_pages = old_pages[reused:]
            else:
                kept_pages = old_pages[:0]
                released_pages = old_pages
            if len(released_pages):
                self.llm.cache_manager._free(released_pages)
        else:
            kept_pages = torch.empty(
                0, dtype=torch.int32, device=self.llm.device
            )

        extension_pages = self.llm.cache_manager.allocate(len(token_ids) - reused)
        prefix_pages = torch.cat([kept_pages, extension_pages])
        page_table[:, : len(token_ids)] = prefix_pages.unsqueeze(0)
        if reused:
            logits = self._extend_prefill(
                token_ids=token_ids,
                cached_len=reused,
                extension_pages=extension_pages,
            )
        else:
            logits = self._prefill(token_ids, prefix_pages, page_table)

        self._prefix_token_ids = list(token_ids)
        self._prefix_pages = prefix_pages
        self._prefix_logits = logits
        return logits, prefix_pages, reused

    def _extend_prefill(
        self,
        *,
        token_ids: list[int],
        cached_len: int,
        extension_pages: torch.Tensor,
    ) -> torch.Tensor:
        req = Req(
            input_ids=torch.empty(len(token_ids), dtype=torch.long),
            cached_len=cached_len,
            output_len=1,
            uid=0,
            sampling_params=SamplingParams(max_tokens=1),
            table_idx=0,
        )
        batch = Batch(reqs=[req], phase="prefill")
        batch.input_ids = torch.tensor(
            token_ids[cached_len:], dtype=torch.long, device=self.llm.device
        )
        batch.positions = torch.arange(
            cached_len, len(token_ids), dtype=torch.int32, device=self.llm.device
        )
        batch.out_loc = extension_pages
        self.llm.model.attn_backend.prepare_metadata(batch)
        with self.llm.ctx.forward_batch(batch):
            return self.llm.model.forward()

    def _prefill(
        self,
        token_ids: list[int],
        prefix_pages: torch.Tensor,
        page_table: torch.Tensor,
    ) -> torch.Tensor:
        req = Req(
            input_ids=torch.empty(len(token_ids), dtype=torch.long),
            cached_len=0,
            output_len=1,
            uid=0,
            sampling_params=SamplingParams(max_tokens=1),
            table_idx=0,
        )
        batch = Batch(reqs=[req], phase="prefill")
        batch.input_ids = torch.tensor(token_ids, dtype=torch.long, device=self.llm.device)
        batch.positions = torch.arange(len(token_ids), dtype=torch.int32, device=self.llm.device)
        batch.out_loc = prefix_pages
        self.llm.model.attn_backend.prepare_metadata(batch)
        with self.llm.ctx.forward_batch(batch):
            return self.llm.model.forward()

    def _decode_step(
        self,
        token_ids: torch.Tensor,
        page_table: torch.Tensor,
        *,
        cached_len: int,
        row_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attempts = len(row_indices)
        pages = self.llm.cache_manager.allocate(attempts)
        page_table[row_indices, cached_len] = pages
        row_ids = row_indices.tolist()
        reqs = [
            Req(
                input_ids=torch.empty(cached_len + 1, dtype=torch.long),
                cached_len=cached_len,
                output_len=1,
                uid=row,
                sampling_params=SamplingParams(max_tokens=1),
                table_idx=row,
            )
            for row in row_ids
        ]
        batch = Batch(reqs=reqs, phase="decode")
        batch.input_ids = token_ids
        batch.positions = torch.full(
            (attempts,), cached_len, dtype=torch.int32, device=self.llm.device
        )
        batch.out_loc = pages
        self.llm.model.attn_backend.prepare_metadata(batch)
        with self.llm.ctx.forward_batch(batch):
            logits = self.llm.model.forward()
        return logits, pages
