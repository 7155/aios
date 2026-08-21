from __future__ import annotations

import pytest
import torch

from aios.models.config import ModelConfig
from aios.models.minimind_ime import (
    BlockAttnResMixer,
    MiniMindIMEForCausalLM,
)


def tiny_config(**changes) -> ModelConfig:
    values = {
        "num_layers": 8,
        "num_qo_heads": 2,
        "num_kv_heads": 1,
        "head_dim": 4,
        "hidden_size": 8,
        "vocab_size": 32,
        "intermediate_size": 16,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "rope_theta": 100_000.0,
        "max_position_embeddings": 32,
        "tie_word_embeddings": True,
        "model_type": "minimind_ime_v3",
        "architecture_revision": "tiny_block_attnres",
        "residual_type": "block_attnres",
        "attnres_num_blocks": 2,
        "attnres_alpha": 1.0,
        "attnres_backend": "eager",
    }
    values.update(changes)
    return ModelConfig(**values)


def initialized_mixer(backend: str) -> BlockAttnResMixer:
    mixer = BlockAttnResMixer(8, 1e-6, backend)
    mixer.query.copy_(torch.tensor([0.3, -0.2, 0.1, 0.4, -0.5, 0.6, 0.2, -0.1]))
    mixer.key_norm.weight.copy_(
        torch.tensor([1.0, 0.9, 1.1, 0.8, 1.2, 1.0, 0.7, 1.3])
    )
    return mixer


@pytest.mark.parametrize("with_partial", [False, True])
def test_direct_attnres_matches_materialized_reference(with_partial: bool) -> None:
    generator = torch.Generator().manual_seed(7)
    bank = torch.randn(4, 5, 8, generator=generator)
    partial = torch.randn(5, 8, generator=generator) if with_partial else None
    reference = initialized_mixer("reference")
    eager = initialized_mixer("eager")
    actual = eager.forward(bank, partial)
    expected = reference.forward(bank, partial)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_zero_query_is_uniform_depth_average() -> None:
    bank = torch.arange(3 * 2 * 8, dtype=torch.float32).reshape(3, 2, 8)
    partial = torch.full((2, 8), 100.0)
    mixer = BlockAttnResMixer(8, 1e-6, "eager")
    mixer.query.zero_()
    mixer.key_norm.weight.fill_(1.0)
    expected = torch.cat((bank, partial.unsqueeze(0)), dim=0).mean(dim=0)
    torch.testing.assert_close(mixer.forward(bank, partial), expected)


def test_attnres_model_state_dict_preserves_checkpoint_names() -> None:
    with torch.device("meta"):
        model = MiniMindIMEForCausalLM(tiny_config())
    names = set(model.state_dict())
    assert "model.layers.0.attn_res_mixer.query" in names
    assert "model.layers.7.mlp_res_mixer.key_norm.weight" in names
    assert "model.final_attn_res_mixer.query" in names
    assert "lm_head.weight" not in names
    assert model.model._layers_per_block == 4
    assert model.model._num_blocks == 2


def test_attnres_config_rejects_partial_migration_and_bad_partition() -> None:
    with pytest.raises(ValueError, match="alpha=1.0"):
        tiny_config(attnres_alpha=0.5)
    with pytest.raises(ValueError, match="divisible"):
        tiny_config(num_layers=7)


@pytest.mark.parametrize("backend", ["compiled", "triton"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_optimized_attnres_matches_eager_bfloat16(backend: str) -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(11)
    bank = torch.randn(
        5, 8, 768, generator=generator, device=device, dtype=torch.bfloat16
    )
    partial = torch.randn(
        8, 768, generator=generator, device=device, dtype=torch.bfloat16
    )
    eager = BlockAttnResMixer(768, 1e-6, "eager")
    optimized = BlockAttnResMixer(768, 1e-6, backend)
    query = torch.randn(768, generator=generator, device=device, dtype=torch.bfloat16)
    weight = torch.randn(768, generator=generator, device=device, dtype=torch.bfloat16)
    for mixer in (eager, optimized):
        mixer.query = query.clone()
        mixer.key_norm.weight = weight.clone()
    expected = eager.forward(bank, partial)
    actual = optimized.forward(bank, partial)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
