from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_minimind_ime.py"
SPEC = importlib.util.spec_from_file_location("export_minimind_ime", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)


def minimal_state(
    *, tied: bool = True, block_attnres: bool = False
) -> dict[str, torch.Tensor]:
    embedding = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    state = {
        "model.embed_tokens.weight": embedding,
        "model.norm.weight": torch.ones(4),
        "model.layers.0.input_layernorm.weight": torch.ones(4),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(4),
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(4, 4),
        "model.layers.0.self_attn.k_proj.weight": torch.zeros(2, 4),
        "model.layers.0.self_attn.v_proj.weight": torch.zeros(2, 4),
        "model.layers.0.self_attn.o_proj.weight": torch.zeros(4, 4),
        "model.layers.0.self_attn.q_norm.weight": torch.ones(2),
        "model.layers.0.self_attn.k_norm.weight": torch.ones(2),
        "model.layers.0.mlp.gate_proj.weight": torch.zeros(8, 4),
        "model.layers.0.mlp.up_proj.weight": torch.zeros(8, 4),
        "model.layers.0.mlp.down_proj.weight": torch.zeros(4, 8),
    }
    if tied:
        state["lm_head.weight"] = embedding.clone()
    if block_attnres:
        for mixer in ("attn_res_mixer", "mlp_res_mixer"):
            state[f"model.layers.0.{mixer}.query"] = torch.zeros(4)
            state[f"model.layers.0.{mixer}.key_norm.weight"] = torch.ones(4)
        state["model.final_attn_res_mixer.query"] = torch.zeros(4)
        state["model.final_attn_res_mixer.key_norm.weight"] = torch.ones(4)
    return state


def config(**changes) -> dict:
    value = {
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "tie_word_embeddings": True,
        "hidden_act": "silu",
    }
    value.update(changes)
    return value


def test_export_rejects_false_tied_embedding_claim() -> None:
    state = minimal_state()
    state["lm_head.weight"] += 1
    with pytest.raises(ValueError, match="weights differ"):
        EXPORTER.build_deployment_config(state, config(), {}, "bfloat16")


def test_export_rejects_missing_untied_lm_head() -> None:
    with pytest.raises(KeyError, match="refusing to tie"):
        EXPORTER.build_deployment_config(
            minimal_state(tied=False),
            config(tie_word_embeddings=False),
            {},
            "bfloat16",
        )


def test_export_rejects_moe_and_unsupported_activation() -> None:
    with pytest.raises(ValueError, match="dense"):
        EXPORTER.build_deployment_config(
            minimal_state(), config(use_moe=True), {}, "bfloat16"
        )
    with pytest.raises(ValueError, match="SiLU"):
        EXPORTER.build_deployment_config(
            minimal_state(), config(hidden_act="gelu"), {}, "bfloat16"
        )


def test_export_casts_to_bfloat16_and_omits_only_verified_tied_head() -> None:
    state = minimal_state()
    deployment = EXPORTER.build_deployment_config(
        state, config(), {}, "bfloat16"
    )
    exported = EXPORTER.prepare_export_state(state, deployment, torch.bfloat16)
    assert "lm_head.weight" not in exported
    assert {tensor.dtype for tensor in exported.values()} == {torch.bfloat16}


def test_mtp_namespace_detection_covers_training_checkpoint_names() -> None:
    assert EXPORTER._is_mtp_tensor("mtp_module.block.mlp.up_proj.weight")
    assert EXPORTER._is_mtp_tensor("model.mtp_head.weight")
    assert not EXPORTER._is_mtp_tensor("model.layers.0.mlp.up_proj.weight")


def test_block_attnres_export_requires_mixers_and_alpha_one() -> None:
    source = config(
        residual_type="block_attnres",
        attnres_num_blocks=1,
        attnres_alpha=1.0,
        architecture_revision="tiny_attnres",
    )
    state = minimal_state(block_attnres=True)
    deployment = EXPORTER.build_deployment_config(
        state, source, {}, "bfloat16"
    )
    assert deployment["residual_type"] == "block_attnres"
    assert deployment["layers_per_attnres_block"] == 1
    names = EXPORTER.required_tensor_names(deployment)
    assert "model.final_attn_res_mixer.query" in names
    exported = EXPORTER.prepare_export_state(
        state, deployment, torch.bfloat16
    )
    assert "model.layers.0.attn_res_mixer.query" in exported

    with pytest.raises(ValueError, match="alpha=1"):
        EXPORTER.build_deployment_config(
            state,
            {**source, "attnres_alpha": 0.5},
            {},
            "bfloat16",
        )


def test_block_attnres_export_rejects_missing_mixer_weight() -> None:
    state = minimal_state(block_attnres=True)
    state.pop("model.layers.0.mlp_res_mixer.query")
    deployment = EXPORTER.build_deployment_config(
        state,
        config(residual_type="block_attnres", attnres_num_blocks=1),
        {},
        "bfloat16",
    )
    with pytest.raises(KeyError, match="mlp_res_mixer.query"):
        EXPORTER.prepare_export_state(state, deployment, torch.bfloat16)
