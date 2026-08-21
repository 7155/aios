#!/usr/bin/env python3
"""Export a validated MiniMind-IME checkpoint as an AIOS deployment bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Deployment weight dtype; BF16 preserves the current 0.1B memory profile.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow generated files in a non-empty output directory to be replaced.",
    )
    return parser.parse_args()


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        converted = value.to_dict()
        return dict(converted) if isinstance(converted, Mapping) else {}
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _checkpoint_state(payload: Any) -> tuple[Mapping[str, Any], dict[str, Any]]:
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint must contain a state-dict mapping")
    metadata = dict(payload)
    for key in ("model", "state_dict", "model_state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping) and candidate:
            return candidate, metadata
    return payload, metadata


def _normalize_name(raw_name: str) -> str:
    name = raw_name
    for prefix in ("_orig_mod.", "module."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    if name.startswith("model.model."):
        name = name[len("model.") :]
    if name == "model.lm_head.weight":
        name = "lm_head.weight"
    return name


def _is_mtp_tensor(name: str) -> bool:
    parts = name.split(".")
    return any(part == "mtp" or part.startswith("mtp_") for part in parts)


def load_checkpoint(
    path: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], tuple[str, ...]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    source_state, metadata = _checkpoint_state(payload)
    state: dict[str, torch.Tensor] = {}
    stripped_mtp: list[str] = []
    for raw_name, tensor in source_state.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        name = _normalize_name(str(raw_name))
        if _is_mtp_tensor(name):
            stripped_mtp.append(name)
            continue
        if not (name.startswith("model.") or name.startswith("lm_head.")):
            continue
        state[name] = tensor.detach().cpu().contiguous()
    if not state:
        raise ValueError("no MiniMind inference tensors were found in the checkpoint")
    return state, metadata, tuple(stripped_mtp)


def read_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _pick(source: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in source and source[name] is not None:
            return source[name]
    return default


def build_deployment_config(
    state: Mapping[str, torch.Tensor],
    training_config: Mapping[str, Any],
    checkpoint_metadata: Mapping[str, Any],
    dtype_name: str,
) -> dict[str, Any]:
    checkpoint_config = _as_mapping(
        checkpoint_metadata.get("config")
        or checkpoint_metadata.get("model_config")
        or checkpoint_metadata.get("args")
    )
    source = {**checkpoint_config, **dict(training_config)}
    if bool(_pick(source, "use_moe", default=False)) or any(
        ".experts." in name for name in state
    ):
        raise ValueError("AIOS MiniMind-IME export supports dense checkpoints only")
    if bool(_pick(source, "inference_rope_scaling", default=False)):
        raise ValueError("AIOS MiniMind-IME export does not support YaRN checkpoints")

    embedding = state.get("model.embed_tokens.weight")
    if embedding is None:
        raise KeyError("checkpoint is missing model.embed_tokens.weight")
    if embedding.ndim != 2:
        raise ValueError("model.embed_tokens.weight must be a matrix")
    vocab_size, hidden_size = embedding.shape

    layer_ids = sorted(
        {
            int(match.group(1))
            for name in state
            if (match := LAYER_RE.match(name)) is not None
        }
    )
    if not layer_ids or layer_ids != list(range(layer_ids[-1] + 1)):
        raise ValueError("model.layers must be contiguous and start at layer 0")

    gate = state.get("model.layers.0.mlp.gate_proj.weight")
    k_proj = state.get("model.layers.0.self_attn.k_proj.weight")
    if gate is None or k_proj is None:
        raise KeyError("checkpoint is missing layer-0 MLP or K projection weights")

    num_heads = int(_pick(source, "num_attention_heads", "n_heads"))
    head_dim = int(_pick(source, "head_dim", default=hidden_size // num_heads))
    if hidden_size != num_heads * head_dim:
        raise ValueError(
            f"hidden_size={hidden_size} does not equal num_heads*head_dim="
            f"{num_heads * head_dim}"
        )
    inferred_kv_heads = k_proj.shape[0] // head_dim
    num_kv_heads = int(
        _pick(
            source,
            "num_key_value_heads",
            "n_kv_heads",
            default=inferred_kv_heads,
        )
    )
    if num_kv_heads != inferred_kv_heads:
        raise ValueError(
            f"num_key_value_heads={num_kv_heads} conflicts with k_proj shape "
            f"({inferred_kv_heads} inferred)"
        )

    hidden_act = str(_pick(source, "hidden_act", default="silu")).lower()
    if hidden_act not in {"silu", "swish"}:
        raise ValueError("AIOS MiniMind-IME backend only implements SiLU activation")

    tie_word_embeddings = bool(
        _pick(
            source,
            "tie_word_embeddings",
            default="lm_head.weight" not in state,
        )
    )
    lm_head = state.get("lm_head.weight")
    if tie_word_embeddings and lm_head is not None:
        if lm_head.shape != embedding.shape or not torch.equal(lm_head, embedding):
            raise ValueError(
                "config says tie_word_embeddings=True but lm_head and embedding "
                "weights differ"
            )
    if not tie_word_embeddings and lm_head is None:
        raise KeyError(
            "untied checkpoint is missing lm_head.weight; refusing to tie it silently"
        )

    residual_type = str(_pick(source, "residual_type", default="standard"))
    if residual_type not in {"standard", "block_attnres"}:
        raise ValueError(f"unsupported residual_type: {residual_type}")
    attnres_num_blocks = int(
        _pick(source, "attnres_num_blocks", default=0)
    )
    attnres_alpha = float(_pick(source, "attnres_alpha", default=1.0))
    if residual_type == "block_attnres":
        if attnres_num_blocks < 1 or len(layer_ids) % attnres_num_blocks:
            raise ValueError(
                "Block AttnRes requires a positive block count that divides "
                "the number of layers"
            )
        if attnres_alpha != 1.0:
            raise ValueError(
                "deployment export requires a fully migrated AttnRes alpha=1 snapshot"
            )
        if str(_pick(source, "attnres_query_init", default="zeros")) != "zeros":
            raise ValueError("only zero-initialized AttnRes query lineage is supported")
    elif attnres_num_blocks:
        raise ValueError("standard residual config must set attnres_num_blocks=0")

    config = dict(source)
    config.update({
        "architectures": ["MiniMindIMEForCausalLM"],
        "model_type": "minimind_ime_v3",
        "vocab_size": int(vocab_size),
        "hidden_size": int(hidden_size),
        "num_hidden_layers": len(layer_ids),
        "intermediate_size": int(gate.shape[0]),
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "hidden_act": "silu",
        "rms_norm_eps": float(
            _pick(source, "rms_norm_eps", "norm_eps", default=1e-6)
        ),
        "rope_theta": float(_pick(source, "rope_theta", default=1_000_000.0)),
        "max_position_embeddings": int(
            _pick(source, "max_position_embeddings", "max_seq_len", default=32768)
        ),
        "tie_word_embeddings": tie_word_embeddings,
        "use_moe": False,
        "inference_rope_scaling": False,
        "mtp_enabled": False,
        "torch_dtype": dtype_name,
        "architecture_revision": str(
            _pick(source, "architecture_revision", default="")
        ),
        "residual_type": residual_type,
        "attnres_num_blocks": attnres_num_blocks,
        "attnres_alpha": attnres_alpha,
        "attnres_backend": "triton",
    })
    if residual_type == "block_attnres":
        config["layers_per_attnres_block"] = len(layer_ids) // attnres_num_blocks
    config.pop("mtp_intermediate_size", None)
    return config


def required_tensor_names(config: Mapping[str, Any]) -> list[str]:
    names = ["model.embed_tokens.weight", "model.norm.weight"]
    for layer in range(int(config["num_hidden_layers"])):
        base = f"model.layers.{layer}"
        names.extend([
            f"{base}.input_layernorm.weight",
            f"{base}.post_attention_layernorm.weight",
            f"{base}.self_attn.q_proj.weight",
            f"{base}.self_attn.k_proj.weight",
            f"{base}.self_attn.v_proj.weight",
            f"{base}.self_attn.o_proj.weight",
            f"{base}.self_attn.q_norm.weight",
            f"{base}.self_attn.k_norm.weight",
            f"{base}.mlp.gate_proj.weight",
            f"{base}.mlp.up_proj.weight",
            f"{base}.mlp.down_proj.weight",
        ])
        if config.get("residual_type", "standard") == "block_attnres":
            names.extend([
                f"{base}.attn_res_mixer.query",
                f"{base}.attn_res_mixer.key_norm.weight",
                f"{base}.mlp_res_mixer.query",
                f"{base}.mlp_res_mixer.key_norm.weight",
            ])
    if config.get("residual_type", "standard") == "block_attnres":
        names.extend([
            "model.final_attn_res_mixer.query",
            "model.final_attn_res_mixer.key_norm.weight",
        ])
    if not bool(config["tie_word_embeddings"]):
        names.append("lm_head.weight")
    return names


def prepare_export_state(
    state: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    missing = [name for name in required_tensor_names(config) if name not in state]
    if missing:
        preview = "\n".join(f"  - {name}" for name in missing[:20])
        raise KeyError(f"checkpoint is missing required tensors:\n{preview}")

    # Export the exact inference contract only. Training checkpoints may carry
    # tied-head aliases or non-persistent experiment tensors that AIOS neither
    # owns nor loads; omitting them keeps parameter and SHA accounting exact.
    exported = {
        name: state[name].detach().to(dtype=dtype, device="cpu").contiguous()
        for name in required_tensor_names(config)
    }
    return exported


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise NotADirectoryError(args.output_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise FileExistsError(
            f"{args.output_dir} is not empty; pass --force to replace generated files"
        )

    tokenizer_sources = {
        name: args.tokenizer_dir / name
        for name in TOKENIZER_FILES
        if (args.tokenizer_dir / name).exists()
    }
    if "tokenizer.json" not in tokenizer_sources:
        raise FileNotFoundError(
            f"{args.tokenizer_dir} does not contain tokenizer.json"
        )

    state, metadata, stripped_mtp = load_checkpoint(args.checkpoint)
    source_config = read_config(args.config)
    config = build_deployment_config(state, source_config, metadata, args.dtype)
    exported = prepare_export_state(state, config, getattr(torch, args.dtype))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.force:
        for name in (
            "model.safetensors",
            "config.json",
            "aios_manifest.json",
            *TOKENIZER_FILES,
        ):
            target = args.output_dir / name
            if target.is_file():
                target.unlink()
    model_path = args.output_dir / "model.safetensors"
    save_file(exported, str(model_path))
    (args.output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    copied: list[str] = []
    for name, source in tokenizer_sources.items():
        shutil.copy2(source, args.output_dir / name)
        copied.append(name)

    parameter_count = sum(tensor.numel() for tensor in exported.values())
    weight_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in exported.values()
    )
    manifest = {
        "schema_version": "aios.minimind_ime_bundle.v3",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": sha256(args.checkpoint),
        "model_sha256": sha256(model_path),
        "dtype": args.dtype,
        "inference_parameter_tensors": len(exported),
        "inference_parameter_count": parameter_count,
        "inference_weight_bytes": weight_bytes,
        "mtp_stripped": bool(stripped_mtp),
        "mtp_tensor_count_stripped": len(stripped_mtp),
        "tied_lm_head_omitted": bool(config["tie_word_embeddings"]),
        "residual_type": config["residual_type"],
        "attnres_num_blocks": config["attnres_num_blocks"],
        "layers_per_attnres_block": config.get("layers_per_attnres_block", 0),
        "attnres_alpha": config["attnres_alpha"],
        "tokenizer_files": copied,
        "config": config,
    }
    (args.output_dir / "aios_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
