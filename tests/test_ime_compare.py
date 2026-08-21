from __future__ import annotations

import json
from pathlib import Path

import pytest

from aios.ime_compare import (
    CompareRequest,
    GenerationSettings,
    ImeCompareService,
    ModelSpec,
    compare_results,
    json_safe,
)


def model_payload(path: Path, label: str) -> dict[str, object]:
    return {
        "label": label,
        "model_path": str(path),
        "backend": "triton",
        "kv_cache_max_tokens": 512,
        "attention_workspace_mib": 8,
        "device": "cuda:0",
    }


def create_model_dir(path: Path) -> Path:
    path.mkdir()
    (path / "config.json").write_text("{}", encoding="utf-8")
    return path


def candidate(text: str) -> dict[str, str]:
    return {"text": text}


def test_model_spec_only_accepts_local_deployment_directory(tmp_path: Path) -> None:
    model_dir = create_model_dir(tmp_path / "model")
    spec = ModelSpec.from_payload(model_payload(model_dir, "模型 A"))
    assert spec.model_path == str(model_dir.resolve())
    assert spec.backend == "triton"

    with pytest.raises(ValueError, match="模型目录不存在"):
        ModelSpec.from_payload(model_payload(tmp_path / "missing", "模型 B"))


def test_generation_settings_validate_candidate_budgets() -> None:
    settings = GenerationSettings.from_payload(
        {"sampling_attempts": 8, "max_sampling_attempts": 24}
    )
    assert settings.to_ime_config().display_candidates == 3

    with pytest.raises(ValueError, match="总预算"):
        GenerationSettings.from_payload(
            {"sampling_attempts": 8, "max_sampling_attempts": 4}
        )


def test_compare_request_keeps_explicit_serial_order(tmp_path: Path) -> None:
    left = create_model_dir(tmp_path / "left")
    right = create_model_dir(tmp_path / "right")
    request = CompareRequest.from_payload(
        {
            "prefix": "今天下班有点晚，",
            "slots": {
                "a": model_payload(left, "214M"),
                "b": model_payload(right, "100M"),
            },
            "targets": ["b", "a", "b"],
            "order": "b_then_a",
            "reset_prefix_cache": True,
        },
        require_local_paths=True,
    )
    assert request.targets == ("b", "a")
    assert request.order == "b_then_a"
    assert request.reset_prefix_cache


def test_compare_results_reports_overlap_rank_and_latency() -> None:
    left = {
        "candidates": [candidate("我马上回去。"), candidate("我晚点联系你。")],
        "latency_ms": 80.0,
    }
    right = {
        "candidates": [candidate("我马上回去"), candidate("明天再说。")],
        "latency_ms": 120.0,
    }
    comparison = compare_results(left, right)
    assert comparison["top3_overlap"] == 1
    assert comparison["same_rank_candidates"] == 1
    assert comparison["top1_character_lcp"] == 5
    assert comparison["faster_slot"] == "a"
    assert comparison["latency_ratio"] == 1.5


def test_demo_service_returns_two_complete_top3_results() -> None:
    slots = {
        "a": ModelSpec("214M", "demo/a", "triton"),
        "b": ModelSpec("100M", "demo/b", "default"),
    }
    service = ImeCompareService(slots, demo=True)
    response = service.compare(
        {
            "prefix": "这个报错我看了一下，",
            "slots": {key: vars(value) for key, value in slots.items()},
            "targets": ["a", "b"],
            "order": "a_then_b",
            "reset_prefix_cache": False,
        }
    )
    assert response["demo"]
    assert response["results"]["a"]["ok"]
    assert response["results"]["b"]["ok"]
    assert len(response["results"]["a"]["result"]["candidates"]) == 3
    assert response["comparison"] is not None
    json.dumps(response, ensure_ascii=False, allow_nan=False)
    initial = service.initial_config()
    assert initial["examples"][0] == "机器学习是"
    assert len(initial["profiles"]) == 2
    assert response["results"]["a"]["result"]["runtime"]["model"]["dtype"] == "bfloat16"
    assert response["results"]["a"]["result"]["runtime"]["model_warmup_ms"] == 0.0


def test_json_safe_replaces_non_finite_scores() -> None:
    payload = json_safe({"negative": float("-inf"), "positive": float("inf")})
    assert payload == {"negative": None, "positive": None}


def test_frontend_assets_contain_required_comparison_controls() -> None:
    root = Path(__file__).resolve().parents[1] / "web" / "ime_compare"
    html = (root / "index.html").read_text(encoding="utf-8")
    javascript = (root / "app.js").read_text(encoding="utf-8")
    assert "AIOS-IME 推理对比台" in html
    assert 'id="candidates-a"' in html
    assert 'id="candidates-b"' in html
    assert 'id="unload-button"' in html
    assert 'id="profile-a"' in html
    assert 'id="profile-b"' in html
    assert html.index('id="prefix-input"') < html.index('id="result-section"')
    assert html.index('id="result-section"') < html.index('class="settings-panel"')
    assert "/api/compare" in javascript
    assert "/api/unload" in javascript
