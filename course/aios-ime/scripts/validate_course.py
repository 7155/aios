#!/usr/bin/env python3
"""Validate the AIOS-IME v5 course structure without requiring CUDA."""

from __future__ import annotations

import json
import py_compile
import re
import sys
from pathlib import Path


SOURCE_REVISION = "9f53740753de36899aa7694cf7dcb5304e58ea54"
COURSE_ROOT = Path(__file__).resolve().parents[1]

PRACTICE_LESSONS = [
    "P01-first-top3",
    "P02-candidate-budget",
    "P03-continuous-typing",
    "P04-model-matrix",
    "P05-profile-attnres",
    "P06-release-gate",
]
MECHANISM_LESSONS = [
    "M01-runtime-map",
    "M02-model-contract",
    "M03-candidate-group-kv",
    "M04-ragged-rng",
    "M05-adaptive-refill",
    "M06-token-lcp",
    "M07-latest-wins",
    "M08-candidate-governance",
    "M09-block-attnres",
    "M10-triton-attnres",
    "M11-evidence-lanes",
]
FORBIDDEN_TERMS = ("\u9762\u8bd5\u9898", "\u9762\u8bd5\u8ffd\u95ee", "\u68c0\u9a8c\u95ee\u9898")


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def expected_files() -> list[Path]:
    files = [
        COURSE_ROOT / "README.md",
        COURSE_ROOT / "curriculum.yaml",
        COURSE_ROOT / "COURSE_STATE.md",
        COURSE_ROOT / "evidence.jsonl",
        COURSE_ROOT / "bridge-matrix.yaml",
        COURSE_ROOT / "CHANGELOG.md",
        COURSE_ROOT / "tracks/practice/README.md",
        COURSE_ROOT / "tracks/mechanism/README.md",
        COURSE_ROOT / "snapshots/README.md",
    ]
    files.extend(
        COURSE_ROOT / f"tracks/practice/lessons/{slug}/README.md"
        for slug in PRACTICE_LESSONS
    )
    files.extend(
        COURSE_ROOT / f"tracks/mechanism/lessons/{slug}/README.md"
        for slug in MECHANISM_LESSONS
    )
    return files


def validate_lesson(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    require(SOURCE_REVISION in text, f"missing source pin: {path}")
    require("```mermaid" in text, f"missing Mermaid diagram: {path}")
    require("## 验收" in text, f"missing acceptance section: {path}")
    require("## 练习题" in text, f"missing exercise section: {path}")
    require(
        any(marker in text for marker in ("```python", "```bash", "```text")),
        f"missing embedded code or runnable command: {path}",
    )
    for term in FORBIDDEN_TERMS:
        require(term not in text, f"forbidden term {term!r} in {path}")


def validate_links(markdown_path: Path) -> None:
    text = markdown_path.read_text(encoding="utf-8")
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
        target = target.split("#", 1)[0].strip()
        if (
            not target
            or target.startswith(("http://", "https://", "mailto:", "#"))
        ):
            continue
        resolved = (markdown_path.parent / target).resolve()
        require(resolved.exists(), f"broken link in {markdown_path}: {target}")


def validate_evidence() -> int:
    path = COURSE_ROOT / "evidence.jsonl"
    ids: set[str] = set()
    count = 0
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"invalid evidence JSON at line {line_number}: {exc}"
            ) from exc
        require("id" in row and "claim" in row and "status" in row,
                f"incomplete evidence row {line_number}")
        require(row["id"] not in ids, f"duplicate evidence id {row['id']}")
        ids.add(row["id"])
        count += 1
    return count


def validate_curriculum() -> None:
    text = (COURSE_ROOT / "curriculum.yaml").read_text(encoding="utf-8")
    require(f"revision: {SOURCE_REVISION}" in text, "curriculum source pin drift")
    for lesson_id in [*(x.split("-", 1)[0] for x in PRACTICE_LESSONS),
                      *(x.split("-", 1)[0] for x in MECHANISM_LESSONS)]:
        require(
            re.search(rf"\bid:\s*{re.escape(lesson_id)}\b", text) is not None,
            f"curriculum missing lesson {lesson_id}",
        )
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        print("[note] PyYAML not installed; required-key checks used.")
    else:
        parsed = yaml.safe_load(text)
        require(isinstance(parsed, dict) and "course" in parsed and "tracks" in parsed,
                "curriculum YAML does not contain course/tracks")
        yaml.safe_load((COURSE_ROOT / "bridge-matrix.yaml").read_text(encoding="utf-8"))


def validate_python_scripts() -> int:
    count = 0
    for path in sorted((COURSE_ROOT / "scripts").glob("*.py")):
        py_compile.compile(str(path), doraise=True)
        count += 1
    return count


def main() -> None:
    for path in expected_files():
        require(path.is_file(), f"missing course file: {path}")

    lesson_paths = [
        *(
            COURSE_ROOT / f"tracks/practice/lessons/{slug}/README.md"
            for slug in PRACTICE_LESSONS
        ),
        *(
            COURSE_ROOT / f"tracks/mechanism/lessons/{slug}/README.md"
            for slug in MECHANISM_LESSONS
        ),
    ]
    for path in lesson_paths:
        validate_lesson(path)

    for path in COURSE_ROOT.rglob("*.md"):
        validate_links(path)

    validate_curriculum()
    evidence_count = validate_evidence()
    script_count = validate_python_scripts()

    print(
        "AIOS-IME course validation passed: "
        f"{len(lesson_paths)} lessons, {evidence_count} evidence rows, "
        f"{script_count} Python scripts."
    )


if __name__ == "__main__":
    try:
        main()
    except ValidationError as exc:
        print(f"course validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
