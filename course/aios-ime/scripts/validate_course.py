#!/usr/bin/env python3
"""Validate the AIOS-IME v5.1 course without requiring CUDA."""

from __future__ import annotations

import json
import py_compile
import re
import sys
from pathlib import Path
from typing import Any


SOURCE_REVISION = "9f53740753de36899aa7694cf7dcb5304e58ea54"
COURSE_VERSION = "5.1"
COURSE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COURSE_ROOT.parents[1]

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
LESSON_IDS = [
    *(item.split("-", 1)[0] for item in PRACTICE_LESSONS),
    *(item.split("-", 1)[0] for item in MECHANISM_LESSONS),
]
MECHANISM_IDS = [item.split("-", 1)[0] for item in MECHANISM_LESSONS]

SUPPORT_DOCS = [
    "README.md",
    "LEARNING_PATHS.md",
    "VISUAL_ATLAS.md",
    "SOURCE_MAP.md",
    "WORKBOOK.md",
    "TROUBLESHOOTING.md",
    "COURSE_STATE.md",
    "CHANGELOG.md",
    "tracks/practice/README.md",
    "tracks/mechanism/README.md",
    "snapshots/README.md",
]
DATA_AND_WEB_FILES = [
    "curriculum.yaml",
    "evidence.jsonl",
    "source-symbols.json",
    "bridge-matrix.yaml",
    "site-manifest.json",
    "index.html",
    "assets/course.css",
    "assets/course.js",
    "assets/course-render.js",
    "assets/course-utils.js",
]
FORBIDDEN_TERMS = (
    "\u9762\u8bd5\u9898",
    "\u9762\u8bd5\u8ffd\u95ee",
    "\u68c0\u9a8c\u95ee\u9898",
)


class ValidationError(RuntimeError):
    """Raised when the course contract is violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def practice_lesson_path(slug: str) -> Path:
    return COURSE_ROOT / f"tracks/practice/lessons/{slug}/README.md"


def mechanism_lesson_path(slug: str) -> Path:
    return COURSE_ROOT / f"tracks/mechanism/lessons/{slug}/README.md"


def lesson_paths() -> list[Path]:
    return [
        *(practice_lesson_path(slug) for slug in PRACTICE_LESSONS),
        *(mechanism_lesson_path(slug) for slug in MECHANISM_LESSONS),
    ]


def expected_files() -> list[Path]:
    files = [COURSE_ROOT / path for path in SUPPORT_DOCS + DATA_AND_WEB_FILES]
    files.extend(lesson_paths())
    files.append(REPO_ROOT / ".github/workflows/aios-ime-course.yml")
    return files


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def validate_markdown_contract(path: Path) -> None:
    text = read_text(path)
    require(
        text.count("```") % 2 == 0,
        f"unbalanced fenced code block in {path.relative_to(REPO_ROOT)}",
    )
    for term in FORBIDDEN_TERMS:
        require(
            term not in text,
            f"forbidden learner-facing term {term!r} in {path.relative_to(REPO_ROOT)}",
        )


def validate_lesson(path: Path) -> None:
    text = read_text(path)
    relative = path.relative_to(REPO_ROOT)

    require(SOURCE_REVISION in text, f"missing source pin: {relative}")
    require("```mermaid" in text, f"missing Mermaid diagram: {relative}")
    require("## 验收" in text, f"missing acceptance section: {relative}")
    require("## 练习题" in text, f"missing exercise section: {relative}")
    require(
        any(marker in text for marker in ("```python", "```bash", "```text")),
        f"missing embedded code or runnable command: {relative}",
    )
    require(
        text.count("<details>") >= 3,
        f"lesson needs at least three answered exercises: {relative}",
    )
    require(
        "<summary>参考答案</summary>" in text,
        f"exercise answers are not directly embedded: {relative}",
    )


def validate_links(markdown_path: Path) -> None:
    text = read_text(markdown_path)
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
        target = target.strip()
        if (
            not target
            or target.startswith(("http://", "https://", "mailto:", "tel:", "#"))
        ):
            continue

        path_part = target.split("#", 1)[0].split("?", 1)[0].strip()
        if not path_part:
            continue

        resolved = (markdown_path.parent / path_part).resolve()
        require(
            resolved.exists(),
            f"broken link in {markdown_path.relative_to(REPO_ROOT)}: {target}",
        )


def load_json(path: Path) -> Any:
    try:
        return json.loads(read_text(path))
    except json.JSONDecodeError as exc:
        raise ValidationError(
            f"invalid JSON in {path.relative_to(REPO_ROOT)}: {exc}"
        ) from exc


def validate_evidence() -> int:
    path = COURSE_ROOT / "evidence.jsonl"
    ids: set[str] = set()
    count = 0

    for line_number, line in enumerate(read_text(path).splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"invalid evidence JSON at line {line_number}: {exc}"
            ) from exc

        require(
            isinstance(row, dict)
            and "id" in row
            and "claim" in row
            and "status" in row,
            f"incomplete evidence row {line_number}",
        )
        require(row["id"] not in ids, f"duplicate evidence id {row['id']}")
        ids.add(row["id"])
        count += 1

    require(count >= 9, "evidence ledger unexpectedly shrank")
    return count


def validate_manifest() -> tuple[int, int]:
    path = COURSE_ROOT / "site-manifest.json"
    manifest = load_json(path)

    require(isinstance(manifest, dict), "site manifest must be an object")
    course = manifest.get("course")
    sections = manifest.get("sections")
    require(isinstance(course, dict), "site manifest missing course")
    require(isinstance(sections, list) and sections, "site manifest missing sections")
    require(course.get("version") == COURSE_VERSION, "site manifest version drift")
    require(
        course.get("source_revision") == SOURCE_REVISION,
        "site manifest source pin drift",
    )

    document_ids: set[str] = set()
    document_paths: set[str] = set()
    lesson_ids: set[str] = set()
    document_count = 0

    for section in sections:
        require(isinstance(section, dict), "site manifest section must be an object")
        require(section.get("id") and section.get("title"), "incomplete site section")
        docs = section.get("docs")
        require(isinstance(docs, list) and docs, "site section has no docs")

        for doc in docs:
            require(isinstance(doc, dict), "site document must be an object")
            doc_id = doc.get("id")
            doc_path = doc.get("path")
            require(isinstance(doc_id, str) and doc_id, "site document missing id")
            require(isinstance(doc_path, str) and doc_path, f"{doc_id} missing path")
            require(doc.get("title"), f"{doc_id} missing title")
            require(doc.get("summary"), f"{doc_id} missing summary")
            require(doc_id not in document_ids, f"duplicate site document id {doc_id}")
            require(doc_path not in document_paths, f"duplicate site path {doc_path}")
            require(".." not in Path(doc_path).parts, f"unsafe site path {doc_path}")

            resolved = COURSE_ROOT / doc_path
            require(resolved.is_file(), f"site document path does not exist: {doc_path}")

            document_ids.add(doc_id)
            document_paths.add(doc_path)
            document_count += 1
            if doc.get("kind") == "lesson":
                lesson_ids.add(doc_id)

    require(
        lesson_ids == set(LESSON_IDS),
        f"site lesson set drift: expected {LESSON_IDS}, got {sorted(lesson_ids)}",
    )

    expected_lesson_paths = {
        str(path.relative_to(COURSE_ROOT)).replace("\\", "/")
        for path in lesson_paths()
    }
    require(
        expected_lesson_paths.issubset(document_paths),
        "site manifest does not include every lesson README",
    )
    require(course.get("home") in document_paths, "site home is not a listed document")

    return document_count, len(lesson_ids)


def validate_source_symbols() -> tuple[int, int]:
    registry = load_json(COURSE_ROOT / "source-symbols.json")
    require(isinstance(registry, dict), "source symbol registry must be an object")
    require(
        registry.get("source_revision") == SOURCE_REVISION,
        "source symbol registry pin drift",
    )

    entries = registry.get("entries")
    require(isinstance(entries, list) and entries, "source symbol registry is empty")

    covered_lessons: set[str] = set()
    symbol_count = 0
    seen: set[tuple[str, str]] = set()

    for entry in entries:
        require(isinstance(entry, dict), "source symbol entry must be an object")
        lesson = entry.get("lesson")
        path_value = entry.get("path")
        symbols = entry.get("symbols")

        require(lesson in MECHANISM_IDS, f"unexpected source lesson {lesson}")
        require(isinstance(path_value, str) and path_value, f"{lesson} missing path")
        require(".." not in Path(path_value).parts, f"unsafe source path {path_value}")
        require(
            isinstance(symbols, list) and symbols,
            f"{lesson}:{path_value} has no symbols",
        )
        require(
            (lesson, path_value) not in seen,
            f"duplicate source entry {lesson}:{path_value}",
        )
        seen.add((lesson, path_value))

        source_path = REPO_ROOT / path_value
        require(source_path.is_file(), f"source anchor path missing: {path_value}")
        source_text = read_text(source_path)

        for symbol in symbols:
            require(isinstance(symbol, str) and symbol, "empty source symbol")
            require(
                symbol in source_text,
                f"source symbol drift: {lesson} -> {path_value} :: {symbol}",
            )
            symbol_count += 1

        covered_lessons.add(lesson)

    require(
        covered_lessons == set(MECHANISM_IDS),
        "not every mechanism lesson has a canonical source anchor",
    )
    return len(entries), symbol_count


def validate_curriculum() -> None:
    path = COURSE_ROOT / "curriculum.yaml"
    text = read_text(path)
    require(f'revision: {SOURCE_REVISION}' in text, "curriculum source pin drift")
    require(f'version: "{COURSE_VERSION}"' in text, "curriculum version drift")
    require("web_portal_manifest_driven: true" in text, "web contract missing")
    require("source_symbols_machine_checked: true" in text, "source contract missing")
    require("stage_artifacts_required: true" in text, "artifact contract missing")

    for lesson_id in LESSON_IDS:
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
        require(
            isinstance(parsed, dict)
            and "course" in parsed
            and "tracks" in parsed
            and "delivery" in parsed,
            "curriculum YAML does not contain course/tracks/delivery",
        )
        yaml.safe_load(read_text(COURSE_ROOT / "bridge-matrix.yaml"))


def validate_web_assets() -> None:
    index = read_text(COURSE_ROOT / "index.html")
    javascript = read_text(COURSE_ROOT / "assets/course.js")
    renderer = read_text(COURSE_ROOT / "assets/course-render.js")
    utilities = read_text(COURSE_ROOT / "assets/course-utils.js")
    stylesheet = read_text(COURSE_ROOT / "assets/course.css")
    readme = read_text(COURSE_ROOT / "README.md")

    for reference in (
        "site-manifest.json",
        "assets/course.css",
        "assets/course.js",
        'type="module"',
        "marked@12.0.2",
        "mermaid@10.9.1",
        "katex@0.16.10",
    ):
        require(reference in index, f"web entry missing pinned asset: {reference}")

    require('id="search-button"' in index, "web entry lacks mobile search control")
    require(
        'fetchJson("site-manifest.json")' in javascript,
        "web app does not load the course manifest",
    )
    require("setSearchOpen" in javascript, "web app lacks responsive search control")
    require("scrollToFragment" in javascript, "web app lacks deep-link scrolling")
    require('aria-current", "page"' in javascript, "web app lacks current-page state")
    require(
        'from "./course-render.js"' in javascript
        and 'from "./course-utils.js"' in javascript,
        "web entry does not use the module boundary",
    )
    require("extractMathBlocks" in renderer, "web app lacks fenced math handling")
    require("renderDiagrams" in renderer, "web app lacks Mermaid handling")
    require("normalizePath" in utilities, "web utility module lacks path normalization")
    require("localStorage" in javascript, "web app lacks local progress")
    require("--paper:" in stylesheet and "--accent:" in stylesheet, "web theme incomplete")
    require(
        "python -m http.server --directory course/aios-ime 8000" in readme,
        "README missing web preview command",
    )

    atlas = read_text(COURSE_ROOT / "VISUAL_ATLAS.md")
    require(
        atlas.count("```mermaid") >= 10,
        "visual atlas must contain at least ten focused diagrams",
    )


def validate_python_scripts() -> int:
    count = 0
    for path in sorted((COURSE_ROOT / "scripts").glob("*.py")):
        py_compile.compile(str(path), doraise=True)
        count += 1
    require(count >= 4, "course scripts unexpectedly shrank")
    return count


def main() -> None:
    for path in expected_files():
        require(path.is_file(), f"missing course file: {path.relative_to(REPO_ROOT)}")

    lessons = lesson_paths()
    for path in lessons:
        validate_lesson(path)

    for path in COURSE_ROOT.rglob("*.md"):
        validate_markdown_contract(path)
        validate_links(path)

    validate_curriculum()
    evidence_count = validate_evidence()
    document_count, manifest_lesson_count = validate_manifest()
    source_entry_count, symbol_count = validate_source_symbols()
    validate_web_assets()
    script_count = validate_python_scripts()

    print(
        "AIOS-IME course validation passed: "
        f"{len(lessons)} lessons, "
        f"{document_count} site documents ({manifest_lesson_count} lessons), "
        f"{evidence_count} evidence rows, "
        f"{source_entry_count} source entries / {symbol_count} symbols, "
        f"{script_count} Python scripts."
    )


if __name__ == "__main__":
    try:
        main()
    except ValidationError as exc:
        print(f"course validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
