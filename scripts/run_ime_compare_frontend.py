#!/usr/bin/env python3
"""Run the local AIOS-IME side-by-side inference frontend."""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from aios.ime_compare import ImeCompareService, ModelSpec, json_safe


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = REPOSITORY_ROOT / "web" / "ime_compare"
STATIC_ROUTES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/styles.css": "styles.css",
    "/app.js": "app.js",
}
MAX_REQUEST_BYTES = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", help="A 侧本地 AIOS 模型目录")
    parser.add_argument("--model-b", help="B 侧本地 AIOS 模型目录；默认与 A 相同")
    parser.add_argument("--label-a", default="当前模型", help="A 侧显示名称")
    parser.add_argument("--label-b", default="对比模型", help="B 侧显示名称")
    parser.add_argument(
        "--backend-a",
        choices=("default", "reference", "eager", "compiled", "triton"),
        default="default",
    )
    parser.add_argument(
        "--backend-b",
        choices=("default", "reference", "eager", "compiled", "triton"),
        default="default",
    )
    parser.add_argument("--kv-cache-max-tokens", type=int, default=512)
    parser.add_argument("--attention-workspace-mib", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        metavar="NAME=LOCAL_PATH",
        help="增加一个可在前端 A/B 两侧快速选择的本地 BF16 模型",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="不加载 CUDA 模型，仅用固定演示数据预览界面",
    )
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    if not args.demo and not args.model_a:
        parser.error("真实推理必须提供 --model-a；只预览界面可使用 --demo")
    if not 1 <= args.port <= 65535:
        parser.error("--port 必须在 1～65535 之间")
    return args


def build_default_slots(args: argparse.Namespace) -> dict[str, ModelSpec]:
    model_a = args.model_a or "demo/current-model"
    model_b = args.model_b or args.model_a or "demo/comparison-model"
    common = {
        "kv_cache_max_tokens": args.kv_cache_max_tokens,
        "attention_workspace_mib": args.attention_workspace_mib,
        "device": args.device,
    }
    return {
        "a": ModelSpec.from_payload(
            {
                **common,
                "label": args.label_a,
                "model_path": model_a,
                "backend": args.backend_a,
            },
            require_local_path=not args.demo,
        ),
        "b": ModelSpec.from_payload(
            {
                **common,
                "label": args.label_b,
                "model_path": model_b,
                "backend": args.backend_b,
            },
            require_local_path=not args.demo,
        ),
    }


def build_profiles(
    args: argparse.Namespace,
    default_slots: dict[str, ModelSpec],
) -> list[ModelSpec]:
    profiles = list(default_slots.values())
    for raw_profile in args.profile:
        if "=" not in raw_profile:
            raise ValueError(f"--profile 必须使用 NAME=LOCAL_PATH 格式：{raw_profile}")
        label, model_path = raw_profile.split("=", 1)
        profiles.append(
            ModelSpec.from_payload(
                {
                    "label": label.strip(),
                    "model_path": model_path.strip(),
                    "backend": "default",
                    "kv_cache_max_tokens": args.kv_cache_max_tokens,
                    "attention_workspace_mib": args.attention_workspace_mib,
                    "device": args.device,
                },
                require_local_path=not args.demo,
            )
        )
    unique_profiles: list[ModelSpec] = []
    seen: set[tuple[object, ...]] = set()
    for profile in profiles:
        key = profile.runtime_key
        if key in seen:
            continue
        seen.add(key)
        unique_profiles.append(profile)
    return unique_profiles


def make_handler(service: ImeCompareService) -> type[BaseHTTPRequestHandler]:
    class ImeCompareHandler(BaseHTTPRequestHandler):
        server_version = "AIOS-IME-Compare/1.0"

        def _write_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(
                json_safe(payload), ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(body)

        def _write_static(self, filename: str) -> None:
            path = STATIC_ROOT / filename
            if not path.is_file():
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "静态资源不存在"})
                return
            body = path.read_bytes()
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(body)

        def _read_payload(self) -> dict[str, Any]:
            raw_length = self.headers.get("Content-Length", "0")
            try:
                content_length = int(raw_length)
            except ValueError as error:
                raise ValueError("Content-Length 无效") from error
            if not 0 < content_length <= MAX_REQUEST_BYTES:
                raise ValueError("请求体为空或超过 1 MiB")
            try:
                payload = json.loads(self.rfile.read(content_length))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("请求体不是有效 JSON") from error
            if not isinstance(payload, dict):
                raise ValueError("请求体必须是 JSON 对象")
            return payload

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
            path = urlparse(self.path).path
            if path in STATIC_ROUTES:
                self._write_static(STATIC_ROUTES[path])
                return
            if path == "/api/config":
                self._write_json(HTTPStatus.OK, service.initial_config())
                return
            if path == "/api/health":
                self._write_json(HTTPStatus.OK, service.health())
                return
            if path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "页面不存在"})

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
            path = urlparse(self.path).path
            try:
                payload = self._read_payload()
                if path == "/api/compare":
                    self._write_json(HTTPStatus.OK, service.compare(payload))
                    return
                if path == "/api/unload":
                    service.shutdown()
                    self._write_json(HTTPStatus.OK, {"ok": True})
                    return
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "API 不存在"})
            except ValueError as error:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except Exception as error:
                self._write_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"{type(error).__name__}: {error}"},
                )

        def log_message(self, message: str, *args: Any) -> None:
            print(f"[web] {self.address_string()} {message % args}")

    return ImeCompareHandler


def main() -> None:
    args = parse_args()
    if not STATIC_ROOT.is_dir():
        raise SystemExit(f"前端静态目录不存在：{STATIC_ROOT}")
    slots = build_default_slots(args)
    try:
        profiles = build_profiles(args, slots)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    service = ImeCompareService(slots, profiles=profiles, demo=args.demo)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    server.daemon_threads = True
    display_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    url = f"http://{display_host}:{args.port}"
    print("AIOS-IME 推理对比台已启动")
    print(f"地址：{url}")
    print("A/B 默认串行推理；模型首次加载和首次 JIT 不计入 engine latency。")
    if args.demo:
        print("当前为 demo 模式，不会加载模型或占用 GPU。")
    if args.open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\n正在关闭前端并释放模型显存……")
    finally:
        server.shutdown()
        server.server_close()
        service.shutdown()


if __name__ == "__main__":
    main()
