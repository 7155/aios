#!/usr/bin/env python3
"""Conceptual and optional real PyTorch CUDA Graph demo."""

from __future__ import annotations


def conceptual() -> None:
    eager_launches = ["qkv", "norm", "attention", "mlp"] * 14
    print("eager CPU launch calls:", len(eager_launches))
    print("graph replay calls:     1 (kernels inside graph remain", len(eager_launches), ")")


def cuda_demo() -> None:
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        print("CUDA unavailable; skipped real graph capture")
        return
    model = torch.nn.Linear(64, 64, bias=False).cuda().half().eval()
    static_x = torch.empty(8, 64, device="cuda", dtype=torch.float16)
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        for _ in range(3):
            static_y = model(static_x)
    side.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_y = model(static_x)
    new_x = torch.randn_like(static_x)
    expected = model(new_x)
    static_x.copy_(new_x)
    graph.replay()
    torch.cuda.synchronize()
    print("max graph/eager diff:", (static_y - expected).abs().max().item())


if __name__ == "__main__":
    conceptual()
    cuda_demo()
