#!/usr/bin/env python3
"""Configuration and KV-budget walkthrough for Lesson 11."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    layers: int = 14
    hidden: int = 768
    q_heads: int = 12
    kv_heads: int = 4
    head_dim: int = 64
    bytes_per_value: int = 2

    def validate(self) -> None:
        if self.hidden != self.q_heads * self.head_dim:
            raise ValueError("hidden != q_heads * head_dim")

    @property
    def kv_bytes_per_token(self) -> int:
        return 2 * self.layers * self.kv_heads * self.head_dim * self.bytes_per_value


config = Config()
config.validate()
print("q_proj shape:", (config.hidden, config.hidden))
print("k_proj shape:", (config.kv_heads * config.head_dim, config.hidden))
print("v_proj shape:", (config.kv_heads * config.head_dim, config.hidden))
print("KV bytes/token:", config.kv_bytes_per_token)
print("256 pages KV MiB:", 256 * config.kv_bytes_per_token / 2**20)

print("\nIntentional conflict:")
try:
    Config(q_heads=10).validate()
except ValueError as error:
    print("rejected ->", error)
