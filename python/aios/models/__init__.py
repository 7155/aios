from .base import BaseLLMModel
from .config import ModelConfig
from .weight import load_weights


def create_model(model_path: str, config: ModelConfig) -> BaseLLMModel:
    model_name = model_path.lower()
    if config.model_type == "qwen3" or "qwen3" in model_name:
        from .qwen3 import Qwen3ForCausalLM

        return Qwen3ForCausalLM(config)
    if config.model_type == "minimind_ime_v3" or "minimind-ime" in model_name:
        from .minimind_ime import MiniMindIMEForCausalLM

        return MiniMindIMEForCausalLM(config)
    
    raise ValueError(f"Unsupported model: {model_path}")


__all__ = ["BaseLLMModel", "ModelConfig", "load_weights", "create_model"]
