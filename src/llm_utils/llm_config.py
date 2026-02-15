from dataclasses import dataclass, field
from typing import Optional, Any
from src.llm_utils.llm_model import LLMModel


@dataclass
class LLMConfig:
    """Configuration for LLM requests."""
    model: LLMModel = LLMModel.GPT_4O_MINI
    max_tokens: int = 16384
    temperature: float = 0.0  # Deterministic output for reproducibility
    top_p: float = 1.0
    top_k: int = 0  # Mainly for local models
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: list[str] = field(default_factory=list)
    seed: Optional[int] = 42
    n_completions: int = 1
    stream: bool = False
    
    # vLLM-specific config (for cluster/local GPU inference)
    max_model_len: Optional[int] = 1024 * 12  # Max sequence length, None = model default
    gpu_memory_utilization: float = 0.9  # Fraction of GPU memory to use
    quantization: Optional[str] = None  # e.g., "awq", "gptq", "fp8", None = no quantization
    dtype: Optional[str] = None  # e.g., "half", "float16", "bfloat16", None = auto
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMConfig":
        """
        Create LLMConfig from a dictionary, handling model string conversion.
        
        Args:
            data: Dict with config values. 'model' can be a string (model_id)
                  or an LLMModel enum.
        
        Returns:
            LLMConfig instance
        """
        if not data:
            return cls()
        
        data = data.copy()  # Don't modify original
        
        # Convert model string to LLMModel enum
        if "model" in data and isinstance(data["model"], str):
            data["model"] = LLMModel.from_string(data["model"])
        
        return cls(**data)