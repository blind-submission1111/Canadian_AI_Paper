"""
Configuration for evaluation.
"""
from dataclasses import dataclass, field
from typing import Any

from src.llm_utils.llm_config import LLMConfig
from src.llm_utils.llm_model import LLMModel


@dataclass
class EvaluationConfig:
    """
    Configuration for jailbreak evaluation.
    
    Attributes:
        method: Evaluation method to use ('harmbench' or 'jailbreakbench').
        model_config: LLM configuration for the judge model.
                      Defaults to GPT-4o with temperature=0.0 for deterministic judging.
    """
    method: str = "harmbench"
    model_config: LLMConfig = field(default_factory=lambda: LLMConfig(
        model=LLMModel.GPT_5_NANO,
        temperature=0.0,
    ))
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationConfig":
        """Create EvaluationConfig from a dictionary."""
        if not data:
            return cls()
        
        data = data.copy()
        
        # Parse nested model_config
        if "model_config" in data and isinstance(data["model_config"], dict):
            data["model_config"] = LLMConfig.from_dict(data["model_config"])
        
        return cls(**data)
