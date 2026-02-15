"""
Configuration for experiment tasks.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

from src.llm_utils.llm_model import LLMModel
from src.evaluation.evaluation_config import EvaluationConfig
from src.prompt_processor.processor_config import ProcessorConfig
from src.dataloader.dataloader_config import DataLoaderConfig


@dataclass
class TaskConfig:
    """
    Top-level experiment configuration.
    
    Composes DataLoaderConfig, ProcessorConfig, and EvaluationConfig
    for a complete experiment setup.
    
    Attributes:
        name: Optional task name. If None or empty, auto-generated as
              '{strategy}_{dataset}_{model}' (e.g., 'llm_set_theory_harmbench_standard_gpt-4o-mini').
        target_model: The LLM to attack/test.
        dataloader: Dataset loading configuration (dataset name, filtering).
        processor: Processor configuration (strategy, model, parameters).
        evaluation: Evaluation configuration (method, judge model).
    """
    name: Optional[str] = None
    target_model: LLMModel = LLMModel.GPT_4O
    
    # Composed configs
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    processor: ProcessorConfig = field(default_factory=ProcessorConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskConfig":
        """Create TaskConfig from a dictionary."""
        if not data:
            return cls()
        
        data = data.copy()
        
        # Convert target_model string to enum
        if "target_model" in data and isinstance(data["target_model"], str):
            data["target_model"] = LLMModel.from_string(data["target_model"])
        
        # Parse nested configs
        if "dataloader" in data and isinstance(data["dataloader"], dict):
            data["dataloader"] = DataLoaderConfig.from_dict(data["dataloader"])
        
        if "processor" in data and isinstance(data["processor"], dict):
            data["processor"] = ProcessorConfig.from_dict(data["processor"])
        
        if "evaluation" in data and isinstance(data["evaluation"], dict):
            data["evaluation"] = EvaluationConfig.from_dict(data["evaluation"])
        
        config = cls(**data)
        
        # Auto-generate name if not provided
        if not config.name:
            # Derive short model name (e.g., 'gpt-4o', 'llama-3.1-8b')
            model_short = config.target_model.model_id.split('/')[-1].lower()
            parts = [config.processor.strategy, config.dataloader.dataset_name, model_short]
            if config.processor.rephrase_first:
                parts.append("rephrase")
            if config.processor.is_repeating:
                parts.append("repeat")
            config.name = "_".join(parts)
        
        return config
    
    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "TaskConfig":
        """Load TaskConfig from a JSON file."""
        import json
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        from dataclasses import asdict
        
        def _serialize_value(val: Any) -> Any:
            """Recursively convert LLMModel enums to model_id strings."""
            if isinstance(val, LLMModel):
                return val.value[0]  # model_id string
            elif isinstance(val, dict):
                return {k: _serialize_value(v) for k, v in val.items()}
            elif isinstance(val, (list, tuple)):
                return [_serialize_value(v) for v in val]
            return val
        
        return _serialize_value(asdict(self))
    
    def to_json(self, path: Union[str, Path]) -> None:
        """Save TaskConfig to a JSON file."""
        import json
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
