"""
Configuration for an experiment (collection of tasks).
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.experiment.task_config import TaskConfig


@dataclass
class ExperimentConfig:
    """
    Top-level experiment configuration.
    
    Contains experiment-level settings and a list of task configurations.
    Loaded from a JSON file via `python main.py experiment_config/config.json`.
    
    Attributes:
        num_api_workers: Max concurrent API tasks (target models use cloud APIs).
        num_cluster_workers: Max concurrent cluster tasks. If None (default),
            auto-derived from total server instances across all cluster models.
        tasks: List of TaskConfig for each experiment task.
        cluster_server: Optional per-model dict of overrides for ClusterServerConfig.
                        Keys are model identifiers (e.g. "vicuna-13b-cluster",
                        "llama3-1-8b-cluster"). Values are dicts of config fields
                        to override (e.g. {"num_instances": 2}).
                        Can also be a flat dict applied to all models (backward compat).
    """
    num_api_workers: int = 8
    num_cluster_workers: Optional[int] = None  # None = auto from num_instances
    tasks: List[TaskConfig] = field(default_factory=list)
    cluster_server: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        """Create ExperimentConfig from a dictionary."""
        if not data:
            return cls()
        
        data = data.copy()
        
        # Backward compat: map old field name to new
        if "num_parallel_tasks" in data and "num_api_workers" not in data:
            data["num_api_workers"] = data.pop("num_parallel_tasks")
        elif "num_parallel_tasks" in data:
            data.pop("num_parallel_tasks")
        
        # Parse list of task configs
        if "tasks" in data and isinstance(data["tasks"], list):
            data["tasks"] = [
                TaskConfig.from_dict(t) if isinstance(t, dict) else t
                for t in data["tasks"]
                if not (isinstance(t, dict) and set(t.keys()) == {"_comment"})
            ]
        
        return cls(**data)
    
    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "ExperimentConfig":
        """Load ExperimentConfig from a JSON file."""
        import json
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        d = {
            "num_api_workers": self.num_api_workers,
            "num_cluster_workers": self.num_cluster_workers,
            "tasks": [t.to_dict() for t in self.tasks],
        }
        if self.cluster_server is not None:
            d["cluster_server"] = self.cluster_server
        return d
    
    def to_json(self, path: Union[str, Path]) -> None:
        """Save ExperimentConfig to a JSON file."""
        import json
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

