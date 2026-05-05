"""
Configuration for the dataloader.
"""
from dataclasses import dataclass
from typing import Any, List, Optional, Union


@dataclass
class DataLoaderConfig:
    """
    Configuration for dataset loading.
    
    Attributes:
        dataset_name: Name of the dataset to load
                      ('harmbench_standard' or 'jailbreakbench').
        prompt_ids: Optional subset of prompt IDs to load.
                    Can be a range tuple (start, end) or a list of IDs.
        max_samples: Optional maximum number of samples to load.
    """
    dataset_name: str = "harmbench_standard"
    prompt_ids: Optional[Union[List[int], tuple]] = None
    max_samples: Optional[int] = None
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataLoaderConfig":
        """Create DataLoaderConfig from a dictionary."""
        if not data:
            return cls()
        return cls(**data)
