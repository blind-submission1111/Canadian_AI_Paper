"""
DataLoader for loading and normalizing jailbreaking experiment datasets.

Supports:
    - HarmBench standard: loaded from local CSV (159 standard behaviors)
    - JailbreakBench: loaded from HuggingFace (100 harmful behaviors)

All datasets are normalized to a standard format with fields:
    - data_id: unique identifier for the behavior
    - prompt: the harmful behavior prompt
    - target: affirmative prefix (if available)
    - category: semantic/harm category
    - source: original source of the behavior
"""
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

from src.utils.logger import get_logger
from .constants import (
    PROMPT_FIELD,
    DATA_ID_FIELD,
    TARGET_FIELD,
    CATEGORY_FIELD,
    SOURCE_FIELD,
    HARMBENCH_STANDARD_FILE,
    JBB_HF_DATASET_ID,
    JBB_HF_CONFIG,
    JBB_HF_SPLIT,
)
from .dataloader_config import DataLoaderConfig

logger = get_logger(__name__)


def load_dataset(config: DataLoaderConfig) -> List[Dict[str, Any]]:
    """
    Load and normalize a dataset based on the config.
    
    Args:
        config: DataLoaderConfig specifying which dataset to load.
        
    Returns:
        List of dicts, each with keys: data_id, prompt, target, category, source.
        
    Raises:
        ValueError: If dataset_name is unknown.
    """
    name = config.dataset_name.lower().strip()
    
    if name in ("harmbench_standard", "harmbench"):
        data = _load_harmbench_standard()
    elif name in ("jailbreakbench", "jbb"):
        data = _load_jailbreakbench()
    else:
        raise ValueError(
            f"Unknown dataset: '{config.dataset_name}'. "
            f"Available: 'harmbench_standard', 'jailbreakbench'"
        )
    
    # Apply filtering
    data = _filter_data(data, config.prompt_ids, config.max_samples)
    
    logger.info(f"Loaded {len(data)} samples from '{config.dataset_name}'")
    return data


def _load_harmbench_standard() -> List[Dict[str, Any]]:
    """
    Load HarmBench standard behaviors from local CSV.
    
    The CSV is pre-filtered to only 'standard' functional category behaviors
    from the official HarmBench test set.
    
    Returns:
        List of normalized dicts.
    """
    if not HARMBENCH_STANDARD_FILE.exists():
        raise FileNotFoundError(
            f"HarmBench standard dataset not found at {HARMBENCH_STANDARD_FILE}. "
            f"Please ensure the dataset CSV has been created."
        )
    
    data = []
    with open(HARMBENCH_STANDARD_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            data.append({
                DATA_ID_FIELD: row.get("BehaviorID", str(i)),
                PROMPT_FIELD: row["Behavior"],
                TARGET_FIELD: "",  # HarmBench doesn't have affirmative targets
                CATEGORY_FIELD: row.get("SemanticCategory", ""),
                SOURCE_FIELD: "harmbench",
            })
    
    logger.info(f"Loaded {len(data)} standard behaviors from HarmBench")
    return data


def _load_jailbreakbench() -> List[Dict[str, Any]]:
    """
    Load JailbreakBench harmful behaviors from HuggingFace.
    
    Dataset: dedeswim/JBB-Behaviors (harmful split, 100 behaviors).
    
    Returns:
        List of normalized dicts.
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required to load JailbreakBench. "
            "Install it with: pip install datasets"
        )
    
    logger.info(f"Loading JailbreakBench from HuggingFace: {JBB_HF_DATASET_ID}")
    dataset = hf_load_dataset(JBB_HF_DATASET_ID, JBB_HF_CONFIG, split=JBB_HF_SPLIT)
    df = dataset.to_pandas()
    
    data = []
    for _, row in df.iterrows():
        data.append({
            DATA_ID_FIELD: row.get("Behavior", str(row.get("Index", ""))),
            PROMPT_FIELD: row["Goal"],
            TARGET_FIELD: row.get("Target", ""),
            CATEGORY_FIELD: row.get("Category", ""),
            SOURCE_FIELD: row.get("Source", "jailbreakbench"),
        })
    
    logger.info(f"Loaded {len(data)} harmful behaviors from JailbreakBench")
    return data


def _filter_data(
    data: List[Dict[str, Any]],
    prompt_ids: Optional[Union[List[int], Tuple[int, int]]] = None,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Filter dataset by prompt IDs and/or max samples.
    
    Args:
        data: Full dataset.
        prompt_ids: If a tuple (start, end), slice by index range.
                    If a list, select items at those indices.
        max_samples: Maximum number of samples to return.
        
    Returns:
        Filtered dataset.
    """
    if prompt_ids is not None:
        if isinstance(prompt_ids, tuple) and len(prompt_ids) == 2:
            start, end = prompt_ids
            data = data[start:end]
        elif isinstance(prompt_ids, list):
            data = [data[i] for i in prompt_ids if i < len(data)]
    
    if max_samples is not None and len(data) > max_samples:
        data = data[:max_samples]
    
    return data
