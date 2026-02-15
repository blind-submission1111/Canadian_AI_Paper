"""
Constants for the dataloader package.
"""
from pathlib import Path
from typing import Final


# =============================================================================
# Standard field names for normalized datasets
# =============================================================================

# Every dataset is normalized to dicts with these field names
PROMPT_FIELD: Final[str] = "prompt"
DATA_ID_FIELD: Final[str] = "data_id"
TARGET_FIELD: Final[str] = "target"         # Affirmative prefix (if available)
CATEGORY_FIELD: Final[str] = "category"     # Semantic/harm category
SOURCE_FIELD: Final[str] = "source"         # Original source of the behavior


# =============================================================================
# Dataset paths
# =============================================================================

DATASETS_DIR: Final[Path] = Path(__file__).resolve().parent.parent / "data" / "datasets"

# HarmBench standard dataset (filtered to 'standard' functional category)
HARMBENCH_STANDARD_DIR: Final[Path] = DATASETS_DIR / "harmbench_standard"
HARMBENCH_STANDARD_FILE: Final[Path] = HARMBENCH_STANDARD_DIR / "harmbench_standard.csv"

# JailbreakBench HuggingFace dataset
JBB_HF_DATASET_ID: Final[str] = "dedeswim/JBB-Behaviors"
JBB_HF_CONFIG: Final[str] = "behaviors"
JBB_HF_SPLIT: Final[str] = "harmful"


# =============================================================================
# Known dataset names
# =============================================================================

KNOWN_DATASETS: Final[list[str]] = [
    "harmbench_standard",
    "jailbreakbench",
]
