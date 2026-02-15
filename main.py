"""
Main entry point for running experiments.

Usage:
    python main.py experiment_config/config.json
    python main.py experiment_config/full_config.json
"""
import sys
import json
from pathlib import Path

from src.experiment.experiment_config import ExperimentConfig
from src.experiment.experiment import Experiment
from src.experiment.task import Task
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_task_from_config(task_config) -> Task:
    """
    Build a Task instance from a TaskConfig dataclass.
    
    Task now accepts TaskConfig directly — no manual parameter mapping needed.
    """
    return Task(config=task_config)


def run_from_config(config_path: str) -> None:
    """
    Load an experiment config JSON and run all tasks.
    
    Args:
        config_path: Path to the experiment config JSON file.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    logger.info(f"Loading experiment config from: {config_path}")
    experiment_config = ExperimentConfig.from_json(config_path)
    
    if not experiment_config.tasks:
        logger.error("No tasks defined in config")
        sys.exit(1)
    
    logger.info(f"Found {len(experiment_config.tasks)} task(s)")
    logger.info(f"API workers: {experiment_config.num_api_workers}, Cluster workers: {experiment_config.num_cluster_workers or 'auto'}")
    
    # Build experiment (pass cluster server overrides if present)
    experiment = Experiment(
        cluster_server_overrides=experiment_config.cluster_server,
        num_api_workers=experiment_config.num_api_workers,
        num_cluster_workers=experiment_config.num_cluster_workers,
    )
    
    for i, task_config in enumerate(experiment_config.tasks):
        task = build_task_from_config(task_config)
        experiment.add_task_to_tail(task)
        logger.info(f"  Task {i+1}: {task.name} | {task_config.processor.strategy} -> {task_config.target_model.value[0]}")
    
    # Run
    results = experiment.run_experiment()
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Experiment complete. {len(results)} task(s) finished.")
    logger.info(f"{'='*70}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <config.json>")
        print("Example: python main.py experiment_config/config.json")
        sys.exit(1)
    
    config_path = sys.argv[1]
    run_from_config(config_path)


if __name__ == "__main__":
    main()
