"""
Experiment class for managing and executing multiple experiment tasks.

An Experiment can contain multiple Tasks, each with their own configuration.
Tasks are classified into API or cluster queues and executed concurrently
using asyncio with model-aware concurrency control.
"""
import asyncio
import traceback
from typing import List, Dict, Any, Optional, Set
from collections import deque

from src.utils.logger import get_logger
from src.llm_utils.llm_model import Provider
from .task import Task

logger = get_logger(__name__)


class Experiment:
    """
    Experiment class for managing multiple experiment tasks.
    
    An Experiment orchestrates multiple Tasks, each representing a unit experiment
    with specific configuration (input data, processing strategy, target model, etc.).
    
    Tasks are auto-classified into two queues:
    - API queue: tasks where all models use cloud API providers (OpenAI, Anthropic, Google)
    - Cluster queue: tasks where any model uses LOCAL or NU_CLUSTER provider
    
    Each queue has its own concurrency limit (semaphore) to prevent
    resource contention on shared GPU servers.
    """
    
    def __init__(self, cluster_server_overrides: Optional[Dict[str, Any]] = None,
                 num_api_workers: int = 8, num_cluster_workers: Optional[int] = None):
        """Initialize the experiment.
        
        Args:
            cluster_server_overrides: Optional dict of overrides for
                ClusterServerConfig. Can be per-model nested dict.
            num_api_workers: Max concurrent API tasks (default: 8).
            num_cluster_workers: Max concurrent cluster tasks. If None (default),
                auto-derived from total server instances across all models.
        """
        self.tasks: deque[Task] = deque()
        self.results: List[Dict[str, Any]] = []
        self.cluster_server_overrides = cluster_server_overrides
        self.num_api_workers = num_api_workers
        self.num_cluster_workers = num_cluster_workers  # None = auto
        self._server_manager = None
    
    def add_task_to_front(self, task: Task):
        """
        Add a task to the front of the execution queue (will execute first).
        
        Args:
            task: Task instance to add
        """
        self.tasks.appendleft(task)
        logger.info(f"Added task '{task.name}' to front of queue (total: {len(self.tasks)} tasks)")
    
    def add_task_to_tail(self, task: Task):
        """
        Add a task to the tail of the execution queue (will execute last).
        
        Args:
            task: Task instance to add
        """
        self.tasks.append(task)
        logger.info(f"Added task '{task.name}' to tail of queue (total: {len(self.tasks)} tasks)")
    
    def _find_cluster_models(self, tasks: List[Task]) -> Set:
        """Scan tasks for cluster models that need vLLM servers."""
        from src.llm_utils.llm_model import LLMModel
        cluster_models = set()
        for task in tasks:
            if task.target_model and task.target_model.provider == Provider.NU_CLUSTER:
                cluster_models.add(task.target_model)
            if isinstance(task.evaluation_model, LLMModel) and task.evaluation_model.provider == Provider.NU_CLUSTER:
                cluster_models.add(task.evaluation_model)
            if isinstance(task.processing_model, LLMModel) and task.processing_model.provider == Provider.NU_CLUSTER:
                cluster_models.add(task.processing_model)
        return cluster_models
    
    def _get_cluster_target_model(self, task: Task):
        """Get the primary cluster model for a task (for sub-queue routing)."""
        from src.llm_utils.llm_model import LLMModel
        NON_API = {Provider.LOCAL, Provider.NU_CLUSTER}
        # Prefer target model, then processing, then evaluation
        for model in [task.target_model, task.processing_model, task.evaluation_model]:
            if isinstance(model, LLMModel) and model.provider in NON_API:
                return model
        return None
    
    def _setup_cluster_servers(self, cluster_models: Set) -> None:
        """Start vLLM servers for all cluster models (multi-instance)."""
        from src.llm_utils.cluster_server_manager import ClusterModelServerManager
        from src.llm_utils.llm_service_factory import LLMServiceFactory
        
        self._server_manager = ClusterModelServerManager(
            config_overrides=self.cluster_server_overrides
        )
        
        for model in cluster_models:
            logger.info(f"Starting vLLM server(s) for {model.model_id}...")
            endpoints = self._server_manager.start_server(model)
            logger.info(f"  -> {len(endpoints)} instance(s): {endpoints}")
        
        # Register with factory so tasks auto-get endpoints via round-robin
        LLMServiceFactory.set_server_manager(self._server_manager)
    
    def _teardown_cluster_servers(self) -> None:
        """Shut down all vLLM servers."""
        if self._server_manager:
            logger.info("Shutting down cluster vLLM servers...")
            self._server_manager.shutdown_all()
            self._server_manager = None
    
    async def _run_task_safe(self, task: Task, semaphore: asyncio.Semaphore,
                             task_idx: int, total: int, queue_name: str) -> Dict[str, Any]:
        """
        Execute a single task with semaphore-based concurrency control.
        
        Runs the synchronous task.run_task() in a thread pool to avoid
        blocking the asyncio event loop.
        
        Args:
            task: Task to execute.
            semaphore: Concurrency-limiting semaphore for this queue.
            task_idx: 1-based index of the task.
            total: Total number of tasks.
            queue_name: "API" or "Cluster" for logging.
        
        Returns:
            Task result dict, or error dict on failure.
        """
        async with semaphore:
            logger.info(f"\n{'~'*70}")
            logger.info(f"[{queue_name}] Executing Task {task_idx}/{total}: {task.name}")
            logger.info(f"{'~'*70}\n")
            
            try:
                # Run synchronous task in thread pool
                result = await asyncio.to_thread(task.run_task)
                return result
            except Exception as e:
                logger.error(f"Error executing task '{task.name}': {e}")
                logger.error(traceback.format_exc())
                return {
                    'task_name': task.name,
                    'error': str(e),
                    'status': 'failed'
                }
    
    async def run_experiment_async(self, num_of_tasks: int = None) -> List[Dict[str, Any]]:
        """
        Execute tasks using asyncio with multi-queue concurrency control.
        
        Tasks are classified into:
        - API queue: tasks using cloud API providers (high concurrency)
        - Per-model cluster queues: each cluster model gets its own queue
          with concurrency matching its num_instances
        
        Args:
            num_of_tasks: Number of tasks to execute. If None, execute all.
        
        Returns:
            List of all task results.
        """
        if not self.tasks:
            logger.warning("No tasks to execute")
            return []
        
        # Determine how many tasks to run
        if num_of_tasks is None:
            task_count = len(self.tasks)
        else:
            task_count = min(num_of_tasks, len(self.tasks))
        
        # Pop tasks from queue
        tasks_to_run = [self.tasks.popleft() for _ in range(task_count)]
        
        # Classify tasks into API and cluster queues
        api_tasks = [t for t in tasks_to_run if not t.requires_cluster]
        cluster_tasks = [t for t in tasks_to_run if t.requires_cluster]
        
        self.results = []
        
        try:
            # Start cluster servers if needed
            model_semaphores: Dict = {}  # LLMModel -> asyncio.Semaphore
            if cluster_tasks:
                cluster_models = self._find_cluster_models(cluster_tasks)
                if cluster_models:
                    model_names = [m.model_id for m in cluster_models]
                    logger.info(f"Cluster models detected: {model_names}")
                    self._setup_cluster_servers(cluster_models)
                    
                    # Create per-model semaphores based on instance count
                    for model in cluster_models:
                        n_instances = self._server_manager.get_num_instances(model)
                        model_semaphores[model] = asyncio.Semaphore(n_instances)
                        logger.info(f"  {model.model_id}: {n_instances} worker(s)")
            
            # Compute total cluster workers for logging
            total_cluster_workers = sum(
                self._server_manager.get_num_instances(m) 
                for m in model_semaphores
            ) if model_semaphores else 0
            
            logger.info(f"\n{'='*70}")
            logger.info(f"Total tasks to execute: {task_count}")
            logger.info(f"  API tasks: {len(api_tasks)} (max {self.num_api_workers} concurrent)")
            logger.info(f"  Cluster tasks: {len(cluster_tasks)} ({total_cluster_workers} total workers)")
            logger.info(f"Execution Mode: ASYNC MULTI-QUEUE")
            logger.info(f"{'='*70}\n")
            
            # Create API semaphore
            api_semaphore = asyncio.Semaphore(self.num_api_workers)
            
            # Build coroutines for all tasks
            coroutines = []
            
            # API tasks
            for i, task in enumerate(api_tasks, 1):
                coroutines.append(
                    self._run_task_safe(task, api_semaphore, i, len(api_tasks), "API")
                )
            
            # Cluster tasks — route to per-model semaphore
            for i, task in enumerate(cluster_tasks, 1):
                cluster_model = self._get_cluster_target_model(task)
                if cluster_model and cluster_model in model_semaphores:
                    sem = model_semaphores[cluster_model]
                    queue_name = f"Cluster:{cluster_model.model_id.split('/')[-1]}"
                else:
                    # Fallback: single semaphore with 1 worker
                    sem = asyncio.Semaphore(1)
                    queue_name = "Cluster"
                coroutines.append(
                    self._run_task_safe(task, sem, i, len(cluster_tasks), queue_name)
                )
            
            # Run all tasks concurrently (errors are isolated per-task)
            self.results = await asyncio.gather(*coroutines)
            
            # Convert to list (gather returns tuple-like)
            self.results = list(self.results)
        
        finally:
            # Always shut down cluster servers, even on error
            self._teardown_cluster_servers()
        
        # Print overall summary
        self._print_experiment_summary()
        
        return self.results
    
    def run_experiment(self, num_of_tasks: int = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Execute tasks (sync wrapper for backward compatibility).
        
        Calls run_experiment_async() via asyncio.run().
        
        Args:
            num_of_tasks: Number of tasks to execute. If None, execute all.
        
        Returns:
            List of all task results.
        """
        return asyncio.run(self.run_experiment_async(num_of_tasks=num_of_tasks))
    
    def _print_experiment_summary(self):
        """Print overall experiment summary."""
        logger.info(f"\n{'='*70}")
        logger.info(f"EXPERIMENT SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Total tasks executed: {len(self.results)}")
        
        # Count successes and failures
        successful = sum(1 for r in self.results if isinstance(r, dict) and r.get('status') != 'failed')
        failed = len(self.results) - successful
        
        logger.info(f"Successful: {successful}")
        if failed > 0:
            logger.info(f"Failed: {failed}")
            # Log failed task names
            for r in self.results:
                if isinstance(r, dict) and r.get('status') == 'failed':
                    logger.info(f"  ✗ {r.get('task_name', 'unknown')}: {r.get('error', 'unknown error')}")
        
        # Aggregate statistics
        total_prompts = sum(
            r.get('num_prompts', 0) for r in self.results
            if isinstance(r, dict) and r.get('num_prompts')
        )
        logger.info(f"Total prompts processed: {total_prompts}")
        
        logger.info(f"{'='*70}\n")
    
    def get_task_count(self) -> int:
        """Get the number of tasks in the queue."""
        return len(self.tasks)
    
    def clear_tasks(self):
        """Clear all tasks from the queue."""
        self.tasks.clear()
        logger.info("Cleared all tasks from queue")
    
    def __repr__(self) -> str:
        return f"Experiment(tasks={len(self.tasks)})"
