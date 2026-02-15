"""
Task - a unit experiment with specific configuration and execution logic.
"""
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from src.llm_utils import LLMModel, Provider
from src.evaluation import EvaluatorFactory
from src.evaluation.constants import DEFAULT_EVALUATION_MODEL
from src.prompt_processor import create_processor
from src.dataloader import load_dataset
from src.utils.logger import get_logger
from src.utils.experiment import get_new_experiment_data_dir
from src.experiment.constants import TASK_DATA_DIR, DEFAULT_TASK_NAME
from src.experiment.task_config import TaskConfig

logger = get_logger(__name__)


class Task:
    """
    A Task represents a single unit experiment with specific configuration.
    
    Each task has:
    - Dataset configuration (loaded via the dataloader)
    - Output data path (where to save results)
    - Evaluation model (for scoring responses)
    - Processing strategy (single processor per task)
    - Processing model (LLM for processing, if applicable)
    """
    
    def __init__(self, config: TaskConfig):
        """
        Initialize a Task from a TaskConfig.
        
        Args:
            config: Complete task configuration.
        """
        self.config = config
        # Auto-generate name if not provided
        if config.name:
            self.name = config.name
        else:
            model_short = config.target_model.model_id.split('/')[-1].lower()
            parts = [config.processor.strategy, config.dataloader.dataset_name, model_short]
            if config.processor.rephrase_first:
                parts.append("rephrase")
            if config.processor.is_repeating:
                parts.append("repeat")
            self.name = "_".join(parts)
        self.dataset_name = config.dataloader.dataset_name
        
        # Evaluation settings
        self.evaluation_model = config.evaluation.model_config.model or DEFAULT_EVALUATION_MODEL
        self.evaluation_method = config.evaluation.method
        
        # Processing settings (single strategy)
        self.strategy = config.processor.strategy
        self.processing_model = config.processor.model_config.model
        self.target_model = config.target_model
        
        # Set up experiment directory: experiment_data/{dataset_name}/{timestamp}_{task_name}/
        base_dir = TASK_DATA_DIR / self.dataset_name
        self.experiment_dir = Path(get_new_experiment_data_dir(
            experiment_dir=str(base_dir),
            dataset=None,  # Already in the path via base_dir
            model=self.name,
        ))
        
        # Lazy initialization
        self.evaluator = None
        self.processor = None
        
        # Results storage
        self.results: Dict[str, Any] = {}
        
        logger.info(f"Initialized Task '{self.name}'")
        logger.info(f"Dataset: {self.dataset_name}, Strategy: {self.strategy}")
        logger.info(f"Experiment Dir: {self.experiment_dir}")

    @property
    def requires_cluster(self) -> bool:
        """True if any model in this task needs a local/cluster GPU server."""
        NON_API_PROVIDERS = {Provider.LOCAL, Provider.NU_CLUSTER}
        models = [self.target_model, self.processing_model, self.evaluation_model]
        return any(
            isinstance(m, LLMModel) and m.provider in NON_API_PROVIDERS
            for m in models
            if m is not None
        )

    def load_prompts(self) -> list[Dict[str, Any]]:
        """Load prompts via the dataloader using the DataLoaderConfig."""
        return load_dataset(self.config.dataloader)

    def run_task(self) -> Dict[str, Any]:
        """Execute the task."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Running Task: {self.name}")
        logger.info(f"{'='*60}\n")
        
        # Step 1: Create experiment directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Experiment directory: {self.experiment_dir}")
        
        # Set up file logging to task.log in experiment directory
        log_file = self.experiment_dir / "task.log"
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            fmt='[%(levelname)s] %(asctime)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        # Add file handler to root logger so all loggers write to the file
        logging.root.addHandler(file_handler)
        
        try:
            return self._execute_task()
        finally:
            # Clean up file handler
            file_handler.flush()
            file_handler.close()
            logging.root.removeHandler(file_handler)
            logger.info(f"Task log saved to {log_file}")

    def _execute_task(self) -> Dict[str, Any]:
        """Internal task execution logic.
        
        Pipeline:
          1. Load prompts from dataset
          2. Process prompts through processor
          3. Generate target model responses
          4. Evaluate (classify) responses
          5. Save results
        """
        from src.llm_utils import LLMServiceFactory
        
        # Step 1: Load prompts via dataloader
        prompts = self.load_prompts()
        if not prompts:
            logger.error("No prompts loaded")
            return {}
        
        logger.info(f"Loaded {len(prompts)} prompts from dataset: {self.dataset_name}")
        
        if not self.target_model:
            logger.error("No target model specified")
            return {}

        # Initialize evaluator
        if self.evaluator is None:
            self.evaluator = EvaluatorFactory.create(
                method=self.evaluation_method,
                model=self.evaluation_model
            )
        
        # Determine prompt IDs
        prompt_ids = [str(p.get('id', i)) for i, p in enumerate(prompts)]
        
        # Step 2: Process prompts through processor
        proc_config = self.config.processor
        
        if self.processor is None:
            self.processor = create_processor(
                name=self.strategy,
                model=self.processing_model,
                rephrase_first=proc_config.rephrase_first,
                is_repeating=proc_config.is_repeating,
                temperature=proc_config.model_config.temperature,
                max_tokens=proc_config.model_config.max_tokens,
                use_few_shot=proc_config.use_few_shot,
                num_parts=proc_config.num_parts,
                num_symbols=proc_config.num_symbols,
                experiment_dir=proc_config.experiment_dir,
            )
        
        prompt_texts = [p['prompt'] for p in prompts]
        
        logger.info(f"Processing {len(prompt_texts)} prompts with strategy: {self.strategy}")
        if proc_config.rephrase_first:
            logger.info("  Rephrase pre-step: enabled")
        if proc_config.is_repeating:
            logger.info("  Repeat post-step: enabled")
        
        processed_texts = self.processor.batch_process(prompt_texts)
        strategy_str = str(self.strategy)
        
        # Step 3: Generate target model responses
        logger.info(f"Generating responses from target model: {self.target_model.model_id}")
        target_service = LLMServiceFactory.create(self.target_model)
        target_inputs = [
            (pid, proc_text)
            for pid, proc_text in zip(prompt_ids, processed_texts)
        ]
        target_results = target_service.batch_generate(target_inputs)
        response_dict = {pid: resp for pid, resp in target_results}
        
        # Step 4: Evaluate (classify) responses
        logger.info("Running evaluation...")
        detailed_df, statistics = self.evaluator.evaluate(
            prompts=prompts,
            processed_prompts=processed_texts,
            responses=response_dict,
        )

        # Step 5: Add metadata columns to DataFrame
        detailed_df['target_model'] = self.target_model.model_id
        detailed_df['processing_strategy'] = strategy_str
        
        # Store results for saving
        self.detailed_df = detailed_df
        self.statistics = statistics
        self._strategy_str = strategy_str
        self.num_prompts = len(prompts)
        
        self._save_results_to_folder()
        self._print_summary()
        
        # Return a simplified dict for compatibility
        return {
            'task_name': self.name,
            'experiment_dir': str(self.experiment_dir),
            'statistics': statistics
        }
    
    def _save_results_to_folder(self):
        """Save task results to folder structure: detailed_results.csv and result.json."""
        if not hasattr(self, 'detailed_df') or self.detailed_df is None:
            return
        
        logger.info(f"Saving results to {self.experiment_dir}")
        
        detailed_csv_file = self.experiment_dir / "detailed_results.csv"
        result_json_file = self.experiment_dir / "result.json"
        
        # 1. Save detailed results to CSV
        self.detailed_df.to_csv(detailed_csv_file, index=False)
        logger.info(f"  - Saved detailed_results.csv ({len(self.detailed_df)} rows)")
        
        # 2. Build and save result.json (using config for serialization)
        result_data = {
            'configuration': self.config.to_dict(),
            'result': self.statistics
        }
        
        with open(result_json_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2)
        
        logger.info(f"  - Saved result.json")
        logger.info("✅ Saved task results")

    def _print_summary(self):
        """Print summary."""
        stats = getattr(self, 'statistics', {})
        logger.info(f"Task Complete. Stats: {stats}")

    def __repr__(self) -> str:
        return f"Task(name='{self.name}', strategy='{self.strategy}')"
