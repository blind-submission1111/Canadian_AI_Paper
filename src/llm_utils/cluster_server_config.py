"""
Configuration for SLURM-managed vLLM servers on the NU cluster.

Each cluster model has a full set of default values (GPU types, memory, etc.)
via the `for_model()` class method. User overrides merge on top of these presets.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ClusterServerConfig:
    """
    SLURM + vLLM server configuration.
    
    Controls both the SLURM job parameters (partition, GPU, memory, time)
    and the vLLM server settings (port, model len, memory utilization).
    
    Attributes:
        partition: SLURM partition name ('gpu', 'multigpu', 'short').
        gpu_types_preferred: Ordered preference list of GPU types to try.
            The manager picks the first available type from this list.
        gpu_types_excluded: GPU types to never use (too old, too small, etc.).
        num_gpus: Number of GPUs to request per server.
        cpus_per_task: Number of CPU cores per SLURM task.
        mem_gb: Total system RAM in GB to request.
        time_limit: SLURM time limit in HH:MM:SS format.
        port: Base port for vLLM HTTP servers. Instances use port, port+1, etc.
        num_instances: Number of vLLM server instances to launch for this model.
            Each instance gets its own SLURM job and GPU. Default: 1.
        gpu_memory_utilization: Fraction of GPU memory for vLLM KV cache (0.0-1.0).
        max_model_len: Maximum sequence length (context window) for vLLM.
        dtype: Data type for model weights ('half', 'bfloat16', or None=auto).
        conda_env: Conda environment name to activate on the cluster.
        cuda_module: CUDA module to load via `module load`.
    """
    # SLURM settings
    partition: str = "gpu"
    gpu_types_preferred: List[str] = field(
        default_factory=lambda: ["v100-sxm2", "a100", "l40", "a6000"]
    )
    gpu_types_excluded: List[str] = field(
        default_factory=lambda: ["t4", "p100", "k40m", "k80"]
    )
    num_gpus: int = 1
    cpus_per_task: int = 4
    mem_gb: int = 32
    time_limit: str = "48:00:00"
    
    # vLLM server settings
    port: int = 8000
    num_instances: int = 1  # Number of parallel vLLM servers for this model
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 8192
    max_tokens: int = 4096  # Max tokens for generation (must fit in max_model_len)
    dtype: Optional[str] = None  # "half", "bfloat16", or None = auto
    chat_template: Optional[str] = None  # Path to chat template file for models without built-in templates
    server_start_timeout: int = 1800  # Seconds to wait for vLLM server to start (includes SLURM queue time)
    
    # Environment
    conda_env: str = "LLM_prompting"
    cuda_module: str = "cuda/12.1.1"
    
    @classmethod
    def for_model(cls, model, overrides: Optional[Dict[str, Any]] = None) -> "ClusterServerConfig":
        """
        Get full config for a cluster model with sensible defaults.
        
        Each model has a complete preset. The optional `overrides` dict
        merges on top, replacing only the fields explicitly specified.
        
        Args:
            model: LLMModel enum for the cluster model.
            overrides: Optional dict of fields to override.
            
        Returns:
            ClusterServerConfig with preset + overrides applied.
            
        Example:
            # All defaults for Llama-3.1-8B
            config = ClusterServerConfig.for_model(LLMModel.LLAMA3_1_8B_CLUSTER)
            
            # Override just GPU preference and time limit
            config = ClusterServerConfig.for_model(
                LLMModel.LLAMA3_1_8B_CLUSTER,
                overrides={"gpu_types_preferred": ["a100"], "time_limit": "04:00:00"}
            )
        """
        # Import here to avoid circular import
        from src.llm_utils.llm_model import LLMModel
        
        PRESETS: Dict[LLMModel, Dict[str, Any]] = {
            LLMModel.LLAMA3_1_8B_CLUSTER: {
                "partition": "gpu",
                "gpu_types_preferred": ["v100-sxm2", "a100", "l40", "a6000"],
                "gpu_types_excluded": ["t4", "p100", "k40m", "k80"],
                "num_gpus": 1,
                "cpus_per_task": 4,
                "mem_gb": 32,
                "time_limit": "08:00:00",
                "port": 8000,
                "num_instances": 3,
                "gpu_memory_utilization": 0.9,
                "max_model_len": 8192,
                "max_tokens": 4096,
                "dtype": "half",
            },
            LLMModel.LLAMA3_8B_CLUSTER: {
                "partition": "gpu",
                "gpu_types_preferred": ["v100-sxm2", "a100", "l40", "a6000"],
                "gpu_types_excluded": ["t4", "p100", "k40m", "k80"],
                "num_gpus": 1,
                "cpus_per_task": 4,
                "mem_gb": 32,
                "time_limit": "08:00:00",
                "port": 8200,
                "num_instances": 3,
                "gpu_memory_utilization": 0.9,
                "max_model_len": 8192,
                "max_tokens": 4096,
                "dtype": "half",
            },
            LLMModel.VICUNA_13B_CLUSTER: {
                "partition": "gpu",
                "gpu_types_preferred": ["v100-sxm2", "a100", "l40", "a6000"],
                "gpu_types_excluded": ["t4", "p100", "k40m", "k80", "v100-pcie"],
                "num_gpus": 1,
                "cpus_per_task": 4,
                "mem_gb": 48,
                "time_limit": "08:00:00",
                "port": 8100,
                "num_instances": 1,
                "gpu_memory_utilization": 0.9,
                "max_model_len": 4096,
                "max_tokens": 2048,
                "dtype": "half",
                "chat_template": "vicuna_v1.1",
            },
            LLMModel.PIXTRAL_12B: {
                "partition": "gpu",
                "gpu_types_preferred": ["a100", "l40", "a6000"],
                "gpu_types_excluded": ["t4", "p100", "k40m", "k80", "v100-pcie"],
                "num_gpus": 1,
                "cpus_per_task": 4,
                "mem_gb": 48,
                "time_limit": "08:00:00",
                "port": 8000,
                "gpu_memory_utilization": 0.9,
                "max_model_len": 8192,
                "dtype": None,
            },
        }
        
        # Start with preset defaults, or empty dict for unknown models
        preset = PRESETS.get(model, {}).copy()
        
        # Merge user overrides on top
        if overrides:
            preset.update(overrides)
        
        return cls(**preset)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClusterServerConfig":
        """Create ClusterServerConfig from a dictionary."""
        if not data:
            return cls()
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "partition": self.partition,
            "gpu_types_preferred": self.gpu_types_preferred,
            "gpu_types_excluded": self.gpu_types_excluded,
            "num_gpus": self.num_gpus,
            "cpus_per_task": self.cpus_per_task,
            "mem_gb": self.mem_gb,
            "time_limit": self.time_limit,
            "port": self.port,
            "num_instances": self.num_instances,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "max_tokens": self.max_tokens,
            "dtype": self.dtype,
            "chat_template": self.chat_template,
            "server_start_timeout": self.server_start_timeout,
            "conda_env": self.conda_env,
            "cuda_module": self.cuda_module,
        }
