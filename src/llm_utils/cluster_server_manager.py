"""
Manages vLLM server lifecycle on SLURM cluster.

Handles:
- Auto-generating sbatch scripts for vLLM servers
- Submitting SLURM jobs and tracking their IDs
- Discovering endpoints via scontrol (no shared files needed)
- Multi-instance support: N servers per model on different ports
- Round-robin endpoint selection for load distribution
"""
import os
import re
import subprocess
import tempfile
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.llm_utils.llm_model import LLMModel, Provider
from src.llm_utils.cluster_server_config import ClusterServerConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ClusterModelServerManager:
    """
    Manages vLLM server lifecycle on SLURM cluster.
    
    Supports multiple server instances per model for parallel task execution.
    Each instance gets its own SLURM job, GPU, and port.
    
    Endpoint discovery uses `scontrol show job` to get the assigned node
    hostname, then constructs the URL as http://<node>:<port>/v1.
    
    Usage:
        manager = ClusterModelServerManager()
        endpoints = manager.start_server(LLMModel.LLAMA3_1_8B_CLUSTER)
        # endpoints = ["http://node1:8000/v1", "http://node2:8001/v1", ...]
        endpoint = manager.get_endpoint(LLMModel.LLAMA3_1_8B_CLUSTER)  # round-robin
        manager.shutdown_all()
    """
    
    def __init__(self, config_overrides: Optional[Dict] = None):
        """
        Initialize the manager.
        
        Args:
            config_overrides: Optional dict of per-model overrides for
                ClusterServerConfig. Can be:
                - A flat dict applied to all models (backward compatible)
                - A nested dict keyed by model identifier for per-model overrides:
                  {"vicuna-13b-cluster": {"num_instances": 2}, ...}
        """
        self.config_overrides = config_overrides or {}
        
        # Multi-instance data structures: model -> list of values
        self.endpoints: Dict[LLMModel, List[str]] = {}
        self.slurm_job_ids: Dict[LLMModel, List[str]] = {}
        self.server_ports: Dict[LLMModel, List[int]] = {}
        
        # Round-robin counter per model (thread-safe)
        self._rr_counters: Dict[LLMModel, int] = {}
        self._rr_lock = threading.Lock()
        
        self._sbatch_dir = Path(tempfile.mkdtemp(prefix="vllm_sbatch_"))
    
    def _get_model_overrides(self, model: LLMModel) -> Dict:
        """
        Get config overrides for a specific model.
        
        Supports both flat overrides (applied to all models) and per-model
        nested overrides keyed by model name (e.g. "vicuna-13b-cluster").
        """
        if not self.config_overrides:
            return {}
        
        # Check if overrides are per-model (nested dict with model keys)
        model_key = model.name.lower().replace("_cluster", "-cluster")
        if model_key in self.config_overrides:
            return self.config_overrides[model_key]
        
        # Also try the raw model name
        if model.name.lower() in self.config_overrides:
            return self.config_overrides[model.name.lower()]
        
        # Check if it's a flat dict (has known ClusterServerConfig fields)
        config_fields = {
            "partition", "gpu_types_preferred", "gpu_types_excluded", "num_gpus",
            "cpus_per_task", "mem_gb", "time_limit", "port", "num_instances",
            "gpu_memory_utilization", "max_model_len", "dtype", "conda_env", "cuda_module"
        }
        if any(k in config_fields for k in self.config_overrides):
            return self.config_overrides  # Flat overrides applied to all models
        
        return {}
    
    def start_server(self, model: LLMModel) -> List[str]:
        """
        Start vLLM server(s) for the given model.
        
        Launches `num_instances` servers on consecutive ports, each as a
        separate SLURM job with its own GPU.
        
        Args:
            model: The cluster model to serve.
            
        Returns:
            List of endpoint URLs (e.g., ["http://node:8000/v1", "http://node:8001/v1"])
            
        Raises:
            RuntimeError: If any server fails to start or no GPU available.
        """
        if model in self.endpoints and self.endpoints[model]:
            logger.info(f"Servers already running for {model.model_id}: {self.endpoints[model]}")
            return self.endpoints[model]
        
        if model.provider != Provider.NU_CLUSTER:
            raise ValueError(f"{model.model_id} is not a cluster model (provider: {model.provider})")
        
        overrides = self._get_model_overrides(model)
        config = ClusterServerConfig.for_model(model, overrides)
        num_instances = config.num_instances
        base_port = config.port
        
        logger.info(f"Starting {num_instances} vLLM server(s) for {model.model_id} "
                     f"(ports {base_port}-{base_port + num_instances - 1})")
        
        # Find available GPU type (same for all instances)
        gpu_type = self._find_available_gpu(config)
        logger.info(f"Selected GPU type: {gpu_type} for {model.model_id}")
        
        # Launch N instances
        self.endpoints[model] = []
        self.slurm_job_ids[model] = []
        self.server_ports[model] = []
        self._rr_counters[model] = 0
        
        for i in range(num_instances):
            instance_port = base_port + i
            instance_config = ClusterServerConfig.for_model(model, {
                **overrides,
                "port": instance_port,
            })
            
            # Generate and submit sbatch for this instance
            sbatch_path = self._generate_sbatch(
                model, instance_config, gpu_type, instance_id=i
            )
            job_id = self._submit_sbatch(sbatch_path)
            
            self.slurm_job_ids[model].append(job_id)
            self.server_ports[model].append(instance_port)
            logger.info(f"  Instance {i}: SLURM job {job_id}, port {instance_port}")
        
        # Wait for all instances to become healthy
        for i in range(num_instances):
            job_id = self.slurm_job_ids[model][i]
            port = self.server_ports[model][i]
            
            instance_config = ClusterServerConfig.for_model(model, {
                **overrides,
                "port": port,
            })
            endpoint = self._wait_for_server(
                model, job_id, instance_config, instance_id=i,
                timeout=instance_config.server_start_timeout,
            )
            self.endpoints[model].append(endpoint)
            logger.info(f"  Instance {i}: ready at {endpoint}")
        
        logger.info(f"All {num_instances} server(s) ready for {model.model_id}")
        return self.endpoints[model]
    
    def _find_available_gpu(self, config: ClusterServerConfig) -> str:
        """
        Find the first available GPU type from the preference list.
        
        Checks `sinfo` for idle/mixed nodes with each preferred GPU type,
        skipping any in the exclude list.
        
        Args:
            config: ClusterServerConfig with gpu_types_preferred/excluded.
            
        Returns:
            GPU type string (e.g., "v100-sxm2").
            
        Raises:
            RuntimeError: If no preferred GPU type has availability.
        """
        excluded = set(config.gpu_types_excluded)
        candidates = [g for g in config.gpu_types_preferred if g not in excluded]
        
        if not candidates:
            raise RuntimeError(
                f"No GPU candidates after filtering. "
                f"Preferred: {config.gpu_types_preferred}, Excluded: {config.gpu_types_excluded}"
            )
        
        for gpu_type in candidates:
            try:
                result = subprocess.run(
                    [
                        "sinfo", "-p", config.partition,
                        "--gres=gpu:" + gpu_type,
                        "--states=idle,mixed",
                        "--noheader",
                        "--format=%n"
                    ],
                    capture_output=True, text=True, timeout=10
                )
                nodes = result.stdout.strip()
                if nodes:
                    available_count = len(nodes.splitlines())
                    logger.info(f"GPU {gpu_type}: {available_count} nodes available")
                    return gpu_type
                else:
                    logger.debug(f"GPU {gpu_type}: no idle/mixed nodes")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.warning(f"Could not check sinfo for {gpu_type}, skipping")
                continue
        
        # Fallback: use first preferred type and let SLURM queue
        fallback = candidates[0]
        logger.warning(
            f"No GPU type immediately available. Falling back to '{fallback}' "
            f"(job will queue until a node is free)"
        )
        return fallback
    
    def _generate_sbatch(
        self, model: LLMModel, config: ClusterServerConfig, gpu_type: str,
        instance_id: int = 0
    ) -> Path:
        """
        Generate sbatch script for a vLLM server instance.
        
        Args:
            model: The model to serve.
            config: SLURM/vLLM configuration (port already set for this instance).
            gpu_type: Selected GPU type.
            instance_id: Instance index (0, 1, 2, ...) for unique naming.
            
        Returns:
            Path to the generated sbatch file.
        """
        model_safe_name = model.name.lower()
        instance_suffix = f"_i{instance_id}" if instance_id > 0 else ""
        
        # Build vLLM serve command
        vllm_args = [
            f"--model {model.model_id}",
            f"--host 0.0.0.0",
            f"--port {config.port}",
            f"--gpu-memory-utilization {config.gpu_memory_utilization}",
            f"--max-model-len {config.max_model_len}",
        ]
        if config.dtype:
            vllm_args.append(f"--dtype {config.dtype}")
        if config.num_gpus > 1:
            vllm_args.append(f"--tensor-parallel-size {config.num_gpus}")
        if config.chat_template:
            # Resolve chat template path relative to the chat_templates directory
            chat_template_dir = Path(__file__).parent / "chat_templates"
            template_file = chat_template_dir / f"{config.chat_template}.jinja"
            if template_file.exists():
                vllm_args.append(f"--chat-template {template_file.resolve()}")
                logger.info(f"Using chat template: {template_file.resolve()}")
            else:
                logger.warning(f"Chat template not found: {template_file}. vLLM will use model default.")
        vllm_cmd = "python -m vllm.entrypoints.openai.api_server \\\n    " + " \\\n    ".join(vllm_args)
        
        script = f"""#!/bin/bash
#SBATCH --job-name=vllm_{model_safe_name}{instance_suffix}
#SBATCH --partition={config.partition}
#SBATCH --nodes=1
#SBATCH --gres=gpu:{gpu_type}:{config.num_gpus}
#SBATCH --cpus-per-task={config.cpus_per_task}
#SBATCH --mem={config.mem_gb}GB
#SBATCH --time={config.time_limit}
#SBATCH --output=logs/vllm_{model_safe_name}{instance_suffix}_%j.out
#SBATCH --error=logs/vllm_{model_safe_name}{instance_suffix}_%j.err

# Setup environment
mkdir -p logs
module load anaconda3/2024.06 {config.cuda_module}
source activate {config.conda_env}

# Use shared HF cache (models pre-downloaded on login node)
# Compute nodes have no internet — must run fully offline
export HF_HOME="${{HF_HOME:-$HOME/.cache/huggingface}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Start vLLM server (blocks until killed by scancel or time limit)
{vllm_cmd}
"""
        
        sbatch_path = self._sbatch_dir / f"vllm_{model_safe_name}{instance_suffix}.sbatch"
        sbatch_path.write_text(script)
        logger.debug(f"Generated sbatch at {sbatch_path}")
        return sbatch_path
    
    def _submit_sbatch(self, sbatch_path: Path) -> str:
        """
        Submit sbatch script and return the SLURM job ID.
        
        Raises:
            RuntimeError: If sbatch submission fails.
        """
        try:
            result = subprocess.run(
                ["sbatch", str(sbatch_path)],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                raise RuntimeError(f"sbatch failed: {result.stderr.strip()}")
            
            # Parse job ID from "Submitted batch job 12345678"
            output = result.stdout.strip()
            job_id = output.split()[-1]
            return job_id
            
        except FileNotFoundError:
            raise RuntimeError(
                "sbatch command not found. "
                "Are you running this on the cluster login node?"
            )
    
    def _get_job_node(self, job_id: str) -> Optional[str]:
        """
        Get the IP address of the node running a SLURM job.
        
        Uses `scontrol show job` to get the NodeList, then
        `scontrol show node` to resolve to an IP address.
        
        Returns:
            IP address string (e.g., "10.99.111.2"), or None if not yet assigned.
        """
        try:
            # Get node name from job
            result = subprocess.run(
                ["scontrol", "show", "job", job_id],
                capture_output=True, text=True, timeout=15
            )
            if result.returncode != 0:
                logger.warning(f"scontrol show job {job_id} failed: rc={result.returncode}, stderr={result.stderr[:200]}")
                return None
            
            match = re.search(r"NodeList=(\S+)", result.stdout)
            if not match:
                logger.warning(f"No NodeList found in scontrol output for job {job_id}")
                return None
            node = match.group(1)
            if not node or node == "(null)":
                return None
            
            # Resolve node name to IP via scontrol show node
            result = subprocess.run(
                ["scontrol", "show", "node", node],
                capture_output=True, text=True, timeout=15
            )
            if result.returncode != 0:
                logger.warning(f"scontrol show node {node} failed: rc={result.returncode}")
                return node  # Fallback to hostname
            
            addr_match = re.search(r"NodeAddr=(\S+)", result.stdout)
            if addr_match:
                addr = addr_match.group(1)
                if addr and addr != node:
                    logger.info(f"Resolved node {node} -> IP {addr}")
                    return addr
            
            # Fallback to hostname if IP not found
            logger.info(f"Using hostname {node} (no separate IP found)")
            return node
            
        except subprocess.TimeoutExpired:
            logger.warning(f"scontrol timed out for job {job_id}")
            return None
        except FileNotFoundError:
            logger.warning("scontrol command not found")
            return None
    
    def _get_job_state(self, job_id: str) -> Optional[str]:
        """
        Get the current SLURM state of a job.
        
        Returns:
            State string (e.g., "RUNNING", "PENDING"), or None if unknown.
        """
        try:
            result = subprocess.run(
                ["squeue", "-j", job_id, "--noheader", "--format=%T"],
                capture_output=True, text=True, timeout=10
            )
            state = result.stdout.strip()
            return state or None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None
    
    def _wait_for_server(
        self, model: LLMModel, job_id: str, config: ClusterServerConfig,
        instance_id: int = 0, timeout: int = 1800, poll_interval: int = 10
    ) -> str:
        """
        Wait for a vLLM server instance to become healthy.
        
        Args:
            model: The model being served.
            job_id: SLURM job ID for this instance.
            config: Server configuration (with correct port for this instance).
            instance_id: Instance index for logging.
            timeout: Max seconds to wait.
            poll_interval: Seconds between polls.
            
        Returns:
            Endpoint URL.
            
        Raises:
            RuntimeError: If server doesn't start within timeout.
        """
        start_time = time.time()
        model_safe_name = model.name.lower()
        instance_suffix = f"[{instance_id}]" if instance_id > 0 else ""
        
        logger.info(f"Waiting for vLLM server{instance_suffix} (job {job_id}, port {config.port}) "
                     f"to start (timeout: {timeout}s)...")
        
        while time.time() - start_time < timeout:
            elapsed = int(time.time() - start_time)
            
            # Check if SLURM job is still alive
            state = self._get_job_state(job_id)
            if state is None:
                raise RuntimeError(
                    f"SLURM job {job_id} is no longer in queue. "
                    f"Check logs/vllm_{model_safe_name}_*.err for errors."
                )
            
            if state == "PENDING":
                logger.info(f"Waiting for server{instance_suffix}... PENDING ({elapsed}s / {timeout}s)")
                time.sleep(poll_interval)
                continue
            
            # Job is RUNNING — discover node
            try:
                sq_result = subprocess.run(
                    ["squeue", "-j", job_id, "--noheader", "--format=%N"],
                    capture_output=True, text=True, timeout=15
                )
                node_name = sq_result.stdout.strip()
            except Exception as e:
                node_name = ""
                logger.debug(f"squeue failed for job {job_id}: {e}")
            
            node = None
            if node_name and node_name != "(null)":
                # Try to resolve hostname to IP
                node = self._get_job_node(job_id)
                if not node:
                    node = node_name  # fallback to raw hostname
            
            if node:
                endpoint = f"http://{node}:{config.port}/v1"
                healthy, err = self._health_check(endpoint)
                if healthy:
                    logger.info(f"vLLM server{instance_suffix} ready at {endpoint}")
                    return endpoint
                logger.info(f"Health check{instance_suffix} {endpoint} failed: {err} ({elapsed}s / {timeout}s)")
            else:
                logger.info(f"Waiting for server{instance_suffix}... node='{node_name}' state={state} ({elapsed}s / {timeout}s)")
            time.sleep(poll_interval)
        
        raise RuntimeError(
            f"vLLM server{instance_suffix} for {model.model_id} did not start within {timeout}s. "
            f"Job ID: {job_id}. Check SLURM logs."
        )
    
    def _health_check(self, endpoint: str) -> tuple:
        """
        Check if the vLLM server is responding.
        
        Args:
            endpoint: Base URL (e.g., "http://node:8000/v1").
            
        Returns:
            Tuple of (is_healthy: bool, error_message: str or None).
        """
        import urllib.request
        import urllib.error
        
        # Strip /v1 to get base URL for health check
        base_url = endpoint.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        
        health_url = f"{base_url}/health"
        try:
            # Bypass HTTP proxy — internal cluster connections must be direct
            proxy_handler = urllib.request.ProxyHandler({})
            opener = urllib.request.build_opener(proxy_handler)
            req = urllib.request.Request(health_url, method="GET")
            with opener.open(req, timeout=5) as resp:
                return (resp.status == 200, None)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            return (False, str(e))
    
    def get_endpoint(self, model: LLMModel) -> str:
        """
        Get endpoint URL for a running model using round-robin selection.
        
        Thread-safe: uses a lock to ensure consistent round-robin across
        concurrent tasks.
        
        Args:
            model: The cluster model.
            
        Returns:
            Endpoint URL string.
            
        Raises:
            KeyError: If no server is running for this model.
        """
        if model not in self.endpoints or not self.endpoints[model]:
            raise KeyError(
                f"No vLLM server running for {model.model_id}. "
                f"Call start_server() first."
            )
        
        endpoints = self.endpoints[model]
        if len(endpoints) == 1:
            return endpoints[0]
        
        # Thread-safe round-robin
        with self._rr_lock:
            idx = self._rr_counters.get(model, 0)
            endpoint = endpoints[idx % len(endpoints)]
            self._rr_counters[model] = idx + 1
        
        return endpoint
    
    def get_num_instances(self, model: LLMModel) -> int:
        """
        Get the number of running server instances for a model.
        
        Returns:
            Number of endpoints (0 if server not started).
        """
        return len(self.endpoints.get(model, []))
    
    def get_server_status(self, model: LLMModel) -> Dict[str, Any]:
        """
        Get the current lifecycle status of servers for a model.
        
        Args:
            model: The cluster model to check.
            
        Returns:
            Dict with:
                - state: "not_started" | "pending" | "running" | "ready" | "failed"
                - job_ids: List of SLURM job IDs
                - endpoints: List of endpoint URLs (healthy ones)
                - num_instances: Number of instances
                - instance_states: List of per-instance SLURM states
        """
        status: Dict[str, Any] = {
            "state": "not_started",
            "job_ids": [],
            "endpoints": [],
            "num_instances": 0,
            "instance_states": [],
        }
        
        if model not in self.slurm_job_ids:
            return status
        
        job_ids = self.slurm_job_ids[model]
        status["job_ids"] = job_ids
        status["num_instances"] = len(job_ids)
        
        # Check each instance
        all_ready = True
        any_failed = False
        
        for i, job_id in enumerate(job_ids):
            slurm_state = self._get_job_state(job_id)
            status["instance_states"].append(slurm_state)
            
            if slurm_state is None:
                any_failed = True
            elif slurm_state != "RUNNING":
                all_ready = False
        
        # Check cached endpoints
        if model in self.endpoints:
            for endpoint in self.endpoints[model]:
                if self._health_check(endpoint)[0]:
                    status["endpoints"].append(endpoint)
        
        if any_failed:
            status["state"] = "failed"
        elif status["endpoints"]:
            status["state"] = "ready" if len(status["endpoints"]) == len(job_ids) else "running"
        elif all_ready:
            status["state"] = "running"
        else:
            status["state"] = "pending"
        
        return status
    
    def shutdown_all(self):
        """Cancel all active SLURM jobs and clean up."""
        for model, job_ids in self.slurm_job_ids.items():
            for job_id in job_ids:
                logger.info(f"Cancelling SLURM job {job_id} ({model.model_id})")
                try:
                    subprocess.run(
                        ["scancel", job_id],
                        capture_output=True, timeout=10
                    )
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    logger.warning(f"Could not cancel job {job_id}")
        
        self.endpoints.clear()
        self.slurm_job_ids.clear()
        self.server_ports.clear()
        self._rr_counters.clear()
        logger.info("All vLLM servers shut down")
    
    def __del__(self):
        """Ensure cleanup on garbage collection."""
        if self.slurm_job_ids:
            self.shutdown_all()
