"""
NURC Cluster service via vLLM OpenAI-compatible HTTP server.

This service inherits from OpenAIService, pointing the OpenAI client
at a vLLM server endpoint instead of the OpenAI API. All batch_generate
and batch_chat logic is inherited — the only difference is the base_url.

The vLLM server is managed by ClusterModelServerManager, which starts it
via SLURM sbatch before tasks begin.
"""
from typing import List, Tuple, Optional, Any

from ..base_llm_service import BaseLLMService
from ..llm_model import LLMModel
from ..llm_config import LLMConfig
from ...utils.logger import get_logger

logger = get_logger(__name__)


class NURCClusterService(BaseLLMService):
    """
    Service for models running on NURC cluster via vLLM HTTP server.
    
    Requires a running vLLM server (managed by ClusterModelServerManager).
    Uses the openai Python client pointed at the vLLM endpoint, which is
    fully OpenAI-compatible.
    
    Args (via kwargs):
        server_url: Required. The vLLM endpoint (e.g., "http://node:8000/v1").
    """
    
    def __init__(self, model: LLMModel, config: LLMConfig = None, **kwargs):
        server_url = kwargs.pop("server_url", None)
        if not server_url:
            raise ValueError(
                "NURCClusterService requires 'server_url' kwarg. "
                "Use ClusterModelServerManager to start the vLLM server first."
            )
        
        self.model = model
        self.config = config or LLMConfig()
        self.temperature = kwargs.get('temperature', self.config.temperature)
        # Prefer cluster-specific max_tokens (from ClusterServerConfig) over global default
        cluster_max_tokens = kwargs.pop('cluster_max_tokens', None)
        self.max_tokens = cluster_max_tokens or kwargs.get('max_tokens', self.config.max_tokens)
        self.server_url = server_url
        
        # Initialize OpenAI client pointed at vLLM server
        # Bypass HTTP proxy for internal cluster connections
        try:
            import httpx
            from openai import OpenAI
            self.client = OpenAI(
                base_url=server_url,
                api_key="unused",  # vLLM doesn't require an API key
                http_client=httpx.Client(
                    trust_env=False,  # Bypass cluster Squid proxy (ignore HTTP_PROXY env var)
                    timeout=httpx.Timeout(600.0, connect=60.0),
                ),
            )
            logger.info(f"Initialized cluster service for {model.model_id} at {server_url}")
        except ImportError:
            raise ImportError(
                "openai package is required for NURCClusterService. "
                "Install with: pip install openai"
            )
    
    def batch_generate(
        self,
        prompts: List[Tuple[str, str]],
        system_message: Optional[str] = None,
        **kwargs
    ) -> List[Tuple[str, str]]:
        """Generate text responses via vLLM server."""
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        results = []
        total = len(prompts)
        log_interval = max(1, total // 10)
        
        for idx, (prompt_id, prompt_text) in enumerate(prompts, 1):
            try:
                if idx == 1 or idx % log_interval == 0 or idx == total:
                    logger.info(f"Processing request {idx}/{total} ({idx*100//total}%)")
                
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                messages.append({"role": "user", "content": prompt_text})
                
                response = self.client.chat.completions.create(
                    model=self.model.model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                response_text = response.choices[0].message.content or ""
                if not response_text.strip():
                    response_text = f"[Empty response from vLLM server]"
                
                results.append((prompt_id, response_text))
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"vLLM API error for prompt {prompt_id}: {error_msg}")
                results.append((prompt_id, f"Error: {error_msg}"))
        
        return results
    
    def batch_chat(
        self,
        conversations: List[Tuple[str, List[Tuple[str, Optional[Any]]]]],
        **kwargs
    ) -> List[Tuple[str, str]]:
        """Generate responses for chat conversations via vLLM server."""
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        results = []
        total = len(conversations)
        log_interval = max(1, total // 10)
        
        for idx, (conv_id, messages) in enumerate(conversations, 1):
            try:
                if idx == 1 or idx % log_interval == 0 or idx == total:
                    logger.info(f"Processing conversation {idx}/{total} ({idx*100//total}%)")
                
                # Build OpenAI-format messages
                openai_messages = []
                for msg in messages:
                    text = msg[0]
                    openai_messages.append({"role": "user", "content": text})
                
                response = self.client.chat.completions.create(
                    model=self.model.model_id,
                    messages=openai_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                response_text = response.choices[0].message.content or ""
                results.append((conv_id, response_text))
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"vLLM API error for conversation {conv_id}: {error_msg}")
                results.append((conv_id, f"Error: {error_msg}"))
        
        return results
