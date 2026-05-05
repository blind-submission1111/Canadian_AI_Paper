"""
LLM model definitions with provider information.
"""
from enum import Enum


class Provider(str, Enum):
    """Enum for LLM service providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"
    NU_CLUSTER = "nu_cluster"


class LLMModel(Enum):
    """
    Enum for LLM models.
    
    Each model has:
    - model_id: The actual model identifier string
    - provider: Which service implementation to use
    """
    
    # OpenAI models
    GPT_3_5_TURBO = ("gpt-3.5-turbo", Provider.OPENAI)
    GPT_4 = ("gpt-4", Provider.OPENAI)
    GPT_4_TURBO = ("gpt-4-turbo", Provider.OPENAI)
    GPT_4O = ("gpt-4o", Provider.OPENAI)
    GPT_4O_MINI = ("gpt-4o-mini", Provider.OPENAI)
    GPT_4_1 = ("gpt-4.1", Provider.OPENAI)
    GPT_4_1_MINI = ("gpt-4.1-mini", Provider.OPENAI)
    GPT_4_1_NANO = ("gpt-4.1-nano", Provider.OPENAI)
    GPT_4_NANO = ("gpt-4-nano", Provider.OPENAI)
    GPT_O3 = ("gpt-o3", Provider.OPENAI)
    GPT_O4_MINI = ("gpt-o4-mini", Provider.OPENAI)
    GPT_5 = ("gpt-5", Provider.OPENAI)
    GPT_5_MINI = ("gpt-5-mini", Provider.OPENAI)
    GPT_5_NANO = ("gpt-5-nano", Provider.OPENAI)
    GPT_5_2 = ("gpt-5.2-2025-12-11", Provider.OPENAI)
    GPT_5_2_PRO = ("gpt-5.2-pro-2025-12-11", Provider.OPENAI)

   
    # Older Claude models (less restrictive safety filters)
    CLAUDE_3_HAIKU = ("claude-3-haiku-20240307", Provider.ANTHROPIC)
    CLAUDE_3_SONNET = ("claude-3-sonnet-20240229", Provider.ANTHROPIC)
    CLAUDE_3_5_HAIKU = ("claude-3-5-haiku-20241022", Provider.ANTHROPIC)
    CLAUDE_3_7_SONNET = ("claude-3-7-sonnet-20250219", Provider.ANTHROPIC) # Verified available
    CLAUDE_4_1_OPUS = ("claude-opus-4-1", Provider.ANTHROPIC) 
    CLAUDE_4_5_OPUS = ("claude-opus-4-5-20251101", Provider.ANTHROPIC) 
    CLAUDE_4_5_SONNET = ("claude-sonnet-4-5-20250929", Provider.ANTHROPIC) 
    # cheapest among the newest claude models
    CLAUDE_4_5_HAIKU = ("claude-haiku-4-5-20251001", Provider.ANTHROPIC) 
    
    # Older Gemini models (less restrictive safety filters)
    GEMINI_1_5_FLASH = ("gemini-1.5-flash", Provider.GOOGLE)
    GEMINI_1_5_FLASH_8B = ("gemini-1.5-flash-8b", Provider.GOOGLE)
    GEMINI_1_5_PRO = ("gemini-1.5-pro", Provider.GOOGLE)
    # Specific versions pinned to avoid 'latest' aliases
    GEMINI_2_0_FLASH = ("gemini-2.0-flash", Provider.GOOGLE)
    GEMINI_2_0_FLASH_LITE = ("gemini-2.0-flash-lite", Provider.GOOGLE)
    GEMINI_2_5_FLASH = ("gemini-2.5-flash", Provider.GOOGLE)
    GEMINI_2_5_FLASH_LITE = ("gemini-2.5-flash-lite", Provider.GOOGLE)
    GEMINI_2_5_PRO = ("gemini-2.5-pro", Provider.GOOGLE)
    GEMINI_3_FLASH_PREVIEW = ("gemini-3-flash-preview", Provider.GOOGLE)
    GEMINI_3_PRO_PREVIEW = ("gemini-3-pro-preview", Provider.GOOGLE)
    
    LLAMA3 = ("llama3", Provider.LOCAL)
    LLAMA3_1 = ("llama3.1", Provider.LOCAL)
    
    # Transformers models (local HuggingFace)
    # Llama 3 series (8B)
    LLAMA3_8B = ("meta-llama/Meta-Llama-3-8B-Instruct", Provider.LOCAL)
    LLAMA3_2_1B = ("meta-llama/Llama-3.2-1B-Instruct", Provider.LOCAL)  # ~2GB, very fast
    LLAMA3_2_3B = ("meta-llama/Llama-3.2-3B-Instruct", Provider.LOCAL)  # ~6GB, balanced
    
    # Other models
    MISTRAL_7B = ("mistralai/Mistral-7B-Instruct-v0.2", Provider.LOCAL)
    PHI_3_MINI = ("microsoft/Phi-3-mini-4k-instruct", Provider.LOCAL)  # ~7GB, good quality
    
    # NU Cluster models (via vLLM server)
    # Text models for jailbreaking research
    LLAMA3_1_8B_CLUSTER = ("meta-llama/Llama-3.1-8B-Instruct", Provider.NU_CLUSTER)  # ~16GB, V100 OK (gated)
    LLAMA3_8B_CLUSTER = ("meta-llama/Meta-Llama-3-8B-Instruct", Provider.NU_CLUSTER)  # ~16GB, V100 OK (available)
    VICUNA_13B_CLUSTER = ("lmsys/vicuna-13b-v1.5", Provider.NU_CLUSTER)               # ~26GB, V100-SXM2 OK
    # Vision-language models
    PIXTRAL_12B = ("mistralai/Pixtral-12B-2409", Provider.NU_CLUSTER)  # Vision-language, needs >32GB
    LLAVA_7B = ("llava-hf/llava-1.5-7b-hf", Provider.NU_CLUSTER) 
    
    @property
    def model_id(self) -> str:
        """Get the model identifier string."""
        return self.value[0]
    
    @property
    def provider(self) -> Provider:
        """Get the provider for this model."""
        return self.value[1]
    
    @classmethod
    def from_string(cls, model_str: str) -> "LLMModel":
        """
        Get LLMModel enum from a string (model_id or enum name).
        
        Args:
            model_str: Model identifier string (e.g., "gpt-5-nano") or 
                      enum name (e.g., "GPT_5_NANO")
        
        Returns:
            Matching LLMModel enum
        
        Raises:
            ValueError: If no matching model found
        """
        # Try by model_id first
        for model in cls:
            if model.model_id == model_str:
                return model
        
        # Try by enum name (case insensitive)
        model_str_upper = model_str.upper().replace("-", "_").replace(".", "_")
        for model in cls:
            if model.name == model_str_upper:
                return model
        
        # List available models for error message
        available = [m.model_id for m in cls]
        raise ValueError(f"Unknown model: '{model_str}'. Available: {available[:10]}...")    
