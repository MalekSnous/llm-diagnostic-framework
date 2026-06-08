"""
Unified LLM client supporting multiple providers.
Provides consistent interface across OpenAI, Anthropic, and Hugging Face models.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

# Provider SDKs (openai, anthropic, transformers, torch) are imported lazily
# inside each client so that importing this module — and the package as a whole
# — does not require every backend to be installed. This keeps test/CI
# environments lightweight and lets users install only the providers they use.

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is optional
    pass


class LLMResponse:
    """Standardized LLM response format."""

    def __init__(
        self,
        text: str,
        model: str,
        tokens_used: int,
        latency_ms: float,
        cost_usd: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.text = text
        self.model = model
        self.tokens_used = tokens_used
        self.latency_ms = latency_ms
        self.cost_usd = cost_usd
        self.metadata = metadata or {}


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs

    @abstractmethod
    def generate(
        self, prompt: str, max_tokens: int = 500, temperature: float = 0.0, **kwargs
    ) -> LLMResponse:
        """Generate response from LLM."""
        pass

    @abstractmethod
    def batch_generate(
        self, prompts: List[str], max_tokens: int = 500, temperature: float = 0.0, **kwargs
    ) -> List[LLMResponse]:
        """Generate responses for multiple prompts."""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI API client (GPT-4, GPT-3.5)."""

    def __init__(self, model_name: str = "gpt-4o-mini", **kwargs):
        super().__init__(model_name, **kwargs)
        import openai

        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Pricing in USD per 1K tokens (must match the per-1K math in
        # _calculate_cost). Values below are the published per-1M-token prices
        # divided by 1000 — e.g. gpt-4o is $2.50/$10.00 per 1M => 0.0025/0.01.
        # A previous version stored gpt-4o/gpt-4o-mini/gpt-4-turbo as per-1M
        # prices here, which inflated all reported costs by ~1000x.
        self.pricing = {
            "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
            "gpt-4o": {"input": 0.0025, "output": 0.01},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        }

    def generate(
        self, prompt: str, max_tokens: int = 300, temperature: float = 0.0, **kwargs
    ) -> LLMResponse:
        """Generate response using OpenAI API."""
        import time

        start = time.time()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        latency_ms = (time.time() - start) * 1000

        # Calculate cost
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = self._calculate_cost(input_tokens, output_tokens)

        return LLMResponse(
            text=response.choices[0].message.content,
            model=self.model_name,
            tokens_used=response.usage.total_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "finish_reason": response.choices[0].finish_reason,
            },
        )

    def batch_generate(
        self, prompts: List[str], max_tokens: int = 500, temperature: float = 0.0, **kwargs
    ) -> List[LLMResponse]:
        """Generate responses for multiple prompts."""
        return [self.generate(p, max_tokens, temperature, **kwargs) for p in prompts]

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD."""
        pricing = self.pricing.get(self.model_name, {"input": 0, "output": 0})
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        return input_cost + output_cost


class AnthropicClient(BaseLLMClient):
    """Anthropic API client (Claude)."""

    def __init__(self, model_name: str = "claude-3-sonnet-20240229", **kwargs):
        super().__init__(model_name, **kwargs)
        from anthropic import Anthropic

        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        # Pricing per 1M tokens
        self.pricing = {
            "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
            "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},
            "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
        }

    def generate(
        self, prompt: str, max_tokens: int = 500, temperature: float = 0.0, **kwargs
    ) -> LLMResponse:
        """Generate response using Anthropic API."""
        import time

        start = time.time()

        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

        latency_ms = (time.time() - start) * 1000

        # Calculate cost
        input_tokens = message.usage.input_tokens
        output_tokens = message.usage.output_tokens
        cost = self._calculate_cost(input_tokens, output_tokens)

        return LLMResponse(
            text=message.content[0].text,
            model=self.model_name,
            tokens_used=input_tokens + output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "stop_reason": message.stop_reason,
            },
        )

    def batch_generate(
        self, prompts: List[str], max_tokens: int = 500, temperature: float = 0.0, **kwargs
    ) -> List[LLMResponse]:
        """Generate responses for multiple prompts."""
        return [self.generate(p, max_tokens, temperature, **kwargs) for p in prompts]

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD."""
        pricing = self.pricing.get(self.model_name, {"input": 0, "output": 0})
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost


class HuggingFaceClient(BaseLLMClient):
    """Local Hugging Face model client (Llama, Mistral, etc.)."""

    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", **kwargs):
        super().__init__(model_name, **kwargs)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cpu":
            torch.set_num_threads(kwargs.get("num_threads", 8))

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # dtype=torch.float16 if self.device == "cuda" else torch.float32,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

    def generate(
        self, prompt: str, max_tokens: int = 500, temperature: float = 0.0, **kwargs
    ) -> LLMResponse:
        """Generate response using local Hugging Face model."""
        import time

        import torch

        start = time.time()

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,  # Évite les prompts trop longs
            max_length=512,  # Limite la longueur d'entrée
        )
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )

        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        latency_ms = (time.time() - start) * 1000
        tokens_used = outputs.shape[1]

        return LLMResponse(
            text=generated_text,
            model=self.model_name,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            cost_usd=0.0,  # Local inference is free
            metadata={"device": self.device},
        )

    def batch_generate(
        self, prompts: List[str], max_tokens: int = 500, temperature: float = 0.0, **kwargs
    ) -> List[LLMResponse]:
        """Generate responses for multiple prompts."""
        # For simplicity, process sequentially
        # TODO: Implement true batching for efficiency
        return [self.generate(p, max_tokens, temperature, **kwargs) for p in prompts]


def get_llm_client(model_name: str, **kwargs) -> BaseLLMClient:
    """Factory function to get appropriate LLM client."""

    if model_name.startswith("gpt-"):
        return OpenAIClient(model_name, **kwargs)
    elif model_name.startswith("claude-"):
        return AnthropicClient(model_name, **kwargs)
    else:
        # Assume it's a Hugging Face model
        return HuggingFaceClient(model_name, **kwargs)


# Example usage
if __name__ == "__main__":
    # Test OpenAI
    client = get_llm_client("gpt-4-turbo-preview")
    response = client.generate("What is 2+2?", max_tokens=200)
    print(f"Response: {response.text}")
    print(f"Cost: ${response.cost_usd:.6f}")
    print(f"Latency: {response.latency_ms:.0f}ms")
