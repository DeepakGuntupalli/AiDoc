"""
LLM Module
Handles HuggingFace LLM API integration.
"""

import os
from typing import Optional, List, Dict, Any
from huggingface_hub import InferenceClient


class HuggingFaceLLM:
    """HuggingFace Inference API LLM wrapper."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ):
        """
        Initialize the HuggingFace LLM.
        
        Args:
            api_key: HuggingFace API key. Falls back to HF_API_KEY env var.
            model_name: Model name to use. Falls back to HF_MODEL env var.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
        """
        self.api_key = api_key or os.getenv("HF_API_KEY", "")
        self.model_name = model_name or os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not self.api_key:
            raise ValueError("HuggingFace API key is required. Set HF_API_KEY environment variable or pass api_key parameter.")
        
        self.client = InferenceClient(
            token=self.api_key
        )
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response from the HuggingFace LLM.
        
        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            
        Returns:
            Generated response text.
        """
        # Build messages
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat_completion(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            error_str = str(e)
            # Try alternative models if current one fails
            fallback_models = [
                "microsoft/Phi-3-mini-4k-instruct",
                "Qwen/Qwen2.5-Coder-32B-Instruct",
                "meta-llama/Llama-3.2-3B-Instruct",
                "google/gemma-2-2b-it"
            ]
            
            for model in fallback_models:
                try:
                    response = self.client.chat_completion(
                        model=model,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    )
                    return response.choices[0].message.content
                except:
                    continue
            
            return f"Error generating response: {error_str}"


# Alias for backward compatibility
GrokLLM = HuggingFaceLLM


def create_huggingface_llm(
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.7
) -> HuggingFaceLLM:
    """
    Factory function to create a HuggingFace LLM instance.
    
    Args:
        api_key: Optional API key. Uses environment variable if not provided.
        model_name: Optional model name. Uses environment variable if not provided.
        temperature: Sampling temperature.
        
    Returns:
        Configured HuggingFaceLLM instance.
    """
    return HuggingFaceLLM(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature
    )
