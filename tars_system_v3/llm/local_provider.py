"""
Local LLM Provider via Ollama.

Uses Ollama's OpenAI-compatible API (localhost:11434) to run models
locally on the Raspberry Pi without cloud dependency.

Install ollama: curl -fsSL https://ollama.com/install.sh | sh
Pull a model:   ollama pull gemma2:2b   (1.6GB, fastest for Pi)
                ollama pull phi3:mini   (2.3GB, good balance)
                ollama pull mistral     (4GB, best quality, needs 8GB Pi)
"""

import json
import os
from typing import List, Dict, Any, Optional

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from hardware.interfaces import ILLMProvider


class OllamaProvider(ILLMProvider):
    """
    Local LLM provider using Ollama's OpenAI-compatible API.

    Ollama exposes an OpenAI-compatible endpoint at /v1/chat/completions
    so this provider works the same way as the OpenAI cloud provider.
    """

    def __init__(
        self,
        model: str = "gemma2:2b",
        base_url: str = None,
        timeout: int = 30,
    ):
        """
        Initialize Ollama provider.

        Args:
            model: Ollama model name (e.g., "gemma2:2b", "phi3:mini", "mistral")
            base_url: Ollama API URL (default: http://localhost:11434)
            timeout: Request timeout in seconds (local models are slower)
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests package required: pip install requests")

        self.model = model
        self.base_url = base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.timeout = timeout
        self._available = None

    def is_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        if self._available is not None:
            return self._available

        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                # Check if our model (or a variant) is available
                self._available = any(
                    self.model in m or m.startswith(self.model.split(":")[0])
                    for m in models
                )
                if not self._available:
                    print(f"[LOCAL-LLM] Model '{self.model}' not found. Available: {models}")
                    print(f"[LOCAL-LLM] Pull it with: ollama pull {self.model}")
                return self._available
        except Exception as e:
            print(f"[LOCAL-LLM] Ollama not reachable at {self.base_url}: {e}")
            self._available = False
            return False

    def chat(self, messages: List[Dict[str, Any]], stream: bool = False, **kwargs) -> Any:
        """
        Send chat completion request to local Ollama.

        Args:
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream (not supported, ignored)
            **kwargs: Additional parameters:
                - model: Override model for this call
                - temperature: Sampling temperature
                - max_tokens: Max response tokens

        Returns:
            Response dict with OpenAI-compatible format
        """
        model = kwargs.pop("model", self.model)

        # Build request payload (Ollama native API)
        payload = {
            "model": model,
            "messages": self._clean_messages(messages),
            "stream": False,
        }

        # Add optional parameters
        options = {}
        if "temperature" in kwargs:
            options["temperature"] = kwargs.pop("temperature")
        if "max_tokens" in kwargs:
            options["num_predict"] = kwargs.pop("max_tokens")
        if options:
            payload["options"] = options

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            # Extract response text from Ollama format
            content = data.get("message", {}).get("content", "")

            # Return in OpenAI-compatible format so existing extraction code works
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": content,
                    }
                }]
            }

        except requests.Timeout:
            print(f"[LOCAL-LLM] Timeout after {self.timeout}s (model may be loading)")
            return {"choices": [{"message": {"content": "Processing timed out."}}]}
        except Exception as e:
            print(f"[LOCAL-LLM] Error: {e}")
            return {"choices": [{"message": {"content": ""}}]}

    def _clean_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Clean messages for Ollama — flatten multimodal content.

        Ollama doesn't support OpenAI's image_url content format,
        so we strip images and keep only text.

        Args:
            messages: Messages potentially containing image content

        Returns:
            Cleaned messages with text-only content
        """
        cleaned = []
        for msg in messages:
            content = msg.get("content", "")

            # Handle multimodal content (list of text + image blocks)
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif part.get("type") == "image_url":
                            text_parts.append("[image provided but local model cannot process images]")
                    elif isinstance(part, str):
                        text_parts.append(part)
                content = " ".join(text_parts)

            cleaned.append({
                "role": msg.get("role", "user"),
                "content": content,
            })

        return cleaned

    def extract_text(self, response: Any) -> str:
        """Extract text from response."""
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
        if isinstance(response, str):
            return response
        return ""
