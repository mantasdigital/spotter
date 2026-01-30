"""
LLM Provider Wrapper.

Wraps the picarx OpenAI LLM to implement the ILLMProvider interface properly.
"""

from typing import List, Dict, Any
from hardware.interfaces import ILLMProvider


class OpenAIProviderWrapper(ILLMProvider):
    """
    Wrapper around picarx OpenAI LLM to implement ILLMProvider interface.

    The picarx.llm.OpenAI class doesn't implement extract_text() method,
    so this wrapper adds it to make it compatible with TARS subsystems.

    This wrapper also proxies all other OpenAI LLM methods to maintain
    full compatibility with VoiceAssistant and other systems.
    """

    def __init__(self, llm_instance):
        """
        Initialize wrapper.

        Args:
            llm_instance: Instance of picarx.llm.OpenAI (or robot_hat.llm.OpenAI)
        """
        self.llm = llm_instance

    # Proxy common OpenAI LLM methods to the underlying instance
    def set_instructions(self, instructions: str):
        """Set system instructions for the LLM."""
        if hasattr(self.llm, 'set_instructions'):
            return self.llm.set_instructions(instructions)

    def set_welcome(self, welcome: str):
        """Set welcome message."""
        if hasattr(self.llm, 'set_welcome'):
            return self.llm.set_welcome(welcome)

    def set_model(self, model: str):
        """Set the model name."""
        if hasattr(self.llm, 'set_model'):
            return self.llm.set_model(model)

    def set_max_messages(self, max_messages: int):
        """Set maximum message history."""
        if hasattr(self.llm, 'set_max_messages'):
            return self.llm.set_max_messages(max_messages)

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        if hasattr(self.llm, 'add_message'):
            return self.llm.add_message(role, content)

    @property
    def messages(self):
        """Get message history."""
        if hasattr(self.llm, 'messages'):
            return self.llm.messages
        return []

    @property
    def model(self):
        """Get current model name."""
        if hasattr(self.llm, 'model'):
            return self.llm.model
        return None

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Any:
        """
        Call LLM chat method.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters (stream, max_tokens, etc.)

        Returns:
            LLM response object
        """
        return self.llm.chat(messages=messages, **kwargs)

    def extract_text(self, response: Any) -> str:
        """
        Extract text content from LLM response.

        Handles various response formats from OpenAI API:
        - String responses (direct return)
        - requests.Response objects (parse JSON)
        - Objects with .content attribute
        - Objects with .choices[0].message.content
        - Dicts with nested content fields

        Args:
            response: Response object from chat()

        Returns:
            str: Extracted text content
        """
        # Already a string
        if isinstance(response, str):
            return response.strip()

        # requests.Response object - parse JSON
        if hasattr(response, 'json') and callable(response.json):
            try:
                data = response.json()
                # Now process as dict
                if isinstance(data, dict):
                    if 'choices' in data and len(data['choices']) > 0:
                        choice = data['choices'][0]
                        if 'message' in choice and 'content' in choice['message']:
                            content = choice['message']['content']
                            if content is not None:
                                return content.strip()
                    # Try simple formats
                    for key in ('content', 'message', 'text', 'response'):
                        if key in data and data[key] is not None and isinstance(data[key], str):
                            return data[key].strip()
            except Exception as e:
                print(f"[WRAPPER] Failed to parse JSON from response: {e}")
                # Try to get response text directly
                if hasattr(response, 'text'):
                    print(f"[WRAPPER] Response.text: {response.text[:200]}")
                return ""

        # OpenAI ChatCompletion object with choices
        if hasattr(response, 'choices') and len(response.choices) > 0:
            choice = response.choices[0]
            if hasattr(choice, 'message'):
                message = choice.message
                if hasattr(message, 'content') and message.content is not None:
                    return message.content.strip()

        # Object with direct .content attribute
        if hasattr(response, 'content') and response.content is not None and isinstance(response.content, str):
            return response.content.strip()

        # Dict with nested structure
        if isinstance(response, dict):
            # Try OpenAI API response format
            if 'choices' in response and len(response['choices']) > 0:
                choice = response['choices'][0]
                if 'message' in choice and 'content' in choice['message']:
                    content = choice['message']['content']
                    if content is not None:
                        return content.strip()

            # Try simple formats
            for key in ('content', 'message', 'text', 'response'):
                if key in response and response[key] is not None and isinstance(response[key], str):
                    return response[key].strip()

        # Try common getter methods
        for method_name in ('get_message', 'get_text', 'get_content'):
            method = getattr(response, method_name, None)
            if callable(method):
                try:
                    text = method()
                    if text is not None and isinstance(text, str):
                        return text.strip()
                except Exception:
                    pass

        # Last resort: convert to string
        print(f"[WRAPPER] Warning: Could not extract text from response, using str(): {type(response)}")
        result = str(response)
        return result.strip() if result is not None else ""
