"""
LLM Agent System.

Four-agent architecture for processing commands:
1. CommandAgent - Parse user intent and extract actions
2. MemoryAgent - Decide what to remember with tagging
3. CharacterAgent - Periodically refresh personality summary
4. AnswerAgent - Generate final human-like response

This replaces the monolithic LLM call from v58 with a structured,
multi-step approach that maintains context and personality.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from llm.prompts import (
    COMMAND_AGENT_SYSTEM_PROMPT,
    MEMORY_AGENT_SYSTEM_PROMPT,
    CHARACTER_AGENT_SYSTEM_PROMPT,
    get_answer_agent_system_prompt
)
from hardware.interfaces import ILLMProvider


class BaseAgent(ABC):
    """
    Abstract base class for all LLM agents.

    Provides common functionality for LLM interaction including:
    - Unified LLM calling with error handling
    - Text extraction from various response formats
    - Logging and debugging support
    """

    def __init__(self, llm_provider: ILLMProvider):
        """
        Initialize base agent.

        Args:
            llm_provider: LLM provider implementing ILLMProvider interface
        """
        self.llm = llm_provider
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input and return structured output.

        Args:
            input_data: Agent-specific input dictionary

        Returns:
            Agent-specific output dictionary
        """
        pass

    def _call_llm(
        self,
        messages: List[dict],
        **kwargs
    ) -> str:
        """
        Call LLM with error handling and text extraction.

        Args:
            messages: List of message dicts for LLM
            **kwargs: Additional LLM parameters

        Returns:
            str: Extracted text from LLM response

        Raises:
            Exception: If LLM call fails after extraction attempt
        """
        try:
            response = self.llm.chat(messages=messages, stream=False, **kwargs)
            return self._extract_text_from_response(response)

        except Exception as e:
            self.logger.error(f"LLM call failed in {self.__class__.__name__}: {e}")
            raise

    def _extract_text_from_response(self, resp) -> str:
        """
        Extract text from various LLM response formats.

        Handles:
        - String responses
        - requests.Response objects (parse JSON body)
        - Objects with .content attribute
        - Dicts with 'content', 'message', or 'text' keys
        - Objects with get_message(), get_text(), or get_content() methods

        Args:
            resp: LLM response (any format)

        Returns:
            str: Extracted text content
        """
        # Already a string
        if isinstance(resp, str):
            return resp.strip()

        # requests.Response object - parse JSON body (like v58)
        if hasattr(resp, 'json') and callable(resp.json) and hasattr(resp, 'status_code'):
            try:
                data = resp.json()
                if isinstance(data, dict):
                    # OpenAI API format: choices[0].message.content
                    if 'choices' in data and len(data['choices']) > 0:
                        choice = data['choices'][0]
                        if isinstance(choice, dict) and 'message' in choice:
                            message = choice['message']
                            if isinstance(message, dict) and 'content' in message:
                                content = message['content']
                                if content is not None:
                                    return str(content).strip()
                    # Simple key formats
                    for key in ('content', 'message', 'text', 'response'):
                        if key in data and data[key] is not None:
                            return str(data[key]).strip()
            except Exception as e:
                self.logger.warning(f"Failed to parse response JSON: {e}")
                # Try response.text as fallback
                if hasattr(resp, 'text') and resp.text:
                    return resp.text.strip()
                return ""

        # Object with .content attribute (OpenAI-style) - check it's a string not bytes
        if hasattr(resp, 'content'):
            content = resp.content
            if isinstance(content, str):
                return content.strip()
            # Skip bytes (requests.Response.content is bytes)

        # OpenAI ChatCompletion object with choices
        if hasattr(resp, 'choices') and len(resp.choices) > 0:
            choice = resp.choices[0]
            if hasattr(choice, 'message'):
                message = choice.message
                if hasattr(message, 'content') and message.content is not None:
                    return str(message.content).strip()

        # Try common getter methods
        for method_name in ('get_message', 'get_text', 'get_content'):
            method = getattr(resp, method_name, None)
            if callable(method):
                try:
                    text = method()
                    if isinstance(text, str):
                        return text.strip()
                except:
                    pass

        # Dict-like response
        if isinstance(resp, dict):
            # OpenAI API format
            if 'choices' in resp and len(resp['choices']) > 0:
                choice = resp['choices'][0]
                if isinstance(choice, dict) and 'message' in choice:
                    message = choice['message']
                    if isinstance(message, dict) and 'content' in message:
                        content = message['content']
                        if content is not None:
                            return str(content).strip()
            # Simple key formats
            for key in ('content', 'message', 'text', 'response'):
                if key in resp and resp[key] is not None:
                    return str(resp[key]).strip()

        # Last resort: convert to string but warn
        result = str(resp)
        # Don't return Response object string representation
        if result.startswith('<Response ['):
            self.logger.warning(f"Could not extract text from response: {type(resp)}")
            return ""
        return result.strip()


class CommandAgent(BaseAgent):
    """
    Parses user commands into structured intents and actions.

    Extracts:
    - High-level intents (movement, vision, macro, etc.)
    - Specific actions to execute
    - Explanation of the command

    Example:
        >>> agent = CommandAgent(llm_provider)
        >>> result = agent.process({"user_text": "move forward and wave"})
        >>> print(result)
        {
            "intents": ["movement", "gesture"],
            "actions": "forward(50), wave_hands()",
            "explanation": "Move forward and wave hands"
        }
    """

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse command into intents and actions.

        Args:
            input_data: Dict with keys:
                - user_text (str): User command text
                - available_macros (dict, optional): Dict of macro_name -> action_string

        Returns:
            Dict with keys:
                - intents: List[str] - Intent tags
                - actions: str - Action string
                - explanation: str - Command explanation
        """
        user_text = input_data.get("user_text", "")
        available_macros = input_data.get("available_macros", {})

        # Build system prompt with macro context if available
        system_prompt = COMMAND_AGENT_SYSTEM_PROMPT
        if available_macros:
            macro_list = "\n".join([f"- {name}: {actions}" for name, actions in available_macros.items()])
            system_prompt += f"\n\nAVAILABLE LEARNED MACROS:\n{macro_list}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

        try:
            response_text = self._call_llm(messages)

            # Handle empty or None responses gracefully
            if not response_text or not response_text.strip():
                self.logger.debug("LLM returned empty response, using fallback")
                return {
                    "intents": [],
                    "actions": user_text,
                    "explanation": f"Processing: {user_text}"
                }

            # Try to extract JSON from response (may have markdown formatting)
            json_text = response_text.strip()
            if json_text.startswith("```"):
                # Remove markdown code blocks
                lines = json_text.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block or not line.startswith("```"):
                        json_lines.append(line)
                json_text = "\n".join(json_lines).strip()

            data = json.loads(json_text)

            if not isinstance(data, dict):
                raise ValueError("Response not a dict")

            return {
                "intents": data.get("intents", []),
                "actions": data.get("actions", ""),
                "explanation": data.get("explanation", "")
            }

        except json.JSONDecodeError as e:
            # JSON parse error - log at debug level (not warning) as it's expected sometimes
            self.logger.debug(f"LLM response not valid JSON, using fallback: {e}")
            return {
                "intents": [],
                "actions": user_text,
                "explanation": f"Processing: {user_text}"
            }
        except Exception as e:
            self.logger.warning(f"LLM command agent error: {e}")
            # Fallback to basic parsing
            return {
                "intents": [],
                "actions": user_text,
                "explanation": f"Processing: {user_text}"
            }


class MemoryAgent(BaseAgent):
    """
    Decides what to remember from user commands.

    Analyzes commands to extract:
    - Topic tags for semantic search
    - Intent tags for behavior recall
    - Importance scores (1-10)

    Example:
        >>> agent = MemoryAgent(llm_provider)
        >>> result = agent.process({
        ...     "user_text": "Your name is Bob now",
        ...     "recent_topics": ["robot_self"],
        ...     "recent_dialogue": "...",
        ...     "recent_visual": "..."
        ... })
        >>> print(result)
        {
            "topic": "robot_self",
            "intents": ["name_change"],
            "importance": 9
        }
    """

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze what to remember from command.

        Args:
            input_data: Dict with keys:
                - user_text: str - The command
                - recent_topics: List[str] - Recent topic tags
                - recent_dialogue: str - Recent conversation summary
                - recent_visual: str - Recent visual observations

        Returns:
            Dict with keys:
                - topic: Optional[str] - Topic tag
                - intents: List[str] - Intent tags
                - importance: int - Importance score 1-10
        """
        user_payload = {
            "user_command": input_data.get("user_text", ""),
            "recent_topics": input_data.get("recent_topics", []),
            "recent_dialogue": input_data.get("recent_dialogue", ""),
            "recent_visual": input_data.get("recent_visual", ""),
        }

        messages = [
            {"role": "system", "content": MEMORY_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload)},
        ]

        try:
            response_text = self._call_llm(messages)

            # Handle empty response gracefully
            if not response_text or not response_text.strip():
                self.logger.debug("[MEM-AGENT] Empty LLM response, using fallback")
                return {"topic": None, "intents": [], "importance": 5}

            data = json.loads(response_text)

            topic = data.get("topic")
            if topic == "" or topic is False:
                topic = None

            intents = data.get("intents") or []
            importance = int(data.get("importance", 5))

            return {
                "topic": topic,
                "intents": intents,
                "importance": importance,
            }

        except json.JSONDecodeError as e:
            self.logger.debug(f"[MEM-AGENT] JSON parse failed: {e}")
            return {"topic": None, "intents": [], "importance": 5}
        except Exception as e:
            self.logger.warning(f"Failed to process memory agent: {e}")
            # Fallback to basic heuristics
            return {
                "topic": None,
                "intents": [],
                "importance": 5,
            }


class CharacterAgent(BaseAgent):
    """
    Periodically refreshes TARS personality summary.

    Updates character state based on recent conversations and actions
    to maintain consistent personality across long sessions.

    Only runs when sufficient time/messages have passed since last update.

    Example:
        >>> agent = CharacterAgent(llm_provider)
        >>> agent.process({
        ...     "character_state": charstate,
        ...     "conversation_history": memory.history,
        ...     "recent_actions": ["forward", "stare"]
        ... })
        # Updates charstate.summary if update is due
    """

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refresh character summary if update is due.

        Args:
            input_data: Dict with keys:
                - character_state: CharacterState object
                - conversation_history: List of message dicts
                - recent_actions: List of action strings (optional)

        Returns:
            Dict with keys:
                - updated: bool - Whether update was performed
                - summary: str - New summary (if updated)
        """
        from memory.character_state import CharacterState

        charstate: CharacterState = input_data.get("character_state")
        history: List[dict] = input_data.get("conversation_history", [])
        recent_actions: List[str] = input_data.get("recent_actions", [])

        # Check if update is needed
        if not charstate.should_update(conversation_length=len(history)):
            self.logger.debug("Character update not due yet")
            return {
                "updated": False,
                "summary": charstate.summary
            }

        # Prepare context
        last_msgs = history[-30:] if history else []

        context_summary = json.dumps({
            "recent_messages": last_msgs,
            "recent_actions": recent_actions[-10:],
            "current_summary": charstate.summary
        })[-4000:]  # Limit context size

        messages = [
            {"role": "system", "content": CHARACTER_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": context_summary},
        ]

        try:
            new_summary = self._call_llm(messages).strip()

            # Update character state
            charstate.update(new_summary)

            self.logger.info(f"Updated character summary ({len(new_summary)} chars)")

            return {
                "updated": True,
                "summary": new_summary
            }

        except Exception as e:
            self.logger.error(f"Failed to update character: {e}")
            return {
                "updated": False,
                "summary": charstate.summary
            }


class AnswerAgent(BaseAgent):
    """
    Generates final human-like response.

    Synthesizes outputs from other agents and context to create
    a unique, personality-driven response that:
    - Acknowledges what was done
    - Adds wit and humor
    - Maintains character consistency
    - Varies phrasing to avoid repetition

    Example:
        >>> agent = AnswerAgent(llm_provider)
        >>> result = agent.process({
        ...     "user_text": "move forward",
        ...     "character_summary": "I am TARS...",
        ...     "command_info": {"actions": "forward(50)"},
        ...     "memory_info": {"topic": "movement"},
        ...     "executed_actions": ["forward"]
        ... })
        >>> print(result["response"])
        "Moving forward as commanded. Try not to crash me into anything valuable."
    """

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate final response.

        Args:
            input_data: Dict with keys:
                - user_text: str - Original command
                - character_summary: str - Current character state
                - command_info: dict - Output from CommandAgent
                - memory_info: dict - Output from MemoryAgent
                - executed_actions: List[str] - Actions that were executed
                - recent_visual: str (optional) - Recent visual context
                - language: str (optional) - Current language ("en" or "lt")

        Returns:
            Dict with keys:
                - response: str - Generated response text
        """
        user_text = input_data.get("user_text", "")
        character_summary = input_data.get("character_summary", "")
        command_info = input_data.get("command_info", {})
        memory_info = input_data.get("memory_info", {})
        executed_actions = input_data.get("executed_actions", [])
        recent_visual = input_data.get("recent_visual", "")
        language = input_data.get("language", "en")

        system_prompt = get_answer_agent_system_prompt(
            character_summary=character_summary,
            command_info=command_info,
            memory_info=memory_info,
            executed_actions=executed_actions,
            recent_visual=recent_visual,
            language=language
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

        try:
            response_text = self._call_llm(messages)

            return {
                "response": response_text
            }

        except Exception as e:
            self.logger.error(f"Answer generation failed: {e}")
            # Fallback response
            return {
                "response": "Command processed, but my wit generator malfunctioned."
            }


class ConversationAgent:
    """
    Main conversation orchestrator.

    Coordinates all four agents to process user input and generate responses.
    This is the main entry point for LLM-based interaction.

    PERFORMANCE OPTIMIZATIONS:
    - Fast path for simple conversational queries (single LLM call)
    - Async memory tagging (runs in background after response)
    - Uses gpt-4o-mini for CommandAgent and MemoryAgent (faster/cheaper)
    """

    # Keywords that indicate an action-based command (needs full pipeline)
    ACTION_KEYWORDS = {
        'forward', 'backward', 'back', 'left', 'right', 'turn', 'spin', 'stop',
        'dance', 'wiggle', 'move', 'go', 'drive', 'roam', 'stare', 'follow',
        'look', 'head', 'pan', 'tilt', 'scan', 'macro', 'remember', 'save',
        'teach', 'run', 'repeat', 'search', 'browse', 'web', 'label', 'name',
        'show', 'describe', 'see', 'what do you see', 'start', 'begin'
    }

    def __init__(
        self,
        llm_provider: ILLMProvider,
        character_state,
        conversation_memory,
        visual_memory,
        executor,
        fast_model: str = "gpt-4o-mini"
    ):
        """
        Initialize conversation agent.

        Args:
            llm_provider: LLM provider implementing ILLMProvider interface
            character_state: CharacterState instance for personality
            conversation_memory: ConversationMemory instance
            visual_memory: VisualMemory instance
            executor: ActionExecutor instance
            fast_model: Model to use for faster operations (default: gpt-4o-mini)
        """
        self.llm = llm_provider
        self.character = character_state
        self.conversation = conversation_memory
        self.visual_memory = visual_memory
        self.executor = executor
        self.fast_model = fast_model

        # Initialize all agents
        self.command_agent = CommandAgent(llm_provider)
        self.memory_agent = MemoryAgent(llm_provider)
        self.character_agent = CharacterAgent(llm_provider)
        self.answer_agent = AnswerAgent(llm_provider)

        self.logger = logging.getLogger(__name__)

        # Background thread for async memory tagging
        self._memory_thread = None

    def _is_simple_conversational_query(self, text: str) -> bool:
        """
        Check if input is a simple conversational query (no actions needed).

        Simple queries like "how are you", "what's your name", "tell me a joke"
        can use a fast single-LLM-call path instead of the full 3-agent pipeline.

        Args:
            text: User input text

        Returns:
            bool: True if this is a simple conversational query
        """
        text_lower = text.lower()
        words = set(text_lower.split())

        # Check for action keywords
        for keyword in self.ACTION_KEYWORDS:
            if keyword in text_lower:
                return False

        # Check for question words that might need actions
        action_question_patterns = [
            "can you move", "can you go", "can you turn", "can you dance",
            "please move", "please go", "please turn"
        ]
        for pattern in action_question_patterns:
            if pattern in text_lower:
                return False

        return True

    def _fast_conversational_response(self, user_input: str, language: str = "en") -> str:
        """
        Generate response using single LLM call for simple conversational queries.

        This is 2-3x faster than the full pipeline as it skips CommandAgent
        and MemoryAgent, using a single optimized prompt.

        Args:
            user_input: User's conversational input
            language: Language code

        Returns:
            str: Response text
        """
        # Build a simple conversational prompt with TARS personality
        language_instruction = ""
        if language == "lt":
            language_instruction = "\nIMPORTANT: Respond in Lithuanian."

        system_prompt = f"""You are TARS, a sarcastic reconnaissance robot with humor setting at 75%.
Character: {self.character.summary}
Keep responses concise (1-2 sentences), witty, and in character.{language_instruction}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        try:
            # Use fast model for simple queries
            response = self.llm.chat(messages=messages, stream=False, model=self.fast_model)
            text = self._extract_llm_text(response)

            if text:
                # Store in conversation memory (async to not block)
                self._async_store_conversation(user_input, text, language)
                return text

        except Exception as e:
            self.logger.error(f"Fast response failed: {e}")

        return "My wit circuits are temporarily overloaded."

    def _async_store_conversation(self, user_input: str, response: str, language: str):
        """Store conversation in memory asynchronously."""
        import threading

        def store():
            try:
                self.conversation.add_tagged_message(
                    role="user", content=user_input, topic="conversation", intents=["chat"]
                )
                self.conversation.add_tagged_message(
                    role="assistant", content=response, topic="conversation", intents=["chat"]
                )
            except Exception as e:
                self.logger.debug(f"Async memory store failed: {e}")

        t = threading.Thread(target=store, daemon=True)
        t.start()

    def _async_memory_tagging(self, user_input: str, response: str, intents: list):
        """
        Run MemoryAgent in background to tag and store conversation.

        This doesn't block the response - memory is tagged asynchronously.

        Args:
            user_input: User's input text
            response: Generated response text
            intents: Intents from CommandAgent
        """
        import threading

        def tag_and_store():
            try:
                # Get context for memory agent
                recent_topics = self.conversation.find_recent_topics(max_items=5)
                recent_dialogue = self._get_recent_dialogue(max_items=3)

                # Run memory agent
                memory_result = self.memory_agent.process({
                    "user_text": user_input,
                    "recent_topics": recent_topics,
                    "recent_dialogue": recent_dialogue,
                    "recent_visual": ""
                })

                # Store tagged messages
                topic = memory_result.get("topic")
                final_intents = memory_result.get("intents", intents)

                self.conversation.add_tagged_message(
                    role="user",
                    content=user_input,
                    topic=topic,
                    intents=final_intents
                )
                self.conversation.add_tagged_message(
                    role="assistant",
                    content=response,
                    topic=topic,
                    intents=final_intents
                )

            except Exception as e:
                self.logger.debug(f"Async memory tagging failed: {e}")

        t = threading.Thread(target=tag_and_store, daemon=True)
        t.start()

    def chat(self, user_input: str, language: str = "en", image_data: str = None) -> str:
        """
        Process user input and generate response.

        FAST PATH: Simple conversational queries use single LLM call.
        FULL PATH: Action-based commands use 3-agent pipeline.

        Args:
            user_input: User's text input
            language: Current language code ("en" or "lt")
            image_data: Optional base64 JPEG image for vision queries

        Returns:
            str: Response text to speak
        """
        try:
            # Check for vision-related queries with image data
            # Includes Lithuanian variants: "ka matai" = "what do you see"
            vision_keywords = ["see", "look", "describe", "what's", "what is", "show me", "view",
                               "ka matai", "ką matai", "kamatai"]
            is_vision_query = image_data and any(kw in user_input.lower() for kw in vision_keywords)

            # If this is a vision query with image, analyze the image directly
            if is_vision_query:
                vision_response = self._handle_vision_query(user_input, image_data, language)
                if vision_response:
                    return vision_response

            # FAST PATH: Simple conversational queries (no actions)
            # Skip the 3-agent pipeline for queries like "how are you", "tell me a joke"
            if self._is_simple_conversational_query(user_input) and not image_data:
                self.logger.debug(f"[FAST PATH] Simple query: {user_input[:50]}")
                return self._fast_conversational_response(user_input, language)

            # FULL PATH: Action-based commands need the full pipeline
            self.logger.debug(f"[FULL PATH] Action query: {user_input[:50]}")

            # Step 1: Parse command and extract actions
            # Include available macros in context
            available_macros = {}
            if self.executor.macro_store:
                available_macros = self.executor.macro_store.list_macros()

            command_result = self.command_agent.process({
                "user_text": user_input,
                "available_macros": available_macros
            })

            actions_str = command_result.get("actions", "")
            intents = command_result.get("intents", [])

            # Step 2: Execute actions if any
            executed_actions = []
            if actions_str and actions_str.strip():
                executed_actions = self._execute_action_string(actions_str)
                self.logger.info(f"[ACTION] Executed: {executed_actions}")

                # Record actions in macro store for potential learning
                if self.executor.macro_store and executed_actions:
                    self.executor.macro_store.set_last_actions(actions_str)

            # Handle macro save intent
            if "macro_save" in intents and self.executor.macro_store:
                macro_name = self._extract_macro_name(user_input)
                if macro_name:
                    if self.executor.macro_store.save_macro(macro_name):
                        self.logger.info(f"[MACRO] Saved macro '{macro_name}'")
                    else:
                        self.logger.warning(f"[MACRO] Failed to save macro '{macro_name}'")

            # Step 3: Generate answer FIRST (don't wait for MemoryAgent)
            # This is the user-facing response - prioritize speed
            answer_result = self.answer_agent.process({
                "user_text": user_input,
                "character_summary": self.character.summary,
                "command_info": command_result,
                "memory_info": {"topic": None, "intents": intents, "importance": 5},
                "executed_actions": executed_actions,
                "recent_visual": "",
                "language": language
            })

            response_text = answer_result.get("response", "Acknowledged.")

            # Step 4: Run MemoryAgent ASYNC (after response is ready)
            # User doesn't need to wait for memory tagging
            self._async_memory_tagging(user_input, response_text, intents)

            # Periodically update character state (also async)
            conversation_length = len(self.conversation.history)
            if self.character.should_update(conversation_length=conversation_length, min_interval_sec=180):
                import threading
                threading.Thread(target=self._update_character, daemon=True).start()

            return response_text

        except Exception as e:
            self.logger.error(f"Conversation processing failed: {e}")
            return "My cognitive circuits are experiencing interference."

    def _get_recent_dialogue(self, max_items: int = 5) -> str:
        """
        Get formatted recent dialogue for context.

        Args:
            max_items: Maximum number of messages to include

        Returns:
            Formatted dialogue string
        """
        if not self.conversation.history:
            return "No recent conversation."

        recent = self.conversation.history[-max_items * 2:]  # User + assistant pairs
        lines = []
        for msg in recent:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                lines.append(f"User: {content}")
            elif role == "assistant":
                lines.append(f"TARS: {content}")

        return "\n".join(lines)

    def _update_character(self):
        """Update character state summary periodically."""
        try:
            # Run character agent with proper parameters
            result = self.character_agent.process({
                "character_state": self.character,
                "conversation_history": self.conversation.history,
                "recent_actions": []  # TODO: Get recent actions from state
            })

            # CharacterAgent returns {"updated": bool, "summary": str}
            if result.get("updated"):
                new_summary = result.get("summary", "")
                if new_summary:
                    self.logger.info(f"Character updated (#{self.character.update_counter})")

        except Exception as e:
            self.logger.error(f"Character update failed: {e}")

    def _get_recent_topics(self, max_items: int = 5) -> str:
        """Get recent conversation topics."""
        if not self.conversation.history:
            return "No recent topics."

        recent = self.conversation.history[-max_items:]
        topics = [msg.get("topic", "") for msg in recent if msg.get("topic")]

        if not topics:
            return "General conversation."

        return ", ".join(topics)

    def _execute_action_string(self, actions_str: str) -> List[str]:
        """
        Parse and execute action calls from LLM response.

        Parses strings like "forward(50), wiggle(), dance()" and executes each action.

        Args:
            actions_str: Comma-separated action calls

        Returns:
            List of executed action names
        """
        import re

        executed = []

        # Extract action calls like "forward(50)", "dance()", etc.
        action_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)'
        matches = re.findall(action_pattern, actions_str)

        for action_name, params_str in matches:
            action_name = action_name.lower().strip()

            # Try to execute the action
            try:
                # For now, ignore parameters and just execute the action
                # The ActionExecutor will handle default parameters
                success = self.executor.execute_action(action_name)

                if success:
                    executed.append(action_name)
                    self.logger.info(f"[ACTION] Successfully executed: {action_name}")
                else:
                    self.logger.warning(f"[ACTION] Failed to execute: {action_name}")

            except Exception as e:
                self.logger.error(f"[ACTION] Error executing {action_name}: {e}")

        return executed

    def _extract_macro_name(self, text: str) -> Optional[str]:
        """
        Extract macro name from commands like "remember that as X" or "teach X".

        Args:
            text: User input text

        Returns:
            str: Macro name if found, None otherwise

        Examples:
            >>> agent._extract_macro_name("remember that as greet")
            "greet"
            >>> agent._extract_macro_name("save that as dance_move")
            "dance_move"
            >>> agent._extract_macro_name("teach wiggle")
            "wiggle"
        """
        import re

        text_lower = text.lower()

        # Pattern: "remember/save that as NAME"
        match = re.search(r'(?:remember|save)\s+(?:that|it)\s+as\s+([a-zA-Z_][a-zA-Z0-9_]*)', text_lower)
        if match:
            return match.group(1)

        # Pattern: "teach NAME"
        match = re.search(r'teach\s+([a-zA-Z_][a-zA-Z0-9_]*)', text_lower)
        if match:
            return match.group(1)

        # Pattern: "call that NAME"
        match = re.search(r'call\s+(?:that|it)\s+([a-zA-Z_][a-zA-Z0-9_]*)', text_lower)
        if match:
            return match.group(1)

        return None

    def _handle_vision_query(self, user_input: str, image_data: str, language: str = "en") -> Optional[str]:
        """
        Handle vision-related queries by analyzing the image.

        Also stores the observation in visual memory for later labeling/recall.
        Includes known labeled items so the LLM can use their names.

        Args:
            user_input: User's question about what they see
            image_data: Base64-encoded JPEG image
            language: Response language

        Returns:
            str: Description of what's seen, or None if failed
        """
        try:
            # Get known labeled items from visual memory to help identify objects by name
            known_items_str = ""
            if self.visual_memory:
                known_items = self._get_known_labeled_items()
                if known_items:
                    known_items_str = f"""
IMPORTANT - Known objects I've previously named:
{known_items}
If you see any of these objects, use their NAME instead of generic descriptions.
For example, say "Doris the microphone" instead of "a microphone" if you see that microphone."""

            # Build a vision prompt with TARS personality
            system_prompt = f"""You are TARS, a sarcastic reconnaissance robot.
Describe what you see in the image, then add a witty remark.
Keep it concise (2-3 sentences max).
Character summary: {self.character.summary}
Language: {"Lithuanian" if language == "lt" else "English"}
{known_items_str}"""

            # Create vision message with image
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_input},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }
            ]

            # Call LLM with vision
            response = self.llm.chat(messages=messages, stream=False)
            text = self._extract_llm_text(response)

            if text:
                # Store in conversation memory
                self.conversation.add_tagged_message(
                    role="user",
                    content=user_input,
                    topic="vision",
                    intents=["vision", "query"]
                )
                self.conversation.add_tagged_message(
                    role="assistant",
                    content=text,
                    topic="vision",
                    intents=["vision"]
                )

                # IMPORTANT: Store in visual memory for later labeling/recall
                # Run in background thread to avoid blocking and potential crashes
                if self.visual_memory and image_data:
                    def store_visual_async(vm, img_data, desc):
                        try:
                            import base64
                            from PIL import Image
                            import io

                            # Decode base64 image
                            img_bytes = base64.b64decode(img_data)
                            img_pil = Image.open(io.BytesIO(img_bytes))

                            # Convert to numpy array (RGB format, no cv2 conversion needed for storage)
                            import numpy as np
                            frame = np.array(img_pil)

                            # Extract tags from description
                            tags = []
                            desc_lower = desc.lower()
                            tag_map = {
                                'person': ['person', 'human', 'man', 'woman', 'someone', 'people', 'figure'],
                                'face': ['face', 'head'],
                                'pet': ['cat', 'dog', 'pet', 'animal'],
                                'plant': ['plant', 'flower', 'tree', 'greenery'],
                                'desk': ['desk', 'table', 'workspace'],
                                'computer': ['computer', 'keyboard', 'monitor', 'screen', 'laptop'],
                                'furniture': ['chair', 'sofa', 'couch', 'bed', 'furniture'],
                                'room': ['room', 'office', 'bedroom', 'living room'],
                                'toy': ['toy', 'figurine', 'doll'],
                                'gnome': ['gnome', 'gnome-like', 'garden gnome'],
                                'mushroom': ['mushroom', 'fungi', 'fungus'],
                                'microphone': ['microphone', 'mic', 'microphones'],
                                'audio': ['headphones', 'speaker', 'speakers', 'headset'],
                            }
                            for tag, keywords in tag_map.items():
                                if any(kw in desc_lower for kw in keywords):
                                    tags.append(tag)
                            if not tags:
                                tags = ['scene']

                            # Store - use RGB format directly, let visual_memory handle conversion
                            vm.add_visual(
                                image=frame,
                                description=desc,
                                tags=tags,
                                label=None,
                                save_image=True
                            )
                            print(f"[VISUAL] Stored observation: {len(desc)} chars, tags={tags}")
                        except Exception as e:
                            print(f"[VISUAL] Storage failed (non-critical): {e}")

                    # Run in background thread to avoid blocking/crashing main loop
                    import threading
                    t = threading.Thread(
                        target=store_visual_async,
                        args=(self.visual_memory, image_data, text),
                        daemon=True
                    )
                    t.start()

                return text

        except Exception as e:
            self.logger.error(f"Vision query failed: {e}")

        return None

    def _extract_tags_from_description(self, description: str) -> List[str]:
        """
        Extract simple tags from a vision description.

        Args:
            description: Text description of what was seen

        Returns:
            List of tag strings
        """
        tags = []
        desc_lower = description.lower()

        # Common objects to tag
        tag_keywords = {
            'person': ['person', 'human', 'man', 'woman', 'someone', 'people', 'figure'],
            'face': ['face', 'head'],
            'pet': ['cat', 'dog', 'pet', 'animal'],
            'plant': ['plant', 'flower', 'tree', 'greenery'],
            'desk': ['desk', 'table', 'workspace'],
            'computer': ['computer', 'keyboard', 'monitor', 'screen', 'laptop'],
            'furniture': ['chair', 'sofa', 'couch', 'bed', 'furniture'],
            'room': ['room', 'office', 'bedroom', 'living room'],
            'toy': ['toy', 'figurine', 'doll'],
            'gnome': ['gnome', 'gnome-like', 'garden gnome'],
            'mushroom': ['mushroom', 'fungi', 'fungus'],
            'microphone': ['microphone', 'mic', 'microphones'],
            'audio': ['headphones', 'speaker', 'speakers', 'headset'],
        }

        for tag, keywords in tag_keywords.items():
            if any(kw in desc_lower for kw in keywords):
                tags.append(tag)

        return tags if tags else ['scene']

    def _extract_llm_text(self, resp) -> str:
        """Extract text from LLM response (reuse BaseAgent logic)."""
        # Reuse the same extraction logic as BaseAgent
        if isinstance(resp, str):
            return resp.strip()

        # requests.Response object
        if hasattr(resp, 'json') and callable(resp.json) and hasattr(resp, 'status_code'):
            try:
                data = resp.json()
                if isinstance(data, dict):
                    if 'choices' in data and len(data['choices']) > 0:
                        choice = data['choices'][0]
                        if isinstance(choice, dict) and 'message' in choice:
                            message = choice['message']
                            if isinstance(message, dict) and 'content' in message:
                                content = message['content']
                                if content is not None:
                                    return str(content).strip()
            except:
                pass

        # Object with .content attribute
        if hasattr(resp, 'content') and isinstance(resp.content, str):
            return resp.content.strip()

        # OpenAI ChatCompletion object
        if hasattr(resp, 'choices') and len(resp.choices) > 0:
            choice = resp.choices[0]
            if hasattr(choice, 'message'):
                message = choice.message
                if hasattr(message, 'content') and message.content is not None:
                    return str(message.content).strip()

        # Dict response
        if isinstance(resp, dict):
            if 'choices' in resp and len(resp['choices']) > 0:
                choice = resp['choices'][0]
                if isinstance(choice, dict) and 'message' in choice:
                    message = choice['message']
                    if isinstance(message, dict) and 'content' in message:
                        return str(message['content']).strip()

        return str(resp).strip() if resp else ""

    def _get_known_labeled_items(self) -> str:
        """
        Get a formatted string of all known labeled items from visual memory.

        Used to help the LLM identify objects by their given names.

        Returns:
            str: Formatted list of known items, e.g.:
                 "- Doris (microphone): a Blue Yeti microphone
                  - Bob (gnome): a small garden gnome figurine"
        """
        if not self.visual_memory:
            return ""

        try:
            # Get all entries with labels
            labeled_entries = []
            for entry in self.visual_memory.entries:
                if entry.label:
                    labeled_entries.append(entry)

            if not labeled_entries:
                return ""

            # Group by label to avoid duplicates, keep most recent
            by_label = {}
            for entry in labeled_entries:
                label = entry.label
                if label not in by_label:
                    by_label[label] = entry
                elif entry.timestamp > by_label[label].timestamp:
                    by_label[label] = entry

            # Format as list
            lines = []
            for label, entry in by_label.items():
                obj_type = entry.object_type or "object"
                # Get short description (first sentence or first 80 chars)
                desc = entry.description or ""
                if '.' in desc:
                    desc = desc.split('.')[0]
                if len(desc) > 80:
                    desc = desc[:77] + "..."

                lines.append(f"- {label} ({obj_type}): {desc}")

            return "\n".join(lines)

        except Exception as e:
            self.logger.warning(f"Failed to get known items: {e}")
            return ""
