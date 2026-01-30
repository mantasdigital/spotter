"""
Action Executor.

Parses and executes action sequences from LLM responses.
Extracted from tars_execute function in voice_active_car_gpt_modified_v58.py lines 3148-3240.
"""

import re
import logging
from typing import Optional, Callable, List

from motion.action_resolver import ActionResolver
from memory.macro_store import MacroStore


logger = logging.getLogger(__name__)


class ActionExecutor:
    """
    Executes action sequences from LLM responses.

    Parses ACTIONS: lines from LLM output and executes individual
    actions through the ActionResolver.

    Attributes:
        action_resolver: ActionResolver instance for looking up actions
        macro_store: Optional MacroStore for recording actions
        action_callback: Optional callback to track executed actions
    """

    def __init__(
        self,
        action_resolver: ActionResolver,
        macro_store: Optional[MacroStore] = None,
        action_callback: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize action executor.

        Args:
            action_resolver: ActionResolver for looking up action functions
            macro_store: Optional macro store for recording actions
            action_callback: Optional callback called for each executed action

        Example:
            >>> executor = ActionExecutor(resolver, macro_store)
            >>> executor.execute("Move forward. ACTIONS: forward(50)")
        """
        self.action_resolver = action_resolver
        self.macro_store = macro_store
        self.action_callback = action_callback

    def clear_accumulated_actions(self):
        """
        Clear accumulated actions for macro recording.

        Call this at the start of processing a new user command to ensure
        each command sequence is recorded separately.
        """
        if self.macro_store:
            self.macro_store.last_actions_line = None
            logger.debug("[EXECUTOR] Cleared accumulated actions for new command")

    def execute(self, text: str) -> List[str]:
        """
        Parse and execute actions from LLM response.

        Extracts the ACTIONS: line from the response and executes each
        individual action function.

        Args:
            text: LLM response text potentially containing ACTIONS: line

        Returns:
            List of executed action strings

        Example:
            >>> text = "Moving forward now. ACTIONS: forward(50), turn_left()"
            >>> executed = executor.execute(text)
            >>> print(executed)
            ['forward(50)', 'turn_left()']
        """
        text = self._to_plain_text(text)

        # Extract ACTIONS: line
        action_match = re.search(r"ACTIONS:\s*(.*)", text, re.IGNORECASE)
        if not action_match:
            logger.debug("[EXECUTOR] No ACTIONS found in response")
            return []

        actions_line = action_match.group(1)
        print(f"[EXECUTOR] Actions line: {actions_line}")

        # Record to macro store if available
        if self.macro_store:
            self.macro_store.set_last_actions(actions_line)

        # Parse individual action calls (allow digits in function names like forward_24cm)
        individual_actions = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)", actions_line)
        print(f"[EXECUTOR] Parsed actions: {individual_actions}")
        if not individual_actions:
            print("[EXECUTOR] No valid action calls found")
            return []

        executed = []
        for act_str in individual_actions:
            act_str = act_str.strip()
            print(f"[EXECUTING] {act_str}")

            # Track action if callback provided
            if self.action_callback:
                self.action_callback(act_str)

            # Execute action
            try:
                action_fn = self.action_resolver[act_str]
                print(f"[EXECUTING] Starting: {act_str}")
                action_fn()
                print(f"[EXECUTING] Completed: {act_str}")
                executed.append(act_str)
            except Exception as e:
                print(f"[EXECUTING] Failed: {act_str} - {e}")

        return executed

    def execute_action(self, action_name: str) -> bool:
        """
        Execute a single action by name.

        Args:
            action_name: Action name (e.g., "forward", "stop", "turn_left_30deg")

        Returns:
            bool: True if action executed successfully, False otherwise

        Example:
            >>> executor.execute_action("forward")
            True
            >>> executor.execute_action("turn_left_45deg")
            True
        """
        try:
            # Try to get action from resolver
            action_fn = self.action_resolver[action_name]
            action_fn()

            # Track action if callback provided
            if self.action_callback:
                self.action_callback(action_name)

            # Record to macro store for later "save macro" command
            # Accumulate actions so multi-commands get saved together
            if self.macro_store:
                action_call = f"{action_name}()"
                if self.macro_store.last_actions_line:
                    # Append to existing actions
                    self.macro_store.last_actions_line += f", {action_call}"
                else:
                    self.macro_store.last_actions_line = action_call
                logger.debug(f"[EXECUTOR] Recorded action for macro: {self.macro_store.last_actions_line}")

            return True

        except Exception as e:
            logger.error(f"Failed to execute action '{action_name}': {e}")
            return False

    def _to_plain_text(self, x) -> str:
        """
        Extract plain text from various response formats.

        Handles strings, objects with text methods, and dictionaries.

        Args:
            x: Response object (str, dict, or object with text methods)

        Returns:
            Plain text string

        Example:
            >>> text = executor._to_plain_text({"response": "Hello"})
            >>> print(text)
            'Hello'
        """
        if isinstance(x, str):
            return x.strip()

        # Try common method names
        for attr in ("get_message", "get_text", "get_content"):
            fn = getattr(x, attr, None)
            if callable(fn):
                try:
                    val = fn()
                    if isinstance(val, str):
                        return val.strip()
                except Exception:
                    pass

        # Try common attribute names
        for attr in ("message", "content", "text"):
            val = getattr(x, attr, None)
            if isinstance(val, str):
                return val.strip()

        # Try dictionary keys
        if isinstance(x, dict):
            for key in ("response", "content", "text", "message"):
                if key in x and isinstance(x[key], str):
                    return x[key].strip()

        # Fallback to string conversion
        return str(x).strip()

    def rewrite_for_stop_command(self, original_prompt: str, llm_response: str) -> str:
        """
        Rewrite LLM response if "stop" command detected in prompt.

        If user command contains "stop" (like "go forward and then stop"),
        force ACTIONS to just 'stop()' for safety.

        Args:
            original_prompt: Original user command
            llm_response: LLM's generated response

        Returns:
            Modified response with stop action if needed

        Note:
            If prompt is purely "stop", doesn't rewrite (emergency stop
            should already be handled). Only rewrites when "stop" appears
            as part of a larger command.

        Example:
            >>> prompt = "move forward then stop"
            >>> response = "Moving. ACTIONS: forward(100)"
            >>> rewritten = executor.rewrite_for_stop_command(prompt, response)
            >>> print(rewritten)
            'Moving. ACTIONS: stop()'
        """
        # Normalize prompt
        norm = re.sub(r"[^\w\s]", " ", original_prompt.lower())
        norm = re.sub(r"\s+", " ", norm).strip()

        # If whole command is just "stop...", don't rewrite
        # (emergency path should already be handled)
        if norm.startswith("stop"):
            return llm_response

        # If "stop" appears anywhere else in command, force hard stop
        if " stop " in f" {norm} ":
            # Ensure we keep any spoken reply but overwrite ACTIONS line
            if "ACTIONS:" in llm_response:
                # Replace everything after ACTIONS: with 'stop()'
                rewritten = re.sub(
                    r"(ACTIONS:\s*)(.*)",
                    r"\1stop()",
                    llm_response,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                return rewritten
            else:
                # No ACTIONS present; append one
                return llm_response.rstrip() + " ACTIONS: stop()"

        return llm_response
