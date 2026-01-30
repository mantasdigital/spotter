"""
Character State Management.

Manages TARS personality/context summary, updated periodically to maintain
consistent character across conversations.
"""

import json
import os
import time
import logging
from typing import Optional


class CharacterState:
    """
    Manages TARS personality and context summary.

    The character state is a compact summary of TARS personality and recent
    context, periodically refreshed by the Character Agent LLM to maintain
    consistency across long conversations.

    Example:
        >>> charstate = CharacterState("/path/to/character.json")
        >>> if charstate.should_update(conversation_length=20):
        ...     charstate.summary = llm_character_agent()
        ...     charstate.save()
    """

    def __init__(self, filepath: str, default_summary: Optional[str] = None):
        """
        Initialize character state.

        Args:
            filepath: Path to JSON file for persistence
            default_summary: Default personality summary if file doesn't exist
        """
        self.filepath = filepath
        self.logger = logging.getLogger(__name__)

        self.summary = default_summary or \
            "You are TARS, a sarcastic reconnaissance robot on a PiCar-X."
        self.last_update_ts = 0.0
        self.update_counter = 0

        self.load()

    def load(self):
        """
        Load character state from JSON file.

        If file doesn't exist or is invalid, keeps default values.
        """
        if not os.path.exists(self.filepath):
            self.logger.info(f"[CHAR] No existing character file at {self.filepath}, using defaults")
            return

        try:
            with open(self.filepath, "r") as f:
                data = json.load(f)

            self.summary = data.get("summary", self.summary)
            self.last_update_ts = data.get("last_update_ts", 0.0)
            self.update_counter = data.get("update_counter", 0)

            self.logger.info(f"[CHAR] Loaded character state from {self.filepath}")

        except Exception as e:
            self.logger.error(f"[CHAR] Failed to load character state: {e}")

    def save(self):
        """
        Persist character state to JSON file.

        Updates the update counter and timestamp before saving.
        """
        try:
            self.last_update_ts = time.time()
            self.update_counter += 1

            with open(self.filepath, "w") as f:
                json.dump({
                    "summary": self.summary,
                    "last_update_ts": self.last_update_ts,
                    "update_counter": self.update_counter
                }, f, indent=2)

            self.logger.info(f"[CHAR] Saved character state to {self.filepath}")

        except Exception as e:
            self.logger.error(f"[CHAR] Failed to save character state: {e}")

    def should_update(
        self,
        conversation_length: int,
        min_interval_sec: float = 180.0,
        min_messages: int = 10
    ) -> bool:
        """
        Check if character summary should be refreshed.

        Args:
            conversation_length: Total number of messages in conversation history
            min_interval_sec: Minimum seconds between updates
            min_messages: Minimum new messages since last update

        Returns:
            bool: True if update is due (either time or message threshold met)

        Example:
            >>> if charstate.should_update(conversation_length=len(memory.history)):
            ...     charstate.summary = refresh_character_summary()
            ...     charstate.save()
        """
        now = time.time()
        time_ok = (now - self.last_update_ts) >= min_interval_sec
        msg_ok = conversation_length >= self.update_counter + min_messages

        return time_ok or msg_ok

    def update(self, new_summary: str):
        """
        Update character summary and save.

        Args:
            new_summary: New personality/context summary from LLM

        Example:
            >>> new_summary = llm_character_agent()
            >>> charstate.update(new_summary)
        """
        self.summary = new_summary
        self.save()
