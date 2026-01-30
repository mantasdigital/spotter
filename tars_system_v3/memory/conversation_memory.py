"""
Conversation Memory Management.

Manages dialogue history, visual observations, and roam path history with
tagging support for semantic search and recall.
"""

import json
import os
import time
import logging
from typing import List, Dict, Optional, Any


class ConversationMemory:
    """
    Manages TARS conversation and visual memory.

    Stores:
    - Dialogue history with optional topic/intent tags
    - Visual observations with timestamps and tags
    - Roam path history (compact snapshots)

    Automatically persists to JSON files with capped limits to prevent
    unbounded growth.

    Example:
        >>> memory = ConversationMemory("~/tars_memory.json", "~/tars_visual.json")
        >>> memory.add_tagged_message("user", "Start roam mode", topic="robot_control", intents=["roam"])
        >>> topics = memory.find_recent_topics(max_items=5)
    """

    def __init__(
        self,
        memory_file: str,
        visual_file: str,
        max_history: int = 50,
        max_visual: int = 100
    ):
        """
        Initialize conversation memory.

        Args:
            memory_file: Path to conversation history JSON file
            visual_file: Path to visual memory JSON file
            max_history: Maximum dialogue messages to retain
            max_visual: Maximum visual observations to retain
        """
        self.memory_file = memory_file
        self.visual_file = visual_file
        self.max_history = max_history
        self.max_visual = max_visual
        self.logger = logging.getLogger(__name__)

        self.history: List[Dict[str, Any]] = self._load_json(memory_file, [])
        self.visual: List[Dict[str, Any]] = self._load_json(visual_file, [])
        self.roam_paths: List[List[List[float]]] = []  # Compact path snapshots

    def _load_json(self, filepath: str, default: Any) -> Any:
        """
        Load JSON file with fallback to default.

        Args:
            filepath: Path to JSON file
            default: Default value if file doesn't exist or is invalid

        Returns:
            Loaded data or default value
        """
        if not os.path.exists(filepath):
            self.logger.info(f"[MEM] No existing file at {filepath}, using default")
            return default

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.logger.info(f"[MEM] Loaded from {filepath}")
            return data
        except Exception as e:
            self.logger.error(f"[MEM] Failed to load {filepath}: {e}")
            return default

    def save(self):
        """
        Persist memory to disk.

        Truncates history and visual memory to max limits before saving.
        Also saves compact roam path snapshots.
        """
        try:
            # Truncate to limits
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history:]

            if len(self.visual) > self.max_visual:
                self.visual = self.visual[-self.max_visual:]

            # Save conversation history
            with open(self.memory_file, 'w') as f:
                json.dump(self.history, f, indent=2)

            # Save visual memory
            with open(self.visual_file, "w") as f:
                json.dump(self.visual, f, indent=2)

            self.logger.info("[MEM] Memory saved to disk")

        except Exception as e:
            self.logger.error(f"[MEM] Failed to save memory: {e}")

    def add_message(self, role: str, content: str):
        """
        Add a simple message to conversation history.

        Args:
            role: Message role ("user", "assistant", "system")
            content: Message content text

        Example:
            >>> memory.add_message("user", "Hello TARS")
            >>> memory.add_message("assistant", "Hello! How can I help?")
        """
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })

        # Auto-save periodically
        if len(self.history) % 4 == 0:
            self.save()

    def add_tagged_message(
        self,
        role: str,
        content: str,
        topic: Optional[str] = None,
        intents: Optional[List[str]] = None
    ):
        """
        Add a tagged message with semantic metadata.

        Args:
            role: Message role ("user", "assistant", "system")
            content: Message content text
            topic: Optional high-level topic tag (e.g., "robot_self", "super_mario")
            intents: Optional intent tags (e.g., ["stare", "follow"])

        Example:
            >>> memory.add_tagged_message(
            ...     "user",
            ...     "Stare at me",
            ...     topic="robot_control",
            ...     intents=["stare", "face_tracking"]
            ... )
        """
        entry: Dict[str, Any] = {
            "role": role,
            "content": content,
            "timestamp": time.time()
        }

        if topic:
            entry["topic"] = topic

        if intents:
            entry["intents"] = intents

        self.history.append(entry)

        # Auto-save periodically
        if len(self.history) % 4 == 0:
            self.save()

    def find_recent_topics(self, max_items: int = 5) -> List[str]:
        """
        Get list of recent distinct topics from conversation history.

        Args:
            max_items: Maximum number of topics to return

        Returns:
            List of topic strings, most recent first

        Example:
            >>> topics = memory.find_recent_topics(max_items=3)
            >>> print(topics)  # ["robot_control", "super_mario", "memory_management"]
        """
        topics = []
        seen = set()

        for msg in reversed(self.history):
            topic = msg.get("topic")
            if not topic or topic in seen:
                continue

            seen.add(topic)
            topics.append(topic)

            if len(topics) >= max_items:
                break

        return topics

    def find_recent_intents(
        self,
        intent_filter: Optional[List[str]] = None,
        max_items: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find recent messages with specific intents.

        Args:
            intent_filter: List of intent tags to search for (OR matching)
            max_items: Maximum number of messages to return

        Returns:
            List of message dicts that match any of the intent filters

        Example:
            >>> # Find messages related to face tracking
            >>> stare_msgs = memory.find_recent_intents(["stare", "follow"])
        """
        if not intent_filter:
            return []

        wanted = set(intent_filter)
        matches = []

        for msg in reversed(self.history):
            intents = set(msg.get("intents", []))
            if not intents:
                continue

            # Check if any wanted intents are in this message's intents
            if intents & wanted:  # Set intersection
                matches.append(msg)
                if len(matches) >= max_items:
                    break

        return matches

    def add_visual(self, entry: Dict[str, Any]):
        """
        Add a visual observation to memory.

        Args:
            entry: Visual memory dict with keys like:
                - timestamp: float
                - file: str (image path)
                - tags: List[str]
                - summary: str (scene description)
                - label: Optional[str] (user-provided label)

        Example:
            >>> memory.add_visual({
            ...     "timestamp": time.time(),
            ...     "file": "~/tars_images/frame_001.jpg",
            ...     "tags": ["person", "indoor", "bright"],
            ...     "summary": "A person standing in a bright indoor space"
            ... })
        """
        self.visual.append(entry)

        # Auto-save periodically
        if len(self.visual) % 5 == 0:
            self.save()

    def find_visual_by_tags(
        self,
        tags: List[str],
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find visual memories by tags.

        Args:
            tags: List of tags to search for (OR matching)
            max_results: Maximum number of results

        Returns:
            List of visual memory dicts that match any of the tags

        Example:
            >>> faces = memory.find_visual_by_tags(["person", "face"])
        """
        wanted = set(tags)
        matches = []

        for visual in reversed(self.visual):
            visual_tags = set(visual.get("tags", []))
            if visual_tags & wanted:
                matches.append(visual)
                if len(matches) >= max_results:
                    break

        return matches

    def get_recent_messages(self, max_items: int = 10) -> List[Dict[str, Any]]:
        """
        Get most recent messages.

        Args:
            max_items: Maximum number of messages to return

        Returns:
            List of recent message dicts
        """
        return self.history[-max_items:] if self.history else []

    def clear(self):
        """
        Clear all memory and delete files.

        WARNING: This is destructive and cannot be undone.
        """
        self.history = []
        self.visual = []
        self.roam_paths = []

        # Delete files
        for filepath in [self.memory_file, self.visual_file]:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    self.logger.info(f"[MEM] Deleted {filepath}")
                except Exception as e:
                    self.logger.error(f"[MEM] Failed to delete {filepath}: {e}")

        self.logger.info("[MEM] Memory cleared")
