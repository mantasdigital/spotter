"""
Visual Memory Command Handlers.

Handles voice commands for labeling and recalling visual memories.
Includes "that mushroom name is doris" and "who is doris" from v58.
"""

import re
import datetime
from typing import Dict, Any, Optional

from vision.visual_memory import VisualMemory


class VisualCommandHandler:
    """
    Handler for visual memory commands.

    Supports:
    - Labeling: "that mushroom name is doris", "call that mushroom bob"
    - Recall: "who is doris", "do you remember doris", "how does doris look"
    """

    # Object identifiers that can be named
    # Includes common STT misrecognitions: "known" for "gnome", "mike" for "mic"
    OBJECT_WORDS_LIST = [
        "human", "person", "figure", "object", "creature", "pet", "gnome", "known",
        "toy", "mushroom", "thing", "one", "cat", "dog", "plant", "item",
        "microphone", "mic", "mike", "keyboard", "computer", "robot", "car",
        "bottle", "cup", "phone", "book", "lamp", "chair", "table", "box"
    ]
    OBJECT_WORDS = r"(?:" + "|".join(OBJECT_WORDS_LIST) + r")"

    # Label patterns - Capture both OBJECT and NAME
    # Group 1 = object type, Group 2 = name
    LABEL_PATTERNS_WITH_OBJECT = [
        # "name that person doris" / "name the gnome bob" / "name that thing fluffy"
        r"name\s+(?:that|the|this)?\s*(" + OBJECT_WORDS + r")\s+([a-zA-Z]+)$",

        # "that person name is doris" / "that mushroom his name is bob"
        r"(?:that|the|this)\s+(" + OBJECT_WORDS + r")\s+(?:his\s+|her\s+)?name\s+(?:is\s+)?([a-zA-Z]+)$",

        # "the person's name is doris" / "that gnome's name is bob"
        r"(?:that|the|this)\s+(" + OBJECT_WORDS + r")'?s?\s+name\s+(?:is\s+)?([a-zA-Z]+)$",

        # "that person is named doris" / "the gnome is named bob"
        r"(?:that|the|this)\s+(" + OBJECT_WORDS + r")\s+is\s+named\s+([a-zA-Z]+)$",

        # "call that person doris" / "call the gnome bob"
        r"call\s+(?:that|the|this)?\s*(" + OBJECT_WORDS + r")\s+([a-zA-Z]+)$",
    ]

    # Label patterns without object (fallback) - only capture NAME
    LABEL_PATTERNS_NAME_ONLY = [
        # "label that as doris" / "label it bob"
        r"label\s+(?:that|it|this)\s+(?:as\s+)?([a-zA-Z]+)$",

        # Simpler: "his/her name is doris" (referring to last seen)
        r"(?:his|her|its)\s+name\s+(?:is\s+)?([a-zA-Z]+)$",

        # "name him/her doris" (referring to last seen person)
        r"name\s+(?:him|her|it|them)\s+([a-zA-Z]+)$",
    ]

    # Recall patterns WITH object type - "who is Bob the gnome", "show me Bob the microphone"
    # Group 1 = name, Group 2 = object type
    RECALL_PATTERNS_WITH_TYPE = [
        r"(?:who|what)\s+is\s+([a-zA-Z]+)\s+the\s+(" + OBJECT_WORDS + r")",
        r"(?:show|shows|host|most)\s+(?:me\s+)?([a-zA-Z]+)\s+the\s+(" + OBJECT_WORDS + r")",
        r"find\s+([a-zA-Z]+)\s+the\s+(" + OBJECT_WORDS + r")",
    ]

    # Recall patterns - basic recall (who/what is X)
    RECALL_PATTERNS = [
        r"(?:who|what)\s+is\s+([a-zA-Z]+)",
        r"(?:who|what)'s\s+([a-zA-Z]+)",  # "who's doris" contraction
        r"do\s+you\s+(?:know|remember)\s+([a-zA-Z]+)",
        r"remember\s+([a-zA-Z]+)",
        r"have\s+you\s+(?:seen|met)\s+([a-zA-Z]+)",
        r"you\s+know\s+([a-zA-Z]+)",
        # "show X" patterns and STT misrecognitions (host/most/shows = show)
        r"(?:show|shows|host|most)\s+(?:me\s+)?([a-zA-Z]+)",
        r"find\s+([a-zA-Z]+)",
        r"where\s+is\s+([a-zA-Z]+)",
        r"locate\s+([a-zA-Z]+)",
    ]

    # Detailed recall patterns - user wants more info (description, first meeting, appearance)
    DETAILED_RECALL_PATTERNS = [
        r"how\s+does\s+([a-zA-Z]+)\s+look",
        r"what\s+does\s+([a-zA-Z]+)\s+look\s+like",
        r"tell\s+me\s+(?:more\s+)?about\s+([a-zA-Z]+)",
        r"describe\s+([a-zA-Z]+)",
        # When/where did you first see X
        r"(?:when|where)\s+did\s+(?:you|i|we)\s+(?:first\s+)?(?:see|meet|find)\s+([a-zA-Z]+)",
        r"(?:when|where)\s+was\s+([a-zA-Z]+)\s+(?:first\s+)?(?:seen|found)",
        # More natural variants
        r"(?:what|how)\s+(?:do\s+you\s+)?know\s+about\s+([a-zA-Z]+)",
        r"info\s+(?:on|about)\s+([a-zA-Z]+)",
        r"details\s+(?:on|about)\s+([a-zA-Z]+)",
    ]

    def __init__(self, visual_memory: VisualMemory):
        """
        Initialize visual command handler.

        Args:
            visual_memory: VisualMemory instance
        """
        self.visual_memory = visual_memory

    def can_handle(self, command: str) -> bool:
        """
        Check if this handler can process the command.

        Args:
            command: Voice command text

        Returns:
            bool: True if command is visual-related
        """
        cmd_lower = command.lower().strip()

        # Check label patterns with object
        for pattern in self.LABEL_PATTERNS_WITH_OBJECT:
            if re.search(pattern, cmd_lower):
                return True

        # Check label patterns without object (fallback)
        for pattern in self.LABEL_PATTERNS_NAME_ONLY:
            if re.search(pattern, cmd_lower):
                return True

        # Check recall patterns with object type (e.g., "who is Bob the gnome")
        for pattern in self.RECALL_PATTERNS_WITH_TYPE:
            if re.search(pattern, cmd_lower):
                return True

        # Check basic recall patterns
        for pattern in self.RECALL_PATTERNS:
            if re.search(pattern, cmd_lower):
                return True

        # Check detailed recall patterns
        for pattern in self.DETAILED_RECALL_PATTERNS:
            if re.search(pattern, cmd_lower):
                return True

        return False

    def handle(self, command: str) -> Dict[str, Any]:
        """
        Handle visual memory command.

        Args:
            command: Voice command text

        Returns:
            Dict with keys:
                - success (bool): Whether command was handled successfully
                - action (str): Action taken
                - message (str): Response message
        """
        cmd_lower = command.lower().strip()

        # Check for labeling with object type first (e.g., "that microphone name is doris")
        for pattern in self.LABEL_PATTERNS_WITH_OBJECT:
            match = re.search(pattern, cmd_lower)
            if match:
                object_type = match.group(1).strip()
                label_name = match.group(2).strip().capitalize()
                # Normalize common STT misrecognitions
                object_type = self._normalize_object_type(object_type)
                return self._handle_label_visual(label_name, object_type)

        # Check for labeling without object type (fallback)
        for pattern in self.LABEL_PATTERNS_NAME_ONLY:
            match = re.search(pattern, cmd_lower)
            if match:
                label_name = match.group(1).strip().capitalize()
                return self._handle_label_visual(label_name, object_type=None)

        # Check for detailed recall (wants description/first meeting)
        for pattern in self.DETAILED_RECALL_PATTERNS:
            match = re.search(pattern, cmd_lower)
            if match:
                name = match.group(1).strip().capitalize()
                return self._handle_recall_visual(name, detailed=True)

        # Check for recall with object type (e.g., "who is Bob the gnome")
        for pattern in self.RECALL_PATTERNS_WITH_TYPE:
            match = re.search(pattern, cmd_lower)
            if match:
                name = match.group(1).strip().capitalize()
                object_type = match.group(2).strip()
                object_type = self._normalize_object_type(object_type)
                return self._handle_recall_visual_with_type(name, object_type)

        # Check for basic recall (who/what is X)
        for pattern in self.RECALL_PATTERNS:
            match = re.search(pattern, cmd_lower)
            if match:
                name = match.group(1).strip().capitalize()
                return self._handle_recall_visual(name, detailed=False)

        return {
            "success": False,
            "action": "unknown",
            "message": "Visual command not recognized"
        }

    def _normalize_object_type(self, obj_type: str) -> str:
        """
        Normalize common STT misrecognitions for object types.

        Args:
            obj_type: Raw object type from STT

        Returns:
            Normalized object type
        """
        normalizations = {
            "known": "gnome",
            "mike": "microphone",
            "mic": "microphone",
        }
        return normalizations.get(obj_type, obj_type)

    def _extract_distinction(self, description: str, index: int) -> str:
        """
        Extract distinguishing features from a description for disambiguation.

        Looks for colors, sizes, locations, or other identifying info.

        Args:
            description: Full description text
            index: Fallback index number if no distinction found

        Returns:
            Short distinguishing phrase (e.g., "the black one", "the large one")
        """
        desc_lower = description.lower()

        # Common distinguishing words to look for
        colors = ["black", "white", "red", "blue", "green", "silver", "gold",
                  "gray", "grey", "orange", "yellow", "pink", "purple", "brown"]
        sizes = ["large", "small", "big", "tiny", "huge", "mini", "compact"]
        positions = ["left", "right", "top", "bottom", "front", "back", "corner"]
        materials = ["metal", "plastic", "wooden", "glass", "leather"]

        # Check for colors first (most distinguishing)
        for color in colors:
            if color in desc_lower:
                return f"the {color} one"

        # Check for sizes
        for size in sizes:
            if size in desc_lower:
                return f"the {size} one"

        # Check for positions/locations
        for pos in positions:
            if pos in desc_lower:
                return f"the one on the {pos}"

        # Check for materials
        for mat in materials:
            if mat in desc_lower:
                return f"the {mat} one"

        # Fallback: use first few words of description or index
        words = description.split()[:4]
        if len(words) >= 2:
            return f"'{' '.join(words)}...'"

        # Last resort: numbered
        ordinals = ["first", "second", "third", "fourth", "fifth"]
        if index <= len(ordinals):
            return f"the {ordinals[index-1]} one"
        return f"#{index}"

    def _handle_label_visual(self, label_name: str, object_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Label the most recent visual memory.

        Args:
            label_name: Name to assign
            object_type: The specific object being labeled (e.g., "microphone", "gnome")

        Returns:
            Result dict
        """
        success = self.visual_memory.label_last_visual(label_name, object_type)

        if success:
            if object_type:
                message = f"Got it! I'll remember that {object_type} as '{label_name}'"
            else:
                message = f"Got it! I've labeled that as '{label_name}'"

            return {
                "success": True,
                "action": "label_visual",
                "message": message,
                "label": label_name,
                "object_type": object_type
            }
        else:
            return {
                "success": False,
                "action": "label_visual",
                "message": "I have no recent visual memory to label",
                "label": label_name
            }

    def _handle_recall_visual(self, name: str, detailed: bool = False) -> Dict[str, Any]:
        """
        Recall a labeled visual memory by name with disambiguation.

        If multiple objects share the same name (e.g., "Bob the gnome" and "Bob the microphone"),
        the response will mention both.

        Args:
            name: Label name to search for
            detailed: If True, include full description and timestamp. If False, just identity.

        Returns:
            Result dict with visual description
        """
        # Use find_all_by_label for disambiguation
        all_matches = self.visual_memory.find_all_by_label(name)

        if not all_matches['matches']:
            return {
                "success": False,
                "action": "recall_visual",
                "message": (
                    f"I don't have any stored images labeled '{name}'. "
                    f"If you show me {name} again and tell me the name, "
                    "I can remember next time."
                ),
                "label": name
            }

        unique_types = all_matches['unique_types']
        by_type = all_matches['by_type']
        has_duplicates = all_matches.get('has_duplicates', False)
        duplicate_types = all_matches.get('duplicate_types', [])

        # If multiple different object types share this name, mention all
        if len(unique_types) > 1:
            # Disambiguate: "I know multiple things named Bob"
            type_list = []
            for obj_type in unique_types:
                count = len(by_type.get(obj_type, []))
                if obj_type != "unknown":
                    if count > 1:
                        type_list.append(f"{count} {obj_type}s")
                    else:
                        type_list.append(f"a {obj_type}")
                else:
                    type_list.append("something")

            if len(type_list) == 2:
                types_str = f"{type_list[0]} and {type_list[1]}"
            else:
                types_str = ", ".join(type_list[:-1]) + f", and {type_list[-1]}"

            response = f"I know {len(all_matches['matches'])} things named {name}: {types_str}. Which one do you mean?"

            return {
                "success": True,
                "action": "recall_visual_ambiguous",
                "message": response,
                "label": name,
                "unique_types": unique_types,
                "by_type": by_type
            }

        # Single object type but multiple instances (e.g., two microphones named Bob)
        if has_duplicates and len(unique_types) == 1:
            obj_type = unique_types[0]
            items = by_type[obj_type]
            count = len(items)

            # Try to extract distinguishing features from descriptions
            distinctions = []
            for i, item in enumerate(items[:3]):  # Max 3 to keep response short
                desc = item.get('description', '')
                # Extract first distinguishing phrase (color, size, location, etc.)
                distinction = self._extract_distinction(desc, i + 1)
                distinctions.append(distinction)

            if obj_type != "unknown":
                items_str = ", ".join(distinctions)
                response = f"I know {count} {obj_type}s named {name}: {items_str}. Which one?"
            else:
                response = f"I know {count} things named {name}. Can you be more specific?"

            return {
                "success": True,
                "action": "recall_visual_ambiguous",
                "message": response,
                "label": name,
                "count": count,
                "object_type": obj_type,
                "items": items
            }

        # Single object type - return normally
        matches = self.visual_memory.find_by_label(name)
        match = matches[0]
        object_type = match.get("object_type")
        description = match.get("description", "something")
        tags = match.get("tags", [])
        timestamp = match.get("timestamp")

        if detailed:
            # Full response with description and first meeting time
            if object_type:
                response = f"{name} is a {object_type}. "
            else:
                response = f"{name}: "

            response += description

            # Add first seen time if available
            if timestamp:
                dt = datetime.datetime.fromtimestamp(timestamp)
                time_str = dt.strftime("%B %d at %I:%M %p")
                response += f" I first saw {name} on {time_str}."
        else:
            # Concise response - just the identity (no extra tags)
            if object_type:
                response = f"{name} is a {object_type}"
            else:
                # No object type stored - use first tag or brief description
                if tags:
                    response = f"{name} is a {tags[0]}"
                else:
                    # Fallback: truncate description to first sentence
                    first_sentence = description.split('.')[0] if '.' in description else description[:100]
                    response = f"{name} is {first_sentence}"

        return {
            "success": True,
            "action": "recall_visual",
            "message": response,
            "label": name,
            "visual_data": match
        }

    def _handle_recall_visual_with_type(self, name: str, object_type: str) -> Dict[str, Any]:
        """
        Recall a specific labeled visual memory by name AND object type.

        Used for disambiguation like "who is Bob the gnome" vs "who is Bob the microphone".

        Args:
            name: Label name to search for
            object_type: Specific object type to filter by

        Returns:
            Result dict with visual description
        """
        # Search with specific object type
        matches = self.visual_memory.find_by_label(name, object_type=object_type)

        if not matches:
            return {
                "success": False,
                "action": "recall_visual",
                "message": (
                    f"I don't have any {object_type} named '{name}'. "
                    f"If you show me {name} the {object_type} and tell me, I can remember."
                ),
                "label": name,
                "object_type": object_type
            }

        # Get best match (should be filtered by object_type already)
        match = matches[0]
        stored_type = match.get("object_type")
        description = match.get("description", "something")
        timestamp = match.get("timestamp")

        # Build response
        if stored_type and stored_type.lower() == object_type.lower():
            response = f"Yes! {name} the {object_type}. {description}"
        else:
            # Found by name but different type - mention both
            response = f"{name} is a {stored_type or 'something'}, not a {object_type}. {description}"

        return {
            "success": True,
            "action": "recall_visual_typed",
            "message": response,
            "label": name,
            "object_type": object_type,
            "visual_data": match
        }
