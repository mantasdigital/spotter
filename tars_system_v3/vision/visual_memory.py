"""
Visual Memory Storage.

Manages storage and retrieval of visual observations with descriptions and tags.
Persists to disk for long-term memory across sessions.
"""

import json
import time
import base64
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
import numpy as np
# Note: Using PIL instead of cv2 for image operations to avoid native crashes

from hardware.interfaces import IVisualMemory


@dataclass
class VisualMemoryEntry:
    """Single visual memory entry."""
    timestamp: float
    description: str
    tags: List[str]
    label: Optional[str]
    image_path: Optional[str]  # Path to saved image
    thumbnail_base64: Optional[str]  # Small thumbnail for quick preview
    object_type: Optional[str] = None  # The specific object being labeled (e.g., "microphone")


class VisualMemory(IVisualMemory):
    """
    Visual memory storage with tagging and search.

    Stores visual observations with descriptions, tags, and optional labels.
    Saves images to disk and maintains a searchable index.
    """

    def __init__(self, storage_dir: str = "data/visual_memory", max_entries: int = 1000):
        """
        Initialize visual memory.

        Args:
            storage_dir: Directory to store visual memory data
            max_entries: Maximum number of entries to keep (oldest removed first)
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.images_dir = self.storage_dir / "images"
        self.images_dir.mkdir(exist_ok=True)

        self.index_file = self.storage_dir / "index.json"
        self.max_entries = max_entries

        self.entries: List[VisualMemoryEntry] = []
        self.load()

    def add_visual(
        self,
        image: np.ndarray,
        description: str,
        tags: List[str],
        label: Optional[str] = None,
        save_image: bool = True
    ):
        """
        Add a visual memory.

        Args:
            image: Image frame as numpy array (RGB or BGR format)
            description: Text description of the scene
            tags: List of tag strings
            label: Optional human-provided label
            save_image: Whether to save the full image to disk
        """
        timestamp = time.time()

        # Generate filename
        image_filename = f"visual_{int(timestamp)}_{len(self.entries)}.jpg"
        image_path = str(self.images_dir / image_filename) if save_image else None

        # Save full image if requested - use PIL instead of cv2 to avoid crashes
        if save_image and image is not None:
            try:
                from PIL import Image as PILImage
                # Handle both RGB and BGR formats
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # Assume RGB from PIL, save directly
                    pil_img = PILImage.fromarray(image)
                    pil_img.save(image_path, "JPEG", quality=85)
                else:
                    # Grayscale or other format
                    pil_img = PILImage.fromarray(image)
                    pil_img.save(image_path, "JPEG", quality=85)
            except Exception as e:
                print(f"[VISUAL] Failed to save image: {e}")
                image_path = None

        # Create thumbnail (base64 encoded) - use PIL instead of cv2
        thumbnail_base64 = None
        if image is not None:
            try:
                from PIL import Image as PILImage
                import io
                pil_img = PILImage.fromarray(image)
                pil_img.thumbnail((64, 48))  # Resize in place
                buffer = io.BytesIO()
                pil_img.save(buffer, format="JPEG", quality=60)
                thumbnail_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            except Exception as e:
                print(f"[VISUAL] Failed to create thumbnail: {e}")

        # Create entry
        entry = VisualMemoryEntry(
            timestamp=timestamp,
            description=description,
            tags=tags,
            label=label,
            image_path=image_path,
            thumbnail_base64=thumbnail_base64
        )

        self.entries.append(entry)

        # Enforce max entries limit
        if len(self.entries) > self.max_entries:
            # Remove oldest entry
            old_entry = self.entries.pop(0)
            # Delete old image file if it exists
            if old_entry.image_path and Path(old_entry.image_path).exists():
                Path(old_entry.image_path).unlink()

        # Auto-save after adding
        self.save()

    def find_by_tags(self, tags: List[str], max_results: int = 5) -> List[dict]:
        """
        Find visual memories by tags.

        Args:
            tags: List of tags to search for
            max_results: Maximum number of results to return

        Returns:
            List of visual memory dicts, sorted by relevance (most recent first)
        """
        if not tags:
            # Return most recent entries
            results = self.entries[-max_results:]
            results.reverse()
            return [self._entry_to_dict(e) for e in results]

        # Search for entries matching any of the tags
        matching = []
        for entry in self.entries:
            # Check if any search tag matches any entry tag
            matches = set(tags) & set(entry.tags)
            if matches:
                # Score by number of matching tags
                score = len(matches)
                matching.append((score, entry))

        # Sort by score (descending), then by timestamp (descending)
        matching.sort(key=lambda x: (x[0], x[1].timestamp), reverse=True)

        # Return top results
        results = [self._entry_to_dict(entry) for score, entry in matching[:max_results]]
        return results

    def find_by_description(self, query: str, max_results: int = 5) -> List[dict]:
        """
        Find visual memories by description text search.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of visual memory dicts
        """
        query_lower = query.lower()
        matching = []

        for entry in self.entries:
            # Simple text matching
            if query_lower in entry.description.lower():
                matching.append(entry)
            elif entry.label and query_lower in entry.label.lower():
                matching.append(entry)

        # Return most recent matches
        matching.sort(key=lambda e: e.timestamp, reverse=True)
        return [self._entry_to_dict(e) for e in matching[:max_results]]

    def get_recent(self, max_results: int = 10) -> List[dict]:
        """
        Get most recent visual memories.

        Args:
            max_results: Maximum number of results to return

        Returns:
            List of visual memory dicts
        """
        results = self.entries[-max_results:]
        results.reverse()
        return [self._entry_to_dict(e) for e in results]

    def label_last_visual(self, label_name: str, object_type: Optional[str] = None) -> bool:
        """
        Attach a label to the most recent visual memory.

        Args:
            label_name: Name/label to attach
            object_type: The specific object being labeled (e.g., "microphone", "gnome")

        Returns:
            bool: True if successfully labeled, False if no recent visual
        """
        if not self.entries:
            return False

        # Get most recent entry
        self.entries[-1].label = label_name
        if object_type:
            self.entries[-1].object_type = object_type
        self.save()
        return True

    def find_by_label(self, label_query: str, object_type: str = None, max_results: int = 3) -> List[dict]:
        """
        Find visual memories by label with improved differentiation.

        Prioritizes:
        1. Exact label match + matching object_type (if specified)
        2. Exact label match (any object_type)
        3. Partial label match

        Args:
            label_query: Label to search for
            object_type: Optional object type to filter by (e.g., "microphone", "gnome", "human")
            max_results: Maximum number of results to return

        Returns:
            List of visual memory dicts, most recent first, with unique object types prioritized
        """
        if not label_query:
            return []

        label_lower = label_query.lower()
        exact_with_type = []  # Exact label + matching object_type
        exact_matches = []    # Exact label match
        partial_matches = []  # Partial label match

        for entry in self.entries:
            if not entry.label:
                continue

            entry_label_lower = entry.label.lower()

            # Check for exact match
            if entry_label_lower == label_lower:
                # If object_type specified and matches, highest priority
                if object_type and entry.object_type and entry.object_type.lower() == object_type.lower():
                    exact_with_type.append(entry)
                else:
                    exact_matches.append(entry)
            # Check for partial match
            elif label_lower in entry_label_lower or entry_label_lower in label_lower:
                partial_matches.append(entry)

        # Combine results: exact+type first, then exact, then partial
        all_matches = exact_with_type + exact_matches + partial_matches

        # Sort by timestamp (most recent first)
        all_matches.sort(key=lambda e: e.timestamp, reverse=True)

        # For better differentiation: if multiple object types exist for same label,
        # ensure we return diverse results (one of each type if possible)
        seen_types = set()
        diverse_results = []
        remaining = []

        for entry in all_matches:
            entry_type = entry.object_type or "unknown"
            if entry_type not in seen_types:
                seen_types.add(entry_type)
                diverse_results.append(entry)
            else:
                remaining.append(entry)

        # Fill up to max_results with diverse first, then remaining
        final_results = diverse_results[:max_results]
        if len(final_results) < max_results:
            final_results.extend(remaining[:max_results - len(final_results)])

        return [self._entry_to_dict(e) for e in final_results]

    def find_all_by_label(self, label_query: str) -> dict:
        """
        Find all visual memories by label, grouped by object type.

        Useful for disambiguation when same name is used for different objects.
        Also detects multiple instances of same type (e.g., two microphones named Bob).

        Args:
            label_query: Label to search for

        Returns:
            Dict with:
                - 'matches': List of all matches
                - 'by_type': Dict mapping object_type to list of matches
                - 'unique_types': List of unique object types found
                - 'has_duplicates': True if multiple items of same type share the name
                - 'duplicate_types': List of types that have multiple instances
        """
        if not label_query:
            return {'matches': [], 'by_type': {}, 'unique_types': [],
                    'has_duplicates': False, 'duplicate_types': []}

        label_lower = label_query.lower()
        matches = []

        for entry in self.entries:
            if not entry.label:
                continue

            entry_label_lower = entry.label.lower()
            if label_lower in entry_label_lower or entry_label_lower in label_lower:
                matches.append(entry)

        # Group by object type
        by_type = {}
        for entry in matches:
            obj_type = entry.object_type or "unknown"
            if obj_type not in by_type:
                by_type[obj_type] = []
            by_type[obj_type].append(self._entry_to_dict(entry))

        # Sort each group by timestamp
        for obj_type in by_type:
            by_type[obj_type].sort(key=lambda e: e['timestamp'], reverse=True)

        # Check for duplicates within same type (e.g., two microphones named Bob)
        duplicate_types = [t for t, items in by_type.items() if len(items) > 1]
        has_duplicates = len(duplicate_types) > 0

        return {
            'matches': [self._entry_to_dict(e) for e in matches],
            'by_type': by_type,
            'unique_types': list(by_type.keys()),
            'has_duplicates': has_duplicates,
            'duplicate_types': duplicate_types
        }

    def save(self):
        """Persist visual memory index to disk."""
        # Convert entries to dicts
        data = {
            "entries": [asdict(e) for e in self.entries],
            "max_entries": self.max_entries
        }

        with open(self.index_file, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
        """Load visual memory index from disk."""
        if not self.index_file.exists():
            return

        try:
            with open(self.index_file, 'r') as f:
                data = json.load(f)

            # Load entries
            self.entries = [
                VisualMemoryEntry(**entry_dict)
                for entry_dict in data.get("entries", [])
            ]

            self.max_entries = data.get("max_entries", self.max_entries)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load visual memory: {e}")
            self.entries = []

    def _entry_to_dict(self, entry: VisualMemoryEntry) -> dict:
        """Convert entry to dictionary for API response."""
        return {
            "timestamp": entry.timestamp,
            "description": entry.description,
            "tags": entry.tags,
            "label": entry.label,
            "image_path": entry.image_path,
            "thumbnail_base64": entry.thumbnail_base64,
            "object_type": entry.object_type
        }

    def clear(self):
        """Clear all visual memories and delete stored images."""
        # Delete all image files
        for entry in self.entries:
            if entry.image_path and Path(entry.image_path).exists():
                Path(entry.image_path).unlink()

        # Clear entries
        self.entries = []
        self.save()


class MockVisualMemory(IVisualMemory):
    """
    Mock visual memory for testing.

    Stores memories in memory without persisting to disk.
    """

    def __init__(self):
        """Initialize mock visual memory."""
        self.memories = []

    def add_visual(
        self,
        image: np.ndarray,
        description: str,
        tags: List[str],
        label: Optional[str] = None
    ):
        """Add a visual memory (stored in memory only)."""
        self.memories.append({
            "timestamp": time.time(),
            "description": description,
            "tags": tags,
            "label": label
        })

    def find_by_tags(self, tags: List[str], max_results: int = 5) -> List[dict]:
        """Find memories by tags."""
        if not tags:
            return self.memories[-max_results:]

        matching = [
            m for m in self.memories
            if any(tag in m["tags"] for tag in tags)
        ]
        return matching[-max_results:]

    def save(self):
        """No-op for mock."""
        pass

    def load(self):
        """No-op for mock."""
        pass
