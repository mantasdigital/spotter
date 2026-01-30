"""
Scene Analysis using Vision-Language Models.

Provides scene understanding by analyzing camera frames with LLM vision capabilities.
Generates natural language descriptions of visual scenes.
"""

import base64
import cv2
import numpy as np
from typing import Optional, Dict, Any, List
from io import BytesIO
from PIL import Image

from hardware.interfaces import ILLMProvider


class SceneAnalyzer:
    """
    Scene analyzer using vision-language models.

    Analyzes camera frames to generate natural language descriptions
    of scenes, objects, and situations.
    """

    def __init__(self, llm_provider: ILLMProvider, model_name: str = "gpt-4o"):
        """
        Initialize scene analyzer.

        Args:
            llm_provider: LLM provider with vision capabilities
            model_name: Name of vision-capable model to use
        """
        self.llm = llm_provider
        self.model_name = model_name

    def analyze_frame(
        self,
        frame: np.ndarray,
        prompt: Optional[str] = None,
        max_size: int = 512
    ) -> str:
        """
        Analyze a camera frame and generate description.

        Args:
            frame: Image frame as numpy array (BGR format from OpenCV)
            prompt: Optional specific question/prompt about the image.
                   If None, generates general scene description.
            max_size: Maximum dimension for image (resized if larger)

        Returns:
            str: Natural language description of the scene
        """
        # Resize if too large
        h, w = frame.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Encode image to base64
        image_base64 = self._encode_image(frame_rgb)

        # Default prompt if none provided
        if prompt is None:
            prompt = (
                "Describe what you see in this image in 1-2 sentences. "
                "Focus on the most important objects, people, or activities."
            )

        # Construct message with image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]

        # Call LLM with vision
        try:
            response = self.llm.chat(
                messages=messages,
                model=self.model_name,
                max_tokens=150
            )
            description = self.llm.extract_text(response)
            return description.strip()

        except Exception as e:
            print(f"Scene analysis error: {e}")
            return "Unable to analyze scene"

    def detect_objects(self, frame: np.ndarray) -> List[str]:
        """
        Detect and list objects in the frame.

        Args:
            frame: Image frame as numpy array

        Returns:
            List of object names detected in the scene
        """
        prompt = (
            "List all objects you can see in this image. "
            "Provide just a comma-separated list of object names."
        )

        description = self.analyze_frame(frame, prompt=prompt)

        # Parse comma-separated list
        objects = [obj.strip() for obj in description.split(',')]
        return objects

    def check_for_obstacle(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Check if there's an obstacle in front of the robot.

        Args:
            frame: Image frame as numpy array

        Returns:
            Dict with keys:
                - has_obstacle (bool): Whether obstacle is detected
                - description (str): What the obstacle is
                - urgency (str): 'high', 'medium', or 'low'
        """
        prompt = (
            "Is there an obstacle directly in front in this image? "
            "Answer in format: 'YES/NO: [description]. Urgency: [high/medium/low]'"
        )

        response = self.analyze_frame(frame, prompt=prompt)

        # Parse response
        has_obstacle = response.upper().startswith("YES")

        # Extract description (between : and .)
        description = "unknown"
        urgency = "medium"

        try:
            if ':' in response:
                parts = response.split(':', 1)[1].split('.')
                if parts:
                    description = parts[0].strip()

            if 'urgency:' in response.lower():
                urgency_part = response.lower().split('urgency:')[1].strip()
                if 'high' in urgency_part:
                    urgency = 'high'
                elif 'low' in urgency_part:
                    urgency = 'low'
                else:
                    urgency = 'medium'
        except Exception:
            pass

        return {
            "has_obstacle": has_obstacle,
            "description": description,
            "urgency": urgency
        }

    def identify_person(self, frame: np.ndarray, face_box: Optional[tuple] = None) -> str:
        """
        Generate description of person in the frame.

        Args:
            frame: Image frame as numpy array
            face_box: Optional (x, y, w, h) bounding box of face

        Returns:
            str: Description of the person
        """
        prompt = (
            "Describe the person in this image. "
            "Include visible characteristics like clothing, posture, or activity. "
            "Keep it brief (1-2 sentences)."
        )

        return self.analyze_frame(frame, prompt=prompt)

    def generate_tags(self, frame: np.ndarray) -> List[str]:
        """
        Generate descriptive tags for the scene.

        Args:
            frame: Image frame as numpy array

        Returns:
            List of tag strings
        """
        prompt = (
            "Generate 3-5 tags that describe this scene. "
            "Provide just the tags separated by commas (e.g., 'indoor, table, laptop')."
        )

        response = self.analyze_frame(frame, prompt=prompt)

        # Parse tags
        tags = [tag.strip().lower() for tag in response.split(',')]
        return tags[:5]  # Limit to 5 tags

    def _encode_image(self, image_rgb: np.ndarray) -> str:
        """
        Encode image to base64 string.

        Args:
            image_rgb: Image in RGB format

        Returns:
            Base64 encoded string
        """
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)

        # Encode as JPEG
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        image_bytes = buffer.getvalue()

        # Encode to base64
        return base64.b64encode(image_bytes).decode('utf-8')


class MockSceneAnalyzer:
    """
    Mock scene analyzer for testing without LLM.

    Returns simulated scene descriptions.
    """

    def __init__(self):
        """Initialize mock analyzer."""
        self.call_count = 0

    def analyze_frame(
        self,
        frame: np.ndarray,
        prompt: Optional[str] = None,
        max_size: int = 512
    ) -> str:
        """Return mock scene description."""
        self.call_count += 1
        descriptions = [
            "A room with furniture and walls",
            "An open space with floor visible",
            "A corridor or hallway"
        ]
        return descriptions[self.call_count % len(descriptions)]

    def detect_objects(self, frame: np.ndarray) -> List[str]:
        """Return mock object list."""
        return ["wall", "floor", "furniture"]

    def check_for_obstacle(self, frame: np.ndarray) -> Dict[str, Any]:
        """Return mock obstacle check."""
        return {
            "has_obstacle": False,
            "description": "clear path",
            "urgency": "low"
        }

    def identify_person(self, frame: np.ndarray, face_box: Optional[tuple] = None) -> str:
        """Return mock person description."""
        return "A person standing in the frame"

    def generate_tags(self, frame: np.ndarray) -> List[str]:
        """Return mock tags."""
        return ["indoor", "room", "space"]
