"""Odds and ends the chat application needs: images, and the conversation so far."""

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from pydantic_ai import BinaryContent


def load_local_image(image_path: str) -> BinaryContent | None:
    """Load a local image file and return BinaryContent"""
    try:
        path = Path(image_path)
        if not path.exists():
            print(f"❌ Image file not found: {image_path}")
            return None

        if not path.is_file():
            print(f"❌ Path is not a file: {image_path}")
            return None

        # Determine media type based on file extension
        extension = path.suffix.lower()
        media_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
        }

        media_type = media_type_map.get(extension, "image/jpeg")

        # Read the image data
        image_data = path.read_bytes()
        return BinaryContent(data=image_data, media_type=media_type)

    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None


@dataclass
class ConversationHistory:
    """Store conversation history for memory"""

    messages: List[Dict[str, Any]]

    def add_message(
        self,
        role: str,
        content: str,
        has_image: bool = False,
        timestamp: datetime | None = None,
    ):
        if timestamp is None:
            timestamp = datetime.now()
        self.messages.append(
            {
                "role": role,
                "content": content,
                "has_image": has_image,
                "timestamp": timestamp.isoformat(),
            }
        )

    def get_recent_messages(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent messages for context"""
        return self.messages[-limit:]

    def get_formatted_history(self, limit: int = 5) -> str:
        """Format recent conversation history for the AI context"""
        recent = self.get_recent_messages(limit)
        formatted = []
        for msg in recent:
            content = msg["content"]
            if msg.get("has_image", False):
                content += " [included an image]"
            formatted.append(f"{msg['role']}: {content}")
        return "\n".join(formatted)


def is_local_image_path(text: str) -> bool:
    """Check if text contains a local image file path"""
    # Look for file:// URLs or local paths ending with image extensions
    file_pattern = r"(?:file://)?[^\s]+\.(jpg|jpeg|png|gif|bmp|webp)"
    return bool(re.search(file_pattern, text, re.IGNORECASE))


def extract_local_image_path(text: str) -> tuple[str, str]:
    """Extract local image path from text and return (text_without_path, image_path)"""
    file_pattern = r"(?:file://)?([^\s]+\.(jpg|jpeg|png|gif|bmp|webp))"
    match = re.search(file_pattern, text, re.IGNORECASE)
    if match:
        image_path = match.group(1)
        # Remove file:// prefix if present
        image_path = image_path.replace("file://", "")
        text_without_path = text.replace(match.group(), "").strip()
        return text_without_path, image_path
    return text, ""
