"""
Data models for chat interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime    import datetime
from typing      import Any


@dataclass
class ChatMessage:
    """One message in a chat session."""

    role:      str          # "user" or "assistant"
    content:   str
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "role":      self.role,
            "content":   self.content,
            "timestamp": self.timestamp,
        }


@dataclass
class ChatSession:
    """A follow-up Q&A conversation tied to a specific paper."""

    arxiv_id:      str
    paper_title:   str
    original_query: str
    messages:      list[ChatMessage] = field(default_factory=list)
    started_at:    str = ""

    def __post_init__(self):
        if not self.started_at:
            self.started_at = datetime.now().isoformat()

    def add_user(self, content: str) -> None:
        self.messages.append(ChatMessage(role="user", content=content))

    def add_assistant(self, content: str) -> None:
        self.messages.append(ChatMessage(role="assistant", content=content))

    def to_dict(self) -> dict[str, Any]:
        return {
            "arxiv_id":       self.arxiv_id,
            "paper_title":    self.paper_title,
            "original_query": self.original_query,
            "started_at":     self.started_at,
            "messages":       [m.to_dict() for m in self.messages],
        }