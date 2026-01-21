"""
Topic definitions for the message bus.

Topics use a hierarchical naming convention:
  <category>.<subsystem>.<event_type>

Wildcards are supported:
  * - matches any single segment
  # - matches zero or more segments
"""

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar


class TopicCategory(Enum):
    """Top-level topic categories."""

    SYSTEM = "system"  # System-wide events (startup, shutdown, errors)
    SUBSYSTEM = "subsystem"  # Subsystem lifecycle events
    SIGNAL = "signal"  # Signal transmissions
    ECHO = "echo"  # Echo replays
    VALUE = "value"  # Value exchange events
    PROCESS = "process"  # Process lifecycle events
    DATA = "data"  # Data storage events


@dataclass(frozen=True)
class Topic:
    """
    A message bus topic with hierarchical structure.

    Topics support wildcard matching for flexible subscriptions.
    """

    path: str

    # Well-known topic patterns
    WILDCARD_SINGLE: ClassVar[str] = "*"
    WILDCARD_MULTI: ClassVar[str] = "#"

    @property
    def segments(self) -> list[str]:
        """Split topic into segments."""
        return self.path.split(".")

    @property
    def category(self) -> str:
        """Get the top-level category."""
        return self.segments[0] if self.segments else ""

    def matches(self, pattern: str) -> bool:
        """
        Check if this topic matches a pattern.

        Supports wildcards:
        - '*' matches exactly one segment
        - '#' matches zero or more segments
        """
        topic_parts = self.segments
        pattern_parts = pattern.split(".")

        return self._match_parts(topic_parts, pattern_parts)

    def _match_parts(self, topic: list[str], pattern: list[str]) -> bool:
        """Recursive pattern matching implementation."""
        if not pattern:
            return not topic

        if pattern[0] == "#":
            # '#' matches zero or more segments
            if len(pattern) == 1:
                return True
            # Try matching remaining pattern at each position
            for i in range(len(topic) + 1):
                if self._match_parts(topic[i:], pattern[1:]):
                    return True
            return False

        if not topic:
            return False

        if pattern[0] == "*" or pattern[0] == topic[0]:
            return self._match_parts(topic[1:], pattern[1:])

        return False

    def __str__(self) -> str:
        return self.path

    def __hash__(self) -> int:
        return hash(self.path)


# Well-known system topics
class SystemTopics:
    """Well-known system-level topics."""

    # System lifecycle
    STARTUP = Topic("system.startup")
    SHUTDOWN = Topic("system.shutdown")
    HEARTBEAT = Topic("system.heartbeat")

    # System errors
    ERROR = Topic("system.error")
    ERROR_FATAL = Topic("system.error.fatal")

    # All system events (wildcard)
    ALL = Topic("system.#")


class SubsystemTopics:
    """Well-known subsystem-level topics."""

    # Subsystem lifecycle (use with subsystem name)
    STARTED = Topic("subsystem.*.started")
    STOPPED = Topic("subsystem.*.stopped")
    FAILED = Topic("subsystem.*.failed")

    # All subsystem events
    ALL = Topic("subsystem.#")

    @staticmethod
    def started(name: str) -> Topic:
        """Topic for a specific subsystem started event."""
        return Topic(f"subsystem.{name}.started")

    @staticmethod
    def stopped(name: str) -> Topic:
        """Topic for a specific subsystem stopped event."""
        return Topic(f"subsystem.{name}.stopped")

    @staticmethod
    def failed(name: str) -> Topic:
        """Topic for a specific subsystem failed event."""
        return Topic(f"subsystem.{name}.failed")

    @staticmethod
    def all_for(name: str) -> Topic:
        """All topics for a specific subsystem."""
        return Topic(f"subsystem.{name}.#")


class SignalTopics:
    """Well-known signal-level topics."""

    # Generic signals
    BROADCAST = Topic("signal.broadcast")

    # All signals
    ALL = Topic("signal.#")

    @staticmethod
    def from_subsystem(name: str) -> Topic:
        """Signals from a specific subsystem."""
        return Topic(f"signal.from.{name}")

    @staticmethod
    def to_subsystem(name: str) -> Topic:
        """Signals to a specific subsystem."""
        return Topic(f"signal.to.{name}")


class ProcessTopics:
    """Well-known process-level topics."""

    # Process lifecycle
    STARTED = Topic("process.*.started")
    COMPLETED = Topic("process.*.completed")
    FAILED = Topic("process.*.failed")

    # All process events
    ALL = Topic("process.#")

    @staticmethod
    def for_subsystem(name: str) -> Topic:
        """Process events for a specific subsystem."""
        return Topic(f"process.{name}.#")


class DataTopics:
    """Well-known data-level topics."""

    # Archive events
    ARCHIVED = Topic("data.archive.stored")
    RETRIEVED = Topic("data.archive.retrieved")

    # Reference events
    REFERENCE_CREATED = Topic("data.reference.created")
    REFERENCE_UPDATED = Topic("data.reference.updated")

    # All data events
    ALL = Topic("data.#")


# Convenience function for creating topics
def topic(path: str) -> Topic:
    """Create a topic from a path string."""
    return Topic(path)
