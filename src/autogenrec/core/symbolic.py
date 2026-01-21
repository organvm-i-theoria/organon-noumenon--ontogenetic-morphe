"""
Symbolic value types for the AutoGenRec system.

All data in the system is treated as symbolic, carrying meaning beyond raw values.
This module defines the core symbolic abstractions used throughout the system.
"""

from datetime import UTC, datetime
from enum import Enum, auto
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from ulid import ULID


class SymbolicValueType(Enum):
    """Types of symbolic values in the system."""

    # Narrative and linguistic
    NARRATIVE = auto()  # Stories, sequences, narrative fragments
    LINGUISTIC = auto()  # Constructed languages, linguistic structures
    DREAM = auto()  # Dream content, visions
    VISION = auto()  # Prophetic or symbolic visions

    # Structural and rule-based
    RULE = auto()  # Compiled rules, constraints
    PATTERN = auto()  # Recognized patterns
    SCHEMA = auto()  # Structural schemas

    # Communication
    SIGNAL = auto()  # System signals
    MESSAGE = auto()  # Inter-subsystem messages
    ECHO = auto()  # Replayed or reflected signals

    # Value and exchange
    TOKEN = auto()  # Symbolic tokens for exchange  # allow-secret
    CURRENCY = auto()  # Symbolic currency units
    ASSET = auto()  # Symbolic assets

    # Identity
    MASK = auto()  # Identity masks
    IDENTITY = auto()  # Identity markers
    ROLE = auto()  # Role assignments

    # Temporal and spatial
    TIMESTAMP = auto()  # Temporal markers
    LOCATION = auto()  # Spatial references
    SCHEDULE = auto()  # Temporal schedules

    # Meta
    REFERENCE = auto()  # References to other values
    ARCHIVE = auto()  # Archived content
    CODE = auto()  # Generated code or instructions
    UNKNOWN = auto()  # Unclassified symbolic content


class SymbolicValue(BaseModel):
    """
    Base class for all symbolic values in the system.

    Symbolic values carry meaning beyond their raw data representation.
    They include metadata about origin, type, and relationships.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    type: SymbolicValueType
    content: Any
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Origin tracking
    source_subsystem: str | None = None
    source_process_id: str | None = None

    # Semantic metadata
    tags: frozenset[str] = Field(default_factory=frozenset)
    meaning: str | None = None  # Human-readable interpretation
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    # Relationships
    parent_id: str | None = None  # ID of value this was derived from
    related_ids: frozenset[str] = Field(default_factory=frozenset)

    def derive(
        self,
        *,
        content: Any | None = None,
        type: SymbolicValueType | None = None,
        meaning: str | None = None,
        tags: frozenset[str] | None = None,
        **kwargs: Any,
    ) -> "SymbolicValue":
        """
        Create a new symbolic value derived from this one.

        The new value inherits properties from this value and
        tracks the derivation relationship via parent_id.
        """
        return SymbolicValue(
            type=type or self.type,
            content=content if content is not None else self.content,
            source_subsystem=kwargs.get("source_subsystem", self.source_subsystem),
            source_process_id=kwargs.get("source_process_id", self.source_process_id),
            tags=tags if tags is not None else self.tags,
            meaning=meaning if meaning is not None else self.meaning,
            confidence=kwargs.get("confidence", self.confidence),
            parent_id=self.id,
            related_ids=frozenset(kwargs.get("related_ids", [])),
        )


class SymbolicInput(BaseModel):
    """
    Container for symbolic inputs to a subsystem.

    Groups multiple symbolic values with shared context and metadata.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    values: tuple[SymbolicValue, ...] = Field(default_factory=tuple)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Context
    source_subsystem: str | None = None
    target_subsystem: str | None = None
    correlation_id: str | None = None  # For tracking related inputs/outputs

    # Processing hints
    priority: int = Field(default=0, ge=-100, le=100)
    deadline: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def primary_value(self) -> SymbolicValue | None:
        """Return the first/primary symbolic value, if any."""
        return self.values[0] if self.values else None

    def with_values(self, *values: SymbolicValue) -> "SymbolicInput":
        """Create a new input with additional values appended."""
        return SymbolicInput(
            values=(*self.values, *values),
            source_subsystem=self.source_subsystem,
            target_subsystem=self.target_subsystem,
            correlation_id=self.correlation_id,
            priority=self.priority,
            deadline=self.deadline,
            metadata=self.metadata,
        )


class SymbolicOutput(BaseModel):
    """
    Container for symbolic outputs from a subsystem.

    Tracks the transformation from inputs to outputs with provenance.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    values: tuple[SymbolicValue, ...] = Field(default_factory=tuple)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Provenance
    source_subsystem: str
    process_id: str
    input_id: str | None = None  # ID of the SymbolicInput that produced this
    correlation_id: str | None = None

    # Processing results
    iterations: int = 1
    success: bool = True
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def primary_value(self) -> SymbolicValue | None:
        """Return the first/primary symbolic value, if any."""
        return self.values[0] if self.values else None

    def as_input(
        self,
        target_subsystem: str | None = None,
        priority: int = 0,
    ) -> SymbolicInput:
        """
        Convert this output to an input for another subsystem.

        This enables the recursive feedback loop where outputs
        become inputs for further processing.
        """
        return SymbolicInput(
            values=self.values,
            source_subsystem=self.source_subsystem,
            target_subsystem=target_subsystem,
            correlation_id=self.correlation_id,
            priority=priority,
            metadata={"derived_from_output_id": self.id},
        )


# Convenience factory functions


def narrative(content: str, **kwargs: Any) -> SymbolicValue:
    """Create a narrative symbolic value."""
    return SymbolicValue(type=SymbolicValueType.NARRATIVE, content=content, **kwargs)


def rule(content: dict[str, Any], **kwargs: Any) -> SymbolicValue:
    """Create a rule symbolic value."""
    return SymbolicValue(type=SymbolicValueType.RULE, content=content, **kwargs)


def token(content: Any, **kwargs: Any) -> SymbolicValue:
    """Create a token symbolic value."""
    return SymbolicValue(type=SymbolicValueType.TOKEN, content=content, **kwargs)


def pattern(content: Any, **kwargs: Any) -> SymbolicValue:
    """Create a pattern symbolic value."""
    return SymbolicValue(type=SymbolicValueType.PATTERN, content=content, **kwargs)


def reference(target_id: str, **kwargs: Any) -> SymbolicValue:
    """Create a reference symbolic value pointing to another value."""
    return SymbolicValue(
        type=SymbolicValueType.REFERENCE,
        content={"target_id": target_id},
        related_ids=frozenset([target_id]),
        **kwargs,
    )
