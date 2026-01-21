"""
Signal, Echo, and Message types for inter-subsystem communication.

Signals represent symbolic communications between subsystems.
Echoes are replayed signals that maintain continuity across recursive cycles.
Messages are the pub/sub carrier for signals on the message bus.
"""

from datetime import UTC, datetime
from enum import Enum, auto
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from ulid import ULID


class SignalDomain(Enum):
    """Domain classification for signals (symbolic analog/digital boundary)."""

    ANALOG = auto()  # Continuous, graded signals
    DIGITAL = auto()  # Discrete, binary signals
    HYBRID = auto()  # Mixed-domain signals


class SignalPriority(Enum):
    """Priority levels for signal processing."""

    LOW = 0
    NORMAL = 50
    HIGH = 75
    CRITICAL = 100


class Signal(BaseModel):
    """
    A symbolic signal for inter-subsystem communication.

    Signals carry data between subsystems and can cross
    analog-digital thresholds, requiring validation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    domain: SignalDomain = SignalDomain.DIGITAL
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Payload
    payload: Any
    payload_type: str = "unknown"

    # Routing
    source: str  # Source subsystem name
    target: str | None = None  # Target subsystem (None = broadcast)
    correlation_id: str | None = None  # For tracking related signals

    # Signal characteristics
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    priority: SignalPriority = SignalPriority.NORMAL

    # Metadata
    tags: frozenset[str] = Field(default_factory=frozenset)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_strength_threshold(self) -> Self:
        """Ensure signal strength meets threshold for transmission."""
        if self.strength < self.threshold and not self.metadata.get("allow_weak"):
            raise ValueError(
                f"Signal strength ({self.strength}) below threshold ({self.threshold})"
            )
        return self

    @property
    def is_broadcast(self) -> bool:
        """Check if this is a broadcast signal."""
        return self.target is None

    @property
    def above_threshold(self) -> bool:
        """Check if signal is above transmission threshold."""
        return self.strength >= self.threshold

    def attenuate(self, factor: float) -> "Signal":
        """Create a new signal with reduced strength."""
        return Signal(
            domain=self.domain,
            payload=self.payload,
            payload_type=self.payload_type,
            source=self.source,
            target=self.target,
            correlation_id=self.correlation_id,
            strength=max(0.0, self.strength * factor),
            threshold=self.threshold,
            priority=self.priority,
            tags=self.tags,
            metadata={**self.metadata, "attenuated_from": self.id, "allow_weak": True},
        )

    def amplify(self, factor: float) -> "Signal":
        """Create a new signal with increased strength."""
        return Signal(
            domain=self.domain,
            payload=self.payload,
            payload_type=self.payload_type,
            source=self.source,
            target=self.target,
            correlation_id=self.correlation_id,
            strength=min(1.0, self.strength * factor),
            threshold=self.threshold,
            priority=self.priority,
            tags=self.tags,
            metadata={**self.metadata, "amplified_from": self.id},
        )

    def convert_domain(self, target_domain: SignalDomain) -> "Signal":
        """Create a new signal converted to a different domain."""
        if self.domain == target_domain:
            return self
        return Signal(
            domain=target_domain,
            payload=self.payload,
            payload_type=self.payload_type,
            source=self.source,
            target=self.target,
            correlation_id=self.correlation_id,
            strength=self.strength,
            threshold=self.threshold,
            priority=self.priority,
            tags=self.tags,
            metadata={
                **self.metadata,
                "converted_from": self.id,
                "original_domain": self.domain.name,
            },
        )


class Echo(BaseModel):
    """
    A replayed signal for maintaining continuity across recursive cycles.

    Echoes preserve the original signal while tracking replay history.
    They enable the system to "remember" past communications.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    original_signal: Signal
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Echo characteristics
    replay_count: int = 1
    decay_factor: float = Field(default=0.9, ge=0.0, le=1.0)

    # History
    replay_history: tuple[str, ...] = Field(default_factory=tuple)  # Subsystems that replayed

    @property
    def effective_strength(self) -> float:
        """Calculate signal strength after decay from replays."""
        return self.original_signal.strength * (self.decay_factor**self.replay_count)

    @property
    def is_viable(self) -> bool:
        """Check if echo still has sufficient strength for processing."""
        return self.effective_strength >= self.original_signal.threshold

    def replay(self, replaying_subsystem: str) -> "Echo":
        """Create a new echo representing another replay iteration."""
        return Echo(
            original_signal=self.original_signal,
            replay_count=self.replay_count + 1,
            decay_factor=self.decay_factor,
            replay_history=(*self.replay_history, replaying_subsystem),
        )

    def to_signal(self, source: str) -> Signal:
        """Convert echo back to a signal for transmission."""
        return Signal(
            domain=self.original_signal.domain,
            payload=self.original_signal.payload,
            payload_type=self.original_signal.payload_type,
            source=source,
            target=self.original_signal.target,
            correlation_id=self.original_signal.correlation_id,
            strength=self.effective_strength,
            threshold=self.original_signal.threshold,
            priority=self.original_signal.priority,
            tags=self.original_signal.tags | frozenset(["echo"]),
            metadata={
                **self.original_signal.metadata,
                "echo_id": self.id,
                "replay_count": self.replay_count,
                "allow_weak": True,
            },
        )


class MessageType(Enum):
    """Types of messages on the message bus."""

    SIGNAL = auto()  # Signal transmission
    ECHO = auto()  # Echo replay
    COMMAND = auto()  # System command
    EVENT = auto()  # System event notification
    REQUEST = auto()  # Request for response
    RESPONSE = auto()  # Response to request
    HEARTBEAT = auto()  # Liveness check


class Message(BaseModel):
    """
    Carrier for messages on the pub/sub message bus.

    Messages wrap signals, echoes, and other payloads for
    topic-based routing between subsystems.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    type: MessageType
    topic: str
    payload: Any
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Routing
    source: str  # Publishing subsystem
    correlation_id: str | None = None  # For request/response pairing
    reply_to: str | None = None  # Topic for responses

    # Delivery
    ttl_seconds: int | None = None  # Time-to-live
    priority: SignalPriority = SignalPriority.NORMAL

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if message has exceeded its TTL."""
        if self.ttl_seconds is None:
            return False
        age = (datetime.now(UTC) - self.created_at).total_seconds()
        return age > self.ttl_seconds

    @classmethod
    def from_signal(cls, signal: Signal, topic: str) -> "Message":
        """Create a message wrapping a signal."""
        return cls(
            type=MessageType.SIGNAL,
            topic=topic,
            payload=signal,
            source=signal.source,
            correlation_id=signal.correlation_id,
            priority=signal.priority,
        )

    @classmethod
    def from_echo(cls, echo: Echo, topic: str, source: str) -> "Message":
        """Create a message wrapping an echo."""
        return cls(
            type=MessageType.ECHO,
            topic=topic,
            payload=echo,
            source=source,
            correlation_id=echo.original_signal.correlation_id,
            priority=echo.original_signal.priority,
        )

    @classmethod
    def event(cls, topic: str, source: str, payload: Any) -> "Message":
        """Create an event message."""
        return cls(
            type=MessageType.EVENT,
            topic=topic,
            payload=payload,
            source=source,
        )

    @classmethod
    def command(cls, topic: str, source: str, payload: Any) -> "Message":
        """Create a command message."""
        return cls(
            type=MessageType.COMMAND,
            topic=topic,
            payload=payload,
            source=source,
        )
