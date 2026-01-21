"""
Subsystem base class extending ProcessLoop with messaging and lifecycle.

Subsystems are the primary building blocks of the AutoGenRec system.
Each subsystem processes symbolic inputs and produces symbolic outputs
while communicating with other subsystems via the message bus.
"""

from abc import abstractmethod
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, ConfigDict, Field
from ulid import ULID

from autogenrec.core.process import ProcessContext, ProcessLoop, ProcessPhase
from autogenrec.core.signals import Message, MessageType, Signal, SignalPriority
from autogenrec.core.symbolic import SymbolicInput, SymbolicOutput, SymbolicValue

if TYPE_CHECKING:
    from autogenrec.bus.message_bus import MessageBus

logger = structlog.get_logger()


class SubsystemType(Enum):
    """Classification of subsystem types."""

    META = auto()  # Meta-system (AnthologyManager)
    CORE_PROCESSING = auto()  # Symbolic interpretation, rule compilation, code generation
    CONFLICT = auto()  # Conflict resolution and arbitration
    DATA = auto()  # Reference, archive, and echo management
    ROUTING = auto()  # Node routing and signal threshold
    VALUE = auto()  # Value exchange, blockchain, monetization
    TEMPORAL = auto()  # Time, location, evolution scheduling
    IDENTITY = auto()  # Mask generation, audience classification
    TRANSFORMATION = auto()  # Process conversion, consumption management
    ACADEMIC = auto()  # Academic and research management


class SubsystemState(Enum):
    """Lifecycle states for a subsystem."""

    CREATED = auto()
    STARTING = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPING = auto()
    STOPPED = auto()
    FAILED = auto()


class SubsystemMetadata(BaseModel):
    """Metadata describing a subsystem."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    display_name: str
    description: str
    type: SubsystemType
    version: str = "0.1.0"
    tags: frozenset[str] = Field(default_factory=frozenset)

    # Capabilities
    input_types: frozenset[str] = Field(default_factory=frozenset)
    output_types: frozenset[str] = Field(default_factory=frozenset)

    # Subscriptions
    subscribed_topics: frozenset[str] = Field(default_factory=frozenset)
    published_topics: frozenset[str] = Field(default_factory=frozenset)


@dataclass
class SubsystemStats:
    """Runtime statistics for a subsystem."""

    started_at: datetime | None = None
    stopped_at: datetime | None = None
    total_inputs_processed: int = 0
    total_outputs_produced: int = 0
    total_messages_received: int = 0
    total_messages_sent: int = 0
    total_errors: int = 0
    last_activity_at: datetime | None = None


MessageHandler = Callable[[Message], Coroutine[Any, Any, None]]


@dataclass
class Subscription:
    """A topic subscription for a subsystem."""

    topic: str
    handler: MessageHandler
    subscription_id: str = field(default_factory=lambda: str(ULID()))


class Subsystem(ProcessLoop[SymbolicInput, SymbolicOutput, dict[str, Any]]):
    """
    Base class for all AutoGenRec subsystems.

    Extends ProcessLoop with:
    - Metadata and lifecycle management
    - Message bus integration
    - Topic subscriptions
    - Signal handling
    - Statistics tracking
    """

    def __init__(
        self,
        metadata: SubsystemMetadata,
        message_bus: "MessageBus | None" = None,
        max_iterations: int = 100,
    ) -> None:
        super().__init__(name=metadata.name, max_iterations=max_iterations)
        self._metadata = metadata
        self._message_bus = message_bus
        self._subsystem_state = SubsystemState.CREATED
        self._stats = SubsystemStats()
        self._subscriptions: dict[str, Subscription] = {}
        self._log = logger.bind(subsystem=metadata.name)

    # --- Properties ---

    @property
    def metadata(self) -> SubsystemMetadata:
        return self._metadata

    @property
    def subsystem_state(self) -> SubsystemState:
        return self._subsystem_state

    @property
    def stats(self) -> SubsystemStats:
        return self._stats

    @property
    def is_running(self) -> bool:
        return self._subsystem_state == SubsystemState.RUNNING

    # --- Lifecycle management ---

    async def start(self) -> None:
        """Start the subsystem and set up subscriptions."""
        if self._subsystem_state not in (SubsystemState.CREATED, SubsystemState.STOPPED):
            raise RuntimeError(f"Cannot start subsystem in state {self._subsystem_state}")

        self._subsystem_state = SubsystemState.STARTING
        self._log.info("subsystem_starting")

        try:
            # Set up topic subscriptions
            if self._message_bus:
                for topic in self._metadata.subscribed_topics:
                    await self.subscribe(topic, self._handle_message)

            # Call subclass initialization
            await self.on_start()

            self._subsystem_state = SubsystemState.RUNNING
            self._stats.started_at = datetime.now(UTC)
            self._log.info("subsystem_started")

        except Exception:
            self._subsystem_state = SubsystemState.FAILED
            self._log.exception("subsystem_start_failed")
            raise

    async def stop(self) -> None:
        """Stop the subsystem and clean up subscriptions."""
        if self._subsystem_state != SubsystemState.RUNNING:
            return

        self._subsystem_state = SubsystemState.STOPPING
        self._log.info("subsystem_stopping")

        try:
            # Call subclass cleanup
            await self.on_stop()

            # Unsubscribe from all topics
            if self._message_bus:
                for sub in list(self._subscriptions.values()):
                    await self.unsubscribe(sub.subscription_id)

            self._subsystem_state = SubsystemState.STOPPED
            self._stats.stopped_at = datetime.now(UTC)
            self._log.info("subsystem_stopped")

        except Exception:
            self._subsystem_state = SubsystemState.FAILED
            self._log.exception("subsystem_stop_failed")
            raise

    # --- Lifecycle hooks (override in subclasses) ---

    async def on_start(self) -> None:
        """Called during startup. Override for custom initialization."""

    async def on_stop(self) -> None:
        """Called during shutdown. Override for custom cleanup."""

    # --- Message bus integration ---

    def set_message_bus(self, bus: "MessageBus") -> None:
        """Set the message bus for this subsystem."""
        self._message_bus = bus

    async def subscribe(self, topic: str, handler: MessageHandler) -> str:
        """Subscribe to a topic with a handler."""
        if not self._message_bus:
            raise RuntimeError("No message bus configured")

        sub = Subscription(topic=topic, handler=handler)
        self._subscriptions[sub.subscription_id] = sub

        await self._message_bus.subscribe(topic, handler)
        self._log.debug("subscribed", topic=topic, subscription_id=sub.subscription_id)

        return sub.subscription_id

    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from a topic."""
        if subscription_id not in self._subscriptions:
            return

        sub = self._subscriptions.pop(subscription_id)
        if self._message_bus:
            await self._message_bus.unsubscribe(sub.topic, sub.handler)
        self._log.debug("unsubscribed", topic=sub.topic, subscription_id=subscription_id)

    async def publish(self, message: Message) -> None:
        """Publish a message to the message bus."""
        if not self._message_bus:
            raise RuntimeError("No message bus configured")

        await self._message_bus.publish(message)
        self._stats.total_messages_sent += 1
        self._stats.last_activity_at = datetime.now(UTC)
        self._log.debug("published", topic=message.topic, message_id=message.id)

    async def send_signal(
        self,
        topic: str,
        payload: Any,
        *,
        target: str | None = None,
        priority: SignalPriority = SignalPriority.NORMAL,
        correlation_id: str | None = None,
    ) -> Signal:
        """Send a signal via the message bus."""
        signal = Signal(
            source=self.name,
            target=target,
            payload=payload,
            payload_type=type(payload).__name__,
            priority=priority,
            correlation_id=correlation_id,
        )

        message = Message.from_signal(signal, topic)
        await self.publish(message)

        return signal

    async def emit_event(self, topic: str, payload: Any) -> None:
        """Emit an event to the message bus."""
        message = Message.event(topic, self.name, payload)
        await self.publish(message)

    # --- Message handling ---

    async def _handle_message(self, message: Message) -> None:
        """Internal message handler that dispatches to appropriate handlers."""
        self._stats.total_messages_received += 1
        self._stats.last_activity_at = datetime.now(UTC)

        try:
            if message.is_expired:
                self._log.debug("message_expired", message_id=message.id)
                return

            match message.type:
                case MessageType.SIGNAL:
                    await self.handle_signal(message.payload)
                case MessageType.ECHO:
                    await self.handle_echo(message.payload)
                case MessageType.COMMAND:
                    await self.handle_command(message)
                case MessageType.EVENT:
                    await self.handle_event(message)
                case MessageType.REQUEST:
                    await self.handle_request(message)
                case _:
                    self._log.warning("unhandled_message_type", type=message.type)

        except Exception:
            self._stats.total_errors += 1
            self._log.exception("message_handling_failed", message_id=message.id)

    # --- Message type handlers (override in subclasses) ---

    async def handle_signal(self, signal: Signal) -> None:
        """Handle an incoming signal. Override in subclasses."""

    async def handle_echo(self, echo: Any) -> None:
        """Handle an incoming echo. Override in subclasses."""

    async def handle_command(self, message: Message) -> None:
        """Handle a command message. Override in subclasses."""

    async def handle_event(self, message: Message) -> None:
        """Handle an event message. Override in subclasses."""

    async def handle_request(self, message: Message) -> None:
        """Handle a request message. Override in subclasses."""

    # --- ProcessLoop phase hooks ---

    async def on_phase_enter(self, phase: ProcessPhase) -> None:
        """Log phase transitions."""
        self._log.debug("phase_entered", phase=phase.name)

    async def on_iteration_start(self, ctx: ProcessContext[dict[str, Any]]) -> None:
        """Track iteration start."""
        self._log.debug("iteration_started", iteration=ctx.iteration)

    async def on_iteration_end(
        self, ctx: ProcessContext[dict[str, Any]], output: SymbolicOutput
    ) -> None:
        """Track iteration completion."""
        self._stats.total_inputs_processed += 1
        self._stats.total_outputs_produced += len(output.values)
        self._stats.last_activity_at = datetime.now(UTC)
        self._log.debug("iteration_completed", iteration=ctx.iteration)

    # --- Abstract methods from ProcessLoop ---

    @abstractmethod
    async def intake(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> SymbolicInput:
        """
        Phase 1: Intake - Gather and validate symbolic inputs.

        Subclasses must implement input validation and normalization.
        """
        ...

    @abstractmethod
    async def process(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> Any:
        """
        Phase 2: Process - Apply core transformation logic.

        Subclasses must implement their specific processing logic.
        """
        ...

    @abstractmethod
    async def evaluate(
        self, intermediate: Any, ctx: ProcessContext[dict[str, Any]]
    ) -> tuple[SymbolicOutput, bool]:
        """
        Phase 3: Evaluate - Assess results and synthesize outputs.

        Subclasses must return (output, should_continue).
        """
        ...

    @abstractmethod
    async def integrate(
        self, output: SymbolicOutput, ctx: ProcessContext[dict[str, Any]]
    ) -> SymbolicInput | None:
        """
        Phase 4: Integrate - Finalize and prepare feedback.

        Return None to terminate, or new input for recursion.
        """
        ...

    # --- Utility methods ---

    def create_output(
        self,
        values: list[SymbolicValue],
        *,
        input_id: str | None = None,
        correlation_id: str | None = None,
        success: bool = True,
        error_message: str | None = None,
    ) -> SymbolicOutput:
        """Create a SymbolicOutput with this subsystem as source."""
        return SymbolicOutput(
            values=tuple(values),
            source_subsystem=self.name,
            process_id=str(ULID()),
            input_id=input_id,
            correlation_id=correlation_id,
            success=success,
            error_message=error_message,
        )
