"""
EchoHandler: Manages the processing and replay of signals and echoes.

Records, repeats, and re-contextualizes signals to maintain continuity
across recursive cycles.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum, auto
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field
from ulid import ULID

from autogenrec.bus.topics import SubsystemTopics
from autogenrec.core.process import ProcessContext
from autogenrec.core.signals import Echo, Message, Signal, SignalDomain
from autogenrec.core.subsystem import Subsystem, SubsystemMetadata, SubsystemType
from autogenrec.core.symbolic import (
    SymbolicInput,
    SymbolicOutput,
    SymbolicValue,
    SymbolicValueType,
)

logger = structlog.get_logger()


class EchoState(Enum):
    """State of an echo in the system."""

    CAPTURED = auto()  # Initially captured
    SCHEDULED = auto()  # Scheduled for replay
    REPLAYING = auto()  # Currently being replayed
    REPLAYED = auto()  # Replay completed
    DECAYED = auto()  # Signal too weak for replay
    EXPIRED = auto()  # Past replay window
    CANCELLED = auto()  # Replay cancelled


class ReplayStrategy(Enum):
    """Strategies for replaying echoes."""

    IMMEDIATE = auto()  # Replay immediately
    DELAYED = auto()  # Replay after a delay
    SCHEDULED = auto()  # Replay at specific time
    TRIGGERED = auto()  # Replay on external trigger
    CYCLIC = auto()  # Replay periodically


class CapturedSignal(BaseModel):
    """A captured signal awaiting potential replay."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    signal: Signal
    topic: str
    state: EchoState = EchoState.CAPTURED

    # Capture metadata
    captured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    captured_by: str = ""  # Subsystem that captured it

    # Replay configuration
    replay_strategy: ReplayStrategy = ReplayStrategy.IMMEDIATE
    replay_at: datetime | None = None
    replay_count: int = 0
    max_replays: int = 3
    replay_interval_seconds: int = 0  # For cyclic replay

    # Decay tracking
    decay_factor: float = Field(default=0.9, ge=0.0, le=1.0)
    min_strength: float = Field(default=0.1, ge=0.0, le=1.0)

    @property
    def effective_strength(self) -> float:
        """Calculate signal strength after decay."""
        return self.signal.strength * (self.decay_factor ** self.replay_count)

    @property
    def is_viable(self) -> bool:
        """Check if signal still has sufficient strength."""
        return self.effective_strength >= self.min_strength

    @property
    def can_replay(self) -> bool:
        """Check if signal can be replayed."""
        return (
            self.state in (EchoState.CAPTURED, EchoState.SCHEDULED, EchoState.REPLAYED)
            and self.replay_count < self.max_replays
            and self.is_viable
        )


class ReplaySchedule(BaseModel):
    """Schedule for replaying an echo."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    captured_signal_id: str
    scheduled_at: datetime
    target_topic: str | None = None  # Override original topic
    target_subsystem: str | None = None  # Override target
    priority_boost: int = 0  # Adjust priority for replay
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ReplayResult:
    """Result of a replay operation."""

    captured_signal_id: str
    success: bool
    echo: Echo | None = None
    error: str | None = None
    replayed_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class EchoStats:
    """Statistics about echo handling."""

    total_captured: int
    total_replayed: int
    total_decayed: int
    total_expired: int
    pending_replays: int
    scheduled_replays: int
    average_decay: float


class SignalBuffer:
    """Buffer for captured signals."""

    def __init__(self, max_size: int = 1000) -> None:
        self._signals: dict[str, CapturedSignal] = {}
        self._by_state: dict[EchoState, set[str]] = {}
        self._by_topic: dict[str, set[str]] = {}
        self._by_source: dict[str, set[str]] = {}
        self._max_size = max_size
        self._log = logger.bind(component="signal_buffer")

    @property
    def size(self) -> int:
        return len(self._signals)

    def add(self, captured: CapturedSignal) -> None:
        """Add a captured signal to the buffer."""
        # Evict oldest if at capacity
        if len(self._signals) >= self._max_size:
            self._evict_oldest()

        self._signals[captured.id] = captured

        # Index
        self._by_state.setdefault(captured.state, set()).add(captured.id)
        self._by_topic.setdefault(captured.topic, set()).add(captured.id)
        self._by_source.setdefault(captured.signal.source, set()).add(captured.id)

        self._log.debug(
            "signal_captured",
            signal_id=captured.id,
            topic=captured.topic,
            strength=captured.signal.strength,
        )

    def get(self, signal_id: str) -> CapturedSignal | None:
        """Get a captured signal by ID."""
        return self._signals.get(signal_id)

    def update_state(self, signal_id: str, new_state: EchoState) -> bool:
        """Update the state of a captured signal."""
        captured = self._signals.get(signal_id)
        if not captured:
            return False

        # Remove from old state index
        old_state = captured.state
        if old_state in self._by_state:
            self._by_state[old_state].discard(signal_id)

        # Create new captured signal with updated state
        new_captured = CapturedSignal(
            id=captured.id,
            signal=captured.signal,
            topic=captured.topic,
            state=new_state,
            captured_at=captured.captured_at,
            captured_by=captured.captured_by,
            replay_strategy=captured.replay_strategy,
            replay_at=captured.replay_at,
            replay_count=captured.replay_count,
            max_replays=captured.max_replays,
            replay_interval_seconds=captured.replay_interval_seconds,
            decay_factor=captured.decay_factor,
            min_strength=captured.min_strength,
        )

        self._signals[signal_id] = new_captured
        self._by_state.setdefault(new_state, set()).add(signal_id)

        return True

    def increment_replay_count(self, signal_id: str) -> bool:
        """Increment the replay count of a captured signal."""
        captured = self._signals.get(signal_id)
        if not captured:
            return False

        new_captured = CapturedSignal(
            id=captured.id,
            signal=captured.signal,
            topic=captured.topic,
            state=captured.state,
            captured_at=captured.captured_at,
            captured_by=captured.captured_by,
            replay_strategy=captured.replay_strategy,
            replay_at=captured.replay_at,
            replay_count=captured.replay_count + 1,
            max_replays=captured.max_replays,
            replay_interval_seconds=captured.replay_interval_seconds,
            decay_factor=captured.decay_factor,
            min_strength=captured.min_strength,
        )

        self._signals[signal_id] = new_captured
        return True

    def get_by_state(self, state: EchoState) -> list[CapturedSignal]:
        """Get all captured signals in a given state."""
        signal_ids = self._by_state.get(state, set())
        return [self._signals[sid] for sid in signal_ids if sid in self._signals]

    def get_by_topic(self, topic: str) -> list[CapturedSignal]:
        """Get all captured signals for a topic."""
        signal_ids = self._by_topic.get(topic, set())
        return [self._signals[sid] for sid in signal_ids if sid in self._signals]

    def get_viable_for_replay(self) -> list[CapturedSignal]:
        """Get all signals viable for replay."""
        viable: list[CapturedSignal] = []
        for captured in self._signals.values():
            if captured.can_replay:
                viable.append(captured)
        return viable

    def remove(self, signal_id: str) -> bool:
        """Remove a captured signal from the buffer."""
        if signal_id not in self._signals:
            return False

        captured = self._signals.pop(signal_id)

        # Remove from indices
        if captured.state in self._by_state:
            self._by_state[captured.state].discard(signal_id)
        if captured.topic in self._by_topic:
            self._by_topic[captured.topic].discard(signal_id)
        if captured.signal.source in self._by_source:
            self._by_source[captured.signal.source].discard(signal_id)

        return True

    def _evict_oldest(self) -> None:
        """Evict the oldest signal from the buffer."""
        if not self._signals:
            return

        oldest_id = min(
            self._signals.keys(),
            key=lambda sid: self._signals[sid].captured_at,
        )
        self.remove(oldest_id)
        self._log.debug("signal_evicted", signal_id=oldest_id)


class ReplayScheduler:
    """Schedules echoes for replay."""

    def __init__(self) -> None:
        self._schedules: dict[str, ReplaySchedule] = {}
        self._by_time: dict[datetime, set[str]] = {}  # Bucketed by minute
        self._log = logger.bind(component="replay_scheduler")

    def schedule(
        self,
        captured_signal_id: str,
        replay_at: datetime,
        target_topic: str | None = None,
        target_subsystem: str | None = None,
        priority_boost: int = 0,
    ) -> ReplaySchedule:
        """Schedule a signal for replay."""
        schedule = ReplaySchedule(
            captured_signal_id=captured_signal_id,
            scheduled_at=replay_at,
            target_topic=target_topic,
            target_subsystem=target_subsystem,
            priority_boost=priority_boost,
        )

        self._schedules[schedule.id] = schedule

        # Bucket by minute
        bucket = replay_at.replace(second=0, microsecond=0)
        self._by_time.setdefault(bucket, set()).add(schedule.id)

        self._log.debug(
            "replay_scheduled",
            schedule_id=schedule.id,
            signal_id=captured_signal_id,
            at=replay_at.isoformat(),
        )

        return schedule

    def get_due(self, now: datetime | None = None) -> list[ReplaySchedule]:
        """Get all schedules that are due for replay."""
        if now is None:
            now = datetime.now(UTC)

        due: list[ReplaySchedule] = []
        for schedule in self._schedules.values():
            if schedule.scheduled_at <= now:
                due.append(schedule)

        return due

    def remove(self, schedule_id: str) -> bool:
        """Remove a schedule."""
        if schedule_id not in self._schedules:
            return False

        schedule = self._schedules.pop(schedule_id)
        bucket = schedule.scheduled_at.replace(second=0, microsecond=0)
        if bucket in self._by_time:
            self._by_time[bucket].discard(schedule_id)

        return True

    @property
    def pending_count(self) -> int:
        return len(self._schedules)


class EchoHandler(Subsystem):
    """
    Manages the processing and replay of signals and echoes.

    Process Loop:
    1. Capture: Receive incoming signals or echoes
    2. Process: Classify and transform as needed
    3. Replay: Reissue or transmit signals to appropriate nodes
    4. Store: Archive signals for future recall
    """

    def __init__(self, buffer_size: int = 1000) -> None:
        metadata = SubsystemMetadata(
            name="echo_handler",
            display_name="Echo Handler",
            description="Manages the processing and replay of signals and echoes",
            type=SubsystemType.DATA,
            tags=frozenset(["echo", "signal", "replay", "continuity"]),
            input_types=frozenset(["SIGNAL", "ECHO"]),
            output_types=frozenset(["SIGNAL", "ECHO"]),
            subscribed_topics=frozenset([
                str(SubsystemTopics.ALL),
                "signal.#",
                "echo.#",
            ]),
            published_topics=frozenset([
                "echo.captured",
                "echo.replayed",
                "echo.decayed",
                "echo.scheduled",
            ]),
        )
        super().__init__(metadata)

        self._buffer = SignalBuffer(max_size=buffer_size)
        self._scheduler = ReplayScheduler()
        self._total_replayed = 0
        self._total_decayed = 0

    @property
    def captured_count(self) -> int:
        return self._buffer.size

    @property
    def scheduled_count(self) -> int:
        return self._scheduler.pending_count

    async def intake(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> SymbolicInput:
        """Phase 1: Capture incoming signals or echoes."""
        if not input_data.values:
            self._log.debug("empty_input")
            return input_data

        # Filter to signal-related types
        supported_types = {SymbolicValueType.SIGNAL, SymbolicValueType.ECHO}
        valid_values = [v for v in input_data.values if v.type in supported_types]

        self._log.debug(
            "intake_complete",
            total=len(input_data.values),
            valid=len(valid_values),
        )

        if len(valid_values) != len(input_data.values):
            return SymbolicInput(
                values=tuple(valid_values),
                source_subsystem=input_data.source_subsystem,
                target_subsystem=input_data.target_subsystem,
                correlation_id=input_data.correlation_id,
                priority=input_data.priority,
                metadata=input_data.metadata,
            )
        return input_data

    async def process(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> list[CapturedSignal]:
        """Phase 2: Classify and transform signals."""
        captured_signals: list[CapturedSignal] = []

        for value in input_data.values:
            captured = self._capture_signal(value)
            if captured:
                self._buffer.add(captured)
                captured_signals.append(captured)

                self._log.debug(
                    "signal_processed",
                    captured_id=captured.id,
                    strength=captured.signal.strength,
                )

        return captured_signals

    async def evaluate(
        self, intermediate: list[CapturedSignal], ctx: ProcessContext[dict[str, Any]]
    ) -> tuple[SymbolicOutput, bool]:
        """Phase 3: Create output with captured signals."""
        values: list[SymbolicValue] = []

        for captured in intermediate:
            value = SymbolicValue(
                type=SymbolicValueType.ECHO,
                content={
                    "captured_id": captured.id,
                    "original_signal_id": captured.signal.id,
                    "topic": captured.topic,
                    "source": captured.signal.source,
                    "strength": captured.signal.strength,
                    "effective_strength": captured.effective_strength,
                    "is_viable": captured.is_viable,
                    "replay_count": captured.replay_count,
                    "max_replays": captured.max_replays,
                },
                source_subsystem=self.name,
                tags=frozenset(["echo", "captured", captured.signal.source]),
                meaning=f"Captured signal from {captured.signal.source}",
                confidence=captured.effective_strength,
            )
            values.append(value)

        output = self.create_output(
            values=values,
            input_id=ctx.metadata.get("input_id"),
        )

        return output, False

    async def integrate(
        self, output: SymbolicOutput, ctx: ProcessContext[dict[str, Any]]
    ) -> SymbolicInput | None:
        """Phase 4: Emit events and archive signals."""
        if self._message_bus and output.values:
            for value in output.values:
                await self.emit_event(
                    "echo.captured",
                    {
                        "captured_id": value.content.get("captured_id"),
                        "topic": value.content.get("topic"),
                        "strength": value.content.get("strength"),
                    },
                )

        return None

    def _capture_signal(self, value: SymbolicValue) -> CapturedSignal | None:
        """Create a CapturedSignal from a SymbolicValue."""
        content = value.content
        if not isinstance(content, dict):
            return None

        try:
            # Extract or create signal
            signal_data = content.get("signal", content)

            signal = Signal(
                id=signal_data.get("id", str(ULID())),
                domain=SignalDomain[signal_data.get("domain", "DIGITAL").upper()],
                payload=signal_data.get("payload", content),
                payload_type=signal_data.get("payload_type", "unknown"),
                source=signal_data.get("source", value.source_subsystem or "unknown"),
                target=signal_data.get("target"),
                correlation_id=signal_data.get("correlation_id"),
                strength=signal_data.get("strength", 1.0),
                threshold=signal_data.get("threshold", 0.5),
                tags=frozenset(signal_data.get("tags", [])) | value.tags,
                metadata={"allow_weak": True},  # Allow capture of weak signals
            )

            # Determine replay strategy
            strategy_str = content.get("replay_strategy", "IMMEDIATE")
            try:
                strategy = ReplayStrategy[strategy_str.upper()]
            except KeyError:
                strategy = ReplayStrategy.IMMEDIATE

            captured = CapturedSignal(
                signal=signal,
                topic=content.get("topic", f"signal.{signal.source}"),
                captured_by=self.name,
                replay_strategy=strategy,
                max_replays=content.get("max_replays", 3),
                decay_factor=content.get("decay_factor", 0.9),
                min_strength=content.get("min_strength", 0.1),
            )

            return captured

        except Exception as e:
            self._log.warning("capture_failed", value_id=value.id, error=str(e))
            return None

    # --- Message handlers ---

    async def handle_signal(self, signal: Signal) -> None:
        """Handle incoming signals for capture."""
        captured = CapturedSignal(
            signal=signal,
            topic=f"signal.{signal.source}",
            captured_by=self.name,
        )
        self._buffer.add(captured)
        self._log.debug("direct_signal_captured", signal_id=signal.id)

    async def handle_echo(self, echo: Echo) -> None:
        """Handle incoming echoes."""
        # Convert echo to captured signal
        captured = CapturedSignal(
            signal=echo.original_signal,
            topic=f"echo.{echo.original_signal.source}",
            captured_by=self.name,
            replay_count=echo.replay_count,
            decay_factor=echo.decay_factor,
        )
        self._buffer.add(captured)
        self._log.debug("echo_captured", echo_id=echo.id)

    async def handle_event(self, message: Message) -> None:
        """Handle incoming events."""
        self._log.debug("event_received", topic=message.topic)

    # --- Public API ---

    def capture(
        self,
        signal: Signal,
        topic: str,
        strategy: ReplayStrategy = ReplayStrategy.IMMEDIATE,
        **kwargs: Any,
    ) -> CapturedSignal:
        """Capture a signal for potential replay."""
        captured = CapturedSignal(
            signal=signal,
            topic=topic,
            captured_by=kwargs.get("captured_by", self.name),
            replay_strategy=strategy,
            max_replays=kwargs.get("max_replays", 3),
            decay_factor=kwargs.get("decay_factor", 0.9),
            min_strength=kwargs.get("min_strength", 0.1),
        )
        self._buffer.add(captured)
        return captured

    def schedule_replay(
        self,
        captured_id: str,
        delay_seconds: int = 0,
        at_time: datetime | None = None,
        **kwargs: Any,
    ) -> ReplaySchedule | None:
        """Schedule a captured signal for replay."""
        captured = self._buffer.get(captured_id)
        if not captured or not captured.can_replay:
            return None

        if at_time is None:
            at_time = datetime.now(UTC) + timedelta(seconds=delay_seconds)

        schedule = self._scheduler.schedule(
            captured_signal_id=captured_id,
            replay_at=at_time,
            target_topic=kwargs.get("target_topic"),
            target_subsystem=kwargs.get("target_subsystem"),
            priority_boost=kwargs.get("priority_boost", 0),
        )

        self._buffer.update_state(captured_id, EchoState.SCHEDULED)
        return schedule

    async def replay(self, captured_id: str) -> ReplayResult:
        """Replay a captured signal immediately."""
        captured = self._buffer.get(captured_id)
        if not captured:
            return ReplayResult(
                captured_signal_id=captured_id,
                success=False,
                error="Signal not found",
            )

        if not captured.can_replay:
            return ReplayResult(
                captured_signal_id=captured_id,
                success=False,
                error="Signal cannot be replayed (decayed or max replays reached)",
            )

        # Create echo
        echo = Echo(
            original_signal=captured.signal,
            replay_count=captured.replay_count + 1,
            decay_factor=captured.decay_factor,
            replay_history=(self.name,),
        )

        # Update state
        self._buffer.update_state(captured_id, EchoState.REPLAYING)

        # Emit echo if message bus available
        if self._message_bus:
            try:
                message = Message.from_echo(echo, captured.topic, self.name)
                await self.publish(message)

                self._buffer.increment_replay_count(captured_id)
                self._buffer.update_state(captured_id, EchoState.REPLAYED)
                self._total_replayed += 1

                await self.emit_event(
                    "echo.replayed",
                    {
                        "captured_id": captured_id,
                        "echo_id": echo.id,
                        "replay_count": echo.replay_count,
                        "effective_strength": echo.effective_strength,
                    },
                )

                return ReplayResult(
                    captured_signal_id=captured_id,
                    success=True,
                    echo=echo,
                )

            except Exception as e:
                self._buffer.update_state(captured_id, EchoState.CAPTURED)
                return ReplayResult(
                    captured_signal_id=captured_id,
                    success=False,
                    error=str(e),
                )
        else:
            # No message bus, just update state
            self._buffer.increment_replay_count(captured_id)
            self._buffer.update_state(captured_id, EchoState.REPLAYED)
            self._total_replayed += 1

            return ReplayResult(
                captured_signal_id=captured_id,
                success=True,
                echo=echo,
            )

    async def process_scheduled(self) -> list[ReplayResult]:
        """Process all scheduled replays that are due."""
        results: list[ReplayResult] = []
        due_schedules = self._scheduler.get_due()

        for schedule in due_schedules:
            result = await self.replay(schedule.captured_signal_id)
            results.append(result)
            self._scheduler.remove(schedule.id)

        return results

    def get_captured(self, captured_id: str) -> CapturedSignal | None:
        """Get a captured signal by ID."""
        return self._buffer.get(captured_id)

    def get_by_state(self, state: EchoState) -> list[CapturedSignal]:
        """Get captured signals by state."""
        return self._buffer.get_by_state(state)

    def get_by_topic(self, topic: str) -> list[CapturedSignal]:
        """Get captured signals by topic."""
        return self._buffer.get_by_topic(topic)

    def get_viable_for_replay(self) -> list[CapturedSignal]:
        """Get all signals viable for replay."""
        return self._buffer.get_viable_for_replay()

    def mark_decayed(self, captured_id: str) -> bool:
        """Mark a signal as decayed."""
        if self._buffer.update_state(captured_id, EchoState.DECAYED):
            self._total_decayed += 1
            return True
        return False

    def remove(self, captured_id: str) -> bool:
        """Remove a captured signal."""
        return self._buffer.remove(captured_id)

    def get_stats(self) -> EchoStats:
        """Get echo handling statistics."""
        viable = self._buffer.get_viable_for_replay()
        total_decay = sum(1.0 - c.effective_strength for c in viable)
        avg_decay = total_decay / len(viable) if viable else 0.0

        return EchoStats(
            total_captured=self._buffer.size,
            total_replayed=self._total_replayed,
            total_decayed=self._total_decayed,
            total_expired=len(self._buffer.get_by_state(EchoState.EXPIRED)),
            pending_replays=len(self._buffer.get_by_state(EchoState.CAPTURED)),
            scheduled_replays=self._scheduler.pending_count,
            average_decay=avg_decay,
        )

    def clear(self) -> int:
        """Clear all captured signals. Returns count cleared."""
        count = self._buffer.size
        self._buffer = SignalBuffer(max_size=self._buffer._max_size)
        self._scheduler = ReplayScheduler()
        return count
