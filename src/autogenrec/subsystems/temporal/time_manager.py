"""
TimeManager: Governs time functions and recursive scheduling.

Manages time functions and cycle tracking, ensuring recursive events
occur in ordered sequence and time-dependent processes are synchronized.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable

import structlog
from pydantic import BaseModel, ConfigDict, Field
from ulid import ULID

from autogenrec.bus.topics import SubsystemTopics
from autogenrec.core.process import ProcessContext
from autogenrec.core.signals import Message
from autogenrec.core.subsystem import Subsystem, SubsystemMetadata, SubsystemType
from autogenrec.core.symbolic import (
    SymbolicInput,
    SymbolicOutput,
    SymbolicValue,
    SymbolicValueType,
)

logger = structlog.get_logger()


class CycleType(Enum):
    """Types of temporal cycles."""

    ONE_TIME = auto()  # Single occurrence
    RECURRING = auto()  # Repeating at interval
    CRON = auto()  # Cron-like schedule
    PHASE = auto()  # Phase-based (e.g., moon phases)
    SEASONAL = auto()  # Seasonal cycles
    RECURSIVE = auto()  # Recursive/self-referencing


class EventStatus(Enum):
    """Status of a scheduled event."""

    PENDING = auto()  # Awaiting execution
    TRIGGERED = auto()  # Has been triggered
    EXECUTED = auto()  # Successfully executed
    SKIPPED = auto()  # Was skipped
    FAILED = auto()  # Execution failed
    CANCELLED = auto()  # Cancelled


class TimeEvent(BaseModel):
    """A scheduled time event."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    name: str
    description: str = ""
    cycle_type: CycleType = CycleType.ONE_TIME
    status: EventStatus = EventStatus.PENDING

    # Timing
    scheduled_at: datetime
    interval: timedelta | None = None  # For recurring
    max_occurrences: int = 0  # 0 = unlimited

    # Execution
    callback_topic: str | None = None  # Topic to publish when triggered
    payload: dict[str, Any] = Field(default_factory=dict)

    # Tracking
    occurrences: int = 0
    last_triggered: datetime | None = None
    next_trigger: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Metadata
    priority: int = 50
    tags: frozenset[str] = Field(default_factory=frozenset)


class Cycle(BaseModel):
    """A temporal cycle definition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    name: str
    cycle_type: CycleType
    description: str = ""

    # Timing
    period: timedelta
    phase_offset: timedelta = timedelta(0)  # Offset from cycle start
    current_phase: int = 0
    total_phases: int = 1

    # State
    started_at: datetime | None = None
    iteration: int = 0

    # Events
    event_ids: frozenset[str] = Field(default_factory=frozenset)

    # Metadata
    tags: frozenset[str] = Field(default_factory=frozenset)


@dataclass
class TriggerResult:
    """Result of triggering an event."""

    event_id: str
    success: bool
    triggered_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    next_trigger: datetime | None = None
    error: str | None = None


@dataclass
class TimeStats:
    """Statistics about time management."""

    total_events: int
    pending_events: int
    executed_events: int
    total_cycles: int
    active_cycles: int
    triggers_processed: int


class EventScheduler:
    """Schedules and manages time events."""

    def __init__(self) -> None:
        self._events: dict[str, TimeEvent] = {}
        self._by_status: dict[EventStatus, set[str]] = {}
        self._log = logger.bind(component="event_scheduler")

    @property
    def event_count(self) -> int:
        return len(self._events)

    def add_event(self, event: TimeEvent) -> None:
        """Add an event to the scheduler."""
        self._events[event.id] = event
        self._by_status.setdefault(event.status, set()).add(event.id)
        self._log.debug("event_added", event_id=event.id, name=event.name)

    def get_event(self, event_id: str) -> TimeEvent | None:
        """Get an event by ID."""
        return self._events.get(event_id)

    def get_due_events(self, now: datetime | None = None) -> list[TimeEvent]:
        """Get all events due for triggering."""
        if now is None:
            now = datetime.now(UTC)

        due: list[TimeEvent] = []
        for event in self._events.values():
            if event.status != EventStatus.PENDING:
                continue
            trigger_time = event.next_trigger or event.scheduled_at
            if trigger_time <= now:
                due.append(event)

        # Sort by scheduled time and priority
        due.sort(key=lambda e: (e.scheduled_at, -e.priority))
        return due

    def trigger_event(self, event_id: str) -> TriggerResult:
        """Trigger an event."""
        event = self._events.get(event_id)
        if not event:
            return TriggerResult(
                event_id=event_id,
                success=False,
                error="Event not found",
            )

        now = datetime.now(UTC)

        # Calculate next trigger for recurring events
        next_trigger = None
        new_status = EventStatus.EXECUTED

        if event.cycle_type == CycleType.RECURRING and event.interval:
            occurrences = event.occurrences + 1
            if event.max_occurrences == 0 or occurrences < event.max_occurrences:
                next_trigger = now + event.interval
                new_status = EventStatus.PENDING

        # Update event
        updated = TimeEvent(
            id=event.id,
            name=event.name,
            description=event.description,
            cycle_type=event.cycle_type,
            status=new_status,
            scheduled_at=event.scheduled_at,
            interval=event.interval,
            max_occurrences=event.max_occurrences,
            callback_topic=event.callback_topic,
            payload=event.payload,
            occurrences=event.occurrences + 1,
            last_triggered=now,
            next_trigger=next_trigger,
            created_at=event.created_at,
            priority=event.priority,
            tags=event.tags,
        )

        # Update status index
        self._by_status[event.status].discard(event_id)
        self._by_status.setdefault(new_status, set()).add(event_id)
        self._events[event_id] = updated

        return TriggerResult(
            event_id=event_id,
            success=True,
            triggered_at=now,
            next_trigger=next_trigger,
        )

    def cancel_event(self, event_id: str) -> bool:
        """Cancel an event."""
        event = self._events.get(event_id)
        if not event:
            return False

        self._by_status[event.status].discard(event_id)
        updated = TimeEvent(
            id=event.id,
            name=event.name,
            description=event.description,
            cycle_type=event.cycle_type,
            status=EventStatus.CANCELLED,
            scheduled_at=event.scheduled_at,
            interval=event.interval,
            max_occurrences=event.max_occurrences,
            callback_topic=event.callback_topic,
            payload=event.payload,
            occurrences=event.occurrences,
            last_triggered=event.last_triggered,
            next_trigger=None,
            created_at=event.created_at,
            priority=event.priority,
            tags=event.tags,
        )
        self._events[event_id] = updated
        self._by_status.setdefault(EventStatus.CANCELLED, set()).add(event_id)
        return True

    def get_by_status(self, status: EventStatus) -> list[TimeEvent]:
        """Get events by status."""
        event_ids = self._by_status.get(status, set())
        return [self._events[eid] for eid in event_ids if eid in self._events]

    def remove_event(self, event_id: str) -> bool:
        """Remove an event."""
        if event_id not in self._events:
            return False
        event = self._events.pop(event_id)
        self._by_status[event.status].discard(event_id)
        return True


class CycleTracker:
    """Tracks temporal cycles."""

    def __init__(self) -> None:
        self._cycles: dict[str, Cycle] = {}
        self._log = logger.bind(component="cycle_tracker")

    @property
    def cycle_count(self) -> int:
        return len(self._cycles)

    def add_cycle(self, cycle: Cycle) -> None:
        """Add a cycle to track."""
        self._cycles[cycle.id] = cycle
        self._log.debug("cycle_added", cycle_id=cycle.id, name=cycle.name)

    def get_cycle(self, cycle_id: str) -> Cycle | None:
        """Get a cycle by ID."""
        return self._cycles.get(cycle_id)

    def start_cycle(self, cycle_id: str) -> Cycle | None:
        """Start a cycle."""
        cycle = self._cycles.get(cycle_id)
        if not cycle:
            return None

        updated = Cycle(
            id=cycle.id,
            name=cycle.name,
            cycle_type=cycle.cycle_type,
            description=cycle.description,
            period=cycle.period,
            phase_offset=cycle.phase_offset,
            current_phase=0,
            total_phases=cycle.total_phases,
            started_at=datetime.now(UTC),
            iteration=0,
            event_ids=cycle.event_ids,
            tags=cycle.tags,
        )
        self._cycles[cycle_id] = updated
        return updated

    def advance_cycle(self, cycle_id: str) -> Cycle | None:
        """Advance a cycle to the next phase."""
        cycle = self._cycles.get(cycle_id)
        if not cycle or not cycle.started_at:
            return None

        new_phase = (cycle.current_phase + 1) % cycle.total_phases
        new_iteration = cycle.iteration + (1 if new_phase == 0 else 0)

        updated = Cycle(
            id=cycle.id,
            name=cycle.name,
            cycle_type=cycle.cycle_type,
            description=cycle.description,
            period=cycle.period,
            phase_offset=cycle.phase_offset,
            current_phase=new_phase,
            total_phases=cycle.total_phases,
            started_at=cycle.started_at,
            iteration=new_iteration,
            event_ids=cycle.event_ids,
            tags=cycle.tags,
        )
        self._cycles[cycle_id] = updated
        return updated

    def get_active_cycles(self) -> list[Cycle]:
        """Get all active cycles."""
        return [c for c in self._cycles.values() if c.started_at is not None]

    def remove_cycle(self, cycle_id: str) -> bool:
        """Remove a cycle."""
        return self._cycles.pop(cycle_id, None) is not None


class TimeManager(Subsystem):
    """
    Governs time functions and recursive scheduling.

    Process Loop:
    1. Register: Receive events with associated times or cycles
    2. Assign: Map events to appropriate cycles or schedules
    3. Monitor: Track event timing and trigger events as needed
    4. Adjust: Modify schedules to account for drift or changes
    """

    def __init__(self) -> None:
        metadata = SubsystemMetadata(
            name="time_manager",
            display_name="Time Manager",
            description="Governs time functions and recursive scheduling",
            type=SubsystemType.TEMPORAL,
            tags=frozenset(["time", "scheduling", "cycles", "events"]),
            input_types=frozenset(["TIMESTAMP", "SCHEDULE", "SCHEMA"]),
            output_types=frozenset(["TIMESTAMP", "SIGNAL"]),
            subscribed_topics=frozenset([
                str(SubsystemTopics.ALL),
                "time.#",
                "schedule.#",
            ]),
            published_topics=frozenset([
                "time.event.triggered",
                "time.event.scheduled",
                "time.cycle.advanced",
            ]),
        )
        super().__init__(metadata)

        self._scheduler = EventScheduler()
        self._tracker = CycleTracker()
        self._triggers_processed = 0

    @property
    def event_count(self) -> int:
        return self._scheduler.event_count

    @property
    def cycle_count(self) -> int:
        return self._tracker.cycle_count

    async def intake(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> SymbolicInput:
        """Phase 1: Register events with times or cycles."""
        if not input_data.values:
            self._log.debug("empty_input")
            return input_data

        self._log.debug("intake_complete", value_count=len(input_data.values))
        return input_data

    async def process(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> list[TimeEvent | Cycle]:
        """Phase 2: Assign events to schedules and process triggers."""
        results: list[TimeEvent | Cycle] = []

        for value in input_data.values:
            if value.type == SymbolicValueType.SCHEDULE:
                cycle = self._parse_cycle(value)
                if cycle:
                    self._tracker.add_cycle(cycle)
                    results.append(cycle)
            else:
                event = self._parse_event(value)
                if event:
                    self._scheduler.add_event(event)
                    results.append(event)

        return results

    async def evaluate(
        self, intermediate: list[TimeEvent | Cycle],
        ctx: ProcessContext[dict[str, Any]],
    ) -> tuple[SymbolicOutput, bool]:
        """Phase 3: Create output with scheduled items."""
        values: list[SymbolicValue] = []

        for item in intermediate:
            if isinstance(item, TimeEvent):
                value = SymbolicValue(
                    type=SymbolicValueType.TIMESTAMP,
                    content={
                        "event_id": item.id,
                        "name": item.name,
                        "scheduled_at": item.scheduled_at.isoformat(),
                        "cycle_type": item.cycle_type.name,
                        "status": item.status.name,
                    },
                    source_subsystem=self.name,
                    tags=item.tags | frozenset(["event", item.cycle_type.name.lower()]),
                    meaning=f"Scheduled: {item.name}",
                    confidence=1.0,
                )
            else:  # Cycle
                value = SymbolicValue(
                    type=SymbolicValueType.SCHEDULE,
                    content={
                        "cycle_id": item.id,
                        "name": item.name,
                        "cycle_type": item.cycle_type.name,
                        "period_seconds": item.period.total_seconds(),
                        "current_phase": item.current_phase,
                        "total_phases": item.total_phases,
                    },
                    source_subsystem=self.name,
                    tags=item.tags | frozenset(["cycle", item.cycle_type.name.lower()]),
                    meaning=f"Cycle: {item.name}",
                    confidence=1.0,
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
        """Phase 4: Emit events for scheduled items."""
        if self._message_bus and output.values:
            for value in output.values:
                if value.type == SymbolicValueType.TIMESTAMP:
                    await self.emit_event(
                        "time.event.scheduled",
                        {
                            "event_id": value.content.get("event_id"),
                            "name": value.content.get("name"),
                        },
                    )
                else:
                    await self.emit_event(
                        "time.cycle.created",
                        {
                            "cycle_id": value.content.get("cycle_id"),
                            "name": value.content.get("name"),
                        },
                    )

        return None

    def _parse_event(self, value: SymbolicValue) -> TimeEvent | None:
        """Parse a TimeEvent from a SymbolicValue."""
        content = value.content
        if not isinstance(content, dict):
            return None

        try:
            scheduled_at_str = content.get("scheduled_at")
            if scheduled_at_str:
                scheduled_at = datetime.fromisoformat(scheduled_at_str.replace("Z", "+00:00"))
            else:
                scheduled_at = datetime.now(UTC) + timedelta(seconds=content.get("delay_seconds", 0))

            cycle_type_str = content.get("cycle_type", "ONE_TIME")
            try:
                cycle_type = CycleType[cycle_type_str.upper()]
            except KeyError:
                cycle_type = CycleType.ONE_TIME

            interval = None
            if interval_seconds := content.get("interval_seconds"):
                interval = timedelta(seconds=interval_seconds)

            return TimeEvent(
                id=content.get("id", str(ULID())),
                name=content.get("name", f"event_{value.id[:8]}"),
                description=content.get("description", ""),
                cycle_type=cycle_type,
                scheduled_at=scheduled_at,
                interval=interval,
                max_occurrences=content.get("max_occurrences", 0),
                callback_topic=content.get("callback_topic"),
                payload=content.get("payload", {}),
                priority=content.get("priority", 50),
                tags=frozenset(content.get("tags", [])) | value.tags,
            )
        except Exception as e:
            self._log.warning("event_parse_failed", value_id=value.id, error=str(e))
            return None

    def _parse_cycle(self, value: SymbolicValue) -> Cycle | None:
        """Parse a Cycle from a SymbolicValue."""
        content = value.content
        if not isinstance(content, dict):
            return None

        try:
            cycle_type_str = content.get("cycle_type", "RECURRING")
            try:
                cycle_type = CycleType[cycle_type_str.upper()]
            except KeyError:
                cycle_type = CycleType.RECURRING

            period_seconds = content.get("period_seconds", 3600)
            period = timedelta(seconds=period_seconds)

            return Cycle(
                id=content.get("id", str(ULID())),
                name=content.get("name", f"cycle_{value.id[:8]}"),
                cycle_type=cycle_type,
                description=content.get("description", ""),
                period=period,
                total_phases=content.get("total_phases", 1),
                tags=frozenset(content.get("tags", [])) | value.tags,
            )
        except Exception as e:
            self._log.warning("cycle_parse_failed", value_id=value.id, error=str(e))
            return None

    # --- Message handlers ---

    async def handle_event(self, message: Message) -> None:
        """Handle incoming events."""
        if message.topic.startswith("time."):
            self._log.debug("time_event_received", topic=message.topic)

    async def handle_signal(self, signal: Any) -> None:
        """Handle incoming signals."""
        self._log.debug("signal_received", signal_id=getattr(signal, "id", "unknown"))

    # --- Public API ---

    def schedule_event(
        self,
        name: str,
        scheduled_at: datetime,
        cycle_type: CycleType = CycleType.ONE_TIME,
        **kwargs: Any,
    ) -> TimeEvent:
        """Schedule a new event."""
        interval = None
        if interval_seconds := kwargs.get("interval_seconds"):
            interval = timedelta(seconds=interval_seconds)

        event = TimeEvent(
            name=name,
            scheduled_at=scheduled_at,
            cycle_type=cycle_type,
            description=kwargs.get("description", ""),
            interval=interval,
            max_occurrences=kwargs.get("max_occurrences", 0),
            callback_topic=kwargs.get("callback_topic"),
            payload=kwargs.get("payload", {}),
            priority=kwargs.get("priority", 50),
            tags=frozenset(kwargs.get("tags", [])),
            next_trigger=scheduled_at,
        )
        self._scheduler.add_event(event)
        return event

    def schedule_recurring(
        self,
        name: str,
        interval_seconds: int,
        **kwargs: Any,
    ) -> TimeEvent:
        """Schedule a recurring event."""
        return self.schedule_event(
            name=name,
            scheduled_at=datetime.now(UTC) + timedelta(seconds=interval_seconds),
            cycle_type=CycleType.RECURRING,
            interval_seconds=interval_seconds,
            **kwargs,
        )

    def create_cycle(
        self,
        name: str,
        period_seconds: int,
        cycle_type: CycleType = CycleType.RECURRING,
        **kwargs: Any,
    ) -> Cycle:
        """Create a new cycle."""
        cycle = Cycle(
            name=name,
            cycle_type=cycle_type,
            description=kwargs.get("description", ""),
            period=timedelta(seconds=period_seconds),
            total_phases=kwargs.get("total_phases", 1),
            tags=frozenset(kwargs.get("tags", [])),
        )
        self._tracker.add_cycle(cycle)
        return cycle

    def get_event(self, event_id: str) -> TimeEvent | None:
        """Get an event by ID."""
        return self._scheduler.get_event(event_id)

    def get_cycle(self, cycle_id: str) -> Cycle | None:
        """Get a cycle by ID."""
        return self._tracker.get_cycle(cycle_id)

    def get_due_events(self) -> list[TimeEvent]:
        """Get all events due for triggering."""
        return self._scheduler.get_due_events()

    async def trigger_due_events(self) -> list[TriggerResult]:
        """Trigger all due events."""
        due = self._scheduler.get_due_events()
        results: list[TriggerResult] = []

        for event in due:
            result = self._scheduler.trigger_event(event.id)
            results.append(result)
            self._triggers_processed += 1

            if result.success and self._message_bus and event.callback_topic:
                await self.emit_event(
                    event.callback_topic,
                    {
                        "event_id": event.id,
                        "name": event.name,
                        "payload": event.payload,
                        "triggered_at": result.triggered_at.isoformat(),
                    },
                )

        return results

    def cancel_event(self, event_id: str) -> bool:
        """Cancel an event."""
        return self._scheduler.cancel_event(event_id)

    def start_cycle(self, cycle_id: str) -> Cycle | None:
        """Start a cycle."""
        return self._tracker.start_cycle(cycle_id)

    def advance_cycle(self, cycle_id: str) -> Cycle | None:
        """Advance a cycle to the next phase."""
        return self._tracker.advance_cycle(cycle_id)

    def get_active_cycles(self) -> list[Cycle]:
        """Get all active cycles."""
        return self._tracker.get_active_cycles()

    def get_stats(self) -> TimeStats:
        """Get time management statistics."""
        pending = len(self._scheduler.get_by_status(EventStatus.PENDING))
        executed = len(self._scheduler.get_by_status(EventStatus.EXECUTED))
        active_cycles = len(self._tracker.get_active_cycles())

        return TimeStats(
            total_events=self._scheduler.event_count,
            pending_events=pending,
            executed_events=executed,
            total_cycles=self._tracker.cycle_count,
            active_cycles=active_cycles,
            triggers_processed=self._triggers_processed,
        )

    def clear(self) -> tuple[int, int]:
        """Clear all events and cycles. Returns (events, cycles) cleared."""
        events = self._scheduler.event_count
        cycles = self._tracker.cycle_count
        self._scheduler = EventScheduler()
        self._tracker = CycleTracker()
        return events, cycles
