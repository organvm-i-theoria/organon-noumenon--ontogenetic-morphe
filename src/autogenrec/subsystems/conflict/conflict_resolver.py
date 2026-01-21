"""
ConflictResolver: Manages detection and resolution of conflicting inputs or states.

Provides recursive reflection and resolution of conflicts, balancing
competing narratives or data within the system.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, auto
from typing import Any

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


class ConflictType(Enum):
    """Types of conflicts that can occur."""

    DATA = auto()  # Conflicting data values
    RULE = auto()  # Conflicting rules or policies
    STATE = auto()  # Conflicting system states
    TEMPORAL = auto()  # Temporal conflicts (ordering, timing)
    SEMANTIC = auto()  # Semantic/meaning conflicts
    RESOURCE = auto()  # Resource contention
    PRIORITY = auto()  # Priority conflicts
    IDENTITY = auto()  # Identity/ownership conflicts


class ConflictSeverity(Enum):
    """Severity levels for conflicts."""

    LOW = 1  # Minor conflict, can be auto-resolved
    MEDIUM = 2  # Moderate conflict, may need attention
    HIGH = 3  # Significant conflict, needs resolution
    CRITICAL = 4  # Critical conflict, blocking


class ResolutionStrategy(Enum):
    """Strategies for resolving conflicts."""

    PRIORITY = auto()  # Higher priority wins
    TIMESTAMP = auto()  # Most recent wins
    CONSENSUS = auto()  # Merge/consensus approach
    ARBITRATION = auto()  # Send to arbitration
    MANUAL = auto()  # Requires manual resolution
    DEFER = auto()  # Defer resolution
    OVERRIDE = auto()  # Override with specific value


class ConflictStatus(Enum):
    """Status of a conflict."""

    DETECTED = auto()  # Conflict has been detected
    ANALYZING = auto()  # Under analysis
    RESOLVING = auto()  # Being resolved
    RESOLVED = auto()  # Resolution complete
    ESCALATED = auto()  # Escalated to arbitration
    DEFERRED = auto()  # Resolution deferred
    FAILED = auto()  # Resolution failed


class ConflictingValue(BaseModel):
    """A value involved in a conflict."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    value: Any
    source: str  # Source subsystem or identifier
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    priority: int = 0
    tags: frozenset[str] = Field(default_factory=frozenset)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Conflict(BaseModel):
    """A detected conflict between values or states."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    conflict_type: ConflictType
    severity: ConflictSeverity = ConflictSeverity.MEDIUM
    status: ConflictStatus = ConflictStatus.DETECTED

    # Conflicting parties
    values: tuple[ConflictingValue, ...] = Field(default_factory=tuple)
    field_name: str = ""  # Field/attribute in conflict

    # Analysis
    description: str = ""
    contradiction_points: tuple[str, ...] = Field(default_factory=tuple)

    # Resolution
    preferred_strategy: ResolutionStrategy = ResolutionStrategy.PRIORITY
    resolution: Any | None = None
    resolution_reason: str = ""

    # Metadata
    detected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    resolved_at: datetime | None = None
    tags: frozenset[str] = Field(default_factory=frozenset)


@dataclass
class ResolutionResult:
    """Result of attempting to resolve a conflict."""

    conflict_id: str
    success: bool
    strategy_used: ResolutionStrategy
    resolved_value: Any = None
    reason: str = ""
    escalated: bool = False
    resolved_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ConflictStats:
    """Statistics about conflict resolution."""

    total_detected: int
    total_resolved: int
    total_escalated: int
    total_failed: int
    resolution_rate: float
    by_type: dict[str, int]
    by_severity: dict[str, int]


class ConflictDetector:
    """Detects conflicts in incoming data."""

    def __init__(self) -> None:
        self._log = logger.bind(component="conflict_detector")

    def detect(
        self,
        values: list[SymbolicValue],
        existing_values: dict[str, Any] | None = None,
    ) -> list[Conflict]:
        """Detect conflicts among values or with existing state."""
        conflicts: list[Conflict] = []
        existing_values = existing_values or {}

        # Check for conflicts among incoming values
        for i, v1 in enumerate(values):
            for v2 in values[i + 1:]:
                conflict = self._check_value_conflict(v1, v2)
                if conflict:
                    conflicts.append(conflict)

        # Check for conflicts with existing values
        for value in values:
            for key, existing in existing_values.items():
                conflict = self._check_existing_conflict(value, key, existing)
                if conflict:
                    conflicts.append(conflict)

        self._log.debug("conflicts_detected", count=len(conflicts))
        return conflicts

    def _check_value_conflict(
        self,
        v1: SymbolicValue,
        v2: SymbolicValue,
    ) -> Conflict | None:
        """Check if two values conflict."""
        # Same type with different content might conflict
        if v1.type != v2.type:
            return None

        # Same source might indicate duplicate/conflict
        if v1.source_subsystem == v2.source_subsystem:
            if v1.content != v2.content:
                return Conflict(
                    conflict_type=ConflictType.DATA,
                    severity=ConflictSeverity.MEDIUM,
                    values=(
                        ConflictingValue(
                            value=v1.content,
                            source=v1.source_subsystem or "unknown",
                            priority=0,
                            tags=v1.tags,
                        ),
                        ConflictingValue(
                            value=v2.content,
                            source=v2.source_subsystem or "unknown",
                            priority=0,
                            tags=v2.tags,
                        ),
                    ),
                    description=f"Conflicting values from same source: {v1.source_subsystem}",
                    contradiction_points=(
                        f"Value 1: {str(v1.content)[:100]}",
                        f"Value 2: {str(v2.content)[:100]}",
                    ),
                )

        # Check for semantic conflicts based on meaning
        if v1.meaning and v2.meaning:
            # Simple heuristic: same meaning with different content
            if v1.meaning == v2.meaning and v1.content != v2.content:
                return Conflict(
                    conflict_type=ConflictType.SEMANTIC,
                    severity=ConflictSeverity.MEDIUM,
                    values=(
                        ConflictingValue(
                            value=v1.content,
                            source=v1.source_subsystem or "unknown",
                            tags=v1.tags,
                        ),
                        ConflictingValue(
                            value=v2.content,
                            source=v2.source_subsystem or "unknown",
                            tags=v2.tags,
                        ),
                    ),
                    description=f"Semantic conflict: same meaning '{v1.meaning}' with different content",
                )

        return None

    def _check_existing_conflict(
        self,
        value: SymbolicValue,
        key: str,
        existing: Any,
    ) -> Conflict | None:
        """Check if a value conflicts with existing state."""
        # Simple check: same key with different value
        if isinstance(value.content, dict) and key in value.content:
            new_value = value.content[key]
            if new_value != existing:
                return Conflict(
                    conflict_type=ConflictType.STATE,
                    severity=ConflictSeverity.MEDIUM,
                    values=(
                        ConflictingValue(
                            value=existing,
                            source="existing_state",
                        ),
                        ConflictingValue(
                            value=new_value,
                            source=value.source_subsystem or "incoming",
                        ),
                    ),
                    field_name=key,
                    description=f"State conflict for field '{key}'",
                )

        return None


class ResolutionEngine:
    """Engine for resolving conflicts."""

    def __init__(self) -> None:
        self._log = logger.bind(component="resolution_engine")

    def resolve(
        self,
        conflict: Conflict,
        strategy: ResolutionStrategy | None = None,
    ) -> ResolutionResult:
        """Resolve a conflict using the specified or default strategy."""
        strategy = strategy or conflict.preferred_strategy

        self._log.debug(
            "resolving_conflict",
            conflict_id=conflict.id,
            strategy=strategy.name,
        )

        if strategy == ResolutionStrategy.PRIORITY:
            return self._resolve_by_priority(conflict)
        elif strategy == ResolutionStrategy.TIMESTAMP:
            return self._resolve_by_timestamp(conflict)
        elif strategy == ResolutionStrategy.CONSENSUS:
            return self._resolve_by_consensus(conflict)
        elif strategy == ResolutionStrategy.ARBITRATION:
            return self._escalate_to_arbitration(conflict)
        elif strategy == ResolutionStrategy.DEFER:
            return self._defer_resolution(conflict)
        else:
            return self._resolve_by_priority(conflict)

    def _resolve_by_priority(self, conflict: Conflict) -> ResolutionResult:
        """Resolve by selecting highest priority value."""
        if not conflict.values:
            return ResolutionResult(
                conflict_id=conflict.id,
                success=False,
                strategy_used=ResolutionStrategy.PRIORITY,
                reason="No values to resolve",
            )

        # Find highest priority
        winner = max(conflict.values, key=lambda v: v.priority)

        return ResolutionResult(
            conflict_id=conflict.id,
            success=True,
            strategy_used=ResolutionStrategy.PRIORITY,
            resolved_value=winner.value,
            reason=f"Selected value from {winner.source} with priority {winner.priority}",
        )

    def _resolve_by_timestamp(self, conflict: Conflict) -> ResolutionResult:
        """Resolve by selecting most recent value."""
        if not conflict.values:
            return ResolutionResult(
                conflict_id=conflict.id,
                success=False,
                strategy_used=ResolutionStrategy.TIMESTAMP,
                reason="No values to resolve",
            )

        # Find most recent
        winner = max(conflict.values, key=lambda v: v.timestamp)

        return ResolutionResult(
            conflict_id=conflict.id,
            success=True,
            strategy_used=ResolutionStrategy.TIMESTAMP,
            resolved_value=winner.value,
            reason=f"Selected most recent value from {winner.source}",
        )

    def _resolve_by_consensus(self, conflict: Conflict) -> ResolutionResult:
        """Attempt to merge/consensus values."""
        if not conflict.values:
            return ResolutionResult(
                conflict_id=conflict.id,
                success=False,
                strategy_used=ResolutionStrategy.CONSENSUS,
                reason="No values to resolve",
            )

        # Try to merge if values are dicts
        merged = {}
        all_dicts = all(isinstance(v.value, dict) for v in conflict.values)

        if all_dicts:
            for v in conflict.values:
                merged.update(v.value)

            return ResolutionResult(
                conflict_id=conflict.id,
                success=True,
                strategy_used=ResolutionStrategy.CONSENSUS,
                resolved_value=merged,
                reason="Merged conflicting dictionaries",
            )

        # Fall back to priority for non-dict types
        return self._resolve_by_priority(conflict)

    def _escalate_to_arbitration(self, conflict: Conflict) -> ResolutionResult:
        """Escalate to arbitration engine."""
        return ResolutionResult(
            conflict_id=conflict.id,
            success=True,
            strategy_used=ResolutionStrategy.ARBITRATION,
            escalated=True,
            reason="Escalated to arbitration engine",
        )

    def _defer_resolution(self, conflict: Conflict) -> ResolutionResult:
        """Defer resolution for later."""
        return ResolutionResult(
            conflict_id=conflict.id,
            success=True,
            strategy_used=ResolutionStrategy.DEFER,
            reason="Resolution deferred",
        )


class ConflictResolver(Subsystem):
    """
    Manages detection and resolution of conflicting inputs or states.

    Process Loop:
    1. Intake: Receive conflicting data
    2. Mirror: Reflect and duplicate inputs to expose contradictions
    3. Analyze: Identify points of conflict
    4. Resolve: Stabilize outcomes or mediate compromises
    """

    def __init__(self) -> None:
        metadata = SubsystemMetadata(
            name="conflict_resolver",
            display_name="Conflict Resolver",
            description="Manages detection and resolution of conflicting inputs or states",
            type=SubsystemType.CONFLICT,
            tags=frozenset(["conflict", "resolution", "reflection", "mediation"]),
            input_types=frozenset(["SCHEMA", "RULE", "REFERENCE"]),
            output_types=frozenset(["SCHEMA", "REFERENCE"]),
            subscribed_topics=frozenset([
                str(SubsystemTopics.ALL),
                "conflict.#",
                "dispute.#",
            ]),
            published_topics=frozenset([
                "conflict.detected",
                "conflict.resolved",
                "conflict.escalated",
            ]),
        )
        super().__init__(metadata)

        self._detector = ConflictDetector()
        self._resolver = ResolutionEngine()
        self._conflicts: dict[str, Conflict] = {}
        self._existing_state: dict[str, Any] = {}

        # Statistics
        self._total_detected = 0
        self._total_resolved = 0
        self._total_escalated = 0
        self._total_failed = 0

    @property
    def conflict_count(self) -> int:
        return len(self._conflicts)

    async def intake(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> SymbolicInput:
        """Phase 1: Receive potentially conflicting data."""
        if not input_data.values:
            self._log.debug("empty_input")
            return input_data

        self._log.debug("intake_complete", value_count=len(input_data.values))
        return input_data

    async def process(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> list[tuple[Conflict, ResolutionResult | None]]:
        """Phase 2 & 3: Detect and analyze conflicts."""
        results: list[tuple[Conflict, ResolutionResult | None]] = []

        # Detect conflicts
        conflicts = self._detector.detect(
            list(input_data.values),
            self._existing_state,
        )

        for conflict in conflicts:
            self._conflicts[conflict.id] = conflict
            self._total_detected += 1

            # Determine if auto-resolution is appropriate
            if conflict.severity in (ConflictSeverity.LOW, ConflictSeverity.MEDIUM):
                resolution = self._resolver.resolve(conflict)
                if resolution.success and not resolution.escalated:
                    self._total_resolved += 1
                elif resolution.escalated:
                    self._total_escalated += 1
                else:
                    self._total_failed += 1
                results.append((conflict, resolution))
            else:
                # High/Critical conflicts may need escalation
                results.append((conflict, None))

        return results

    async def evaluate(
        self, intermediate: list[tuple[Conflict, ResolutionResult | None]],
        ctx: ProcessContext[dict[str, Any]],
    ) -> tuple[SymbolicOutput, bool]:
        """Phase 4: Create output with resolution results."""
        values: list[SymbolicValue] = []

        for conflict, resolution in intermediate:
            if resolution and resolution.success:
                value = SymbolicValue(
                    type=SymbolicValueType.SCHEMA,
                    content={
                        "conflict_id": conflict.id,
                        "conflict_type": conflict.conflict_type.name,
                        "severity": conflict.severity.name,
                        "status": "resolved" if not resolution.escalated else "escalated",
                        "strategy": resolution.strategy_used.name,
                        "resolved_value": resolution.resolved_value,
                        "reason": resolution.reason,
                    },
                    source_subsystem=self.name,
                    tags=frozenset(["conflict", "resolution", conflict.conflict_type.name.lower()]),
                    meaning=f"Conflict resolved: {conflict.description[:50]}",
                    confidence=1.0 if resolution.success else 0.5,
                )
            else:
                value = SymbolicValue(
                    type=SymbolicValueType.SCHEMA,
                    content={
                        "conflict_id": conflict.id,
                        "conflict_type": conflict.conflict_type.name,
                        "severity": conflict.severity.name,
                        "status": "pending",
                        "description": conflict.description,
                        "contradiction_points": list(conflict.contradiction_points),
                    },
                    source_subsystem=self.name,
                    tags=frozenset(["conflict", "pending", conflict.severity.name.lower()]),
                    meaning=f"Conflict pending: {conflict.description[:50]}",
                    confidence=0.3,
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
        """Phase 4: Emit conflict events."""
        if self._message_bus and output.values:
            for value in output.values:
                status = value.content.get("status")
                if status == "resolved":
                    await self.emit_event(
                        "conflict.resolved",
                        {
                            "conflict_id": value.content.get("conflict_id"),
                            "strategy": value.content.get("strategy"),
                        },
                    )
                elif status == "escalated":
                    await self.emit_event(
                        "conflict.escalated",
                        {
                            "conflict_id": value.content.get("conflict_id"),
                        },
                    )
                else:
                    await self.emit_event(
                        "conflict.detected",
                        {
                            "conflict_id": value.content.get("conflict_id"),
                            "severity": value.content.get("severity"),
                        },
                    )

        return None

    # --- Message handlers ---

    async def handle_event(self, message: Message) -> None:
        """Handle incoming events."""
        if message.topic.startswith("conflict."):
            self._log.debug("conflict_event_received", topic=message.topic)

    async def handle_signal(self, signal: Any) -> None:
        """Handle incoming signals."""
        self._log.debug("signal_received", signal_id=getattr(signal, "id", "unknown"))

    # --- Public API ---

    def detect_conflicts(
        self,
        values: list[SymbolicValue],
    ) -> list[Conflict]:
        """Detect conflicts in a list of values."""
        conflicts = self._detector.detect(values, self._existing_state)
        for conflict in conflicts:
            self._conflicts[conflict.id] = conflict
            self._total_detected += 1
        return conflicts

    def resolve_conflict(
        self,
        conflict_id: str,
        strategy: ResolutionStrategy | None = None,
    ) -> ResolutionResult | None:
        """Resolve a specific conflict."""
        conflict = self._conflicts.get(conflict_id)
        if not conflict:
            return None

        result = self._resolver.resolve(conflict, strategy)

        if result.success and not result.escalated:
            self._total_resolved += 1
            # Update conflict status
            del self._conflicts[conflict_id]
        elif result.escalated:
            self._total_escalated += 1
        else:
            self._total_failed += 1

        return result

    def get_conflict(self, conflict_id: str) -> Conflict | None:
        """Get a conflict by ID."""
        return self._conflicts.get(conflict_id)

    def get_conflicts_by_type(self, conflict_type: ConflictType) -> list[Conflict]:
        """Get all conflicts of a specific type."""
        return [c for c in self._conflicts.values() if c.conflict_type == conflict_type]

    def get_conflicts_by_severity(self, severity: ConflictSeverity) -> list[Conflict]:
        """Get all conflicts of a specific severity."""
        return [c for c in self._conflicts.values() if c.severity == severity]

    def get_pending_conflicts(self) -> list[Conflict]:
        """Get all pending (unresolved) conflicts."""
        return list(self._conflicts.values())

    def set_state(self, key: str, value: Any) -> None:
        """Set a value in the existing state for conflict detection."""
        self._existing_state[key] = value

    def get_stats(self) -> ConflictStats:
        """Get conflict resolution statistics."""
        by_type: dict[str, int] = {}
        by_severity: dict[str, int] = {}

        for conflict in self._conflicts.values():
            by_type[conflict.conflict_type.name] = by_type.get(conflict.conflict_type.name, 0) + 1
            by_severity[conflict.severity.name] = by_severity.get(conflict.severity.name, 0) + 1

        total = self._total_resolved + self._total_failed + self._total_escalated
        rate = self._total_resolved / total if total > 0 else 0.0

        return ConflictStats(
            total_detected=self._total_detected,
            total_resolved=self._total_resolved,
            total_escalated=self._total_escalated,
            total_failed=self._total_failed,
            resolution_rate=rate,
            by_type=by_type,
            by_severity=by_severity,
        )

    def clear(self) -> int:
        """Clear all conflicts. Returns count cleared."""
        count = len(self._conflicts)
        self._conflicts.clear()
        return count
