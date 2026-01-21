"""
EvolutionScheduler: Manages timing and progression of growth and mutation cycles.

Governs symbolic growth, mutation, and renewal processes, modeling
expansion, branching, and transformation within recursive systems.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum, auto
from typing import Any
import random

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


class GrowthPhase(Enum):
    """Phases of growth in evolution cycles."""

    DORMANT = auto()  # Inactive, waiting
    GERMINATION = auto()  # Initial growth
    GROWTH = auto()  # Active expansion
    BLOOM = auto()  # Peak expression
    HARVEST = auto()  # Collection of results
    DECAY = auto()  # Winding down
    RENEWAL = auto()  # Preparing for next cycle


class MutationType(Enum):
    """Types of mutations that can occur."""

    ADDITION = auto()  # Add new elements
    REMOVAL = auto()  # Remove elements
    MODIFICATION = auto()  # Modify existing elements
    RECOMBINATION = auto()  # Combine elements in new ways
    INVERSION = auto()  # Reverse/invert elements
    DUPLICATION = auto()  # Duplicate elements


class StabilityLevel(Enum):
    """Stability levels for patterns."""

    UNSTABLE = 1  # Likely to change
    FRAGILE = 2  # May change under stress
    STABLE = 3  # Resistant to change
    ROBUST = 4  # Very resistant to change
    FIXED = 5  # Will not change


class MutationPath(BaseModel):
    """A potential mutation path for a pattern."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    mutation_type: MutationType
    description: str = ""
    probability: float = Field(default=0.5, ge=0.0, le=1.0)
    impact: float = Field(default=0.5, ge=0.0, le=1.0)  # How much change
    required_phase: GrowthPhase | None = None


class GrowthPattern(BaseModel):
    """A pattern undergoing growth and evolution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    name: str
    description: str = ""
    phase: GrowthPhase = GrowthPhase.DORMANT
    stability: StabilityLevel = StabilityLevel.STABLE

    # Content
    content: Any
    generation: int = 0  # How many evolution cycles
    parent_id: str | None = None

    # Evolution tracking
    mutation_history: tuple[str, ...] = Field(default_factory=tuple)  # Mutation IDs
    viable: bool = True
    fitness: float = Field(default=0.5, ge=0.0, le=1.0)

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_mutated: datetime | None = None
    tags: frozenset[str] = Field(default_factory=frozenset)


class Mutation(BaseModel):
    """A mutation applied to a pattern."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    pattern_id: str
    mutation_type: MutationType
    description: str = ""

    # Changes
    before: Any
    after: Any
    changes: dict[str, Any] = Field(default_factory=dict)

    # Assessment
    fitness_delta: float = 0.0  # Change in fitness
    success: bool = True

    # Metadata
    applied_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EvolutionCycle(BaseModel):
    """A complete evolution cycle."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    name: str
    description: str = ""

    # State
    current_phase: GrowthPhase = GrowthPhase.DORMANT
    iteration: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Patterns
    pattern_ids: frozenset[str] = Field(default_factory=frozenset)

    # Configuration
    phase_duration: timedelta = timedelta(hours=1)
    auto_advance: bool = False

    # Results
    patterns_evolved: int = 0
    mutations_applied: int = 0
    patterns_culled: int = 0

    # Metadata
    tags: frozenset[str] = Field(default_factory=frozenset)


@dataclass
class EvolutionResult:
    """Result of an evolution step."""

    pattern_id: str
    success: bool
    new_phase: GrowthPhase | None = None
    mutation_applied: Mutation | None = None
    culled: bool = False
    reason: str = ""


@dataclass
class EvolutionStats:
    """Statistics about evolution."""

    total_patterns: int
    total_cycles: int
    active_cycles: int
    total_mutations: int
    average_fitness: float
    by_phase: dict[str, int]


class MutationEngine:
    """Engine for generating and applying mutations."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._log = logger.bind(component="mutation_engine")

    def generate_mutation(
        self,
        pattern: GrowthPattern,
        mutation_type: MutationType | None = None,
    ) -> Mutation:
        """Generate a mutation for a pattern."""
        if mutation_type is None:
            mutation_type = self._rng.choice(list(MutationType))

        before = pattern.content
        after, changes, description = self._apply_mutation_logic(before, mutation_type)

        # Calculate fitness impact
        fitness_delta = self._calculate_fitness_delta(pattern, mutation_type)

        return Mutation(
            pattern_id=pattern.id,
            mutation_type=mutation_type,
            description=description,
            before=before,
            after=after,
            changes=changes,
            fitness_delta=fitness_delta,
        )

    def _apply_mutation_logic(
        self,
        content: Any,
        mutation_type: MutationType,
    ) -> tuple[Any, dict[str, Any], str]:
        """Apply mutation logic to content."""
        new_content: Any
        changes: dict[str, Any] = {}
        description = ""

        if isinstance(content, dict):
            if mutation_type == MutationType.ADDITION:
                key = f"evolved_{len(content)}"
                new_content = {**content, key: "new_value"}
                changes = {"added": key}
                description = f"Added key: {key}"
                return new_content, changes, description

            elif mutation_type == MutationType.REMOVAL and content:
                key = self._rng.choice(list(content.keys()))
                new_content = {k: v for k, v in content.items() if k != key}
                changes = {"removed": key}
                description = f"Removed key: {key}"
                return new_content, changes, description

            elif mutation_type == MutationType.MODIFICATION and content:
                key = self._rng.choice(list(content.keys()))
                new_content = {**content, key: f"modified_{content[key]}"}
                changes = {"modified": key}
                description = f"Modified key: {key}"
                return new_content, changes, description

            elif mutation_type == MutationType.DUPLICATION and content:
                key = self._rng.choice(list(content.keys()))
                new_key = f"{key}_dup"
                new_content = {**content, new_key: content[key]}
                changes = {"duplicated": key, "to": new_key}
                description = f"Duplicated {key} to {new_key}"
                return new_content, changes, description

        elif isinstance(content, list):
            if mutation_type == MutationType.ADDITION:
                new_content = [*content, "new_element"]
                changes = {"added_at": len(content)}
                description = "Added new element"
                return new_content, changes, description

            elif mutation_type == MutationType.REMOVAL and content:
                idx = self._rng.randint(0, len(content) - 1)
                new_content = [v for i, v in enumerate(content) if i != idx]
                changes = {"removed_at": idx}
                description = f"Removed element at index {idx}"
                return new_content, changes, description

            elif mutation_type == MutationType.INVERSION:
                new_content = list(reversed(content))
                changes = {"inverted": True}
                description = "Inverted list order"
                return new_content, changes, description

        # Default: wrap in mutation marker
        new_content = {"mutated": content, "type": mutation_type.name}
        changes = {"wrapped": True}
        description = f"Applied {mutation_type.name} mutation"
        return new_content, changes, description

    def _calculate_fitness_delta(
        self,
        pattern: GrowthPattern,
        mutation_type: MutationType,
    ) -> float:
        """Calculate the fitness impact of a mutation."""
        # Stability affects mutation impact
        stability_factor = 1.0 - (pattern.stability.value * 0.15)

        # Different mutations have different typical impacts
        base_impact = {
            MutationType.ADDITION: 0.1,
            MutationType.REMOVAL: -0.05,
            MutationType.MODIFICATION: 0.05,
            MutationType.RECOMBINATION: 0.15,
            MutationType.INVERSION: 0.0,
            MutationType.DUPLICATION: 0.02,
        }

        base = base_impact.get(mutation_type, 0.0)
        variance = self._rng.uniform(-0.1, 0.1)

        return (base + variance) * stability_factor


class PhaseManager:
    """Manages growth phase transitions."""

    PHASE_ORDER = [
        GrowthPhase.DORMANT,
        GrowthPhase.GERMINATION,
        GrowthPhase.GROWTH,
        GrowthPhase.BLOOM,
        GrowthPhase.HARVEST,
        GrowthPhase.DECAY,
        GrowthPhase.RENEWAL,
    ]

    def __init__(self) -> None:
        self._log = logger.bind(component="phase_manager")

    def next_phase(self, current: GrowthPhase) -> GrowthPhase:
        """Get the next phase in the cycle."""
        idx = self.PHASE_ORDER.index(current)
        next_idx = (idx + 1) % len(self.PHASE_ORDER)
        return self.PHASE_ORDER[next_idx]

    def can_mutate(self, phase: GrowthPhase) -> bool:
        """Check if mutations are allowed in this phase."""
        return phase in (GrowthPhase.GROWTH, GrowthPhase.BLOOM)

    def is_harvest_phase(self, phase: GrowthPhase) -> bool:
        """Check if this is a harvest phase."""
        return phase == GrowthPhase.HARVEST


class EvolutionScheduler(Subsystem):
    """
    Manages timing and progression of growth and mutation cycles.

    Process Loop:
    1. Intake: Receive seasonal or symbolic input
    2. Process: Apply mutation logic to generate new patterns
    3. Evaluate: Assess new patterns for stability and coherence
    4. Integrate: Incorporate viable outcomes back into the system
    """

    def __init__(self, seed: int | None = None) -> None:
        metadata = SubsystemMetadata(
            name="evolution_scheduler",
            display_name="Evolution Scheduler",
            description="Manages timing and progression of growth and mutation cycles",
            type=SubsystemType.TEMPORAL,
            tags=frozenset(["evolution", "growth", "mutation", "cycles"]),
            input_types=frozenset(["SCHEMA", "PATTERN"]),
            output_types=frozenset(["SCHEMA", "PATTERN"]),
            subscribed_topics=frozenset([
                str(SubsystemTopics.ALL),
                "evolution.#",
                "growth.#",
            ]),
            published_topics=frozenset([
                "evolution.pattern.created",
                "evolution.mutation.applied",
                "evolution.phase.changed",
                "evolution.cycle.completed",
            ]),
        )
        super().__init__(metadata)

        self._mutation_engine = MutationEngine(seed)
        self._phase_manager = PhaseManager()
        self._patterns: dict[str, GrowthPattern] = {}
        self._cycles: dict[str, EvolutionCycle] = {}
        self._mutations: dict[str, Mutation] = {}

    @property
    def pattern_count(self) -> int:
        return len(self._patterns)

    @property
    def cycle_count(self) -> int:
        return len(self._cycles)

    async def intake(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> SymbolicInput:
        """Phase 1: Receive patterns for evolution."""
        if not input_data.values:
            self._log.debug("empty_input")
            return input_data

        self._log.debug("intake_complete", value_count=len(input_data.values))
        return input_data

    async def process(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> list[EvolutionResult]:
        """Phase 2: Apply mutation logic to patterns."""
        results: list[EvolutionResult] = []

        for value in input_data.values:
            pattern = self._parse_pattern(value)
            if pattern:
                self._patterns[pattern.id] = pattern

                # Check if we should mutate
                if self._phase_manager.can_mutate(pattern.phase):
                    mutation = self._mutation_engine.generate_mutation(pattern)
                    self._mutations[mutation.id] = mutation

                    # Apply mutation
                    evolved_pattern = self._apply_mutation(pattern, mutation)
                    self._patterns[evolved_pattern.id] = evolved_pattern

                    results.append(EvolutionResult(
                        pattern_id=evolved_pattern.id,
                        success=True,
                        mutation_applied=mutation,
                        reason="Mutation applied",
                    ))
                else:
                    results.append(EvolutionResult(
                        pattern_id=pattern.id,
                        success=True,
                        reason="Pattern registered (not in mutation phase)",
                    ))

        return results

    async def evaluate(
        self, intermediate: list[EvolutionResult],
        ctx: ProcessContext[dict[str, Any]],
    ) -> tuple[SymbolicOutput, bool]:
        """Phase 3: Assess patterns for stability and coherence."""
        values: list[SymbolicValue] = []

        for result in intermediate:
            pattern = self._patterns.get(result.pattern_id)
            if not pattern:
                continue

            value = SymbolicValue(
                type=SymbolicValueType.PATTERN,
                content={
                    "pattern_id": pattern.id,
                    "name": pattern.name,
                    "phase": pattern.phase.name,
                    "generation": pattern.generation,
                    "fitness": pattern.fitness,
                    "stability": pattern.stability.name,
                    "viable": pattern.viable,
                    "mutation_applied": result.mutation_applied.mutation_type.name if result.mutation_applied else None,
                },
                source_subsystem=self.name,
                tags=pattern.tags | frozenset(["evolution", pattern.phase.name.lower()]),
                meaning=f"Evolved: {pattern.name} (gen {pattern.generation})",
                confidence=pattern.fitness,
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
        """Phase 4: Emit events for evolved patterns."""
        if self._message_bus and output.values:
            for value in output.values:
                if value.content.get("mutation_applied"):
                    await self.emit_event(
                        "evolution.mutation.applied",
                        {
                            "pattern_id": value.content.get("pattern_id"),
                            "mutation_type": value.content.get("mutation_applied"),
                            "generation": value.content.get("generation"),
                        },
                    )
                else:
                    await self.emit_event(
                        "evolution.pattern.created",
                        {
                            "pattern_id": value.content.get("pattern_id"),
                            "name": value.content.get("name"),
                        },
                    )

        return None

    def _parse_pattern(self, value: SymbolicValue) -> GrowthPattern | None:
        """Parse a GrowthPattern from a SymbolicValue."""
        content = value.content
        if not isinstance(content, dict):
            content = {"data": content}

        try:
            phase_str = content.get("phase", "DORMANT")
            try:
                phase = GrowthPhase[phase_str.upper()]
            except KeyError:
                phase = GrowthPhase.DORMANT

            stability_str = content.get("stability", "STABLE")
            try:
                stability = StabilityLevel[stability_str.upper()]
            except KeyError:
                stability = StabilityLevel.STABLE

            return GrowthPattern(
                id=content.get("id", str(ULID())),
                name=content.get("name", f"pattern_{value.id[:8]}"),
                description=content.get("description", ""),
                phase=phase,
                stability=stability,
                content=content.get("content", content),
                generation=content.get("generation", 0),
                parent_id=content.get("parent_id"),
                fitness=content.get("fitness", 0.5),
                tags=frozenset(content.get("tags", [])) | value.tags,
            )
        except Exception as e:
            self._log.warning("pattern_parse_failed", value_id=value.id, error=str(e))
            return None

    def _apply_mutation(
        self,
        pattern: GrowthPattern,
        mutation: Mutation,
    ) -> GrowthPattern:
        """Apply a mutation to a pattern."""
        new_fitness = max(0.0, min(1.0, pattern.fitness + mutation.fitness_delta))

        return GrowthPattern(
            id=str(ULID()),
            name=pattern.name,
            description=pattern.description,
            phase=pattern.phase,
            stability=pattern.stability,
            content=mutation.after,
            generation=pattern.generation + 1,
            parent_id=pattern.id,
            mutation_history=(*pattern.mutation_history, mutation.id),
            viable=new_fitness > 0.2,
            fitness=new_fitness,
            last_mutated=datetime.now(UTC),
            tags=pattern.tags | frozenset([mutation.mutation_type.name.lower()]),
        )

    # --- Message handlers ---

    async def handle_event(self, message: Message) -> None:
        """Handle incoming events."""
        if message.topic.startswith("evolution."):
            self._log.debug("evolution_event_received", topic=message.topic)

    async def handle_signal(self, signal: Any) -> None:
        """Handle incoming signals."""
        self._log.debug("signal_received", signal_id=getattr(signal, "id", "unknown"))

    # --- Public API ---

    def create_pattern(
        self,
        name: str,
        content: Any,
        phase: GrowthPhase = GrowthPhase.DORMANT,
        **kwargs: Any,
    ) -> GrowthPattern:
        """Create a new growth pattern."""
        pattern = GrowthPattern(
            name=name,
            content=content,
            phase=phase,
            description=kwargs.get("description", ""),
            stability=kwargs.get("stability", StabilityLevel.STABLE),
            fitness=kwargs.get("fitness", 0.5),
            tags=frozenset(kwargs.get("tags", [])),
        )
        self._patterns[pattern.id] = pattern
        return pattern

    def mutate_pattern(
        self,
        pattern_id: str,
        mutation_type: MutationType | None = None,
    ) -> tuple[GrowthPattern | None, Mutation | None]:
        """Mutate a pattern."""
        pattern = self._patterns.get(pattern_id)
        if not pattern:
            return None, None

        mutation = self._mutation_engine.generate_mutation(pattern, mutation_type)
        self._mutations[mutation.id] = mutation

        evolved = self._apply_mutation(pattern, mutation)
        self._patterns[evolved.id] = evolved

        return evolved, mutation

    def advance_phase(self, pattern_id: str) -> GrowthPattern | None:
        """Advance a pattern to the next phase."""
        pattern = self._patterns.get(pattern_id)
        if not pattern:
            return None

        next_phase = self._phase_manager.next_phase(pattern.phase)

        updated = GrowthPattern(
            id=pattern.id,
            name=pattern.name,
            description=pattern.description,
            phase=next_phase,
            stability=pattern.stability,
            content=pattern.content,
            generation=pattern.generation,
            parent_id=pattern.parent_id,
            mutation_history=pattern.mutation_history,
            viable=pattern.viable,
            fitness=pattern.fitness,
            created_at=pattern.created_at,
            last_mutated=pattern.last_mutated,
            tags=pattern.tags,
        )
        self._patterns[pattern_id] = updated
        return updated

    def create_cycle(
        self,
        name: str,
        pattern_ids: list[str],
        **kwargs: Any,
    ) -> EvolutionCycle:
        """Create an evolution cycle."""
        cycle = EvolutionCycle(
            name=name,
            description=kwargs.get("description", ""),
            pattern_ids=frozenset(pattern_ids),
            phase_duration=timedelta(seconds=kwargs.get("phase_duration_seconds", 3600)),
            auto_advance=kwargs.get("auto_advance", False),
            tags=frozenset(kwargs.get("tags", [])),
        )
        self._cycles[cycle.id] = cycle
        return cycle

    def start_cycle(self, cycle_id: str) -> EvolutionCycle | None:
        """Start an evolution cycle."""
        cycle = self._cycles.get(cycle_id)
        if not cycle:
            return None

        updated = EvolutionCycle(
            id=cycle.id,
            name=cycle.name,
            description=cycle.description,
            current_phase=GrowthPhase.GERMINATION,
            iteration=1,
            started_at=datetime.now(UTC),
            pattern_ids=cycle.pattern_ids,
            phase_duration=cycle.phase_duration,
            auto_advance=cycle.auto_advance,
            tags=cycle.tags,
        )
        self._cycles[cycle_id] = updated
        return updated

    def get_pattern(self, pattern_id: str) -> GrowthPattern | None:
        """Get a pattern by ID."""
        return self._patterns.get(pattern_id)

    def get_cycle(self, cycle_id: str) -> EvolutionCycle | None:
        """Get a cycle by ID."""
        return self._cycles.get(cycle_id)

    def get_patterns_by_phase(self, phase: GrowthPhase) -> list[GrowthPattern]:
        """Get patterns in a specific phase."""
        return [p for p in self._patterns.values() if p.phase == phase]

    def get_viable_patterns(self) -> list[GrowthPattern]:
        """Get all viable patterns."""
        return [p for p in self._patterns.values() if p.viable]

    def cull_unviable(self) -> int:
        """Remove unviable patterns. Returns count removed."""
        unviable = [p.id for p in self._patterns.values() if not p.viable]
        for pid in unviable:
            del self._patterns[pid]
        return len(unviable)

    def get_stats(self) -> EvolutionStats:
        """Get evolution statistics."""
        by_phase: dict[str, int] = {}
        total_fitness = 0.0

        for pattern in self._patterns.values():
            by_phase[pattern.phase.name] = by_phase.get(pattern.phase.name, 0) + 1
            total_fitness += pattern.fitness

        avg_fitness = total_fitness / len(self._patterns) if self._patterns else 0.0
        active_cycles = len([c for c in self._cycles.values() if c.started_at and not c.completed_at])

        return EvolutionStats(
            total_patterns=len(self._patterns),
            total_cycles=len(self._cycles),
            active_cycles=active_cycles,
            total_mutations=len(self._mutations),
            average_fitness=avg_fitness,
            by_phase=by_phase,
        )

    def clear(self) -> tuple[int, int, int]:
        """Clear all patterns, cycles, mutations. Returns counts cleared."""
        patterns = len(self._patterns)
        cycles = len(self._cycles)
        mutations = len(self._mutations)
        self._patterns.clear()
        self._cycles.clear()
        self._mutations.clear()
        return patterns, cycles, mutations
