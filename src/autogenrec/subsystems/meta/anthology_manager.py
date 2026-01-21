"""
AnthologyManager: The meta-system for the Recursive-Generative architecture.

Functions as the registry, index, and access layer for all subsystems,
processes, and artifacts in the system.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field
from ulid import ULID

from autogenrec.bus.topics import SubsystemTopics, SystemTopics, Topic
from autogenrec.core.process import ProcessContext
from autogenrec.core.registry import ProcessRegistry, SubsystemRegistry
from autogenrec.core.signals import Message, MessageType
from autogenrec.core.subsystem import Subsystem, SubsystemMetadata, SubsystemType
from autogenrec.core.symbolic import (
    SymbolicInput,
    SymbolicOutput,
    SymbolicValue,
    SymbolicValueType,
)

logger = structlog.get_logger()


class AnthologyEntry(BaseModel):
    """A normalized entry in the anthology registry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    entry_type: str  # "subsystem", "process", "artifact", "event"
    name: str
    source: str  # Source subsystem
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Normalized schema fields
    category: str
    tags: frozenset[str] = Field(default_factory=frozenset)
    description: str | None = None

    # Original data
    raw_data: dict[str, Any] = Field(default_factory=dict)

    # Relationships
    related_entries: frozenset[str] = Field(default_factory=frozenset)


@dataclass
class QueryResult:
    """Result from a cross-system query."""

    entries: list[AnthologyEntry]
    total_count: int
    query_time_ms: float
    query: dict[str, Any] = field(default_factory=dict)


class CrossQueryEngine:
    """Engine for cross-system queries against the anthology."""

    def __init__(self, entries: dict[str, AnthologyEntry]) -> None:
        self._entries = entries
        self._log = logger.bind(component="cross_query_engine")

    def query(
        self,
        *,
        entry_type: str | None = None,
        source: str | None = None,
        category: str | None = None,
        tags: set[str] | None = None,
        name_contains: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> QueryResult:
        """
        Query the anthology with filters.

        Args:
            entry_type: Filter by entry type
            source: Filter by source subsystem
            category: Filter by category
            tags: Filter by tags (entries must have all specified tags)
            name_contains: Filter by name substring
            limit: Maximum results to return
            offset: Number of results to skip

        Returns:
            QueryResult with matching entries
        """
        start = datetime.now(UTC)

        # Apply filters
        results: list[AnthologyEntry] = []
        for entry in self._entries.values():
            if entry_type and entry.entry_type != entry_type:
                continue
            if source and entry.source != source:
                continue
            if category and entry.category != category:
                continue
            if tags and not tags <= entry.tags:
                continue
            if name_contains and name_contains.lower() not in entry.name.lower():
                continue
            results.append(entry)

        # Sort by created_at descending
        results.sort(key=lambda e: e.created_at, reverse=True)

        total = len(results)
        results = results[offset : offset + limit]

        elapsed = (datetime.now(UTC) - start).total_seconds() * 1000

        return QueryResult(
            entries=results,
            total_count=total,
            query_time_ms=elapsed,
            query={
                "entry_type": entry_type,
                "source": source,
                "category": category,
                "tags": list(tags) if tags else None,
                "name_contains": name_contains,
            },
        )

    def get_related(self, entry_id: str) -> list[AnthologyEntry]:
        """Get entries related to a given entry."""
        entry = self._entries.get(entry_id)
        if not entry:
            return []

        return [
            self._entries[rid]
            for rid in entry.related_entries
            if rid in self._entries
        ]


class AnthologyManager(Subsystem):
    """
    Meta-system that maintains the registry and index for all subsystems.

    Subscribes to all system topics to collect and normalize outputs
    from every subsystem into a unified, searchable anthology.

    Process Loop (5 phases):
    1. Collect: Ingest outputs and state data from all subsystems
    2. Normalize: Convert diverse data into consistent schema entries
    3. Register: Assign unique identifiers and process metadata
    4. Expose: Provide search and query capabilities
    5. Archive: Store in durable storage
    """

    # List of all subsystem names to be registered
    SUBSYSTEM_NAMES = [
        "academia_manager",
        "evolution_scheduler",
        "conflict_resolver",
        "symbolic_interpreter",
        "arbitration_engine",
        "node_router",
        "rule_compiler",
        "code_generator",
        "reference_manager",
        "echo_handler",
        "archive_manager",
        "mask_generator",
        "value_exchange_manager",
        "blockchain_simulator",
        "process_monetizer",
        "audience_classifier",
        "location_resolver",
        "signal_threshold_guard",
        "time_manager",
        "process_converter",
        "consumption_manager",
    ]

    def __init__(
        self,
        process_registry: ProcessRegistry | None = None,
        subsystem_registry: SubsystemRegistry | None = None,
    ) -> None:
        metadata = SubsystemMetadata(
            name="anthology_manager",
            display_name="Anthology Manager",
            description="Meta-system registry and index for all subsystems",
            type=SubsystemType.META,
            tags=frozenset(["registry", "meta", "process", "index"]),
            subscribed_topics=frozenset([
                str(SystemTopics.ALL),
                str(SubsystemTopics.ALL),
                "signal.#",
                "process.#",
                "data.#",
            ]),
            published_topics=frozenset([
                "anthology.entry.created",
                "anthology.query.result",
            ]),
        )
        super().__init__(metadata)

        self._process_registry = process_registry or ProcessRegistry()
        self._subsystem_registry = subsystem_registry or SubsystemRegistry()
        self._entries: dict[str, AnthologyEntry] = {}
        self._query_engine = CrossQueryEngine(self._entries)
        self._normalizers: dict[str, Callable[[Any], AnthologyEntry | None]] = {}

        # Register default normalizers
        self._register_default_normalizers()

    @property
    def process_registry(self) -> ProcessRegistry:
        return self._process_registry

    @property
    def subsystem_registry(self) -> SubsystemRegistry:
        return self._subsystem_registry

    @property
    def query_engine(self) -> CrossQueryEngine:
        return self._query_engine

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def _register_default_normalizers(self) -> None:
        """Register default normalizers for common data types."""

        def normalize_subsystem_event(data: dict[str, Any]) -> AnthologyEntry | None:
            if "subsystem" not in data:
                return None
            return AnthologyEntry(
                entry_type="event",
                name=f"subsystem.{data.get('event', 'unknown')}",
                source=data.get("subsystem", "unknown"),
                category="lifecycle",
                tags=frozenset(["subsystem", "event"]),
                raw_data=data,
            )

        def normalize_process_result(data: dict[str, Any]) -> AnthologyEntry | None:
            if "process_id" not in data:
                return None
            return AnthologyEntry(
                entry_type="process",
                name=data.get("process_name", "unknown"),
                source=data.get("source", "unknown"),
                category="execution",
                tags=frozenset(["process", "result"]),
                raw_data=data,
            )

        self._normalizers["subsystem_event"] = normalize_subsystem_event
        self._normalizers["process_result"] = normalize_process_result

    def register_normalizer(
        self,
        name: str,
        normalizer: Callable[[Any], AnthologyEntry | None],
    ) -> None:
        """Register a custom normalizer for a data type."""
        self._normalizers[name] = normalizer

    # --- ProcessLoop implementation ---

    async def intake(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> SymbolicInput:
        """Phase 1: Collect - Validate and prepare input data."""
        # Basic validation
        if not input_data.values:
            self._log.debug("empty_input")
            return input_data

        self._log.debug(
            "intake",
            value_count=len(input_data.values),
            source=input_data.source_subsystem,
        )

        return input_data

    async def process(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> list[AnthologyEntry]:
        """Phase 2 & 3: Normalize and Register - Convert to anthology entries."""
        entries: list[AnthologyEntry] = []

        for value in input_data.values:
            entry = self._normalize_value(value)
            if entry:
                self._entries[entry.id] = entry
                entries.append(entry)
                self._log.debug(
                    "entry_registered",
                    entry_id=entry.id,
                    entry_type=entry.entry_type,
                )

        return entries

    async def evaluate(
        self, intermediate: list[AnthologyEntry], ctx: ProcessContext[dict[str, Any]]
    ) -> tuple[SymbolicOutput, bool]:
        """Phase 4: Expose - Create output with registered entries."""
        values = [
            SymbolicValue(
                type=SymbolicValueType.REFERENCE,
                content={"entry_id": entry.id, "entry_type": entry.entry_type},
                source_subsystem=self.name,
                tags=entry.tags,
            )
            for entry in intermediate
        ]

        output = self.create_output(
            values=values,
            input_id=ctx.metadata.get("input_id"),
        )

        # Don't continue iterating
        return output, False

    async def integrate(
        self, output: SymbolicOutput, ctx: ProcessContext[dict[str, Any]]
    ) -> SymbolicInput | None:
        """Phase 5: Archive - Emit events and return."""
        # Publish entry created events
        if self._message_bus and output.values:
            for value in output.values:
                await self.emit_event(
                    "anthology.entry.created",
                    value.content,
                )

        # No recursion needed
        return None

    def _normalize_value(self, value: SymbolicValue) -> AnthologyEntry | None:
        """Normalize a symbolic value into an anthology entry."""
        # Try type-specific normalizers first
        type_name = value.type.name.lower()
        if type_name in self._normalizers:
            return self._normalizers[type_name](value.content)

        # Generic normalization
        return AnthologyEntry(
            entry_type="artifact",
            name=value.meaning or f"value_{value.id[:8]}",
            source=value.source_subsystem or "unknown",
            category=value.type.name.lower(),
            tags=value.tags,
            description=value.meaning,
            raw_data={"content": value.content, "value_id": value.id},
        )

    # --- Message handlers ---

    async def handle_event(self, message: Message) -> None:
        """Handle system and subsystem events."""
        topic = Topic(message.topic)

        if topic.category == "subsystem":
            # Index subsystem lifecycle events
            entry = AnthologyEntry(
                entry_type="event",
                name=message.topic,
                source=message.source,
                category="lifecycle",
                tags=frozenset(["subsystem", "event", message.source]),
                raw_data={"payload": message.payload, "message_id": message.id},
            )
            self._entries[entry.id] = entry

        elif topic.category == "system":
            # Index system events
            entry = AnthologyEntry(
                entry_type="event",
                name=message.topic,
                source="system",
                category="system",
                tags=frozenset(["system", "event"]),
                raw_data={"payload": message.payload, "message_id": message.id},
            )
            self._entries[entry.id] = entry

    async def handle_signal(self, signal: Any) -> None:
        """Handle incoming signals for indexing."""
        if hasattr(signal, "id"):
            entry = AnthologyEntry(
                entry_type="signal",
                name=f"signal_{signal.id[:8]}",
                source=getattr(signal, "source", "unknown"),
                category="communication",
                tags=frozenset(["signal"]),
                raw_data={"signal_id": signal.id},
            )
            self._entries[entry.id] = entry

    # --- Query API ---

    def query(self, **kwargs: Any) -> QueryResult:
        """Query the anthology."""
        return self._query_engine.query(**kwargs)

    def get_entry(self, entry_id: str) -> AnthologyEntry | None:
        """Get an entry by ID."""
        return self._entries.get(entry_id)

    def get_entries_by_source(self, source: str) -> list[AnthologyEntry]:
        """Get all entries from a source subsystem."""
        return self.query(source=source).entries

    def get_entries_by_type(self, entry_type: str) -> list[AnthologyEntry]:
        """Get all entries of a specific type."""
        return self.query(entry_type=entry_type).entries

    def search(self, name_contains: str) -> list[AnthologyEntry]:
        """Search entries by name."""
        return self.query(name_contains=name_contains).entries

    def get_subsystem_names(self) -> list[str]:
        """Get list of all known subsystem names."""
        return list(self.SUBSYSTEM_NAMES)

    def export_map(self) -> dict[str, Any]:
        """Export the full system map for external use."""
        return {
            "entry_count": len(self._entries),
            "entries_by_type": {
                etype: len(self.query(entry_type=etype).entries)
                for etype in {"subsystem", "process", "artifact", "event", "signal"}
            },
            "subsystems": self.SUBSYSTEM_NAMES,
            "exported_at": datetime.now(UTC).isoformat(),
        }
