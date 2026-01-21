"""
ReferenceManager: Establishes and maintains canonical references.

Manages a reference graph that provides stable anchoring for recursive
knowledge across all subsystems.
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


class ReferenceType(Enum):
    """Types of canonical references."""

    CANONICAL = auto()  # Authoritative source
    CITATION = auto()  # Citation/link to another reference
    DERIVED = auto()  # Derived from another reference
    ALIAS = auto()  # Alternative name/identifier
    VERSIONED = auto()  # Specific version of a reference
    EXTERNAL = auto()  # External reference (URL, etc.)


class ReferenceStatus(Enum):
    """Status of a reference."""

    ACTIVE = auto()  # Currently valid
    DEPRECATED = auto()  # Marked for removal
    SUPERSEDED = auto()  # Replaced by newer version
    DRAFT = auto()  # Not yet validated
    ARCHIVED = auto()  # Historical, no longer active


class ValidationResult(Enum):
    """Result of reference validation."""

    VALID = auto()
    INVALID = auto()
    INCOMPLETE = auto()
    NEEDS_UPDATE = auto()


class Reference(BaseModel):
    """A canonical reference in the system."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    name: str
    ref_type: ReferenceType = ReferenceType.CANONICAL
    status: ReferenceStatus = ReferenceStatus.ACTIVE

    # Content
    content: Any
    content_hash: str = ""  # For integrity checking

    # Metadata
    description: str = ""
    source: str = ""  # Origin subsystem or external source
    version: int = 1
    tags: frozenset[str] = Field(default_factory=frozenset)

    # Relationships
    parent_id: str | None = None  # For derived/versioned references
    superseded_by: str | None = None  # ID of superseding reference
    related_ids: frozenset[str] = Field(default_factory=frozenset)

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    validated_at: datetime | None = None


class ReferenceEdge(BaseModel):
    """An edge in the reference graph."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    source_id: str
    target_id: str
    edge_type: str  # "cites", "derives_from", "alias_of", "supersedes", "relates_to"
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ReferenceQuery:
    """Query parameters for searching references."""

    name_contains: str | None = None
    ref_type: ReferenceType | None = None
    status: ReferenceStatus | None = None
    tags: set[str] | None = None
    source: str | None = None
    limit: int = 100
    offset: int = 0


@dataclass
class QueryResult:
    """Result of a reference query."""

    references: list[Reference]
    total_count: int
    query_time_ms: float


class ReferenceGraph:
    """Graph structure for managing reference relationships."""

    def __init__(self) -> None:
        self._references: dict[str, Reference] = {}
        self._edges: dict[str, ReferenceEdge] = {}
        self._outgoing: dict[str, set[str]] = {}  # ref_id -> set of edge_ids
        self._incoming: dict[str, set[str]] = {}  # ref_id -> set of edge_ids
        self._by_name: dict[str, set[str]] = {}  # name -> set of ref_ids
        self._log = logger.bind(component="reference_graph")

    @property
    def reference_count(self) -> int:
        return len(self._references)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def add_reference(self, reference: Reference) -> None:
        """Add a reference to the graph."""
        self._references[reference.id] = reference
        self._outgoing.setdefault(reference.id, set())
        self._incoming.setdefault(reference.id, set())
        self._by_name.setdefault(reference.name.lower(), set()).add(reference.id)

        self._log.debug("reference_added", ref_id=reference.id, name=reference.name)

    def add_edge(self, edge: ReferenceEdge) -> None:
        """Add an edge between references."""
        if edge.source_id not in self._references:
            raise ValueError(f"Source reference not found: {edge.source_id}")
        if edge.target_id not in self._references:
            raise ValueError(f"Target reference not found: {edge.target_id}")

        self._edges[edge.id] = edge
        self._outgoing[edge.source_id].add(edge.id)
        self._incoming[edge.target_id].add(edge.id)

        self._log.debug(
            "edge_added",
            edge_id=edge.id,
            source=edge.source_id,
            target=edge.target_id,
            type=edge.edge_type,
        )

    def get_reference(self, ref_id: str) -> Reference | None:
        """Get a reference by ID."""
        return self._references.get(ref_id)

    def get_by_name(self, name: str) -> list[Reference]:
        """Get references by name (case-insensitive)."""
        ref_ids = self._by_name.get(name.lower(), set())
        return [self._references[rid] for rid in ref_ids if rid in self._references]

    def get_related(self, ref_id: str, edge_type: str | None = None) -> list[Reference]:
        """Get references related to the given reference."""
        related: list[Reference] = []

        # Outgoing edges
        for edge_id in self._outgoing.get(ref_id, set()):
            edge = self._edges.get(edge_id)
            if edge and (edge_type is None or edge.edge_type == edge_type):
                target = self._references.get(edge.target_id)
                if target:
                    related.append(target)

        # Incoming edges
        for edge_id in self._incoming.get(ref_id, set()):
            edge = self._edges.get(edge_id)
            if edge and (edge_type is None or edge.edge_type == edge_type):
                source = self._references.get(edge.source_id)
                if source:
                    related.append(source)

        return related

    def get_ancestors(self, ref_id: str, max_depth: int = 10) -> list[Reference]:
        """Get all ancestor references (via parent_id chain)."""
        ancestors: list[Reference] = []
        current_id = ref_id
        depth = 0

        while depth < max_depth:
            ref = self._references.get(current_id)
            if not ref or not ref.parent_id:
                break
            parent = self._references.get(ref.parent_id)
            if not parent:
                break
            ancestors.append(parent)
            current_id = parent.id
            depth += 1

        return ancestors

    def get_descendants(self, ref_id: str) -> list[Reference]:
        """Get all references that derive from this one."""
        descendants: list[Reference] = []
        for ref in self._references.values():
            if ref.parent_id == ref_id:
                descendants.append(ref)
        return descendants

    def search(self, query: ReferenceQuery) -> QueryResult:
        """Search references with filters."""
        start = datetime.now(UTC)
        results: list[Reference] = []

        for ref in self._references.values():
            if query.name_contains and query.name_contains.lower() not in ref.name.lower():
                continue
            if query.ref_type and ref.ref_type != query.ref_type:
                continue
            if query.status and ref.status != query.status:
                continue
            if query.source and ref.source != query.source:
                continue
            if query.tags and not query.tags <= ref.tags:
                continue
            results.append(ref)

        # Sort by created_at descending
        results.sort(key=lambda r: r.created_at, reverse=True)

        total = len(results)
        results = results[query.offset : query.offset + query.limit]

        elapsed = (datetime.now(UTC) - start).total_seconds() * 1000

        return QueryResult(
            references=results,
            total_count=total,
            query_time_ms=elapsed,
        )

    def remove_reference(self, ref_id: str) -> bool:
        """Remove a reference and its edges from the graph."""
        if ref_id not in self._references:
            return False

        ref = self._references.pop(ref_id)

        # Remove from name index
        if ref.name.lower() in self._by_name:
            self._by_name[ref.name.lower()].discard(ref_id)

        # Remove edges
        for edge_id in list(self._outgoing.get(ref_id, set())):
            if edge_id in self._edges:
                edge = self._edges.pop(edge_id)
                self._incoming[edge.target_id].discard(edge_id)

        for edge_id in list(self._incoming.get(ref_id, set())):
            if edge_id in self._edges:
                edge = self._edges.pop(edge_id)
                self._outgoing[edge.source_id].discard(edge_id)

        self._outgoing.pop(ref_id, None)
        self._incoming.pop(ref_id, None)

        self._log.debug("reference_removed", ref_id=ref_id)
        return True


class ReferenceValidator:
    """Validates references for completeness and integrity."""

    def __init__(self) -> None:
        self._log = logger.bind(component="reference_validator")

    def validate(self, reference: Reference, graph: ReferenceGraph) -> ValidationResult:
        """Validate a reference."""
        # Check required fields
        if not reference.name:
            return ValidationResult.INCOMPLETE
        if reference.content is None:
            return ValidationResult.INCOMPLETE

        # Check parent exists if specified
        if reference.parent_id and not graph.get_reference(reference.parent_id):
            return ValidationResult.INVALID

        # Check if superseded
        if reference.superseded_by:
            successor = graph.get_reference(reference.superseded_by)
            if successor and successor.status == ReferenceStatus.ACTIVE:
                return ValidationResult.NEEDS_UPDATE

        # Check related references exist
        for related_id in reference.related_ids:
            if not graph.get_reference(related_id):
                return ValidationResult.INCOMPLETE

        return ValidationResult.VALID


class ReferenceManager(Subsystem):
    """
    Establishes and maintains canonical references for the system.

    Process Loop:
    1. Intake: Receive references or canonical records
    2. Process: Validate authenticity and completeness
    3. Evaluate: Update and refine as needed
    4. Integrate: Distribute references to other subsystems
    """

    def __init__(self) -> None:
        metadata = SubsystemMetadata(
            name="reference_manager",
            display_name="Reference Manager",
            description="Establishes and maintains canonical references",
            type=SubsystemType.DATA,
            tags=frozenset(["reference", "canonical", "knowledge", "graph"]),
            input_types=frozenset(["REFERENCE", "SCHEMA"]),
            output_types=frozenset(["REFERENCE"]),
            subscribed_topics=frozenset([
                str(SubsystemTopics.ALL),
                "reference.create.#",
                "reference.update.#",
                "reference.query.#",
            ]),
            published_topics=frozenset([
                "reference.created",
                "reference.updated",
                "reference.validated",
                "reference.deprecated",
            ]),
        )
        super().__init__(metadata)

        self._graph = ReferenceGraph()
        self._validator = ReferenceValidator()

    @property
    def reference_count(self) -> int:
        return self._graph.reference_count

    @property
    def edge_count(self) -> int:
        return self._graph.edge_count

    async def intake(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> SymbolicInput:
        """Phase 1: Receive references or canonical records."""
        if not input_data.values:
            self._log.debug("empty_input")
            return input_data

        # Filter to reference-related types
        supported_types = {SymbolicValueType.REFERENCE, SymbolicValueType.SCHEMA}
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
    ) -> list[tuple[Reference, ValidationResult]]:
        """Phase 2: Validate and process references."""
        results: list[tuple[Reference, ValidationResult]] = []

        for value in input_data.values:
            reference = self._parse_reference(value)
            if reference:
                # Add to graph first
                self._graph.add_reference(reference)

                # Add edges for relationships
                self._add_relationship_edges(reference)

                # Validate
                validation_result = self._validator.validate(reference, self._graph)
                results.append((reference, validation_result))

                self._log.debug(
                    "reference_processed",
                    ref_id=reference.id,
                    validation=validation_result.name,
                )

        return results

    async def evaluate(
        self, intermediate: list[tuple[Reference, ValidationResult]],
        ctx: ProcessContext[dict[str, Any]],
    ) -> tuple[SymbolicOutput, bool]:
        """Phase 3: Create output with processed references."""
        values: list[SymbolicValue] = []

        for reference, validation_result in intermediate:
            value = SymbolicValue(
                type=SymbolicValueType.REFERENCE,
                content={
                    "ref_id": reference.id,
                    "name": reference.name,
                    "ref_type": reference.ref_type.name,
                    "status": reference.status.name,
                    "version": reference.version,
                    "validation": validation_result.name,
                    "related_count": len(reference.related_ids),
                },
                source_subsystem=self.name,
                tags=reference.tags | frozenset(["reference", reference.ref_type.name.lower()]),
                meaning=f"Reference: {reference.name}",
                confidence=1.0 if validation_result == ValidationResult.VALID else 0.7,
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
        """Phase 4: Emit events for references."""
        if self._message_bus and output.values:
            for value in output.values:
                await self.emit_event(
                    "reference.created",
                    {
                        "ref_id": value.content.get("ref_id"),
                        "name": value.content.get("name"),
                        "ref_type": value.content.get("ref_type"),
                    },
                )

        return None

    def _parse_reference(self, value: SymbolicValue) -> Reference | None:
        """Parse a SymbolicValue into a Reference."""
        content = value.content
        if not isinstance(content, dict):
            return None

        try:
            ref_type_str = content.get("ref_type", "CANONICAL")
            try:
                ref_type = ReferenceType[ref_type_str.upper()]
            except KeyError:
                ref_type = ReferenceType.CANONICAL

            status_str = content.get("status", "ACTIVE")
            try:
                status = ReferenceStatus[status_str.upper()]
            except KeyError:
                status = ReferenceStatus.ACTIVE

            reference = Reference(
                id=content.get("id", str(ULID())),
                name=content.get("name", f"ref_{value.id[:8]}"),
                ref_type=ref_type,
                status=status,
                content=content.get("content", content),
                content_hash=content.get("content_hash", ""),
                description=content.get("description", ""),
                source=content.get("source", value.source_subsystem or ""),
                version=content.get("version", 1),
                tags=frozenset(content.get("tags", [])) | value.tags,
                parent_id=content.get("parent_id"),
                superseded_by=content.get("superseded_by"),
                related_ids=frozenset(content.get("related_ids", [])),
            )

            return reference

        except Exception as e:
            self._log.warning("parse_failed", value_id=value.id, error=str(e))
            return None

    def _add_relationship_edges(self, reference: Reference) -> None:
        """Add edges for reference relationships."""
        # Parent relationship
        if reference.parent_id:
            try:
                edge = ReferenceEdge(
                    source_id=reference.id,
                    target_id=reference.parent_id,
                    edge_type="derives_from",
                )
                self._graph.add_edge(edge)
            except ValueError:
                pass  # Parent doesn't exist yet

        # Related references
        for related_id in reference.related_ids:
            try:
                edge = ReferenceEdge(
                    source_id=reference.id,
                    target_id=related_id,
                    edge_type="relates_to",
                )
                self._graph.add_edge(edge)
            except ValueError:
                pass  # Related ref doesn't exist yet

    # --- Message handlers ---

    async def handle_event(self, message: Message) -> None:
        """Handle incoming events."""
        if message.topic.startswith("reference.query"):
            self._log.debug("query_request_received", message_id=message.id)

    async def handle_signal(self, signal: Any) -> None:
        """Handle incoming signals."""
        self._log.debug("signal_received", signal_id=getattr(signal, "id", "unknown"))

    # --- Public API ---

    def create_reference(
        self,
        name: str,
        content: Any,
        ref_type: ReferenceType = ReferenceType.CANONICAL,
        **kwargs: Any,
    ) -> Reference:
        """Create and add a new reference."""
        reference = Reference(
            name=name,
            content=content,
            ref_type=ref_type,
            tags=frozenset(kwargs.get("tags", [])),
            description=kwargs.get("description", ""),
            source=kwargs.get("source", ""),
            parent_id=kwargs.get("parent_id"),
            related_ids=frozenset(kwargs.get("related_ids", [])),
        )
        self._graph.add_reference(reference)
        self._add_relationship_edges(reference)
        return reference

    def resolve(self, ref_id: str) -> Reference | None:
        """Resolve a reference by ID."""
        return self._graph.get_reference(ref_id)

    def resolve_by_name(self, name: str) -> list[Reference]:
        """Resolve references by name."""
        return self._graph.get_by_name(name)

    def get_related(self, ref_id: str, edge_type: str | None = None) -> list[Reference]:
        """Get related references."""
        return self._graph.get_related(ref_id, edge_type)

    def search(self, query: ReferenceQuery) -> QueryResult:
        """Search references."""
        return self._graph.search(query)

    def link(self, source_id: str, target_id: str, edge_type: str = "relates_to") -> ReferenceEdge:
        """Create a link between two references."""
        edge = ReferenceEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
        )
        self._graph.add_edge(edge)
        return edge

    def deprecate(self, ref_id: str, superseded_by: str | None = None) -> bool:
        """Deprecate a reference."""
        ref = self._graph.get_reference(ref_id)
        if not ref:
            return False

        # Create updated reference with deprecated status
        # Since Reference is frozen, we need to remove and re-add
        self._graph.remove_reference(ref_id)

        new_ref = Reference(
            id=ref.id,
            name=ref.name,
            ref_type=ref.ref_type,
            status=ReferenceStatus.DEPRECATED,
            content=ref.content,
            content_hash=ref.content_hash,
            description=ref.description,
            source=ref.source,
            version=ref.version,
            tags=ref.tags,
            parent_id=ref.parent_id,
            superseded_by=superseded_by,
            related_ids=ref.related_ids,
            created_at=ref.created_at,
            updated_at=datetime.now(UTC),
        )
        self._graph.add_reference(new_ref)
        return True

    def validate(self, ref_id: str) -> ValidationResult:
        """Validate a specific reference."""
        ref = self._graph.get_reference(ref_id)
        if not ref:
            return ValidationResult.INVALID
        return self._validator.validate(ref, self._graph)

    def clear(self) -> int:
        """Clear all references. Returns count cleared."""
        count = self._graph.reference_count
        self._graph = ReferenceGraph()
        return count
