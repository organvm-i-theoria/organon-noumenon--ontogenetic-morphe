"""
LocationResolver: Resolves spatial references and manages symbolic places.

Determines and manages symbolic places, resolves ambiguous references into
structured locations, and maintains a coherent spatial ontology.
"""

from dataclasses import dataclass
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


class PlaceType(Enum):
    """Types of symbolic places."""

    PHYSICAL = auto()  # Real-world location
    VIRTUAL = auto()  # Digital/virtual space
    CONCEPTUAL = auto()  # Abstract/conceptual space
    MYTHIC = auto()  # Mythological/symbolic place
    LIMINAL = auto()  # Threshold/transitional space
    NESTED = auto()  # Place within a place


class SpatialRelation(Enum):
    """Types of spatial relationships."""

    CONTAINS = auto()  # A contains B
    CONTAINED_BY = auto()  # A is within B
    ADJACENT = auto()  # A is next to B
    CONNECTED = auto()  # A is connected to B
    DISTANT = auto()  # A is far from B
    OVERLAPS = auto()  # A partially overlaps B
    OPPOSITE = auto()  # A is opposite to B
    PARALLEL = auto()  # A runs parallel to B


class ResolutionStatus(Enum):
    """Status of a location resolution."""

    RESOLVED = auto()  # Successfully resolved
    AMBIGUOUS = auto()  # Multiple matches
    UNRESOLVED = auto()  # No match found
    PARTIAL = auto()  # Partially resolved


class Place(BaseModel):
    """A symbolic place in the spatial ontology."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    name: str
    place_type: PlaceType = PlaceType.PHYSICAL
    description: str = ""

    # Coordinates (symbolic or actual)
    coordinates: tuple[float, float, float] | None = None
    symbolic_position: str | None = None  # e.g., "center", "threshold", "above"

    # Hierarchy
    parent_id: str | None = None  # Container place
    aliases: frozenset[str] = Field(default_factory=frozenset)

    # Properties
    properties: dict[str, Any] = Field(default_factory=dict)
    tags: frozenset[str] = Field(default_factory=frozenset)

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SpatialMarker(BaseModel):
    """A marker identifying a spatial reference."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    identifier: str  # The raw identifier (name, coordinate, etc.)
    marker_type: str = "name"  # name, coordinate, reference, symbolic

    # Context
    context: dict[str, Any] = Field(default_factory=dict)
    source: str | None = None


class SpatialLink(BaseModel):
    """A relationship between two places."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    source_id: str
    target_id: str
    relation: SpatialRelation
    bidirectional: bool = False
    weight: float = 1.0  # Strength/distance of relationship
    properties: dict[str, Any] = Field(default_factory=dict)


class LocationAssignment(BaseModel):
    """An assignment of an entity to a location."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    entity_id: str
    entity_type: str
    place_id: str
    assigned_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    valid_until: datetime | None = None
    properties: dict[str, Any] = Field(default_factory=dict)


@dataclass
class ResolutionResult:
    """Result of resolving a spatial reference."""

    marker: SpatialMarker
    status: ResolutionStatus
    place: Place | None = None
    candidates: list[Place] | None = None  # For ambiguous results
    confidence: float = 0.0
    error: str | None = None


@dataclass
class LocationStats:
    """Statistics about the location system."""

    total_places: int
    places_by_type: dict[str, int]
    total_links: int
    total_assignments: int


class PlaceRegistry:
    """Registry of symbolic places."""

    def __init__(self) -> None:
        self._places: dict[str, Place] = {}
        self._by_name: dict[str, set[str]] = {}  # name -> place IDs
        self._by_type: dict[PlaceType, set[str]] = {}
        self._children: dict[str, set[str]] = {}  # parent_id -> child IDs
        self._log = logger.bind(component="place_registry")

    @property
    def place_count(self) -> int:
        return len(self._places)

    def add_place(self, place: Place) -> None:
        """Add a place to the registry."""
        self._places[place.id] = place

        # Index by name and aliases
        self._by_name.setdefault(place.name.lower(), set()).add(place.id)
        for alias in place.aliases:
            self._by_name.setdefault(alias.lower(), set()).add(place.id)

        # Index by type
        self._by_type.setdefault(place.place_type, set()).add(place.id)

        # Index children
        if place.parent_id:
            self._children.setdefault(place.parent_id, set()).add(place.id)

        self._log.debug("place_added", place_id=place.id, name=place.name)

    def get_place(self, place_id: str) -> Place | None:
        """Get a place by ID."""
        return self._places.get(place_id)

    def find_by_name(self, name: str) -> list[Place]:
        """Find places by name or alias."""
        place_ids = self._by_name.get(name.lower(), set())
        return [self._places[pid] for pid in place_ids if pid in self._places]

    def find_by_type(self, place_type: PlaceType) -> list[Place]:
        """Find places by type."""
        place_ids = self._by_type.get(place_type, set())
        return [self._places[pid] for pid in place_ids if pid in self._places]

    def get_children(self, parent_id: str) -> list[Place]:
        """Get child places of a parent."""
        child_ids = self._children.get(parent_id, set())
        return [self._places[cid] for cid in child_ids if cid in self._places]

    def get_ancestors(self, place_id: str) -> list[Place]:
        """Get ancestor places (parent chain)."""
        ancestors: list[Place] = []
        place = self._places.get(place_id)
        while place and place.parent_id:
            parent = self._places.get(place.parent_id)
            if parent:
                ancestors.append(parent)
                place = parent
            else:
                break
        return ancestors

    def update_place(self, place: Place) -> bool:
        """Update a place."""
        if place.id not in self._places:
            return False

        old_place = self._places[place.id]

        # Remove old indexes
        self._by_name[old_place.name.lower()].discard(place.id)
        for alias in old_place.aliases:
            if alias.lower() in self._by_name:
                self._by_name[alias.lower()].discard(place.id)
        if old_place.parent_id:
            self._children[old_place.parent_id].discard(place.id)

        # Add new place and indexes
        self.add_place(place)
        return True

    def remove_place(self, place_id: str) -> bool:
        """Remove a place."""
        place = self._places.pop(place_id, None)
        if not place:
            return False

        # Remove from indexes
        self._by_name[place.name.lower()].discard(place_id)
        for alias in place.aliases:
            if alias.lower() in self._by_name:
                self._by_name[alias.lower()].discard(place_id)
        self._by_type[place.place_type].discard(place_id)
        if place.parent_id:
            self._children[place.parent_id].discard(place_id)

        return True

    def search(
        self,
        query: str,
        place_type: PlaceType | None = None,
        limit: int = 10,
    ) -> list[Place]:
        """Search for places."""
        query_lower = query.lower()
        results: list[tuple[int, Place]] = []

        for place in self._places.values():
            if place_type and place.place_type != place_type:
                continue

            score = 0
            if place.name.lower() == query_lower:
                score = 100
            elif query_lower in place.name.lower():
                score = 50
            elif any(query_lower in alias.lower() for alias in place.aliases):
                score = 40
            elif query_lower in place.description.lower():
                score = 20

            if score > 0:
                results.append((score, place))

        results.sort(key=lambda x: -x[0])
        return [place for _, place in results[:limit]]


class SpatialResolver:
    """Resolves spatial references to places."""

    def __init__(self, registry: PlaceRegistry) -> None:
        self._registry = registry
        self._links: dict[str, SpatialLink] = {}
        self._by_source: dict[str, set[str]] = {}  # source_id -> link IDs
        self._by_target: dict[str, set[str]] = {}  # target_id -> link IDs
        self._log = logger.bind(component="spatial_resolver")

    @property
    def link_count(self) -> int:
        return len(self._links)

    def add_link(self, link: SpatialLink) -> None:
        """Add a spatial relationship."""
        self._links[link.id] = link
        self._by_source.setdefault(link.source_id, set()).add(link.id)
        self._by_target.setdefault(link.target_id, set()).add(link.id)

        # If bidirectional, also index reverse
        if link.bidirectional:
            self._by_source.setdefault(link.target_id, set()).add(link.id)
            self._by_target.setdefault(link.source_id, set()).add(link.id)

        self._log.debug("link_added", link_id=link.id, relation=link.relation.name)

    def get_related(
        self,
        place_id: str,
        relation: SpatialRelation | None = None,
    ) -> list[tuple[Place, SpatialLink]]:
        """Get related places."""
        results: list[tuple[Place, SpatialLink]] = []

        # Get outgoing links
        for link_id in self._by_source.get(place_id, set()):
            link = self._links.get(link_id)
            if not link:
                continue
            if relation and link.relation != relation:
                continue

            target_id = link.target_id if link.source_id == place_id else link.source_id
            target = self._registry.get_place(target_id)
            if target:
                results.append((target, link))

        return results

    def resolve(self, marker: SpatialMarker) -> ResolutionResult:
        """Resolve a spatial marker to a place."""
        identifier = marker.identifier

        # Try exact name match
        matches = self._registry.find_by_name(identifier)

        if len(matches) == 1:
            return ResolutionResult(
                marker=marker,
                status=ResolutionStatus.RESOLVED,
                place=matches[0],
                confidence=1.0,
            )
        elif len(matches) > 1:
            # Ambiguous - try to narrow down with context
            best_match = self._disambiguate(matches, marker.context)
            if best_match:
                return ResolutionResult(
                    marker=marker,
                    status=ResolutionStatus.RESOLVED,
                    place=best_match,
                    confidence=0.8,
                )
            return ResolutionResult(
                marker=marker,
                status=ResolutionStatus.AMBIGUOUS,
                candidates=matches,
                confidence=0.5,
            )

        # Try search
        search_results = self._registry.search(identifier, limit=5)
        if search_results:
            if len(search_results) == 1:
                return ResolutionResult(
                    marker=marker,
                    status=ResolutionStatus.RESOLVED,
                    place=search_results[0],
                    confidence=0.7,
                )
            return ResolutionResult(
                marker=marker,
                status=ResolutionStatus.AMBIGUOUS,
                candidates=search_results,
                confidence=0.4,
            )

        # Unresolved
        return ResolutionResult(
            marker=marker,
            status=ResolutionStatus.UNRESOLVED,
            confidence=0.0,
            error=f"No place found for '{identifier}'",
        )

    def _disambiguate(
        self,
        candidates: list[Place],
        context: dict[str, Any],
    ) -> Place | None:
        """Try to disambiguate between candidates using context."""
        if not context:
            return None

        # Check for type hint
        if type_hint := context.get("place_type"):
            try:
                target_type = PlaceType[type_hint.upper()]
                typed = [p for p in candidates if p.place_type == target_type]
                if len(typed) == 1:
                    return typed[0]
            except (KeyError, AttributeError):
                pass

        # Check for parent hint
        if parent_name := context.get("parent"):
            parents = self._registry.find_by_name(parent_name)
            if parents:
                parent_ids = {p.id for p in parents}
                children = [p for p in candidates if p.parent_id in parent_ids]
                if len(children) == 1:
                    return children[0]

        return None

    def compute_path(
        self,
        from_id: str,
        to_id: str,
        max_depth: int = 10,
    ) -> list[Place] | None:
        """Compute a path between two places."""
        if from_id == to_id:
            place = self._registry.get_place(from_id)
            return [place] if place else None

        visited: set[str] = set()
        queue: list[tuple[str, list[str]]] = [(from_id, [from_id])]

        while queue and len(visited) < 1000:
            current_id, path = queue.pop(0)

            if len(path) > max_depth:
                continue

            if current_id in visited:
                continue
            visited.add(current_id)

            # Check all related places
            for place, _link in self.get_related(current_id):
                if place.id == to_id:
                    full_path = path + [place.id]
                    return [
                        p
                        for pid in full_path
                        if (p := self._registry.get_place(pid)) is not None
                    ]

                if place.id not in visited:
                    queue.append((place.id, path + [place.id]))

        return None


class LocationResolver(Subsystem):
    """
    Resolves spatial references and manages symbolic places.

    Process Loop:
    1. Collect: Receive place identifiers or spatial markers
    2. Resolve: Match identifiers against the spatial ontology
    3. Assign: Link resolved locations to entities or events
    4. Archive: Store location definitions for reuse and lookup
    """

    def __init__(self) -> None:
        metadata = SubsystemMetadata(
            name="location_resolver",
            display_name="Location Resolver",
            description="Resolves spatial references and manages symbolic places",
            type=SubsystemType.TEMPORAL,
            tags=frozenset(["location", "spatial", "geography", "places"]),
            input_types=frozenset(["LOCATION", "REFERENCE", "MARKER"]),
            output_types=frozenset(["LOCATION", "ASSIGNMENT"]),
            subscribed_topics=frozenset([
                str(SubsystemTopics.ALL),
                "location.#",
                "spatial.#",
            ]),
            published_topics=frozenset([
                "location.resolved",
                "location.created",
                "location.assigned",
            ]),
        )
        super().__init__(metadata)

        self._registry = PlaceRegistry()
        self._resolver = SpatialResolver(self._registry)
        self._assignments: dict[str, LocationAssignment] = {}
        self._assignments_by_entity: dict[str, set[str]] = {}
        self._assignments_by_place: dict[str, set[str]] = {}

    @property
    def place_count(self) -> int:
        return self._registry.place_count

    @property
    def link_count(self) -> int:
        return self._resolver.link_count

    @property
    def assignment_count(self) -> int:
        return len(self._assignments)

    async def intake(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> SymbolicInput:
        """Phase 1: Collect place identifiers and spatial markers."""
        if not input_data.values:
            self._log.debug("empty_input")
            return input_data

        self._log.debug("intake_complete", value_count=len(input_data.values))
        return input_data

    async def process(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> list[ResolutionResult | Place | LocationAssignment]:
        """Phase 2: Resolve markers and process place definitions."""
        results: list[ResolutionResult | Place | LocationAssignment] = []

        for value in input_data.values:
            content = value.content
            if not isinstance(content, dict):
                continue

            action = content.get("action", "resolve")

            if action == "create":
                place = self._create_place_from_value(value)
                if place:
                    self._registry.add_place(place)
                    results.append(place)

            elif action == "resolve":
                marker = self._create_marker_from_value(value)
                if marker:
                    result = self._resolver.resolve(marker)
                    results.append(result)

            elif action == "assign":
                assignment = self._create_assignment_from_value(value)
                if assignment:
                    self._add_assignment(assignment)
                    results.append(assignment)

            elif action == "link":
                link = self._create_link_from_value(value)
                if link:
                    self._resolver.add_link(link)

        return results

    async def evaluate(
        self,
        intermediate: list[ResolutionResult | Place | LocationAssignment],
        ctx: ProcessContext[dict[str, Any]],
    ) -> tuple[SymbolicOutput, bool]:
        """Phase 3: Create output from resolved locations."""
        values: list[SymbolicValue] = []

        for item in intermediate:
            if isinstance(item, ResolutionResult):
                value = self._result_to_value(item)
            elif isinstance(item, Place):
                value = self._place_to_value(item)
            elif isinstance(item, LocationAssignment):
                value = self._assignment_to_value(item)
            else:
                continue

            if value:
                values.append(value)

        output = self.create_output(
            values=values,
            input_id=ctx.metadata.get("input_id"),
        )

        return output, False

    async def integrate(
        self, output: SymbolicOutput, ctx: ProcessContext[dict[str, Any]]
    ) -> SymbolicInput | None:
        """Phase 4: Emit events for resolved locations."""
        if self._message_bus and output.values:
            for value in output.values:
                content = value.content
                if not isinstance(content, dict):
                    continue

                if "place_id" in content and "resolution_status" not in content:
                    await self.emit_event(
                        "location.created",
                        {"place_id": content.get("place_id"), "name": content.get("name")},
                    )
                elif "resolution_status" in content:
                    await self.emit_event(
                        "location.resolved",
                        {
                            "identifier": content.get("identifier"),
                            "status": content.get("resolution_status"),
                            "place_id": content.get("place_id"),
                        },
                    )
                elif "assignment_id" in content:
                    await self.emit_event(
                        "location.assigned",
                        {
                            "assignment_id": content.get("assignment_id"),
                            "entity_id": content.get("entity_id"),
                            "place_id": content.get("place_id"),
                        },
                    )

        return None

    def _create_place_from_value(self, value: SymbolicValue) -> Place | None:
        """Create a Place from a SymbolicValue."""
        content = value.content
        if not isinstance(content, dict):
            return None

        try:
            place_type_str = content.get("place_type", "PHYSICAL")
            try:
                place_type = PlaceType[place_type_str.upper()]
            except KeyError:
                place_type = PlaceType.PHYSICAL

            coordinates = None
            if coords := content.get("coordinates"):
                if isinstance(coords, (list, tuple)) and len(coords) >= 3:
                    coordinates = tuple(float(c) for c in coords[:3])

            return Place(
                id=content.get("id", str(ULID())),
                name=content.get("name", f"place_{value.id[:8]}"),
                place_type=place_type,
                description=content.get("description", ""),
                coordinates=coordinates,
                symbolic_position=content.get("symbolic_position"),
                parent_id=content.get("parent_id"),
                aliases=frozenset(content.get("aliases", [])),
                properties=content.get("properties", {}),
                tags=frozenset(content.get("tags", [])) | value.tags,
            )
        except Exception as e:
            self._log.warning("place_parse_failed", value_id=value.id, error=str(e))
            return None

    def _create_marker_from_value(self, value: SymbolicValue) -> SpatialMarker | None:
        """Create a SpatialMarker from a SymbolicValue."""
        content = value.content
        if not isinstance(content, dict):
            return None

        identifier = content.get("identifier") or content.get("name")
        if not identifier:
            return None

        return SpatialMarker(
            identifier=identifier,
            marker_type=content.get("marker_type", "name"),
            context=content.get("context", {}),
            source=content.get("source"),
        )

    def _create_assignment_from_value(self, value: SymbolicValue) -> LocationAssignment | None:
        """Create a LocationAssignment from a SymbolicValue."""
        content = value.content
        if not isinstance(content, dict):
            return None

        entity_id = content.get("entity_id")
        place_id = content.get("place_id")
        if not entity_id or not place_id:
            return None

        return LocationAssignment(
            entity_id=entity_id,
            entity_type=content.get("entity_type", "unknown"),
            place_id=place_id,
            properties=content.get("properties", {}),
        )

    def _create_link_from_value(self, value: SymbolicValue) -> SpatialLink | None:
        """Create a SpatialLink from a SymbolicValue."""
        content = value.content
        if not isinstance(content, dict):
            return None

        source_id = content.get("source_id")
        target_id = content.get("target_id")
        if not source_id or not target_id:
            return None

        try:
            relation_str = content.get("relation", "CONNECTED")
            try:
                relation = SpatialRelation[relation_str.upper()]
            except KeyError:
                relation = SpatialRelation.CONNECTED

            return SpatialLink(
                source_id=source_id,
                target_id=target_id,
                relation=relation,
                bidirectional=content.get("bidirectional", False),
                weight=content.get("weight", 1.0),
                properties=content.get("properties", {}),
            )
        except Exception as e:
            self._log.warning("link_parse_failed", value_id=value.id, error=str(e))
            return None

    def _result_to_value(self, result: ResolutionResult) -> SymbolicValue:
        """Convert ResolutionResult to SymbolicValue."""
        content: dict[str, Any] = {
            "identifier": result.marker.identifier,
            "resolution_status": result.status.name,
            "confidence": result.confidence,
        }

        if result.place:
            content["place_id"] = result.place.id
            content["place_name"] = result.place.name
            content["place_type"] = result.place.place_type.name

        if result.candidates:
            content["candidate_count"] = len(result.candidates)
            content["candidate_names"] = [c.name for c in result.candidates[:5]]

        if result.error:
            content["error"] = result.error

        return SymbolicValue(
            type=SymbolicValueType.LOCATION,
            content=content,
            source_subsystem=self.name,
            tags=frozenset(["resolution", result.status.name.lower()]),
            meaning=f"Resolved: {result.marker.identifier} -> {result.status.name}",
            confidence=result.confidence,
        )

    def _place_to_value(self, place: Place) -> SymbolicValue:
        """Convert Place to SymbolicValue."""
        return SymbolicValue(
            type=SymbolicValueType.LOCATION,
            content={
                "place_id": place.id,
                "name": place.name,
                "place_type": place.place_type.name,
                "description": place.description,
                "parent_id": place.parent_id,
                "aliases": list(place.aliases),
            },
            source_subsystem=self.name,
            tags=place.tags | frozenset(["place", place.place_type.name.lower()]),
            meaning=f"Place: {place.name}",
            confidence=1.0,
        )

    def _assignment_to_value(self, assignment: LocationAssignment) -> SymbolicValue:
        """Convert LocationAssignment to SymbolicValue."""
        return SymbolicValue(
            type=SymbolicValueType.REFERENCE,
            content={
                "assignment_id": assignment.id,
                "entity_id": assignment.entity_id,
                "entity_type": assignment.entity_type,
                "place_id": assignment.place_id,
                "assigned_at": assignment.assigned_at.isoformat(),
            },
            source_subsystem=self.name,
            tags=frozenset(["assignment", assignment.entity_type]),
            meaning=f"Assignment: {assignment.entity_id} -> {assignment.place_id}",
            confidence=1.0,
        )

    def _add_assignment(self, assignment: LocationAssignment) -> None:
        """Add an assignment to storage."""
        self._assignments[assignment.id] = assignment
        self._assignments_by_entity.setdefault(assignment.entity_id, set()).add(assignment.id)
        self._assignments_by_place.setdefault(assignment.place_id, set()).add(assignment.id)

    # --- Message handlers ---

    async def handle_event(self, message: Message) -> None:
        """Handle incoming events."""
        if message.topic.startswith("location."):
            self._log.debug("location_event_received", topic=message.topic)

    async def handle_signal(self, signal: Any) -> None:
        """Handle incoming signals."""
        self._log.debug("signal_received", signal_id=getattr(signal, "id", "unknown"))

    # --- Public API ---

    def create_place(
        self,
        name: str,
        place_type: PlaceType = PlaceType.PHYSICAL,
        **kwargs: Any,
    ) -> Place:
        """Create and register a new place."""
        coordinates = None
        if coords := kwargs.get("coordinates"):
            if isinstance(coords, (list, tuple)) and len(coords) >= 3:
                coordinates = tuple(float(c) for c in coords[:3])

        place = Place(
            name=name,
            place_type=place_type,
            description=kwargs.get("description", ""),
            coordinates=coordinates,
            symbolic_position=kwargs.get("symbolic_position"),
            parent_id=kwargs.get("parent_id"),
            aliases=frozenset(kwargs.get("aliases", [])),
            properties=kwargs.get("properties", {}),
            tags=frozenset(kwargs.get("tags", [])),
        )
        self._registry.add_place(place)
        return place

    def get_place(self, place_id: str) -> Place | None:
        """Get a place by ID."""
        return self._registry.get_place(place_id)

    def find_by_name(self, name: str) -> list[Place]:
        """Find places by name or alias."""
        return self._registry.find_by_name(name)

    def resolve(self, identifier: str, **context: Any) -> ResolutionResult:
        """Resolve a spatial identifier."""
        marker = SpatialMarker(identifier=identifier, context=context)
        return self._resolver.resolve(marker)

    def link_places(
        self,
        source_id: str,
        target_id: str,
        relation: SpatialRelation,
        bidirectional: bool = False,
        **kwargs: Any,
    ) -> SpatialLink:
        """Create a spatial link between places."""
        link = SpatialLink(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            bidirectional=bidirectional,
            weight=kwargs.get("weight", 1.0),
            properties=kwargs.get("properties", {}),
        )
        self._resolver.add_link(link)
        return link

    def get_related(
        self,
        place_id: str,
        relation: SpatialRelation | None = None,
    ) -> list[tuple[Place, SpatialLink]]:
        """Get places related to a given place."""
        return self._resolver.get_related(place_id, relation)

    def assign_entity(
        self,
        entity_id: str,
        place_id: str,
        entity_type: str = "unknown",
        **kwargs: Any,
    ) -> LocationAssignment:
        """Assign an entity to a place."""
        assignment = LocationAssignment(
            entity_id=entity_id,
            entity_type=entity_type,
            place_id=place_id,
            properties=kwargs.get("properties", {}),
        )
        self._add_assignment(assignment)
        return assignment

    def get_entity_locations(self, entity_id: str) -> list[Place]:
        """Get all places an entity is assigned to."""
        assignment_ids = self._assignments_by_entity.get(entity_id, set())
        places: list[Place] = []
        for aid in assignment_ids:
            assignment = self._assignments.get(aid)
            if assignment:
                place = self._registry.get_place(assignment.place_id)
                if place:
                    places.append(place)
        return places

    def get_entities_at_place(self, place_id: str) -> list[LocationAssignment]:
        """Get all entities at a place."""
        assignment_ids = self._assignments_by_place.get(place_id, set())
        return [self._assignments[aid] for aid in assignment_ids if aid in self._assignments]

    def find_path(self, from_id: str, to_id: str) -> list[Place] | None:
        """Find a path between two places."""
        return self._resolver.compute_path(from_id, to_id)

    def search_places(
        self,
        query: str,
        place_type: PlaceType | None = None,
        limit: int = 10,
    ) -> list[Place]:
        """Search for places."""
        return self._registry.search(query, place_type, limit)

    def get_stats(self) -> LocationStats:
        """Get location system statistics."""
        by_type: dict[str, int] = {}
        for pt in PlaceType:
            count = len(self._registry.find_by_type(pt))
            if count > 0:
                by_type[pt.name] = count

        return LocationStats(
            total_places=self._registry.place_count,
            places_by_type=by_type,
            total_links=self._resolver.link_count,
            total_assignments=len(self._assignments),
        )

    def clear(self) -> tuple[int, int, int]:
        """Clear all data. Returns (places, links, assignments) cleared."""
        places = self._registry.place_count
        links = self._resolver.link_count
        assignments = len(self._assignments)

        self._registry = PlaceRegistry()
        self._resolver = SpatialResolver(self._registry)
        self._assignments.clear()
        self._assignments_by_entity.clear()
        self._assignments_by_place.clear()

        return places, links, assignments
