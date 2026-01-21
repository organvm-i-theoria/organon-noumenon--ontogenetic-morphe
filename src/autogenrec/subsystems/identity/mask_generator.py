"""
MaskGenerator: Creates symbolic identity masks.

Generates and manages symbolic masks that abstract identity, represent roles
or states, and help maintain privacy within the system.
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
from autogenrec.core.signals import Message
from autogenrec.core.subsystem import Subsystem, SubsystemMetadata, SubsystemType
from autogenrec.core.symbolic import (
    SymbolicInput,
    SymbolicOutput,
    SymbolicValue,
    SymbolicValueType,
)

logger = structlog.get_logger()


class MaskType(Enum):
    """Types of symbolic masks."""

    IDENTITY = auto()  # Full identity mask
    ROLE = auto()  # Role-based mask
    TEMPORAL = auto()  # Time-limited mask
    COMPOSITE = auto()  # Composed from multiple masks
    ANONYMOUS = auto()  # Privacy-preserving mask
    PSEUDONYMOUS = auto()  # Consistent but unlinkable


class MaskState(Enum):
    """State of a mask."""

    ACTIVE = auto()  # Currently in use
    SUSPENDED = auto()  # Temporarily inactive
    EXPIRED = auto()  # Past valid period
    REVOKED = auto()  # Permanently disabled


class Mask(BaseModel):
    """A symbolic identity mask."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    name: str
    mask_type: MaskType = MaskType.IDENTITY
    state: MaskState = MaskState.ACTIVE

    # Identity attributes encoded in mask
    attributes: frozenset[str] = Field(default_factory=frozenset)
    roles: frozenset[str] = Field(default_factory=frozenset)

    # Temporal properties
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    valid_from: datetime | None = None
    valid_until: datetime | None = None

    # Composition (for COMPOSITE type)
    parent_mask_ids: frozenset[str] = Field(default_factory=frozenset)

    # Privacy level (0 = transparent, 1 = fully opaque)
    opacity: float = 0.5

    # Metadata
    description: str = ""
    entity_id: str | None = None  # The entity this mask represents
    tags: frozenset[str] = Field(default_factory=frozenset)


class MaskAssignment(BaseModel):
    """Assignment of a mask to an entity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    mask_id: str
    entity_id: str
    entity_type: str = "user"
    assigned_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    assigned_by: str | None = None
    context: str = ""  # Context for this assignment
    is_primary: bool = False


class MaskLayer(BaseModel):
    """A layer in a mask composition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mask_id: str
    layer_order: int
    visibility: float = 1.0  # How visible this layer is
    filter_attributes: frozenset[str] = Field(default_factory=frozenset)


@dataclass
class GenerationResult:
    """Result of mask generation."""

    mask_id: str
    success: bool
    mask: Mask | None = None
    error: str | None = None


@dataclass
class AssignmentResult:
    """Result of mask assignment."""

    assignment_id: str
    success: bool
    assignment: MaskAssignment | None = None
    error: str | None = None


@dataclass
class MaskStats:
    """Statistics about masks."""

    total_masks: int
    active_masks: int
    expired_masks: int
    total_assignments: int
    masks_by_type: dict[str, int]


class MaskRegistry:
    """Registry of masks."""

    def __init__(self) -> None:
        self._masks: dict[str, Mask] = {}
        self._by_entity: dict[str, set[str]] = {}  # entity_id -> mask IDs
        self._by_type: dict[MaskType, set[str]] = {}
        self._assignments: dict[str, MaskAssignment] = {}
        self._entity_assignments: dict[str, set[str]] = {}  # entity_id -> assignment IDs
        self._log = logger.bind(component="mask_registry")

    @property
    def mask_count(self) -> int:
        return len(self._masks)

    @property
    def assignment_count(self) -> int:
        return len(self._assignments)

    def register_mask(self, mask: Mask) -> None:
        """Register a new mask."""
        self._masks[mask.id] = mask
        self._by_type.setdefault(mask.mask_type, set()).add(mask.id)
        if mask.entity_id:
            self._by_entity.setdefault(mask.entity_id, set()).add(mask.id)
        self._log.debug("mask_registered", mask_id=mask.id, mask_type=mask.mask_type.name)

    def get_mask(self, mask_id: str) -> Mask | None:
        """Get a mask by ID."""
        return self._masks.get(mask_id)

    def get_by_entity(self, entity_id: str) -> list[Mask]:
        """Get masks for an entity."""
        mask_ids = self._by_entity.get(entity_id, set())
        return [self._masks[mid] for mid in mask_ids if mid in self._masks]

    def get_by_type(self, mask_type: MaskType) -> list[Mask]:
        """Get masks by type."""
        mask_ids = self._by_type.get(mask_type, set())
        return [self._masks[mid] for mid in mask_ids if mid in self._masks]

    def update_mask(self, mask_id: str, **updates: Any) -> Mask | None:
        """Update a mask's properties."""
        mask = self._masks.get(mask_id)
        if not mask:
            return None

        # Build new mask with updates
        data = {
            "id": mask.id,
            "name": updates.get("name", mask.name),
            "mask_type": updates.get("mask_type", mask.mask_type),
            "state": updates.get("state", mask.state),
            "attributes": updates.get("attributes", mask.attributes),
            "roles": updates.get("roles", mask.roles),
            "created_at": mask.created_at,
            "valid_from": updates.get("valid_from", mask.valid_from),
            "valid_until": updates.get("valid_until", mask.valid_until),
            "parent_mask_ids": updates.get("parent_mask_ids", mask.parent_mask_ids),
            "opacity": updates.get("opacity", mask.opacity),
            "description": updates.get("description", mask.description),
            "entity_id": updates.get("entity_id", mask.entity_id),
            "tags": updates.get("tags", mask.tags),
        }
        updated = Mask(**data)
        self._masks[mask_id] = updated
        return updated

    def assign_mask(self, assignment: MaskAssignment) -> None:
        """Assign a mask to an entity."""
        self._assignments[assignment.id] = assignment
        self._entity_assignments.setdefault(assignment.entity_id, set()).add(assignment.id)
        self._log.debug(
            "mask_assigned",
            assignment_id=assignment.id,
            mask_id=assignment.mask_id,
            entity_id=assignment.entity_id,
        )

    def get_assignment(self, assignment_id: str) -> MaskAssignment | None:
        """Get an assignment by ID."""
        return self._assignments.get(assignment_id)

    def get_entity_assignments(self, entity_id: str) -> list[MaskAssignment]:
        """Get all assignments for an entity."""
        assignment_ids = self._entity_assignments.get(entity_id, set())
        return [self._assignments[aid] for aid in assignment_ids if aid in self._assignments]

    def revoke_assignment(self, assignment_id: str) -> bool:
        """Revoke a mask assignment."""
        assignment = self._assignments.get(assignment_id)
        if not assignment:
            return False
        del self._assignments[assignment_id]
        if assignment.entity_id in self._entity_assignments:
            self._entity_assignments[assignment.entity_id].discard(assignment_id)
        return True

    def get_active_masks(self) -> list[Mask]:
        """Get all active masks."""
        now = datetime.now(UTC)
        active = []
        for mask in self._masks.values():
            if mask.state != MaskState.ACTIVE:
                continue
            if mask.valid_until and mask.valid_until < now:
                continue
            if mask.valid_from and mask.valid_from > now:
                continue
            active.append(mask)
        return active

    def get_expired_masks(self) -> list[Mask]:
        """Get all expired masks."""
        now = datetime.now(UTC)
        return [
            m for m in self._masks.values()
            if m.valid_until and m.valid_until < now
        ]


class MaskComposer:
    """Composes masks from layers."""

    def __init__(self, registry: MaskRegistry) -> None:
        self._registry = registry
        self._log = logger.bind(component="mask_composer")

    def compose(
        self,
        name: str,
        layers: list[MaskLayer],
        **kwargs: Any,
    ) -> Mask | None:
        """Compose a new mask from layers."""
        if not layers:
            return None

        # Gather parent masks
        parent_ids = set()
        combined_attributes: set[str] = set()
        combined_roles: set[str] = set()
        max_opacity = 0.0

        for layer in sorted(layers, key=lambda l: l.layer_order):
            parent_mask = self._registry.get_mask(layer.mask_id)
            if not parent_mask:
                continue

            parent_ids.add(parent_mask.id)

            # Apply visibility filtering
            if layer.filter_attributes:
                # Only include specified attributes
                for attr in layer.filter_attributes:
                    if attr in parent_mask.attributes:
                        combined_attributes.add(attr)
            else:
                combined_attributes.update(parent_mask.attributes)

            combined_roles.update(parent_mask.roles)
            max_opacity = max(max_opacity, parent_mask.opacity * layer.visibility)

        # Create composite mask
        return Mask(
            name=name,
            mask_type=MaskType.COMPOSITE,
            attributes=frozenset(combined_attributes),
            roles=frozenset(combined_roles),
            parent_mask_ids=frozenset(parent_ids),
            opacity=min(1.0, max_opacity),
            description=kwargs.get("description", f"Composite of {len(parent_ids)} masks"),
            entity_id=kwargs.get("entity_id"),
            tags=frozenset(kwargs.get("tags", [])),
        )


class MaskGenerator(Subsystem):
    """
    Creates and assigns symbolic masks.

    Process Loop:
    1. Collect: Gather identity attributes and parameters
    2. Generate: Create symbolic masks based on input
    3. Assign: Link masks to entities, roles, or processes
    4. Update: Modify or reassign masks as identities evolve
    """

    def __init__(self) -> None:
        metadata = SubsystemMetadata(
            name="mask_generator",
            display_name="Mask Generator",
            description="Creates symbolic identity masks",
            type=SubsystemType.IDENTITY,
            tags=frozenset(["mask", "identity", "symbolic", "privacy"]),
            input_types=frozenset(["IDENTITY", "ROLE", "PATTERN"]),
            output_types=frozenset(["MASK", "IDENTITY", "ASSIGNMENT"]),
            subscribed_topics=frozenset([
                str(SubsystemTopics.ALL),
                "identity.#",
                "mask.#",
            ]),
            published_topics=frozenset([
                "mask.created",
                "mask.assigned",
                "mask.revoked",
                "mask.expired",
            ]),
        )
        super().__init__(metadata)

        self._registry = MaskRegistry()
        self._composer = MaskComposer(self._registry)

    @property
    def mask_count(self) -> int:
        return self._registry.mask_count

    @property
    def assignment_count(self) -> int:
        return self._registry.assignment_count

    async def intake(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> SymbolicInput:
        """Phase 1: Collect identity attributes and parameters."""
        if not input_data.values:
            self._log.debug("empty_input")
            return input_data

        self._log.debug("intake_complete", value_count=len(input_data.values))
        return input_data

    async def process(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> list[GenerationResult | AssignmentResult]:
        """Phase 2: Generate and assign masks."""
        results: list[GenerationResult | AssignmentResult] = []

        for value in input_data.values:
            content = value.content
            if not isinstance(content, dict):
                continue

            action = content.get("action", "generate")

            if action == "generate":
                results.append(self._generate_from_value(value))
            elif action == "assign":
                results.append(self._assign_from_value(value))
            elif action == "compose":
                results.append(self._compose_from_value(value))

        return results

    async def evaluate(
        self,
        intermediate: list[GenerationResult | AssignmentResult],
        ctx: ProcessContext[dict[str, Any]],
    ) -> tuple[SymbolicOutput, bool]:
        """Phase 3: Create output from results."""
        values: list[SymbolicValue] = []

        for result in intermediate:
            if isinstance(result, GenerationResult):
                value = SymbolicValue(
                    type=SymbolicValueType.PATTERN,
                    content={
                        "mask_id": result.mask_id,
                        "success": result.success,
                        "mask_name": result.mask.name if result.mask else None,
                        "mask_type": result.mask.mask_type.name if result.mask else None,
                        "error": result.error,
                    },
                    source_subsystem=self.name,
                    tags=frozenset(["mask", "generated"]),
                    meaning="Mask generation result",
                    confidence=1.0 if result.success else 0.0,
                )
            else:  # AssignmentResult
                value = SymbolicValue(
                    type=SymbolicValueType.REFERENCE,
                    content={
                        "assignment_id": result.assignment_id,
                        "success": result.success,
                        "error": result.error,
                    },
                    source_subsystem=self.name,
                    tags=frozenset(["mask", "assignment"]),
                    meaning="Mask assignment result",
                    confidence=1.0 if result.success else 0.0,
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
        """Phase 4: Emit events for mask operations."""
        if self._message_bus and output.values:
            for value in output.values:
                content = value.content
                if not isinstance(content, dict):
                    continue

                if "mask_id" in content and content.get("success"):
                    await self.emit_event("mask.created", {"mask_id": content["mask_id"]})
                elif "assignment_id" in content and content.get("success"):
                    await self.emit_event("mask.assigned", {"assignment_id": content["assignment_id"]})

        return None

    def _generate_from_value(self, value: SymbolicValue) -> GenerationResult:
        """Generate a mask from a SymbolicValue."""
        content = value.content
        if not isinstance(content, dict):
            return GenerationResult(mask_id="", success=False, error="Invalid content")

        try:
            type_str = content.get("mask_type", "IDENTITY")
            try:
                mask_type = MaskType[type_str.upper()]
            except KeyError:
                mask_type = MaskType.IDENTITY

            # Parse temporal properties
            valid_from = None
            valid_until = None
            if vf := content.get("valid_from"):
                valid_from = datetime.fromisoformat(vf) if isinstance(vf, str) else vf
            if vu := content.get("valid_until"):
                valid_until = datetime.fromisoformat(vu) if isinstance(vu, str) else vu

            # For temporal masks, default to 24h validity
            if mask_type == MaskType.TEMPORAL and not valid_until:
                valid_until = datetime.now(UTC) + timedelta(hours=24)

            mask = Mask(
                name=content.get("name", f"mask_{value.id[:8]}"),
                mask_type=mask_type,
                attributes=frozenset(content.get("attributes", [])),
                roles=frozenset(content.get("roles", [])),
                valid_from=valid_from,
                valid_until=valid_until,
                opacity=float(content.get("opacity", 0.5)),
                description=content.get("description", ""),
                entity_id=content.get("entity_id"),
                tags=frozenset(content.get("tags", [])) | value.tags,
            )
            self._registry.register_mask(mask)
            return GenerationResult(mask_id=mask.id, success=True, mask=mask)

        except Exception as e:
            self._log.warning("generation_failed", value_id=value.id, error=str(e))
            return GenerationResult(mask_id="", success=False, error=str(e))

    def _assign_from_value(self, value: SymbolicValue) -> AssignmentResult:
        """Create a mask assignment from a SymbolicValue."""
        content = value.content
        if not isinstance(content, dict):
            return AssignmentResult(assignment_id="", success=False, error="Invalid content")

        try:
            mask_id = content.get("mask_id")
            entity_id = content.get("entity_id")

            if not mask_id or not entity_id:
                return AssignmentResult(
                    assignment_id="",
                    success=False,
                    error="mask_id and entity_id required",
                )

            # Verify mask exists
            if not self._registry.get_mask(mask_id):
                return AssignmentResult(
                    assignment_id="",
                    success=False,
                    error=f"Mask {mask_id} not found",
                )

            assignment = MaskAssignment(
                mask_id=mask_id,
                entity_id=entity_id,
                entity_type=content.get("entity_type", "user"),
                assigned_by=content.get("assigned_by"),
                context=content.get("context", ""),
                is_primary=content.get("is_primary", False),
            )
            self._registry.assign_mask(assignment)
            return AssignmentResult(
                assignment_id=assignment.id,
                success=True,
                assignment=assignment,
            )

        except Exception as e:
            self._log.warning("assignment_failed", value_id=value.id, error=str(e))
            return AssignmentResult(assignment_id="", success=False, error=str(e))

    def _compose_from_value(self, value: SymbolicValue) -> GenerationResult:
        """Compose a mask from layers."""
        content = value.content
        if not isinstance(content, dict):
            return GenerationResult(mask_id="", success=False, error="Invalid content")

        try:
            layers_data = content.get("layers", [])
            layers = []
            for i, layer_data in enumerate(layers_data):
                if isinstance(layer_data, dict):
                    layers.append(MaskLayer(
                        mask_id=layer_data.get("mask_id", ""),
                        layer_order=layer_data.get("layer_order", i),
                        visibility=float(layer_data.get("visibility", 1.0)),
                        filter_attributes=frozenset(layer_data.get("filter_attributes", [])),
                    ))

            mask = self._composer.compose(
                name=content.get("name", f"composite_{value.id[:8]}"),
                layers=layers,
                description=content.get("description"),
                entity_id=content.get("entity_id"),
                tags=content.get("tags", []),
            )

            if mask:
                self._registry.register_mask(mask)
                return GenerationResult(mask_id=mask.id, success=True, mask=mask)
            else:
                return GenerationResult(mask_id="", success=False, error="Composition failed")

        except Exception as e:
            self._log.warning("compose_failed", value_id=value.id, error=str(e))
            return GenerationResult(mask_id="", success=False, error=str(e))

    # --- Message handlers ---

    async def handle_event(self, message: Message) -> None:
        """Handle incoming events."""
        if message.topic.startswith("mask.") or message.topic.startswith("identity."):
            self._log.debug("event_received", topic=message.topic)

    async def handle_signal(self, signal: Any) -> None:
        """Handle incoming signals."""
        self._log.debug("signal_received", signal_id=getattr(signal, "id", "unknown"))

    # --- Public API ---

    def generate_mask(
        self,
        name: str,
        mask_type: MaskType = MaskType.IDENTITY,
        **kwargs: Any,
    ) -> Mask:
        """Generate a new mask."""
        mask = Mask(
            name=name,
            mask_type=mask_type,
            attributes=frozenset(kwargs.get("attributes", [])),
            roles=frozenset(kwargs.get("roles", [])),
            valid_from=kwargs.get("valid_from"),
            valid_until=kwargs.get("valid_until"),
            opacity=float(kwargs.get("opacity", 0.5)),
            description=kwargs.get("description", ""),
            entity_id=kwargs.get("entity_id"),
            tags=frozenset(kwargs.get("tags", [])),
        )
        self._registry.register_mask(mask)
        return mask

    def generate_temporal_mask(
        self,
        name: str,
        duration: timedelta | None = None,
        **kwargs: Any,
    ) -> Mask:
        """Generate a time-limited mask."""
        now = datetime.now(UTC)
        valid_until = now + (duration or timedelta(hours=24))

        return self.generate_mask(
            name=name,
            mask_type=MaskType.TEMPORAL,
            valid_from=now,
            valid_until=valid_until,
            **kwargs,
        )

    def generate_anonymous_mask(
        self,
        **kwargs: Any,
    ) -> Mask:
        """Generate an anonymous mask for privacy."""
        return self.generate_mask(
            name=f"anon_{str(ULID())[:8]}",
            mask_type=MaskType.ANONYMOUS,
            opacity=1.0,  # Fully opaque
            **kwargs,
        )

    def get_mask(self, mask_id: str) -> Mask | None:
        """Get a mask by ID."""
        return self._registry.get_mask(mask_id)

    def get_masks_for_entity(self, entity_id: str) -> list[Mask]:
        """Get all masks for an entity."""
        return self._registry.get_by_entity(entity_id)

    def assign_mask(
        self,
        mask_id: str,
        entity_id: str,
        **kwargs: Any,
    ) -> MaskAssignment | None:
        """Assign a mask to an entity."""
        mask = self._registry.get_mask(mask_id)
        if not mask:
            return None

        assignment = MaskAssignment(
            mask_id=mask_id,
            entity_id=entity_id,
            entity_type=kwargs.get("entity_type", "user"),
            assigned_by=kwargs.get("assigned_by"),
            context=kwargs.get("context", ""),
            is_primary=kwargs.get("is_primary", False),
        )
        self._registry.assign_mask(assignment)
        return assignment

    def get_entity_assignments(self, entity_id: str) -> list[MaskAssignment]:
        """Get all mask assignments for an entity."""
        return self._registry.get_entity_assignments(entity_id)

    def revoke_assignment(self, assignment_id: str) -> bool:
        """Revoke a mask assignment."""
        return self._registry.revoke_assignment(assignment_id)

    def compose_mask(
        self,
        name: str,
        mask_ids: list[str],
        **kwargs: Any,
    ) -> Mask | None:
        """Compose a new mask from existing masks."""
        layers = [
            MaskLayer(mask_id=mid, layer_order=i)
            for i, mid in enumerate(mask_ids)
        ]
        mask = self._composer.compose(name, layers, **kwargs)
        if mask:
            self._registry.register_mask(mask)
        return mask

    def update_mask_state(self, mask_id: str, state: MaskState) -> Mask | None:
        """Update a mask's state."""
        return self._registry.update_mask(mask_id, state=state)

    def suspend_mask(self, mask_id: str) -> Mask | None:
        """Suspend a mask."""
        return self.update_mask_state(mask_id, MaskState.SUSPENDED)

    def revoke_mask(self, mask_id: str) -> Mask | None:
        """Revoke a mask permanently."""
        return self.update_mask_state(mask_id, MaskState.REVOKED)

    def get_active_masks(self) -> list[Mask]:
        """Get all currently active masks."""
        return self._registry.get_active_masks()

    def check_expired_masks(self) -> list[Mask]:
        """Get masks that have expired."""
        expired = self._registry.get_expired_masks()
        # Update their state
        for mask in expired:
            if mask.state == MaskState.ACTIVE:
                self._registry.update_mask(mask.id, state=MaskState.EXPIRED)
        return expired

    def get_stats(self) -> MaskStats:
        """Get mask statistics."""
        masks_by_type: dict[str, int] = {}
        active_count = 0
        expired_count = 0

        for mask in self._registry._masks.values():
            type_name = mask.mask_type.name
            masks_by_type[type_name] = masks_by_type.get(type_name, 0) + 1

            if mask.state == MaskState.ACTIVE:
                active_count += 1
            elif mask.state == MaskState.EXPIRED:
                expired_count += 1

        return MaskStats(
            total_masks=self._registry.mask_count,
            active_masks=active_count,
            expired_masks=expired_count,
            total_assignments=self._registry.assignment_count,
            masks_by_type=masks_by_type,
        )

    def clear(self) -> tuple[int, int]:
        """Clear all data. Returns (masks, assignments) cleared."""
        masks = self._registry.mask_count
        assignments = self._registry.assignment_count
        self._registry = MaskRegistry()
        self._composer = MaskComposer(self._registry)
        return masks, assignments
