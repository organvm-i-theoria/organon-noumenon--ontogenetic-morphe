"""
AudienceClassifier: Categorizes users into segments.

Analyzes and classifies audience members for targeted processing,
segmentation, and access control.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
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


class SegmentType(Enum):
    """Types of audience segments."""

    TIER = auto()  # Hierarchical access tier
    BEHAVIORAL = auto()  # Based on behavior patterns
    DEMOGRAPHIC = auto()  # Based on attributes
    CONTEXTUAL = auto()  # Based on context/situation
    CUSTOM = auto()  # User-defined


class AccessLevel(Enum):
    """Access levels for segments."""

    NONE = 0
    BASIC = 1
    STANDARD = 2
    PREMIUM = 3
    VIP = 4
    ADMIN = 5


class RuleOperator(Enum):
    """Operators for classification rules."""

    EQUALS = auto()
    NOT_EQUALS = auto()
    GREATER_THAN = auto()
    LESS_THAN = auto()
    CONTAINS = auto()
    NOT_CONTAINS = auto()
    IN = auto()
    NOT_IN = auto()
    MATCHES = auto()  # Regex match
    EXISTS = auto()


class Segment(BaseModel):
    """An audience segment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    name: str
    segment_type: SegmentType = SegmentType.TIER
    description: str = ""

    # Access control
    access_level: AccessLevel = AccessLevel.STANDARD

    # Segment properties
    priority: int = 0  # Higher = evaluated first
    is_default: bool = False  # Catch-all segment
    is_exclusive: bool = False  # User can only be in one exclusive segment
    max_members: int | None = None  # Optional member limit

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tags: frozenset[str] = Field(default_factory=frozenset)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClassificationRule(BaseModel):
    """A rule for classifying users into segments."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    name: str
    segment_id: str
    priority: int = 0  # Higher = evaluated first

    # Rule conditions
    attribute: str  # The attribute to check
    operator: RuleOperator = RuleOperator.EQUALS
    value: Any = None  # Comparison value

    # Optional weight for scoring
    weight: float = 1.0

    # Active state
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AudienceMember(BaseModel):
    """An audience member (user/participant)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    external_id: str | None = None  # ID from external system
    name: str = ""

    # Attributes for classification
    attributes: dict[str, Any] = Field(default_factory=dict)

    # Interaction metrics
    interaction_count: int = 0
    last_interaction: datetime | None = None
    engagement_score: float = 0.0

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tags: frozenset[str] = Field(default_factory=frozenset)


class SegmentMembership(BaseModel):
    """Membership of a user in a segment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    member_id: str
    segment_id: str
    assigned_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    assigned_by: str = "classifier"  # "classifier", "manual", "rule:{rule_id}"
    confidence: float = 1.0  # Classification confidence
    is_active: bool = True


@dataclass
class ClassificationResult:
    """Result of classifying a member."""

    member_id: str
    success: bool
    segments: list[str] = field(default_factory=list)  # Segment IDs
    confidences: dict[str, float] = field(default_factory=dict)  # segment_id -> confidence
    error: str | None = None


@dataclass
class SegmentStats:
    """Statistics about segments."""

    total_segments: int
    total_members: int
    total_memberships: int
    members_by_segment: dict[str, int]
    segments_by_type: dict[str, int]


class SegmentRegistry:
    """Registry of segments."""

    def __init__(self) -> None:
        self._segments: dict[str, Segment] = {}
        self._by_type: dict[SegmentType, set[str]] = {}
        self._log = logger.bind(component="segment_registry")

    @property
    def segment_count(self) -> int:
        return len(self._segments)

    def register_segment(self, segment: Segment) -> None:
        """Register a segment."""
        self._segments[segment.id] = segment
        self._by_type.setdefault(segment.segment_type, set()).add(segment.id)
        self._log.debug("segment_registered", segment_id=segment.id, name=segment.name)

    def get_segment(self, segment_id: str) -> Segment | None:
        """Get a segment by ID."""
        return self._segments.get(segment_id)

    def get_by_name(self, name: str) -> Segment | None:
        """Get a segment by name."""
        for segment in self._segments.values():
            if segment.name == name:
                return segment
        return None

    def get_by_type(self, segment_type: SegmentType) -> list[Segment]:
        """Get segments by type."""
        segment_ids = self._by_type.get(segment_type, set())
        return [self._segments[sid] for sid in segment_ids if sid in self._segments]

    def get_all(self) -> list[Segment]:
        """Get all segments sorted by priority."""
        return sorted(self._segments.values(), key=lambda s: -s.priority)

    def get_default_segment(self) -> Segment | None:
        """Get the default segment."""
        for segment in self._segments.values():
            if segment.is_default:
                return segment
        return None


class RuleEngine:
    """Engine for evaluating classification rules."""

    def __init__(self) -> None:
        self._rules: dict[str, ClassificationRule] = {}
        self._by_segment: dict[str, set[str]] = {}  # segment_id -> rule IDs
        self._log = logger.bind(component="rule_engine")

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    def add_rule(self, rule: ClassificationRule) -> None:
        """Add a classification rule."""
        self._rules[rule.id] = rule
        self._by_segment.setdefault(rule.segment_id, set()).add(rule.id)
        self._log.debug("rule_added", rule_id=rule.id, segment_id=rule.segment_id)

    def get_rule(self, rule_id: str) -> ClassificationRule | None:
        """Get a rule by ID."""
        return self._rules.get(rule_id)

    def get_rules_for_segment(self, segment_id: str) -> list[ClassificationRule]:
        """Get rules for a segment."""
        rule_ids = self._by_segment.get(segment_id, set())
        return [self._rules[rid] for rid in rule_ids if rid in self._rules]

    def evaluate_rule(self, rule: ClassificationRule, member: AudienceMember) -> tuple[bool, float]:
        """Evaluate a rule against a member. Returns (matches, confidence)."""
        if not rule.is_active:
            return False, 0.0

        # Get the attribute value
        attr_value = member.attributes.get(rule.attribute)

        # Handle EXISTS operator specially
        if rule.operator == RuleOperator.EXISTS:
            matches = attr_value is not None
            return matches, rule.weight if matches else 0.0

        if attr_value is None:
            return False, 0.0

        # Evaluate based on operator
        try:
            if rule.operator == RuleOperator.EQUALS:
                matches = attr_value == rule.value
            elif rule.operator == RuleOperator.NOT_EQUALS:
                matches = attr_value != rule.value
            elif rule.operator == RuleOperator.GREATER_THAN:
                matches = attr_value > rule.value
            elif rule.operator == RuleOperator.LESS_THAN:
                matches = attr_value < rule.value
            elif rule.operator == RuleOperator.CONTAINS:
                matches = rule.value in str(attr_value)
            elif rule.operator == RuleOperator.NOT_CONTAINS:
                matches = rule.value not in str(attr_value)
            elif rule.operator == RuleOperator.IN:
                matches = attr_value in rule.value
            elif rule.operator == RuleOperator.NOT_IN:
                matches = attr_value not in rule.value
            elif rule.operator == RuleOperator.MATCHES:
                import re
                matches = bool(re.match(rule.value, str(attr_value)))
            else:
                matches = False
        except Exception as e:
            self._log.warning("rule_eval_error", rule_id=rule.id, error=str(e))
            return False, 0.0

        return matches, rule.weight if matches else 0.0

    def evaluate_member(
        self,
        member: AudienceMember,
        segment_id: str,
    ) -> tuple[bool, float]:
        """Evaluate all rules for a segment against a member."""
        rules = self.get_rules_for_segment(segment_id)
        if not rules:
            return False, 0.0

        # All rules must match for segment classification
        total_weight = 0.0
        match_weight = 0.0

        for rule in rules:
            matches, weight = self.evaluate_rule(rule, member)
            total_weight += rule.weight
            if matches:
                match_weight += weight
            else:
                # If any required rule fails, segment doesn't match
                return False, 0.0

        confidence = match_weight / total_weight if total_weight > 0 else 0.0
        return True, confidence

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule."""
        rule = self._rules.get(rule_id)
        if not rule:
            return False

        del self._rules[rule_id]
        if rule.segment_id in self._by_segment:
            self._by_segment[rule.segment_id].discard(rule_id)
        return True


class MemberRegistry:
    """Registry of audience members and their segment memberships."""

    def __init__(self) -> None:
        self._members: dict[str, AudienceMember] = {}
        self._by_external_id: dict[str, str] = {}  # external_id -> member_id
        self._memberships: dict[str, SegmentMembership] = {}
        self._member_segments: dict[str, set[str]] = {}  # member_id -> membership IDs
        self._segment_members: dict[str, set[str]] = {}  # segment_id -> membership IDs
        self._log = logger.bind(component="member_registry")

    @property
    def member_count(self) -> int:
        return len(self._members)

    @property
    def membership_count(self) -> int:
        return len(self._memberships)

    def register_member(self, member: AudienceMember) -> None:
        """Register a member."""
        self._members[member.id] = member
        if member.external_id:
            self._by_external_id[member.external_id] = member.id
        self._log.debug("member_registered", member_id=member.id)

    def get_member(self, member_id: str) -> AudienceMember | None:
        """Get a member by ID."""
        return self._members.get(member_id)

    def get_by_external_id(self, external_id: str) -> AudienceMember | None:
        """Get a member by external ID."""
        member_id = self._by_external_id.get(external_id)
        if member_id:
            return self._members.get(member_id)
        return None

    def update_member(self, member_id: str, **updates: Any) -> AudienceMember | None:
        """Update a member's attributes."""
        member = self._members.get(member_id)
        if not member:
            return None

        # Merge attributes
        new_attributes = dict(member.attributes)
        if "attributes" in updates:
            new_attributes.update(updates.pop("attributes"))

        updated = AudienceMember(
            id=member.id,
            external_id=updates.get("external_id", member.external_id),
            name=updates.get("name", member.name),
            attributes=new_attributes,
            interaction_count=updates.get("interaction_count", member.interaction_count),
            last_interaction=updates.get("last_interaction", member.last_interaction),
            engagement_score=updates.get("engagement_score", member.engagement_score),
            created_at=member.created_at,
            tags=updates.get("tags", member.tags),
        )
        self._members[member_id] = updated
        return updated

    def add_membership(self, membership: SegmentMembership) -> None:
        """Add a segment membership."""
        self._memberships[membership.id] = membership
        self._member_segments.setdefault(membership.member_id, set()).add(membership.id)
        self._segment_members.setdefault(membership.segment_id, set()).add(membership.id)
        self._log.debug(
            "membership_added",
            member_id=membership.member_id,
            segment_id=membership.segment_id,
        )

    def get_member_segments(self, member_id: str) -> list[SegmentMembership]:
        """Get all segment memberships for a member."""
        membership_ids = self._member_segments.get(member_id, set())
        return [
            self._memberships[mid]
            for mid in membership_ids
            if mid in self._memberships and self._memberships[mid].is_active
        ]

    def get_segment_members(self, segment_id: str) -> list[SegmentMembership]:
        """Get all memberships for a segment."""
        membership_ids = self._segment_members.get(segment_id, set())
        return [
            self._memberships[mid]
            for mid in membership_ids
            if mid in self._memberships and self._memberships[mid].is_active
        ]

    def remove_membership(self, membership_id: str) -> bool:
        """Remove a membership (deactivate)."""
        membership = self._memberships.get(membership_id)
        if not membership:
            return False

        # Create deactivated version
        updated = SegmentMembership(
            id=membership.id,
            member_id=membership.member_id,
            segment_id=membership.segment_id,
            assigned_at=membership.assigned_at,
            assigned_by=membership.assigned_by,
            confidence=membership.confidence,
            is_active=False,
        )
        self._memberships[membership_id] = updated
        return True

    def clear_member_segments(self, member_id: str) -> int:
        """Clear all segment memberships for a member."""
        membership_ids = list(self._member_segments.get(member_id, set()))
        count = 0
        for mid in membership_ids:
            if self.remove_membership(mid):
                count += 1
        return count

    def get_member_count_for_segment(self, segment_id: str) -> int:
        """Get the count of active members in a segment."""
        memberships = self.get_segment_members(segment_id)
        return len([m for m in memberships if m.is_active])


class AudienceClassifier(Subsystem):
    """
    Categorizes users and participants.

    Process Loop:
    1. Collect: Gather data on user interactions and characteristics
    2. Classify: Group users into segments based on rules
    3. Assign: Determine access levels and roles per segment
    4. Reclassify: Update segment assignments as new data arrives
    """

    def __init__(self) -> None:
        metadata = SubsystemMetadata(
            name="audience_classifier",
            display_name="Audience Classifier",
            description="Categorizes users into segments",
            type=SubsystemType.IDENTITY,
            tags=frozenset(["audience", "classification", "segmentation", "access"]),
            input_types=frozenset(["IDENTITY", "PATTERN", "INTERACTION"]),
            output_types=frozenset(["IDENTITY", "ROLE", "SEGMENT"]),
            subscribed_topics=frozenset([
                str(SubsystemTopics.ALL),
                "audience.#",
                "classification.#",
            ]),
            published_topics=frozenset([
                "audience.classified",
                "audience.segment.assigned",
                "audience.segment.removed",
            ]),
        )
        super().__init__(metadata)

        self._segments = SegmentRegistry()
        self._rules = RuleEngine()
        self._members = MemberRegistry()

    @property
    def segment_count(self) -> int:
        return self._segments.segment_count

    @property
    def member_count(self) -> int:
        return self._members.member_count

    @property
    def rule_count(self) -> int:
        return self._rules.rule_count

    async def intake(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> SymbolicInput:
        """Phase 1: Collect audience data."""
        if not input_data.values:
            self._log.debug("empty_input")
            return input_data

        self._log.debug("intake_complete", value_count=len(input_data.values))
        return input_data

    async def process(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> list[ClassificationResult]:
        """Phase 2: Analyze and classify audience."""
        results: list[ClassificationResult] = []

        for value in input_data.values:
            content = value.content
            if not isinstance(content, dict):
                continue

            action = content.get("action", "classify")

            if action == "register":
                result = self._register_from_value(value)
                results.append(result)
            elif action == "classify":
                result = self._classify_from_value(value)
                results.append(result)
            elif action == "update":
                result = self._update_from_value(value)
                results.append(result)

        return results

    async def evaluate(
        self,
        intermediate: list[ClassificationResult],
        ctx: ProcessContext[dict[str, Any]],
    ) -> tuple[SymbolicOutput, bool]:
        """Phase 3: Prepare classification results."""
        values: list[SymbolicValue] = []

        for result in intermediate:
            value = SymbolicValue(
                type=SymbolicValueType.PATTERN,
                content={
                    "member_id": result.member_id,
                    "success": result.success,
                    "segments": result.segments,
                    "confidences": result.confidences,
                    "error": result.error,
                },
                source_subsystem=self.name,
                tags=frozenset(["classification", "audience"]),
                meaning="Classification result",
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
        """Phase 4: Report classification results."""
        if self._message_bus and output.values:
            for value in output.values:
                content = value.content
                if not isinstance(content, dict):
                    continue

                if content.get("success") and content.get("segments"):
                    await self.emit_event(
                        "audience.classified",
                        {
                            "member_id": content.get("member_id"),
                            "segments": content.get("segments"),
                        },
                    )

        return None

    def _register_from_value(self, value: SymbolicValue) -> ClassificationResult:
        """Register a member from a SymbolicValue."""
        content = value.content
        if not isinstance(content, dict):
            return ClassificationResult(member_id="", success=False, error="Invalid content")

        try:
            member = AudienceMember(
                id=content.get("id", str(ULID())),
                external_id=content.get("external_id"),
                name=content.get("name", ""),
                attributes=content.get("attributes", {}),
                tags=frozenset(content.get("tags", [])) | value.tags,
            )
            self._members.register_member(member)

            # Auto-classify if requested
            if content.get("auto_classify", True):
                return self.classify_member(member.id)

            return ClassificationResult(member_id=member.id, success=True)

        except Exception as e:
            self._log.warning("register_failed", error=str(e))
            return ClassificationResult(member_id="", success=False, error=str(e))

    def _classify_from_value(self, value: SymbolicValue) -> ClassificationResult:
        """Classify a member from a SymbolicValue."""
        content = value.content
        if not isinstance(content, dict):
            return ClassificationResult(member_id="", success=False, error="Invalid content")

        member_id = content.get("member_id")
        if not member_id:
            return ClassificationResult(member_id="", success=False, error="member_id required")

        return self.classify_member(member_id)

    def _update_from_value(self, value: SymbolicValue) -> ClassificationResult:
        """Update a member and reclassify."""
        content = value.content
        if not isinstance(content, dict):
            return ClassificationResult(member_id="", success=False, error="Invalid content")

        try:
            member_id = content.get("member_id")
            if not member_id:
                return ClassificationResult(member_id="", success=False, error="member_id required")

            # Update attributes
            updated = self._members.update_member(
                member_id,
                attributes=content.get("attributes", {}),
                interaction_count=content.get("interaction_count"),
                engagement_score=content.get("engagement_score"),
            )
            if not updated:
                return ClassificationResult(member_id=member_id, success=False, error="Member not found")

            # Reclassify
            if content.get("reclassify", True):
                return self.reclassify_member(member_id)

            return ClassificationResult(member_id=member_id, success=True)

        except Exception as e:
            return ClassificationResult(member_id="", success=False, error=str(e))

    # --- Message handlers ---

    async def handle_event(self, message: Message) -> None:
        """Handle incoming events."""
        if message.topic.startswith("audience.") or message.topic.startswith("classification."):
            self._log.debug("event_received", topic=message.topic)

    async def handle_signal(self, signal: Any) -> None:
        """Handle incoming signals."""
        self._log.debug("signal_received", signal_id=getattr(signal, "id", "unknown"))

    # --- Public API ---

    def create_segment(
        self,
        name: str,
        segment_type: SegmentType = SegmentType.TIER,
        **kwargs: Any,
    ) -> Segment:
        """Create a new segment."""
        access_str = kwargs.get("access_level", "STANDARD")
        if isinstance(access_str, str):
            try:
                access_level = AccessLevel[access_str.upper()]
            except KeyError:
                access_level = AccessLevel.STANDARD
        else:
            access_level = access_str

        segment = Segment(
            name=name,
            segment_type=segment_type,
            description=kwargs.get("description", ""),
            access_level=access_level,
            priority=kwargs.get("priority", 0),
            is_default=kwargs.get("is_default", False),
            is_exclusive=kwargs.get("is_exclusive", False),
            max_members=kwargs.get("max_members"),
            tags=frozenset(kwargs.get("tags", [])),
            metadata=kwargs.get("metadata", {}),
        )
        self._segments.register_segment(segment)
        return segment

    def get_segment(self, segment_id: str) -> Segment | None:
        """Get a segment by ID."""
        return self._segments.get_segment(segment_id)

    def get_segment_by_name(self, name: str) -> Segment | None:
        """Get a segment by name."""
        return self._segments.get_by_name(name)

    def add_rule(
        self,
        name: str,
        segment_id: str,
        attribute: str,
        operator: RuleOperator = RuleOperator.EQUALS,
        value: Any = None,
        **kwargs: Any,
    ) -> ClassificationRule:
        """Add a classification rule."""
        rule = ClassificationRule(
            name=name,
            segment_id=segment_id,
            attribute=attribute,
            operator=operator,
            value=value,
            priority=kwargs.get("priority", 0),
            weight=float(kwargs.get("weight", 1.0)),
            is_active=kwargs.get("is_active", True),
        )
        self._rules.add_rule(rule)
        return rule

    def get_rules_for_segment(self, segment_id: str) -> list[ClassificationRule]:
        """Get all rules for a segment."""
        return self._rules.get_rules_for_segment(segment_id)

    def register_member(
        self,
        external_id: str | None = None,
        name: str = "",
        attributes: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AudienceMember:
        """Register a new audience member."""
        member = AudienceMember(
            external_id=external_id,
            name=name,
            attributes=attributes or {},
            tags=frozenset(kwargs.get("tags", [])),
        )
        self._members.register_member(member)
        return member

    def get_member(self, member_id: str) -> AudienceMember | None:
        """Get a member by ID."""
        return self._members.get_member(member_id)

    def get_member_by_external_id(self, external_id: str) -> AudienceMember | None:
        """Get a member by external ID."""
        return self._members.get_by_external_id(external_id)

    def update_member_attributes(
        self,
        member_id: str,
        attributes: dict[str, Any],
    ) -> AudienceMember | None:
        """Update a member's attributes."""
        return self._members.update_member(member_id, attributes=attributes)

    def classify_member(self, member_id: str) -> ClassificationResult:
        """Classify a member into segments."""
        member = self._members.get_member(member_id)
        if not member:
            return ClassificationResult(
                member_id=member_id,
                success=False,
                error="Member not found",
            )

        segments: list[str] = []
        confidences = {}

        # Evaluate against all segments in priority order
        for segment in self._segments.get_all():
            # Check member limit
            if segment.max_members:
                current_count = self._members.get_member_count_for_segment(segment.id)
                if current_count >= segment.max_members:
                    continue

            # Evaluate rules
            rules = self._rules.get_rules_for_segment(segment.id)
            if not rules:
                # Segment with no rules - use as default only
                if segment.is_default and not segments:
                    segments.append(segment.id)
                    confidences[segment.id] = 1.0
                    # Create membership for default segment
                    membership = SegmentMembership(
                        member_id=member_id,
                        segment_id=segment.id,
                        assigned_by="default",
                        confidence=1.0,
                    )
                    self._members.add_membership(membership)
                continue

            matches, confidence = self._rules.evaluate_member(member, segment.id)
            if matches:
                segments.append(segment.id)
                confidences[segment.id] = confidence

                # Create membership
                membership = SegmentMembership(
                    member_id=member_id,
                    segment_id=segment.id,
                    assigned_by="classifier",
                    confidence=confidence,
                )
                self._members.add_membership(membership)

                # If exclusive, stop here
                if segment.is_exclusive:
                    break

        # Apply default segment if no matches
        if not segments:
            default = self._segments.get_default_segment()
            if default:
                segments.append(default.id)
                confidences[default.id] = 1.0
                membership = SegmentMembership(
                    member_id=member_id,
                    segment_id=default.id,
                    assigned_by="default",
                    confidence=1.0,
                )
                self._members.add_membership(membership)

        return ClassificationResult(
            member_id=member_id,
            success=True,
            segments=segments,
            confidences=confidences,
        )

    def reclassify_member(self, member_id: str) -> ClassificationResult:
        """Reclassify a member (clear existing and classify again)."""
        # Clear existing memberships
        self._members.clear_member_segments(member_id)
        # Reclassify
        return self.classify_member(member_id)

    def assign_to_segment(
        self,
        member_id: str,
        segment_id: str,
        **kwargs: Any,
    ) -> SegmentMembership | None:
        """Manually assign a member to a segment."""
        member = self._members.get_member(member_id)
        segment = self._segments.get_segment(segment_id)

        if not member or not segment:
            return None

        # Check member limit
        if segment.max_members:
            current_count = self._members.get_member_count_for_segment(segment_id)
            if current_count >= segment.max_members:
                return None

        membership = SegmentMembership(
            member_id=member_id,
            segment_id=segment_id,
            assigned_by=kwargs.get("assigned_by", "manual"),
            confidence=float(kwargs.get("confidence", 1.0)),
        )
        self._members.add_membership(membership)
        return membership

    def get_member_segments(self, member_id: str) -> list[Segment]:
        """Get all segments a member belongs to."""
        memberships = self._members.get_member_segments(member_id)
        return [
            seg
            for m in memberships
            if (seg := self._segments.get_segment(m.segment_id)) is not None
        ]

    def get_segment_members(self, segment_id: str) -> list[AudienceMember]:
        """Get all members in a segment."""
        memberships = self._members.get_segment_members(segment_id)
        return [
            mem
            for m in memberships
            if (mem := self._members.get_member(m.member_id)) is not None
        ]

    def get_member_access_level(self, member_id: str) -> AccessLevel:
        """Get the highest access level for a member."""
        segments = self.get_member_segments(member_id)
        if not segments:
            return AccessLevel.NONE
        return max(segments, key=lambda s: s.access_level.value).access_level

    def get_stats(self) -> SegmentStats:
        """Get classification statistics."""
        members_by_segment: dict[str, int] = {}
        for segment in self._segments.get_all():
            members_by_segment[segment.name] = self._members.get_member_count_for_segment(segment.id)

        segments_by_type: dict[str, int] = {}
        for segment in self._segments.get_all():
            type_name = segment.segment_type.name
            segments_by_type[type_name] = segments_by_type.get(type_name, 0) + 1

        return SegmentStats(
            total_segments=self._segments.segment_count,
            total_members=self._members.member_count,
            total_memberships=self._members.membership_count,
            members_by_segment=members_by_segment,
            segments_by_type=segments_by_type,
        )

    def clear(self) -> tuple[int, int, int]:
        """Clear all data. Returns (segments, members, rules) cleared."""
        segments = self._segments.segment_count
        members = self._members.member_count
        rules = self._rules.rule_count
        self._segments = SegmentRegistry()
        self._rules = RuleEngine()
        self._members = MemberRegistry()
        return segments, members, rules
