"""
ConsumptionManager: Monitors and manages consumption events.

Governs ingestion, evaluation, and tracking of consumption within the system,
ensuring safety and balance in symbolic resource usage.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
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


class ResourceType(Enum):
    """Types of consumable resources."""

    TOKEN = auto()  # Generic token # allow-secret
    COMPUTE = auto()  # Compute resources
    STORAGE = auto()  # Storage resources
    BANDWIDTH = auto()  # Network bandwidth
    API_CALL = auto()  # API calls
    CONTENT = auto()  # Content/media
    ENERGY = auto()  # Energy units
    CUSTOM = auto()  # Custom resource


class RiskLevel(Enum):
    """Risk levels for consumption."""

    SAFE = auto()  # No risk
    LOW = auto()  # Low risk
    MEDIUM = auto()  # Medium risk
    HIGH = auto()  # High risk
    CRITICAL = auto()  # Critical risk - block


class ConsumptionStatus(Enum):
    """Status of a consumption event."""

    PENDING = auto()  # Awaiting evaluation
    APPROVED = auto()  # Approved for consumption
    CONSUMED = auto()  # Successfully consumed
    REJECTED = auto()  # Rejected by policy
    RATE_LIMITED = auto()  # Rate limited
    QUOTA_EXCEEDED = auto()  # Quota exceeded


class ConsumptionEvent(BaseModel):
    """A consumption event."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    consumer_id: str  # Who is consuming
    resource_type: ResourceType
    resource_id: str | None = None  # Specific resource ID

    # Consumption details
    amount: Decimal = Decimal("1")
    unit: str = "unit"
    status: ConsumptionStatus = ConsumptionStatus.PENDING

    # Evaluation
    risk_level: RiskLevel = RiskLevel.SAFE
    risk_score: float = 0.0
    risk_factors: tuple[str, ...] = Field(default_factory=tuple)

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    evaluated_at: datetime | None = None
    consumed_at: datetime | None = None
    context: str = ""
    tags: frozenset[str] = Field(default_factory=frozenset)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConsumptionQuota(BaseModel):
    """A consumption quota/limit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    name: str
    consumer_id: str | None = None  # None = global quota
    resource_type: ResourceType

    # Limits
    max_amount: Decimal
    period: timedelta | None = None  # None = lifetime limit
    unit: str = "unit"

    # Current usage
    current_usage: Decimal = Decimal("0")
    period_start: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # State
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RiskRule(BaseModel):
    """A rule for evaluating consumption risk."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    name: str
    description: str = ""

    # Matching
    resource_type: ResourceType | None = None  # None = all types
    consumer_pattern: str | None = None  # Pattern to match consumer

    # Risk assessment
    condition: str  # Condition expression
    risk_level: RiskLevel = RiskLevel.LOW
    risk_score: float = 0.1

    priority: int = 0
    is_active: bool = True


class UsageMetrics(BaseModel):
    """Usage metrics for a consumer/resource."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    consumer_id: str
    resource_type: ResourceType
    period_start: datetime
    period_end: datetime

    # Metrics
    total_events: int = 0
    total_consumed: Decimal = Decimal("0")
    total_rejected: int = 0
    average_risk_score: float = 0.0
    peak_usage: Decimal = Decimal("0")


@dataclass
class EvaluationResult:
    """Result of consumption evaluation."""

    event_id: str
    approved: bool
    risk_level: RiskLevel = RiskLevel.SAFE
    risk_score: float = 0.0
    risk_factors: list[str] = field(default_factory=list)
    rejection_reason: str | None = None


@dataclass
class ConsumptionStats:
    """Statistics about consumption."""

    total_events: int
    consumed_count: int
    rejected_count: int
    total_quotas: int
    total_rules: int
    usage_by_type: dict[str, Decimal]


class EventLog:
    """Log of consumption events."""

    def __init__(self) -> None:
        self._events: dict[str, ConsumptionEvent] = {}
        self._by_consumer: dict[str, list[str]] = {}  # consumer_id -> event IDs
        self._by_type: dict[ResourceType, list[str]] = {}
        self._log = logger.bind(component="event_log")

    @property
    def event_count(self) -> int:
        return len(self._events)

    def log_event(self, event: ConsumptionEvent) -> None:
        """Log a consumption event."""
        self._events[event.id] = event
        self._by_consumer.setdefault(event.consumer_id, []).append(event.id)
        self._by_type.setdefault(event.resource_type, []).append(event.id)
        self._log.debug(
            "event_logged",
            event_id=event.id,
            consumer=event.consumer_id,
            status=event.status.name,
        )

    def update_event(self, event_id: str, **updates: Any) -> ConsumptionEvent | None:
        """Update an event."""
        event = self._events.get(event_id)
        if not event:
            return None

        data = {
            "id": event.id,
            "consumer_id": event.consumer_id,
            "resource_type": event.resource_type,
            "resource_id": event.resource_id,
            "amount": updates.get("amount", event.amount),
            "unit": updates.get("unit", event.unit),
            "status": updates.get("status", event.status),
            "risk_level": updates.get("risk_level", event.risk_level),
            "risk_score": updates.get("risk_score", event.risk_score),
            "risk_factors": updates.get("risk_factors", event.risk_factors),
            "created_at": event.created_at,
            "evaluated_at": updates.get("evaluated_at", event.evaluated_at),
            "consumed_at": updates.get("consumed_at", event.consumed_at),
            "context": updates.get("context", event.context),
            "tags": updates.get("tags", event.tags),
            "metadata": updates.get("metadata", event.metadata),
        }
        updated = ConsumptionEvent(**data)
        self._events[event_id] = updated
        return updated

    def get_event(self, event_id: str) -> ConsumptionEvent | None:
        """Get an event by ID."""
        return self._events.get(event_id)

    def get_by_consumer(
        self,
        consumer_id: str,
        since: datetime | None = None,
    ) -> list[ConsumptionEvent]:
        """Get events for a consumer."""
        event_ids = self._by_consumer.get(consumer_id, [])
        events = [self._events[eid] for eid in event_ids if eid in self._events]
        if since:
            events = [e for e in events if e.created_at >= since]
        return events

    def get_by_type(
        self,
        resource_type: ResourceType,
        since: datetime | None = None,
    ) -> list[ConsumptionEvent]:
        """Get events by resource type."""
        event_ids = self._by_type.get(resource_type, [])
        events = [self._events[eid] for eid in event_ids if eid in self._events]
        if since:
            events = [e for e in events if e.created_at >= since]
        return events

    def get_consumed(self) -> list[ConsumptionEvent]:
        """Get all consumed events."""
        return [e for e in self._events.values() if e.status == ConsumptionStatus.CONSUMED]


class QuotaManager:
    """Manages consumption quotas."""

    def __init__(self) -> None:
        self._quotas: dict[str, ConsumptionQuota] = {}
        self._by_consumer: dict[str, list[str]] = {}
        self._global_quotas: list[str] = []
        self._log = logger.bind(component="quota_manager")

    @property
    def quota_count(self) -> int:
        return len(self._quotas)

    def add_quota(self, quota: ConsumptionQuota) -> None:
        """Add a quota."""
        self._quotas[quota.id] = quota
        if quota.consumer_id:
            self._by_consumer.setdefault(quota.consumer_id, []).append(quota.id)
        else:
            self._global_quotas.append(quota.id)
        self._log.debug("quota_added", quota_id=quota.id, name=quota.name)

    def get_quota(self, quota_id: str) -> ConsumptionQuota | None:
        """Get a quota by ID."""
        return self._quotas.get(quota_id)

    def get_applicable_quotas(
        self,
        consumer_id: str,
        resource_type: ResourceType,
    ) -> list[ConsumptionQuota]:
        """Get quotas applicable to a consumer and resource type."""
        applicable = []

        # Consumer-specific quotas
        for qid in self._by_consumer.get(consumer_id, []):
            quota = self._quotas.get(qid)
            if quota and quota.is_active and quota.resource_type == resource_type:
                applicable.append(quota)

        # Global quotas
        for qid in self._global_quotas:
            quota = self._quotas.get(qid)
            if quota and quota.is_active and quota.resource_type == resource_type:
                applicable.append(quota)

        return applicable

    def check_quota(
        self,
        quota: ConsumptionQuota,
        amount: Decimal,
    ) -> tuple[bool, Decimal]:
        """Check if quota allows consumption. Returns (allowed, remaining)."""
        # Check if period needs reset
        if quota.period:
            now = datetime.now(UTC)
            if now >= quota.period_start + quota.period:
                # Reset period
                quota = ConsumptionQuota(
                    id=quota.id,
                    name=quota.name,
                    consumer_id=quota.consumer_id,
                    resource_type=quota.resource_type,
                    max_amount=quota.max_amount,
                    period=quota.period,
                    unit=quota.unit,
                    current_usage=Decimal("0"),
                    period_start=now,
                    is_active=quota.is_active,
                    created_at=quota.created_at,
                )
                self._quotas[quota.id] = quota

        remaining = quota.max_amount - quota.current_usage
        allowed = remaining >= amount
        return allowed, remaining

    def consume_quota(
        self,
        quota_id: str,
        amount: Decimal,
    ) -> bool:
        """Consume from a quota."""
        quota = self._quotas.get(quota_id)
        if not quota:
            return False

        allowed, _ = self.check_quota(quota, amount)
        if not allowed:
            return False

        # Update usage
        updated = ConsumptionQuota(
            id=quota.id,
            name=quota.name,
            consumer_id=quota.consumer_id,
            resource_type=quota.resource_type,
            max_amount=quota.max_amount,
            period=quota.period,
            unit=quota.unit,
            current_usage=quota.current_usage + amount,
            period_start=quota.period_start,
            is_active=quota.is_active,
            created_at=quota.created_at,
        )
        self._quotas[quota_id] = updated
        return True


class RiskEvaluator:
    """Evaluates consumption risk."""

    def __init__(self) -> None:
        self._rules: dict[str, RiskRule] = {}
        self._log = logger.bind(component="risk_evaluator")

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    def add_rule(self, rule: RiskRule) -> None:
        """Add a risk rule."""
        self._rules[rule.id] = rule
        self._log.debug("rule_added", rule_id=rule.id, name=rule.name)

    def get_rule(self, rule_id: str) -> RiskRule | None:
        """Get a rule by ID."""
        return self._rules.get(rule_id)

    def evaluate(
        self,
        event: ConsumptionEvent,
    ) -> EvaluationResult:
        """Evaluate consumption risk."""
        risk_factors: list[str] = []
        max_risk_level = RiskLevel.SAFE
        total_risk_score = 0.0
        matching_rules = 0

        for rule in sorted(self._rules.values(), key=lambda r: -r.priority):
            if not rule.is_active:
                continue

            # Check resource type match
            if rule.resource_type and rule.resource_type != event.resource_type:
                continue

            # Check consumer pattern match
            if rule.consumer_pattern:
                import re
                if not re.match(rule.consumer_pattern, event.consumer_id):
                    continue

            # Evaluate condition
            if self._evaluate_condition(rule.condition, event):
                risk_factors.append(rule.name)
                total_risk_score += rule.risk_score
                matching_rules += 1

                if rule.risk_level.value > max_risk_level.value:
                    max_risk_level = rule.risk_level

        # Average risk score
        avg_score = total_risk_score / matching_rules if matching_rules > 0 else 0.0

        # Determine approval
        approved = max_risk_level not in (RiskLevel.CRITICAL, RiskLevel.HIGH)
        rejection_reason = None
        if not approved:
            rejection_reason = f"Risk level {max_risk_level.name}: {', '.join(risk_factors)}"

        return EvaluationResult(
            event_id=event.id,
            approved=approved,
            risk_level=max_risk_level,
            risk_score=avg_score,
            risk_factors=risk_factors,
            rejection_reason=rejection_reason,
        )

    def _evaluate_condition(self, condition: str, event: ConsumptionEvent) -> bool:
        """Evaluate a condition against an event."""
        try:
            # Simple condition evaluation
            if condition == "always":
                return True
            elif condition == "never":
                return False
            elif condition.startswith("amount>"):
                threshold = Decimal(condition.split(">")[1])
                return event.amount > threshold
            elif condition.startswith("amount<"):
                threshold = Decimal(condition.split("<")[1])
                return event.amount < threshold
            elif condition.startswith("amount>="):
                threshold = Decimal(condition.split(">=")[1])
                return event.amount >= threshold
            elif condition.startswith("tag:"):
                tag = condition.split(":")[1]
                return tag in event.tags
            else:
                return False
        except Exception:
            return False


class ConsumptionManager(Subsystem):
    """
    Monitors and manages consumption events.

    Process Loop:
    1. Intake: Receive content or resource inputs for consumption
    2. Evaluate: Check inputs for validity, safety, and compliance
    3. Log: Record consumption events and associated metadata
    4. Feedback: Provide metrics and adjustments to related subsystems
    """

    def __init__(self) -> None:
        metadata = SubsystemMetadata(
            name="consumption_manager",
            display_name="Consumption Manager",
            description="Monitors and manages consumption events",
            type=SubsystemType.TRANSFORMATION,
            tags=frozenset(["consumption", "monitoring", "resources", "quota"]),
            input_types=frozenset(["TOKEN", "SIGNAL", "RESOURCE"]),
            output_types=frozenset(["SIGNAL", "REFERENCE", "METRIC"]),
            subscribed_topics=frozenset([
                str(SubsystemTopics.ALL),
                "consumption.#",
                "resource.#",
            ]),
            published_topics=frozenset([
                "consumption.consumed",
                "consumption.rejected",
                "consumption.quota.exceeded",
            ]),
        )
        super().__init__(metadata)

        self._event_log = EventLog()
        self._quota_manager = QuotaManager()
        self._risk_evaluator = RiskEvaluator()

    @property
    def event_count(self) -> int:
        return self._event_log.event_count

    @property
    def quota_count(self) -> int:
        return self._quota_manager.quota_count

    @property
    def rule_count(self) -> int:
        return self._risk_evaluator.rule_count

    async def intake(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> SymbolicInput:
        """Phase 1: Receive consumption inputs."""
        if not input_data.values:
            self._log.debug("empty_input")
            return input_data

        self._log.debug("intake_complete", value_count=len(input_data.values))
        return input_data

    async def process(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> list[EvaluationResult]:
        """Phase 2: Evaluate and process consumption."""
        results: list[EvaluationResult] = []

        for value in input_data.values:
            content = value.content
            if not isinstance(content, dict):
                continue

            action = content.get("action", "consume")

            if action == "consume":
                result = self._consume_from_value(value)
                results.append(result)

        return results

    async def evaluate(
        self,
        intermediate: list[EvaluationResult],
        ctx: ProcessContext[dict[str, Any]],
    ) -> tuple[SymbolicOutput, bool]:
        """Phase 3: Prepare consumption results."""
        values: list[SymbolicValue] = []

        for result in intermediate:
            value = SymbolicValue(
                type=SymbolicValueType.SIGNAL,
                content={
                    "event_id": result.event_id,
                    "approved": result.approved,
                    "risk_level": result.risk_level.name,
                    "risk_score": result.risk_score,
                    "risk_factors": result.risk_factors,
                    "rejection_reason": result.rejection_reason,
                },
                source_subsystem=self.name,
                tags=frozenset(["consumption", "evaluation"]),
                meaning="Consumption evaluation result",
                confidence=1.0 - result.risk_score,
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
        """Phase 4: Emit events and provide feedback."""
        if self._message_bus and output.values:
            for value in output.values:
                content = value.content
                if not isinstance(content, dict):
                    continue

                if content.get("approved"):
                    await self.emit_event(
                        "consumption.consumed",
                        {"event_id": content.get("event_id")},
                    )
                else:
                    await self.emit_event(
                        "consumption.rejected",
                        {
                            "event_id": content.get("event_id"),
                            "reason": content.get("rejection_reason"),
                        },
                    )

        return None

    def _consume_from_value(self, value: SymbolicValue) -> EvaluationResult:
        """Process a consumption request from a SymbolicValue."""
        content = value.content
        if not isinstance(content, dict):
            return EvaluationResult(
                event_id="",
                approved=False,
                rejection_reason="Invalid content",
            )

        try:
            type_str = content.get("resource_type", "TOKEN")
            try:
                resource_type = ResourceType[type_str.upper()]
            except KeyError:
                resource_type = ResourceType.TOKEN

            event = ConsumptionEvent(
                consumer_id=content.get("consumer_id", "unknown"),
                resource_type=resource_type,
                resource_id=content.get("resource_id"),
                amount=Decimal(str(content.get("amount", "1"))),
                unit=content.get("unit", "unit"),
                context=content.get("context", ""),
                tags=frozenset(content.get("tags", [])) | value.tags,
                metadata=content.get("metadata", {}),
            )

            return self.consume(event)

        except Exception as e:
            return EvaluationResult(
                event_id="",
                approved=False,
                rejection_reason=str(e),
            )

    # --- Message handlers ---

    async def handle_event(self, message: Message) -> None:
        """Handle incoming events."""
        if message.topic.startswith("consumption.") or message.topic.startswith("resource."):
            self._log.debug("event_received", topic=message.topic)

    async def handle_signal(self, signal: Any) -> None:
        """Handle incoming signals."""
        self._log.debug("signal_received", signal_id=getattr(signal, "id", "unknown"))

    # --- Public API ---

    def create_event(
        self,
        consumer_id: str,
        resource_type: ResourceType,
        amount: Decimal = Decimal("1"),
        **kwargs: Any,
    ) -> ConsumptionEvent:
        """Create a consumption event."""
        return ConsumptionEvent(
            consumer_id=consumer_id,
            resource_type=resource_type,
            resource_id=kwargs.get("resource_id"),
            amount=amount,
            unit=kwargs.get("unit", "unit"),
            context=kwargs.get("context", ""),
            tags=frozenset(kwargs.get("tags", [])),
            metadata=kwargs.get("metadata", {}),
        )

    def consume(self, event: ConsumptionEvent) -> EvaluationResult:
        """Evaluate and process a consumption event."""
        # Log the pending event
        self._event_log.log_event(event)

        # Evaluate risk
        eval_result = self._risk_evaluator.evaluate(event)

        # Check quotas
        quotas = self._quota_manager.get_applicable_quotas(
            event.consumer_id,
            event.resource_type,
        )

        quota_ok = True
        quota_reason = None
        for quota in quotas:
            allowed, remaining = self._quota_manager.check_quota(quota, event.amount)
            if not allowed:
                quota_ok = False
                quota_reason = f"Quota '{quota.name}' exceeded (remaining: {remaining})"
                break

        # Determine final approval
        approved = eval_result.approved and quota_ok
        rejection_reason = eval_result.rejection_reason or quota_reason

        # Update event status
        if approved:
            # Consume from quotas
            for quota in quotas:
                self._quota_manager.consume_quota(quota.id, event.amount)

            self._event_log.update_event(
                event.id,
                status=ConsumptionStatus.CONSUMED,
                risk_level=eval_result.risk_level,
                risk_score=eval_result.risk_score,
                risk_factors=tuple(eval_result.risk_factors),
                evaluated_at=datetime.now(UTC),
                consumed_at=datetime.now(UTC),
            )
        elif not quota_ok:
            self._event_log.update_event(
                event.id,
                status=ConsumptionStatus.QUOTA_EXCEEDED,
                risk_level=eval_result.risk_level,
                risk_score=eval_result.risk_score,
                risk_factors=tuple(eval_result.risk_factors),
                evaluated_at=datetime.now(UTC),
            )
        else:
            self._event_log.update_event(
                event.id,
                status=ConsumptionStatus.REJECTED,
                risk_level=eval_result.risk_level,
                risk_score=eval_result.risk_score,
                risk_factors=tuple(eval_result.risk_factors),
                evaluated_at=datetime.now(UTC),
            )

        return EvaluationResult(
            event_id=event.id,
            approved=approved,
            risk_level=eval_result.risk_level,
            risk_score=eval_result.risk_score,
            risk_factors=eval_result.risk_factors,
            rejection_reason=rejection_reason,
        )

    def get_event(self, event_id: str) -> ConsumptionEvent | None:
        """Get a consumption event."""
        return self._event_log.get_event(event_id)

    def get_consumer_events(
        self,
        consumer_id: str,
        since: datetime | None = None,
    ) -> list[ConsumptionEvent]:
        """Get events for a consumer."""
        return self._event_log.get_by_consumer(consumer_id, since)

    def add_quota(
        self,
        name: str,
        resource_type: ResourceType,
        max_amount: Decimal,
        **kwargs: Any,
    ) -> ConsumptionQuota:
        """Add a consumption quota."""
        quota = ConsumptionQuota(
            name=name,
            consumer_id=kwargs.get("consumer_id"),
            resource_type=resource_type,
            max_amount=max_amount,
            period=kwargs.get("period"),
            unit=kwargs.get("unit", "unit"),
            is_active=kwargs.get("is_active", True),
        )
        self._quota_manager.add_quota(quota)
        return quota

    def get_quota(self, quota_id: str) -> ConsumptionQuota | None:
        """Get a quota."""
        return self._quota_manager.get_quota(quota_id)

    def check_quota(
        self,
        consumer_id: str,
        resource_type: ResourceType,
        amount: Decimal,
    ) -> tuple[bool, Decimal | None]:
        """Check if consumption is within quota. Returns (allowed, min_remaining)."""
        quotas = self._quota_manager.get_applicable_quotas(consumer_id, resource_type)
        if not quotas:
            return True, None

        min_remaining = None
        for quota in quotas:
            allowed, remaining = self._quota_manager.check_quota(quota, amount)
            if not allowed:
                return False, remaining
            if min_remaining is None or remaining < min_remaining:
                min_remaining = remaining

        return True, min_remaining

    def add_risk_rule(
        self,
        name: str,
        condition: str,
        risk_level: RiskLevel = RiskLevel.LOW,
        **kwargs: Any,
    ) -> RiskRule:
        """Add a risk evaluation rule."""
        type_str = kwargs.get("resource_type")
        resource_type = None
        if type_str:
            try:
                resource_type = ResourceType[type_str.upper()] if isinstance(type_str, str) else type_str
            except KeyError:
                pass

        rule = RiskRule(
            name=name,
            description=kwargs.get("description", ""),
            resource_type=resource_type,
            consumer_pattern=kwargs.get("consumer_pattern"),
            condition=condition,
            risk_level=risk_level,
            risk_score=float(kwargs.get("risk_score", 0.1)),
            priority=kwargs.get("priority", 0),
            is_active=kwargs.get("is_active", True),
        )
        self._risk_evaluator.add_rule(rule)
        return rule

    def get_metrics(
        self,
        consumer_id: str,
        resource_type: ResourceType,
        period: timedelta | None = None,
    ) -> UsageMetrics:
        """Get usage metrics for a consumer."""
        now = datetime.now(UTC)
        since = now - period if period else None
        period_start = since or datetime.min.replace(tzinfo=UTC)

        events = self._event_log.get_by_consumer(consumer_id, since)
        type_events = [e for e in events if e.resource_type == resource_type]

        total_consumed = Decimal("0")
        total_rejected = 0
        total_risk = 0.0
        peak = Decimal("0")

        for event in type_events:
            if event.status == ConsumptionStatus.CONSUMED:
                total_consumed += event.amount
                if event.amount > peak:
                    peak = event.amount
            elif event.status in (ConsumptionStatus.REJECTED, ConsumptionStatus.QUOTA_EXCEEDED):
                total_rejected += 1
            total_risk += event.risk_score

        avg_risk = total_risk / len(type_events) if type_events else 0.0

        return UsageMetrics(
            consumer_id=consumer_id,
            resource_type=resource_type,
            period_start=period_start,
            period_end=now,
            total_events=len(type_events),
            total_consumed=total_consumed,
            total_rejected=total_rejected,
            average_risk_score=avg_risk,
            peak_usage=peak,
        )

    def get_stats(self) -> ConsumptionStats:
        """Get consumption statistics."""
        consumed = self._event_log.get_consumed()

        usage_by_type: dict[str, Decimal] = {}
        for event in consumed:
            key = event.resource_type.name
            usage_by_type[key] = usage_by_type.get(key, Decimal("0")) + event.amount

        rejected_count = len([
            e for e in self._event_log._events.values()
            if e.status in (ConsumptionStatus.REJECTED, ConsumptionStatus.QUOTA_EXCEEDED)
        ])

        return ConsumptionStats(
            total_events=self._event_log.event_count,
            consumed_count=len(consumed),
            rejected_count=rejected_count,
            total_quotas=self._quota_manager.quota_count,
            total_rules=self._risk_evaluator.rule_count,
            usage_by_type=usage_by_type,
        )

    def clear(self) -> tuple[int, int, int]:
        """Clear all data. Returns (events, quotas, rules) cleared."""
        events = self._event_log.event_count
        quotas = self._quota_manager.quota_count
        rules = self._risk_evaluator.rule_count
        self._event_log = EventLog()
        self._quota_manager = QuotaManager()
        self._risk_evaluator = RiskEvaluator()
        return events, quotas, rules
