"""
ArbitrationEngine: Oversees dispute resolution through structured arbitration processes.

Governs resolution through structured arbitration, modeling fairness and
balance in symbolic systems.
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


class DisputeType(Enum):
    """Types of disputes that can be arbitrated."""

    VALUE = auto()  # Dispute over values
    RULE = auto()  # Dispute over rule interpretation
    RESOURCE = auto()  # Resource allocation dispute
    PRIORITY = auto()  # Priority/precedence dispute
    OWNERSHIP = auto()  # Ownership/attribution dispute
    PROCESS = auto()  # Process execution dispute
    INTERPRETATION = auto()  # Interpretation dispute
    OTHER = auto()  # Other disputes


class DisputeStatus(Enum):
    """Status of a dispute."""

    SUBMITTED = auto()  # Dispute submitted
    EVIDENCE_COLLECTION = auto()  # Collecting evidence
    DELIBERATION = auto()  # Under deliberation
    DECIDED = auto()  # Decision made
    APPEALED = auto()  # Decision appealed
    CLOSED = auto()  # Dispute closed
    DISMISSED = auto()  # Dispute dismissed


class VerdictType(Enum):
    """Types of verdicts."""

    IN_FAVOR_PARTY_A = auto()
    IN_FAVOR_PARTY_B = auto()
    COMPROMISE = auto()
    SPLIT_DECISION = auto()
    DISMISSED = auto()
    DEFERRED = auto()


class Party(BaseModel):
    """A party in a dispute."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    name: str
    subsystem: str | None = None  # Representing subsystem
    role: str = "claimant"  # "claimant" or "respondent"
    position: str = ""  # Party's position/claim
    tags: frozenset[str] = Field(default_factory=frozenset)


class Evidence(BaseModel):
    """Evidence submitted in a dispute."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    party_id: str  # Submitting party
    evidence_type: str = "document"  # "document", "data", "testimony", "reference"
    content: Any
    description: str = ""
    weight: float = Field(default=1.0, ge=0.0, le=1.0)  # Evidential weight
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    verified: bool = False
    tags: frozenset[str] = Field(default_factory=frozenset)


class Argument(BaseModel):
    """An argument made by a party."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    party_id: str
    claim: str
    supporting_evidence: tuple[str, ...] = Field(default_factory=tuple)  # Evidence IDs
    strength: float = Field(default=0.5, ge=0.0, le=1.0)
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ArbitrationRule(BaseModel):
    """A rule applied during arbitration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    name: str
    description: str
    applies_to: tuple[DisputeType, ...] = Field(default_factory=tuple)  # Empty = all
    condition: str = ""  # Condition for applying rule
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    priority: int = 50


class Verdict(BaseModel):
    """A verdict in a dispute."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    dispute_id: str
    verdict_type: VerdictType
    summary: str
    reasoning: str = ""

    # Decision details
    prevailing_party_id: str | None = None
    awarded_value: Any | None = None
    conditions: tuple[str, ...] = Field(default_factory=tuple)

    # Metadata
    rules_applied: tuple[str, ...] = Field(default_factory=tuple)  # Rule IDs
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    decided_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    appealable: bool = True


class Dispute(BaseModel):
    """A dispute submitted for arbitration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    dispute_type: DisputeType
    status: DisputeStatus = DisputeStatus.SUBMITTED

    # Parties
    parties: tuple[Party, ...] = Field(default_factory=tuple)

    # Subject
    subject: str  # What the dispute is about
    description: str = ""
    disputed_value: Any | None = None

    # Evidence and arguments
    evidence: tuple[Evidence, ...] = Field(default_factory=tuple)
    arguments: tuple[Argument, ...] = Field(default_factory=tuple)

    # Resolution
    verdict: Verdict | None = None
    related_conflict_id: str | None = None  # If escalated from ConflictResolver

    # Metadata
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    deadline: datetime | None = None
    tags: frozenset[str] = Field(default_factory=frozenset)


@dataclass
class DeliberationResult:
    """Result of deliberation on a dispute."""

    dispute_id: str
    scores: dict[str, float]  # party_id -> score
    arguments_evaluated: int
    evidence_weighed: int
    rules_applied: list[str]
    recommendation: VerdictType


@dataclass
class ArbitrationStats:
    """Statistics about arbitration."""

    total_disputes: int
    disputes_decided: int
    disputes_pending: int
    disputes_dismissed: int
    by_type: dict[str, int]
    average_resolution_time_hours: float


class DeliberationEngine:
    """Engine for deliberating on disputes."""

    def __init__(self) -> None:
        self._rules: dict[str, ArbitrationRule] = {}
        self._log = logger.bind(component="deliberation_engine")

        # Add default rules
        self._add_default_rules()

    def _add_default_rules(self) -> None:
        """Add default arbitration rules."""
        defaults = [
            ArbitrationRule(
                name="evidence_weight",
                description="Stronger evidence carries more weight",
                priority=100,
            ),
            ArbitrationRule(
                name="temporal_precedence",
                description="Earlier claims have precedence in ownership disputes",
                applies_to=(DisputeType.OWNERSHIP,),
                priority=75,
            ),
            ArbitrationRule(
                name="rule_hierarchy",
                description="Higher-level rules override lower-level rules",
                applies_to=(DisputeType.RULE,),
                priority=80,
            ),
            ArbitrationRule(
                name="compromise_preference",
                description="Prefer compromise when positions are equally strong",
                priority=50,
            ),
        ]
        for rule in defaults:
            self._rules[rule.id] = rule

    def add_rule(self, rule: ArbitrationRule) -> None:
        """Add an arbitration rule."""
        self._rules[rule.id] = rule

    def deliberate(self, dispute: Dispute) -> DeliberationResult:
        """Deliberate on a dispute and produce a recommendation."""
        # Get applicable rules
        applicable_rules = self._get_applicable_rules(dispute)

        # Score each party's arguments and evidence
        scores: dict[str, float] = {}
        for party in dispute.parties:
            score = self._score_party(party, dispute, applicable_rules)
            scores[party.id] = score

        # Determine recommendation
        recommendation = self._determine_recommendation(scores, dispute)

        return DeliberationResult(
            dispute_id=dispute.id,
            scores=scores,
            arguments_evaluated=len(dispute.arguments),
            evidence_weighed=len(dispute.evidence),
            rules_applied=[r.name for r in applicable_rules],
            recommendation=recommendation,
        )

    def _get_applicable_rules(self, dispute: Dispute) -> list[ArbitrationRule]:
        """Get rules applicable to a dispute."""
        applicable = []
        for rule in self._rules.values():
            if not rule.applies_to or dispute.dispute_type in rule.applies_to:
                applicable.append(rule)
        return sorted(applicable, key=lambda r: r.priority, reverse=True)

    def _score_party(
        self,
        party: Party,
        dispute: Dispute,
        rules: list[ArbitrationRule],
    ) -> float:
        """Score a party's case."""
        score = 0.0
        count = 0

        # Score evidence
        for evidence in dispute.evidence:
            if evidence.party_id == party.id:
                weight = evidence.weight
                if evidence.verified:
                    weight *= 1.2  # Bonus for verified evidence
                score += min(weight, 1.0)
                count += 1

        # Score arguments
        for argument in dispute.arguments:
            if argument.party_id == party.id:
                # Arguments supported by evidence are stronger
                evidence_support = len(argument.supporting_evidence) * 0.1
                score += argument.strength + min(evidence_support, 0.3)
                count += 1

        # Normalize
        return score / max(count, 1)

    def _determine_recommendation(
        self,
        scores: dict[str, float],
        dispute: Dispute,
    ) -> VerdictType:
        """Determine verdict recommendation based on scores."""
        if len(scores) < 2:
            return VerdictType.DISMISSED

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_score = sorted_scores[0][1]
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0

        # Check for clear winner
        if top_score > second_score * 1.5:  # Clear margin
            if sorted_scores[0][0] == dispute.parties[0].id if dispute.parties else None:
                return VerdictType.IN_FAVOR_PARTY_A
            return VerdictType.IN_FAVOR_PARTY_B

        # Close scores suggest compromise
        if abs(top_score - second_score) < 0.2:
            return VerdictType.COMPROMISE

        return VerdictType.SPLIT_DECISION


class ArbitrationEngine(Subsystem):
    """
    Oversees dispute resolution through structured arbitration processes.

    Process Loop:
    1. Submit: Disputes are submitted to the engine
    2. Deliberate: Structured deliberation applies defined rules
    3. Decide: Issue symbolic judgments
    4. Record: Log rulings and update system state
    """

    def __init__(self) -> None:
        metadata = SubsystemMetadata(
            name="arbitration_engine",
            display_name="Arbitration Engine",
            description="Oversees dispute resolution through structured arbitration",
            type=SubsystemType.CONFLICT,
            tags=frozenset(["arbitration", "dispute", "resolution", "judgment"]),
            input_types=frozenset(["SCHEMA", "RULE"]),
            output_types=frozenset(["SCHEMA", "RULE"]),
            subscribed_topics=frozenset([
                str(SubsystemTopics.ALL),
                "arbitration.#",
                "dispute.#",
                "conflict.escalated",
            ]),
            published_topics=frozenset([
                "arbitration.dispute.submitted",
                "arbitration.verdict.issued",
                "arbitration.dispute.closed",
            ]),
        )
        super().__init__(metadata)

        self._deliberation = DeliberationEngine()
        self._disputes: dict[str, Dispute] = {}
        self._verdicts: dict[str, Verdict] = {}

    @property
    def dispute_count(self) -> int:
        return len(self._disputes)

    @property
    def verdict_count(self) -> int:
        return len(self._verdicts)

    async def intake(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> SymbolicInput:
        """Phase 1: Receive disputes for arbitration."""
        if not input_data.values:
            self._log.debug("empty_input")
            return input_data

        self._log.debug("intake_complete", value_count=len(input_data.values))
        return input_data

    async def process(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> list[tuple[Dispute, DeliberationResult | None, Verdict | None]]:
        """Phase 2 & 3: Deliberate and decide on disputes."""
        results: list[tuple[Dispute, DeliberationResult | None, Verdict | None]] = []

        for value in input_data.values:
            dispute = self._parse_dispute(value)
            if dispute:
                self._disputes[dispute.id] = dispute

                # Check if ready for deliberation
                if len(dispute.parties) >= 2 and (dispute.evidence or dispute.arguments):
                    # Deliberate
                    deliberation = self._deliberation.deliberate(dispute)

                    # Render verdict
                    verdict = self._render_verdict(dispute, deliberation)
                    self._verdicts[verdict.id] = verdict

                    # Update dispute status
                    dispute = Dispute(
                        id=dispute.id,
                        dispute_type=dispute.dispute_type,
                        status=DisputeStatus.DECIDED,
                        parties=dispute.parties,
                        subject=dispute.subject,
                        description=dispute.description,
                        disputed_value=dispute.disputed_value,
                        evidence=dispute.evidence,
                        arguments=dispute.arguments,
                        verdict=verdict,
                        related_conflict_id=dispute.related_conflict_id,
                        submitted_at=dispute.submitted_at,
                        deadline=dispute.deadline,
                        tags=dispute.tags,
                    )
                    self._disputes[dispute.id] = dispute

                    results.append((dispute, deliberation, verdict))
                else:
                    # Not ready for deliberation
                    results.append((dispute, None, None))

        return results

    async def evaluate(
        self, intermediate: list[tuple[Dispute, DeliberationResult | None, Verdict | None]],
        ctx: ProcessContext[dict[str, Any]],
    ) -> tuple[SymbolicOutput, bool]:
        """Phase 4: Create output with verdicts."""
        values: list[SymbolicValue] = []

        for dispute, deliberation, verdict in intermediate:
            if verdict:
                value = SymbolicValue(
                    type=SymbolicValueType.SCHEMA,
                    content={
                        "dispute_id": dispute.id,
                        "dispute_type": dispute.dispute_type.name,
                        "status": dispute.status.name,
                        "verdict_id": verdict.id,
                        "verdict_type": verdict.verdict_type.name,
                        "summary": verdict.summary,
                        "reasoning": verdict.reasoning,
                        "prevailing_party": verdict.prevailing_party_id,
                        "confidence": verdict.confidence,
                        "rules_applied": list(verdict.rules_applied),
                    },
                    source_subsystem=self.name,
                    tags=frozenset(["arbitration", "verdict", verdict.verdict_type.name.lower()]),
                    meaning=verdict.summary[:100],
                    confidence=verdict.confidence,
                )
            else:
                value = SymbolicValue(
                    type=SymbolicValueType.SCHEMA,
                    content={
                        "dispute_id": dispute.id,
                        "dispute_type": dispute.dispute_type.name,
                        "status": dispute.status.name,
                        "subject": dispute.subject,
                        "parties": len(dispute.parties),
                        "evidence_count": len(dispute.evidence),
                    },
                    source_subsystem=self.name,
                    tags=frozenset(["arbitration", "pending", dispute.status.name.lower()]),
                    meaning=f"Dispute pending: {dispute.subject[:50]}",
                    confidence=0.5,
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
        """Phase 4: Emit arbitration events."""
        if self._message_bus and output.values:
            for value in output.values:
                if value.content.get("verdict_id"):
                    await self.emit_event(
                        "arbitration.verdict.issued",
                        {
                            "dispute_id": value.content.get("dispute_id"),
                            "verdict_id": value.content.get("verdict_id"),
                            "verdict_type": value.content.get("verdict_type"),
                        },
                    )
                else:
                    await self.emit_event(
                        "arbitration.dispute.submitted",
                        {
                            "dispute_id": value.content.get("dispute_id"),
                            "dispute_type": value.content.get("dispute_type"),
                        },
                    )

        return None

    def _parse_dispute(self, value: SymbolicValue) -> Dispute | None:
        """Parse a Dispute from a SymbolicValue."""
        content = value.content
        if not isinstance(content, dict):
            return None

        try:
            dispute_type_str = content.get("dispute_type", "OTHER")
            try:
                dispute_type = DisputeType[dispute_type_str.upper()]
            except KeyError:
                dispute_type = DisputeType.OTHER

            # Parse parties
            parties: list[Party] = []
            for p_data in content.get("parties", []):
                parties.append(Party(
                    name=p_data.get("name", "Unknown"),
                    subsystem=p_data.get("subsystem"),
                    role=p_data.get("role", "claimant"),
                    position=p_data.get("position", ""),
                ))

            # Parse evidence
            evidence: list[Evidence] = []
            for e_data in content.get("evidence", []):
                evidence.append(Evidence(
                    party_id=e_data.get("party_id", ""),
                    evidence_type=e_data.get("evidence_type", "document"),
                    content=e_data.get("content"),
                    description=e_data.get("description", ""),
                    weight=e_data.get("weight", 1.0),
                    verified=e_data.get("verified", False),
                ))

            # Parse arguments
            arguments: list[Argument] = []
            for a_data in content.get("arguments", []):
                arguments.append(Argument(
                    party_id=a_data.get("party_id", ""),
                    claim=a_data.get("claim", ""),
                    supporting_evidence=tuple(a_data.get("supporting_evidence", [])),
                    strength=a_data.get("strength", 0.5),
                ))

            return Dispute(
                id=content.get("id", str(ULID())),
                dispute_type=dispute_type,
                parties=tuple(parties),
                subject=content.get("subject", ""),
                description=content.get("description", ""),
                disputed_value=content.get("disputed_value"),
                evidence=tuple(evidence),
                arguments=tuple(arguments),
                related_conflict_id=content.get("related_conflict_id"),
                tags=frozenset(content.get("tags", [])) | value.tags,
            )

        except Exception as e:
            self._log.warning("dispute_parse_failed", value_id=value.id, error=str(e))
            return None

    def _render_verdict(
        self,
        dispute: Dispute,
        deliberation: DeliberationResult,
    ) -> Verdict:
        """Render a verdict based on deliberation."""
        # Determine prevailing party
        prevailing_party_id = None
        if deliberation.scores:
            prevailing_party_id = max(deliberation.scores.items(), key=lambda x: x[1])[0]

        # Generate reasoning
        reasoning_parts = []
        if deliberation.rules_applied:
            reasoning_parts.append(f"Applied rules: {', '.join(deliberation.rules_applied)}")
        if deliberation.evidence_weighed > 0:
            reasoning_parts.append(f"Evaluated {deliberation.evidence_weighed} pieces of evidence")
        if deliberation.arguments_evaluated > 0:
            reasoning_parts.append(f"Considered {deliberation.arguments_evaluated} arguments")

        reasoning = "; ".join(reasoning_parts)

        # Generate summary
        summary_map = {
            VerdictType.IN_FAVOR_PARTY_A: "Verdict in favor of first party",
            VerdictType.IN_FAVOR_PARTY_B: "Verdict in favor of second party",
            VerdictType.COMPROMISE: "Compromise verdict reached",
            VerdictType.SPLIT_DECISION: "Split decision rendered",
            VerdictType.DISMISSED: "Dispute dismissed",
            VerdictType.DEFERRED: "Decision deferred",
        }
        summary = summary_map.get(deliberation.recommendation, "Verdict rendered")

        # Calculate confidence based on score differential
        score_values = list(deliberation.scores.values())
        if len(score_values) >= 2:
            sorted_scores = sorted(score_values, reverse=True)
            differential = sorted_scores[0] - sorted_scores[1]
            confidence = min(0.5 + differential, 0.95)
        else:
            confidence = 0.7

        return Verdict(
            dispute_id=dispute.id,
            verdict_type=deliberation.recommendation,
            summary=summary,
            reasoning=reasoning,
            prevailing_party_id=prevailing_party_id,
            awarded_value=dispute.disputed_value if prevailing_party_id else None,
            rules_applied=tuple(deliberation.rules_applied),
            confidence=confidence,
        )

    # --- Message handlers ---

    async def handle_event(self, message: Message) -> None:
        """Handle incoming events."""
        if message.topic == "conflict.escalated":
            self._log.info("conflict_escalated_received", message_id=message.id)

    async def handle_signal(self, signal: Any) -> None:
        """Handle incoming signals."""
        self._log.debug("signal_received", signal_id=getattr(signal, "id", "unknown"))

    # --- Public API ---

    def submit_dispute(
        self,
        subject: str,
        dispute_type: DisputeType,
        parties: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Dispute:
        """Submit a new dispute for arbitration."""
        party_objs = [
            Party(
                name=p.get("name", "Unknown"),
                subsystem=p.get("subsystem"),
                role=p.get("role", "claimant"),
                position=p.get("position", ""),
            )
            for p in parties
        ]

        dispute = Dispute(
            dispute_type=dispute_type,
            parties=tuple(party_objs),
            subject=subject,
            description=kwargs.get("description", ""),
            disputed_value=kwargs.get("disputed_value"),
            related_conflict_id=kwargs.get("related_conflict_id"),
            tags=frozenset(kwargs.get("tags", [])),
        )

        self._disputes[dispute.id] = dispute
        return dispute

    def add_evidence(
        self,
        dispute_id: str,
        party_id: str,
        content: Any,
        **kwargs: Any,
    ) -> Evidence | None:
        """Add evidence to a dispute."""
        dispute = self._disputes.get(dispute_id)
        if not dispute:
            return None

        evidence = Evidence(
            party_id=party_id,
            evidence_type=kwargs.get("evidence_type", "document"),
            content=content,
            description=kwargs.get("description", ""),
            weight=kwargs.get("weight", 1.0),
            verified=kwargs.get("verified", False),
        )

        # Update dispute with new evidence
        updated = Dispute(
            id=dispute.id,
            dispute_type=dispute.dispute_type,
            status=DisputeStatus.EVIDENCE_COLLECTION,
            parties=dispute.parties,
            subject=dispute.subject,
            description=dispute.description,
            disputed_value=dispute.disputed_value,
            evidence=(*dispute.evidence, evidence),
            arguments=dispute.arguments,
            verdict=dispute.verdict,
            related_conflict_id=dispute.related_conflict_id,
            submitted_at=dispute.submitted_at,
            deadline=dispute.deadline,
            tags=dispute.tags,
        )
        self._disputes[dispute_id] = updated

        return evidence

    def add_argument(
        self,
        dispute_id: str,
        party_id: str,
        claim: str,
        **kwargs: Any,
    ) -> Argument | None:
        """Add an argument to a dispute."""
        dispute = self._disputes.get(dispute_id)
        if not dispute:
            return None

        argument = Argument(
            party_id=party_id,
            claim=claim,
            supporting_evidence=tuple(kwargs.get("supporting_evidence", [])),
            strength=kwargs.get("strength", 0.5),
        )

        # Update dispute with new argument
        updated = Dispute(
            id=dispute.id,
            dispute_type=dispute.dispute_type,
            status=dispute.status,
            parties=dispute.parties,
            subject=dispute.subject,
            description=dispute.description,
            disputed_value=dispute.disputed_value,
            evidence=dispute.evidence,
            arguments=(*dispute.arguments, argument),
            verdict=dispute.verdict,
            related_conflict_id=dispute.related_conflict_id,
            submitted_at=dispute.submitted_at,
            deadline=dispute.deadline,
            tags=dispute.tags,
        )
        self._disputes[dispute_id] = updated

        return argument

    def deliberate(self, dispute_id: str) -> DeliberationResult | None:
        """Deliberate on a dispute."""
        dispute = self._disputes.get(dispute_id)
        if not dispute:
            return None

        return self._deliberation.deliberate(dispute)

    def render_verdict(self, dispute_id: str) -> Verdict | None:
        """Render a verdict for a dispute."""
        dispute = self._disputes.get(dispute_id)
        if not dispute:
            return None

        deliberation = self._deliberation.deliberate(dispute)
        verdict = self._render_verdict(dispute, deliberation)
        self._verdicts[verdict.id] = verdict

        # Update dispute
        updated = Dispute(
            id=dispute.id,
            dispute_type=dispute.dispute_type,
            status=DisputeStatus.DECIDED,
            parties=dispute.parties,
            subject=dispute.subject,
            description=dispute.description,
            disputed_value=dispute.disputed_value,
            evidence=dispute.evidence,
            arguments=dispute.arguments,
            verdict=verdict,
            related_conflict_id=dispute.related_conflict_id,
            submitted_at=dispute.submitted_at,
            deadline=dispute.deadline,
            tags=dispute.tags,
        )
        self._disputes[dispute_id] = updated

        return verdict

    def get_dispute(self, dispute_id: str) -> Dispute | None:
        """Get a dispute by ID."""
        return self._disputes.get(dispute_id)

    def get_verdict(self, verdict_id: str) -> Verdict | None:
        """Get a verdict by ID."""
        return self._verdicts.get(verdict_id)

    def get_disputes_by_type(self, dispute_type: DisputeType) -> list[Dispute]:
        """Get disputes by type."""
        return [d for d in self._disputes.values() if d.dispute_type == dispute_type]

    def get_pending_disputes(self) -> list[Dispute]:
        """Get all pending disputes."""
        return [
            d for d in self._disputes.values()
            if d.status not in (DisputeStatus.DECIDED, DisputeStatus.CLOSED, DisputeStatus.DISMISSED)
        ]

    def add_rule(self, rule: ArbitrationRule) -> None:
        """Add an arbitration rule."""
        self._deliberation.add_rule(rule)

    def get_stats(self) -> ArbitrationStats:
        """Get arbitration statistics."""
        by_type: dict[str, int] = {}
        pending = 0
        decided = 0
        dismissed = 0

        for dispute in self._disputes.values():
            by_type[dispute.dispute_type.name] = by_type.get(dispute.dispute_type.name, 0) + 1
            if dispute.status == DisputeStatus.DECIDED:
                decided += 1
            elif dispute.status == DisputeStatus.DISMISSED:
                dismissed += 1
            elif dispute.status not in (DisputeStatus.CLOSED,):
                pending += 1

        # Calculate average resolution time (rough estimate)
        resolved_times = []
        for dispute in self._disputes.values():
            if dispute.verdict and dispute.verdict.decided_at:
                delta = dispute.verdict.decided_at - dispute.submitted_at
                resolved_times.append(delta.total_seconds() / 3600)

        avg_time = sum(resolved_times) / len(resolved_times) if resolved_times else 0.0

        return ArbitrationStats(
            total_disputes=len(self._disputes),
            disputes_decided=decided,
            disputes_pending=pending,
            disputes_dismissed=dismissed,
            by_type=by_type,
            average_resolution_time_hours=avg_time,
        )

    def clear(self) -> tuple[int, int]:
        """Clear all disputes and verdicts. Returns (disputes_cleared, verdicts_cleared)."""
        disputes = len(self._disputes)
        verdicts = len(self._verdicts)
        self._disputes.clear()
        self._verdicts.clear()
        return disputes, verdicts
