"""Tests for conflict subsystems: ConflictResolver and ArbitrationEngine."""

import pytest
from datetime import UTC, datetime

from autogenrec.core.subsystem import SubsystemType
from autogenrec.core.symbolic import (
    SymbolicInput,
    SymbolicValue,
    SymbolicValueType,
)
from autogenrec.subsystems.conflict.conflict_resolver import (
    Conflict,
    ConflictDetector,
    ConflictResolver,
    ConflictSeverity,
    ConflictType,
    ConflictingValue,
    ResolutionEngine,
    ResolutionResult,
    ResolutionStrategy,
)
from autogenrec.subsystems.conflict.arbitration_engine import (
    ArbitrationEngine,
    ArbitrationRule,
    Argument,
    DeliberationEngine,
    Dispute,
    DisputeType,
    Evidence,
    Party,
    Verdict,
    VerdictType,
)


class TestConflictDetector:
    def test_detect_no_conflicts(self) -> None:
        detector = ConflictDetector()
        values = [
            SymbolicValue(
                type=SymbolicValueType.SCHEMA,
                content={"key": "value1"},
                source_subsystem="source1",
            ),
            SymbolicValue(
                type=SymbolicValueType.SCHEMA,
                content={"key": "value2"},
                source_subsystem="source2",
            ),
        ]

        conflicts = detector.detect(values)
        # Different sources, different content - no conflict
        assert len(conflicts) == 0

    def test_detect_same_source_conflict(self) -> None:
        detector = ConflictDetector()
        values = [
            SymbolicValue(
                type=SymbolicValueType.SCHEMA,
                content={"key": "value1"},
                source_subsystem="same_source",
            ),
            SymbolicValue(
                type=SymbolicValueType.SCHEMA,
                content={"key": "value2"},
                source_subsystem="same_source",
            ),
        ]

        conflicts = detector.detect(values)
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == ConflictType.DATA

    def test_detect_semantic_conflict(self) -> None:
        detector = ConflictDetector()
        values = [
            SymbolicValue(
                type=SymbolicValueType.SCHEMA,
                content={"data": 1},
                meaning="The result",
            ),
            SymbolicValue(
                type=SymbolicValueType.SCHEMA,
                content={"data": 2},
                meaning="The result",
            ),
        ]

        conflicts = detector.detect(values)
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == ConflictType.DATA


class TestResolutionEngine:
    def test_resolve_by_priority(self) -> None:
        engine = ResolutionEngine()
        conflict = Conflict(
            conflict_type=ConflictType.DATA,
            values=(
                ConflictingValue(value="low", source="a", priority=10),
                ConflictingValue(value="high", source="b", priority=90),
            ),
        )

        result = engine.resolve(conflict, ResolutionStrategy.PRIORITY)
        assert result.success
        assert result.resolved_value == "high"

    def test_resolve_by_timestamp(self) -> None:
        engine = ResolutionEngine()
        earlier = datetime(2024, 1, 1, tzinfo=UTC)
        later = datetime(2024, 6, 1, tzinfo=UTC)

        conflict = Conflict(
            conflict_type=ConflictType.DATA,
            values=(
                ConflictingValue(value="old", source="a", timestamp=earlier),
                ConflictingValue(value="new", source="b", timestamp=later),
            ),
        )

        result = engine.resolve(conflict, ResolutionStrategy.TIMESTAMP)
        assert result.success
        assert result.resolved_value == "new"

    def test_resolve_by_consensus_dicts(self) -> None:
        engine = ResolutionEngine()
        conflict = Conflict(
            conflict_type=ConflictType.DATA,
            values=(
                ConflictingValue(value={"a": 1}, source="x"),
                ConflictingValue(value={"b": 2}, source="y"),
            ),
        )

        result = engine.resolve(conflict, ResolutionStrategy.CONSENSUS)
        assert result.success
        assert result.resolved_value == {"a": 1, "b": 2}

    def test_escalate_to_arbitration(self) -> None:
        engine = ResolutionEngine()
        conflict = Conflict(
            conflict_type=ConflictType.RULE,
            values=(
                ConflictingValue(value="rule1", source="a"),
                ConflictingValue(value="rule2", source="b"),
            ),
        )

        result = engine.resolve(conflict, ResolutionStrategy.ARBITRATION)
        assert result.success
        assert result.escalated


class TestConflictResolver:
    def test_create_resolver(self) -> None:
        resolver = ConflictResolver()
        assert resolver.name == "conflict_resolver"
        assert resolver.metadata.type == SubsystemType.CONFLICT

    def test_detect_conflicts(self) -> None:
        resolver = ConflictResolver()
        values = [
            SymbolicValue(
                type=SymbolicValueType.SCHEMA,
                content={"key": "value1"},
                source_subsystem="same",
            ),
            SymbolicValue(
                type=SymbolicValueType.SCHEMA,
                content={"key": "value2"},
                source_subsystem="same",
            ),
        ]

        conflicts = resolver.detect_conflicts(values)
        assert len(conflicts) == 1

    def test_resolve_conflict(self) -> None:
        resolver = ConflictResolver()
        values = [
            SymbolicValue(
                type=SymbolicValueType.SCHEMA,
                content={"key": "value1"},
                source_subsystem="same",
            ),
            SymbolicValue(
                type=SymbolicValueType.SCHEMA,
                content={"key": "value2"},
                source_subsystem="same",
            ),
        ]

        conflicts = resolver.detect_conflicts(values)
        result = resolver.resolve_conflict(conflicts[0].id, ResolutionStrategy.PRIORITY)

        assert result is not None
        assert result.success

    def test_get_stats(self) -> None:
        resolver = ConflictResolver()
        values = [
            SymbolicValue(
                type=SymbolicValueType.SCHEMA,
                content={"key": "value1"},
                source_subsystem="same",
            ),
            SymbolicValue(
                type=SymbolicValueType.SCHEMA,
                content={"key": "value2"},
                source_subsystem="same",
            ),
        ]

        resolver.detect_conflicts(values)
        stats = resolver.get_stats()
        assert stats.total_detected == 1

    @pytest.mark.asyncio
    async def test_full_process_loop(self) -> None:
        resolver = ConflictResolver()
        from autogenrec.core.process import ProcessContext

        input_data = SymbolicInput(
            values=(
                SymbolicValue(
                    type=SymbolicValueType.SCHEMA,
                    content={"field": "value1"},
                    source_subsystem="source",
                ),
                SymbolicValue(
                    type=SymbolicValueType.SCHEMA,
                    content={"field": "value2"},
                    source_subsystem="source",
                ),
            )
        )
        ctx = ProcessContext[dict](iteration=1, started_at=datetime.now(UTC))

        filtered = await resolver.intake(input_data, ctx)
        results = await resolver.process(filtered, ctx)
        output, should_continue = await resolver.evaluate(results, ctx)

        assert not should_continue
        # Should have detected conflict
        assert len(output.values) >= 1


class TestDeliberationEngine:
    def test_deliberate_simple_dispute(self) -> None:
        engine = DeliberationEngine()

        party_a = Party(name="Party A", role="claimant", position="I am right")
        party_b = Party(name="Party B", role="respondent", position="No, I am right")

        dispute = Dispute(
            dispute_type=DisputeType.VALUE,
            parties=(party_a, party_b),
            subject="Test dispute",
            evidence=(
                Evidence(party_id=party_a.id, content="Evidence A", weight=0.8),
                Evidence(party_id=party_b.id, content="Evidence B", weight=0.6),
            ),
            arguments=(
                Argument(party_id=party_a.id, claim="Claim A", strength=0.7),
                Argument(party_id=party_b.id, claim="Claim B", strength=0.5),
            ),
        )

        result = engine.deliberate(dispute)
        assert result.dispute_id == dispute.id
        assert len(result.scores) == 2
        assert result.arguments_evaluated == 2
        assert result.evidence_weighed == 2


class TestArbitrationEngine:
    def test_create_engine(self) -> None:
        engine = ArbitrationEngine()
        assert engine.name == "arbitration_engine"
        assert engine.metadata.type == SubsystemType.CONFLICT

    def test_submit_dispute(self) -> None:
        engine = ArbitrationEngine()
        dispute = engine.submit_dispute(
            subject="Test dispute",
            dispute_type=DisputeType.VALUE,
            parties=[
                {"name": "Party A", "role": "claimant"},
                {"name": "Party B", "role": "respondent"},
            ],
        )

        assert dispute.subject == "Test dispute"
        assert len(dispute.parties) == 2
        assert engine.dispute_count == 1

    def test_add_evidence(self) -> None:
        engine = ArbitrationEngine()
        dispute = engine.submit_dispute(
            subject="Test",
            dispute_type=DisputeType.VALUE,
            parties=[{"name": "A"}, {"name": "B"}],
        )

        evidence = engine.add_evidence(
            dispute.id,
            dispute.parties[0].id,
            {"document": "proof"},
            description="Supporting evidence",
        )

        assert evidence is not None
        updated = engine.get_dispute(dispute.id)
        assert len(updated.evidence) == 1

    def test_add_argument(self) -> None:
        engine = ArbitrationEngine()
        dispute = engine.submit_dispute(
            subject="Test",
            dispute_type=DisputeType.VALUE,
            parties=[{"name": "A"}, {"name": "B"}],
        )

        argument = engine.add_argument(
            dispute.id,
            dispute.parties[0].id,
            claim="I should win",
            strength=0.8,
        )

        assert argument is not None
        updated = engine.get_dispute(dispute.id)
        assert len(updated.arguments) == 1

    def test_render_verdict(self) -> None:
        engine = ArbitrationEngine()
        dispute = engine.submit_dispute(
            subject="Test",
            dispute_type=DisputeType.VALUE,
            parties=[{"name": "A"}, {"name": "B"}],
        )

        # Add evidence and arguments
        engine.add_evidence(
            dispute.id, dispute.parties[0].id, {"proof": 1}, weight=0.9
        )
        engine.add_argument(
            dispute.id, dispute.parties[0].id, "Strong claim", strength=0.8
        )
        engine.add_evidence(
            dispute.id, dispute.parties[1].id, {"weak": True}, weight=0.3
        )
        engine.add_argument(
            dispute.id, dispute.parties[1].id, "Weak claim", strength=0.3
        )

        verdict = engine.render_verdict(dispute.id)
        assert verdict is not None
        assert verdict.verdict_type != VerdictType.DISMISSED

    def test_get_stats(self) -> None:
        engine = ArbitrationEngine()
        engine.submit_dispute(
            subject="Test 1",
            dispute_type=DisputeType.VALUE,
            parties=[{"name": "A"}, {"name": "B"}],
        )
        engine.submit_dispute(
            subject="Test 2",
            dispute_type=DisputeType.RULE,
            parties=[{"name": "C"}, {"name": "D"}],
        )

        stats = engine.get_stats()
        assert stats.total_disputes == 2
        assert stats.by_type["VALUE"] == 1
        assert stats.by_type["RULE"] == 1

    @pytest.mark.asyncio
    async def test_full_process_loop(self) -> None:
        engine = ArbitrationEngine()
        from autogenrec.core.process import ProcessContext

        input_data = SymbolicInput(
            values=(
                SymbolicValue(
                    type=SymbolicValueType.SCHEMA,
                    content={
                        "dispute_type": "VALUE",
                        "subject": "Test dispute",
                        "parties": [
                            {"name": "Party A", "role": "claimant", "position": "I am right"},
                            {"name": "Party B", "role": "respondent", "position": "No, I am"},
                        ],
                        "evidence": [
                            {"party_id": "", "content": "Evidence", "weight": 0.8},
                        ],
                        "arguments": [
                            {"party_id": "", "claim": "My claim", "strength": 0.7},
                        ],
                    },
                ),
            )
        )
        ctx = ProcessContext[dict](iteration=1, started_at=datetime.now(UTC))

        filtered = await engine.intake(input_data, ctx)
        results = await engine.process(filtered, ctx)
        output, should_continue = await engine.evaluate(results, ctx)

        assert not should_continue
        assert len(output.values) >= 1
