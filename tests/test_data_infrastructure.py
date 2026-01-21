"""Tests for data infrastructure subsystems: ReferenceManager, ArchiveManager, EchoHandler."""

import pytest
from datetime import UTC, datetime, timedelta

from autogenrec.core.subsystem import SubsystemType
from autogenrec.core.signals import Signal, SignalDomain
from autogenrec.core.symbolic import (
    SymbolicInput,
    SymbolicValue,
    SymbolicValueType,
)
from autogenrec.subsystems.data.reference_manager import (
    Reference,
    ReferenceEdge,
    ReferenceGraph,
    ReferenceManager,
    ReferenceQuery,
    ReferenceStatus,
    ReferenceType,
    ReferenceValidator,
    ValidationResult,
)
from autogenrec.subsystems.data.archive_manager import (
    ArchiveCategory,
    ArchiveManager,
    ArchiveRecord,
    ArchiveStatus,
    RetentionPolicy,
    SearchQuery,
)
from autogenrec.subsystems.data.echo_handler import (
    CapturedSignal,
    EchoHandler,
    EchoState,
    ReplayStrategy,
    SignalBuffer,
)


class TestReferenceGraph:
    def test_add_and_get_reference(self) -> None:
        graph = ReferenceGraph()
        ref = Reference(name="test_ref", content={"data": "test"})
        graph.add_reference(ref)

        retrieved = graph.get_reference(ref.id)
        assert retrieved is not None
        assert retrieved.name == "test_ref"

    def test_get_by_name(self) -> None:
        graph = ReferenceGraph()
        ref1 = Reference(name="MyRef", content="data1")
        ref2 = Reference(name="myref", content="data2")  # Case insensitive
        graph.add_reference(ref1)
        graph.add_reference(ref2)

        refs = graph.get_by_name("MYREF")
        assert len(refs) == 2

    def test_add_and_get_edge(self) -> None:
        graph = ReferenceGraph()
        ref1 = Reference(name="source", content="s")
        ref2 = Reference(name="target", content="t")
        graph.add_reference(ref1)
        graph.add_reference(ref2)

        edge = ReferenceEdge(
            source_id=ref1.id,
            target_id=ref2.id,
            edge_type="relates_to",
        )
        graph.add_edge(edge)

        assert graph.edge_count == 1

    def test_get_related(self) -> None:
        graph = ReferenceGraph()
        ref1 = Reference(name="source", content="s")
        ref2 = Reference(name="target", content="t")
        graph.add_reference(ref1)
        graph.add_reference(ref2)

        edge = ReferenceEdge(
            source_id=ref1.id,
            target_id=ref2.id,
            edge_type="cites",
        )
        graph.add_edge(edge)

        related = graph.get_related(ref1.id)
        assert len(related) == 1
        assert related[0].id == ref2.id

        # Filter by edge type
        related_cites = graph.get_related(ref1.id, edge_type="cites")
        assert len(related_cites) == 1

        related_other = graph.get_related(ref1.id, edge_type="derives_from")
        assert len(related_other) == 0

    def test_search(self) -> None:
        graph = ReferenceGraph()
        ref1 = Reference(
            name="canonical_source",
            content="data",
            ref_type=ReferenceType.CANONICAL,
            tags=frozenset(["important"]),
        )
        ref2 = Reference(
            name="citation",
            content="data",
            ref_type=ReferenceType.CITATION,
        )
        graph.add_reference(ref1)
        graph.add_reference(ref2)

        # Search by type
        result = graph.search(ReferenceQuery(ref_type=ReferenceType.CANONICAL))
        assert result.total_count == 1
        assert result.references[0].name == "canonical_source"

        # Search by name
        result = graph.search(ReferenceQuery(name_contains="source"))
        assert result.total_count == 1

        # Search by tags
        result = graph.search(ReferenceQuery(tags={"important"}))
        assert result.total_count == 1

    def test_remove_reference(self) -> None:
        graph = ReferenceGraph()
        ref = Reference(name="test", content="data")
        graph.add_reference(ref)
        assert graph.reference_count == 1

        removed = graph.remove_reference(ref.id)
        assert removed
        assert graph.reference_count == 0


class TestReferenceValidator:
    def test_validate_valid_reference(self) -> None:
        validator = ReferenceValidator()
        graph = ReferenceGraph()
        ref = Reference(name="valid_ref", content={"data": "test"})
        graph.add_reference(ref)

        result = validator.validate(ref, graph)
        assert result == ValidationResult.VALID

    def test_validate_incomplete_reference(self) -> None:
        validator = ReferenceValidator()
        graph = ReferenceGraph()
        ref = Reference(name="", content={"data": "test"})

        result = validator.validate(ref, graph)
        assert result == ValidationResult.INCOMPLETE

    def test_validate_with_missing_parent(self) -> None:
        validator = ReferenceValidator()
        graph = ReferenceGraph()
        ref = Reference(
            name="child",
            content="data",
            parent_id="nonexistent",
        )
        graph.add_reference(ref)

        result = validator.validate(ref, graph)
        assert result == ValidationResult.INVALID


class TestReferenceManager:
    def test_create_manager(self) -> None:
        manager = ReferenceManager()
        assert manager.name == "reference_manager"
        assert manager.metadata.type == SubsystemType.DATA

    def test_create_reference(self) -> None:
        manager = ReferenceManager()
        ref = manager.create_reference(
            name="new_ref",
            content={"key": "value"},
            ref_type=ReferenceType.CANONICAL,
            tags=["test"],
        )

        assert ref.name == "new_ref"
        assert manager.reference_count == 1

    def test_resolve(self) -> None:
        manager = ReferenceManager()
        ref = manager.create_reference("test", {"data": 1})

        resolved = manager.resolve(ref.id)
        assert resolved is not None
        assert resolved.name == "test"

    def test_link_references(self) -> None:
        manager = ReferenceManager()
        ref1 = manager.create_reference("source", {"data": 1})
        ref2 = manager.create_reference("target", {"data": 2})

        edge = manager.link(ref1.id, ref2.id, "cites")
        assert edge is not None
        assert manager.edge_count == 1

    @pytest.mark.asyncio
    async def test_full_process_loop(self) -> None:
        manager = ReferenceManager()
        from autogenrec.core.process import ProcessContext

        input_data = SymbolicInput(
            values=(
                SymbolicValue(
                    type=SymbolicValueType.REFERENCE,
                    content={
                        "name": "test_reference",
                        "content": {"key": "value"},
                        "ref_type": "CANONICAL",
                    },
                ),
            )
        )
        ctx = ProcessContext[dict](iteration=1, started_at=datetime.now(UTC))

        filtered = await manager.intake(input_data, ctx)
        results = await manager.process(filtered, ctx)
        output, should_continue = await manager.evaluate(results, ctx)

        assert not should_continue
        assert len(output.values) == 1
        assert manager.reference_count == 1


class TestArchiveManager:
    def test_create_manager(self) -> None:
        manager = ArchiveManager()
        assert manager.name == "archive_manager"
        assert manager.metadata.type == SubsystemType.DATA

    def test_archive_and_retrieve(self) -> None:
        manager = ArchiveManager()
        record = manager.archive(
            title="Test Record",
            content={"data": "test"},
            category=ArchiveCategory.DATA,
            tags=["test"],
        )

        assert record.title == "Test Record"
        assert manager.record_count == 1

        retrieved = manager.retrieve(record.id)
        assert retrieved is not None
        assert retrieved.title == "Test Record"

    def test_search_text(self) -> None:
        manager = ArchiveManager()
        manager.archive("Important Document", {"text": "critical information"})
        manager.archive("Another Record", {"text": "unrelated data"})

        result = manager.search_text("important")
        assert result.total_count == 1
        assert result.records[0].title == "Important Document"

    def test_search_by_category(self) -> None:
        manager = ArchiveManager()
        manager.archive("System Log", {"log": "data"}, category=ArchiveCategory.SYSTEM)
        manager.archive("Audit Trail", {"audit": "data"}, category=ArchiveCategory.AUDIT)

        result = manager.search(SearchQuery(category=ArchiveCategory.AUDIT))
        assert result.total_count == 1
        assert result.records[0].title == "Audit Trail"

    def test_delete(self) -> None:
        manager = ArchiveManager()
        record = manager.archive("To Delete", {"data": "temp"})
        assert manager.record_count == 1

        deleted = manager.delete(record.id)
        assert deleted
        assert manager.record_count == 0

    def test_retention_policy(self) -> None:
        manager = ArchiveManager()
        policy = RetentionPolicy(
            name="short_retention",
            category=ArchiveCategory.DATA,
            retention_days=30,
            archive_after_days=7,
            auto_delete=True,
            priority=10,
        )
        manager.add_retention_policy(policy)

        record = manager.archive("Test", {"data": 1}, category=ArchiveCategory.DATA)
        assert record.expires_at is not None

    def test_get_stats(self) -> None:
        manager = ArchiveManager()
        manager.archive("Record 1", {}, category=ArchiveCategory.DATA)
        manager.archive("Record 2", {}, category=ArchiveCategory.SYSTEM)

        stats = manager.get_stats()
        assert stats.total_records == 2
        assert stats.active_records == 2
        assert stats.by_category["DATA"] == 1
        assert stats.by_category["SYSTEM"] == 1

    @pytest.mark.asyncio
    async def test_full_process_loop(self) -> None:
        manager = ArchiveManager()
        from autogenrec.core.process import ProcessContext

        input_data = SymbolicInput(
            values=(
                SymbolicValue(
                    type=SymbolicValueType.ARCHIVE,
                    content={
                        "title": "Archived Item",
                        "content": {"important": "data"},
                        "category": "DATA",
                    },
                ),
            )
        )
        ctx = ProcessContext[dict](iteration=1, started_at=datetime.now(UTC))

        filtered = await manager.intake(input_data, ctx)
        records = await manager.process(filtered, ctx)
        output, should_continue = await manager.evaluate(records, ctx)

        assert not should_continue
        assert len(output.values) == 1
        assert manager.record_count == 1


class TestSignalBuffer:
    def test_add_and_get(self) -> None:
        buffer = SignalBuffer()
        signal = Signal(
            source="test",
            payload={"data": "test"},
            payload_type="dict",
        )
        captured = CapturedSignal(signal=signal, topic="test.topic")
        buffer.add(captured)

        retrieved = buffer.get(captured.id)
        assert retrieved is not None
        assert retrieved.topic == "test.topic"

    def test_buffer_eviction(self) -> None:
        buffer = SignalBuffer(max_size=2)
        for i in range(3):
            signal = Signal(source=f"source_{i}", payload=i, payload_type="int")
            captured = CapturedSignal(signal=signal, topic=f"topic_{i}")
            buffer.add(captured)

        assert buffer.size == 2

    def test_get_by_state(self) -> None:
        buffer = SignalBuffer()
        signal = Signal(source="test", payload=1, payload_type="int")
        captured = CapturedSignal(signal=signal, topic="topic", state=EchoState.CAPTURED)
        buffer.add(captured)

        by_state = buffer.get_by_state(EchoState.CAPTURED)
        assert len(by_state) == 1

    def test_update_state(self) -> None:
        buffer = SignalBuffer()
        signal = Signal(source="test", payload=1, payload_type="int")
        captured = CapturedSignal(signal=signal, topic="topic")
        buffer.add(captured)

        buffer.update_state(captured.id, EchoState.SCHEDULED)
        retrieved = buffer.get(captured.id)
        assert retrieved.state == EchoState.SCHEDULED


class TestEchoHandler:
    def test_create_handler(self) -> None:
        handler = EchoHandler()
        assert handler.name == "echo_handler"
        assert handler.metadata.type == SubsystemType.DATA

    def test_capture_signal(self) -> None:
        handler = EchoHandler()
        signal = Signal(
            source="test_source",
            payload={"data": "test"},
            payload_type="dict",
        )
        captured = handler.capture(signal, "test.topic")

        assert captured is not None
        assert handler.captured_count == 1

    def test_schedule_replay(self) -> None:
        handler = EchoHandler()
        signal = Signal(source="test", payload=1, payload_type="int")
        captured = handler.capture(signal, "test.topic")

        schedule = handler.schedule_replay(captured.id, delay_seconds=60)
        assert schedule is not None
        assert handler.scheduled_count == 1

    @pytest.mark.asyncio
    async def test_replay(self) -> None:
        handler = EchoHandler()
        signal = Signal(source="test", payload={"msg": "hello"}, payload_type="dict")
        captured = handler.capture(signal, "test.topic")

        result = await handler.replay(captured.id)
        assert result.success
        assert result.echo is not None
        assert result.echo.replay_count == 1

    def test_signal_decay(self) -> None:
        handler = EchoHandler()
        signal = Signal(source="test", payload=1, payload_type="int", strength=1.0)
        captured = handler.capture(
            signal,
            "test.topic",
            decay_factor=0.5,
            min_strength=0.1,
        )

        assert captured.effective_strength == 1.0
        assert captured.is_viable

    def test_get_viable_for_replay(self) -> None:
        handler = EchoHandler()
        signal1 = Signal(source="test1", payload=1, payload_type="int", strength=1.0)
        signal2 = Signal(source="test2", payload=2, payload_type="int", strength=0.6)

        handler.capture(signal1, "topic1", min_strength=0.1)
        handler.capture(
            signal2, "topic2", min_strength=0.1,
            strategy=ReplayStrategy.IMMEDIATE,
        )

        # Second signal is too weak
        viable = handler.get_viable_for_replay()
        # Both should be viable initially (before decay)
        assert len(viable) == 2

    def test_get_stats(self) -> None:
        handler = EchoHandler()
        signal = Signal(source="test", payload=1, payload_type="int")
        handler.capture(signal, "test.topic")

        stats = handler.get_stats()
        assert stats.total_captured == 1
        assert stats.pending_replays == 1

    @pytest.mark.asyncio
    async def test_full_process_loop(self) -> None:
        handler = EchoHandler()
        from autogenrec.core.process import ProcessContext

        input_data = SymbolicInput(
            values=(
                SymbolicValue(
                    type=SymbolicValueType.SIGNAL,
                    content={
                        "signal": {
                            "source": "test_source",
                            "payload": {"data": "test"},
                            "payload_type": "dict",
                            "strength": 0.9,
                        },
                        "topic": "test.topic",
                    },
                ),
            )
        )
        ctx = ProcessContext[dict](iteration=1, started_at=datetime.now(UTC))

        filtered = await handler.intake(input_data, ctx)
        captured = await handler.process(filtered, ctx)
        output, should_continue = await handler.evaluate(captured, ctx)

        assert not should_continue
        assert len(output.values) == 1
        assert handler.captured_count == 1
