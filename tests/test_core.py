"""Tests for core abstractions."""

import pytest

from autogenrec.core.symbolic import (
    SymbolicInput,
    SymbolicOutput,
    SymbolicValue,
    SymbolicValueType,
    narrative,
    pattern,
    reference,
    rule,
    token,
)
from autogenrec.core.signals import (
    Echo,
    Message,
    MessageType,
    Signal,
    SignalDomain,
    SignalPriority,
)
from autogenrec.bus.topics import Topic
from autogenrec.bus.message_bus import MessageBus


class TestSymbolicValue:
    def test_create_symbolic_value(self) -> None:
        value = SymbolicValue(
            type=SymbolicValueType.NARRATIVE,
            content="A story begins",
        )
        assert value.type == SymbolicValueType.NARRATIVE
        assert value.content == "A story begins"
        assert value.id is not None
        assert value.confidence == 1.0

    def test_derive_symbolic_value(self) -> None:
        original = SymbolicValue(
            type=SymbolicValueType.NARRATIVE,
            content="Original content",
            meaning="Test meaning",
        )
        derived = original.derive(
            content="Derived content",
            type=SymbolicValueType.PATTERN,
        )
        assert derived.parent_id == original.id
        assert derived.content == "Derived content"
        assert derived.type == SymbolicValueType.PATTERN
        assert derived.meaning == "Test meaning"

    def test_factory_functions(self) -> None:
        n = narrative("A tale", meaning="Symbolic story")
        assert n.type == SymbolicValueType.NARRATIVE

        r = rule({"condition": "x > 0"})
        assert r.type == SymbolicValueType.RULE

        t = token(42)
        assert t.type == SymbolicValueType.TOKEN

        p = pattern([1, 2, 3])
        assert p.type == SymbolicValueType.PATTERN

        ref = reference("target-id")
        assert ref.type == SymbolicValueType.REFERENCE
        assert "target-id" in ref.related_ids


class TestSymbolicInput:
    def test_create_input(self) -> None:
        value = narrative("Test")
        input_data = SymbolicInput(
            values=(value,),
            source_subsystem="test_source",
        )
        assert input_data.primary_value == value
        assert input_data.source_subsystem == "test_source"

    def test_with_values(self) -> None:
        v1 = narrative("First")
        v2 = narrative("Second")
        input_data = SymbolicInput(values=(v1,))
        extended = input_data.with_values(v2)
        assert len(extended.values) == 2


class TestSignal:
    def test_create_signal(self) -> None:
        signal = Signal(
            source="test_source",
            payload={"test": "data"},
        )
        assert signal.source == "test_source"
        assert signal.payload == {"test": "data"}
        assert signal.above_threshold

    def test_signal_threshold_validation(self) -> None:
        with pytest.raises(ValueError, match="below threshold"):
            Signal(
                source="test",
                payload="data",
                strength=0.3,
                threshold=0.5,
            )

    def test_attenuate_signal(self) -> None:
        signal = Signal(source="test", payload="data", strength=1.0)
        attenuated = signal.attenuate(0.5)
        assert attenuated.strength == 0.5
        assert attenuated.metadata.get("attenuated_from") == signal.id

    def test_convert_domain(self) -> None:
        signal = Signal(
            source="test",
            payload="data",
            domain=SignalDomain.ANALOG,
        )
        converted = signal.convert_domain(SignalDomain.DIGITAL)
        assert converted.domain == SignalDomain.DIGITAL
        assert converted.metadata.get("original_domain") == "ANALOG"


class TestEcho:
    def test_create_echo(self) -> None:
        signal = Signal(source="test", payload="data")
        echo = Echo(original_signal=signal)
        assert echo.original_signal == signal
        assert echo.replay_count == 1

    def test_echo_decay(self) -> None:
        signal = Signal(source="test", payload="data", strength=1.0)
        echo = Echo(original_signal=signal, decay_factor=0.9)
        assert echo.effective_strength == pytest.approx(0.9)

        replayed = echo.replay("subsystem")
        assert replayed.effective_strength == pytest.approx(0.81)

    def test_echo_to_signal(self) -> None:
        original = Signal(source="original", payload="data")
        echo = Echo(original_signal=original)
        signal = echo.to_signal("new_source")
        assert signal.source == "new_source"
        assert "echo" in signal.tags


class TestTopic:
    def test_topic_matching(self) -> None:
        topic = Topic("system.subsystem.event")
        assert topic.matches("system.subsystem.event")
        assert topic.matches("system.*.event")
        assert topic.matches("system.#")
        assert topic.matches("#")
        assert not topic.matches("other.subsystem.event")
        assert not topic.matches("system.subsystem")

    def test_topic_segments(self) -> None:
        topic = Topic("a.b.c")
        assert topic.segments == ["a", "b", "c"]
        assert topic.category == "a"


class TestMessageBus:
    @pytest.mark.asyncio
    async def test_publish_subscribe(self) -> None:
        bus = MessageBus()
        received: list[Message] = []

        async def handler(msg: Message) -> None:
            received.append(msg)

        await bus.subscribe("test.topic", handler)

        message = Message(
            type=MessageType.EVENT,
            topic="test.topic",
            payload="test data",
            source="test",
        )
        delivered = await bus.publish(message)

        assert delivered == 1
        assert len(received) == 1
        assert received[0].payload == "test data"

    @pytest.mark.asyncio
    async def test_wildcard_subscription(self) -> None:
        bus = MessageBus()
        received: list[Message] = []

        async def handler(msg: Message) -> None:
            received.append(msg)

        await bus.subscribe("test.#", handler)

        msg1 = Message.event("test.a", "src", "data1")
        msg2 = Message.event("test.b.c", "src", "data2")
        msg3 = Message.event("other.x", "src", "data3")

        await bus.publish(msg1)
        await bus.publish(msg2)
        await bus.publish(msg3)

        assert len(received) == 2

    @pytest.mark.asyncio
    async def test_unsubscribe(self) -> None:
        bus = MessageBus()
        received: list[Message] = []

        async def handler(msg: Message) -> None:
            received.append(msg)

        sub_id = await bus.subscribe("test", handler)
        await bus.unsubscribe(sub_id)

        await bus.publish(Message.event("test", "src", "data"))
        assert len(received) == 0
