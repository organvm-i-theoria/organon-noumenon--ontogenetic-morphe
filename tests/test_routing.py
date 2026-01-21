"""Tests for routing subsystems: NodeRouter and SignalThresholdGuard."""

import pytest
from datetime import UTC, datetime

from autogenrec.core.subsystem import SubsystemType
from autogenrec.core.signals import Signal, SignalDomain
from autogenrec.core.symbolic import (
    SymbolicInput,
    SymbolicValue,
    SymbolicValueType,
)
from autogenrec.subsystems.routing.node_router import (
    Node,
    NodeRouter,
    NodeStatus,
    NodeType,
    Route,
    RouteOptimizer,
    RouteStatus,
    RouteType,
    RoutingTable,
)
from autogenrec.subsystems.routing.signal_threshold_guard import (
    ConversionMode,
    ConversionPolicy,
    DomainConverter,
    SignalThresholdGuard,
    ThresholdPolicy,
    ThresholdType,
    ThresholdValidator,
    ValidationResult,
)


class TestRoutingTable:
    def test_add_and_get_node(self) -> None:
        table = RoutingTable()
        node = Node(name="test_node", node_type=NodeType.SUBSYSTEM)
        table.add_node(node)

        retrieved = table.get_node(node.id)
        assert retrieved is not None
        assert retrieved.name == "test_node"

    def test_get_node_by_name(self) -> None:
        table = RoutingTable()
        node = Node(name="named_node", node_type=NodeType.ENDPOINT)
        table.add_node(node)

        retrieved = table.get_node_by_name("named_node")
        assert retrieved is not None
        assert retrieved.id == node.id

    def test_add_and_get_route(self) -> None:
        table = RoutingTable()
        source = Node(name="source", node_type=NodeType.SUBSYSTEM)
        target = Node(name="target", node_type=NodeType.SUBSYSTEM)
        table.add_node(source)
        table.add_node(target)

        route = Route(
            name="test_route",
            source_id=source.id,
            target_ids=(target.id,),
            topic_pattern="test.*",
        )
        table.add_route(route)

        retrieved = table.get_route(route.id)
        assert retrieved is not None
        assert retrieved.name == "test_route"

    def test_add_route_invalid_nodes(self) -> None:
        table = RoutingTable()
        source = Node(name="source", node_type=NodeType.SUBSYSTEM)
        table.add_node(source)

        route = Route(
            name="invalid_route",
            source_id=source.id,
            target_ids=("nonexistent",),
        )

        with pytest.raises(ValueError):
            table.add_route(route)

    def test_find_routes(self) -> None:
        table = RoutingTable()
        source = Node(name="source", node_type=NodeType.SUBSYSTEM)
        target1 = Node(name="target1", node_type=NodeType.SUBSYSTEM)
        target2 = Node(name="target2", node_type=NodeType.SUBSYSTEM)
        table.add_node(source)
        table.add_node(target1)
        table.add_node(target2)

        route1 = Route(
            name="route1",
            source_id=source.id,
            target_ids=(target1.id,),
            topic_pattern="events.*",
            priority=75,
        )
        route2 = Route(
            name="route2",
            source_id=source.id,
            target_ids=(target2.id,),
            topic_pattern="signals.*",
            priority=50,
        )
        table.add_route(route1)
        table.add_route(route2)

        # Find by source
        routes = table.find_routes(source_id=source.id)
        assert len(routes) == 2

        # Find by topic
        routes = table.find_routes(topic="events.test")
        assert len(routes) == 1
        assert routes[0].name == "route1"

    def test_remove_node_removes_routes(self) -> None:
        table = RoutingTable()
        source = Node(name="source", node_type=NodeType.SUBSYSTEM)
        target = Node(name="target", node_type=NodeType.SUBSYSTEM)
        table.add_node(source)
        table.add_node(target)

        route = Route(
            name="route",
            source_id=source.id,
            target_ids=(target.id,),
        )
        table.add_route(route)

        assert table.route_count == 1
        table.remove_node(source.id)
        assert table.route_count == 0


class TestRouteOptimizer:
    def test_select_highest_priority(self) -> None:
        optimizer = RouteOptimizer()

        routes = [
            Route(name="low", source_id="s", target_ids=("t",), priority=25),
            Route(name="high", source_id="s", target_ids=("t",), priority=75),
            Route(name="medium", source_id="s", target_ids=("t",), priority=50),
        ]

        selected = optimizer.select_route(routes)
        assert selected is not None
        assert selected.name == "high"

    def test_select_failover_route(self) -> None:
        optimizer = RouteOptimizer()

        routes = [
            Route(
                name="primary",
                source_id="s",
                target_ids=("t",),
                route_type=RouteType.FAILOVER,
                priority=100,
            ),
            Route(
                name="backup",
                source_id="s",
                target_ids=("t2",),
                route_type=RouteType.FAILOVER,
                priority=50,
            ),
        ]

        selected = optimizer.select_route(routes)
        assert selected is not None
        assert selected.name == "primary"

    def test_skip_disabled_routes(self) -> None:
        optimizer = RouteOptimizer()

        routes = [
            Route(
                name="disabled",
                source_id="s",
                target_ids=("t",),
                status=RouteStatus.DISABLED,
                priority=100,
            ),
            Route(
                name="active",
                source_id="s",
                target_ids=("t",),
                status=RouteStatus.ACTIVE,
                priority=50,
            ),
        ]

        selected = optimizer.select_route(routes)
        assert selected is not None
        assert selected.name == "active"


class TestNodeRouter:
    def test_create_router(self) -> None:
        router = NodeRouter()
        assert router.name == "node_router"
        assert router.metadata.type == SubsystemType.ROUTING

    def test_add_node(self) -> None:
        router = NodeRouter()
        node = router.add_node("test_node", NodeType.SUBSYSTEM)

        assert node.name == "test_node"
        assert router.node_count == 1

    def test_add_route(self) -> None:
        router = NodeRouter()
        source = router.add_node("source", NodeType.SUBSYSTEM)
        target = router.add_node("target", NodeType.SUBSYSTEM)

        route = router.add_route(
            "test_route",
            source.id,
            [target.id],
            RouteType.DIRECT,
            topic_pattern="test.*",
        )

        assert route.name == "test_route"
        assert router.route_count == 1

    def test_route_message(self) -> None:
        router = NodeRouter()
        source = router.add_node("source", NodeType.SUBSYSTEM)
        target = router.add_node("target", NodeType.SUBSYSTEM)

        router.add_route(
            "route",
            source.id,
            [target.id],
            topic_pattern="events.#",
        )

        targets = router.route("source", "events.test")
        assert len(targets) == 1
        assert targets[0] == "target"

    def test_route_no_match(self) -> None:
        router = NodeRouter()
        source = router.add_node("source", NodeType.SUBSYSTEM)
        target = router.add_node("target", NodeType.SUBSYSTEM)

        router.add_route(
            "route",
            source.id,
            [target.id],
            topic_pattern="events.*",
        )

        targets = router.route("source", "signals.test")
        assert len(targets) == 0

    @pytest.mark.asyncio
    async def test_full_process_loop(self) -> None:
        router = NodeRouter()
        from autogenrec.core.process import ProcessContext

        # First register nodes
        source = router.add_node("source", NodeType.SUBSYSTEM)
        target = router.add_node("target", NodeType.SUBSYSTEM)
        router.add_route("route", source.id, [target.id], topic_pattern="*")

        # Then process a signal routing request
        input_data = SymbolicInput(
            values=(
                SymbolicValue(
                    type=SymbolicValueType.SIGNAL,
                    content={
                        "source": "source",
                        "topic": "test.event",
                        "payload": {"data": "test"},
                    },
                ),
            )
        )
        ctx = ProcessContext[dict](iteration=1, started_at=datetime.now(UTC))

        filtered = await router.intake(input_data, ctx)
        decisions = await router.process(filtered, ctx)
        output, should_continue = await router.evaluate(decisions, ctx)

        assert not should_continue
        assert len(output.values) == 1
        assert output.values[0].content["target_nodes"] == ["target"]


class TestThresholdValidator:
    def test_validate_passes(self) -> None:
        validator = ThresholdValidator()
        signal = Signal(
            source="test",
            payload={"data": 1},
            payload_type="dict",
            strength=0.8,
        )

        report = validator.validate(signal)
        assert report.result in (ValidationResult.PASSED, ValidationResult.WARNING)

    def test_validate_weak_signal(self) -> None:
        validator = ThresholdValidator()
        signal = Signal(
            source="test",
            payload={"data": 1},
            payload_type="dict",
            strength=0.05,
            metadata={"allow_weak": True},
        )

        report = validator.validate(signal)
        # Should be adjusted since strength is below minimum
        assert report.result in (ValidationResult.ADJUSTED, ValidationResult.WARNING)

    def test_add_custom_policy(self) -> None:
        validator = ThresholdValidator()
        policy = ThresholdPolicy(
            name="strict_strength",
            threshold_type=ThresholdType.STRENGTH,
            min_value=0.9,
            action_on_fail="block",
            priority=200,
        )
        validator.add_policy(policy)

        signal = Signal(
            source="test",
            payload={},
            payload_type="dict",
            strength=0.7,
        )

        report = validator.validate(signal)
        assert report.result == ValidationResult.BLOCKED


class TestDomainConverter:
    def test_convert_analog_to_digital(self) -> None:
        converter = DomainConverter()
        signal = Signal(
            domain=SignalDomain.ANALOG,
            source="test",
            payload={"data": 1},
            payload_type="dict",
            strength=0.8,
        )

        converted, report = converter.convert(signal, SignalDomain.DIGITAL)

        assert report.success
        assert converted is not None
        assert converted.domain == SignalDomain.DIGITAL

    def test_convert_digital_to_analog(self) -> None:
        converter = DomainConverter()
        signal = Signal(
            domain=SignalDomain.DIGITAL,
            source="test",
            payload={"data": 1},
            payload_type="dict",
            strength=0.9,
        )

        converted, report = converter.convert(signal, SignalDomain.ANALOG)

        assert report.success
        assert converted is not None
        assert converted.domain == SignalDomain.ANALOG

    def test_no_conversion_same_domain(self) -> None:
        converter = DomainConverter()
        signal = Signal(
            domain=SignalDomain.DIGITAL,
            source="test",
            payload={},
            payload_type="dict",
        )

        converted, report = converter.convert(signal, SignalDomain.DIGITAL)

        assert report.success
        assert converted == signal  # Same object since no conversion needed


class TestSignalThresholdGuard:
    def test_create_guard(self) -> None:
        guard = SignalThresholdGuard()
        assert guard.name == "signal_threshold_guard"
        assert guard.metadata.type == SubsystemType.ROUTING

    def test_validate_signal(self) -> None:
        guard = SignalThresholdGuard()
        signal = Signal(
            source="test",
            payload={"data": 1},
            payload_type="dict",
            strength=0.8,
        )

        report = guard.validate(signal)
        assert report.result in (ValidationResult.PASSED, ValidationResult.WARNING, ValidationResult.ADJUSTED)

    def test_convert_signal(self) -> None:
        guard = SignalThresholdGuard()
        signal = Signal(
            domain=SignalDomain.ANALOG,
            source="test",
            payload={"data": 1},
            payload_type="dict",
            strength=0.8,
        )

        converted, report = guard.convert(signal, SignalDomain.DIGITAL)

        assert report.success
        assert converted is not None
        assert converted.domain == SignalDomain.DIGITAL

    def test_validate_and_convert(self) -> None:
        guard = SignalThresholdGuard()
        signal = Signal(
            domain=SignalDomain.ANALOG,
            source="test",
            payload={"data": 1},
            payload_type="dict",
            strength=0.8,
        )

        converted, validation, conversion = guard.validate_and_convert(
            signal, SignalDomain.DIGITAL
        )

        assert converted is not None
        assert validation.result in (ValidationResult.PASSED, ValidationResult.WARNING, ValidationResult.ADJUSTED)
        assert conversion is not None
        assert conversion.success

    def test_get_stats(self) -> None:
        guard = SignalThresholdGuard()
        signal = Signal(
            source="test",
            payload={},
            payload_type="dict",
            strength=0.9,
        )

        guard.validate(signal)
        guard.validate(signal)

        stats = guard.get_stats()
        assert stats.total_validated == 2

    @pytest.mark.asyncio
    async def test_full_process_loop(self) -> None:
        guard = SignalThresholdGuard()
        from autogenrec.core.process import ProcessContext

        input_data = SymbolicInput(
            values=(
                SymbolicValue(
                    type=SymbolicValueType.SIGNAL,
                    content={
                        "signal": {
                            "domain": "ANALOG",
                            "source": "test_source",
                            "payload": {"data": "test"},
                            "strength": 0.8,
                        },
                    },
                ),
            ),
            metadata={"target_domain": "DIGITAL"},
        )
        ctx = ProcessContext[dict](iteration=1, started_at=datetime.now(UTC), metadata={"target_domain": "DIGITAL"})

        filtered = await guard.intake(input_data, ctx)
        results = await guard.process(filtered, ctx)
        output, should_continue = await guard.evaluate(results, ctx)

        assert not should_continue
        assert len(output.values) == 1
        assert output.values[0].type == SymbolicValueType.SIGNAL
