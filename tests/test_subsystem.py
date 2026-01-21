"""Tests for subsystem base class and orchestrator."""

import pytest

from autogenrec.core.subsystem import SubsystemMetadata, SubsystemType
from autogenrec.bus.message_bus import MessageBus
from autogenrec.core.registry import SubsystemRegistry
from autogenrec.subsystems.meta.anthology_manager import AnthologyManager
from autogenrec.runtime.orchestrator import Orchestrator, create_default_orchestrator


class TestSubsystemMetadata:
    def test_create_metadata(self) -> None:
        metadata = SubsystemMetadata(
            name="test_subsystem",
            display_name="Test Subsystem",
            description="A test subsystem",
            type=SubsystemType.CORE_PROCESSING,
            tags=frozenset(["test", "example"]),
        )
        assert metadata.name == "test_subsystem"
        assert metadata.type == SubsystemType.CORE_PROCESSING
        assert "test" in metadata.tags


class TestAnthologyManager:
    def test_create_anthology_manager(self) -> None:
        manager = AnthologyManager()
        assert manager.name == "anthology_manager"
        assert manager.metadata.type == SubsystemType.META

    def test_subsystem_names(self) -> None:
        manager = AnthologyManager()
        names = manager.get_subsystem_names()
        assert len(names) == 21
        assert "symbolic_interpreter" in names


class TestSubsystemRegistry:
    def test_register_subsystem(self) -> None:
        registry = SubsystemRegistry()
        manager = AnthologyManager()
        registry.register(manager)
        assert "anthology_manager" in registry
        assert registry.get("anthology_manager") == manager

    def test_get_by_type(self) -> None:
        registry = SubsystemRegistry()
        manager = AnthologyManager()
        registry.register(manager)
        meta_subsystems = registry.get_by_type(SubsystemType.META)
        assert len(meta_subsystems) == 1
        assert meta_subsystems[0] == manager


class TestOrchestrator:
    def test_create_orchestrator(self) -> None:
        orchestrator = Orchestrator()
        assert orchestrator.state.name == "CREATED"
        assert orchestrator.message_bus is not None

    def test_create_default_orchestrator(self) -> None:
        orchestrator = create_default_orchestrator()
        # Check that factories are registered
        assert "anthology_manager" in orchestrator._subsystem_factories
        assert "symbolic_interpreter" in orchestrator._subsystem_factories
        assert len(orchestrator._subsystem_factories) == 22  # 21 subsystems + anthology

    @pytest.mark.asyncio
    async def test_orchestrator_lifecycle(self) -> None:
        orchestrator = create_default_orchestrator()
        async with orchestrator.run_context():
            assert orchestrator.is_running
            health = orchestrator.get_health()
            assert health["state"] == "RUNNING"
            assert health["subsystems"]["total"] == 22

        assert orchestrator.state.name == "STOPPED"
