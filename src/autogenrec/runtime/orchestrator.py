"""
Orchestrator: System startup, shutdown, and lifecycle management.

Coordinates all subsystems and manages the overall system lifecycle.
"""

from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from enum import Enum, auto
from typing import Any, AsyncIterator

import anyio
import structlog

from autogenrec.bus.message_bus import MessageBus
from autogenrec.bus.topics import SystemTopics
from autogenrec.core.registry import ProcessRegistry, SubsystemRegistry
from autogenrec.core.signals import Message
from autogenrec.core.subsystem import Subsystem, SubsystemState

logger = structlog.get_logger()


class OrchestratorState(Enum):
    """Lifecycle states for the orchestrator."""

    CREATED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    FAILED = auto()


SubsystemFactory = Callable[[], Subsystem]


class Orchestrator:
    """
    Manages system startup, shutdown, and overall lifecycle.

    Responsibilities:
    - Initialize shared resources (message bus, registries)
    - Start and stop subsystems in proper order
    - Handle graceful shutdown
    - Provide system health monitoring
    """

    def __init__(self) -> None:
        self._state = OrchestratorState.CREATED
        self._message_bus = MessageBus()
        self._process_registry = ProcessRegistry()
        self._subsystem_registry = SubsystemRegistry()
        self._subsystem_factories: dict[str, SubsystemFactory] = {}
        self._started_at: datetime | None = None
        self._stopped_at: datetime | None = None
        self._log = logger.bind(component="orchestrator")

    @property
    def state(self) -> OrchestratorState:
        return self._state

    @property
    def message_bus(self) -> MessageBus:
        return self._message_bus

    @property
    def process_registry(self) -> ProcessRegistry:
        return self._process_registry

    @property
    def subsystem_registry(self) -> SubsystemRegistry:
        return self._subsystem_registry

    @property
    def is_running(self) -> bool:
        return self._state == OrchestratorState.RUNNING

    @property
    def uptime_seconds(self) -> float | None:
        """Get system uptime in seconds."""
        if not self._started_at:
            return None
        end_time = self._stopped_at or datetime.now(UTC)
        return (end_time - self._started_at).total_seconds()

    def register_subsystem_factory(self, name: str, factory: SubsystemFactory) -> None:
        """Register a factory for creating a subsystem."""
        self._subsystem_factories[name] = factory

    def register_subsystem(self, subsystem: Subsystem) -> None:
        """Register an existing subsystem instance."""
        subsystem.set_message_bus(self._message_bus)
        self._subsystem_registry.register(subsystem)

    async def start(self) -> None:
        """Start the orchestrator and all registered subsystems."""
        if self._state != OrchestratorState.CREATED:
            raise RuntimeError(f"Cannot start orchestrator in state {self._state}")

        self._state = OrchestratorState.STARTING
        self._started_at = datetime.now(UTC)
        self._log.info("orchestrator_starting")

        try:
            # Create subsystems from factories
            for name, factory in self._subsystem_factories.items():
                self._log.debug("creating_subsystem", name=name)
                subsystem = factory()
                subsystem.set_message_bus(self._message_bus)
                self._subsystem_registry.register(subsystem)

            # Start all subsystems concurrently
            async with anyio.create_task_group() as tg:
                for subsystem in self._subsystem_registry:
                    tg.start_soon(self._start_subsystem, subsystem)

            # Publish system startup event
            await self._message_bus.publish(
                Message.event(
                    str(SystemTopics.STARTUP),
                    "orchestrator",
                    {"started_at": self._started_at.isoformat()},
                )
            )

            self._state = OrchestratorState.RUNNING
            self._log.info(
                "orchestrator_started",
                subsystem_count=len(self._subsystem_registry),
            )

        except Exception:
            self._state = OrchestratorState.FAILED
            self._log.exception("orchestrator_start_failed")
            raise

    async def _start_subsystem(self, subsystem: Subsystem) -> None:
        """Start a single subsystem with error handling."""
        try:
            await subsystem.start()
            self._log.debug("subsystem_started", name=subsystem.name)
        except Exception:
            self._log.exception("subsystem_start_failed", name=subsystem.name)
            raise

    async def stop(self) -> None:
        """Stop all subsystems and the orchestrator."""
        if self._state != OrchestratorState.RUNNING:
            return

        self._state = OrchestratorState.STOPPING
        self._log.info("orchestrator_stopping")

        try:
            # Publish shutdown event before stopping subsystems
            await self._message_bus.publish(
                Message.event(
                    str(SystemTopics.SHUTDOWN),
                    "orchestrator",
                    {"stopping_at": datetime.now(UTC).isoformat()},
                )
            )

            # Stop all subsystems concurrently
            async with anyio.create_task_group() as tg:
                for subsystem in self._subsystem_registry:
                    tg.start_soon(self._stop_subsystem, subsystem)

            # Clear message bus subscriptions
            self._message_bus.clear()

            self._stopped_at = datetime.now(UTC)
            self._state = OrchestratorState.STOPPED
            self._log.info(
                "orchestrator_stopped",
                uptime_seconds=self.uptime_seconds,
            )

        except Exception:
            self._state = OrchestratorState.FAILED
            self._log.exception("orchestrator_stop_failed")
            raise

    async def _stop_subsystem(self, subsystem: Subsystem) -> None:
        """Stop a single subsystem with error handling."""
        try:
            await subsystem.stop()
            self._log.debug("subsystem_stopped", name=subsystem.name)
        except Exception:
            self._log.exception("subsystem_stop_failed", name=subsystem.name)

    @asynccontextmanager
    async def run_context(self) -> AsyncIterator["Orchestrator"]:
        """Async context manager for running the orchestrator."""
        await self.start()
        try:
            yield self
        finally:
            await self.stop()

    def get_health(self) -> dict[str, Any]:
        """Get system health status."""
        running_subsystems = [
            s.name
            for s in self._subsystem_registry
            if s.subsystem_state == SubsystemState.RUNNING
        ]
        failed_subsystems = [
            s.name
            for s in self._subsystem_registry
            if s.subsystem_state == SubsystemState.FAILED
        ]

        return {
            "state": self._state.name,
            "uptime_seconds": self.uptime_seconds,
            "subsystems": {
                "total": len(self._subsystem_registry),
                "running": len(running_subsystems),
                "failed": len(failed_subsystems),
            },
            "message_bus": {
                "subscriptions": self._message_bus.stats.total_subscriptions,
                "messages_published": self._message_bus.stats.total_messages_published,
                "messages_delivered": self._message_bus.stats.total_messages_delivered,
            },
            "process_registry": {
                "total_entries": len(self._process_registry),
            },
        }


def create_default_orchestrator() -> Orchestrator:
    """Create an orchestrator with all default subsystems registered."""
    from autogenrec.subsystems.academic.academia_manager import AcademiaManager
    from autogenrec.subsystems.conflict.arbitration_engine import ArbitrationEngine
    from autogenrec.subsystems.conflict.conflict_resolver import ConflictResolver
    from autogenrec.subsystems.core_processing.code_generator import CodeGenerator
    from autogenrec.subsystems.core_processing.rule_compiler import RuleCompiler
    from autogenrec.subsystems.core_processing.symbolic_interpreter import SymbolicInterpreter
    from autogenrec.subsystems.data.archive_manager import ArchiveManager
    from autogenrec.subsystems.data.echo_handler import EchoHandler
    from autogenrec.subsystems.data.reference_manager import ReferenceManager
    from autogenrec.subsystems.identity.audience_classifier import AudienceClassifier
    from autogenrec.subsystems.identity.mask_generator import MaskGenerator
    from autogenrec.subsystems.meta.anthology_manager import AnthologyManager
    from autogenrec.subsystems.routing.node_router import NodeRouter
    from autogenrec.subsystems.routing.signal_threshold_guard import SignalThresholdGuard
    from autogenrec.subsystems.temporal.evolution_scheduler import EvolutionScheduler
    from autogenrec.subsystems.temporal.location_resolver import LocationResolver
    from autogenrec.subsystems.temporal.time_manager import TimeManager
    from autogenrec.subsystems.transformation.consumption_manager import ConsumptionManager
    from autogenrec.subsystems.transformation.process_converter import ProcessConverter
    from autogenrec.subsystems.value.blockchain_simulator import BlockchainSimulator
    from autogenrec.subsystems.value.process_monetizer import ProcessMonetizer
    from autogenrec.subsystems.value.value_exchange_manager import ValueExchangeManager

    orchestrator = Orchestrator()

    # Register all subsystem factories
    orchestrator.register_subsystem_factory(
        "anthology_manager",
        lambda: AnthologyManager(
            process_registry=orchestrator.process_registry,
            subsystem_registry=orchestrator.subsystem_registry,
        ),
    )
    orchestrator.register_subsystem_factory("symbolic_interpreter", SymbolicInterpreter)
    orchestrator.register_subsystem_factory("rule_compiler", RuleCompiler)
    orchestrator.register_subsystem_factory("code_generator", CodeGenerator)
    orchestrator.register_subsystem_factory("conflict_resolver", ConflictResolver)
    orchestrator.register_subsystem_factory("arbitration_engine", ArbitrationEngine)
    orchestrator.register_subsystem_factory("reference_manager", ReferenceManager)
    orchestrator.register_subsystem_factory("archive_manager", ArchiveManager)
    orchestrator.register_subsystem_factory("echo_handler", EchoHandler)
    orchestrator.register_subsystem_factory("node_router", NodeRouter)
    orchestrator.register_subsystem_factory("signal_threshold_guard", SignalThresholdGuard)
    orchestrator.register_subsystem_factory("value_exchange_manager", ValueExchangeManager)
    orchestrator.register_subsystem_factory("blockchain_simulator", BlockchainSimulator)
    orchestrator.register_subsystem_factory("process_monetizer", ProcessMonetizer)
    orchestrator.register_subsystem_factory("time_manager", TimeManager)
    orchestrator.register_subsystem_factory("location_resolver", LocationResolver)
    orchestrator.register_subsystem_factory("evolution_scheduler", EvolutionScheduler)
    orchestrator.register_subsystem_factory("mask_generator", MaskGenerator)
    orchestrator.register_subsystem_factory("audience_classifier", AudienceClassifier)
    orchestrator.register_subsystem_factory("process_converter", ProcessConverter)
    orchestrator.register_subsystem_factory("consumption_manager", ConsumptionManager)
    orchestrator.register_subsystem_factory("academia_manager", AcademiaManager)

    return orchestrator
