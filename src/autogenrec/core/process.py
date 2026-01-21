"""
ProcessLoop base class implementing the 4-phase recursive pattern.

All subsystems follow the pattern: Intake → Process → Evaluate → Integrate
with outputs potentially feeding back as inputs for recursive execution.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, auto
from typing import Any, Generic, TypeVar

import structlog
from ulid import ULID

logger = structlog.get_logger()


class ProcessPhase(Enum):
    """Phases of the process loop."""

    IDLE = auto()
    INTAKE = auto()
    PROCESS = auto()
    EVALUATE = auto()
    INTEGRATE = auto()


class ProcessState(Enum):
    """Execution state of a process."""

    PENDING = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
ContextT = TypeVar("ContextT")


@dataclass
class ProcessContext(Generic[ContextT]):
    """Shared context passed through all phases of a process loop iteration."""

    iteration: int
    started_at: datetime
    data: ContextT | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessResult(Generic[OutputT]):
    """Result of a complete process loop execution."""

    id: str
    iterations: int
    outputs: list[OutputT]
    state: ProcessState
    started_at: datetime
    completed_at: datetime
    error: Exception | None = None

    @property
    def duration_ms(self) -> float:
        """Total execution duration in milliseconds."""
        delta = self.completed_at - self.started_at
        return delta.total_seconds() * 1000


class ProcessLoop(ABC, Generic[InputT, OutputT, ContextT]):
    """
    Abstract base class for the 4-phase recursive process pattern.

    Subclasses implement the four phase methods:
    - intake(): Gather and validate inputs
    - process(): Apply core transformation logic
    - evaluate(): Assess results and determine next steps
    - integrate(): Finalize outputs and prepare for feedback

    The loop supports both single execution and recursive execution
    where outputs can feed back as inputs.
    """

    def __init__(self, name: str, max_iterations: int = 100) -> None:
        self.name = name
        self.max_iterations = max_iterations
        self._current_phase = ProcessPhase.IDLE
        self._state = ProcessState.PENDING
        self._log = logger.bind(process=name)

    @property
    def current_phase(self) -> ProcessPhase:
        return self._current_phase

    @property
    def state(self) -> ProcessState:
        return self._state

    @asynccontextmanager
    async def _phase_context(self, phase: ProcessPhase) -> AsyncIterator[None]:
        """Context manager for tracking phase transitions."""
        previous_phase = self._current_phase
        self._current_phase = phase
        self._log.debug("phase_entered", phase=phase.name)
        try:
            await self.on_phase_enter(phase)
            yield
            await self.on_phase_exit(phase)
        except Exception:
            self._log.exception("phase_error", phase=phase.name)
            raise
        finally:
            self._current_phase = previous_phase

    # --- Lifecycle hooks (override in subclasses) ---

    async def on_phase_enter(self, phase: ProcessPhase) -> None:
        """Called when entering a phase. Override for custom behavior."""

    async def on_phase_exit(self, phase: ProcessPhase) -> None:
        """Called when exiting a phase. Override for custom behavior."""

    async def on_iteration_start(self, ctx: ProcessContext[ContextT]) -> None:
        """Called at the start of each iteration. Override for custom behavior."""

    async def on_iteration_end(self, ctx: ProcessContext[ContextT], output: OutputT) -> None:
        """Called at the end of each iteration. Override for custom behavior."""

    # --- Abstract phase methods (must be implemented by subclasses) ---

    @abstractmethod
    async def intake(
        self, input_data: InputT, ctx: ProcessContext[ContextT]
    ) -> InputT:
        """
        Phase 1: Intake - Gather and validate inputs.

        Collect symbolic inputs, validate format, and prepare for processing.
        May transform or normalize the input.

        Args:
            input_data: The raw input to process
            ctx: Shared context for this iteration

        Returns:
            Validated/normalized input ready for processing
        """
        ...

    @abstractmethod
    async def process(
        self, input_data: InputT, ctx: ProcessContext[ContextT]
    ) -> Any:
        """
        Phase 2: Process - Apply core transformation logic.

        Execute the main processing logic, applying interpretive frameworks,
        transformations, or computations.

        Args:
            input_data: Validated input from intake phase
            ctx: Shared context for this iteration

        Returns:
            Intermediate result to be evaluated
        """
        ...

    @abstractmethod
    async def evaluate(
        self, intermediate: Any, ctx: ProcessContext[ContextT]
    ) -> tuple[OutputT, bool]:
        """
        Phase 3: Evaluate - Assess results and determine next steps.

        Synthesize outputs, assess quality, and decide whether to continue
        iteration or terminate.

        Args:
            intermediate: Result from process phase
            ctx: Shared context for this iteration

        Returns:
            Tuple of (output, should_continue) where should_continue
            indicates if another iteration is needed
        """
        ...

    @abstractmethod
    async def integrate(
        self, output: OutputT, ctx: ProcessContext[ContextT]
    ) -> InputT | None:
        """
        Phase 4: Integrate - Finalize outputs and prepare feedback.

        Feed outputs back into the system for further iteration if needed.
        Returns None to signal termination, or a new input for recursion.

        Args:
            output: Evaluated output from this iteration
            ctx: Shared context for this iteration

        Returns:
            New input for next iteration, or None to terminate
        """
        ...

    # --- Execution methods ---

    async def run_once(
        self,
        input_data: InputT,
        context_data: ContextT | None = None,
    ) -> OutputT:
        """
        Execute a single iteration of the process loop.

        Args:
            input_data: Input to process
            context_data: Optional context data to pass through phases

        Returns:
            Output from this single iteration
        """
        ctx = ProcessContext(
            iteration=1,
            started_at=datetime.now(UTC),
            data=context_data,
        )

        await self.on_iteration_start(ctx)

        async with self._phase_context(ProcessPhase.INTAKE):
            validated = await self.intake(input_data, ctx)

        async with self._phase_context(ProcessPhase.PROCESS):
            intermediate = await self.process(validated, ctx)

        async with self._phase_context(ProcessPhase.EVALUATE):
            output, _ = await self.evaluate(intermediate, ctx)

        async with self._phase_context(ProcessPhase.INTEGRATE):
            await self.integrate(output, ctx)

        await self.on_iteration_end(ctx, output)

        return output

    async def run(
        self,
        input_data: InputT,
        context_data: ContextT | None = None,
    ) -> ProcessResult[OutputT]:
        """
        Execute the full recursive process loop.

        Runs iterations until:
        - evaluate() returns should_continue=False
        - integrate() returns None
        - max_iterations is reached

        Args:
            input_data: Initial input to process
            context_data: Optional context data to pass through phases

        Returns:
            ProcessResult containing all outputs and execution metadata
        """
        run_id = str(ULID())
        started_at = datetime.now(UTC)
        outputs: list[OutputT] = []
        current_input: InputT | None = input_data
        iteration = 0

        self._state = ProcessState.RUNNING
        self._log.info("process_started", run_id=run_id)

        try:
            while current_input is not None and iteration < self.max_iterations:
                iteration += 1
                ctx = ProcessContext(
                    iteration=iteration,
                    started_at=datetime.now(UTC),
                    data=context_data,
                )

                await self.on_iteration_start(ctx)

                async with self._phase_context(ProcessPhase.INTAKE):
                    validated = await self.intake(current_input, ctx)

                async with self._phase_context(ProcessPhase.PROCESS):
                    intermediate = await self.process(validated, ctx)

                async with self._phase_context(ProcessPhase.EVALUATE):
                    output, should_continue = await self.evaluate(intermediate, ctx)
                    outputs.append(output)

                async with self._phase_context(ProcessPhase.INTEGRATE):
                    if should_continue:
                        current_input = await self.integrate(output, ctx)
                    else:
                        await self.integrate(output, ctx)
                        current_input = None

                await self.on_iteration_end(ctx, output)

            self._state = ProcessState.COMPLETED
            self._log.info(
                "process_completed",
                run_id=run_id,
                iterations=iteration,
            )

            return ProcessResult(
                id=run_id,
                iterations=iteration,
                outputs=outputs,
                state=ProcessState.COMPLETED,
                started_at=started_at,
                completed_at=datetime.now(UTC),
            )

        except Exception as e:
            self._state = ProcessState.FAILED
            self._log.exception("process_failed", run_id=run_id, iteration=iteration)
            return ProcessResult(
                id=run_id,
                iterations=iteration,
                outputs=outputs,
                state=ProcessState.FAILED,
                started_at=started_at,
                completed_at=datetime.now(UTC),
                error=e,
            )
