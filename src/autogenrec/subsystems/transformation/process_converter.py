"""
ProcessConverter: Transforms workflows into derivative outputs.

Converts processes between formats and representations, enabling the translation
of activity into new structures and forms.
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


class ConversionFormat(Enum):
    """Output formats for conversion."""

    JSON = auto()  # JSON structure
    YAML = auto()  # YAML structure
    XML = auto()  # XML structure
    SCHEMA = auto()  # Schema definition
    GRAPH = auto()  # Graph representation
    COMPRESSED = auto()  # Compressed binary
    EXECUTABLE = auto()  # Executable form
    TEMPLATE = auto()  # Template form
    SUMMARY = auto()  # Summarized form


class ConversionStrategy(Enum):
    """Strategies for conversion."""

    DIRECT = auto()  # Direct 1:1 mapping
    TRANSFORM = auto()  # Apply transformation rules
    COMPRESS = auto()  # Lossy compression
    EXPAND = auto()  # Expand with defaults
    MERGE = auto()  # Merge multiple inputs
    SPLIT = auto()  # Split into parts


class ConversionStatus(Enum):
    """Status of a conversion."""

    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    PARTIAL = auto()  # Partially converted


class SourceProcess(BaseModel):
    """A source process to convert."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    name: str
    description: str = ""

    # Process definition
    steps: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    inputs: tuple[str, ...] = Field(default_factory=tuple)
    outputs: tuple[str, ...] = Field(default_factory=tuple)

    # Metadata
    version: str = "1.0"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tags: frozenset[str] = Field(default_factory=frozenset)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversionRule(BaseModel):
    """A rule for converting processes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    name: str
    description: str = ""

    # Rule definition
    source_pattern: str  # Pattern to match in source
    target_template: str  # Template for output
    strategy: ConversionStrategy = ConversionStrategy.TRANSFORM

    # Conditions
    applies_to_formats: frozenset[ConversionFormat] = Field(default_factory=frozenset)
    priority: int = 0

    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ConvertedOutput(BaseModel):
    """A converted output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    source_id: str
    format: ConversionFormat
    strategy: ConversionStrategy

    # Output content
    content: Any  # The converted content
    content_type: str = "application/json"

    # Quality metrics
    fidelity: float = 1.0  # How faithful to original (0-1)
    compression_ratio: float = 1.0  # Size ratio vs original

    # Metadata
    converted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    rules_applied: tuple[str, ...] = Field(default_factory=tuple)
    warnings: tuple[str, ...] = Field(default_factory=tuple)


class ConversionJob(BaseModel):
    """A conversion job."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    source_id: str
    target_format: ConversionFormat
    strategy: ConversionStrategy = ConversionStrategy.TRANSFORM
    status: ConversionStatus = ConversionStatus.PENDING

    # Results
    output_id: str | None = None
    error: str | None = None

    # Timing
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass
class ConversionResult:
    """Result of a conversion operation."""

    job_id: str
    success: bool
    output: ConvertedOutput | None = None
    fidelity: float = 0.0
    error: str | None = None


@dataclass
class ConverterStats:
    """Statistics about conversions."""

    total_processes: int
    total_rules: int
    total_conversions: int
    successful_conversions: int
    average_fidelity: float
    conversions_by_format: dict[str, int]


class ProcessRegistry:
    """Registry of source processes."""

    def __init__(self) -> None:
        self._processes: dict[str, SourceProcess] = {}
        self._log = logger.bind(component="process_registry")

    @property
    def process_count(self) -> int:
        return len(self._processes)

    def register(self, process: SourceProcess) -> None:
        """Register a source process."""
        self._processes[process.id] = process
        self._log.debug("process_registered", process_id=process.id, name=process.name)

    def get(self, process_id: str) -> SourceProcess | None:
        """Get a process by ID."""
        return self._processes.get(process_id)

    def get_by_name(self, name: str) -> SourceProcess | None:
        """Get a process by name."""
        for p in self._processes.values():
            if p.name == name:
                return p
        return None

    def list_all(self) -> list[SourceProcess]:
        """List all processes."""
        return list(self._processes.values())


class RuleEngine:
    """Engine for applying conversion rules."""

    def __init__(self) -> None:
        self._rules: dict[str, ConversionRule] = {}
        self._log = logger.bind(component="rule_engine")

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    def add_rule(self, rule: ConversionRule) -> None:
        """Add a conversion rule."""
        self._rules[rule.id] = rule
        self._log.debug("rule_added", rule_id=rule.id, name=rule.name)

    def get_rule(self, rule_id: str) -> ConversionRule | None:
        """Get a rule by ID."""
        return self._rules.get(rule_id)

    def get_applicable_rules(
        self,
        target_format: ConversionFormat,
    ) -> list[ConversionRule]:
        """Get rules applicable to a format."""
        applicable = []
        for rule in self._rules.values():
            if not rule.is_active:
                continue
            if rule.applies_to_formats and target_format not in rule.applies_to_formats:
                continue
            applicable.append(rule)
        return sorted(applicable, key=lambda r: -r.priority)


class ConversionEngine:
    """Engine for performing conversions."""

    def __init__(
        self,
        process_registry: ProcessRegistry,
        rule_engine: RuleEngine,
    ) -> None:
        self._processes = process_registry
        self._rules = rule_engine
        self._outputs: dict[str, ConvertedOutput] = {}
        self._jobs: dict[str, ConversionJob] = {}
        self._log = logger.bind(component="conversion_engine")

    @property
    def output_count(self) -> int:
        return len(self._outputs)

    @property
    def job_count(self) -> int:
        return len(self._jobs)

    def convert(
        self,
        source_id: str,
        target_format: ConversionFormat,
        strategy: ConversionStrategy = ConversionStrategy.TRANSFORM,
        **options: Any,
    ) -> ConversionResult:
        """Convert a process to target format."""
        # Create job
        job = ConversionJob(
            source_id=source_id,
            target_format=target_format,
            strategy=strategy,
        )
        self._jobs[job.id] = job

        # Get source
        source = self._processes.get(source_id)
        if not source:
            return self._fail_job(job, "Source process not found")

        # Update job status
        job = ConversionJob(
            id=job.id,
            source_id=job.source_id,
            target_format=job.target_format,
            strategy=job.strategy,
            status=ConversionStatus.IN_PROGRESS,
            created_at=job.created_at,
            started_at=datetime.now(UTC),
        )
        self._jobs[job.id] = job

        # Perform conversion
        try:
            output = self._perform_conversion(source, target_format, strategy, options)
            return self._complete_job(job, output)
        except Exception as e:
            return self._fail_job(job, str(e))

    def _perform_conversion(
        self,
        source: SourceProcess,
        target_format: ConversionFormat,
        strategy: ConversionStrategy,
        options: dict[str, Any],
    ) -> ConvertedOutput:
        """Perform the actual conversion."""
        rules = self._rules.get_applicable_rules(target_format)
        rules_applied = [r.id for r in rules]
        warnings: list[str] = []

        # Convert based on format
        if target_format == ConversionFormat.JSON:
            content = self._to_json(source)
        elif target_format == ConversionFormat.YAML:
            content = self._to_yaml(source)
        elif target_format == ConversionFormat.SCHEMA:
            content = self._to_schema(source)
        elif target_format == ConversionFormat.GRAPH:
            content = self._to_graph(source)
        elif target_format == ConversionFormat.SUMMARY:
            content = self._to_summary(source)
            warnings.append("Summary is lossy - some details omitted")
        elif target_format == ConversionFormat.COMPRESSED:
            content = self._to_compressed(source)
        elif target_format == ConversionFormat.TEMPLATE:
            content = self._to_template(source)
        else:
            content = self._to_json(source)

        # Calculate fidelity
        fidelity = self._calculate_fidelity(source, content, strategy)

        # Calculate compression ratio
        import json
        original_size = len(json.dumps(self._to_json(source)))
        output_size = len(json.dumps(content)) if isinstance(content, (dict, list)) else len(str(content))
        compression_ratio = output_size / original_size if original_size > 0 else 1.0

        return ConvertedOutput(
            source_id=source.id,
            format=target_format,
            strategy=strategy,
            content=content,
            fidelity=fidelity,
            compression_ratio=compression_ratio,
            rules_applied=tuple(rules_applied),
            warnings=tuple(warnings),
        )

    def _to_json(self, source: SourceProcess) -> dict[str, Any]:
        """Convert to JSON structure."""
        return {
            "id": source.id,
            "name": source.name,
            "description": source.description,
            "version": source.version,
            "steps": list(source.steps),
            "inputs": list(source.inputs),
            "outputs": list(source.outputs),
            "metadata": source.metadata,
            "tags": list(source.tags),
        }

    def _to_yaml(self, source: SourceProcess) -> dict[str, Any]:
        """Convert to YAML-friendly structure."""
        # Same structure as JSON but with comments as metadata
        result = self._to_json(source)
        result["_comments"] = {
            "description": f"Process: {source.name}",
            "version": source.version,
        }
        return result

    def _to_schema(self, source: SourceProcess) -> dict[str, Any]:
        """Convert to schema definition."""
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": source.name,
            "description": source.description,
            "type": "object",
            "properties": {
                "inputs": {
                    "type": "object",
                    "properties": {inp: {"type": "string"} for inp in source.inputs},
                },
                "outputs": {
                    "type": "object",
                    "properties": {out: {"type": "string"} for out in source.outputs},
                },
                "steps": {
                    "type": "array",
                    "items": {"type": "object"},
                    "minItems": len(source.steps),
                },
            },
        }

    def _to_graph(self, source: SourceProcess) -> dict[str, Any]:
        """Convert to graph representation."""
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, str]] = []

        # Input nodes
        for i, inp in enumerate(source.inputs):
            nodes.append({"id": f"input_{i}", "type": "input", "label": inp})

        # Step nodes
        for i, step in enumerate(source.steps):
            node_id = f"step_{i}"
            nodes.append({
                "id": node_id,
                "type": "step",
                "label": step.get("name", f"Step {i+1}"),
                "data": step,
            })
            # Connect to previous
            if i == 0:
                for j in range(len(source.inputs)):
                    edges.append({"from": f"input_{j}", "to": node_id})
            else:
                edges.append({"from": f"step_{i-1}", "to": node_id})

        # Output nodes
        for i, out in enumerate(source.outputs):
            out_id = f"output_{i}"
            nodes.append({"id": out_id, "type": "output", "label": out})
            if source.steps:
                edges.append({"from": f"step_{len(source.steps)-1}", "to": out_id})

        return {"nodes": nodes, "edges": edges}

    def _to_summary(self, source: SourceProcess) -> dict[str, Any]:
        """Convert to summarized form."""
        return {
            "name": source.name,
            "step_count": len(source.steps),
            "input_count": len(source.inputs),
            "output_count": len(source.outputs),
            "tags": list(source.tags)[:5],  # Limit tags
        }

    def _to_compressed(self, source: SourceProcess) -> dict[str, Any]:
        """Convert to compressed form."""
        import hashlib
        import json

        full_data = self._to_json(source)
        data_str = json.dumps(full_data, sort_keys=True)
        checksum = hashlib.sha256(data_str.encode()).hexdigest()[:16]

        return {
            "id": source.id,
            "name": source.name,
            "checksum": checksum,
            "size": len(data_str),
            "step_count": len(source.steps),
        }

    def _to_template(self, source: SourceProcess) -> dict[str, Any]:
        """Convert to template form."""
        return {
            "template_name": f"{source.name}_template",
            "placeholders": {
                "inputs": {inp: f"{{{{ {inp} }}}}" for inp in source.inputs},
                "outputs": {out: f"{{{{ {out} }}}}" for out in source.outputs},
            },
            "step_templates": [
                {"name": step.get("name", f"step_{i}"), "template": "{{ step_logic }}"}
                for i, step in enumerate(source.steps)
            ],
        }

    def _calculate_fidelity(
        self,
        source: SourceProcess,
        output: Any,
        strategy: ConversionStrategy,
    ) -> float:
        """Calculate conversion fidelity."""
        if strategy == ConversionStrategy.DIRECT:
            return 1.0
        elif strategy == ConversionStrategy.COMPRESS:
            return 0.7  # Lossy
        elif strategy == ConversionStrategy.TRANSFORM:
            return 0.95
        elif strategy == ConversionStrategy.EXPAND:
            return 0.9
        else:
            return 0.85

    def _complete_job(
        self,
        job: ConversionJob,
        output: ConvertedOutput,
    ) -> ConversionResult:
        """Complete a conversion job."""
        self._outputs[output.id] = output

        updated = ConversionJob(
            id=job.id,
            source_id=job.source_id,
            target_format=job.target_format,
            strategy=job.strategy,
            status=ConversionStatus.COMPLETED,
            output_id=output.id,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=datetime.now(UTC),
        )
        self._jobs[job.id] = updated

        return ConversionResult(
            job_id=job.id,
            success=True,
            output=output,
            fidelity=output.fidelity,
        )

    def _fail_job(self, job: ConversionJob, error: str) -> ConversionResult:
        """Fail a conversion job."""
        updated = ConversionJob(
            id=job.id,
            source_id=job.source_id,
            target_format=job.target_format,
            strategy=job.strategy,
            status=ConversionStatus.FAILED,
            error=error,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=datetime.now(UTC),
        )
        self._jobs[job.id] = updated

        return ConversionResult(
            job_id=job.id,
            success=False,
            error=error,
        )

    def get_output(self, output_id: str) -> ConvertedOutput | None:
        """Get a converted output."""
        return self._outputs.get(output_id)

    def get_job(self, job_id: str) -> ConversionJob | None:
        """Get a conversion job."""
        return self._jobs.get(job_id)

    def get_successful_conversions(self) -> list[ConvertedOutput]:
        """Get all successful conversions."""
        return list(self._outputs.values())


class ProcessConverter(Subsystem):
    """
    Transforms processes into derivative outputs.

    Process Loop:
    1. Intake: Receive defined processes and parameters
    2. Transform: Apply conversion rules to generate outputs
    3. Evaluate: Assess the efficiency and fidelity of converted outputs
    4. Integrate: Integrate results back into the system
    """

    def __init__(self) -> None:
        metadata = SubsystemMetadata(
            name="process_converter",
            display_name="Process Converter",
            description="Transforms workflows into derivative outputs",
            type=SubsystemType.TRANSFORMATION,
            tags=frozenset(["conversion", "transformation", "process", "workflow"]),
            input_types=frozenset(["REFERENCE", "PATTERN", "CODE", "PROCESS"]),
            output_types=frozenset(["CODE", "SCHEMA", "PATTERN", "JSON"]),
            subscribed_topics=frozenset([
                str(SubsystemTopics.ALL),
                "conversion.#",
                "process.#",
            ]),
            published_topics=frozenset([
                "conversion.completed",
                "conversion.failed",
                "process.registered",
            ]),
        )
        super().__init__(metadata)

        self._processes = ProcessRegistry()
        self._rules = RuleEngine()
        self._engine = ConversionEngine(self._processes, self._rules)

    @property
    def process_count(self) -> int:
        return self._processes.process_count

    @property
    def rule_count(self) -> int:
        return self._rules.rule_count

    @property
    def conversion_count(self) -> int:
        return self._engine.output_count

    async def intake(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> SymbolicInput:
        """Phase 1: Receive processes and parameters."""
        if not input_data.values:
            self._log.debug("empty_input")
            return input_data

        self._log.debug("intake_complete", value_count=len(input_data.values))
        return input_data

    async def process(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> list[ConversionResult]:
        """Phase 2: Transform processes."""
        results: list[ConversionResult] = []

        for value in input_data.values:
            content = value.content
            if not isinstance(content, dict):
                continue

            action = content.get("action", "convert")

            if action == "register":
                result = self._register_from_value(value)
                results.append(result)
            elif action == "convert":
                result = self._convert_from_value(value)
                results.append(result)

        return results

    async def evaluate(
        self,
        intermediate: list[ConversionResult],
        ctx: ProcessContext[dict[str, Any]],
    ) -> tuple[SymbolicOutput, bool]:
        """Phase 3: Prepare converted outputs."""
        values: list[SymbolicValue] = []

        for result in intermediate:
            value = SymbolicValue(
                type=SymbolicValueType.PATTERN,
                content={
                    "job_id": result.job_id,
                    "success": result.success,
                    "fidelity": result.fidelity,
                    "output_id": result.output.id if result.output else None,
                    "format": result.output.format.name if result.output else None,
                    "error": result.error,
                },
                source_subsystem=self.name,
                tags=frozenset(["conversion", "result"]),
                meaning="Conversion result",
                confidence=result.fidelity if result.success else 0.0,
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
        """Phase 4: Emit events for conversions."""
        if self._message_bus and output.values:
            for value in output.values:
                content = value.content
                if not isinstance(content, dict):
                    continue

                if content.get("success"):
                    await self.emit_event(
                        "conversion.completed",
                        {
                            "job_id": content.get("job_id"),
                            "output_id": content.get("output_id"),
                            "fidelity": content.get("fidelity"),
                        },
                    )
                else:
                    await self.emit_event(
                        "conversion.failed",
                        {
                            "job_id": content.get("job_id"),
                            "error": content.get("error"),
                        },
                    )

        return None

    def _register_from_value(self, value: SymbolicValue) -> ConversionResult:
        """Register a process from a SymbolicValue."""
        content = value.content
        if not isinstance(content, dict):
            return ConversionResult(job_id="", success=False, error="Invalid content")

        try:
            process = SourceProcess(
                name=content.get("name", f"process_{value.id[:8]}"),
                description=content.get("description", ""),
                steps=tuple(content.get("steps", [])),
                inputs=tuple(content.get("inputs", [])),
                outputs=tuple(content.get("outputs", [])),
                version=content.get("version", "1.0"),
                tags=frozenset(content.get("tags", [])) | value.tags,
                metadata=content.get("metadata", {}),
            )
            self._processes.register(process)
            return ConversionResult(job_id=process.id, success=True)

        except Exception as e:
            return ConversionResult(job_id="", success=False, error=str(e))

    def _convert_from_value(self, value: SymbolicValue) -> ConversionResult:
        """Convert a process from a SymbolicValue request."""
        content = value.content
        if not isinstance(content, dict):
            return ConversionResult(job_id="", success=False, error="Invalid content")

        source_id = content.get("source_id")
        if not source_id:
            return ConversionResult(job_id="", success=False, error="source_id required")

        format_str = content.get("format", "JSON")
        try:
            target_format = ConversionFormat[format_str.upper()]
        except KeyError:
            target_format = ConversionFormat.JSON

        strategy_str = content.get("strategy", "TRANSFORM")
        try:
            strategy = ConversionStrategy[strategy_str.upper()]
        except KeyError:
            strategy = ConversionStrategy.TRANSFORM

        return self._engine.convert(source_id, target_format, strategy)

    # --- Message handlers ---

    async def handle_event(self, message: Message) -> None:
        """Handle incoming events."""
        if message.topic.startswith("conversion.") or message.topic.startswith("process."):
            self._log.debug("event_received", topic=message.topic)

    async def handle_signal(self, signal: Any) -> None:
        """Handle incoming signals."""
        self._log.debug("signal_received", signal_id=getattr(signal, "id", "unknown"))

    # --- Public API ---

    def register_process(
        self,
        name: str,
        steps: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> SourceProcess:
        """Register a source process."""
        process = SourceProcess(
            name=name,
            description=kwargs.get("description", ""),
            steps=tuple(steps or []),
            inputs=tuple(kwargs.get("inputs", [])),
            outputs=tuple(kwargs.get("outputs", [])),
            version=kwargs.get("version", "1.0"),
            tags=frozenset(kwargs.get("tags", [])),
            metadata=kwargs.get("metadata", {}),
        )
        self._processes.register(process)
        return process

    def get_process(self, process_id: str) -> SourceProcess | None:
        """Get a process by ID."""
        return self._processes.get(process_id)

    def add_rule(
        self,
        name: str,
        source_pattern: str,
        target_template: str,
        **kwargs: Any,
    ) -> ConversionRule:
        """Add a conversion rule."""
        formats = kwargs.get("applies_to_formats", [])
        if formats:
            format_set = frozenset(
                ConversionFormat[f.upper()] if isinstance(f, str) else f
                for f in formats
            )
        else:
            format_set = frozenset()

        strategy_str = kwargs.get("strategy", "TRANSFORM")
        if isinstance(strategy_str, str):
            try:
                strategy = ConversionStrategy[strategy_str.upper()]
            except KeyError:
                strategy = ConversionStrategy.TRANSFORM
        else:
            strategy = strategy_str

        rule = ConversionRule(
            name=name,
            source_pattern=source_pattern,
            target_template=target_template,
            strategy=strategy,
            applies_to_formats=format_set,
            priority=kwargs.get("priority", 0),
            is_active=kwargs.get("is_active", True),
        )
        self._rules.add_rule(rule)
        return rule

    def convert(
        self,
        source_id: str,
        target_format: ConversionFormat,
        strategy: ConversionStrategy = ConversionStrategy.TRANSFORM,
    ) -> ConversionResult:
        """Convert a process to target format."""
        return self._engine.convert(source_id, target_format, strategy)

    def convert_by_name(
        self,
        name: str,
        target_format: ConversionFormat,
        strategy: ConversionStrategy = ConversionStrategy.TRANSFORM,
    ) -> ConversionResult:
        """Convert a process by name."""
        process = self._processes.get_by_name(name)
        if not process:
            return ConversionResult(job_id="", success=False, error=f"Process '{name}' not found")
        return self._engine.convert(process.id, target_format, strategy)

    def get_output(self, output_id: str) -> ConvertedOutput | None:
        """Get a converted output."""
        return self._engine.get_output(output_id)

    def get_job(self, job_id: str) -> ConversionJob | None:
        """Get a conversion job."""
        return self._engine.get_job(job_id)

    def get_stats(self) -> ConverterStats:
        """Get conversion statistics."""
        outputs = self._engine.get_successful_conversions()

        by_format: dict[str, int] = {}
        total_fidelity = 0.0
        for output in outputs:
            by_format[output.format.name] = by_format.get(output.format.name, 0) + 1
            total_fidelity += output.fidelity

        avg_fidelity = total_fidelity / len(outputs) if outputs else 0.0

        return ConverterStats(
            total_processes=self._processes.process_count,
            total_rules=self._rules.rule_count,
            total_conversions=self._engine.job_count,
            successful_conversions=len(outputs),
            average_fidelity=avg_fidelity,
            conversions_by_format=by_format,
        )

    def clear(self) -> tuple[int, int, int]:
        """Clear all data. Returns (processes, rules, outputs) cleared."""
        processes = self._processes.process_count
        rules = self._rules.rule_count
        outputs = self._engine.output_count
        self._processes = ProcessRegistry()
        self._rules = RuleEngine()
        self._engine = ConversionEngine(self._processes, self._rules)
        return processes, rules, outputs
