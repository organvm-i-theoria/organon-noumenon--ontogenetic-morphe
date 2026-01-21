"""
RuleCompiler: Compiles and validates symbolic rules.

Transforms rule definitions into executable constraints with full
validation, dependency resolution, and compilation to executable form.
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


class RuleType(Enum):
    """Types of symbolic rules."""

    CONSTRAINT = auto()  # Limits or boundaries
    TRANSFORMATION = auto()  # Change operations
    TRIGGER = auto()  # Event-based activation
    VALIDATION = auto()  # Input/output validation
    POLICY = auto()  # Behavioral policies
    MAPPING = auto()  # Value mappings
    COMPOSITE = auto()  # Combination of rules


class RuleStatus(Enum):
    """Status of a rule in the compilation pipeline."""

    DRAFT = auto()  # Initial state
    PARSED = auto()  # Structure analyzed
    VALIDATED = auto()  # Passed validation
    COMPILED = auto()  # Executable form generated
    ACTIVE = auto()  # In use
    DEPRECATED = auto()  # Marked for removal
    INVALID = auto()  # Failed validation


class RulePriority(Enum):
    """Priority levels for rule execution."""

    LOWEST = 0
    LOW = 25
    NORMAL = 50
    HIGH = 75
    HIGHEST = 100


class ValidationError(BaseModel):
    """A validation error for a rule."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str
    message: str
    location: str | None = None
    severity: str = "error"  # "error", "warning", "info"


class RuleCondition(BaseModel):
    """A condition within a rule."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    field: str
    operator: str  # "eq", "ne", "gt", "lt", "gte", "lte", "in", "contains", "matches"
    value: Any
    negate: bool = False


class RuleAction(BaseModel):
    """An action to execute when a rule matches."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action_type: str  # "set", "transform", "emit", "block", "allow", "log"
    target: str
    value: Any | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class RuleDefinition(BaseModel):
    """Definition of a symbolic rule."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    name: str
    description: str = ""
    rule_type: RuleType = RuleType.CONSTRAINT
    priority: RulePriority = RulePriority.NORMAL
    enabled: bool = True

    # Rule structure
    conditions: tuple[RuleCondition, ...] = Field(default_factory=tuple)
    actions: tuple[RuleAction, ...] = Field(default_factory=tuple)
    match_all: bool = True  # True = AND, False = OR for conditions

    # Dependencies
    depends_on: frozenset[str] = Field(default_factory=frozenset)  # Rule IDs
    conflicts_with: frozenset[str] = Field(default_factory=frozenset)  # Rule IDs

    # Metadata
    tags: frozenset[str] = Field(default_factory=frozenset)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    version: int = 1


class CompiledRule(BaseModel):
    """A compiled rule ready for execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    definition_id: str  # Reference to original definition
    name: str
    rule_type: RuleType
    priority: int  # Numeric priority for ordering
    status: RuleStatus = RuleStatus.COMPILED

    # Compiled form
    condition_bytecode: str  # Serialized condition logic
    action_bytecode: str  # Serialized action logic

    # Metadata
    compiled_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    validation_errors: tuple[ValidationError, ...] = Field(default_factory=tuple)
    execution_count: int = 0

    def can_execute(self) -> bool:
        """Check if rule is in executable state."""
        return self.status in (RuleStatus.COMPILED, RuleStatus.ACTIVE) and not any(
            e.severity == "error" for e in self.validation_errors
        )


@dataclass
class CompilationResult:
    """Result of compiling a rule definition."""

    definition_id: str
    success: bool
    compiled_rule: CompiledRule | None = None
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)


class ValidationEngine:
    """Validates rule definitions for correctness and consistency."""

    VALID_OPERATORS = {
        "eq", "ne", "gt", "lt", "gte", "lte",
        "in", "not_in", "contains", "not_contains",
        "matches", "exists", "not_exists",
    }

    VALID_ACTION_TYPES = {
        "set", "transform", "emit", "block", "allow",
        "log", "notify", "escalate", "delegate",
    }

    def __init__(self) -> None:
        self._log = logger.bind(component="validation_engine")

    def validate(
        self,
        definition: RuleDefinition,
        existing_rules: dict[str, RuleDefinition] | None = None,
    ) -> tuple[list[ValidationError], list[ValidationError]]:
        """
        Validate a rule definition.

        Returns: (errors, warnings)
        """
        errors: list[ValidationError] = []
        warnings: list[ValidationError] = []
        existing_rules = existing_rules or {}

        # Basic validation
        if not definition.name:
            errors.append(ValidationError(
                code="EMPTY_NAME",
                message="Rule name cannot be empty",
                location="name",
            ))

        if not definition.conditions and not definition.actions:
            errors.append(ValidationError(
                code="EMPTY_RULE",
                message="Rule must have at least one condition or action",
                location="conditions/actions",
            ))

        # Validate conditions
        for i, condition in enumerate(definition.conditions):
            cond_errors, cond_warnings = self._validate_condition(condition, i)
            errors.extend(cond_errors)
            warnings.extend(cond_warnings)

        # Validate actions
        for i, action in enumerate(definition.actions):
            action_errors, action_warnings = self._validate_action(action, i)
            errors.extend(action_errors)
            warnings.extend(action_warnings)

        # Validate dependencies
        dep_errors = self._validate_dependencies(definition, existing_rules)
        errors.extend(dep_errors)

        # Check for conflicts
        conflict_warnings = self._check_conflicts(definition, existing_rules)
        warnings.extend(conflict_warnings)

        self._log.debug(
            "validation_complete",
            rule_id=definition.id,
            errors=len(errors),
            warnings=len(warnings),
        )

        return errors, warnings

    def _validate_condition(
        self, condition: RuleCondition, index: int
    ) -> tuple[list[ValidationError], list[ValidationError]]:
        """Validate a single condition."""
        errors: list[ValidationError] = []
        warnings: list[ValidationError] = []
        loc = f"conditions[{index}]"

        if not condition.field:
            errors.append(ValidationError(
                code="EMPTY_FIELD",
                message="Condition field cannot be empty",
                location=f"{loc}.field",
            ))

        if condition.operator not in self.VALID_OPERATORS:
            errors.append(ValidationError(
                code="INVALID_OPERATOR",
                message=f"Unknown operator: {condition.operator}",
                location=f"{loc}.operator",
            ))

        # Type-specific validation
        if condition.operator in ("gt", "lt", "gte", "lte"):
            if not isinstance(condition.value, (int, float)):
                warnings.append(ValidationError(
                    code="TYPE_MISMATCH",
                    message=f"Comparison operator '{condition.operator}' typically used with numbers",
                    location=f"{loc}.value",
                    severity="warning",
                ))

        if condition.operator in ("in", "not_in"):
            if not isinstance(condition.value, (list, tuple, set, frozenset)):
                errors.append(ValidationError(
                    code="TYPE_MISMATCH",
                    message=f"Operator '{condition.operator}' requires a collection value",
                    location=f"{loc}.value",
                ))

        return errors, warnings

    def _validate_action(
        self, action: RuleAction, index: int
    ) -> tuple[list[ValidationError], list[ValidationError]]:
        """Validate a single action."""
        errors: list[ValidationError] = []
        warnings: list[ValidationError] = []
        loc = f"actions[{index}]"

        if action.action_type not in self.VALID_ACTION_TYPES:
            errors.append(ValidationError(
                code="INVALID_ACTION_TYPE",
                message=f"Unknown action type: {action.action_type}",
                location=f"{loc}.action_type",
            ))

        if not action.target:
            errors.append(ValidationError(
                code="EMPTY_TARGET",
                message="Action target cannot be empty",
                location=f"{loc}.target",
            ))

        # Action-specific validation
        if action.action_type == "set" and action.value is None:
            warnings.append(ValidationError(
                code="MISSING_VALUE",
                message="'set' action without value will set target to None",
                location=f"{loc}.value",
                severity="warning",
            ))

        return errors, warnings

    def _validate_dependencies(
        self,
        definition: RuleDefinition,
        existing_rules: dict[str, RuleDefinition],
    ) -> list[ValidationError]:
        """Validate rule dependencies."""
        errors: list[ValidationError] = []

        for dep_id in definition.depends_on:
            if dep_id not in existing_rules:
                errors.append(ValidationError(
                    code="MISSING_DEPENDENCY",
                    message=f"Dependency not found: {dep_id}",
                    location="depends_on",
                ))

        # Check for circular dependencies
        if self._has_circular_dependency(definition.id, definition.depends_on, existing_rules):
            errors.append(ValidationError(
                code="CIRCULAR_DEPENDENCY",
                message="Circular dependency detected",
                location="depends_on",
            ))

        return errors

    def _has_circular_dependency(
        self,
        rule_id: str,
        depends_on: frozenset[str],
        existing_rules: dict[str, RuleDefinition],
        visited: set[str] | None = None,
    ) -> bool:
        """Check for circular dependencies."""
        if visited is None:
            visited = set()

        if rule_id in visited:
            return True
        visited.add(rule_id)

        for dep_id in depends_on:
            if dep_id in existing_rules:
                dep_rule = existing_rules[dep_id]
                if self._has_circular_dependency(
                    dep_id, dep_rule.depends_on, existing_rules, visited
                ):
                    return True

        return False

    def _check_conflicts(
        self,
        definition: RuleDefinition,
        existing_rules: dict[str, RuleDefinition],
    ) -> list[ValidationError]:
        """Check for potential conflicts with existing rules."""
        warnings: list[ValidationError] = []

        for conflict_id in definition.conflicts_with:
            if conflict_id in existing_rules:
                existing = existing_rules[conflict_id]
                if existing.enabled:
                    warnings.append(ValidationError(
                        code="ACTIVE_CONFLICT",
                        message=f"Conflicts with active rule: {existing.name}",
                        location="conflicts_with",
                        severity="warning",
                    ))

        return warnings


class CompilationEngine:
    """Compiles rule definitions into executable form."""

    def __init__(self) -> None:
        self._validation_engine = ValidationEngine()
        self._log = logger.bind(component="compilation_engine")

    def compile(
        self,
        definition: RuleDefinition,
        existing_rules: dict[str, RuleDefinition] | None = None,
    ) -> CompilationResult:
        """Compile a rule definition into executable form."""
        existing_rules = existing_rules or {}

        # First validate
        errors, warnings = self._validation_engine.validate(definition, existing_rules)

        if any(e.severity == "error" for e in errors):
            self._log.debug(
                "compilation_failed",
                rule_id=definition.id,
                error_count=len(errors),
            )
            return CompilationResult(
                definition_id=definition.id,
                success=False,
                errors=errors,
                warnings=warnings,
            )

        # Generate bytecode
        condition_bytecode = self._compile_conditions(definition.conditions, definition.match_all)
        action_bytecode = self._compile_actions(definition.actions)

        compiled = CompiledRule(
            definition_id=definition.id,
            name=definition.name,
            rule_type=definition.rule_type,
            priority=definition.priority.value,
            status=RuleStatus.COMPILED,
            condition_bytecode=condition_bytecode,
            action_bytecode=action_bytecode,
            validation_errors=tuple(e for e in errors if e.severity != "error"),
        )

        self._log.debug(
            "compilation_success",
            rule_id=definition.id,
            compiled_id=compiled.id,
        )

        return CompilationResult(
            definition_id=definition.id,
            success=True,
            compiled_rule=compiled,
            errors=errors,
            warnings=warnings,
        )

    def _compile_conditions(
        self,
        conditions: tuple[RuleCondition, ...],
        match_all: bool,
    ) -> str:
        """Compile conditions into bytecode representation."""
        # Generate a simple bytecode representation
        op = "AND" if match_all else "OR"
        parts: list[str] = []

        for cond in conditions:
            neg = "NOT " if cond.negate else ""
            parts.append(f"{neg}({cond.field} {cond.operator} {repr(cond.value)})")

        return f" {op} ".join(parts) if parts else "TRUE"

    def _compile_actions(self, actions: tuple[RuleAction, ...]) -> str:
        """Compile actions into bytecode representation."""
        parts: list[str] = []

        for action in actions:
            params = ",".join(f"{k}={repr(v)}" for k, v in action.parameters.items())
            if params:
                parts.append(f"{action.action_type}({action.target},{repr(action.value)},{params})")
            else:
                parts.append(f"{action.action_type}({action.target},{repr(action.value)})")

        return ";".join(parts) if parts else "NOOP"


class RuleCompiler(Subsystem):
    """
    Compiles and validates symbolic rules.

    Process Loop:
    1. Intake: Receive and parse rule definitions
    2. Process: Validate rules against existing rules and dependencies
    3. Evaluate: Compile valid rules into executable form
    4. Integrate: Store compiled rules and emit events
    """

    def __init__(self) -> None:
        metadata = SubsystemMetadata(
            name="rule_compiler",
            display_name="Rule Compiler",
            description="Compiles and validates symbolic rules",
            type=SubsystemType.CORE_PROCESSING,
            tags=frozenset(["rules", "compilation", "validation", "constraints"]),
            input_types=frozenset(["RULE", "SCHEMA"]),
            output_types=frozenset(["RULE", "CODE"]),
            subscribed_topics=frozenset([
                str(SubsystemTopics.ALL),
                "rule.submit.#",
                "rule.validate.#",
                "rule.compile.#",
            ]),
            published_topics=frozenset([
                "rule.compiled",
                "rule.validated",
                "rule.error",
            ]),
        )
        super().__init__(metadata)

        self._compilation_engine = CompilationEngine()
        self._validation_engine = ValidationEngine()
        self._definitions: dict[str, RuleDefinition] = {}
        self._compiled_rules: dict[str, CompiledRule] = {}

    @property
    def definition_count(self) -> int:
        return len(self._definitions)

    @property
    def compiled_count(self) -> int:
        return len(self._compiled_rules)

    async def intake(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> SymbolicInput:
        """Phase 1: Receive and parse rule definitions."""
        if not input_data.values:
            self._log.debug("empty_input")
            return input_data

        # Filter to rule-related types
        supported_types = {SymbolicValueType.RULE, SymbolicValueType.SCHEMA}
        valid_values = [v for v in input_data.values if v.type in supported_types]

        self._log.debug(
            "intake_complete",
            total=len(input_data.values),
            valid=len(valid_values),
        )

        if len(valid_values) != len(input_data.values):
            return SymbolicInput(
                values=tuple(valid_values),
                source_subsystem=input_data.source_subsystem,
                target_subsystem=input_data.target_subsystem,
                correlation_id=input_data.correlation_id,
                priority=input_data.priority,
                metadata=input_data.metadata,
            )
        return input_data

    async def process(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> list[CompilationResult]:
        """Phase 2: Validate and compile rules."""
        results: list[CompilationResult] = []

        for value in input_data.values:
            definition = self._parse_definition(value)
            if definition:
                # Store definition
                self._definitions[definition.id] = definition

                # Compile the rule
                result = self._compilation_engine.compile(
                    definition,
                    self._definitions,
                )
                results.append(result)

                if result.success and result.compiled_rule:
                    self._compiled_rules[result.compiled_rule.id] = result.compiled_rule

                self._log.debug(
                    "rule_processed",
                    definition_id=definition.id,
                    success=result.success,
                    errors=len(result.errors),
                )

        return results

    async def evaluate(
        self, intermediate: list[CompilationResult], ctx: ProcessContext[dict[str, Any]]
    ) -> tuple[SymbolicOutput, bool]:
        """Phase 3: Create output with compiled rules."""
        values: list[SymbolicValue] = []

        for result in intermediate:
            if result.success and result.compiled_rule:
                # Create a CODE value for the compiled rule
                value = SymbolicValue(
                    type=SymbolicValueType.CODE,
                    content={
                        "compiled_id": result.compiled_rule.id,
                        "definition_id": result.definition_id,
                        "name": result.compiled_rule.name,
                        "rule_type": result.compiled_rule.rule_type.name,
                        "priority": result.compiled_rule.priority,
                        "condition_bytecode": result.compiled_rule.condition_bytecode,
                        "action_bytecode": result.compiled_rule.action_bytecode,
                        "can_execute": result.compiled_rule.can_execute(),
                    },
                    source_subsystem=self.name,
                    tags=frozenset(["rule", "compiled", result.compiled_rule.rule_type.name.lower()]),
                    meaning=f"Compiled rule: {result.compiled_rule.name}",
                    confidence=1.0 if not result.warnings else 0.9,
                )
                values.append(value)
            else:
                # Create error value for failed compilation
                value = SymbolicValue(
                    type=SymbolicValueType.SCHEMA,
                    content={
                        "definition_id": result.definition_id,
                        "success": False,
                        "errors": [
                            {"code": e.code, "message": e.message, "location": e.location}
                            for e in result.errors
                        ],
                        "warnings": [
                            {"code": w.code, "message": w.message, "location": w.location}
                            for w in result.warnings
                        ],
                    },
                    source_subsystem=self.name,
                    tags=frozenset(["rule", "error", "validation"]),
                    meaning=f"Compilation failed with {len(result.errors)} errors",
                    confidence=0.0,
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
        """Phase 4: Emit events for compiled rules."""
        if self._message_bus and output.values:
            for value in output.values:
                if value.type == SymbolicValueType.CODE:
                    await self.emit_event(
                        "rule.compiled",
                        {
                            "compiled_id": value.content.get("compiled_id"),
                            "name": value.content.get("name"),
                            "rule_type": value.content.get("rule_type"),
                        },
                    )
                elif value.type == SymbolicValueType.SCHEMA and not value.content.get("success"):
                    await self.emit_event(
                        "rule.error",
                        {
                            "definition_id": value.content.get("definition_id"),
                            "errors": value.content.get("errors"),
                        },
                    )

        return None

    def _parse_definition(self, value: SymbolicValue) -> RuleDefinition | None:
        """Parse a SymbolicValue into a RuleDefinition."""
        content = value.content
        if not isinstance(content, dict):
            self._log.warning("invalid_rule_content", value_id=value.id)
            return None

        try:
            # Extract conditions
            conditions: list[RuleCondition] = []
            for cond_data in content.get("conditions", []):
                conditions.append(RuleCondition(
                    field=cond_data.get("field", ""),
                    operator=cond_data.get("operator", "eq"),
                    value=cond_data.get("value"),
                    negate=cond_data.get("negate", False),
                ))

            # Extract actions
            actions: list[RuleAction] = []
            for action_data in content.get("actions", []):
                actions.append(RuleAction(
                    action_type=action_data.get("action_type", "log"),
                    target=action_data.get("target", ""),
                    value=action_data.get("value"),
                    parameters=action_data.get("parameters", {}),
                ))

            # Create definition
            rule_type_str = content.get("rule_type", "CONSTRAINT")
            try:
                rule_type = RuleType[rule_type_str.upper()]
            except KeyError:
                rule_type = RuleType.CONSTRAINT

            priority_str = content.get("priority", "NORMAL")
            try:
                priority = RulePriority[priority_str.upper()]
            except KeyError:
                priority = RulePriority.NORMAL

            definition = RuleDefinition(
                id=content.get("id", str(ULID())),
                name=content.get("name", f"rule_{value.id[:8]}"),
                description=content.get("description", ""),
                rule_type=rule_type,
                priority=priority,
                enabled=content.get("enabled", True),
                conditions=tuple(conditions),
                actions=tuple(actions),
                match_all=content.get("match_all", True),
                depends_on=frozenset(content.get("depends_on", [])),
                conflicts_with=frozenset(content.get("conflicts_with", [])),
                tags=frozenset(content.get("tags", [])) | value.tags,
            )

            return definition

        except Exception as e:
            self._log.warning("parse_failed", value_id=value.id, error=str(e))
            return None

    # --- Message handlers ---

    async def handle_event(self, message: Message) -> None:
        """Handle incoming events."""
        if message.topic.startswith("rule.submit"):
            self._log.debug("rule_submission_received", message_id=message.id)
        elif message.topic.startswith("rule.validate"):
            self._log.debug("validation_request_received", message_id=message.id)

    async def handle_signal(self, signal: Any) -> None:
        """Handle incoming signals."""
        self._log.debug("signal_received", signal_id=getattr(signal, "id", "unknown"))

    # --- Query API ---

    def get_definition(self, definition_id: str) -> RuleDefinition | None:
        """Get a rule definition by ID."""
        return self._definitions.get(definition_id)

    def get_compiled_rule(self, compiled_id: str) -> CompiledRule | None:
        """Get a compiled rule by ID."""
        return self._compiled_rules.get(compiled_id)

    def get_rules_by_type(self, rule_type: RuleType) -> list[CompiledRule]:
        """Get all compiled rules of a specific type."""
        return [
            r for r in self._compiled_rules.values()
            if r.rule_type == rule_type
        ]

    def get_executable_rules(self) -> list[CompiledRule]:
        """Get all rules that can be executed."""
        return [r for r in self._compiled_rules.values() if r.can_execute()]

    def validate_rule(self, definition: RuleDefinition) -> tuple[list[ValidationError], list[ValidationError]]:
        """Validate a rule definition without compiling."""
        return self._validation_engine.validate(definition, self._definitions)

    def deactivate_rule(self, compiled_id: str) -> bool:
        """Deactivate a compiled rule."""
        if compiled_id in self._compiled_rules:
            # Create a new rule with deprecated status
            old = self._compiled_rules[compiled_id]
            # Since CompiledRule is frozen, we need to remove it
            # In a real system, we'd update status in a mutable way
            del self._compiled_rules[compiled_id]
            self._log.info("rule_deactivated", compiled_id=compiled_id)
            return True
        return False

    def clear_rules(self) -> tuple[int, int]:
        """Clear all rules. Returns (definitions_cleared, compiled_cleared)."""
        def_count = len(self._definitions)
        comp_count = len(self._compiled_rules)
        self._definitions.clear()
        self._compiled_rules.clear()
        return def_count, comp_count
