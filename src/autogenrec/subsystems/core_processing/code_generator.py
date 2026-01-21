"""
CodeGenerator: Transforms symbolic structures into executable instructions.

Translates abstract symbolic structures into actionable code, bridging the gap
between expression and execution.
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


class OutputLanguage(Enum):
    """Target output languages."""

    PYTHON = auto()
    JAVASCRIPT = auto()
    JSON = auto()
    YAML = auto()
    SQL = auto()
    BASH = auto()
    PSEUDOCODE = auto()
    SYMBOLIC = auto()  # Internal symbolic representation


class GenerationStrategy(Enum):
    """Code generation strategies."""

    TEMPLATE = auto()  # Use templates
    TRANSFORM = auto()  # Direct transformation
    COMPOSE = auto()  # Compose from parts
    OPTIMIZE = auto()  # Generate optimized code


class ValidationStatus(Enum):
    """Code validation status."""

    PENDING = auto()
    VALID = auto()
    INVALID = auto()
    WARNINGS = auto()


class SymbolicStructure(BaseModel):
    """A symbolic structure to generate code from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    name: str
    structure_type: str = "generic"  # rule, schema, pattern, workflow, etc.

    # Structure definition
    definition: dict[str, Any] = Field(default_factory=dict)
    inputs: tuple[str, ...] = Field(default_factory=tuple)
    outputs: tuple[str, ...] = Field(default_factory=tuple)

    # Metadata
    version: str = "1.0"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tags: frozenset[str] = Field(default_factory=frozenset)


class CodeTemplate(BaseModel):
    """A code template for generation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    name: str
    language: OutputLanguage
    description: str = ""

    # Template content
    template: str
    placeholders: tuple[str, ...] = Field(default_factory=tuple)

    # Matching
    applies_to: frozenset[str] = Field(default_factory=frozenset)  # structure types
    priority: int = 0

    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class GeneratedCode(BaseModel):
    """Generated code output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    source_id: str  # Source structure ID
    language: OutputLanguage
    strategy: GenerationStrategy

    # Generated content
    code: str
    entry_point: str | None = None  # Main function/class name
    dependencies: tuple[str, ...] = Field(default_factory=tuple)

    # Validation
    validation_status: ValidationStatus = ValidationStatus.PENDING
    validation_errors: tuple[str, ...] = Field(default_factory=tuple)
    validation_warnings: tuple[str, ...] = Field(default_factory=tuple)

    # Metadata
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    template_id: str | None = None
    line_count: int = 0
    tags: frozenset[str] = Field(default_factory=frozenset)


class ExecutionPlan(BaseModel):
    """A plan for executing generated code."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    code_id: str
    name: str = ""

    # Execution steps
    steps: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    environment: dict[str, str] = Field(default_factory=dict)

    # Resources
    required_inputs: tuple[str, ...] = Field(default_factory=tuple)
    expected_outputs: tuple[str, ...] = Field(default_factory=tuple)

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


@dataclass
class GenerationResult:
    """Result of code generation."""

    code_id: str
    success: bool
    code: GeneratedCode | None = None
    error: str | None = None


@dataclass
class ValidationResult:
    """Result of code validation."""

    code_id: str
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class GeneratorStats:
    """Statistics about code generation."""

    total_structures: int
    total_templates: int
    total_generated: int
    valid_count: int
    by_language: dict[str, int]


class StructureRegistry:
    """Registry of symbolic structures."""

    def __init__(self) -> None:
        self._structures: dict[str, SymbolicStructure] = {}
        self._by_type: dict[str, list[str]] = {}
        self._log = logger.bind(component="structure_registry")

    @property
    def structure_count(self) -> int:
        return len(self._structures)

    def register(self, structure: SymbolicStructure) -> None:
        """Register a structure."""
        self._structures[structure.id] = structure
        self._by_type.setdefault(structure.structure_type, []).append(structure.id)
        self._log.debug("structure_registered", structure_id=structure.id, name=structure.name)

    def get(self, structure_id: str) -> SymbolicStructure | None:
        """Get a structure by ID."""
        return self._structures.get(structure_id)

    def get_by_type(self, structure_type: str) -> list[SymbolicStructure]:
        """Get structures by type."""
        ids = self._by_type.get(structure_type, [])
        return [self._structures[sid] for sid in ids if sid in self._structures]


class TemplateRegistry:
    """Registry of code templates."""

    def __init__(self) -> None:
        self._templates: dict[str, CodeTemplate] = {}
        self._by_language: dict[OutputLanguage, list[str]] = {}
        self._log = logger.bind(component="template_registry")

    @property
    def template_count(self) -> int:
        return len(self._templates)

    def register(self, template: CodeTemplate) -> None:
        """Register a template."""
        self._templates[template.id] = template
        self._by_language.setdefault(template.language, []).append(template.id)
        self._log.debug("template_registered", template_id=template.id, name=template.name)

    def get(self, template_id: str) -> CodeTemplate | None:
        """Get a template by ID."""
        return self._templates.get(template_id)

    def get_for_language(self, language: OutputLanguage) -> list[CodeTemplate]:
        """Get templates for a language."""
        ids = self._by_language.get(language, [])
        return [self._templates[tid] for tid in ids if tid in self._templates]

    def find_template(
        self,
        language: OutputLanguage,
        structure_type: str,
    ) -> CodeTemplate | None:
        """Find the best matching template."""
        templates = self.get_for_language(language)
        for template in sorted(templates, key=lambda t: -t.priority):
            if not template.is_active:
                continue
            if not template.applies_to or structure_type in template.applies_to:
                return template
        return None


class CodeCompiler:
    """Compiles symbolic structures into code."""

    def __init__(
        self,
        structures: StructureRegistry,
        templates: TemplateRegistry,
    ) -> None:
        self._structures = structures
        self._templates = templates
        self._generated: dict[str, GeneratedCode] = {}
        self._plans: dict[str, ExecutionPlan] = {}
        self._log = logger.bind(component="code_compiler")

    @property
    def generated_count(self) -> int:
        return len(self._generated)

    def generate(
        self,
        source_id: str,
        language: OutputLanguage,
        strategy: GenerationStrategy = GenerationStrategy.TEMPLATE,
    ) -> GenerationResult:
        """Generate code from a symbolic structure."""
        structure = self._structures.get(source_id)
        if not structure:
            return GenerationResult(code_id="", success=False, error="Structure not found")

        try:
            # Generate based on strategy
            if strategy == GenerationStrategy.TEMPLATE:
                code = self._generate_from_template(structure, language)
            elif strategy == GenerationStrategy.TRANSFORM:
                code = self._generate_transform(structure, language)
            elif strategy == GenerationStrategy.COMPOSE:
                code = self._generate_compose(structure, language)
            else:
                code = self._generate_transform(structure, language)

            self._generated[code.id] = code
            return GenerationResult(code_id=code.id, success=True, code=code)

        except Exception as e:
            return GenerationResult(code_id="", success=False, error=str(e))

    def _generate_from_template(
        self,
        structure: SymbolicStructure,
        language: OutputLanguage,
    ) -> GeneratedCode:
        """Generate using templates."""
        template = self._templates.find_template(language, structure.structure_type)

        if template:
            code = self._apply_template(template, structure)
            template_id = template.id
        else:
            code = self._generate_default(structure, language)
            template_id = None

        return GeneratedCode(
            source_id=structure.id,
            language=language,
            strategy=GenerationStrategy.TEMPLATE,
            code=code,
            entry_point=structure.name,
            template_id=template_id,
            line_count=code.count("\n") + 1,
            tags=structure.tags,
        )

    def _generate_transform(
        self,
        structure: SymbolicStructure,
        language: OutputLanguage,
    ) -> GeneratedCode:
        """Generate via direct transformation."""
        code = self._generate_default(structure, language)

        return GeneratedCode(
            source_id=structure.id,
            language=language,
            strategy=GenerationStrategy.TRANSFORM,
            code=code,
            entry_point=structure.name,
            line_count=code.count("\n") + 1,
            tags=structure.tags,
        )

    def _generate_compose(
        self,
        structure: SymbolicStructure,
        language: OutputLanguage,
    ) -> GeneratedCode:
        """Generate by composing parts."""
        parts = []

        # Header
        parts.append(self._generate_header(structure, language))

        # Imports/dependencies
        parts.append(self._generate_imports(structure, language))

        # Main body
        parts.append(self._generate_body(structure, language))

        # Footer
        parts.append(self._generate_footer(structure, language))

        code = "\n\n".join(p for p in parts if p)

        return GeneratedCode(
            source_id=structure.id,
            language=language,
            strategy=GenerationStrategy.COMPOSE,
            code=code,
            entry_point=structure.name,
            line_count=code.count("\n") + 1,
            tags=structure.tags,
        )

    def _apply_template(
        self,
        template: CodeTemplate,
        structure: SymbolicStructure,
    ) -> str:
        """Apply a template to a structure."""
        code = template.template

        # Replace placeholders
        replacements = {
            "name": structure.name,
            "inputs": ", ".join(structure.inputs),
            "outputs": ", ".join(structure.outputs),
            "version": structure.version,
        }

        # Add definition values
        for key, value in structure.definition.items():
            replacements[key] = str(value) if not isinstance(value, str) else value

        for placeholder, value in replacements.items():
            code = code.replace(f"{{{{ {placeholder} }}}}", value)
            code = code.replace(f"{{{{{placeholder}}}}}", value)

        return code

    def _generate_default(
        self,
        structure: SymbolicStructure,
        language: OutputLanguage,
    ) -> str:
        """Generate default code for a language."""
        if language == OutputLanguage.PYTHON:
            return self._to_python(structure)
        elif language == OutputLanguage.JAVASCRIPT:
            return self._to_javascript(structure)
        elif language == OutputLanguage.JSON:
            return self._to_json(structure)
        elif language == OutputLanguage.YAML:
            return self._to_yaml(structure)
        elif language == OutputLanguage.SQL:
            return self._to_sql(structure)
        elif language == OutputLanguage.BASH:
            return self._to_bash(structure)
        elif language == OutputLanguage.PSEUDOCODE:
            return self._to_pseudocode(structure)
        else:
            return self._to_symbolic(structure)

    def _to_python(self, structure: SymbolicStructure) -> str:
        """Generate Python code."""
        lines = [
            f'"""',
            f'Generated from: {structure.name}',
            f'Type: {structure.structure_type}',
            f'Version: {structure.version}',
            f'"""',
            '',
            f'def {self._to_identifier(structure.name)}({", ".join(structure.inputs)}):',
            f'    """',
            f'    {structure.structure_type} implementation.',
            f'    ',
            f'    Args:',
        ]

        for inp in structure.inputs:
            lines.append(f'        {inp}: Input parameter')

        lines.extend([
            f'    ',
            f'    Returns:',
            f'        tuple: ({", ".join(structure.outputs)})',
            f'    """',
        ])

        # Add definition-based logic
        if structure.definition:
            lines.append(f'    # Definition: {structure.definition}')

        # Default implementation
        if structure.outputs:
            outputs = ", ".join(f"None  # {out}" for out in structure.outputs)
            lines.append(f'    return {outputs}')
        else:
            lines.append('    pass')

        return "\n".join(lines)

    def _to_javascript(self, structure: SymbolicStructure) -> str:
        """Generate JavaScript code."""
        lines = [
            '/**',
            f' * Generated from: {structure.name}',
            f' * Type: {structure.structure_type}',
            f' * Version: {structure.version}',
            ' */',
            '',
            f'function {self._to_identifier(structure.name)}({", ".join(structure.inputs)}) {{',
        ]

        if structure.outputs:
            lines.append(f'    // Returns: {", ".join(structure.outputs)}')
            lines.append('    return {')
            for out in structure.outputs:
                lines.append(f'        {out}: null,')
            lines.append('    };')
        else:
            lines.append('    // No outputs')

        lines.append('}')

        return "\n".join(lines)

    def _to_json(self, structure: SymbolicStructure) -> str:
        """Generate JSON representation."""
        import json
        return json.dumps({
            "name": structure.name,
            "type": structure.structure_type,
            "version": structure.version,
            "inputs": list(structure.inputs),
            "outputs": list(structure.outputs),
            "definition": structure.definition,
        }, indent=2)

    def _to_yaml(self, structure: SymbolicStructure) -> str:
        """Generate YAML representation."""
        lines = [
            f'# {structure.name}',
            f'name: {structure.name}',
            f'type: {structure.structure_type}',
            f'version: "{structure.version}"',
            'inputs:',
        ]
        for inp in structure.inputs:
            lines.append(f'  - {inp}')
        lines.append('outputs:')
        for out in structure.outputs:
            lines.append(f'  - {out}')

        if structure.definition:
            lines.append('definition:')
            for key, value in structure.definition.items():
                lines.append(f'  {key}: {value}')

        return "\n".join(lines)

    def _to_sql(self, structure: SymbolicStructure) -> str:
        """Generate SQL representation."""
        lines = [
            f'-- Generated from: {structure.name}',
            f'-- Type: {structure.structure_type}',
            '',
        ]

        # Create a stored procedure
        inputs = ", ".join(f"p_{inp} TEXT" for inp in structure.inputs)
        lines.extend([
            f'CREATE OR REPLACE FUNCTION {self._to_identifier(structure.name)}({inputs})',
            f'RETURNS TABLE({", ".join(f"{out} TEXT" for out in structure.outputs)}) AS $$',
            'BEGIN',
            '    -- Implementation',
            '    RETURN QUERY SELECT ' + ", ".join(f"NULL::{out}" for out in structure.outputs) + ";",
            'END;',
            '$$ LANGUAGE plpgsql;',
        ])

        return "\n".join(lines)

    def _to_bash(self, structure: SymbolicStructure) -> str:
        """Generate Bash script."""
        lines = [
            '#!/bin/bash',
            f'# Generated from: {structure.name}',
            f'# Type: {structure.structure_type}',
            '',
            f'{self._to_identifier(structure.name)}() {{',
        ]

        for i, inp in enumerate(structure.inputs, 1):
            lines.append(f'    local {inp}="${i}"')

        lines.extend([
            '    ',
            '    # Implementation',
            '    echo "Processing..."',
            '}',
        ])

        return "\n".join(lines)

    def _to_pseudocode(self, structure: SymbolicStructure) -> str:
        """Generate pseudocode."""
        lines = [
            f'PROCEDURE {structure.name}',
            f'  TYPE: {structure.structure_type}',
            '',
            '  INPUTS:',
        ]
        for inp in structure.inputs:
            lines.append(f'    - {inp}')

        lines.extend([
            '',
            '  OUTPUTS:',
        ])
        for out in structure.outputs:
            lines.append(f'    - {out}')

        lines.extend([
            '',
            '  BEGIN',
            '    // Process inputs',
            '    // Generate outputs',
            '  END',
        ])

        return "\n".join(lines)

    def _to_symbolic(self, structure: SymbolicStructure) -> str:
        """Generate symbolic representation."""
        return f"""SYMBOLIC_STRUCTURE {{
  name: "{structure.name}"
  type: {structure.structure_type}
  version: "{structure.version}"
  
  inputs: [{", ".join(structure.inputs)}]
  outputs: [{", ".join(structure.outputs)}]
  
  definition: {structure.definition}
}}"""

    def _generate_header(
        self,
        structure: SymbolicStructure,
        language: OutputLanguage,
    ) -> str:
        """Generate code header."""
        if language == OutputLanguage.PYTHON:
            return f'"""\n{structure.name}\n\nGenerated: {datetime.now(UTC).isoformat()}\n"""'
        elif language == OutputLanguage.JAVASCRIPT:
            return f'/**\n * {structure.name}\n * Generated: {datetime.now(UTC).isoformat()}\n */'
        else:
            return f'# {structure.name}'

    def _generate_imports(
        self,
        structure: SymbolicStructure,
        language: OutputLanguage,
    ) -> str:
        """Generate imports/dependencies."""
        if language == OutputLanguage.PYTHON:
            return "from typing import Any"
        elif language == OutputLanguage.JAVASCRIPT:
            return "// No imports required"
        else:
            return ""

    def _generate_body(
        self,
        structure: SymbolicStructure,
        language: OutputLanguage,
    ) -> str:
        """Generate main body."""
        return self._generate_default(structure, language)

    def _generate_footer(
        self,
        structure: SymbolicStructure,
        language: OutputLanguage,
    ) -> str:
        """Generate code footer."""
        if language == OutputLanguage.PYTHON:
            return f'\nif __name__ == "__main__":\n    {self._to_identifier(structure.name)}()'
        elif language == OutputLanguage.JAVASCRIPT:
            return f'\n// Export\nmodule.exports = {{ {self._to_identifier(structure.name)} }};'
        else:
            return ""

    def _to_identifier(self, name: str) -> str:
        """Convert name to valid identifier."""
        import re
        identifier = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        if identifier[0].isdigit():
            identifier = '_' + identifier
        return identifier.lower()

    def validate(self, code_id: str) -> ValidationResult:
        """Validate generated code."""
        code = self._generated.get(code_id)
        if not code:
            return ValidationResult(code_id=code_id, valid=False, errors=["Code not found"])

        errors: list[str] = []
        warnings: list[str] = []

        # Basic validation
        if not code.code.strip():
            errors.append("Generated code is empty")

        # Language-specific validation
        if code.language == OutputLanguage.PYTHON:
            self._validate_python(code.code, errors, warnings)
        elif code.language == OutputLanguage.JSON:
            self._validate_json(code.code, errors, warnings)

        # Update code with validation status
        if errors:
            status = ValidationStatus.INVALID
        elif warnings:
            status = ValidationStatus.WARNINGS
        else:
            status = ValidationStatus.VALID

        updated = GeneratedCode(
            id=code.id,
            source_id=code.source_id,
            language=code.language,
            strategy=code.strategy,
            code=code.code,
            entry_point=code.entry_point,
            dependencies=code.dependencies,
            validation_status=status,
            validation_errors=tuple(errors),
            validation_warnings=tuple(warnings),
            generated_at=code.generated_at,
            template_id=code.template_id,
            line_count=code.line_count,
            tags=code.tags,
        )
        self._generated[code_id] = updated

        return ValidationResult(
            code_id=code_id,
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _validate_python(
        self,
        code: str,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        """Validate Python code."""
        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")

        # Check for common issues
        if "import *" in code:
            warnings.append("Wildcard imports are discouraged")

    def _validate_json(
        self,
        code: str,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        """Validate JSON code."""
        import json
        try:
            json.loads(code)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON: {e}")

    def create_execution_plan(self, code_id: str) -> ExecutionPlan | None:
        """Create an execution plan for generated code."""
        code = self._generated.get(code_id)
        if not code:
            return None

        structure = self._structures.get(code.source_id)

        steps = [
            {"step": 1, "action": "setup", "description": "Prepare environment"},
            {"step": 2, "action": "load", "description": "Load dependencies"},
            {"step": 3, "action": "execute", "description": f"Run {code.entry_point}"},
            {"step": 4, "action": "collect", "description": "Collect outputs"},
        ]

        plan = ExecutionPlan(
            code_id=code_id,
            name=f"Plan for {code.entry_point}",
            steps=tuple(steps),
            environment={"language": code.language.name},
            required_inputs=structure.inputs if structure else (),
            expected_outputs=structure.outputs if structure else (),
        )
        self._plans[plan.id] = plan
        return plan

    def get_code(self, code_id: str) -> GeneratedCode | None:
        """Get generated code."""
        return self._generated.get(code_id)

    def get_plan(self, plan_id: str) -> ExecutionPlan | None:
        """Get an execution plan."""
        return self._plans.get(plan_id)


class CodeGenerator(Subsystem):
    """
    Transforms symbolic structures into executable code.

    Process Loop:
    1. Submit: Receive symbolic input
    2. Compile: Convert symbolic forms into code
    3. Test: Validate the generated code for correctness
    4. Integrate: Deploy compiled code into system runtime
    """

    def __init__(self) -> None:
        metadata = SubsystemMetadata(
            name="code_generator",
            display_name="Code Generator",
            description="Transforms symbolic structures into executable instructions",
            type=SubsystemType.CORE_PROCESSING,
            tags=frozenset(["code", "generation", "executable", "compilation"]),
            input_types=frozenset(["RULE", "SCHEMA", "PATTERN", "STRUCTURE"]),
            output_types=frozenset(["CODE", "PLAN"]),
            subscribed_topics=frozenset([
                str(SubsystemTopics.ALL),
                "code.#",
                "generation.#",
            ]),
            published_topics=frozenset([
                "code.generated",
                "code.validated",
                "code.deployed",
            ]),
        )
        super().__init__(metadata)

        self._structures = StructureRegistry()
        self._templates = TemplateRegistry()
        self._compiler = CodeCompiler(self._structures, self._templates)

    @property
    def structure_count(self) -> int:
        return self._structures.structure_count

    @property
    def template_count(self) -> int:
        return self._templates.template_count

    @property
    def generated_count(self) -> int:
        return self._compiler.generated_count

    async def intake(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> SymbolicInput:
        """Phase 1: Receive symbolic structures."""
        if not input_data.values:
            self._log.debug("empty_input")
            return input_data

        self._log.debug("intake_complete", value_count=len(input_data.values))
        return input_data

    async def process(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> list[GenerationResult]:
        """Phase 2: Generate code."""
        results: list[GenerationResult] = []

        for value in input_data.values:
            content = value.content
            if not isinstance(content, dict):
                continue

            action = content.get("action", "generate")

            if action == "register":
                result = self._register_from_value(value)
                results.append(result)
            elif action == "generate":
                result = self._generate_from_value(value)
                results.append(result)

        return results

    async def evaluate(
        self,
        intermediate: list[GenerationResult],
        ctx: ProcessContext[dict[str, Any]],
    ) -> tuple[SymbolicOutput, bool]:
        """Phase 3: Validate and package generated code."""
        values: list[SymbolicValue] = []

        for result in intermediate:
            value = SymbolicValue(
                type=SymbolicValueType.CODE,
                content={
                    "code_id": result.code_id,
                    "success": result.success,
                    "language": result.code.language.name if result.code else None,
                    "line_count": result.code.line_count if result.code else 0,
                    "error": result.error,
                },
                source_subsystem=self.name,
                tags=frozenset(["code", "generated"]),
                meaning="Code generation result",
                confidence=1.0 if result.success else 0.0,
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
        """Phase 4: Distribute generated code."""
        if self._message_bus and output.values:
            for value in output.values:
                content = value.content
                if not isinstance(content, dict):
                    continue

                if content.get("success"):
                    await self.emit_event(
                        "code.generated",
                        {
                            "code_id": content.get("code_id"),
                            "language": content.get("language"),
                        },
                    )

        return None

    def _register_from_value(self, value: SymbolicValue) -> GenerationResult:
        """Register a structure from a SymbolicValue."""
        content = value.content
        if not isinstance(content, dict):
            return GenerationResult(code_id="", success=False, error="Invalid content")

        try:
            structure = SymbolicStructure(
                name=content.get("name", f"structure_{value.id[:8]}"),
                structure_type=content.get("structure_type", "generic"),
                definition=content.get("definition", {}),
                inputs=tuple(content.get("inputs", [])),
                outputs=tuple(content.get("outputs", [])),
                version=content.get("version", "1.0"),
                tags=frozenset(content.get("tags", [])) | value.tags,
            )
            self._structures.register(structure)
            return GenerationResult(code_id=structure.id, success=True)

        except Exception as e:
            return GenerationResult(code_id="", success=False, error=str(e))

    def _generate_from_value(self, value: SymbolicValue) -> GenerationResult:
        """Generate code from a SymbolicValue request."""
        content = value.content
        if not isinstance(content, dict):
            return GenerationResult(code_id="", success=False, error="Invalid content")

        source_id = content.get("source_id")
        if not source_id:
            return GenerationResult(code_id="", success=False, error="source_id required")

        lang_str = content.get("language", "PYTHON")
        try:
            language = OutputLanguage[lang_str.upper()]
        except KeyError:
            language = OutputLanguage.PYTHON

        strategy_str = content.get("strategy", "TEMPLATE")
        try:
            strategy = GenerationStrategy[strategy_str.upper()]
        except KeyError:
            strategy = GenerationStrategy.TEMPLATE

        return self._compiler.generate(source_id, language, strategy)

    # --- Message handlers ---

    async def handle_event(self, message: Message) -> None:
        """Handle incoming events."""
        if message.topic.startswith("code.") or message.topic.startswith("generation."):
            self._log.debug("event_received", topic=message.topic)

    async def handle_signal(self, signal: Any) -> None:
        """Handle incoming signals."""
        self._log.debug("signal_received", signal_id=getattr(signal, "id", "unknown"))

    # --- Public API ---

    def register_structure(
        self,
        name: str,
        structure_type: str = "generic",
        **kwargs: Any,
    ) -> SymbolicStructure:
        """Register a symbolic structure."""
        structure = SymbolicStructure(
            name=name,
            structure_type=structure_type,
            definition=kwargs.get("definition", {}),
            inputs=tuple(kwargs.get("inputs", [])),
            outputs=tuple(kwargs.get("outputs", [])),
            version=kwargs.get("version", "1.0"),
            tags=frozenset(kwargs.get("tags", [])),
        )
        self._structures.register(structure)
        return structure

    def get_structure(self, structure_id: str) -> SymbolicStructure | None:
        """Get a structure by ID."""
        return self._structures.get(structure_id)

    def add_template(
        self,
        name: str,
        language: OutputLanguage,
        template: str,
        **kwargs: Any,
    ) -> CodeTemplate:
        """Add a code template."""
        code_template = CodeTemplate(
            name=name,
            language=language,
            description=kwargs.get("description", ""),
            template=template,
            placeholders=tuple(kwargs.get("placeholders", [])),
            applies_to=frozenset(kwargs.get("applies_to", [])),
            priority=kwargs.get("priority", 0),
            is_active=kwargs.get("is_active", True),
        )
        self._templates.register(code_template)
        return code_template

    def get_template(self, template_id: str) -> CodeTemplate | None:
        """Get a template by ID."""
        return self._templates.get(template_id)

    def generate(
        self,
        source_id: str,
        language: OutputLanguage = OutputLanguage.PYTHON,
        strategy: GenerationStrategy = GenerationStrategy.TEMPLATE,
    ) -> GenerationResult:
        """Generate code from a structure."""
        return self._compiler.generate(source_id, language, strategy)

    def generate_from_definition(
        self,
        name: str,
        definition: dict[str, Any],
        language: OutputLanguage = OutputLanguage.PYTHON,
        **kwargs: Any,
    ) -> GenerationResult:
        """Register a structure and generate code in one step."""
        structure = self.register_structure(
            name=name,
            structure_type=kwargs.get("structure_type", "generic"),
            definition=definition,
            inputs=kwargs.get("inputs", []),
            outputs=kwargs.get("outputs", []),
        )
        return self.generate(structure.id, language)

    def get_code(self, code_id: str) -> GeneratedCode | None:
        """Get generated code."""
        return self._compiler.get_code(code_id)

    def validate(self, code_id: str) -> ValidationResult:
        """Validate generated code."""
        return self._compiler.validate(code_id)

    def create_execution_plan(self, code_id: str) -> ExecutionPlan | None:
        """Create an execution plan for code."""
        return self._compiler.create_execution_plan(code_id)

    def get_stats(self) -> GeneratorStats:
        """Get generation statistics."""
        by_language: dict[str, int] = {}
        valid_count = 0

        for code in self._compiler._generated.values():
            by_language[code.language.name] = by_language.get(code.language.name, 0) + 1
            if code.validation_status == ValidationStatus.VALID:
                valid_count += 1

        return GeneratorStats(
            total_structures=self._structures.structure_count,
            total_templates=self._templates.template_count,
            total_generated=self._compiler.generated_count,
            valid_count=valid_count,
            by_language=by_language,
        )

    def clear(self) -> tuple[int, int, int]:
        """Clear all data. Returns (structures, templates, generated) cleared."""
        structures = self._structures.structure_count
        templates = self._templates.template_count
        generated = self._compiler.generated_count
        self._structures = StructureRegistry()
        self._templates = TemplateRegistry()
        self._compiler = CodeCompiler(self._structures, self._templates)
        return structures, templates, generated
