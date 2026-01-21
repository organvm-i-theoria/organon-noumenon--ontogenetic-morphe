"""Tests for core processing subsystems: SymbolicInterpreter and RuleCompiler."""

import pytest
from datetime import UTC, datetime

from autogenrec.core.subsystem import SubsystemType
from autogenrec.core.symbolic import (
    SymbolicInput,
    SymbolicValue,
    SymbolicValueType,
)
from autogenrec.subsystems.core_processing.symbolic_interpreter import (
    ExtractedSymbol,
    InterpretationEngine,
    InterpretiveFramework,
    Pattern,
    PatternRecognizer,
    SymbolCategory,
    SymbolExtractor,
    SymbolicInterpreter,
)
from autogenrec.subsystems.core_processing.rule_compiler import (
    CompilationEngine,
    CompiledRule,
    RuleAction,
    RuleCompiler,
    RuleCondition,
    RuleDefinition,
    RulePriority,
    RuleStatus,
    RuleType,
    ValidationEngine,
    ValidationError,
)


class TestSymbolExtractor:
    def test_extract_archetype_symbols(self) -> None:
        extractor = SymbolExtractor()
        content = "The hero embarked on a journey guided by a wise mentor."
        symbols = extractor.extract(content)

        symbol_names = {s.name for s in symbols}
        assert "hero" in symbol_names
        assert "journey" in symbol_names
        assert "mentor" in symbol_names

    def test_extract_elemental_symbols(self) -> None:
        extractor = SymbolExtractor()
        content = "Water flowed through the ancient fire pit."
        symbols = extractor.extract(content)

        symbol_names = {s.name for s in symbols}
        assert "water" in symbol_names
        assert "fire" in symbol_names

    def test_extract_with_positions(self) -> None:
        extractor = SymbolExtractor()
        content = "The hero found a key"
        symbols = extractor.extract(content)

        hero_symbol = next((s for s in symbols if s.name == "hero"), None)
        assert hero_symbol is not None
        assert hero_symbol.position == content.lower().find("hero")

    def test_extract_empty_content(self) -> None:
        extractor = SymbolExtractor()
        symbols = extractor.extract("")
        assert len(symbols) == 0

    def test_extract_no_symbols(self) -> None:
        extractor = SymbolExtractor()
        content = "The quick brown fox jumps over the lazy dog."
        symbols = extractor.extract(content)
        # Should have no symbols from our keyword lists
        assert len(symbols) == 0


class TestPatternRecognizer:
    def test_recognize_hero_journey_pattern(self) -> None:
        recognizer = PatternRecognizer()
        symbols = [
            ExtractedSymbol(
                name="hero",
                category=SymbolCategory.ARCHETYPE,
                raw_fragment="the hero",
                position=0,
            ),
            ExtractedSymbol(
                name="journey",
                category=SymbolCategory.ACTION,
                raw_fragment="began journey",
                position=10,
            ),
        ]
        patterns = recognizer.recognize(symbols)

        assert len(patterns) >= 1
        pattern_names = {p.name for p in patterns}
        assert "hero_journey" in pattern_names

    def test_recognize_shadow_encounter(self) -> None:
        recognizer = PatternRecognizer()
        symbols = [
            ExtractedSymbol(
                name="shadow",
                category=SymbolCategory.ARCHETYPE,
                raw_fragment="the shadow",
                position=0,
            ),
            ExtractedSymbol(
                name="conflict",
                category=SymbolCategory.RELATIONSHIP,
                raw_fragment="great conflict",
                position=10,
            ),
        ]
        patterns = recognizer.recognize(symbols)

        pattern_names = {p.name for p in patterns}
        assert "shadow_encounter" in pattern_names

    def test_no_patterns_from_unrelated_symbols(self) -> None:
        recognizer = PatternRecognizer()
        symbols = [
            ExtractedSymbol(
                name="water",
                category=SymbolCategory.ELEMENT,
                raw_fragment="clear water",
                position=0,
            ),
        ]
        patterns = recognizer.recognize(symbols)
        # With only one element symbol, no patterns should match
        assert len(patterns) == 0


class TestInterpretationEngine:
    def test_interpret_narrative(self) -> None:
        engine = InterpretationEngine()
        value = SymbolicValue(
            type=SymbolicValueType.NARRATIVE,
            content="The hero crossed the threshold into the dark forest.",
        )
        interpretation = engine.interpret(value)

        assert interpretation.source_id == value.id
        assert len(interpretation.symbols) > 0
        assert interpretation.synthesis != ""

    def test_interpret_with_framework(self) -> None:
        engine = InterpretationEngine()
        value = SymbolicValue(
            type=SymbolicValueType.DREAM,
            content="A shadow figure pursued me through water.",
        )

        interpretation = engine.interpret(value, InterpretiveFramework.JUNGIAN)
        assert interpretation.framework == InterpretiveFramework.JUNGIAN
        assert "Archetypal analysis" in interpretation.synthesis

    def test_interpret_dict_content(self) -> None:
        engine = InterpretationEngine()
        value = SymbolicValue(
            type=SymbolicValueType.NARRATIVE,
            content={"text": "The hero found the sacred key."},
        )
        interpretation = engine.interpret(value)

        symbol_names = {s.name for s in interpretation.symbols}
        assert "hero" in symbol_names
        assert "key" in symbol_names


class TestSymbolicInterpreter:
    def test_create_interpreter(self) -> None:
        interpreter = SymbolicInterpreter()
        assert interpreter.name == "symbolic_interpreter"
        assert interpreter.metadata.type == SubsystemType.CORE_PROCESSING

    def test_supported_types(self) -> None:
        interpreter = SymbolicInterpreter()
        assert SymbolicValueType.NARRATIVE in interpreter.SUPPORTED_TYPES
        assert SymbolicValueType.DREAM in interpreter.SUPPORTED_TYPES
        assert SymbolicValueType.VISION in interpreter.SUPPORTED_TYPES
        assert SymbolicValueType.TOKEN not in interpreter.SUPPORTED_TYPES

    @pytest.mark.asyncio
    async def test_intake_filters_unsupported(self) -> None:
        interpreter = SymbolicInterpreter()
        from autogenrec.core.process import ProcessContext

        input_data = SymbolicInput(
            values=(
                SymbolicValue(type=SymbolicValueType.NARRATIVE, content="story"),
                SymbolicValue(type=SymbolicValueType.TOKEN, content="invalid"),
            )
        )
        ctx = ProcessContext[dict](iteration=1, started_at=datetime.now(UTC))

        result = await interpreter.intake(input_data, ctx)
        assert len(result.values) == 1
        assert result.values[0].type == SymbolicValueType.NARRATIVE

    @pytest.mark.asyncio
    async def test_full_process_loop(self) -> None:
        interpreter = SymbolicInterpreter()
        from autogenrec.core.process import ProcessContext

        input_data = SymbolicInput(
            values=(
                SymbolicValue(
                    type=SymbolicValueType.NARRATIVE,
                    content="The hero began a journey and crossed a threshold.",
                ),
            )
        )
        ctx = ProcessContext[dict](iteration=1, started_at=datetime.now(UTC))

        # Run through phases
        filtered = await interpreter.intake(input_data, ctx)
        interpretations = await interpreter.process(filtered, ctx)
        output, should_continue = await interpreter.evaluate(interpretations, ctx)

        assert not should_continue
        assert len(output.values) > 0

        # Check that patterns and schemas were created
        value_types = {v.type for v in output.values}
        assert SymbolicValueType.PATTERN in value_types or SymbolicValueType.SCHEMA in value_types

    def test_set_framework(self) -> None:
        interpreter = SymbolicInterpreter()
        interpreter.set_framework(InterpretiveFramework.JUNGIAN)
        assert interpreter.active_framework == InterpretiveFramework.JUNGIAN


class TestValidationEngine:
    def test_validate_valid_rule(self) -> None:
        engine = ValidationEngine()
        definition = RuleDefinition(
            name="test_rule",
            conditions=(
                RuleCondition(field="status", operator="eq", value="active"),
            ),
            actions=(
                RuleAction(action_type="log", target="audit_log"),
            ),
        )
        errors, warnings = engine.validate(definition)
        assert len([e for e in errors if e.severity == "error"]) == 0

    def test_validate_empty_name(self) -> None:
        engine = ValidationEngine()
        definition = RuleDefinition(
            name="",
            conditions=(
                RuleCondition(field="status", operator="eq", value="active"),
            ),
        )
        errors, warnings = engine.validate(definition)
        error_codes = {e.code for e in errors}
        assert "EMPTY_NAME" in error_codes

    def test_validate_invalid_operator(self) -> None:
        engine = ValidationEngine()
        definition = RuleDefinition(
            name="test_rule",
            conditions=(
                RuleCondition(field="status", operator="invalid_op", value="active"),
            ),
        )
        errors, warnings = engine.validate(definition)
        error_codes = {e.code for e in errors}
        assert "INVALID_OPERATOR" in error_codes

    def test_validate_invalid_action_type(self) -> None:
        engine = ValidationEngine()
        definition = RuleDefinition(
            name="test_rule",
            actions=(
                RuleAction(action_type="invalid_action", target="target"),
            ),
        )
        errors, warnings = engine.validate(definition)
        error_codes = {e.code for e in errors}
        assert "INVALID_ACTION_TYPE" in error_codes

    def test_validate_missing_dependency(self) -> None:
        engine = ValidationEngine()
        definition = RuleDefinition(
            name="test_rule",
            depends_on=frozenset(["nonexistent_rule"]),
            conditions=(
                RuleCondition(field="status", operator="eq", value="active"),
            ),
        )
        errors, warnings = engine.validate(definition, {})
        error_codes = {e.code for e in errors}
        assert "MISSING_DEPENDENCY" in error_codes


class TestCompilationEngine:
    def test_compile_valid_rule(self) -> None:
        engine = CompilationEngine()
        definition = RuleDefinition(
            name="test_rule",
            rule_type=RuleType.CONSTRAINT,
            conditions=(
                RuleCondition(field="status", operator="eq", value="active"),
            ),
            actions=(
                RuleAction(action_type="log", target="audit"),
            ),
        )
        result = engine.compile(definition)

        assert result.success
        assert result.compiled_rule is not None
        assert result.compiled_rule.name == "test_rule"
        assert result.compiled_rule.can_execute()

    def test_compile_invalid_rule_fails(self) -> None:
        engine = CompilationEngine()
        definition = RuleDefinition(
            name="",  # Invalid
            conditions=(
                RuleCondition(field="status", operator="invalid", value="x"),
            ),
        )
        result = engine.compile(definition)

        assert not result.success
        assert result.compiled_rule is None
        assert len(result.errors) > 0

    def test_compile_conditions_and_actions(self) -> None:
        engine = CompilationEngine()
        definition = RuleDefinition(
            name="complex_rule",
            conditions=(
                RuleCondition(field="age", operator="gte", value=18),
                RuleCondition(field="status", operator="eq", value="active"),
            ),
            actions=(
                RuleAction(action_type="allow", target="access"),
                RuleAction(action_type="log", target="audit"),
            ),
            match_all=True,
        )
        result = engine.compile(definition)

        assert result.success
        assert "AND" in result.compiled_rule.condition_bytecode
        assert "allow" in result.compiled_rule.action_bytecode
        assert "log" in result.compiled_rule.action_bytecode


class TestRuleCompiler:
    def test_create_compiler(self) -> None:
        compiler = RuleCompiler()
        assert compiler.name == "rule_compiler"
        assert compiler.metadata.type == SubsystemType.CORE_PROCESSING

    @pytest.mark.asyncio
    async def test_intake_filters_types(self) -> None:
        compiler = RuleCompiler()
        from autogenrec.core.process import ProcessContext

        input_data = SymbolicInput(
            values=(
                SymbolicValue(
                    type=SymbolicValueType.RULE,
                    content={"name": "test"},
                ),
                SymbolicValue(
                    type=SymbolicValueType.NARRATIVE,
                    content="invalid",
                ),
            )
        )
        ctx = ProcessContext[dict](iteration=1, started_at=datetime.now(UTC))

        result = await compiler.intake(input_data, ctx)
        assert len(result.values) == 1
        assert result.values[0].type == SymbolicValueType.RULE

    @pytest.mark.asyncio
    async def test_full_process_loop(self) -> None:
        compiler = RuleCompiler()
        from autogenrec.core.process import ProcessContext

        input_data = SymbolicInput(
            values=(
                SymbolicValue(
                    type=SymbolicValueType.RULE,
                    content={
                        "name": "access_control",
                        "rule_type": "CONSTRAINT",
                        "conditions": [
                            {"field": "role", "operator": "eq", "value": "admin"}
                        ],
                        "actions": [
                            {"action_type": "allow", "target": "admin_panel"}
                        ],
                    },
                ),
            )
        )
        ctx = ProcessContext[dict](iteration=1, started_at=datetime.now(UTC))

        # Run through phases
        filtered = await compiler.intake(input_data, ctx)
        results = await compiler.process(filtered, ctx)
        output, should_continue = await compiler.evaluate(results, ctx)

        assert not should_continue
        assert len(output.values) > 0
        assert output.values[0].type == SymbolicValueType.CODE
        assert output.values[0].content["name"] == "access_control"

    def test_get_rules_by_type(self) -> None:
        compiler = RuleCompiler()

        # Manually add some compiled rules for testing
        rule1 = CompiledRule(
            definition_id="def1",
            name="rule1",
            rule_type=RuleType.CONSTRAINT,
            priority=50,
            condition_bytecode="TRUE",
            action_bytecode="NOOP",
        )
        rule2 = CompiledRule(
            definition_id="def2",
            name="rule2",
            rule_type=RuleType.TRIGGER,
            priority=50,
            condition_bytecode="TRUE",
            action_bytecode="NOOP",
        )
        compiler._compiled_rules[rule1.id] = rule1
        compiler._compiled_rules[rule2.id] = rule2

        constraints = compiler.get_rules_by_type(RuleType.CONSTRAINT)
        assert len(constraints) == 1
        assert constraints[0].name == "rule1"

    def test_clear_rules(self) -> None:
        compiler = RuleCompiler()
        compiler._definitions["test"] = RuleDefinition(name="test")
        compiler._compiled_rules["test"] = CompiledRule(
            definition_id="test",
            name="test",
            rule_type=RuleType.CONSTRAINT,
            priority=50,
            condition_bytecode="TRUE",
            action_bytecode="NOOP",
        )

        def_count, comp_count = compiler.clear_rules()
        assert def_count == 1
        assert comp_count == 1
        assert compiler.definition_count == 0
        assert compiler.compiled_count == 0


class TestRuleDefinitionParsing:
    @pytest.mark.asyncio
    async def test_parse_complete_rule(self) -> None:
        compiler = RuleCompiler()
        from autogenrec.core.process import ProcessContext

        input_data = SymbolicInput(
            values=(
                SymbolicValue(
                    type=SymbolicValueType.RULE,
                    content={
                        "name": "complete_rule",
                        "description": "A complete test rule",
                        "rule_type": "TRANSFORMATION",
                        "priority": "HIGH",
                        "enabled": True,
                        "conditions": [
                            {"field": "input.type", "operator": "eq", "value": "request"},
                            {"field": "input.valid", "operator": "eq", "value": True},
                        ],
                        "actions": [
                            {"action_type": "transform", "target": "output", "value": {"status": "processed"}},
                            {"action_type": "emit", "target": "events", "parameters": {"topic": "processed"}},
                        ],
                        "match_all": True,
                        "tags": ["test", "transformation"],
                    },
                ),
            )
        )
        ctx = ProcessContext[dict](iteration=1, started_at=datetime.now(UTC))

        filtered = await compiler.intake(input_data, ctx)
        results = await compiler.process(filtered, ctx)

        assert len(results) == 1
        assert results[0].success

        # Check the compiled rule
        compiled = results[0].compiled_rule
        assert compiled.name == "complete_rule"
        assert compiled.rule_type == RuleType.TRANSFORMATION
        assert compiled.priority == RulePriority.HIGH.value
