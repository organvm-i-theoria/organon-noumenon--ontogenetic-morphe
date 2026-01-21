"""Tests for Week 8: Transformation subsystems."""

from datetime import timedelta
from decimal import Decimal

import pytest

from autogenrec.subsystems.transformation.process_converter import (
    ProcessConverter,
    ConversionFormat,
    ConversionStrategy,
    ConversionStatus,
)
from autogenrec.subsystems.transformation.consumption_manager import (
    ConsumptionManager,
    ResourceType,
    RiskLevel,
    ConsumptionStatus,
)
from autogenrec.subsystems.core_processing.code_generator import (
    CodeGenerator,
    OutputLanguage,
    GenerationStrategy,
    ValidationStatus,
)


# ============================================================================
# ProcessConverter Tests
# ============================================================================

class TestProcessConverterBasics:
    """Basic ProcessConverter tests."""

    def test_initialization(self):
        """Test ProcessConverter initializes correctly."""
        pc = ProcessConverter()
        assert pc.name == "process_converter"
        assert pc.process_count == 0
        assert pc.rule_count == 0
        assert pc.conversion_count == 0

    def test_register_process(self):
        """Test registering a process."""
        pc = ProcessConverter()
        process = pc.register_process(
            name="test_process",
            steps=[{"name": "step1"}, {"name": "step2"}],
            inputs=["input1", "input2"],
            outputs=["output1"],
            description="A test process",
        )
        assert process is not None
        assert process.name == "test_process"
        assert len(process.steps) == 2
        assert len(process.inputs) == 2
        assert pc.process_count == 1

    def test_get_process(self):
        """Test retrieving a process by ID."""
        pc = ProcessConverter()
        process = pc.register_process(name="test")
        retrieved = pc.get_process(process.id)
        assert retrieved is not None
        assert retrieved.id == process.id


class TestConversionFormats:
    """Test different conversion formats."""

    def test_convert_to_json(self):
        """Test converting to JSON format."""
        pc = ProcessConverter()
        process = pc.register_process(
            name="json_test",
            inputs=["x", "y"],
            outputs=["result"],
        )
        result = pc.convert(process.id, ConversionFormat.JSON)
        assert result.success is True
        assert result.output is not None
        assert result.output.format == ConversionFormat.JSON
        assert isinstance(result.output.content, dict)
        assert result.output.content["name"] == "json_test"

    def test_convert_to_yaml(self):
        """Test converting to YAML format."""
        pc = ProcessConverter()
        process = pc.register_process(name="yaml_test")
        result = pc.convert(process.id, ConversionFormat.YAML)
        assert result.success is True
        assert result.output.format == ConversionFormat.YAML

    def test_convert_to_schema(self):
        """Test converting to schema format."""
        pc = ProcessConverter()
        process = pc.register_process(
            name="schema_test",
            inputs=["a", "b"],
            outputs=["c"],
        )
        result = pc.convert(process.id, ConversionFormat.SCHEMA)
        assert result.success is True
        assert result.output.format == ConversionFormat.SCHEMA
        assert "$schema" in result.output.content

    def test_convert_to_graph(self):
        """Test converting to graph format."""
        pc = ProcessConverter()
        process = pc.register_process(
            name="graph_test",
            steps=[{"name": "s1"}, {"name": "s2"}],
            inputs=["in"],
            outputs=["out"],
        )
        result = pc.convert(process.id, ConversionFormat.GRAPH)
        assert result.success is True
        assert "nodes" in result.output.content
        assert "edges" in result.output.content

    def test_convert_to_summary(self):
        """Test converting to summary format."""
        pc = ProcessConverter()
        process = pc.register_process(
            name="summary_test",
            steps=[{"name": "s1"}],
        )
        result = pc.convert(process.id, ConversionFormat.SUMMARY)
        assert result.success is True
        assert result.output.content["step_count"] == 1
        assert len(result.output.warnings) > 0  # Summary is lossy

    def test_convert_to_compressed(self):
        """Test converting to compressed format."""
        pc = ProcessConverter()
        process = pc.register_process(name="compress_test")
        result = pc.convert(process.id, ConversionFormat.COMPRESSED)
        assert result.success is True
        assert "checksum" in result.output.content

    def test_convert_to_template(self):
        """Test converting to template format."""
        pc = ProcessConverter()
        process = pc.register_process(
            name="template_test",
            inputs=["x"],
            outputs=["y"],
        )
        result = pc.convert(process.id, ConversionFormat.TEMPLATE)
        assert result.success is True
        assert "placeholders" in result.output.content


class TestConversionStrategies:
    """Test different conversion strategies."""

    def test_direct_strategy(self):
        """Test direct conversion strategy."""
        pc = ProcessConverter()
        process = pc.register_process(name="direct_test")
        result = pc.convert(process.id, ConversionFormat.JSON, ConversionStrategy.DIRECT)
        assert result.success is True
        assert result.fidelity == 1.0

    def test_transform_strategy(self):
        """Test transform conversion strategy."""
        pc = ProcessConverter()
        process = pc.register_process(name="transform_test")
        result = pc.convert(process.id, ConversionFormat.JSON, ConversionStrategy.TRANSFORM)
        assert result.success is True
        assert result.output.strategy == ConversionStrategy.TRANSFORM

    def test_compress_strategy(self):
        """Test compress conversion strategy."""
        pc = ProcessConverter()
        process = pc.register_process(name="compress_test")
        result = pc.convert(process.id, ConversionFormat.JSON, ConversionStrategy.COMPRESS)
        assert result.success is True
        assert result.fidelity < 1.0  # Lossy


class TestConversionRules:
    """Test conversion rules."""

    def test_add_rule(self):
        """Test adding a conversion rule."""
        pc = ProcessConverter()
        rule = pc.add_rule(
            name="test_rule",
            source_pattern="*",
            target_template="{{ name }}",
            applies_to_formats=[ConversionFormat.JSON],
        )
        assert rule is not None
        assert pc.rule_count == 1

    def test_convert_by_name(self):
        """Test converting by process name."""
        pc = ProcessConverter()
        pc.register_process(name="named_process")
        result = pc.convert_by_name("named_process", ConversionFormat.JSON)
        assert result.success is True

    def test_convert_by_name_not_found(self):
        """Test converting with nonexistent name."""
        pc = ProcessConverter()
        result = pc.convert_by_name("nonexistent", ConversionFormat.JSON)
        assert result.success is False
        assert "not found" in result.error


class TestConverterStats:
    """Test converter statistics."""

    def test_get_stats(self):
        """Test getting conversion statistics."""
        pc = ProcessConverter()
        p1 = pc.register_process(name="p1")
        p2 = pc.register_process(name="p2")
        pc.convert(p1.id, ConversionFormat.JSON)
        pc.convert(p2.id, ConversionFormat.YAML)
        
        stats = pc.get_stats()
        assert stats.total_processes == 2
        assert stats.successful_conversions == 2
        assert stats.average_fidelity > 0

    def test_clear(self):
        """Test clearing all data."""
        pc = ProcessConverter()
        pc.register_process(name="p1")
        pc.add_rule("r1", "*", "{{ name }}")
        
        processes, rules, outputs = pc.clear()
        assert processes == 1
        assert rules == 1
        assert pc.process_count == 0


# ============================================================================
# ConsumptionManager Tests
# ============================================================================

class TestConsumptionManagerBasics:
    """Basic ConsumptionManager tests."""

    def test_initialization(self):
        """Test ConsumptionManager initializes correctly."""
        cm = ConsumptionManager()
        assert cm.name == "consumption_manager"
        assert cm.event_count == 0
        assert cm.quota_count == 0
        assert cm.rule_count == 0

    def test_create_and_consume(self):
        """Test creating and consuming an event."""
        cm = ConsumptionManager()
        event = cm.create_event(
            consumer_id="user_123",
            resource_type=ResourceType.TOKEN,
            amount=Decimal("10"),
        )
        result = cm.consume(event)
        assert result.approved is True
        assert cm.event_count == 1

    def test_get_event(self):
        """Test getting an event by ID."""
        cm = ConsumptionManager()
        event = cm.create_event("user_123", ResourceType.TOKEN)
        cm.consume(event)
        retrieved = cm.get_event(event.id)
        assert retrieved is not None
        assert retrieved.id == event.id


class TestResourceTypes:
    """Test different resource types."""

    def test_compute_resource(self):
        """Test compute resource consumption."""
        cm = ConsumptionManager()
        event = cm.create_event("user", ResourceType.COMPUTE, Decimal("100"))
        result = cm.consume(event)
        assert result.approved is True

    def test_storage_resource(self):
        """Test storage resource consumption."""
        cm = ConsumptionManager()
        event = cm.create_event("user", ResourceType.STORAGE, Decimal("1024"))
        result = cm.consume(event)
        assert result.approved is True

    def test_api_call_resource(self):
        """Test API call resource consumption."""
        cm = ConsumptionManager()
        event = cm.create_event("user", ResourceType.API_CALL, Decimal("1"))
        result = cm.consume(event)
        assert result.approved is True


class TestConsumptionQuotas:
    """Test consumption quotas."""

    def test_add_quota(self):
        """Test adding a quota."""
        cm = ConsumptionManager()
        quota = cm.add_quota(
            name="daily_tokens",
            resource_type=ResourceType.TOKEN,
            max_amount=Decimal("1000"),
            consumer_id="user_123",
        )
        assert quota is not None
        assert cm.quota_count == 1

    def test_quota_enforcement(self):
        """Test quota is enforced."""
        cm = ConsumptionManager()
        cm.add_quota(
            name="limit",
            resource_type=ResourceType.TOKEN,
            max_amount=Decimal("10"),
            consumer_id="user_123",
        )
        
        # First consumption should succeed
        event1 = cm.create_event("user_123", ResourceType.TOKEN, Decimal("8"))
        result1 = cm.consume(event1)
        assert result1.approved is True
        
        # Second consumption should fail (quota exceeded)
        event2 = cm.create_event("user_123", ResourceType.TOKEN, Decimal("5"))
        result2 = cm.consume(event2)
        assert result2.approved is False
        assert "exceeded" in result2.rejection_reason.lower()

    def test_global_quota(self):
        """Test global quota (no consumer_id)."""
        cm = ConsumptionManager()
        cm.add_quota(
            name="global_limit",
            resource_type=ResourceType.TOKEN,
            max_amount=Decimal("100"),
        )
        
        event = cm.create_event("any_user", ResourceType.TOKEN, Decimal("50"))
        result = cm.consume(event)
        assert result.approved is True

    def test_check_quota(self):
        """Test checking quota without consuming."""
        cm = ConsumptionManager()
        cm.add_quota(
            name="check_test",
            resource_type=ResourceType.TOKEN,
            max_amount=Decimal("100"),
            consumer_id="user_123",
        )
        
        allowed, remaining = cm.check_quota("user_123", ResourceType.TOKEN, Decimal("30"))
        assert allowed is True
        assert remaining == Decimal("100")


class TestRiskRules:
    """Test risk evaluation rules."""

    def test_add_risk_rule(self):
        """Test adding a risk rule."""
        cm = ConsumptionManager()
        rule = cm.add_risk_rule(
            name="high_amount",
            condition="amount>100",
            risk_level=RiskLevel.HIGH,
            risk_score=0.8,
        )
        assert rule is not None
        assert cm.rule_count == 1

    def test_risk_evaluation(self):
        """Test risk evaluation rejects high risk."""
        cm = ConsumptionManager()
        cm.add_risk_rule(
            name="block_high",
            condition="amount>1000",
            risk_level=RiskLevel.CRITICAL,
        )
        
        event = cm.create_event("user", ResourceType.TOKEN, Decimal("2000"))
        result = cm.consume(event)
        assert result.approved is False
        assert result.risk_level == RiskLevel.CRITICAL

    def test_risk_with_tags(self):
        """Test risk rule with tag condition."""
        cm = ConsumptionManager()
        cm.add_risk_rule(
            name="risky_tag",
            condition="tag:dangerous",
            risk_level=RiskLevel.HIGH,
        )
        
        event = cm.create_event(
            "user",
            ResourceType.TOKEN,
            tags=["dangerous"],
        )
        result = cm.consume(event)
        assert result.approved is False


class TestConsumptionMetrics:
    """Test consumption metrics."""

    def test_get_metrics(self):
        """Test getting usage metrics."""
        cm = ConsumptionManager()
        
        for i in range(5):
            event = cm.create_event("user_123", ResourceType.TOKEN, Decimal("10"))
            cm.consume(event)
        
        metrics = cm.get_metrics(
            "user_123",
            ResourceType.TOKEN,
            period=timedelta(hours=1),
        )
        assert metrics.total_events == 5
        assert metrics.total_consumed == Decimal("50")

    def test_get_consumer_events(self):
        """Test getting events for a consumer."""
        cm = ConsumptionManager()
        
        for i in range(3):
            event = cm.create_event("user_123", ResourceType.TOKEN)
            cm.consume(event)
        
        events = cm.get_consumer_events("user_123")
        assert len(events) == 3


class TestConsumptionStats:
    """Test consumption statistics."""

    def test_get_stats(self):
        """Test getting consumption statistics."""
        cm = ConsumptionManager()
        cm.add_quota("q1", ResourceType.TOKEN, Decimal("100"))
        cm.add_risk_rule("r1", "always", RiskLevel.LOW)
        
        event = cm.create_event("user", ResourceType.TOKEN, Decimal("10"))
        cm.consume(event)
        
        stats = cm.get_stats()
        assert stats.total_events == 1
        assert stats.total_quotas == 1
        assert stats.total_rules == 1

    def test_clear(self):
        """Test clearing all data."""
        cm = ConsumptionManager()
        cm.add_quota("q1", ResourceType.TOKEN, Decimal("100"))
        cm.add_risk_rule("r1", "always", RiskLevel.LOW)
        event = cm.create_event("user", ResourceType.TOKEN)
        cm.consume(event)
        
        events, quotas, rules = cm.clear()
        assert events == 1
        assert quotas == 1
        assert rules == 1
        assert cm.event_count == 0


# ============================================================================
# CodeGenerator Tests
# ============================================================================

class TestCodeGeneratorBasics:
    """Basic CodeGenerator tests."""

    def test_initialization(self):
        """Test CodeGenerator initializes correctly."""
        cg = CodeGenerator()
        assert cg.name == "code_generator"
        assert cg.structure_count == 0
        assert cg.template_count == 0
        assert cg.generated_count == 0

    def test_register_structure(self):
        """Test registering a structure."""
        cg = CodeGenerator()
        structure = cg.register_structure(
            name="test_function",
            structure_type="function",
            inputs=["x", "y"],
            outputs=["result"],
            definition={"operation": "add"},
        )
        assert structure is not None
        assert structure.name == "test_function"
        assert cg.structure_count == 1

    def test_get_structure(self):
        """Test getting a structure by ID."""
        cg = CodeGenerator()
        structure = cg.register_structure(name="test")
        retrieved = cg.get_structure(structure.id)
        assert retrieved is not None
        assert retrieved.id == structure.id


class TestOutputLanguages:
    """Test different output languages."""

    def test_generate_python(self):
        """Test generating Python code."""
        cg = CodeGenerator()
        structure = cg.register_structure(
            name="add_numbers",
            inputs=["a", "b"],
            outputs=["sum"],
        )
        result = cg.generate(structure.id, OutputLanguage.PYTHON)
        assert result.success is True
        assert "def add_numbers" in result.code.code
        assert result.code.language == OutputLanguage.PYTHON

    def test_generate_javascript(self):
        """Test generating JavaScript code."""
        cg = CodeGenerator()
        structure = cg.register_structure(
            name="multiply",
            inputs=["x", "y"],
            outputs=["product"],
        )
        result = cg.generate(structure.id, OutputLanguage.JAVASCRIPT)
        assert result.success is True
        assert "function multiply" in result.code.code
        assert result.code.language == OutputLanguage.JAVASCRIPT

    def test_generate_json(self):
        """Test generating JSON code."""
        cg = CodeGenerator()
        structure = cg.register_structure(name="json_test")
        result = cg.generate(structure.id, OutputLanguage.JSON)
        assert result.success is True
        # Should be valid JSON
        import json
        parsed = json.loads(result.code.code)
        assert parsed["name"] == "json_test"

    def test_generate_yaml(self):
        """Test generating YAML code."""
        cg = CodeGenerator()
        structure = cg.register_structure(name="yaml_test")
        result = cg.generate(structure.id, OutputLanguage.YAML)
        assert result.success is True
        assert "name: yaml_test" in result.code.code

    def test_generate_sql(self):
        """Test generating SQL code."""
        cg = CodeGenerator()
        structure = cg.register_structure(
            name="get_user",
            inputs=["user_id"],
            outputs=["name", "email"],
        )
        result = cg.generate(structure.id, OutputLanguage.SQL)
        assert result.success is True
        assert "CREATE OR REPLACE FUNCTION" in result.code.code

    def test_generate_bash(self):
        """Test generating Bash code."""
        cg = CodeGenerator()
        structure = cg.register_structure(
            name="deploy",
            inputs=["env"],
        )
        result = cg.generate(structure.id, OutputLanguage.BASH)
        assert result.success is True
        assert "#!/bin/bash" in result.code.code

    def test_generate_pseudocode(self):
        """Test generating pseudocode."""
        cg = CodeGenerator()
        structure = cg.register_structure(name="algorithm")
        result = cg.generate(structure.id, OutputLanguage.PSEUDOCODE)
        assert result.success is True
        assert "PROCEDURE algorithm" in result.code.code


class TestGenerationStrategies:
    """Test different generation strategies."""

    def test_template_strategy(self):
        """Test template-based generation."""
        cg = CodeGenerator()
        structure = cg.register_structure(name="template_test")
        result = cg.generate(structure.id, OutputLanguage.PYTHON, GenerationStrategy.TEMPLATE)
        assert result.success is True
        assert result.code.strategy == GenerationStrategy.TEMPLATE

    def test_transform_strategy(self):
        """Test transform-based generation."""
        cg = CodeGenerator()
        structure = cg.register_structure(name="transform_test")
        result = cg.generate(structure.id, OutputLanguage.PYTHON, GenerationStrategy.TRANSFORM)
        assert result.success is True
        assert result.code.strategy == GenerationStrategy.TRANSFORM

    def test_compose_strategy(self):
        """Test compose-based generation."""
        cg = CodeGenerator()
        structure = cg.register_structure(name="compose_test")
        result = cg.generate(structure.id, OutputLanguage.PYTHON, GenerationStrategy.COMPOSE)
        assert result.success is True
        assert result.code.strategy == GenerationStrategy.COMPOSE


class TestCodeTemplates:
    """Test code templates."""

    def test_add_template(self):
        """Test adding a code template."""
        cg = CodeGenerator()
        template = cg.add_template(
            name="python_function",
            language=OutputLanguage.PYTHON,
            template="def {{ name }}({{ inputs }}):\n    pass",
            placeholders=["name", "inputs"],
            applies_to=["function"],
        )
        assert template is not None
        assert cg.template_count == 1

    def test_template_application(self):
        """Test template is applied."""
        cg = CodeGenerator()
        cg.add_template(
            name="custom_template",
            language=OutputLanguage.PYTHON,
            template="# Custom: {{ name }}\ndef {{ name }}():\n    print('custom')",
            applies_to=["function"],
            priority=100,  # High priority
        )
        
        structure = cg.register_structure(
            name="my_func",
            structure_type="function",
        )
        result = cg.generate(structure.id, OutputLanguage.PYTHON)
        assert result.success is True
        assert "# Custom: my_func" in result.code.code


class TestCodeValidation:
    """Test code validation."""

    def test_validate_python(self):
        """Test validating Python code."""
        cg = CodeGenerator()
        structure = cg.register_structure(name="valid_python")
        result = cg.generate(structure.id, OutputLanguage.PYTHON)
        
        validation = cg.validate(result.code_id)
        assert validation.valid is True
        assert len(validation.errors) == 0

    def test_validate_json(self):
        """Test validating JSON code."""
        cg = CodeGenerator()
        structure = cg.register_structure(name="valid_json")
        result = cg.generate(structure.id, OutputLanguage.JSON)
        
        validation = cg.validate(result.code_id)
        assert validation.valid is True


class TestExecutionPlans:
    """Test execution plan generation."""

    def test_create_execution_plan(self):
        """Test creating an execution plan."""
        cg = CodeGenerator()
        structure = cg.register_structure(
            name="planned_func",
            inputs=["x"],
            outputs=["y"],
        )
        gen_result = cg.generate(structure.id, OutputLanguage.PYTHON)
        
        plan = cg.create_execution_plan(gen_result.code_id)
        assert plan is not None
        assert len(plan.steps) == 4
        assert plan.required_inputs == ("x",)
        assert plan.expected_outputs == ("y",)


class TestGeneratorConvenience:
    """Test convenience methods."""

    def test_generate_from_definition(self):
        """Test generating from definition in one step."""
        cg = CodeGenerator()
        result = cg.generate_from_definition(
            name="quick_func",
            definition={"type": "math"},
            language=OutputLanguage.PYTHON,
            inputs=["a", "b"],
            outputs=["c"],
        )
        assert result.success is True
        assert cg.structure_count == 1

    def test_get_code(self):
        """Test getting generated code."""
        cg = CodeGenerator()
        structure = cg.register_structure(name="test")
        result = cg.generate(structure.id, OutputLanguage.PYTHON)
        
        code = cg.get_code(result.code_id)
        assert code is not None
        assert code.id == result.code_id


class TestGeneratorStats:
    """Test generator statistics."""

    def test_get_stats(self):
        """Test getting generation statistics."""
        cg = CodeGenerator()
        s1 = cg.register_structure(name="s1")
        s2 = cg.register_structure(name="s2")
        cg.generate(s1.id, OutputLanguage.PYTHON)
        cg.generate(s2.id, OutputLanguage.JAVASCRIPT)
        
        stats = cg.get_stats()
        assert stats.total_structures == 2
        assert stats.total_generated == 2
        assert stats.by_language["PYTHON"] == 1
        assert stats.by_language["JAVASCRIPT"] == 1

    def test_clear(self):
        """Test clearing all data."""
        cg = CodeGenerator()
        cg.register_structure(name="s1")
        cg.add_template("t1", OutputLanguage.PYTHON, "# template")
        
        structures, templates, generated = cg.clear()
        assert structures == 1
        assert templates == 1
        assert cg.structure_count == 0
