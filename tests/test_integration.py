"""Integration tests for cross-subsystem interactions.

These tests verify that subsystems can work together to accomplish
complex workflows spanning multiple components.
"""

from datetime import datetime, timedelta, UTC
from decimal import Decimal

import pytest

# Import all subsystems we'll test together
from autogenrec.subsystems.value.value_exchange_manager import (
    ValueExchangeManager,
    CurrencyType,
)
from autogenrec.subsystems.value.blockchain_simulator import BlockchainSimulator
from autogenrec.subsystems.value.process_monetizer import (
    ProcessMonetizer,
    ProductType,
    RevenueModel,
)
from autogenrec.subsystems.identity.mask_generator import (
    MaskGenerator,
    MaskType,
)
from autogenrec.subsystems.identity.audience_classifier import (
    AudienceClassifier,
    SegmentType,
    AccessLevel,
    RuleOperator,
)
from autogenrec.subsystems.transformation.process_converter import (
    ProcessConverter,
    ConversionFormat,
)
from autogenrec.subsystems.transformation.consumption_manager import (
    ConsumptionManager,
    ResourceType,
    RiskLevel,
)
from autogenrec.subsystems.core_processing.code_generator import (
    CodeGenerator,
    OutputLanguage,
)
from autogenrec.subsystems.academic.academia_manager import (
    AcademiaManager,
    PublicationType,
)
from autogenrec.subsystems.temporal.time_manager import TimeManager, CycleType
from autogenrec.subsystems.temporal.evolution_scheduler import EvolutionScheduler
from autogenrec.subsystems.temporal.location_resolver import (
    LocationResolver,
    PlaceType,
    SpatialRelation,
    ResolutionStatus,
)


# ============================================================================
# Integration Test: Value Flow Pipeline
# ============================================================================

class TestValueFlowIntegration:
    """Test value flow: Exchange -> Blockchain -> Monetization."""

    def test_complete_value_transfer_flow(self):
        """Test a complete value transfer with blockchain recording."""
        # Setup subsystems
        exchange = ValueExchangeManager()
        blockchain = BlockchainSimulator()

        # Create accounts
        alice = exchange.create_account("Alice", "user_alice", CurrencyType.TOKEN, Decimal("100"))
        bob = exchange.create_account("Bob", "user_bob", CurrencyType.TOKEN, Decimal("50"))

        # Perform transfer
        result = exchange.transfer(alice.id, bob.id, Decimal("25"))
        assert result.success is True

        # Record on blockchain
        tx_result = blockchain.submit_transaction(
            sender=alice.id,
            recipient=bob.id,
            data={
                "type": "transfer",
                "amount": "25",
                "exchange_tx_id": result.transaction_id,
            },
        )
        assert tx_result.valid is True

        # Mine block
        block_result = blockchain.mine_block()
        assert block_result.success is True
        assert block_result.transaction_count == 1

        # Verify final balances
        alice_final = exchange.get_account(alice.id)
        bob_final = exchange.get_account(bob.id)
        assert alice_final.balance == Decimal("75")
        assert bob_final.balance == Decimal("75")

    def test_monetization_with_exchange(self):
        """Test process monetization generating exchange transactions."""
        # Setup subsystems
        exchange = ValueExchangeManager()
        monetizer = ProcessMonetizer()

        # Create accounts
        platform = exchange.create_account("Platform", "platform", CurrencyType.TOKEN, Decimal("1000"))
        creator = exchange.create_account("Creator", "creator", CurrencyType.TOKEN, Decimal("0"))

        # Register and monetize a process
        process = monetizer.register_process(
            name="Premium API",
            owner_id="creator",
            product_type=ProductType.SERVICE,
            revenue_model=RevenueModel.USAGE_BASED,
            usage_rate=Decimal("10"),  # For USAGE_BASED, use usage_rate not base_price
        )
        monetizer.activate_process(process.id)

        # Record usage
        monetizer.record_usage(process.id, "user_1", Decimal("5"))
        monetizer.record_usage(process.id, "user_2", Decimal("3"))

        # Get process to check revenue
        updated_process = monetizer.get_process(process.id)
        assert updated_process.total_revenue == Decimal("80")  # (5+3) * 10

        # Create payout
        payout = monetizer.create_payout(process.id)
        assert payout is not None

        # Transfer payout to creator (simulating the actual payment)
        result = exchange.transfer(platform.id, creator.id, payout.net_amount)
        assert result.success is True

        creator_final = exchange.get_account(creator.id)
        assert creator_final.balance == payout.net_amount


# ============================================================================
# Integration Test: Identity and Classification Pipeline
# ============================================================================

class TestIdentityClassificationIntegration:
    """Test identity: Mask -> Classification -> Access."""

    def test_masked_user_classification(self):
        """Test classifying users with masked identities."""
        # Setup subsystems
        masks = MaskGenerator()
        classifier = AudienceClassifier()

        # Create masks for users
        user_mask = masks.generate_mask(
            name="premium_user_mask",
            mask_type=MaskType.ROLE,
            roles=["premium"],
            attributes=["verified"],
        )

        anon_mask = masks.generate_anonymous_mask()

        # Create segments
        premium_segment = classifier.create_segment(
            name="premium",
            segment_type=SegmentType.TIER,
            access_level=AccessLevel.PREMIUM,
        )
        classifier.add_rule(
            name="premium_check",
            segment_id=premium_segment.id,
            attribute="tier",
            operator=RuleOperator.EQUALS,
            value="premium",
        )

        basic_segment = classifier.create_segment(
            name="basic",
            is_default=True,
            access_level=AccessLevel.BASIC,
        )

        # Register members with mask-derived attributes
        member1 = classifier.register_member(
            external_id=user_mask.id,  # Link to mask
            attributes={"tier": "premium", "masked": True},
        )

        member2 = classifier.register_member(
            external_id=anon_mask.id,
            attributes={"tier": "basic", "masked": True},
        )

        # Classify
        result1 = classifier.classify_member(member1.id)
        result2 = classifier.classify_member(member2.id)

        assert premium_segment.id in result1.segments
        assert basic_segment.id in result2.segments

        # Check access levels
        level1 = classifier.get_member_access_level(member1.id)
        level2 = classifier.get_member_access_level(member2.id)

        assert level1 == AccessLevel.PREMIUM
        assert level2 == AccessLevel.BASIC


# ============================================================================
# Integration Test: Process Transformation Pipeline
# ============================================================================

class TestProcessTransformationIntegration:
    """Test process: Convert -> Generate Code -> Validate."""

    def test_process_to_code_pipeline(self):
        """Test converting a process definition to executable code."""
        # Setup subsystems
        converter = ProcessConverter()
        generator = CodeGenerator()

        # Register a process
        process = converter.register_process(
            name="data_pipeline",
            steps=[
                {"name": "load", "action": "read_file"},
                {"name": "transform", "action": "apply_rules"},
                {"name": "save", "action": "write_output"},
            ],
            inputs=["input_file", "rules"],
            outputs=["output_file"],
        )

        # Convert to schema
        schema_result = converter.convert(process.id, ConversionFormat.SCHEMA)
        assert schema_result.success is True

        # Convert to JSON for code generation
        json_result = converter.convert(process.id, ConversionFormat.JSON)
        assert json_result.success is True

        # Register structure in code generator
        structure = generator.register_structure(
            name="data_pipeline",
            structure_type="workflow",
            definition=json_result.output.content,
            inputs=["input_file", "rules"],
            outputs=["output_file"],
        )

        # Generate Python code
        code_result = generator.generate(structure.id, OutputLanguage.PYTHON)
        assert code_result.success is True
        assert "def data_pipeline" in code_result.code.code

        # Validate the code
        validation = generator.validate(code_result.code_id)
        assert validation.valid is True

    def test_code_generation_with_consumption_tracking(self):
        """Test code generation with resource consumption tracking."""
        # Setup subsystems
        generator = CodeGenerator()
        consumption = ConsumptionManager()

        # Add quota for code generation
        consumption.add_quota(
            name="code_generation_limit",
            resource_type=ResourceType.COMPUTE,
            max_amount=Decimal("100"),
            consumer_id="system",
        )

        # Track consumption for generation
        event = consumption.create_event(
            consumer_id="system",
            resource_type=ResourceType.COMPUTE,
            amount=Decimal("10"),
            context="code_generation",
        )
        result = consumption.consume(event)
        assert result.approved is True

        # Generate code
        structure = generator.register_structure(
            name="tracked_function",
            inputs=["x"],
            outputs=["y"],
        )
        code_result = generator.generate(structure.id, OutputLanguage.PYTHON)
        assert code_result.success is True

        # Check remaining quota
        allowed, remaining = consumption.check_quota(
            "system",
            ResourceType.COMPUTE,
            Decimal("50"),
        )
        assert allowed is True
        assert remaining == Decimal("90")


# ============================================================================
# Integration Test: Academic Research Pipeline
# ============================================================================

class TestAcademicResearchIntegration:
    """Test academic: Research -> Publications -> Archives."""

    def test_complete_research_lifecycle(self):
        """Test a complete research project lifecycle."""
        # Setup subsystem
        academia = AcademiaManager()

        # Create a research project
        project = academia.create_project(
            title="Symbolic Processing Study",
            description="Research on symbolic data processing",
            methodology="Mixed methods",
            objectives=[
                "Define symbolic processing framework",
                "Implement prototype",
                "Evaluate performance",
            ],
            lead_researcher_id="researcher_1",
        )

        # Start the project
        academia.start_project(project.id)

        # Update progress
        academia.update_project_progress(project.id, 25.0)
        academia.update_project_progress(project.id, 50.0)
        academia.update_project_progress(project.id, 75.0)

        # Create publication from research
        pub = academia.create_publication(
            title="Symbolic Processing: A Framework",
            publication_type=PublicationType.PAPER,
            abstract="This paper presents a framework for symbolic processing...",
            content="Introduction\n\nSymbolic processing is...",
            author_ids=["researcher_1"],
            project_id=project.id,
        )

        # Add citations
        citation1 = academia.add_citation(
            title="Prior Work on Symbols",
            authors=["Smith, J.", "Doe, A."],
            year=2023,
        )
        citation2 = academia.add_citation(
            title="Data Processing Fundamentals",
            authors=["Johnson, B."],
            year=2022,
        )

        # Complete project
        academia.complete_project(project.id)

        # Publish the paper
        published = academia.publish_publication(
            pub.id,
            venue="Journal of Symbolic Computing",
            doi="10.1234/jsc.2024.001",
        )
        assert published.is_published is True

        # Archive the publication
        archive = academia.archive_publication(pub.id)
        assert archive is not None

        # Archive the project
        project_archive = academia.archive_project(project.id)
        assert project_archive is not None

        # Verify stats
        stats = academia.get_stats()
        assert stats.completed_projects == 1
        assert stats.published_count == 1
        assert stats.total_citations == 2
        assert stats.total_archives == 2


# ============================================================================
# Integration Test: Temporal and Spatial Management
# ============================================================================

class TestTemporalSpatialIntegration:
    """Test temporal and spatial subsystems together."""

    def test_scheduled_evolution_at_location(self):
        """Test evolution scheduling with location context."""
        # Setup subsystems
        time_mgr = TimeManager()
        evolution = EvolutionScheduler()
        location = LocationResolver()

        # Create locations
        lab = location.create_place("Research Lab", PlaceType.PHYSICAL)
        server = location.create_place("Cloud Server", PlaceType.VIRTUAL)
        location.link_places(lab.id, server.id, SpatialRelation.CONNECTED)

        # Create evolution pattern
        pattern = evolution.create_pattern(
            name="model_training",
            content={"type": "ml_model", "version": "1.0"},
            description="ML model training evolution",
        )

        # Schedule an event in time manager
        training_event = time_mgr.schedule_event(
            name="training_cycle",
            scheduled_at=datetime.now(UTC) + timedelta(hours=1),
            cycle_type=CycleType.ONE_TIME,
        )
        assert training_event is not None

        # Assign entity to location
        assignment = location.assign_entity(
            entity_id=pattern.id,
            place_id=server.id,
            entity_type="evolution_pattern",
        )
        assert assignment is not None

        # Resolve location by name
        resolved = location.resolve("Cloud Server")
        assert resolved.status == ResolutionStatus.RESOLVED
        assert resolved.place is not None
        assert resolved.place.id == server.id

        # Mutate the pattern (simulate evolution)
        mutated, mutation = evolution.mutate_pattern(pattern.id)
        assert mutated is not None
        assert mutation is not None
        assert mutated.generation == 1


# ============================================================================
# Integration Test: Full System Flow
# ============================================================================

class TestFullSystemIntegration:
    """Test a complex flow spanning many subsystems."""

    def test_research_monetization_flow(self):
        """Test research output being monetized and tracked."""
        # Setup subsystems
        academia = AcademiaManager()
        monetizer = ProcessMonetizer()
        exchange = ValueExchangeManager()
        consumption = ConsumptionManager()
        masks = MaskGenerator()

        # 1. Create research project
        project = academia.create_project(
            title="AI Algorithm Research",
            lead_researcher_id="researcher_1",
        )
        academia.start_project(project.id)
        academia.complete_project(project.id)

        # 2. Create publication
        pub = academia.create_publication(
            title="Novel AI Algorithm",
            project_id=project.id,
            author_ids=["researcher_1"],
        )
        academia.publish_publication(pub.id, venue="AI Conference")

        # 3. Monetize the algorithm
        algorithm = monetizer.register_process(
            name="Novel AI Algorithm Implementation",
            owner_id="researcher_1",
            product_type=ProductType.LICENSE,
            revenue_model=RevenueModel.FIXED_PRICE,
            base_price=Decimal("1000"),
        )
        monetizer.activate_process(algorithm.id)

        # 4. Create masked identity for researcher
        researcher_mask = masks.generate_mask(
            name="researcher_public_identity",
            mask_type=MaskType.PSEUDONYMOUS,
            entity_id="researcher_1",
        )

        # 5. Set up exchange accounts
        platform = exchange.create_account(
            "Platform",
            "platform",
            CurrencyType.TOKEN,
            Decimal("10000"),
        )
        researcher = exchange.create_account(
            "Researcher",
            "researcher_1",
            CurrencyType.TOKEN,
            Decimal("0"),
        )

        # 6. Track consumption of the algorithm
        consumption.add_quota(
            name="api_calls",
            resource_type=ResourceType.API_CALL,
            max_amount=Decimal("1000"),
        )

        # 7. Simulate usage and payment
        monetizer.record_usage(algorithm.id, "company_a", Decimal("1"))
        payout = monetizer.create_payout(algorithm.id)

        # 8. Execute payment
        payment_result = exchange.transfer(
            platform.id,
            researcher.id,
            payout.net_amount,
        )
        assert payment_result.success is True

        # Verify final state
        researcher_balance = exchange.get_account(researcher.id)
        assert researcher_balance.balance > Decimal("0")

        updated_algorithm = monetizer.get_process(algorithm.id)
        assert updated_algorithm.total_revenue == Decimal("1000")

    def test_audience_segmented_content_delivery(self):
        """Test content delivery based on audience classification."""
        # Setup subsystems
        classifier = AudienceClassifier()
        masks = MaskGenerator()
        consumption = ConsumptionManager()

        # 1. Create audience segments
        free_tier = classifier.create_segment(
            name="free",
            access_level=AccessLevel.BASIC,
            is_default=True,
        )

        premium_tier = classifier.create_segment(
            name="premium",
            access_level=AccessLevel.PREMIUM,
        )
        classifier.add_rule(
            "premium_check",
            premium_tier.id,
            "subscription",
            RuleOperator.EQUALS,
            "premium",
        )

        # 2. Register members
        free_user = classifier.register_member(
            external_id="user_free",
            attributes={"subscription": "free"},
        )
        premium_user = classifier.register_member(
            external_id="user_premium",
            attributes={"subscription": "premium"},
        )

        # 3. Classify users
        classifier.classify_member(free_user.id)
        classifier.classify_member(premium_user.id)

        # 4. Create masks based on access
        free_mask = masks.generate_mask(
            name="free_access",
            mask_type=MaskType.ROLE,
            roles=["basic_viewer"],
        )
        premium_mask = masks.generate_mask(
            name="premium_access",
            mask_type=MaskType.ROLE,
            roles=["full_access", "download"],
        )

        # 5. Set consumption quotas by tier
        consumption.add_quota(
            name="free_api_limit",
            resource_type=ResourceType.API_CALL,
            max_amount=Decimal("10"),
            consumer_id="user_free",
        )
        consumption.add_quota(
            name="premium_api_limit",
            resource_type=ResourceType.API_CALL,
            max_amount=Decimal("1000"),
            consumer_id="user_premium",
        )

        # 6. Test access levels
        free_level = classifier.get_member_access_level(free_user.id)
        premium_level = classifier.get_member_access_level(premium_user.id)

        assert free_level == AccessLevel.BASIC
        assert premium_level == AccessLevel.PREMIUM

        # 7. Test consumption limits
        # Free user should hit quota faster
        for i in range(15):
            event = consumption.create_event(
                "user_free",
                ResourceType.API_CALL,
                Decimal("1"),
            )
            result = consumption.consume(event)
            if i < 10:
                assert result.approved is True
            else:
                assert result.approved is False

        # Premium user has much higher quota
        for i in range(100):
            event = consumption.create_event(
                "user_premium",
                ResourceType.API_CALL,
                Decimal("1"),
            )
            result = consumption.consume(event)
            assert result.approved is True
