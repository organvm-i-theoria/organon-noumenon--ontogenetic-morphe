#!/usr/bin/env python3
"""
Example: Full System Demo

Demonstrates:
- Complete workflow spanning multiple subsystems
- Identity management with masks and audience classification
- Academic research lifecycle
- Monetization and value exchange
- Temporal scheduling and evolution

This shows the complete organon-noumenon architecture in action.
"""

from datetime import datetime, UTC, timedelta
from decimal import Decimal

# Identity subsystems
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

# Academic subsystems
from autogenrec.subsystems.academic.academia_manager import (
    AcademiaManager,
    PublicationType,
)

# Value subsystems
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

# Transformation subsystems
from autogenrec.subsystems.transformation.consumption_manager import (
    ConsumptionManager,
    ResourceType,
)
from autogenrec.subsystems.core_processing.code_generator import (
    CodeGenerator,
    OutputLanguage,
)

# Temporal subsystems
from autogenrec.subsystems.temporal.time_manager import TimeManager, CycleType
from autogenrec.subsystems.temporal.location_resolver import (
    LocationResolver,
    PlaceType,
    SpatialRelation,
)


def main():
    print("=" * 70)
    print("Full System Demo: Research-to-Revenue Pipeline")
    print("=" * 70)
    print()
    print("This demo simulates a complete workflow where:")
    print("  - A researcher creates and publishes academic work")
    print("  - The work is monetized as a product")
    print("  - Users with different access levels consume the product")
    print("  - Revenue is tracked and distributed")
    print()

    # =========================================================================
    # Initialize all subsystems
    # =========================================================================
    print("Initializing Subsystems...")
    print("-" * 40)

    masks = MaskGenerator()
    classifier = AudienceClassifier()
    academia = AcademiaManager()
    exchange = ValueExchangeManager()
    blockchain = BlockchainSimulator()
    monetizer = ProcessMonetizer()
    consumption = ConsumptionManager()
    generator = CodeGenerator()
    time_mgr = TimeManager()
    location = LocationResolver()

    print("  All 10 subsystems initialized")
    print()

    # =========================================================================
    # Phase 1: Set up locations and identities
    # =========================================================================
    print("=" * 70)
    print("Phase 1: Environment Setup")
    print("=" * 70)

    # Create locations
    print("\n[Locations]")
    lab = location.create_place("AI Research Lab", PlaceType.VIRTUAL)
    marketplace = location.create_place("Digital Marketplace", PlaceType.VIRTUAL)
    location.link_places(lab.id, marketplace.id, SpatialRelation.CONNECTED)
    print(f"  Created: {lab.name} <-> {marketplace.name}")

    # Create researcher identity with pseudonymous mask
    print("\n[Identity: Researcher]")
    researcher_mask = masks.generate_mask(
        name="Dr. Anonymous",
        mask_type=MaskType.PSEUDONYMOUS,
        entity_id="researcher_001",
        roles=["researcher", "author"],
    )
    print(f"  Mask: {researcher_mask.name}")
    print(f"  Type: {researcher_mask.mask_type.name}")
    print(f"  Roles: {list(researcher_mask.roles)}")

    # Create audience segments
    print("\n[Audience Segments]")
    basic_segment = classifier.create_segment(
        name="Basic Users",
        segment_type=SegmentType.TIER,
        access_level=AccessLevel.BASIC,
        is_default=True,
    )

    premium_segment = classifier.create_segment(
        name="Premium Users",
        segment_type=SegmentType.TIER,
        access_level=AccessLevel.PREMIUM,
    )
    classifier.add_rule(
        name="premium_subscription",
        segment_id=premium_segment.id,
        attribute="subscription",
        operator=RuleOperator.EQUALS,
        value="premium",
    )

    vip_segment = classifier.create_segment(
        name="VIP",
        segment_type=SegmentType.TIER,
        access_level=AccessLevel.VIP,
    )
    classifier.add_rule(
        name="enterprise_license",
        segment_id=vip_segment.id,
        attribute="license",
        operator=RuleOperator.EQUALS,
        value="vip",
    )

    print(f"  Created: {basic_segment.name} (default)")
    print(f"  Created: {premium_segment.name}")
    print(f"  Created: {vip_segment.name}")

    # =========================================================================
    # Phase 2: Research and Publication
    # =========================================================================
    print()
    print("=" * 70)
    print("Phase 2: Research and Publication")
    print("=" * 70)

    # Create research project
    print("\n[Research Project]")
    project = academia.create_project(
        title="Advanced Pattern Recognition in Symbolic Systems",
        description="Novel approaches to pattern extraction from symbolic data",
        methodology="Mixed methods: theoretical modeling + empirical validation",
        objectives=[
            "Develop pattern recognition framework",
            "Implement prototype system",
            "Validate on benchmark datasets",
        ],
        lead_researcher_id=researcher_mask.entity_id,
    )
    print(f"  Project: {project.title}")
    print(f"  Status: {project.status.name}")

    # Simulate research progress
    academia.start_project(project.id)
    academia.update_project_progress(project.id, 25.0)
    academia.update_project_progress(project.id, 50.0)
    academia.update_project_progress(project.id, 75.0)
    academia.complete_project(project.id)
    print(f"  Progress: 0% -> 25% -> 50% -> 75% -> 100%")
    print(f"  Status: COMPLETED")

    # Create and publish paper
    print("\n[Publication]")
    paper = academia.create_publication(
        title="SymboliQ: A Framework for Symbolic Pattern Recognition",
        publication_type=PublicationType.PAPER,
        abstract="We present SymboliQ, a novel framework for extracting "
                 "and processing symbolic patterns from unstructured data...",
        content="[Full paper content would be here]",
        author_ids=[researcher_mask.entity_id],
        project_id=project.id,
    )

    # Add citations
    academia.add_citation(
        title="Foundations of Symbolic AI",
        authors=["Smith, J.", "Johnson, K."],
        year=2023,
        venue="AI Journal",
    )
    academia.add_citation(
        title="Pattern Recognition in Complex Systems",
        authors=["Williams, R."],
        year=2024,
        venue="ICML",
    )

    published = academia.publish_publication(
        paper.id,
        venue="International Conference on Symbolic Computing",
        doi="10.1234/icsc.2024.001",
    )
    print(f"  Paper: {published.title}")
    print(f"  Venue: {published.venue}")
    print(f"  DOI: {published.doi}")

    # Archive the research
    archive = academia.archive_publication(paper.id)
    project_archive = academia.archive_project(project.id)
    print(f"  Archived: paper and project")

    # =========================================================================
    # Phase 3: Monetization Setup
    # =========================================================================
    print()
    print("=" * 70)
    print("Phase 3: Monetization")
    print("=" * 70)

    # Create accounts
    print("\n[Accounts]")
    platform_account = exchange.create_account(
        "Platform",
        "platform",
        CurrencyType.TOKEN,
        Decimal("100000"),
    )
    researcher_account = exchange.create_account(
        "Researcher",
        researcher_mask.entity_id,
        CurrencyType.TOKEN,
        Decimal("0"),
    )
    print(f"  Platform: {platform_account.balance} TOKENS")
    print(f"  Researcher: {researcher_account.balance} TOKENS")

    # Register monetizable product
    print("\n[Product Registration]")
    api_product = monetizer.register_process(
        name="SymboliQ API Access",
        owner_id=researcher_mask.entity_id,
        product_type=ProductType.SERVICE,
        revenue_model=RevenueModel.USAGE_BASED,
        usage_rate=Decimal("10"),  # 10 tokens per API call
        description="API access to the SymboliQ pattern recognition engine",
    )
    monetizer.activate_process(api_product.id)
    print(f"  Product: {api_product.name}")
    print(f"  Price: {api_product.usage_rate} TOKENS per call")

    # Generate SDK code for the product
    print("\n[SDK Generation]")
    sdk_structure = generator.register_structure(
        name="symboliq_sdk",
        structure_type="module",
        definition={
            "functions": ["recognize_pattern", "extract_symbols", "validate_input"],
            "classes": ["SymboliQClient", "PatternResult"],
        },
        inputs=["api_key", "data"],
        outputs=["PatternResult"],
    )
    sdk_code = generator.generate(sdk_structure.id, OutputLanguage.PYTHON)
    print(f"  Generated SDK code ({len(sdk_code.code.code)} chars)")

    # =========================================================================
    # Phase 4: User Activity and Consumption
    # =========================================================================
    print()
    print("=" * 70)
    print("Phase 4: User Activity")
    print("=" * 70)

    # Register users
    print("\n[User Registration]")
    users = [
        ("user_basic", {"name": "Basic User", "subscription": "free"}),
        ("user_premium", {"name": "Premium User", "subscription": "premium"}),
        ("user_enterprise", {"name": "Enterprise Corp", "license": "vip"}),
    ]

    registered_users = []
    for user_id, attrs in users:
        member = classifier.register_member(external_id=user_id, attributes=attrs)
        classifier.classify_member(member.id)
        access = classifier.get_member_access_level(member.id)
        registered_users.append((user_id, member, access))
        print(f"  {attrs['name']}: {access.name} access")

    # Set up consumption quotas per tier
    print("\n[Consumption Quotas]")
    quotas = [
        ("user_basic", "10"),
        ("user_premium", "100"),
        ("user_enterprise", "1000"),
    ]
    for user_id, limit in quotas:
        consumption.add_quota(
            name=f"{user_id}_api_quota",
            resource_type=ResourceType.API_CALL,
            max_amount=Decimal(limit),
            consumer_id=user_id,
        )
        print(f"  {user_id}: {limit} API calls/period")

    # Simulate API usage
    print("\n[API Usage Simulation]")
    usage_patterns = [
        ("user_basic", 8),
        ("user_premium", 45),
        ("user_enterprise", 200),
    ]

    for user_id, calls in usage_patterns:
        successful = 0
        for i in range(calls):
            event = consumption.create_event(
                consumer_id=user_id,
                resource_type=ResourceType.API_CALL,
                amount=Decimal("1"),
                context="symboliq_api",
            )
            result = consumption.consume(event)
            if result.approved:
                # Record monetization
                monetizer.record_usage(api_product.id, user_id, Decimal("1"))
                successful += 1

        print(f"  {user_id}: {successful}/{calls} calls successful")

    # =========================================================================
    # Phase 5: Revenue Distribution
    # =========================================================================
    print()
    print("=" * 70)
    print("Phase 5: Revenue Distribution")
    print("=" * 70)

    # Check total revenue
    updated_product = monetizer.get_process(api_product.id)
    print(f"\n[Revenue Summary]")
    print(f"  Total API Calls: {updated_product.usage_count}")
    print(f"  Total Revenue: {updated_product.total_revenue} TOKENS")

    # Create and process payout
    print("\n[Payout Processing]")
    payout = monetizer.create_payout(api_product.id)
    print(f"  Gross Amount: {payout.amount} TOKENS")
    print(f"  Platform Fee (10%): {payout.fee} TOKENS")
    print(f"  Net to Researcher: {payout.net_amount} TOKENS")

    # Execute transfer
    transfer_result = exchange.transfer(
        platform_account.id,
        researcher_account.id,
        payout.net_amount,
    )

    # Record on blockchain
    tx = blockchain.submit_transaction(
        sender=platform_account.id,
        recipient=researcher_account.id,
        data={
            "type": "royalty_payout",
            "product_id": api_product.id,
            "amount": str(payout.net_amount),
            "payout_id": payout.id,
        },
    )
    blockchain.mine_block()
    print(f"  Blockchain TX: {tx.transaction_id[:16]}...")

    # Final balances
    researcher_final = exchange.get_account(researcher_account.id)
    print(f"\n[Final Balance]")
    print(f"  Researcher: {researcher_final.balance} TOKENS")

    # =========================================================================
    # Phase 6: System Statistics
    # =========================================================================
    print()
    print("=" * 70)
    print("Phase 6: System Statistics")
    print("=" * 70)

    print("\n[Subsystem Metrics]")

    # Identity
    mask_stats = masks.get_stats()
    classifier_stats = classifier.get_stats()
    print(f"  Identity:")
    print(f"    Masks: {mask_stats.total_masks}")
    print(f"    Segments: {classifier_stats.total_segments}")
    print(f"    Members: {classifier_stats.total_members}")

    # Academic
    academic_stats = academia.get_stats()
    print(f"  Academic:")
    print(f"    Projects: {academic_stats.total_projects} ({academic_stats.completed_projects} completed)")
    print(f"    Publications: {academic_stats.total_publications} ({academic_stats.published_count} published)")
    print(f"    Citations: {academic_stats.total_citations}")
    print(f"    Archives: {academic_stats.total_archives}")

    # Value
    exchange_stats = exchange.get_stats()
    monetizer_stats = monetizer.get_stats()
    blockchain_stats = blockchain.get_stats()
    print(f"  Value Economy:")
    print(f"    Accounts: {exchange_stats.total_accounts}")
    print(f"    Transactions: {exchange_stats.total_transactions}")
    print(f"    Products: {monetizer_stats.active_processes}")
    print(f"    Revenue: {monetizer_stats.total_revenue} TOKENS")
    print(f"    Blockchain Blocks: {blockchain_stats.block_height}")

    # Consumption
    consumption_stats = consumption.get_stats()
    print(f"  Consumption:")
    print(f"    Total Events: {consumption_stats.total_events}")
    print(f"    Approved: {consumption_stats.consumed_count}")
    print(f"    Denied: {consumption_stats.rejected_count}")

    # Code Generation
    generator_stats = generator.get_stats()
    print(f"  Code Generation:")
    print(f"    Structures: {generator_stats.total_structures}")
    print(f"    Generations: {generator_stats.total_generated}")

    print()
    print("=" * 70)
    print("Full System Demo Complete!")
    print("=" * 70)
    print()
    print("Summary:")
    print("  - Created pseudonymous researcher identity")
    print("  - Completed research project with publication")
    print("  - Monetized research as API product")
    print("  - Served users across 3 access tiers")
    print("  - Distributed revenue to researcher")
    print("  - All transactions recorded on blockchain")


if __name__ == "__main__":
    main()
