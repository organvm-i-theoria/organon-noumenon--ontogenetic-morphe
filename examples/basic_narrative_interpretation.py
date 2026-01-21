#!/usr/bin/env python3
"""
Example: Identity and Classification

Demonstrates:
- MaskGenerator creating identity masks
- AudienceClassifier segmenting and classifying users
- Access level management

This shows the identity subsystem:
ENTITY -> MASK -> SEGMENT -> ACCESS_LEVEL
"""

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


def main():
    print("=" * 60)
    print("Identity and Classification Example")
    print("=" * 60)
    print()

    # Initialize subsystems
    masks = MaskGenerator()
    classifier = AudienceClassifier()

    # =========================================================================
    # Step 1: Create identity masks
    # =========================================================================
    print("Step 1: Creating Identity Masks")
    print("-" * 40)

    # Create different types of masks
    admin_mask = masks.generate_mask(
        name="admin_identity",
        mask_type=MaskType.ROLE,
        entity_id="user_001",
        roles=["admin", "moderator"],
        attributes=["verified", "trusted"],
    )
    print(f"  Created: {admin_mask.name}")
    print(f"    Type: {admin_mask.mask_type.name}")
    print(f"    Roles: {list(admin_mask.roles)}")

    premium_mask = masks.generate_mask(
        name="premium_user",
        mask_type=MaskType.PSEUDONYMOUS,
        entity_id="user_002",
        roles=["subscriber"],
    )
    print(f"  Created: {premium_mask.name}")
    print(f"    Type: {premium_mask.mask_type.name}")

    guest_mask = masks.generate_anonymous_mask()
    print(f"  Created: anonymous mask")
    print(f"    Type: {guest_mask.mask_type.name}")
    print()

    # =========================================================================
    # Step 2: Create audience segments
    # =========================================================================
    print("Step 2: Creating Audience Segments")
    print("-" * 40)

    # Create tiered segments
    admin_segment = classifier.create_segment(
        name="administrators",
        segment_type=SegmentType.TIER,
        access_level=AccessLevel.ADMIN,
        description="System administrators",
    )
    classifier.add_rule(
        name="is_admin",
        segment_id=admin_segment.id,
        attribute="role",
        operator=RuleOperator.EQUALS,
        value="admin",
    )
    print(f"  Created: {admin_segment.name}")
    print(f"    Access: {admin_segment.access_level.name}")

    premium_segment = classifier.create_segment(
        name="premium_users",
        segment_type=SegmentType.TIER,
        access_level=AccessLevel.PREMIUM,
        description="Premium subscribers",
    )
    classifier.add_rule(
        name="is_premium",
        segment_id=premium_segment.id,
        attribute="subscription",
        operator=RuleOperator.IN,
        value=["premium", "enterprise"],
    )
    print(f"  Created: {premium_segment.name}")
    print(f"    Access: {premium_segment.access_level.name}")

    basic_segment = classifier.create_segment(
        name="basic_users",
        segment_type=SegmentType.TIER,
        access_level=AccessLevel.BASIC,
        is_default=True,
        description="Free tier users",
    )
    print(f"  Created: {basic_segment.name} (default)")
    print(f"    Access: {basic_segment.access_level.name}")
    print()

    # =========================================================================
    # Step 3: Register and classify members
    # =========================================================================
    print("Step 3: Registering and Classifying Members")
    print("-" * 40)

    members = [
        ("admin_user", {"role": "admin", "subscription": "enterprise"}),
        ("premium_user", {"role": "user", "subscription": "premium"}),
        ("basic_user", {"role": "user", "subscription": "free"}),
        ("guest_user", {}),
    ]

    for external_id, attrs in members:
        member = classifier.register_member(
            external_id=external_id,
            attributes=attrs,
        )
        result = classifier.classify_member(member.id)
        access = classifier.get_member_access_level(member.id)

        print(f"  {external_id}:")
        print(f"    Segments: {[s[:8] + '...' for s in result.segments]}")
        print(f"    Access Level: {access.name}")
    print()

    # =========================================================================
    # Step 4: Mask composition
    # =========================================================================
    print("Step 4: Mask Composition")
    print("-" * 40)

    # Compose masks to create layered identity
    layer1 = masks.generate_mask(
        name="base_identity",
        mask_type=MaskType.ROLE,
        roles=["user"],
    )

    layer2 = masks.generate_mask(
        name="enhanced_identity",
        mask_type=MaskType.ROLE,
        roles=["premium"],
        attributes=["verified"],
    )

    composed = masks.compose_mask("composite_identity", [layer1.id, layer2.id])
    print(f"  Composed mask: {composed.name}")
    print(f"    Combined roles: {list(composed.roles)}")
    print(f"    Combined attributes: {list(composed.attributes)}")
    print()

    # =========================================================================
    # Step 5: Statistics
    # =========================================================================
    print("Step 5: Statistics")
    print("-" * 40)

    mask_stats = masks.get_stats()
    classifier_stats = classifier.get_stats()

    print(f"  Mask Generator:")
    print(f"    Total masks: {mask_stats.total_masks}")
    print(f"    Active: {mask_stats.active_masks}")
    print(f"    By type: {mask_stats.masks_by_type}")

    print(f"  Audience Classifier:")
    print(f"    Segments: {classifier_stats.total_segments}")
    print(f"    Members: {classifier_stats.total_members}")
    print(f"    Memberships: {classifier_stats.total_memberships}")

    print()
    print("=" * 60)
    print("Example Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
