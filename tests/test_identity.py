"""Tests for Week 7: Identity and Classification subsystems."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from autogenrec.subsystems.identity.mask_generator import (
    MaskGenerator,
    MaskType,
    MaskState,
    Mask,
    MaskLayer,
)
from autogenrec.subsystems.identity.audience_classifier import (
    AudienceClassifier,
    SegmentType,
    AccessLevel,
    RuleOperator,
)


# ============================================================================
# MaskGenerator Tests
# ============================================================================

class TestMaskGeneratorBasics:
    """Basic MaskGenerator tests."""

    def test_initialization(self):
        """Test MaskGenerator initializes correctly."""
        mg = MaskGenerator()
        assert mg.name == "mask_generator"
        assert mg.mask_count == 0
        assert mg.assignment_count == 0

    def test_generate_basic_mask(self):
        """Test generating a basic mask."""
        mg = MaskGenerator()
        mask = mg.generate_mask(
            name="test_mask",
            mask_type=MaskType.IDENTITY,
            attributes=["name", "email"],
            roles=["user"],
        )
        assert mask is not None
        assert mask.name == "test_mask"
        assert mask.mask_type == MaskType.IDENTITY
        assert "name" in mask.attributes
        assert "email" in mask.attributes
        assert "user" in mask.roles
        assert mg.mask_count == 1

    def test_get_mask(self):
        """Test retrieving a mask by ID."""
        mg = MaskGenerator()
        mask = mg.generate_mask(name="test_mask")
        retrieved = mg.get_mask(mask.id)
        assert retrieved is not None
        assert retrieved.id == mask.id
        assert retrieved.name == mask.name

    def test_get_nonexistent_mask(self):
        """Test getting a nonexistent mask returns None."""
        mg = MaskGenerator()
        assert mg.get_mask("nonexistent") is None


class TestMaskTypes:
    """Test different mask types."""

    def test_temporal_mask(self):
        """Test generating a temporal mask."""
        mg = MaskGenerator()
        mask = mg.generate_temporal_mask(
            name="temp_mask",
            duration=timedelta(hours=1),
        )
        assert mask.mask_type == MaskType.TEMPORAL
        assert mask.valid_from is not None
        assert mask.valid_until is not None
        assert mask.valid_until > mask.valid_from

    def test_temporal_mask_default_duration(self):
        """Test temporal mask default 24h duration."""
        mg = MaskGenerator()
        mask = mg.generate_temporal_mask(name="temp_mask")
        assert mask.mask_type == MaskType.TEMPORAL
        # Should have approximately 24h validity
        diff = mask.valid_until - mask.valid_from
        assert diff >= timedelta(hours=23)

    def test_anonymous_mask(self):
        """Test generating an anonymous mask."""
        mg = MaskGenerator()
        mask = mg.generate_anonymous_mask()
        assert mask.mask_type == MaskType.ANONYMOUS
        assert mask.opacity == 1.0  # Fully opaque
        assert mask.name.startswith("anon_")

    def test_role_mask(self):
        """Test generating a role mask."""
        mg = MaskGenerator()
        mask = mg.generate_mask(
            name="admin_mask",
            mask_type=MaskType.ROLE,
            roles=["admin", "moderator"],
        )
        assert mask.mask_type == MaskType.ROLE
        assert "admin" in mask.roles
        assert "moderator" in mask.roles


class TestMaskAssignment:
    """Test mask assignment functionality."""

    def test_assign_mask(self):
        """Test assigning a mask to an entity."""
        mg = MaskGenerator()
        mask = mg.generate_mask(name="test_mask")
        assignment = mg.assign_mask(
            mask_id=mask.id,
            entity_id="user_123",
            entity_type="user",
        )
        assert assignment is not None
        assert assignment.mask_id == mask.id
        assert assignment.entity_id == "user_123"
        assert mg.assignment_count == 1

    def test_assign_nonexistent_mask(self):
        """Test assigning a nonexistent mask returns None."""
        mg = MaskGenerator()
        assignment = mg.assign_mask(
            mask_id="nonexistent",
            entity_id="user_123",
        )
        assert assignment is None

    def test_get_entity_assignments(self):
        """Test getting all assignments for an entity."""
        mg = MaskGenerator()
        mask1 = mg.generate_mask(name="mask1")
        mask2 = mg.generate_mask(name="mask2")
        
        mg.assign_mask(mask1.id, "user_123")
        mg.assign_mask(mask2.id, "user_123")
        
        assignments = mg.get_entity_assignments("user_123")
        assert len(assignments) == 2

    def test_revoke_assignment(self):
        """Test revoking a mask assignment."""
        mg = MaskGenerator()
        mask = mg.generate_mask(name="test_mask")
        assignment = mg.assign_mask(mask.id, "user_123")
        
        result = mg.revoke_assignment(assignment.id)
        assert result is True
        
        # Assignment should be gone
        assignments = mg.get_entity_assignments("user_123")
        assert len(assignments) == 0


class TestMaskComposition:
    """Test mask composition functionality."""

    def test_compose_masks(self):
        """Test composing multiple masks."""
        mg = MaskGenerator()
        mask1 = mg.generate_mask(
            name="base_mask",
            attributes=["name", "email"],
            roles=["user"],
        )
        mask2 = mg.generate_mask(
            name="premium_mask",
            attributes=["phone"],
            roles=["premium"],
        )
        
        composed = mg.compose_mask(
            name="combined_mask",
            mask_ids=[mask1.id, mask2.id],
        )
        
        assert composed is not None
        assert composed.mask_type == MaskType.COMPOSITE
        assert "name" in composed.attributes
        assert "email" in composed.attributes
        assert "phone" in composed.attributes
        assert "user" in composed.roles
        assert "premium" in composed.roles
        assert mask1.id in composed.parent_mask_ids
        assert mask2.id in composed.parent_mask_ids

    def test_compose_empty_list(self):
        """Test composing with empty list returns None."""
        mg = MaskGenerator()
        composed = mg.compose_mask(name="empty", mask_ids=[])
        assert composed is None


class TestMaskStateManagement:
    """Test mask state management."""

    def test_suspend_mask(self):
        """Test suspending a mask."""
        mg = MaskGenerator()
        mask = mg.generate_mask(name="test_mask")
        
        updated = mg.suspend_mask(mask.id)
        assert updated is not None
        assert updated.state == MaskState.SUSPENDED

    def test_revoke_mask(self):
        """Test revoking a mask."""
        mg = MaskGenerator()
        mask = mg.generate_mask(name="test_mask")
        
        updated = mg.revoke_mask(mask.id)
        assert updated is not None
        assert updated.state == MaskState.REVOKED

    def test_get_active_masks(self):
        """Test getting active masks."""
        mg = MaskGenerator()
        mask1 = mg.generate_mask(name="active_mask")
        mask2 = mg.generate_mask(name="suspended_mask")
        mg.suspend_mask(mask2.id)
        
        active = mg.get_active_masks()
        assert len(active) == 1
        assert active[0].id == mask1.id

    def test_check_expired_masks(self):
        """Test checking for expired masks."""
        mg = MaskGenerator()
        # Create an already-expired mask
        past = datetime.now(UTC) - timedelta(hours=1)
        mask = mg.generate_mask(
            name="expired_mask",
            mask_type=MaskType.TEMPORAL,
            valid_until=past,
        )
        
        expired = mg.check_expired_masks()
        assert len(expired) == 1
        assert expired[0].id == mask.id


class TestMaskStats:
    """Test mask statistics."""

    def test_get_stats(self):
        """Test getting mask statistics."""
        mg = MaskGenerator()
        mg.generate_mask(name="mask1", mask_type=MaskType.IDENTITY)
        mg.generate_mask(name="mask2", mask_type=MaskType.ROLE)
        mg.generate_mask(name="mask3", mask_type=MaskType.IDENTITY)
        
        stats = mg.get_stats()
        assert stats.total_masks == 3
        assert stats.active_masks == 3
        assert stats.masks_by_type["IDENTITY"] == 2
        assert stats.masks_by_type["ROLE"] == 1

    def test_clear(self):
        """Test clearing all data."""
        mg = MaskGenerator()
        mg.generate_mask(name="mask1")
        mask = mg.generate_mask(name="mask2")
        mg.assign_mask(mask.id, "user_123")
        
        masks, assignments = mg.clear()
        assert masks == 2
        assert assignments == 1
        assert mg.mask_count == 0
        assert mg.assignment_count == 0


# ============================================================================
# AudienceClassifier Tests
# ============================================================================

class TestAudienceClassifierBasics:
    """Basic AudienceClassifier tests."""

    def test_initialization(self):
        """Test AudienceClassifier initializes correctly."""
        ac = AudienceClassifier()
        assert ac.name == "audience_classifier"
        assert ac.segment_count == 0
        assert ac.member_count == 0
        assert ac.rule_count == 0

    def test_create_segment(self):
        """Test creating a segment."""
        ac = AudienceClassifier()
        segment = ac.create_segment(
            name="premium",
            segment_type=SegmentType.TIER,
            description="Premium users",
            access_level=AccessLevel.PREMIUM,
        )
        assert segment is not None
        assert segment.name == "premium"
        assert segment.segment_type == SegmentType.TIER
        assert segment.access_level == AccessLevel.PREMIUM
        assert ac.segment_count == 1

    def test_get_segment(self):
        """Test getting a segment by ID."""
        ac = AudienceClassifier()
        segment = ac.create_segment(name="test")
        retrieved = ac.get_segment(segment.id)
        assert retrieved is not None
        assert retrieved.id == segment.id

    def test_get_segment_by_name(self):
        """Test getting a segment by name."""
        ac = AudienceClassifier()
        segment = ac.create_segment(name="premium")
        retrieved = ac.get_segment_by_name("premium")
        assert retrieved is not None
        assert retrieved.id == segment.id


class TestSegmentTypes:
    """Test different segment types."""

    def test_tier_segment(self):
        """Test creating a tier segment."""
        ac = AudienceClassifier()
        segment = ac.create_segment(
            name="gold",
            segment_type=SegmentType.TIER,
            priority=10,
        )
        assert segment.segment_type == SegmentType.TIER
        assert segment.priority == 10

    def test_behavioral_segment(self):
        """Test creating a behavioral segment."""
        ac = AudienceClassifier()
        segment = ac.create_segment(
            name="frequent_buyers",
            segment_type=SegmentType.BEHAVIORAL,
        )
        assert segment.segment_type == SegmentType.BEHAVIORAL

    def test_default_segment(self):
        """Test creating a default segment."""
        ac = AudienceClassifier()
        segment = ac.create_segment(
            name="basic",
            is_default=True,
        )
        assert segment.is_default is True


class TestClassificationRules:
    """Test classification rules."""

    def test_add_rule(self):
        """Test adding a classification rule."""
        ac = AudienceClassifier()
        segment = ac.create_segment(name="premium")
        rule = ac.add_rule(
            name="subscription_check",
            segment_id=segment.id,
            attribute="subscription",
            operator=RuleOperator.EQUALS,
            value="premium",
        )
        assert rule is not None
        assert rule.segment_id == segment.id
        assert rule.attribute == "subscription"
        assert ac.rule_count == 1

    def test_get_rules_for_segment(self):
        """Test getting rules for a segment."""
        ac = AudienceClassifier()
        segment = ac.create_segment(name="test")
        ac.add_rule("rule1", segment.id, "attr1", RuleOperator.EQUALS, "val1")
        ac.add_rule("rule2", segment.id, "attr2", RuleOperator.GREATER_THAN, 10)
        
        rules = ac.get_rules_for_segment(segment.id)
        assert len(rules) == 2


class TestMemberManagement:
    """Test member management."""

    def test_register_member(self):
        """Test registering a member."""
        ac = AudienceClassifier()
        member = ac.register_member(
            external_id="ext_123",
            name="John Doe",
            attributes={"age": 30, "country": "US"},
        )
        assert member is not None
        assert member.external_id == "ext_123"
        assert member.name == "John Doe"
        assert member.attributes["age"] == 30
        assert ac.member_count == 1

    def test_get_member(self):
        """Test getting a member by ID."""
        ac = AudienceClassifier()
        member = ac.register_member(name="Test User")
        retrieved = ac.get_member(member.id)
        assert retrieved is not None
        assert retrieved.id == member.id

    def test_get_member_by_external_id(self):
        """Test getting a member by external ID."""
        ac = AudienceClassifier()
        member = ac.register_member(external_id="ext_123")
        retrieved = ac.get_member_by_external_id("ext_123")
        assert retrieved is not None
        assert retrieved.id == member.id

    def test_update_member_attributes(self):
        """Test updating member attributes."""
        ac = AudienceClassifier()
        member = ac.register_member(attributes={"level": 1})
        updated = ac.update_member_attributes(
            member.id,
            attributes={"level": 2, "premium": True},
        )
        assert updated is not None
        assert updated.attributes["level"] == 2
        assert updated.attributes["premium"] is True


class TestClassification:
    """Test member classification."""

    def test_classify_member_basic(self):
        """Test basic member classification."""
        ac = AudienceClassifier()
        
        # Create segment with rule
        segment = ac.create_segment(name="premium")
        ac.add_rule(
            name="premium_check",
            segment_id=segment.id,
            attribute="subscription",
            operator=RuleOperator.EQUALS,
            value="premium",
        )
        
        # Register and classify member
        member = ac.register_member(attributes={"subscription": "premium"})
        result = ac.classify_member(member.id)
        
        assert result.success is True
        assert segment.id in result.segments

    def test_classify_member_no_match(self):
        """Test classification with no matching segment."""
        ac = AudienceClassifier()
        
        # Create segment with rule
        segment = ac.create_segment(name="premium")
        ac.add_rule(
            name="premium_check",
            segment_id=segment.id,
            attribute="subscription",
            operator=RuleOperator.EQUALS,
            value="premium",
        )
        
        # Register member that doesn't match
        member = ac.register_member(attributes={"subscription": "basic"})
        result = ac.classify_member(member.id)
        
        assert result.success is True
        assert segment.id not in result.segments

    def test_classify_with_default_segment(self):
        """Test classification falls back to default segment."""
        ac = AudienceClassifier()
        
        # Create default segment
        default = ac.create_segment(name="basic", is_default=True)
        
        # Create premium segment with rule
        premium = ac.create_segment(name="premium")
        ac.add_rule(
            name="premium_check",
            segment_id=premium.id,
            attribute="subscription",
            operator=RuleOperator.EQUALS,
            value="premium",
        )
        
        # Member that doesn't match premium
        member = ac.register_member(attributes={"subscription": "free"})
        result = ac.classify_member(member.id)
        
        assert result.success is True
        assert default.id in result.segments
        assert premium.id not in result.segments

    def test_classify_multiple_segments(self):
        """Test classification into multiple segments."""
        ac = AudienceClassifier()
        
        segment1 = ac.create_segment(name="active")
        ac.add_rule("active_check", segment1.id, "status", RuleOperator.EQUALS, "active")
        
        segment2 = ac.create_segment(name="premium")
        ac.add_rule("premium_check", segment2.id, "tier", RuleOperator.EQUALS, "premium")
        
        member = ac.register_member(attributes={"status": "active", "tier": "premium"})
        result = ac.classify_member(member.id)
        
        assert result.success is True
        assert segment1.id in result.segments
        assert segment2.id in result.segments

    def test_exclusive_segment(self):
        """Test exclusive segment stops further classification."""
        ac = AudienceClassifier()
        
        # Create exclusive VIP segment with highest priority
        vip = ac.create_segment(name="vip", is_exclusive=True, priority=100)
        ac.add_rule("vip_check", vip.id, "vip", RuleOperator.EQUALS, True)
        
        # Create another segment
        active = ac.create_segment(name="active", priority=50)
        ac.add_rule("active_check", active.id, "status", RuleOperator.EQUALS, "active")
        
        # Member matches both, but VIP is exclusive
        member = ac.register_member(attributes={"vip": True, "status": "active"})
        result = ac.classify_member(member.id)
        
        assert result.success is True
        assert vip.id in result.segments
        assert active.id not in result.segments  # Stopped due to exclusive


class TestRuleOperators:
    """Test different rule operators."""

    def test_equals_operator(self):
        """Test EQUALS operator."""
        ac = AudienceClassifier()
        segment = ac.create_segment(name="test")
        ac.add_rule("r", segment.id, "value", RuleOperator.EQUALS, "match")
        
        member = ac.register_member(attributes={"value": "match"})
        result = ac.classify_member(member.id)
        assert segment.id in result.segments

    def test_greater_than_operator(self):
        """Test GREATER_THAN operator."""
        ac = AudienceClassifier()
        segment = ac.create_segment(name="test")
        ac.add_rule("r", segment.id, "age", RuleOperator.GREATER_THAN, 18)
        
        member = ac.register_member(attributes={"age": 25})
        result = ac.classify_member(member.id)
        assert segment.id in result.segments
        
        member2 = ac.register_member(attributes={"age": 15})
        result2 = ac.classify_member(member2.id)
        assert segment.id not in result2.segments

    def test_contains_operator(self):
        """Test CONTAINS operator."""
        ac = AudienceClassifier()
        segment = ac.create_segment(name="test")
        ac.add_rule("r", segment.id, "email", RuleOperator.CONTAINS, "@company.com")
        
        member = ac.register_member(attributes={"email": "user@company.com"})
        result = ac.classify_member(member.id)
        assert segment.id in result.segments

    def test_in_operator(self):
        """Test IN operator."""
        ac = AudienceClassifier()
        segment = ac.create_segment(name="test")
        ac.add_rule("r", segment.id, "country", RuleOperator.IN, ["US", "CA", "UK"])
        
        member = ac.register_member(attributes={"country": "US"})
        result = ac.classify_member(member.id)
        assert segment.id in result.segments
        
        member2 = ac.register_member(attributes={"country": "FR"})
        result2 = ac.classify_member(member2.id)
        assert segment.id not in result2.segments

    def test_exists_operator(self):
        """Test EXISTS operator."""
        ac = AudienceClassifier()
        segment = ac.create_segment(name="test")
        ac.add_rule("r", segment.id, "premium_features", RuleOperator.EXISTS, None)
        
        member = ac.register_member(attributes={"premium_features": ["a", "b"]})
        result = ac.classify_member(member.id)
        assert segment.id in result.segments
        
        member2 = ac.register_member(attributes={"other": "value"})
        result2 = ac.classify_member(member2.id)
        assert segment.id not in result2.segments


class TestManualAssignment:
    """Test manual segment assignment."""

    def test_manual_assignment(self):
        """Test manually assigning a member to a segment."""
        ac = AudienceClassifier()
        segment = ac.create_segment(name="special")
        member = ac.register_member(name="Test User")
        
        membership = ac.assign_to_segment(member.id, segment.id)
        assert membership is not None
        assert membership.member_id == member.id
        assert membership.segment_id == segment.id

    def test_assignment_respects_max_members(self):
        """Test assignment respects max_members limit."""
        ac = AudienceClassifier()
        segment = ac.create_segment(name="limited", max_members=1)
        
        member1 = ac.register_member(name="User 1")
        member2 = ac.register_member(name="User 2")
        
        m1 = ac.assign_to_segment(member1.id, segment.id)
        m2 = ac.assign_to_segment(member2.id, segment.id)
        
        assert m1 is not None
        assert m2 is None  # Should fail due to limit


class TestAccessLevels:
    """Test access level functionality."""

    def test_get_member_access_level(self):
        """Test getting member's access level."""
        ac = AudienceClassifier()
        
        basic = ac.create_segment(name="basic", access_level=AccessLevel.BASIC)
        premium = ac.create_segment(name="premium", access_level=AccessLevel.PREMIUM)
        
        member = ac.register_member(name="Test User")
        ac.assign_to_segment(member.id, basic.id)
        ac.assign_to_segment(member.id, premium.id)
        
        level = ac.get_member_access_level(member.id)
        assert level == AccessLevel.PREMIUM  # Highest level

    def test_no_membership_returns_none_access(self):
        """Test unassigned member has NONE access."""
        ac = AudienceClassifier()
        member = ac.register_member(name="Test User")
        level = ac.get_member_access_level(member.id)
        assert level == AccessLevel.NONE


class TestReclassification:
    """Test member reclassification."""

    def test_reclassify_member(self):
        """Test reclassifying a member after attribute update."""
        ac = AudienceClassifier()
        
        basic = ac.create_segment(name="basic", is_default=True)
        premium = ac.create_segment(name="premium")
        ac.add_rule("premium_check", premium.id, "tier", RuleOperator.EQUALS, "premium")
        
        # Initially basic
        member = ac.register_member(attributes={"tier": "basic"})
        result1 = ac.classify_member(member.id)
        assert basic.id in result1.segments
        assert premium.id not in result1.segments
        
        # Update to premium
        ac.update_member_attributes(member.id, {"tier": "premium"})
        result2 = ac.reclassify_member(member.id)
        
        assert premium.id in result2.segments


class TestAudienceStats:
    """Test audience statistics."""

    def test_get_stats(self):
        """Test getting classification statistics."""
        ac = AudienceClassifier()
        
        s1 = ac.create_segment(name="segment1", segment_type=SegmentType.TIER)
        s2 = ac.create_segment(name="segment2", segment_type=SegmentType.BEHAVIORAL)
        
        m1 = ac.register_member(name="User 1")
        m2 = ac.register_member(name="User 2")
        
        ac.assign_to_segment(m1.id, s1.id)
        ac.assign_to_segment(m2.id, s1.id)
        ac.assign_to_segment(m2.id, s2.id)
        
        stats = ac.get_stats()
        assert stats.total_segments == 2
        assert stats.total_members == 2
        assert stats.total_memberships == 3
        assert stats.members_by_segment["segment1"] == 2
        assert stats.members_by_segment["segment2"] == 1

    def test_clear(self):
        """Test clearing all data."""
        ac = AudienceClassifier()
        
        segment = ac.create_segment(name="test")
        ac.add_rule("r", segment.id, "attr", RuleOperator.EQUALS, "val")
        ac.register_member(name="User")
        
        segments, members, rules = ac.clear()
        assert segments == 1
        assert members == 1
        assert rules == 1
        assert ac.segment_count == 0
        assert ac.member_count == 0
        assert ac.rule_count == 0
