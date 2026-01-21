---
title: AudienceClassifier
system: Recursive–Generative Organizational Body
type: subsystem
category: identity
tags: [audience, classification, segments, access]
dependencies: [MaskGenerator]
---

# AudienceClassifier

The **AudienceClassifier** categorizes users and entities into segments based on rules, managing access levels and multi-segment membership.

## Overview

| Property | Value |
|----------|-------|
| Category | Identity |
| Module | `autogenrec.subsystems.identity.audience_classifier` |
| Dependencies | MaskGenerator |

## Domain Models

### Enums

```python
class SegmentType(Enum):
    TIER = auto()          # Access tier (basic, premium, VIP)
    BEHAVIORAL = auto()    # Based on behavior patterns
    DEMOGRAPHIC = auto()   # Based on demographics
    CUSTOM = auto()        # Custom-defined segment

class AccessLevel(Enum):
    NONE = auto()          # No access
    BASIC = auto()         # Basic access
    STANDARD = auto()      # Standard access
    PREMIUM = auto()       # Premium access
    VIP = auto()           # VIP access
    ADMIN = auto()         # Administrative access

class RuleOperator(Enum):
    EQUALS = auto()        # Exact match
    NOT_EQUALS = auto()    # Not equal
    GREATER_THAN = auto()  # Greater than
    LESS_THAN = auto()     # Less than
    IN = auto()            # Value in list
    NOT_IN = auto()        # Value not in list
    CONTAINS = auto()      # String contains
    MATCHES = auto()       # Regex match
```

### Core Models

- **Segment**: Audience segment with type, access level, rules
- **ClassificationRule**: Rule for segment membership
- **Member**: Registered member with attributes
- **ClassificationResult**: Result of classifying a member

## Process Loop

1. **Intake**: Receive member attributes and classification requests
2. **Process**: Apply rules to determine segment membership
3. **Evaluate**: Resolve conflicts, determine highest access level
4. **Integrate**: Update memberships, emit classification events

## Public API

### Segment Creation

```python
from autogenrec.subsystems.identity.audience_classifier import (
    AudienceClassifier, SegmentType, AccessLevel, RuleOperator
)

classifier = AudienceClassifier()

# Create tiered segments
basic = classifier.create_segment(
    name="Basic Users",
    segment_type=SegmentType.TIER,
    access_level=AccessLevel.BASIC,
    is_default=True,  # Fallback segment
)

premium = classifier.create_segment(
    name="Premium Users",
    segment_type=SegmentType.TIER,
    access_level=AccessLevel.PREMIUM,
)

# Add classification rules
classifier.add_rule(
    name="premium_subscription",
    segment_id=premium.id,
    attribute="subscription",
    operator=RuleOperator.EQUALS,
    value="premium",
)
```

### Member Registration and Classification

```python
# Register member with attributes
member = classifier.register_member(
    external_id="user_123",
    attributes={
        "name": "John Doe",
        "subscription": "premium",
        "region": "US",
    },
)

# Classify member against all rules
result = classifier.classify_member(member.id)
# ClassificationResult with matched segments

# Get effective access level
access = classifier.get_member_access_level(member.id)
# Returns highest AccessLevel from matched segments
```

### Statistics

```python
stats = classifier.get_stats()
# SegmentStats with:
#   total_segments, total_members
#   total_memberships (members can belong to multiple segments)
```

## Rule Evaluation

Rules are evaluated in priority order. A member matches a segment if **all** rules for that segment pass. Members can belong to multiple segments simultaneously.

Access level resolution returns the **highest** level from all matched segments.

## Integration

The AudienceClassifier works with:
- **MaskGenerator**: For identity abstraction of classified members

## Example

See `examples/basic_narrative_interpretation.py` for segment creation, rule definition, and member classification.
