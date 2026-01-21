---
title: MaskGenerator
system: Recursive–Generative Organizational Body
type: subsystem
category: identity
tags: [mask, identity, privacy, roles]
dependencies: []
---

# MaskGenerator

The **MaskGenerator** creates and manages symbolic identity masks that abstract identity, represent roles or states, and help maintain privacy within the system.

## Overview

| Property | Value |
|----------|-------|
| Category | Identity |
| Module | `autogenrec.subsystems.identity.mask_generator` |
| Dependencies | None |

## Domain Models

### Enums

```python
class MaskType(Enum):
    IDENTITY = auto()      # Full identity mask
    ROLE = auto()          # Role-based mask
    TEMPORAL = auto()      # Time-limited mask
    COMPOSITE = auto()     # Composed from multiple masks
    ANONYMOUS = auto()     # Privacy-preserving mask
    PSEUDONYMOUS = auto()  # Consistent but unlinkable

class MaskState(Enum):
    ACTIVE = auto()        # Currently in use
    SUSPENDED = auto()     # Temporarily inactive
    EXPIRED = auto()       # Past valid period
    REVOKED = auto()       # Permanently disabled
```

### Core Models

- **Mask**: Identity mask with type, state, attributes, roles, opacity level
- **MaskAssignment**: Links a mask to an entity with context
- **MaskLayer**: Layer in a composite mask with precedence

## Process Loop

1. **Intake**: Receive identity attributes, parameters, and mask requests
2. **Process**: Generate masks based on type and attributes
3. **Evaluate**: Validate mask properties and compositions
4. **Integrate**: Assign masks to entities, emit creation events

## Public API

### Mask Generation

```python
from autogenrec.subsystems.identity.mask_generator import (
    MaskGenerator, MaskType
)

masks = MaskGenerator()

# Generate different mask types
admin_mask = masks.generate_mask(
    name="admin_identity",
    mask_type=MaskType.ROLE,
    entity_id="user_001",
    roles=["admin", "moderator"],
    attributes=["verified", "trusted"],
)

# Anonymous mask for privacy
guest = masks.generate_anonymous_mask()

# Pseudonymous mask (consistent identifier)
pseudonym = masks.generate_mask(
    name="researcher_alias",
    mask_type=MaskType.PSEUDONYMOUS,
    entity_id="researcher_001",
    roles=["author"],
)
```

### Mask Composition

```python
# Create base layers
base = masks.generate_mask(
    name="base_identity",
    mask_type=MaskType.ROLE,
    roles=["user"],
)

premium = masks.generate_mask(
    name="premium_layer",
    mask_type=MaskType.ROLE,
    roles=["premium"],
    attributes=["verified"],
)

# Compose into single mask
composite = masks.compose_mask(
    "composite_identity",
    [base.id, premium.id]
)
# Result: roles={"user", "premium"}, attributes={"verified"}
```

### Mask Management

```python
# Retrieve mask
mask = masks.get_mask(mask_id)

# Suspend/revoke
masks.suspend_mask(mask_id)
masks.revoke_mask(mask_id)

# Check if entity has mask
has_mask = masks.entity_has_mask("user_001", mask_id)
```

### Statistics

```python
stats = masks.get_stats()
# MaskStats with:
#   total_masks, active_masks
#   masks_by_type (dict mapping MaskType -> count)
```

## Interpretation

Masks abstract identity into symbolic representations:
- **IDENTITY**: Full representation of an entity
- **ROLE**: Functional capabilities without personal identity
- **TEMPORAL**: Time-bounded access or permissions
- **COMPOSITE**: Layered combination of multiple masks
- **ANONYMOUS**: No linkable identity information
- **PSEUDONYMOUS**: Consistent identifier without real identity

## Example

See `examples/basic_narrative_interpretation.py` for identity mask creation and composition.
