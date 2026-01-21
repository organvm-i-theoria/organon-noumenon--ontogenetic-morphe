---
title: ReferenceManager
system: Recursive–Generative Organizational Body
type: subsystem
category: data
tags: [reference, canonical, graph, resolution]
dependencies: []
---

# ReferenceManager

The **ReferenceManager** maintains canonical references, managing a reference graph for resolution, creation, and relationship queries.

## Overview

| Property | Value |
|----------|-------|
| Category | Data & Records |
| Module | `autogenrec.subsystems.data.reference_manager` |
| Dependencies | None |

## Domain Models

### Enums

```python
class ReferenceType(Enum):
    CANONICAL = auto()     # Authoritative reference
    ALIAS = auto()         # Alternative name
    DERIVED = auto()       # Computed reference
    EXTERNAL = auto()      # External source

class RelationType(Enum):
    PARENT = auto()        # Parent reference
    CHILD = auto()         # Child reference
    SIBLING = auto()       # Related reference
    ALIAS_OF = auto()      # Alias relationship
```

### Core Models

- **Reference**: Canonical reference with type, target, metadata
- **ReferenceLink**: Relationship between references
- **ResolveResult**: Result of reference resolution

## Process Loop

1. **Intake**: Receive reference creation/resolution requests
2. **Process**: Build reference graph, resolve relationships
3. **Evaluate**: Validate references, detect cycles
4. **Integrate**: Update graph, emit reference events

## Public API

### Reference Creation

```python
from autogenrec.subsystems.data.reference_manager import (
    ReferenceManager, ReferenceType, RelationType
)

refs = ReferenceManager()

# Create canonical reference
ref = await refs.create(
    name="main_config",
    reference_type=ReferenceType.CANONICAL,
    target="config://system/main",
    metadata={"version": "1.0"},
)

# Create alias
alias = await refs.create(
    name="config",
    reference_type=ReferenceType.ALIAS,
    target=ref.id,
)
```

### Resolution

```python
# Resolve reference by name
result = await refs.resolve("main_config")
if result.found:
    print(f"Target: {result.reference.target}")

# Resolve with aliases
result = await refs.resolve("config", follow_aliases=True)
```

### Relationships

```python
# Link references
await refs.link(
    source_id=parent_ref.id,
    target_id=child_ref.id,
    relation_type=RelationType.PARENT,
)

# Get related references
children = await refs.get_related(
    reference_id=parent_ref.id,
    relation_type=RelationType.CHILD,
)
```

### Statistics

```python
stats = refs.get_stats()
# ReferenceStats with:
#   total_references, references_by_type
#   total_links
```

## Reference Graph

References form a directed graph:

```
canonical_ref
    ├── alias_1 (ALIAS_OF)
    ├── alias_2 (ALIAS_OF)
    └── child_ref (CHILD)
            └── grandchild_ref (CHILD)
```

## Integration

The ReferenceManager provides:
- **AcademiaManager**: Citation references
- **ArchiveManager**: Archive references
- **All subsystems**: Canonical naming

## Example

The ReferenceManager is used internally for maintaining canonical references across the system.
