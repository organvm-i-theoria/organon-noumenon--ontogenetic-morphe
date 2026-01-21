---
title: ConflictResolver
system: Recursive–Generative Organizational Body
type: subsystem
category: conflict
tags: [conflict, resolution, detection, reconciliation]
dependencies: [ArbitrationEngine]
---

# ConflictResolver

The **ConflictResolver** detects and resolves conflicting inputs using configurable resolution strategies including priority, timestamp, and consensus-based approaches.

## Overview

| Property | Value |
|----------|-------|
| Category | Conflict & Resolution |
| Module | `autogenrec.subsystems.conflict.conflict_resolver` |
| Dependencies | ArbitrationEngine |

## Domain Models

### Enums

```python
class ConflictType(Enum):
    VALUE = auto()         # Conflicting values
    TEMPORAL = auto()      # Time-based conflict
    STRUCTURAL = auto()    # Schema conflict
    SEMANTIC = auto()      # Meaning conflict

class ResolutionStrategy(Enum):
    PRIORITY = auto()      # Highest priority wins
    TIMESTAMP = auto()     # Most recent wins
    CONSENSUS = auto()     # Majority agreement
    MERGE = auto()         # Combine values
    ESCALATE = auto()      # Send to arbitration

class ConflictStatus(Enum):
    DETECTED = auto()
    RESOLVING = auto()
    RESOLVED = auto()
    ESCALATED = auto()
    UNRESOLVABLE = auto()
```

### Core Models

- **Conflict**: Detected conflict with type, sources, status
- **Resolution**: Applied resolution with strategy, outcome
- **ConflictInput**: Input involved in a conflict

## Process Loop

1. **Intake**: Receive potentially conflicting inputs
2. **Process**: Detect conflicts, analyze type and severity
3. **Evaluate**: Apply resolution strategy, determine outcome
4. **Integrate**: Update state, emit resolution events

## Public API

### Conflict Detection

```python
from autogenrec.subsystems.conflict.conflict_resolver import (
    ConflictResolver, ConflictType, ResolutionStrategy
)
from autogenrec.core.symbolic import SymbolicValue, SymbolicValueType

resolver = ConflictResolver()

# Detect conflicts between symbolic values
conflict = await resolver.detect_conflict(
    inputs=[value_a, value_b],
    conflict_type=ConflictType.VALUE,
)

if conflict:
    print(f"Conflict detected: {conflict.conflict_type.name}")
```

### Resolution

```python
# Resolve using priority strategy
resolution = await resolver.resolve(
    conflict_id=conflict.id,
    strategy=ResolutionStrategy.PRIORITY,
)

# Resolve using timestamp (most recent wins)
resolution = await resolver.resolve(
    conflict_id=conflict.id,
    strategy=ResolutionStrategy.TIMESTAMP,
)

# Resolve using consensus
resolution = await resolver.resolve(
    conflict_id=conflict.id,
    strategy=ResolutionStrategy.CONSENSUS,
)
```

### Escalation

```python
# Escalate unresolvable conflicts to arbitration
if conflict.status == ConflictStatus.UNRESOLVABLE:
    await resolver.escalate(
        conflict_id=conflict.id,
        reason="Cannot resolve automatically",
    )
```

### Statistics

```python
stats = resolver.get_stats()
# ConflictStats with:
#   total_conflicts, resolved_conflicts
#   escalated_conflicts
#   resolution_by_strategy
```

## Resolution Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| PRIORITY | Highest priority wins | Authoritative sources |
| TIMESTAMP | Most recent wins | Time-sensitive data |
| CONSENSUS | Majority agreement | Multiple sources |
| MERGE | Combine values | Compatible data |
| ESCALATE | Human/arbitration | Complex conflicts |

## Integration

The ConflictResolver works with:
- **ArbitrationEngine**: For escalated conflicts
- **All subsystems**: Conflict detection on inputs

## Example

The ConflictResolver handles conflicts detected during symbolic processing workflows.
