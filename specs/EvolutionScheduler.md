---
title: EvolutionScheduler
system: Recursive–Generative Organizational Body
type: subsystem
category: temporal
tags: [evolution, growth, mutation, patterns]
dependencies: [TimeManager]
---

# EvolutionScheduler

The **EvolutionScheduler** manages the timing and progression of growth and mutation cycles, governing symbolic evolution within recursive systems.

## Overview

| Property | Value |
|----------|-------|
| Category | Temporal & Spatial |
| Module | `autogenrec.subsystems.temporal.evolution_scheduler` |
| Dependencies | TimeManager |

## Domain Models

### Enums

```python
class GrowthPhase(Enum):
    DORMANT = auto()       # Inactive, awaiting trigger
    GERMINATING = auto()   # Initial growth
    GROWING = auto()       # Active development
    MATURE = auto()        # Fully developed
    DECLINING = auto()     # Past peak

class MutationType(Enum):
    ADDITION = auto()      # Add new capability
    MODIFICATION = auto()  # Modify existing
    DELETION = auto()      # Remove capability
    RECOMBINATION = auto() # Combine elements

class Stability(Enum):
    UNSTABLE = auto()
    STABILIZING = auto()
    STABLE = auto()
    CRYSTALLIZED = auto()
```

### Core Models

- **GrowthPattern**: Evolvable pattern with content, phase, fitness, generation
- **Mutation**: Applied mutation with type, impact, success status

## Process Loop

1. **Intake**: Receive evolution triggers, pattern definitions
2. **Process**: Apply mutations, advance growth phases
3. **Evaluate**: Assess fitness, determine stability
4. **Integrate**: Update patterns, emit evolution events

## Public API

### Pattern Creation

```python
from autogenrec.subsystems.temporal.evolution_scheduler import (
    EvolutionScheduler, GrowthPhase, MutationType
)

evolution = EvolutionScheduler()

# Create an evolvable pattern
pattern = evolution.create_pattern(
    name="data_processor",
    content={
        "type": "processor",
        "version": "1.0",
        "capabilities": ["parse", "validate"],
        "efficiency": 0.7,
    },
    phase=GrowthPhase.DORMANT,
    description="A pattern that evolves over time",
)
```

### Mutation

```python
# Apply mutation
mutated, mutation = evolution.mutate_pattern(
    pattern_id=pattern.id,
    mutation_type=MutationType.ADDITION,
)

print(f"Generation: {mutated.generation}")
print(f"Fitness: {mutated.fitness:.3f}")
print(f"Mutation Impact: {mutation.impact:.3f}")
print(f"Success: {mutation.successful}")
```

### Phase Advancement

```python
# Manually advance growth phase
advanced = evolution.advance_phase(pattern.id)
print(f"New Phase: {advanced.phase.name}")

# Phases progress: DORMANT -> GERMINATING -> GROWING -> MATURE -> DECLINING
```

### Pattern Queries

```python
# Get pattern by ID
pattern = evolution.get_pattern(pattern_id)

# List patterns by phase
growing = evolution.list_patterns_by_phase(GrowthPhase.GROWING)

# Get pattern fitness over time
# (tracked through generation increments)
```

### Statistics

```python
stats = evolution.get_stats()
# EvolutionStats with:
#   total_patterns, patterns_by_phase
#   total_mutations, successful_mutations
```

## Evolution Model

Patterns evolve through:

1. **Mutations**: Random modifications that affect fitness
2. **Phase Transitions**: Natural progression through growth stages
3. **Fitness**: Numeric measure (0.0-1.0) of pattern quality
4. **Stability**: How settled the pattern has become

```
DORMANT → GERMINATING → GROWING → MATURE → DECLINING
           (mutations increase fitness)
```

## Integration

The EvolutionScheduler works with:
- **TimeManager**: For scheduled evolution cycles
- **ProcessConverter**: For converting evolved patterns

## Example

See `examples/recursive_process_demo.py` for evolution cycles with mutations and phase advancement.
