---
title: LocationResolver
system: Recursive–Generative Organizational Body
type: subsystem
category: temporal
tags: [location, spatial, places, relationships]
dependencies: []
---

# LocationResolver

The **LocationResolver** resolves spatial references, manages place definitions, and computes relationships between locations in the system.

## Overview

| Property | Value |
|----------|-------|
| Category | Temporal & Spatial |
| Module | `autogenrec.subsystems.temporal.location_resolver` |
| Dependencies | None |

## Domain Models

### Enums

```python
class PlaceType(Enum):
    PHYSICAL = auto()      # Physical location
    VIRTUAL = auto()       # Virtual/digital space
    CONCEPTUAL = auto()    # Abstract concept space
    HYBRID = auto()        # Mixed physical/virtual

class SpatialRelation(Enum):
    CONTAINS = auto()      # A contains B
    WITHIN = auto()        # A is within B
    ADJACENT = auto()      # A is next to B
    CONNECTED = auto()     # A connects to B
    OVERLAPS = auto()      # A and B overlap

class ResolutionStatus(Enum):
    RESOLVED = auto()      # Successfully resolved
    AMBIGUOUS = auto()     # Multiple matches
    NOT_FOUND = auto()     # No match found
    ERROR = auto()         # Resolution error
```

### Core Models

- **Place**: Location with type, coordinates, metadata
- **SpatialLink**: Relationship between two places
- **ResolutionResult**: Result of resolving a location reference

## Process Loop

1. **Intake**: Receive location queries, place definitions
2. **Process**: Resolve references, compute relationships
3. **Evaluate**: Validate results, handle ambiguity
4. **Integrate**: Update spatial graph, emit resolution events

## Public API

### Place Management

```python
from autogenrec.subsystems.temporal.location_resolver import (
    LocationResolver, PlaceType, SpatialRelation, ResolutionStatus
)

location = LocationResolver()

# Create places
lab = location.create_place(
    name="AI Research Lab",
    place_type=PlaceType.VIRTUAL,
    metadata={"department": "research"},
)

server = location.create_place(
    name="Cloud Server",
    place_type=PlaceType.VIRTUAL,
    metadata={"region": "us-east-1"},
)

# Get place by ID
place = location.get_place(lab.id)
```

### Spatial Relationships

```python
# Link two places
location.link_places(
    place_id_a=lab.id,
    place_id_b=server.id,
    relation=SpatialRelation.CONNECTED,
)

# Query relationships
connected = location.get_connected_places(lab.id)

# Check specific relationship
is_connected = location.are_related(
    lab.id, server.id, SpatialRelation.CONNECTED
)
```

### Resolution

```python
# Resolve a location reference by name
result = location.resolve("Cloud Server")

if result.status == ResolutionStatus.RESOLVED:
    print(f"Found: {result.place.name}")
    print(f"Type: {result.place.place_type.name}")
elif result.status == ResolutionStatus.AMBIGUOUS:
    print(f"Multiple matches: {result.candidates}")
elif result.status == ResolutionStatus.NOT_FOUND:
    print("Location not found")
```

### Queries

```python
# List all places of a type
virtual_places = location.list_places_by_type(PlaceType.VIRTUAL)

# Find places matching criteria
matches = location.search_places(
    name_pattern="*Server*",
    place_type=PlaceType.VIRTUAL,
)
```

### Statistics

```python
stats = location.get_stats()
# LocationStats with:
#   total_places, places_by_type
#   total_links
```

## Spatial Model

Places form a graph connected by relationships:

```
Lab ──CONNECTED── Server
 │                  │
 └──CONTAINS── Dataset Storage
```

## Integration

The LocationResolver provides:
- **All subsystems**: Spatial context for operations
- **EvolutionScheduler**: Location-aware evolution

## Example

See `examples/recursive_process_demo.py` for place creation and resolution.
