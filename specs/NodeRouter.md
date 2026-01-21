---
title: NodeRouter
system: Recursive–Generative Organizational Body
type: subsystem
category: routing
tags: [routing, nodes, broadcast, connections]
dependencies: []
---

# NodeRouter

The **NodeRouter** manages connections and routing between symbolic nodes, providing route management and broadcast capabilities.

## Overview

| Property | Value |
|----------|-------|
| Category | Routing & Communication |
| Module | `autogenrec.subsystems.routing.node_router` |
| Dependencies | None |

## Domain Models

### Enums

```python
class RouteType(Enum):
    DIRECT = auto()        # Point-to-point
    BROADCAST = auto()     # One-to-many
    MULTICAST = auto()     # Selected targets
    CONDITIONAL = auto()   # Rule-based routing

class RouteStatus(Enum):
    ACTIVE = auto()        # Route is active
    INACTIVE = auto()      # Route is disabled
    DEGRADED = auto()      # Partial functionality
```

### Core Models

- **Route**: Route definition with source, target, type
- **Node**: Routing node with connections
- **RoutingResult**: Result of routing operation

## Process Loop

1. **Intake**: Receive routing requests, node registrations
2. **Process**: Compute routes, manage connections
3. **Evaluate**: Validate paths, check connectivity
4. **Integrate**: Execute routing, emit routing events

## Public API

### Node Management

```python
from autogenrec.subsystems.routing.node_router import (
    NodeRouter, RouteType
)

router = NodeRouter()

# Register nodes
source_node = await router.register_node(
    name="processor",
    metadata={"type": "worker"},
)

target_node = await router.register_node(
    name="aggregator",
    metadata={"type": "collector"},
)
```

### Route Management

```python
# Add direct route
route = await router.add_route(
    source_id=source_node.id,
    target_id=target_node.id,
    route_type=RouteType.DIRECT,
)

# Get route
route = await router.get_route(source_id, target_id)

# Remove route
await router.remove_route(route.id)
```

### Message Routing

```python
# Route a message
result = await router.route(
    source_id=source_node.id,
    message=symbolic_message,
)

# Broadcast to all connected nodes
results = await router.broadcast(
    source_id=source_node.id,
    message=broadcast_message,
)
```

### Statistics

```python
stats = router.get_stats()
# RouterStats with:
#   total_nodes, active_nodes
#   total_routes, active_routes
#   messages_routed
```

## Routing Topology

Nodes form a directed graph:

```
      ┌─────────┐
      │ Source  │
      └────┬────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌───────┐    ┌───────┐
│Target1│    │Target2│
└───────┘    └───────┘
```

## Integration

The NodeRouter provides:
- **All subsystems**: Inter-node communication
- **EchoHandler**: Signal distribution

## Example

The NodeRouter manages distributed processing topology for the system.
