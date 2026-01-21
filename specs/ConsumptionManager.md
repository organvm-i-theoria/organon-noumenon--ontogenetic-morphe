---
title: ConsumptionManager
system: Recursive–Generative Organizational Body
type: subsystem
category: transformation
tags: [consumption, quotas, resources, usage]
dependencies: []
---

# ConsumptionManager

The **ConsumptionManager** monitors and manages consumption events, enforcing quotas and tracking resource usage across the system.

## Overview

| Property | Value |
|----------|-------|
| Category | Transformation |
| Module | `autogenrec.subsystems.transformation.consumption_manager` |
| Dependencies | None |

## Domain Models

### Enums

```python
class ResourceType(Enum):
    COMPUTE = auto()       # Compute resources
    STORAGE = auto()       # Storage space
    BANDWIDTH = auto()     # Network bandwidth
    API_CALL = auto()      # API invocations
    TOKEN = auto()         # System tokens # allow-secret
    CUSTOM = auto()        # Custom resource

class ConsumptionStatus(Enum):
    APPROVED = auto()      # Consumption allowed
    DENIED = auto()        # Quota exceeded
    PARTIAL = auto()       # Partially fulfilled
    PENDING = auto()       # Awaiting approval
```

### Core Models

- **ConsumptionEvent**: Request to consume resources
- **ConsumptionResult**: Outcome with status and remaining quota
- **Quota**: Resource limit for a consumer

## Process Loop

1. **Intake**: Receive consumption requests
2. **Process**: Check quotas, calculate availability
3. **Evaluate**: Approve or deny based on limits
4. **Integrate**: Update usage, emit consumption events

## Public API

### Quota Management

```python
from autogenrec.subsystems.transformation.consumption_manager import (
    ConsumptionManager, ResourceType
)
from decimal import Decimal

consumption = ConsumptionManager()

# Add quota for a consumer
consumption.add_quota(
    name="user_api_quota",
    resource_type=ResourceType.API_CALL,
    max_amount=Decimal("100"),  # 100 API calls
    consumer_id="user_001",
)

# Add compute quota
consumption.add_quota(
    name="evolution_compute",
    resource_type=ResourceType.COMPUTE,
    max_amount=Decimal("1000"),
    consumer_id="evolution_system",
)
```

### Consumption

```python
# Create consumption event
event = consumption.create_event(
    consumer_id="user_001",
    resource_type=ResourceType.API_CALL,
    amount=Decimal("1"),
    context="symboliq_api",
)

# Attempt to consume
result = consumption.consume(event)

if result.approved:
    print("Consumption approved")
    print(f"Remaining: {result.remaining}")
else:
    print("Quota exceeded")
```

### Quota Checks

```python
# Check if consumption is allowed
allowed, remaining = consumption.check_quota(
    consumer_id="user_001",
    resource_type=ResourceType.API_CALL,
    amount=Decimal("10"),
)

print(f"Allowed: {allowed}")
print(f"Remaining: {remaining}")
```

### Quota Management

```python
# Get current usage
usage = consumption.get_usage(
    consumer_id="user_001",
    resource_type=ResourceType.API_CALL,
)

# Reset quota (e.g., at period start)
consumption.reset_quota(
    consumer_id="user_001",
    resource_type=ResourceType.API_CALL,
)

# Update quota limit
consumption.update_quota(
    consumer_id="user_001",
    resource_type=ResourceType.API_CALL,
    new_max=Decimal("200"),
)
```

### Statistics

```python
stats = consumption.get_stats()
# ConsumptionStats with:
#   total_events
#   consumed_count (approved)
#   rejected_count (denied)
```

## Quota Model

Each quota tracks:
- **max_amount**: Total allowed consumption
- **consumed**: Amount already used
- **remaining**: Available quota

Quotas can be:
- Per-consumer (user-specific limits)
- Global (system-wide limits)
- Per-resource-type (different limits for different resources)

## Integration

The ConsumptionManager works with:
- **ProcessMonetizer**: Metered billing
- **EvolutionScheduler**: Resource limits for evolution

## Example

See `examples/recursive_process_demo.py` for quota setup and consumption tracking.
