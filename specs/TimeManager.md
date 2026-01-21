---
title: TimeManager
system: Recursive–Generative Organizational Body
type: subsystem
category: temporal
tags: [time, scheduling, cycles, events]
dependencies: []
---

# TimeManager

The **TimeManager** governs time functions, recursive scheduling, cycle tracking, and event triggers within the system.

## Overview

| Property | Value |
|----------|-------|
| Category | Temporal & Spatial |
| Module | `autogenrec.subsystems.temporal.time_manager` |
| Dependencies | None |

## Domain Models

### Enums

```python
class CycleType(Enum):
    DAILY = auto()         # Daily cycle
    WEEKLY = auto()        # Weekly cycle
    MONTHLY = auto()       # Monthly cycle
    SEASONAL = auto()      # Seasonal (quarterly)
    ANNUAL = auto()        # Annual cycle
    CUSTOM = auto()        # Custom interval

class EventStatus(Enum):
    SCHEDULED = auto()     # Waiting to fire
    TRIGGERED = auto()     # Has been triggered
    CANCELLED = auto()     # Cancelled before firing
    EXPIRED = auto()       # Past due, not triggered
```

### Core Models

- **ScheduledEvent**: Event with trigger time, callback, recurrence
- **Cycle**: Recurring time cycle with period and phase
- **TimePoint**: Specific moment with context

## Process Loop

1. **Intake**: Receive scheduling requests, cycle definitions
2. **Process**: Calculate trigger times, manage cycle phases
3. **Evaluate**: Check for due events, validate schedules
4. **Integrate**: Fire events, advance cycles, emit notifications

## Public API

### Event Scheduling

```python
from autogenrec.subsystems.temporal.time_manager import (
    TimeManager, CycleType
)
from datetime import datetime, timedelta, UTC

time_mgr = TimeManager()

# Schedule a one-time event
event = time_mgr.schedule_event(
    name="project_deadline",
    trigger_at=datetime.now(UTC) + timedelta(days=7),
    data={"project_id": "proj_001"},
)

# Schedule recurring event
daily_event = time_mgr.schedule_recurring(
    name="daily_backup",
    cycle_type=CycleType.DAILY,
    start_at=datetime.now(UTC),
    data={"action": "backup"},
)
```

### Cycle Management

```python
# Create a cycle
cycle = time_mgr.create_cycle(
    name="quarterly_review",
    cycle_type=CycleType.SEASONAL,
    start_at=datetime.now(UTC),
)

# Advance cycle phase
time_mgr.advance_cycle(cycle.id)

# Get current phase
phase = time_mgr.get_cycle_phase(cycle.id)
```

### Event Management

```python
# Get pending events
pending = time_mgr.get_pending_events()

# Cancel event
time_mgr.cancel_event(event.id)

# Manually trigger event
time_mgr.trigger_event(event.id)

# Check and fire due events
fired = time_mgr.process_due_events()
```

### Statistics

```python
stats = time_mgr.get_stats()
# TimeStats with:
#   total_events, pending_events, triggered_events
#   total_cycles, active_cycles
```

## Cycle Types

| Type | Period | Use Case |
|------|--------|----------|
| DAILY | 24 hours | Daily tasks |
| WEEKLY | 7 days | Weekly reports |
| MONTHLY | ~30 days | Monthly billing |
| SEASONAL | ~90 days | Quarterly reviews |
| ANNUAL | ~365 days | Annual planning |
| CUSTOM | User-defined | Any interval |

## Integration

The TimeManager provides scheduling for:
- **EvolutionScheduler**: Timing mutation cycles
- **AcademiaManager**: Research deadlines
- **All subsystems**: Event-driven coordination

## Example

See `examples/recursive_process_demo.py` for time-based cycle management.
