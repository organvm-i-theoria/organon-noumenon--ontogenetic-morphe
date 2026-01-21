---
title: EchoHandler
system: Recursive–Generative Organizational Body
type: subsystem
category: data
tags: [echo, signal, replay, decay]
dependencies: [SignalThresholdGuard]
---

# EchoHandler

The **EchoHandler** processes and replays signals, managing signal capture, replay scheduling, and decay over time.

## Overview

| Property | Value |
|----------|-------|
| Category | Data & Records |
| Module | `autogenrec.subsystems.data.echo_handler` |
| Dependencies | SignalThresholdGuard |

## Domain Models

### Enums

```python
class EchoState(Enum):
    CAPTURED = auto()      # Signal captured
    QUEUED = auto()        # Ready for replay
    REPLAYING = auto()     # Currently replaying
    DECAYED = auto()       # Signal has decayed
    EXPIRED = auto()       # Past replay window

class DecayRate(Enum):
    NONE = auto()          # No decay
    SLOW = auto()          # Gradual decay
    MEDIUM = auto()        # Moderate decay
    FAST = auto()          # Rapid decay
```

### Core Models

- **Echo**: Captured signal with original content, decay state
- **ReplaySchedule**: Scheduled replay with timing
- **DecayProfile**: Decay configuration

## Process Loop

1. **Intake**: Capture signals for echo processing
2. **Process**: Apply decay, schedule replays
3. **Evaluate**: Check signal strength, validate replay conditions
4. **Integrate**: Execute replays, emit echo events

## Public API

### Signal Capture

```python
from autogenrec.subsystems.data.echo_handler import (
    EchoHandler, DecayRate
)

handler = EchoHandler()

# Capture a signal for echo
echo = await handler.capture(
    signal=original_signal,
    decay_rate=DecayRate.SLOW,
    max_replays=3,
)
```

### Replay

```python
# Schedule a replay
schedule = await handler.schedule_replay(
    echo_id=echo.id,
    delay_seconds=60,
)

# Immediate replay
replayed = await handler.replay(echo.id)

# Get replay history
history = await handler.get_replay_history(echo.id)
```

### Decay Management

```python
# Check current signal strength
strength = handler.get_signal_strength(echo.id)

# Apply decay manually
await handler.apply_decay(echo.id)

# Process all decayed echoes
decayed = await handler.process_decayed()
```

### Statistics

```python
stats = handler.get_stats()
# EchoStats with:
#   total_echoes, active_echoes
#   total_replays
#   echoes_by_state
```

## Decay Model

Signals decay over time, reducing strength:

```
Strength: 1.0 → 0.8 → 0.6 → 0.4 → 0.2 → 0.0 (decayed)
```

| Decay Rate | Half-life |
|------------|-----------|
| NONE | Infinite |
| SLOW | Long |
| MEDIUM | Moderate |
| FAST | Short |

## Integration

The EchoHandler works with:
- **SignalThresholdGuard**: Signal validation
- **NodeRouter**: Signal distribution

## Example

The EchoHandler manages signal echoes for distributed processing scenarios.
