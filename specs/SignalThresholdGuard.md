---
title: SignalThresholdGuard
system: Recursive–Generative Organizational Body
type: subsystem
category: routing
tags: [signal, threshold, validation, analog, digital]
dependencies: []
---

# SignalThresholdGuard

The **SignalThresholdGuard** validates signals across analog/digital boundaries, enforcing threshold policies and domain conversion.

## Overview

| Property | Value |
|----------|-------|
| Category | Routing & Communication |
| Module | `autogenrec.subsystems.routing.signal_threshold_guard` |
| Dependencies | None |

## Domain Models

### Enums

```python
class SignalDomain(Enum):
    ANALOG = auto()        # Continuous signal
    DIGITAL = auto()       # Discrete signal
    HYBRID = auto()        # Mixed domain

class ThresholdPolicy(Enum):
    STRICT = auto()        # Must exceed threshold
    LENIENT = auto()       # Approximate threshold
    ADAPTIVE = auto()      # Dynamic threshold

class ValidationStatus(Enum):
    VALID = auto()         # Signal passes
    INVALID = auto()       # Signal rejected
    MARGINAL = auto()      # Near threshold
```

### Core Models

- **ThresholdRule**: Threshold definition with policy
- **SignalValidation**: Validation result with status
- **ConversionResult**: Domain conversion result

## Process Loop

1. **Intake**: Receive signals for validation
2. **Process**: Apply threshold rules, perform conversions
3. **Evaluate**: Determine validity, check margins
4. **Integrate**: Pass or reject signals, emit validation events

## Public API

### Threshold Configuration

```python
from autogenrec.subsystems.routing.signal_threshold_guard import (
    SignalThresholdGuard, SignalDomain, ThresholdPolicy
)

guard = SignalThresholdGuard()

# Set threshold rule
guard.set_threshold(
    signal_type="input_signal",
    min_value=0.1,
    max_value=1.0,
    policy=ThresholdPolicy.STRICT,
)
```

### Signal Validation

```python
# Validate a signal
result = await guard.validate(
    signal=input_signal,
    signal_type="input_signal",
)

if result.status == ValidationStatus.VALID:
    # Process signal
    pass
elif result.status == ValidationStatus.MARGINAL:
    # Handle edge case
    pass
else:
    # Reject signal
    pass
```

### Domain Conversion

```python
# Convert analog to digital
digital_signal = await guard.convert(
    signal=analog_signal,
    target_domain=SignalDomain.DIGITAL,
)

# Convert digital to analog
analog_signal = await guard.convert(
    signal=digital_signal,
    target_domain=SignalDomain.ANALOG,
)
```

### Statistics

```python
stats = guard.get_stats()
# GuardStats with:
#   total_validations, valid_count, invalid_count
#   total_conversions
```

## Threshold Model

Signals must pass threshold checks:

```
Signal Value
     │
     │  ┌─────────────────┐
     │  │   VALID RANGE   │
     │  └─────────────────┘
     │    min          max
     └────────────────────────►
```

## Integration

The SignalThresholdGuard provides:
- **EchoHandler**: Signal validation
- **NodeRouter**: Pre-routing validation
- **All subsystems**: Input validation

## Example

The SignalThresholdGuard ensures signal integrity for cross-domain communication.
