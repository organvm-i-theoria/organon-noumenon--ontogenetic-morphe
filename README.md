# organon-noumenon--ontogenetic-morphe

*An instrument for symbolic essences, producing developmental forms.*

## Overview

**organon-noumenon--ontogenetic-morphe** implements a symbolic processing system with 22 interconnected subsystems following the recursive-generative architectural pattern.

| Term | Meaning | Maps To |
|------|---------|---------|
| **Organon** | Instrument of reasoning | ProcessLoop, MessageBus — the system framework |
| **Noumenon** | Thing-in-itself | SymbolicValue — data carrying meaning beyond raw values |
| **Ontogenetic** | Being-developing | Recursive 4-phase loops with feedback |
| **Morphē** | Form | Emergent patterns, schemas, structures |

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```bash
python -m autogenrec.runtime
```

## Project Structure

```
src/autogenrec/
├── core/           # ProcessLoop, Subsystem, SymbolicValue, Signal, Registry
├── bus/            # MessageBus with wildcard topic matching
├── subsystems/     # 22 subsystems organized by category
│   ├── meta/       # AnthologyManager (the recursive key)
│   ├── core_processing/
│   ├── conflict/
│   ├── data/
│   ├── routing/
│   ├── value/
│   ├── temporal/
│   ├── identity/
│   ├── transformation/
│   └── academic/
├── storage/        # Persistence layer
└── runtime/        # Orchestrator with graceful start/stop
```

## Testing

```bash
pytest tests/
```

## Type Checking

```bash
mypy src/
```

## Architecture

The system follows a 4-phase recursive process loop:

```
Intake → Process → Evaluate → Integrate
   ↑                              │
   └──────── feedback ────────────┘
```

Each subsystem processes symbolic inputs (noumena) through this developmental cycle (ontogenesis), producing emergent forms (morphē) — all orchestrated by the system framework (organon).
