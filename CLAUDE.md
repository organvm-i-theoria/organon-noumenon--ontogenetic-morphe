# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**organon-noumenon--ontogenetic-morphe** is a symbolic processing system implementing the Recursive–Generative Organizational Body architecture. It provides both markdown-based architectural specifications and a Python implementation with 22 interconnected subsystems.

The name reflects the architecture:
- **Organon** (instrument) → The system framework (ProcessLoop, MessageBus)
- **Noumenon** (thing-in-itself) → SymbolicValues carrying meaning beyond raw data
- **Ontogenetic** (being-developing) → Recursive 4-phase process loops
- **Morphē** (form) → Emergent patterns, schemas, structures

## Repository Structure

```
├── specs/              # Markdown architectural specifications
├── src/autogenrec/     # Python implementation
│   ├── core/           # ProcessLoop, Subsystem, SymbolicValue, Signal, Registry
│   ├── bus/            # MessageBus with wildcard topic matching
│   ├── subsystems/     # 22 subsystems organized by category
│   ├── storage/        # Persistence layer (stub)
│   └── runtime/        # Orchestrator
├── tests/              # pytest tests
└── examples/           # Usage examples
```

### Meta-System

**AnthologyManager** — The central registry and index for all subsystems. It maintains the unified process registry and serves as the access layer for the entire architecture.

### Subsystem Categories

**Core Processing:**
- SymbolicInterpreter — Interprets symbolic inputs (dreams, narratives, constructed languages)
- RuleCompiler — Compiles and validates symbolic rules
- CodeGenerator — Transforms symbolic structures into executable instructions

**Conflict & Resolution:**
- ConflictResolver — Detects and resolves conflicting inputs
- ArbitrationEngine — Structured dispute resolution

**Data & Records:**
- ReferenceManager — Maintains canonical references
- ArchiveManager — Preserves and retrieves records
- EchoHandler — Processes and replays signals

**Routing & Communication:**
- NodeRouter — Manages connections between symbolic nodes
- SignalThresholdGuard — Validates signals across analog/digital boundaries

**Value & Exchange:**
- ValueExchangeManager — Symbolic trade and value exchange
- BlockchainSimulator — Distributed ledger logic for symbolic economies
- ProcessMonetizer — Converts processes into monetizable outputs

**Temporal & Spatial:**
- TimeManager — Governs time functions and recursive scheduling
- LocationResolver — Resolves spatial references
- EvolutionScheduler — Manages growth and mutation cycles

**Identity & Classification:**
- MaskGenerator — Creates symbolic identity masks
- AudienceClassifier — Categorizes users into segments

**Transformation:**
- ProcessConverter — Transforms workflows into derivative outputs
- ConsumptionManager — Monitors and manages consumption events

**Academic:**
- AcademiaManager — Manages learning, research, and publication cycles

## Working with the Codebase

### Adding/Modifying Subsystems

1. **Extend the Subsystem base class** — Implement the 4 abstract methods: `intake`, `process`, `evaluate`, `integrate`
2. **Define SubsystemMetadata** — Include name, type, tags, input/output types
3. **Register in orchestrator** — Add factory to `create_default_orchestrator()`
4. **Update spec file** — Keep specs in sync with implementation

### Core Abstractions

- **ProcessLoop** — 4-phase pattern with lifecycle hooks
- **Subsystem** — ProcessLoop + messaging + lifecycle management
- **SymbolicValue** — Typed symbolic data with provenance tracking
- **Signal/Echo** — Inter-subsystem communication with threshold validation
- **MessageBus** — Async pub/sub with wildcard topic matching

### Running the System

```bash
# Install
pip install -e ".[dev]"

# Run
python -m autogenrec.runtime

# Test
pytest tests/

# Type check
mypy src/
```

## Architecture Principles

- **Recursive**: Outputs feed back as inputs; systems reference themselves
- **Generative**: New patterns emerge from mutation and transformation
- **Symbolic**: All data is treated as symbolic, carrying meaning beyond raw values
- **Process-oriented**: Everything is modeled as a process with inputs, transformations, and outputs
