# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**organon-noumenon--ontogenetic-morphe** is a symbolic processing system implementing the Recursive–Generative Organizational Body architecture. It provides both markdown-based architectural specifications and a Python implementation with 22 interconnected subsystems.

The name reflects the architecture:
- **Organon** (instrument) → The system framework (ProcessLoop, MessageBus)
- **Noumenon** (thing-in-itself) → SymbolicValues carrying meaning beyond raw data
- **Ontogenetic** (being-developing) → Recursive 4-phase process loops
- **Morphē** (form) → Emergent patterns, schemas, structures

## Architecture

### 4-Phase Process Loop

Every subsystem follows the recursive 4-phase pattern:

```
┌─────────┐    ┌─────────┐    ┌──────────┐    ┌───────────┐
│ INTAKE  │───►│ PROCESS │───►│ EVALUATE │───►│ INTEGRATE │
└─────────┘    └─────────┘    └──────────┘    └─────┬─────┘
     ▲                                              │
     │              feedback loop                   │
     └──────────────────────────────────────────────┘
```

### Subsystem Dependency Graph

```
                    ┌─────────────────┐
                    │ AnthologyManager│ (meta-system)
                    └────────┬────────┘
                             │ registers all
    ┌────────────────────────┼────────────────────────┐
    │                        │                        │
    ▼                        ▼                        ▼
┌───────────┐        ┌───────────────┐        ┌─────────────┐
│ Identity  │        │ Core Process  │        │   Value     │
├───────────┤        ├───────────────┤        ├─────────────┤
│ Mask      │        │ Symbolic      │        │ Exchange    │
│ Generator │        │ Interpreter   │───────►│ Manager     │
│     │     │        │      │        │        │      │      │
│     ▼     │        │      ▼        │        │      ▼      │
│ Audience  │        │ Rule          │        │ Blockchain  │
│ Classifier│        │ Compiler      │───────►│ Simulator   │
└───────────┘        │      │        │        │      │      │
                     │      ▼        │        │      ▼      │
                     │ Code          │        │ Process     │
                     │ Generator     │        │ Monetizer   │
                     └───────────────┘        └─────────────┘

┌───────────┐        ┌───────────────┐        ┌─────────────┐
│ Temporal  │        │    Data       │        │  Routing    │
├───────────┤        ├───────────────┤        ├─────────────┤
│ Time      │        │ Reference     │        │ Node        │
│ Manager   │───────►│ Manager       │        │ Router      │
│     │     │        │      │        │        │      │      │
│     ▼     │        │      ▼        │        │      ▼      │
│ Evolution │        │ Archive       │        │ Signal      │
│ Scheduler │        │ Manager       │        │ Threshold   │
│     │     │        │      │        │        │ Guard       │
│     ▼     │        │      ▼        │        └─────────────┘
│ Location  │        │ Echo          │
│ Resolver  │        │ Handler       │
└───────────┘        └───────────────┘

┌───────────────┐    ┌───────────────┐
│ Transformation│    │   Conflict    │
├───────────────┤    ├───────────────┤
│ Process       │    │ Conflict      │
│ Converter     │    │ Resolver      │
│      │        │    │      │        │
│      ▼        │    │      ▼        │
│ Consumption   │    │ Arbitration   │
│ Manager       │    │ Engine        │
└───────────────┘    └───────────────┘

┌───────────────┐
│   Academic    │
├───────────────┤
│ Academia      │
│ Manager       │
└───────────────┘
```

## Repository Structure

```
├── specs/              # Markdown architectural specifications (22 files)
├── src/autogenrec/     # Python implementation
│   ├── core/           # ProcessLoop, Subsystem, SymbolicValue, Signal, Registry
│   ├── bus/            # MessageBus with wildcard topic matching
│   ├── subsystems/     # 22 subsystems organized by category
│   │   ├── meta/           # AnthologyManager
│   │   ├── core_processing/# SymbolicInterpreter, RuleCompiler, CodeGenerator
│   │   ├── conflict/       # ConflictResolver, ArbitrationEngine
│   │   ├── data/           # ReferenceManager, ArchiveManager, EchoHandler
│   │   ├── routing/        # NodeRouter, SignalThresholdGuard
│   │   ├── value/          # ValueExchangeManager, BlockchainSimulator, ProcessMonetizer
│   │   ├── temporal/       # TimeManager, EvolutionScheduler, LocationResolver
│   │   ├── identity/       # MaskGenerator, AudienceClassifier
│   │   ├── transformation/ # ProcessConverter, ConsumptionManager
│   │   └── academic/       # AcademiaManager
│   ├── storage/        # Persistence layer (stub)
│   └── runtime/        # Orchestrator
├── tests/              # pytest tests
└── examples/           # Usage examples (5 demos)
```

## Working with the Codebase

### Implementation Pattern

Each subsystem follows this structure:

```python
# 1. Domain Models (frozen Pydantic)
class MyModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str = Field(default_factory=lambda: str(ULID()))
    # ... fields

# 2. Enums for type safety
class MyStatus(Enum):
    ACTIVE = auto()
    INACTIVE = auto()

# 3. Subsystem class
class MySubsystem(Subsystem):
    def __init__(self):
        super().__init__(SubsystemMetadata(...))
        self._state: dict[str, MyModel] = {}
    
    # 4. Four-phase implementation
    async def intake(self, input: SymbolicInput, ctx: ProcessContext) -> Any:
        # Validate, filter, log
        pass
    
    async def process(self, data: Any, ctx: ProcessContext) -> Any:
        # Core transformation logic
        pass
    
    async def evaluate(self, result: Any, ctx: ProcessContext) -> tuple[Any, bool]:
        # Synthesize output, decide continuation
        pass
    
    async def integrate(self, output: Any, ctx: ProcessContext) -> SymbolicOutput:
        # Emit events, prepare feedback
        pass
```

### Adding/Modifying Subsystems

1. **Extend the Subsystem base class** — Implement the 4 abstract methods
2. **Define SubsystemMetadata** — Include name, type, tags, input/output types
3. **Create domain models** — Use frozen Pydantic BaseModel
4. **Register in orchestrator** — Add factory to `create_default_orchestrator()`
5. **Update spec file** — Keep specs in sync with implementation
6. **Add tests** — Unit tests for all public methods

### Core Abstractions

- **ProcessLoop** — 4-phase pattern (intake, process, evaluate, integrate)
- **Subsystem** — ProcessLoop + messaging + lifecycle management
- **SymbolicValue** — Typed symbolic data with provenance tracking
- **Signal/Echo** — Inter-subsystem communication with threshold validation
- **MessageBus** — Async pub/sub with wildcard topic matching

### Running the System

```bash
# Install
pip install -e ".[dev]"

# Run orchestrator
python -m autogenrec.runtime

# Run tests
pytest tests/

# Run specific test file
pytest tests/test_academia_manager.py -v

# Run with coverage
pytest tests/ --cov=src/autogenrec

# Type check
mypy src/

# Run examples
python examples/full_system_demo.py
python examples/value_exchange_demo.py
python examples/recursive_process_demo.py
```

## Architecture Principles

- **Recursive**: Outputs feed back as inputs; systems reference themselves
- **Generative**: New patterns emerge from mutation and transformation
- **Symbolic**: All data is treated as symbolic, carrying meaning beyond raw values
- **Process-oriented**: Everything is modeled as a process with inputs, transformations, and outputs
- **Immutable**: Domain models use frozen dataclasses for thread safety

## Examples

The `examples/` directory contains working demonstrations:

| Example | Description |
|---------|-------------|
| `basic_narrative_interpretation.py` | Identity masks and audience classification |
| `value_exchange_demo.py` | Value economy with blockchain recording |
| `conflict_resolution_demo.py` | Academic research lifecycle |
| `recursive_process_demo.py` | Evolution cycles with mutations |
| `full_system_demo.py` | Complete research-to-revenue pipeline |

<!-- ORGANVM:AUTO:START -->
## System Context (auto-generated — do not edit)

**Organ:** ORGAN-I (Theory) | **Tier:** standard | **Status:** CANDIDATE
**Org:** `unknown` | **Repo:** `organon-noumenon--ontogenetic-morphe`

### Edges
- **Produces** → `unknown`: unknown

### Siblings in Theory
`recursive-engine--generative-entity`, `auto-revision-epistemic-engine`, `narratological-algorithmic-lenses`, `call-function--ontological`, `sema-metra--alchemica-mundi`, `system-governance-framework`, `cognitive-archaelogy-tribunal`, `a-recursive-root`, `radix-recursiva-solve-coagula-redi`, `.github`, `nexus--babel-alexandria-`, `reverse-engine-recursive-run`, `4-ivi374-F0Rivi4`, `cog-init-1-0-`, `collective-persona-operations` ... and 4 more

### Governance
- Foundational theory layer. No upstream dependencies.

*Last synced: 2026-02-24T12:41:28Z*
<!-- ORGANVM:AUTO:END -->


## ⚡ Conductor OS Integration
This repository is a managed component of the ORGANVM meta-workspace.
- **Orchestration:** Use `conductor patch` for system status and work queue.
- **Lifecycle:** Follow the `FRAME -> SHAPE -> BUILD -> PROVE` workflow.
- **Governance:** Promotions are managed via `conductor wip promote`.
- **Intelligence:** Conductor MCP tools are available for routing and mission synthesis.
