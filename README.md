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
# Clone the repository
git clone https://github.com/yourorg/organon-noumenon--ontogenetic-morphe.git
cd organon-noumenon--ontogenetic-morphe

# Install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

```bash
# Run the orchestrator
python -m autogenrec.runtime

# Try an example
python examples/full_system_demo.py
```

## Project Structure

```
src/autogenrec/
├── core/           # ProcessLoop, Subsystem, SymbolicValue, Signal, Registry
├── bus/            # MessageBus with wildcard topic matching
├── subsystems/     # 22 subsystems organized by category
│   ├── meta/       # AnthologyManager (the recursive key)
│   ├── core_processing/  # SymbolicInterpreter, RuleCompiler, CodeGenerator
│   ├── conflict/         # ConflictResolver, ArbitrationEngine
│   ├── data/             # ReferenceManager, ArchiveManager, EchoHandler
│   ├── routing/          # NodeRouter, SignalThresholdGuard
│   ├── value/            # ValueExchangeManager, BlockchainSimulator, ProcessMonetizer
│   ├── temporal/         # TimeManager, EvolutionScheduler, LocationResolver
│   ├── identity/         # MaskGenerator, AudienceClassifier
│   ├── transformation/   # ProcessConverter, ConsumptionManager
│   └── academic/         # AcademiaManager
├── storage/        # Persistence layer
└── runtime/        # Orchestrator with graceful start/stop
```

## Architecture

The system follows a 4-phase recursive process loop:

```
Intake → Process → Evaluate → Integrate
   ↑                              │
   └──────── feedback ────────────┘
```

Each subsystem processes symbolic inputs (noumena) through this developmental cycle (ontogenesis), producing emergent forms (morphē) — all orchestrated by the system framework (organon).

## Examples

| Example | Description |
|---------|-------------|
| `basic_narrative_interpretation.py` | Identity masks and audience classification |
| `value_exchange_demo.py` | Value economy with blockchain recording |
| `conflict_resolution_demo.py` | Academic research lifecycle |
| `recursive_process_demo.py` | Evolution cycles with mutations |
| `full_system_demo.py` | Complete research-to-revenue pipeline |

Run any example:
```bash
python examples/full_system_demo.py
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_academia_manager.py

# Run with coverage report
pytest tests/ --cov=src/autogenrec --cov-report=html

# Run integration tests only
pytest tests/test_integration.py -v
```

## Type Checking

```bash
mypy src/
```

## Specifications

Each subsystem has a corresponding specification in `specs/`:

- Consistent 4-phase process loop (Intake → Process → Evaluate → Integrate)
- Domain models and enums
- Public API documentation
- Usage examples

## Contributing

### Development Setup

1. Fork the repository
2. Clone your fork
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Create a feature branch: `git checkout -b feature/your-feature`

### Code Style

- Use type hints for all function signatures
- Follow PEP 8 style guidelines
- Use frozen Pydantic models for domain objects
- Implement the 4-phase pattern for new subsystems

### Pull Request Process

1. Ensure all tests pass: `pytest tests/`
2. Ensure type checking passes: `mypy src/`
3. Update specs if modifying subsystem APIs
4. Add tests for new functionality
5. Update CLAUDE.md if adding new subsystems

### Adding a New Subsystem

1. Create module in appropriate `subsystems/` subdirectory
2. Extend `Subsystem` base class with 4-phase methods
3. Define domain models as frozen Pydantic BaseModel
4. Register in orchestrator's `create_default_orchestrator()`
5. Create corresponding spec in `specs/`
6. Add comprehensive tests
7. Add example usage if appropriate

## License

[License information here]
