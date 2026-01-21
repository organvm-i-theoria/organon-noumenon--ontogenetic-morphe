---
title: AnthologyManager
system: Recursive–Generative Organizational Body
type: meta-system
category: meta
tags: [registry, meta, process, index]
dependencies: []
---

# AnthologyManager

The **AnthologyManager** is the meta-system that maintains a registry of all subsystems, processes, and artifacts. It serves as the map and key for the entire architecture.

## Overview

| Property | Value |
|----------|-------|
| Category | Meta-System |
| Module | `autogenrec.subsystems.meta.anthology_manager` |
| Dependencies | None (all subsystems depend on it) |

## Domain Models

### Core Models

- **ProcessEntry**: Registry entry for a subsystem process
- **ArtifactEntry**: Registry entry for an artifact
- **QueryResult**: Result of cross-system query

## Process Loop

1. **Intake**: Collect outputs and state data from all subsystems
2. **Process**: Normalize data into consistent schema entries
3. **Evaluate**: Assign identifiers, validate metadata
4. **Integrate**: Provide search and query capabilities

## Registered Subsystems

The AnthologyManager registers and indexes all 21 subsystems:

### Core Processing
| Subsystem | Function |
|-----------|----------|
| SymbolicInterpreter | Interprets dreams, symbols, narratives |
| RuleCompiler | Compiles and validates symbolic rules |
| CodeGenerator | Transforms structures into executable code |

### Conflict & Resolution
| Subsystem | Function |
|-----------|----------|
| ConflictResolver | Detects and resolves conflicts |
| ArbitrationEngine | Structured dispute resolution |

### Data & Records
| Subsystem | Function |
|-----------|----------|
| ReferenceManager | Maintains canonical references |
| ArchiveManager | Preserves and retrieves records |
| EchoHandler | Processes and replays signals |

### Routing & Communication
| Subsystem | Function |
|-----------|----------|
| NodeRouter | Manages node connections and routing |
| SignalThresholdGuard | Validates signals across boundaries |

### Value & Exchange
| Subsystem | Function |
|-----------|----------|
| ValueExchangeManager | Symbolic trade and value exchange |
| BlockchainSimulator | Distributed ledger simulation |
| ProcessMonetizer | Converts processes to revenue |

### Temporal & Spatial
| Subsystem | Function |
|-----------|----------|
| TimeManager | Time functions and scheduling |
| LocationResolver | Spatial reference resolution |
| EvolutionScheduler | Growth and mutation cycles |

### Identity & Classification
| Subsystem | Function |
|-----------|----------|
| MaskGenerator | Creates symbolic identity masks |
| AudienceClassifier | Categorizes users into segments |

### Transformation
| Subsystem | Function |
|-----------|----------|
| ProcessConverter | Transforms workflows |
| ConsumptionManager | Monitors consumption events |

### Academic
| Subsystem | Function |
|-----------|----------|
| AcademiaManager | Learning, research, publication cycles |

## Public API

### Process Registry

```python
from autogenrec.subsystems.meta.anthology_manager import AnthologyManager

anthology = AnthologyManager()

# Register a process
entry = await anthology.register_process(
    name="custom_processor",
    subsystem="transformation",
    inputs=["raw_data"],
    outputs=["processed_data"],
    metadata={"version": "1.0"},
)

# Get process by name
process = await anthology.get_process("custom_processor")

# List all processes
processes = await anthology.list_processes()
```

### Cross-System Queries

```python
# Search across subsystems
results = await anthology.query(
    search="pattern recognition",
    subsystems=["core_processing", "transformation"],
)

# Get subsystem dependencies
deps = await anthology.get_dependencies("AcademiaManager")
```

### System Map

```python
# Export full system map
system_map = await anthology.export_map()

# Get subsystem graph
graph = await anthology.get_subsystem_graph()
```

### Statistics

```python
stats = anthology.get_stats()
# AnthologyStats with:
#   total_subsystems, total_processes
#   total_artifacts
```

## Architecture Role

The AnthologyManager is the **recursive key**:

```
┌─────────────────────────────────────────┐
│           AnthologyManager              │
│  ┌─────────────────────────────────┐   │
│  │       Process Registry          │   │
│  │  ┌─────┐ ┌─────┐ ┌─────┐       │   │
│  │  │ Sub │ │ Sub │ │ Sub │ ...   │   │
│  │  │  1  │ │  2  │ │  3  │       │   │
│  │  └─────┘ └─────┘ └─────┘       │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │     Cross-Query Engine          │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │      Export Pipeline            │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

## Components

- **ProcessRegistry**: Stores definitions of all subsystem processes
- **CrossQueryEngine**: Enables navigation across subsystems
- **ExportPipeline**: Packages the full system map for external use

## Example

See `examples/full_system_demo.py` for comprehensive system usage spanning multiple subsystems.
