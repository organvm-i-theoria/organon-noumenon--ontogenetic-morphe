---
title: ProcessConverter
system: Recursive–Generative Organizational Body
type: subsystem
category: transformation
tags: [conversion, transformation, formats, processes]
dependencies: []
---

# ProcessConverter

The **ProcessConverter** transforms workflows into derivative outputs, converting processes between different formats and representations.

## Overview

| Property | Value |
|----------|-------|
| Category | Transformation |
| Module | `autogenrec.subsystems.transformation.process_converter` |
| Dependencies | None |

## Domain Models

### Enums

```python
class ConversionFormat(Enum):
    JSON = auto()          # JSON representation
    YAML = auto()          # YAML representation
    SCHEMA = auto()        # JSON Schema
    GRAPH = auto()         # Graph representation
    EXECUTABLE = auto()    # Executable form

class ConversionStatus(Enum):
    SUCCESS = auto()
    PARTIAL = auto()       # Partial conversion
    FAILED = auto()
```

### Core Models

- **RegisteredProcess**: Process definition with steps, inputs, outputs
- **ConversionResult**: Result with status, output, errors
- **ConvertedOutput**: Output in target format with content

## Process Loop

1. **Intake**: Receive process definitions, conversion requests
2. **Process**: Parse input, apply conversion rules
3. **Evaluate**: Validate output, check completeness
4. **Integrate**: Return converted output, emit events

## Public API

### Process Registration

```python
from autogenrec.subsystems.transformation.process_converter import (
    ProcessConverter, ConversionFormat
)

converter = ProcessConverter()

# Register a process
process = converter.register_process(
    name="data_pipeline",
    steps=[
        {"name": "intake", "action": "receive_data"},
        {"name": "process", "action": "transform_data"},
        {"name": "evaluate", "action": "validate_output"},
        {"name": "integrate", "action": "emit_results"},
    ],
    inputs=["raw_data"],
    outputs=["processed_data", "metrics"],
    metadata={"version": "1.0"},
)
```

### Conversion

```python
# Convert to JSON
json_result = converter.convert(process.id, ConversionFormat.JSON)
if json_result.success:
    print(json_result.output.content)

# Convert to YAML
yaml_result = converter.convert(process.id, ConversionFormat.YAML)

# Convert to schema
schema_result = converter.convert(process.id, ConversionFormat.SCHEMA)
```

### Process Queries

```python
# Get process by ID
process = converter.get_process(process_id)

# List all processes
processes = converter.list_processes()

# Search by name
matches = converter.search_processes(name_pattern="*pipeline*")
```

### Statistics

```python
stats = converter.get_stats()
# ConverterStats with:
#   total_processes
#   total_conversions
#   conversions_by_format
```

## Supported Formats

| Format | Description |
|--------|-------------|
| JSON | Standard JSON object |
| YAML | Human-readable YAML |
| SCHEMA | JSON Schema for validation |
| GRAPH | Node/edge graph structure |
| EXECUTABLE | Runnable form |

## Integration

The ProcessConverter works with:
- **EvolutionScheduler**: Converting evolved patterns
- **CodeGenerator**: Input for code generation

## Example

See `examples/recursive_process_demo.py` for process registration and format conversion.
