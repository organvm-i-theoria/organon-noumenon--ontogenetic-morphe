---
title: CodeGenerator
system: Recursive–Generative Organizational Body
type: subsystem
category: core_processing
tags: [code, generation, transformation, executable]
dependencies: [ProcessConverter]
---

# CodeGenerator

The **CodeGenerator** transforms symbolic structures into executable instructions, generating code in multiple languages from registered structures.

## Overview

| Property | Value |
|----------|-------|
| Category | Core Processing |
| Module | `autogenrec.subsystems.core_processing.code_generator` |
| Dependencies | ProcessConverter |

## Domain Models

### Enums

```python
class OutputLanguage(Enum):
    PYTHON = auto()        # Python code
    JAVASCRIPT = auto()    # JavaScript code
    TYPESCRIPT = auto()    # TypeScript code
    JSON = auto()          # JSON configuration
    YAML = auto()          # YAML configuration

class GenerationStatus(Enum):
    SUCCESS = auto()
    PARTIAL = auto()       # Partial generation
    FAILED = auto()
```

### Core Models

- **CodeStructure**: Registered structure with functions, classes, definition
- **GeneratedCode**: Generated code with language, content
- **GenerationResult**: Result with code, status, errors

## Process Loop

1. **Intake**: Receive structure definitions, generation requests
2. **Process**: Parse structure, apply code templates
3. **Evaluate**: Validate generated code, check syntax
4. **Integrate**: Return generated code, emit events

## Public API

### Structure Registration

```python
from autogenrec.subsystems.core_processing.code_generator import (
    CodeGenerator, OutputLanguage
)

generator = CodeGenerator()

# Register a code structure
structure = generator.register_structure(
    name="symboliq_sdk",
    structure_type="module",
    definition={
        "functions": ["recognize_pattern", "extract_symbols", "validate_input"],
        "classes": ["SymboliQClient", "PatternResult"],
    },
    inputs=["api_key", "data"],
    outputs=["PatternResult"],
)
```

### Code Generation

```python
# Generate Python code
result = generator.generate(structure.id, OutputLanguage.PYTHON)

if result.success:
    print(result.code.code)
    # Output: Generated Python module with functions and classes

# Generate TypeScript
ts_result = generator.generate(structure.id, OutputLanguage.TYPESCRIPT)

# Generate JSON config
json_result = generator.generate(structure.id, OutputLanguage.JSON)
```

### Structure Queries

```python
# Get structure by ID
structure = generator.get_structure(structure_id)

# List all structures
structures = generator.list_structures()
```

### Statistics

```python
stats = generator.get_stats()
# GeneratorStats with:
#   total_structures
#   total_generated
#   generations_by_language
```

## Supported Languages

| Language | Output Type |
|----------|-------------|
| PYTHON | .py module |
| JAVASCRIPT | .js module |
| TYPESCRIPT | .ts module |
| JSON | .json config |
| YAML | .yaml config |

## Generated Code Pattern

For a registered structure, generated code includes:
- **Functions**: Stub implementations
- **Classes**: Class definitions with methods
- **Imports**: Required imports
- **Documentation**: Docstrings and comments

## Integration

The CodeGenerator works with:
- **ProcessConverter**: Input structure conversion
- **SymbolicInterpreter**: Interpreting symbolic inputs

## Example

See `examples/full_system_demo.py` for SDK code generation from registered structures.
