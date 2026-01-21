---
title: RuleCompiler
system: Recursive–Generative Organizational Body
type: subsystem
category: core_processing
tags: [rules, compilation, validation, executable]
dependencies: []
---

# RuleCompiler

The **RuleCompiler** compiles and validates symbolic rules, transforming rule definitions into executable form.

## Overview

| Property | Value |
|----------|-------|
| Category | Core Processing |
| Module | `autogenrec.subsystems.core_processing.rule_compiler` |
| Dependencies | None |

## Domain Models

### Enums

```python
class RuleType(Enum):
    INTERPRETATION = auto()    # Interpretation rules
    TRANSFORMATION = auto()    # Transformation rules
    VALIDATION = auto()        # Validation rules
    ROUTING = auto()           # Routing rules

class CompilationStatus(Enum):
    SUCCESS = auto()           # Compiled successfully
    WARNING = auto()           # Compiled with warnings
    ERROR = auto()             # Compilation failed

class ValidationLevel(Enum):
    SYNTAX = auto()            # Syntax validation only
    SEMANTIC = auto()          # Semantic validation
    FULL = auto()              # Full validation
```

### Core Models

- **Rule**: Rule definition with conditions, actions
- **CompiledRule**: Executable compiled form
- **CompilationResult**: Compilation outcome with errors/warnings

## Process Loop

1. **Intake**: Receive rule definitions
2. **Process**: Parse and compile rules
3. **Evaluate**: Validate syntax and semantics
4. **Integrate**: Return compiled rules, emit compilation events

## Public API

### Rule Definition

```python
from autogenrec.subsystems.core_processing.rule_compiler import (
    RuleCompiler, RuleType, ValidationLevel
)

compiler = RuleCompiler()

# Define a rule
rule = await compiler.define_rule(
    name="transform_symbol",
    rule_type=RuleType.TRANSFORMATION,
    condition="input.type == 'symbol'",
    action="return transform(input.value)",
)
```

### Compilation

```python
# Compile a single rule
result = await compiler.compile(rule.id)

if result.status == CompilationStatus.SUCCESS:
    compiled = result.compiled_rule
    # Use compiled rule
elif result.status == CompilationStatus.WARNING:
    # Handle warnings
    for warning in result.warnings:
        print(f"Warning: {warning}")
else:
    # Handle errors
    for error in result.errors:
        print(f"Error: {error}")
```

### Validation

```python
# Validate rule before compilation
validation = await compiler.validate(
    rule.id,
    level=ValidationLevel.FULL,
)

if validation.is_valid:
    # Proceed with compilation
    pass
```

### Rule Execution

```python
# Execute compiled rule
output = await compiler.execute(
    compiled_rule_id,
    input_data=symbolic_input,
)
```

### Statistics

```python
stats = compiler.get_stats()
# CompilerStats with:
#   total_rules, compiled_rules
#   compilation_errors, compilation_warnings
```

## Compilation Pipeline

```
Rule Definition
      │
      ▼
┌─────────────┐
│   Parse     │──► Syntax Tree
└─────────────┘
      │
      ▼
┌─────────────┐
│  Validate   │──► Validation Result
└─────────────┘
      │
      ▼
┌─────────────┐
│  Compile    │──► Executable Rule
└─────────────┘
```

## Integration

The RuleCompiler provides:
- **SymbolicInterpreter**: Interpretation rules
- **ConflictResolver**: Resolution rules
- **All subsystems**: Rule-based logic

## Example

The RuleCompiler transforms symbolic rules into executable form for system-wide use.
