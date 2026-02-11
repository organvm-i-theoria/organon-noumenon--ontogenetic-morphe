# Organon Noumenon: Ontogenetic Morphe

[![CI](https://github.com/organvm-i-theoria/organon-noumenon--ontogenetic-morphe/actions/workflows/ci.yml/badge.svg)](https://github.com/organvm-i-theoria/organon-noumenon--ontogenetic-morphe/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-pending-lightgrey)](https://github.com/organvm-i-theoria/organon-noumenon--ontogenetic-morphe)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/organvm-i-theoria/organon-noumenon--ontogenetic-morphe/blob/main/LICENSE)
[![Organ I](https://img.shields.io/badge/Organ-I%20Theoria-8B5CF6)](https://github.com/organvm-i-theoria)
[![Status](https://img.shields.io/badge/status-active-brightgreen)](https://github.com/organvm-i-theoria/organon-noumenon--ontogenetic-morphe)
[![Python](https://img.shields.io/badge/lang-Python-informational)](https://github.com/organvm-i-theoria/organon-noumenon--ontogenetic-morphe)


[![ORGAN-I: Theory](https://img.shields.io/badge/ORGAN--I-Theory-1a237e?style=flat-square)](https://github.com/organvm-i-theoria)
[![Python](https://img.shields.io/badge/python-≥3.11-blue?style=flat-square)]()
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)

**A 22-subsystem symbolic processing architecture implementing recursive-generative organizational bodies through typed symbolic values, async message passing, and four-phase process loops.**

The name encodes the system's theoretical commitment: *Organon* (instrument of thought), *Noumenon* (the thing-in-itself, beyond mere appearance), *Ontogenetic* (the development of being through time), *Morphe* (form as structuring principle). Together: an instrument for apprehending the deep forms through which symbolic entities develop.

> Package: `autogenrec` (AutoGenRec — Recursive-Generative Organizational Body)

---

[Problem Statement](#problem-statement) | [Core Concepts](#core-concepts) | [Architecture](#architecture) | [Installation & Usage](#installation--usage) | [Examples](#examples) | [Downstream Implementation](#downstream-implementation) | [Validation](#validation) | [Roadmap](#roadmap) | [Cross-References](#cross-references) | [Contributing](#contributing) | [License](#license) | [Author & Contact](#author--contact)

---

## Problem Statement

Most symbolic processing frameworks treat symbols as inert data: strings with lookup tables, tokens with embeddings, or nodes in static graphs. They lack a model for how symbolic meaning *develops* — how a narrative fragment accrues provenance, how a rule compiles into executable form, how a dream-symbol decays into echo and is eventually archived or transformed into currency.

Existing approaches fall into predictable traps:

- **Rule engines** (Drools, CLIPS) handle production rules but have no concept of symbolic value, identity, or temporal decay
- **Agent frameworks** (LangChain, CrewAI) orchestrate LLM calls but treat the orchestration layer as mere plumbing, not as a domain with its own semantics
- **Message brokers** (RabbitMQ, Kafka) provide pub/sub infrastructure but impose no type discipline on the symbolic content flowing through them
- **Actor systems** (Akka, Ray) manage concurrency but their actor model does not distinguish between a signal that should attenuate over distance and a message that should persist until consumed

Ontogenetic Morphe addresses the gap between these paradigms. It provides a typed symbolic processing architecture where every value carries provenance and lineage, every signal has physical properties (strength, threshold, attenuation), every subsystem follows the same recursive lifecycle, and the whole system communicates through a single async message bus with topic-based routing.

The core insight is that symbolic processing is not a pipeline but a *metabolism* — a recursive cycle of intake, processing, evaluation, and integration that mirrors biological ontogenesis. Each subsystem is a specialized organ within a larger body, and the body's coherence emerges from the message bus, not from a central controller.

## Core Concepts

### The Four-Phase Process Loop

Every computation in the system follows a single abstract protocol defined by `ProcessLoop[InputT, OutputT, ContextT]`:

```
INTAKE → PROCESS → EVALUATE → INTEGRATE
  ↑                    │
  └────────────────────┘
        (feedback: should_continue)
```

The `evaluate` phase returns a boolean `should_continue` flag, creating a natural recursion: a subsystem keeps cycling until its own evaluation criteria are satisfied. This is not an arbitrary design choice — it models the recursive self-modification that characterizes living systems. A rule compiler does not simply compile; it compiles, evaluates the result, and recompiles if the output fails its own criteria.

The generic type parameters (`InputT`, `OutputT`, `ContextT`) ensure that each subsystem's loop is fully typed. A `SymbolicInterpreter` takes `SymbolicValue` in and produces interpreted structures out; a `CodeGenerator` takes compiled rules in and produces executable artifacts out. The type system prevents cross-wiring at the interface level.

### Symbolic Values and Provenance

`SymbolicValue` is the atomic unit of meaning in the system — a frozen Pydantic model carrying:

- **ULID identity** — globally unique, time-sortable, no coordination required
- **Value type** — one of 23 enumerated kinds: `NARRATIVE`, `DREAM`, `RULE`, `PATTERN`, `TOKEN`, `CURRENCY`, `MASK`, `SIGNAL`, `ECHO`, `REFERENCE`, `ARCHIVE`, `CONFLICT`, `RESOLUTION`, `NODE`, `ROUTE`, `THRESHOLD`, `PROCESS`, `CONSUMPTION`, `EVOLUTION`, `LOCATION`, `AUDIENCE`, `ACADEMIA`, `ANTHOLOGY`
- **Provenance chain** — every value records its origin; the `derive()` method creates a new value linked to its parent, building a directed acyclic graph of symbolic lineage

This is not a tagged union for dispatch convenience. The type system reflects an ontological claim: these 23 categories constitute the *morphological vocabulary* of the system. A `DREAM` value behaves differently from a `RULE` value not because of arbitrary business logic but because dreams and rules have different ontogenetic trajectories — dreams decay into echoes, rules compile into executable form.

### Signals, Echoes, and Temporal Decay

The system distinguishes three message primitives with distinct physical semantics:

- **Signal** — carries `strength`, `threshold`, and `attenuation`. A signal propagates only when `strength >= threshold`, and its strength decreases by the attenuation factor at each hop. This models attention, salience, and the natural filtering that prevents every subsystem from responding to every event.
- **Echo** — a signal's residue after it falls below threshold. Echoes decay exponentially over time, modeling how past events influence the present with diminishing force. The `EchoHandler` subsystem manages echo persistence and retrieval.
- **Message** — the general-purpose envelope with TTL (time-to-live), priority levels, and correlation IDs for request/response patterns. Messages expire; signals attenuate; echoes decay. Each has the temporal semantics appropriate to its role.

### The Message Bus

`MessageBus` provides async pub/sub using `anyio` (backend-agnostic: works with both asyncio and trio). Topic matching supports two wildcard patterns:

- `*` matches exactly one topic segment (e.g., `subsystem.*.started` matches `subsystem.interpreter.started`)
- `#` matches zero or more segments (e.g., `value.#` matches `value.created`, `value.derived.from.dream`, etc.)

This is deliberately modeled on AMQP topic exchanges, but implemented in-process with zero network overhead. The bus is the system's nervous system — all inter-subsystem communication flows through it, making the entire communication topology observable, interceptable, and testable.

### Subsystem Lifecycle

Every subsystem extends `ProcessLoop` with a managed lifecycle state machine:

```
CREATED → STARTING → RUNNING → STOPPING → STOPPED
```

Transitions are enforced: you cannot call `process()` on a `STOPPED` subsystem, and `start()` is idempotent on a `RUNNING` one. Each subsystem maintains its own statistics (messages sent/received, processing cycles, errors) and declares its topic subscriptions at construction time.

The `Orchestrator` creates all 22 subsystems, starts them concurrently via `anyio` task groups, and provides ordered shutdown. This concurrent lifecycle management is the difference between 22 independent scripts and a coherent organism.

## Architecture

### The 22 Subsystems

Organized into 8 functional categories:

| Category | Subsystems | Role |
|----------|-----------|------|
| **Meta** | AnthologyManager | Manages collections of symbolic works across subsystems |
| **Core Processing** | SymbolicInterpreter, RuleCompiler, CodeGenerator | Intake → interpretation → compilation → executable output |
| **Value Economy** | ValueExchangeManager, BlockchainSimulator, ProcessMonetizer | Symbolic value creation, exchange, provenance ledger, monetization |
| **Identity** | MaskGenerator, AudienceClassifier | Identity construction, persona management, audience segmentation |
| **Temporal** | TimeManager, EvolutionScheduler, LocationResolver | Temporal coordination, evolutionary scheduling, spatial context |
| **Data** | ReferenceManager, ArchiveManager, EchoHandler | Reference tracking, archival, echo decay management |
| **Conflict** | ConflictResolver, ArbitrationEngine | Contradiction detection, multi-party arbitration |
| **Routing** | NodeRouter, SignalThresholdGuard | Message routing, signal strength filtering |
| **Transformation** | ProcessConverter, ConsumptionManager | Format conversion, consumption tracking |
| **Academic** | AcademiaManager | Citation management, academic output formatting |

This is not a microservices decomposition — it is an *anatomical* decomposition. Each subsystem exists because the symbolic processing domain requires a distinct organ for that function, not because of scaling or deployment concerns.

### Dependency Graph

The subsystems communicate exclusively through the message bus. There are no direct imports between subsystem modules. This means:

1. Any subsystem can be replaced without affecting others
2. New subsystems can be added by subscribing to existing topics
3. The full system can be tested by injecting messages and observing bus traffic
4. Subsystem combinations can be composed for specific use cases (you do not need all 22)

## Installation & Usage

### Requirements

- Python >= 3.11
- Dependencies: `anyio>=4.0`, `pydantic>=2.0`, `structlog>=24.0`, `python-ulid>=2.0`

### Install

```bash
# From source
git clone https://github.com/organvm-i-theoria/organon-noumenon--ontogenetic-morphe.git
cd organon-noumenon--ontogenetic-morphe
pip install -e ".[dev]"
```

### Quick Start

```python
import anyio
from autogenrec.core.symbolic import SymbolicValue, ValueType
from autogenrec.core.messaging import MessageBus, Message
from autogenrec.orchestrator import Orchestrator

async def main():
    # Create a symbolic value
    dream = SymbolicValue(value_type=ValueType.DREAM, content="recursive mirror")
    
    # Derive a new value with lineage tracking
    pattern = dream.derive(value_type=ValueType.PATTERN, content="self-reference detected")
    
    # Start the full 22-subsystem orchestrator
    orchestrator = Orchestrator()
    async with orchestrator:
        # Publish a message to the bus
        await orchestrator.bus.publish(
            "value.created",
            Message(payload=pattern)
        )

anyio.run(main)
```

### Development

```bash
# Run tests
pytest

# Type checking
mypy src/

# Linting
ruff check src/

# Property-based tests (via Hypothesis)
pytest tests/ -m hypothesis
```

## Examples

The repository includes 5 example scripts demonstrating progressively complex usage:

1. **Basic symbolic values** — Creating, typing, and deriving symbolic values with provenance chains
2. **Message bus patterns** — Topic subscriptions, wildcard matching, correlation-based request/response
3. **Single subsystem lifecycle** — Starting, processing, evaluating, and stopping a subsystem in isolation
4. **Multi-subsystem composition** — Wiring 3-4 subsystems together for a focused task
5. **Full system demo** — A research-to-revenue pipeline across 10 subsystems: a narrative value enters the SymbolicInterpreter, gets compiled by RuleCompiler, passes through ConflictResolver, is monetized by ProcessMonetizer, and exits as a tracked academic output via AcademiaManager

The full system demo is the canonical integration test — it demonstrates that the message bus topology works end-to-end without any subsystem holding a direct reference to another.

## Downstream Implementation

Ontogenetic Morphe provides the theoretical substrate for concrete applications across the organ system:

- **ORGAN-II (Poiesis)** — Generative art systems can use `SymbolicValue` with `DREAM`, `NARRATIVE`, and `MASK` types as the semantic layer beneath generative processes. The `MaskGenerator` subsystem directly supports identity-construction workflows in performance and experiential art.
- **ORGAN-III (Ergon)** — The `ValueExchangeManager`, `BlockchainSimulator`, and `ProcessMonetizer` subsystems provide a typed value-economy framework that can underpin SaaS product logic, particularly for platforms involving digital asset provenance or process-based billing.
- **ORGAN-IV (Taxis)** — The `Orchestrator` pattern and `MessageBus` architecture serve as a reference implementation for the governance and routing layer, demonstrating how autonomous subsystems coordinate without central control.

The dependency direction is strictly I → II → III: downstream organs may import from Ontogenetic Morphe, but this library has zero dependencies on any other organ's code.

## Validation

### Current State

- **13 test files** covering core abstractions, message bus, subsystem lifecycle, and integration scenarios
- **Type checking** via mypy in strict mode across the full `src/` tree
- **Property-based testing** via Hypothesis for symbolic value derivation invariants (provenance chains never break, ULID ordering is preserved, frozen models remain immutable after derive)
- **22 specification documents** — each subsystem has a dedicated spec describing its topic subscriptions, message formats, and behavioral contracts

### Quality Criteria

| Criterion | Status |
|-----------|--------|
| All 22 subsystems instantiate and start | Passing |
| ProcessLoop generic types enforce at boundaries | Passing |
| SymbolicValue.derive() maintains provenance DAG | Passing |
| Signal attenuation below threshold produces Echo | Passing |
| MessageBus wildcard matching (*, #) | Passing |
| Full system demo completes without deadlock | Passing |

## Roadmap

### Near-Term

- [ ] Persistent echo store (currently in-memory only — echoes are lost on shutdown)
- [ ] Subsystem dependency declarations (explicit "requires" graph for partial orchestration)
- [ ] Metrics export (OpenTelemetry integration for bus traffic observability)
- [ ] Symbolic value serialization format (portable representation beyond Pydantic JSON)

### Medium-Term

- [ ] Distributed message bus option (bridge to NATS or Redis Streams for multi-process deployments)
- [ ] Visual topology inspector (render subsystem graph and live message flow)
- [ ] Plugin subsystem API (third-party subsystems that register via entry points)

### Long-Term

- [ ] Formal verification of the ProcessLoop protocol (TLA+ or similar)
- [ ] Cross-organ symbolic value federation (shared provenance across ORGAN-I/II/III boundaries)

## Cross-References

### Within ORGAN-I (Theory)

- **[recursive-engine](https://github.com/organvm-i-theoria/recursive-engine--generative-entity)** — Recursive computation engine; shares the self-referential processing philosophy but operates at a different abstraction level (computation vs. symbolic metabolism)
- **[a-]() repos** — Additional theoretical explorations within the organ

### Across the Organ System

- **[ORGAN-II: Poiesis](https://github.com/organvm-ii-poiesis)** — Art organ; downstream consumer of symbolic value types
- **[ORGAN-III: Ergon](https://github.com/organvm-iii-ergon)** — Commerce organ; downstream consumer of value economy subsystems
- **[ORGAN-IV: Taxis](https://github.com/organvm-iv-taxis)** — Orchestration organ; reference consumer of bus and orchestrator patterns
- **[Meta-ORGANVM](https://github.com/meta-organvm)** — Umbrella organization coordinating all eight organs

## Related Work

Ontogenetic Morphe draws on and departs from several traditions:

- **Luhmann's autopoietic systems theory** — The recursive self-production of the ProcessLoop echoes autopoiesis, but we add typed symbolic values as the medium of self-production, making the theory computationally tractable
- **Whitehead's process philosophy** — The four-phase loop (INTAKE → PROCESS → EVALUATE → INTEGRATE) parallels Whitehead's phases of concrescence, but grounds them in async Python rather than metaphysical speculation
- **AMQP/Actor model hybrids** — The message bus borrows AMQP topic semantics while the subsystem lifecycle borrows from actor supervision trees; the combination is original
- **Digital humanities symbolic processing** — Projects like HuCit and CIDOC-CRM provide symbolic ontologies for cultural objects; Ontogenetic Morphe provides the *processing architecture* that such ontologies lack

## Contributing

Contributions are welcome. The architecture is designed for extensibility — new subsystems can be added by:

1. Extending `Subsystem` with appropriate generic type parameters
2. Declaring topic subscriptions in the constructor
3. Implementing the four-phase `ProcessLoop` methods
4. Registering with the `Orchestrator`

Please ensure all contributions include:
- Type annotations (mypy strict mode)
- Tests (pytest, property-based where applicable)
- A specification document following the existing 22-spec format

See the org-level [contributing guidelines](https://github.com/organvm-i-theoria/.github/blob/main/CONTRIBUTING.md) for code of conduct and PR process.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Author & Contact

**[@4444J99](https://github.com/4444J99)** — Creator of the eight-organ creative-institutional system.

- Org: [organvm-i-theoria](https://github.com/organvm-i-theoria) (ORGAN-I: Theory)
- Meta: [meta-organvm](https://github.com/meta-organvm)
