"""
organon-noumenon--ontogenetic-morphe

An instrument for symbolic essences, producing developmental forms.

A symbolic processing system implementing 22 interconnected subsystems
following the recursive-generative architectural pattern.

- Organon (instrument) → ProcessLoop, MessageBus
- Noumenon (thing-in-itself) → SymbolicValue
- Ontogenetic (being-developing) → Recursive 4-phase loops
- Morphē (form) → Emergent patterns, structures
"""

__version__ = "0.1.0"

from autogenrec.core.process import ProcessLoop, ProcessPhase, ProcessState
from autogenrec.core.subsystem import Subsystem
from autogenrec.core.symbolic import SymbolicInput, SymbolicOutput, SymbolicValue, SymbolicValueType

__all__ = [
    "__version__",
    "ProcessLoop",
    "ProcessPhase",
    "ProcessState",
    "Subsystem",
    "SymbolicInput",
    "SymbolicOutput",
    "SymbolicValue",
    "SymbolicValueType",
]
