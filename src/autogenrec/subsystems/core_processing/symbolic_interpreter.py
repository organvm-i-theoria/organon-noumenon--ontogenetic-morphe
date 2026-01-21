"""
SymbolicInterpreter: Interprets symbolic inputs like dreams, visions, and narratives.

Transforms symbolic material into structured guidance and insights through
pattern extraction and interpretation frameworks.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, auto
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field
from ulid import ULID

from autogenrec.bus.topics import SubsystemTopics
from autogenrec.core.process import ProcessContext
from autogenrec.core.signals import Message
from autogenrec.core.subsystem import Subsystem, SubsystemMetadata, SubsystemType
from autogenrec.core.symbolic import (
    SymbolicInput,
    SymbolicOutput,
    SymbolicValue,
    SymbolicValueType,
)

logger = structlog.get_logger()


class SymbolCategory(Enum):
    """Categories of symbolic elements."""

    ARCHETYPE = auto()  # Universal patterns (hero, shadow, etc.)
    ELEMENT = auto()  # Basic elements (water, fire, earth, air)
    ENTITY = auto()  # Beings, characters, creatures
    ACTION = auto()  # Actions, movements, transformations
    OBJECT = auto()  # Significant objects, artifacts
    PLACE = auto()  # Locations, spaces, realms
    EMOTION = auto()  # Emotional states, feelings
    RELATIONSHIP = auto()  # Connections, bonds, conflicts
    THRESHOLD = auto()  # Transitions, boundaries, passages
    UNKNOWN = auto()  # Unclassified symbols


class InterpretiveFramework(Enum):
    """Frameworks for interpreting symbolic content."""

    JUNGIAN = auto()  # Collective unconscious, archetypes
    NARRATIVE = auto()  # Story structure, plot patterns
    MYTHOLOGICAL = auto()  # Mythic themes and motifs
    ALCHEMICAL = auto()  # Transformation stages
    STRUCTURAL = auto()  # Structural analysis
    PHENOMENOLOGICAL = auto()  # Experiential meaning


class ExtractedSymbol(BaseModel):
    """A symbol extracted from symbolic content."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    name: str
    category: SymbolCategory
    raw_fragment: str  # Original text fragment
    position: int  # Position in source content
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    associations: frozenset[str] = Field(default_factory=frozenset)
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Pattern(BaseModel):
    """A pattern recognized in symbolic content."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    name: str
    pattern_type: str  # e.g., "journey", "transformation", "encounter"
    symbols: tuple[str, ...] = Field(default_factory=tuple)  # Symbol IDs
    framework: InterpretiveFramework
    description: str
    significance: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Interpretation(BaseModel):
    """An interpretation of symbolic content."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    source_id: str  # ID of the source SymbolicValue
    framework: InterpretiveFramework
    symbols: tuple[ExtractedSymbol, ...] = Field(default_factory=tuple)
    patterns: tuple[Pattern, ...] = Field(default_factory=tuple)
    synthesis: str  # Overall interpretation
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class InterpretationContext:
    """Context maintained during interpretation."""

    symbols: list[ExtractedSymbol] = field(default_factory=list)
    patterns: list[Pattern] = field(default_factory=list)
    active_framework: InterpretiveFramework = InterpretiveFramework.NARRATIVE


class SymbolExtractor:
    """Extracts symbols from symbolic content."""

    # Symbol patterns for extraction (keyword-based for demonstration)
    SYMBOL_KEYWORDS: dict[SymbolCategory, set[str]] = {
        SymbolCategory.ARCHETYPE: {
            "hero", "shadow", "mentor", "trickster", "sage", "mother",
            "father", "child", "self", "anima", "animus", "warrior",
        },
        SymbolCategory.ELEMENT: {
            "water", "fire", "earth", "air", "wind", "flame", "ocean",
            "river", "mountain", "forest", "storm", "light", "darkness",
        },
        SymbolCategory.ENTITY: {
            "dragon", "serpent", "bird", "wolf", "lion", "eagle",
            "spirit", "ghost", "angel", "demon", "god", "goddess",
        },
        SymbolCategory.ACTION: {
            "journey", "transformation", "death", "rebirth", "ascent",
            "descent", "battle", "flight", "pursuit", "return", "crossing",
        },
        SymbolCategory.OBJECT: {
            "sword", "key", "ring", "crown", "mirror", "door", "gate",
            "bridge", "tower", "treasure", "book", "chalice", "stone",
        },
        SymbolCategory.PLACE: {
            "temple", "cave", "garden", "forest", "desert", "abyss",
            "kingdom", "underworld", "heaven", "home", "wilderness",
        },
        SymbolCategory.EMOTION: {
            "fear", "love", "anger", "joy", "sorrow", "wonder",
            "dread", "hope", "despair", "longing", "peace",
        },
        SymbolCategory.RELATIONSHIP: {
            "bond", "conflict", "union", "separation", "betrayal",
            "alliance", "sacrifice", "gift", "oath", "promise",
        },
        SymbolCategory.THRESHOLD: {
            "threshold", "boundary", "passage", "transition", "portal",
            "beginning", "ending", "awakening", "initiation",
        },
    }

    def __init__(self) -> None:
        self._log = logger.bind(component="symbol_extractor")

    def extract(self, content: str) -> list[ExtractedSymbol]:
        """Extract symbols from content."""
        symbols: list[ExtractedSymbol] = []
        content_lower = content.lower()
        words = content_lower.split()

        for category, keywords in self.SYMBOL_KEYWORDS.items():
            for keyword in keywords:
                pos = content_lower.find(keyword)
                if pos != -1:
                    # Find the context around the symbol
                    start = max(0, pos - 20)
                    end = min(len(content), pos + len(keyword) + 20)
                    fragment = content[start:end].strip()

                    symbol = ExtractedSymbol(
                        name=keyword,
                        category=category,
                        raw_fragment=fragment,
                        position=pos,
                        confidence=0.8 if keyword in words else 0.6,
                        associations=self._get_associations(keyword, category),
                    )
                    symbols.append(symbol)

        self._log.debug("symbols_extracted", count=len(symbols))
        return symbols

    def _get_associations(self, keyword: str, category: SymbolCategory) -> frozenset[str]:
        """Get semantic associations for a symbol."""
        # Basic association mapping
        associations: dict[str, set[str]] = {
            "water": {"emotion", "unconscious", "purification", "flow"},
            "fire": {"transformation", "passion", "destruction", "illumination"},
            "journey": {"growth", "quest", "discovery", "initiation"},
            "shadow": {"unconscious", "repressed", "integration", "darkness"},
            "hero": {"ego", "courage", "quest", "transformation"},
            "threshold": {"change", "opportunity", "risk", "growth"},
        }
        return frozenset(associations.get(keyword, {category.name.lower()}))


class PatternRecognizer:
    """Recognizes patterns in extracted symbols."""

    # Pattern templates
    PATTERN_TEMPLATES: dict[str, dict[str, Any]] = {
        "hero_journey": {
            "required_categories": {SymbolCategory.ARCHETYPE, SymbolCategory.ACTION},
            "keywords": {"hero", "journey", "return", "transformation"},
            "framework": InterpretiveFramework.MYTHOLOGICAL,
            "description": "Classic hero's journey narrative pattern",
        },
        "death_rebirth": {
            "required_categories": {SymbolCategory.ACTION, SymbolCategory.THRESHOLD},
            "keywords": {"death", "rebirth", "transformation", "descent", "ascent"},
            "framework": InterpretiveFramework.ALCHEMICAL,
            "description": "Death and rebirth transformation cycle",
        },
        "shadow_encounter": {
            "required_categories": {SymbolCategory.ARCHETYPE, SymbolCategory.RELATIONSHIP},
            "keywords": {"shadow", "darkness", "conflict", "battle"},
            "framework": InterpretiveFramework.JUNGIAN,
            "description": "Encounter with the shadow self",
        },
        "threshold_crossing": {
            "required_categories": {SymbolCategory.THRESHOLD, SymbolCategory.PLACE},
            "keywords": {"threshold", "door", "gate", "passage", "crossing"},
            "framework": InterpretiveFramework.NARRATIVE,
            "description": "Crossing a significant threshold or boundary",
        },
        "elemental_transformation": {
            "required_categories": {SymbolCategory.ELEMENT, SymbolCategory.ACTION},
            "keywords": {"fire", "water", "transformation", "change"},
            "framework": InterpretiveFramework.ALCHEMICAL,
            "description": "Transformation through elemental forces",
        },
    }

    def __init__(self) -> None:
        self._log = logger.bind(component="pattern_recognizer")

    def recognize(self, symbols: list[ExtractedSymbol]) -> list[Pattern]:
        """Recognize patterns from extracted symbols."""
        patterns: list[Pattern] = []
        symbol_categories = {s.category for s in symbols}
        symbol_names = {s.name for s in symbols}

        for pattern_name, template in self.PATTERN_TEMPLATES.items():
            required_categories = template["required_categories"]
            keywords = template["keywords"]

            # Check if required categories are present
            if required_categories <= symbol_categories:
                # Check if any keywords match
                matching_keywords = keywords & symbol_names
                if matching_keywords:
                    matching_symbol_ids = tuple(
                        s.id for s in symbols if s.name in matching_keywords
                    )
                    significance = len(matching_keywords) / len(keywords)

                    pattern = Pattern(
                        name=pattern_name,
                        pattern_type=pattern_name.replace("_", " "),
                        symbols=matching_symbol_ids,
                        framework=template["framework"],
                        description=template["description"],
                        significance=min(1.0, significance + 0.3),
                    )
                    patterns.append(pattern)

        self._log.debug("patterns_recognized", count=len(patterns))
        return patterns


class InterpretationEngine:
    """Engine for interpreting symbolic content."""

    def __init__(self) -> None:
        self._extractor = SymbolExtractor()
        self._recognizer = PatternRecognizer()
        self._log = logger.bind(component="interpretation_engine")

    def interpret(
        self,
        value: SymbolicValue,
        framework: InterpretiveFramework = InterpretiveFramework.NARRATIVE,
    ) -> Interpretation:
        """Interpret a symbolic value."""
        content = self._extract_content(value)
        symbols = self._extractor.extract(content)
        patterns = self._recognizer.recognize(symbols)

        # Generate synthesis based on patterns and symbols
        synthesis = self._synthesize(symbols, patterns, framework)

        # Calculate overall confidence
        symbol_confidence = sum(s.confidence for s in symbols) / max(len(symbols), 1)
        pattern_confidence = sum(p.significance for p in patterns) / max(len(patterns), 1)
        overall_confidence = (symbol_confidence + pattern_confidence) / 2

        interpretation = Interpretation(
            source_id=value.id,
            framework=framework,
            symbols=tuple(symbols),
            patterns=tuple(patterns),
            synthesis=synthesis,
            confidence=overall_confidence,
            metadata={
                "content_length": len(content),
                "value_type": value.type.name,
            },
        )

        self._log.debug(
            "interpretation_complete",
            source_id=value.id,
            symbols=len(symbols),
            patterns=len(patterns),
            confidence=overall_confidence,
        )

        return interpretation

    def _extract_content(self, value: SymbolicValue) -> str:
        """Extract textual content from a symbolic value."""
        if isinstance(value.content, str):
            return value.content
        if isinstance(value.content, dict):
            # Try common keys for content
            for key in ("text", "content", "narrative", "description", "body"):
                if key in value.content:
                    return str(value.content[key])
            return str(value.content)
        return str(value.content)

    def _synthesize(
        self,
        symbols: list[ExtractedSymbol],
        patterns: list[Pattern],
        framework: InterpretiveFramework,
    ) -> str:
        """Synthesize an interpretation from symbols and patterns."""
        if not symbols and not patterns:
            return "No significant symbolic content identified."

        parts: list[str] = []

        # Describe patterns first
        if patterns:
            pattern_desc = ", ".join(p.name.replace("_", " ") for p in patterns)
            parts.append(f"Patterns identified: {pattern_desc}.")

        # Summarize symbols by category
        by_category: dict[SymbolCategory, list[str]] = {}
        for symbol in symbols:
            by_category.setdefault(symbol.category, []).append(symbol.name)

        if by_category:
            for category, names in sorted(by_category.items(), key=lambda x: x[0].name):
                unique_names = list(dict.fromkeys(names))[:5]
                parts.append(f"{category.name.title()}: {', '.join(unique_names)}.")

        # Framework-specific insight
        if framework == InterpretiveFramework.JUNGIAN:
            parts.append("Archetypal analysis suggests engagement with unconscious material.")
        elif framework == InterpretiveFramework.MYTHOLOGICAL:
            parts.append("Mythic themes indicate connection to universal narrative patterns.")
        elif framework == InterpretiveFramework.ALCHEMICAL:
            parts.append("Alchemical perspective reveals transformative processes at work.")

        return " ".join(parts)


class SymbolicInterpreter(Subsystem):
    """
    Interprets symbolic inputs and transforms them into structured guidance.

    Process Loop:
    1. Intake: Gather and filter symbolic inputs (dreams, visions, narratives)
    2. Process: Apply interpretive frameworks and extract patterns
    3. Evaluate: Synthesize interpretations into structured outcomes
    4. Integrate: Emit events and prepare feedback for further processing
    """

    SUPPORTED_TYPES = {
        SymbolicValueType.NARRATIVE,
        SymbolicValueType.DREAM,
        SymbolicValueType.VISION,
        SymbolicValueType.LINGUISTIC,
    }

    def __init__(self) -> None:
        metadata = SubsystemMetadata(
            name="symbolic_interpreter",
            display_name="Symbolic Interpreter",
            description="Interprets symbolic inputs and transforms them into structured guidance",
            type=SubsystemType.CORE_PROCESSING,
            tags=frozenset(["symbolism", "interpretation", "dreams", "narratives", "patterns"]),
            input_types=frozenset(["NARRATIVE", "DREAM", "VISION", "LINGUISTIC"]),
            output_types=frozenset(["PATTERN", "SCHEMA", "RULE"]),
            subscribed_topics=frozenset([
                str(SubsystemTopics.ALL),
                "symbolic.input.#",
                "interpretation.request.#",
            ]),
            published_topics=frozenset([
                "interpretation.complete",
                "pattern.extracted",
                "symbol.identified",
            ]),
        )
        super().__init__(metadata)

        self._engine = InterpretationEngine()
        self._interpretations: dict[str, Interpretation] = {}
        self._active_framework = InterpretiveFramework.NARRATIVE

    @property
    def interpretation_count(self) -> int:
        return len(self._interpretations)

    @property
    def active_framework(self) -> InterpretiveFramework:
        return self._active_framework

    def set_framework(self, framework: InterpretiveFramework) -> None:
        """Set the active interpretive framework."""
        self._active_framework = framework
        self._log.info("framework_changed", framework=framework.name)

    async def intake(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> SymbolicInput:
        """Phase 1: Collect and filter symbolic inputs."""
        if not input_data.values:
            self._log.debug("empty_input")
            return input_data

        # Filter to supported input types
        valid_values = [v for v in input_data.values if v.type in self.SUPPORTED_TYPES]

        if len(valid_values) != len(input_data.values):
            filtered_count = len(input_data.values) - len(valid_values)
            self._log.debug("filtered_unsupported", filtered=filtered_count)

        self._log.debug(
            "intake_complete",
            total=len(input_data.values),
            valid=len(valid_values),
        )

        # Store framework hint from metadata if present
        if "framework" in input_data.metadata:
            try:
                self._active_framework = InterpretiveFramework[
                    input_data.metadata["framework"].upper()
                ]
            except (KeyError, AttributeError):
                pass

        if len(valid_values) != len(input_data.values):
            return SymbolicInput(
                values=tuple(valid_values),
                source_subsystem=input_data.source_subsystem,
                target_subsystem=input_data.target_subsystem,
                correlation_id=input_data.correlation_id,
                priority=input_data.priority,
                metadata=input_data.metadata,
            )
        return input_data

    async def process(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> list[Interpretation]:
        """Phase 2: Apply interpretive frameworks to symbolic inputs."""
        interpretations: list[Interpretation] = []

        for value in input_data.values:
            interpretation = self._engine.interpret(value, self._active_framework)
            self._interpretations[interpretation.id] = interpretation
            interpretations.append(interpretation)

            self._log.debug(
                "value_interpreted",
                value_id=value.id,
                interpretation_id=interpretation.id,
                symbols=len(interpretation.symbols),
                patterns=len(interpretation.patterns),
            )

        return interpretations

    async def evaluate(
        self, intermediate: list[Interpretation], ctx: ProcessContext[dict[str, Any]]
    ) -> tuple[SymbolicOutput, bool]:
        """Phase 3: Synthesize interpretations into structured outputs."""
        values: list[SymbolicValue] = []

        for interpretation in intermediate:
            # Create pattern values for each recognized pattern
            for pattern in interpretation.patterns:
                pattern_value = SymbolicValue(
                    type=SymbolicValueType.PATTERN,
                    content={
                        "pattern_id": pattern.id,
                        "name": pattern.name,
                        "type": pattern.pattern_type,
                        "description": pattern.description,
                        "significance": pattern.significance,
                        "framework": pattern.framework.name,
                        "symbol_ids": list(pattern.symbols),
                    },
                    source_subsystem=self.name,
                    tags=frozenset(["pattern", pattern.name, pattern.framework.name.lower()]),
                    meaning=pattern.description,
                    confidence=pattern.significance,
                    parent_id=interpretation.source_id,
                )
                values.append(pattern_value)

            # Create a schema value for the overall interpretation
            schema_value = SymbolicValue(
                type=SymbolicValueType.SCHEMA,
                content={
                    "interpretation_id": interpretation.id,
                    "framework": interpretation.framework.name,
                    "synthesis": interpretation.synthesis,
                    "symbol_count": len(interpretation.symbols),
                    "pattern_count": len(interpretation.patterns),
                    "symbols": [
                        {
                            "id": s.id,
                            "name": s.name,
                            "category": s.category.name,
                            "confidence": s.confidence,
                        }
                        for s in interpretation.symbols
                    ],
                },
                source_subsystem=self.name,
                tags=frozenset(["interpretation", "schema", interpretation.framework.name.lower()]),
                meaning=interpretation.synthesis,
                confidence=interpretation.confidence,
                parent_id=interpretation.source_id,
            )
            values.append(schema_value)

        output = self.create_output(
            values=values,
            input_id=ctx.metadata.get("input_id"),
        )

        # Don't continue iterating - interpretations are complete
        return output, False

    async def integrate(
        self, output: SymbolicOutput, ctx: ProcessContext[dict[str, Any]]
    ) -> SymbolicInput | None:
        """Phase 4: Emit events and prepare feedback."""
        if self._message_bus and output.values:
            # Emit events for patterns
            for value in output.values:
                if value.type == SymbolicValueType.PATTERN:
                    await self.emit_event(
                        "pattern.extracted",
                        {
                            "pattern_id": value.content.get("pattern_id"),
                            "name": value.content.get("name"),
                            "significance": value.content.get("significance"),
                        },
                    )
                elif value.type == SymbolicValueType.SCHEMA:
                    await self.emit_event(
                        "interpretation.complete",
                        {
                            "interpretation_id": value.content.get("interpretation_id"),
                            "framework": value.content.get("framework"),
                            "symbol_count": value.content.get("symbol_count"),
                            "pattern_count": value.content.get("pattern_count"),
                        },
                    )

        # No recursion needed
        return None

    # --- Message handlers ---

    async def handle_event(self, message: Message) -> None:
        """Handle incoming events."""
        if message.topic.startswith("interpretation.request"):
            # Handle interpretation requests
            self._log.debug("interpretation_request_received", message_id=message.id)

    async def handle_signal(self, signal: Any) -> None:
        """Handle incoming signals."""
        self._log.debug("signal_received", signal_id=getattr(signal, "id", "unknown"))

    # --- Query API ---

    def get_interpretation(self, interpretation_id: str) -> Interpretation | None:
        """Get an interpretation by ID."""
        return self._interpretations.get(interpretation_id)

    def get_interpretations_by_framework(
        self, framework: InterpretiveFramework
    ) -> list[Interpretation]:
        """Get all interpretations using a specific framework."""
        return [
            i for i in self._interpretations.values()
            if i.framework == framework
        ]

    def get_high_confidence_patterns(
        self, min_confidence: float = 0.7
    ) -> list[Pattern]:
        """Get patterns above a confidence threshold."""
        patterns: list[Pattern] = []
        for interpretation in self._interpretations.values():
            for pattern in interpretation.patterns:
                if pattern.significance >= min_confidence:
                    patterns.append(pattern)
        return patterns

    def clear_interpretations(self) -> int:
        """Clear all stored interpretations. Returns count cleared."""
        count = len(self._interpretations)
        self._interpretations.clear()
        return count
