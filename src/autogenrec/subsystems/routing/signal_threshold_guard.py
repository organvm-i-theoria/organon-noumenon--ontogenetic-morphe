"""
SignalThresholdGuard: Monitors and validates signals across analog/digital thresholds.

Validates, monitors, and transforms signals as they cross analog-digital
boundaries, ensuring fidelity and coherence during conversions.
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
from autogenrec.core.signals import Message, Signal, SignalDomain, SignalPriority
from autogenrec.core.subsystem import Subsystem, SubsystemMetadata, SubsystemType
from autogenrec.core.symbolic import (
    SymbolicInput,
    SymbolicOutput,
    SymbolicValue,
    SymbolicValueType,
)

logger = structlog.get_logger()


class ThresholdType(Enum):
    """Types of thresholds for signal validation."""

    STRENGTH = auto()  # Signal strength threshold
    NOISE = auto()  # Noise level threshold
    FREQUENCY = auto()  # Signal frequency threshold
    AMPLITUDE = auto()  # Signal amplitude threshold
    CUSTOM = auto()  # Custom threshold


class ConversionMode(Enum):
    """Modes for analog-digital conversion."""

    QUANTIZE = auto()  # Simple quantization
    SAMPLE_HOLD = auto()  # Sample and hold
    INTERPOLATE = auto()  # Interpolation
    ADAPTIVE = auto()  # Adaptive conversion based on signal characteristics


class ValidationResult(Enum):
    """Result of signal validation."""

    PASSED = auto()  # Signal passed validation
    FAILED = auto()  # Signal failed validation
    ADJUSTED = auto()  # Signal was adjusted to pass
    BLOCKED = auto()  # Signal was blocked
    WARNING = auto()  # Signal passed with warnings


class ThresholdPolicy(BaseModel):
    """Policy for threshold validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    name: str
    threshold_type: ThresholdType = ThresholdType.STRENGTH

    # Threshold values
    min_value: float = 0.0
    max_value: float = 1.0
    target_value: float | None = None  # For adjustment

    # Behavior
    action_on_fail: str = "block"  # "block", "adjust", "warn", "pass"
    allow_adjustment: bool = True
    adjustment_factor: float = 0.1  # How much to adjust

    # Domain-specific
    applies_to_domain: SignalDomain | None = None  # None = all domains

    # Metadata
    priority: int = 50
    enabled: bool = True
    description: str = ""
    tags: frozenset[str] = Field(default_factory=frozenset)


class ConversionPolicy(BaseModel):
    """Policy for domain conversion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    name: str
    source_domain: SignalDomain
    target_domain: SignalDomain
    mode: ConversionMode = ConversionMode.QUANTIZE

    # Conversion parameters
    quantization_levels: int = 256  # For digital conversion
    sample_rate: float = 1.0  # For sampling
    interpolation_factor: int = 2  # For interpolation

    # Quality settings
    preserve_strength: bool = True
    preserve_metadata: bool = True
    min_output_strength: float = 0.1

    # Metadata
    priority: int = 50
    enabled: bool = True
    description: str = ""


@dataclass
class ValidationReport:
    """Report from validating a signal."""

    signal_id: str
    result: ValidationResult
    original_strength: float
    final_strength: float
    policies_applied: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    validated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ConversionReport:
    """Report from converting a signal."""

    signal_id: str
    source_domain: SignalDomain
    target_domain: SignalDomain
    success: bool
    policy_used: str | None = None
    original_strength: float = 0.0
    converted_strength: float = 0.0
    metadata_preserved: bool = True
    converted_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class GuardStats:
    """Statistics for the threshold guard."""

    total_validated: int
    total_passed: int
    total_failed: int
    total_adjusted: int
    total_blocked: int
    total_converted: int
    conversion_success_rate: float


class ThresholdValidator:
    """Validates signals against threshold policies."""

    def __init__(self) -> None:
        self._policies: dict[str, ThresholdPolicy] = {}
        self._log = logger.bind(component="threshold_validator")

        # Add default policies
        self._add_default_policies()

    def _add_default_policies(self) -> None:
        """Add default threshold policies."""
        defaults = [
            ThresholdPolicy(
                name="strength_minimum",
                threshold_type=ThresholdType.STRENGTH,
                min_value=0.1,
                max_value=1.0,
                action_on_fail="adjust",
                allow_adjustment=True,
                adjustment_factor=0.2,
                priority=100,
            ),
            ThresholdPolicy(
                name="digital_strength",
                threshold_type=ThresholdType.STRENGTH,
                min_value=0.5,
                max_value=1.0,
                action_on_fail="warn",
                applies_to_domain=SignalDomain.DIGITAL,
                priority=50,
            ),
            ThresholdPolicy(
                name="analog_noise",
                threshold_type=ThresholdType.NOISE,
                min_value=0.0,
                max_value=0.3,
                action_on_fail="adjust",
                applies_to_domain=SignalDomain.ANALOG,
                priority=50,
            ),
        ]
        for policy in defaults:
            self._policies[policy.id] = policy

    def add_policy(self, policy: ThresholdPolicy) -> None:
        """Add a threshold policy."""
        self._policies[policy.id] = policy
        self._log.debug("policy_added", policy_id=policy.id, name=policy.name)

    def remove_policy(self, policy_id: str) -> bool:
        """Remove a threshold policy."""
        if policy_id in self._policies:
            del self._policies[policy_id]
            return True
        return False

    def validate(self, signal: Signal) -> ValidationReport:
        """Validate a signal against all applicable policies."""
        policies_applied: list[str] = []
        warnings: list[str] = []
        errors: list[str] = []
        final_strength = signal.strength
        result = ValidationResult.PASSED

        # Get applicable policies sorted by priority
        applicable = [
            p for p in self._policies.values()
            if p.enabled and (p.applies_to_domain is None or p.applies_to_domain == signal.domain)
        ]
        applicable.sort(key=lambda p: p.priority, reverse=True)

        for policy in applicable:
            policies_applied.append(policy.name)
            passed, new_strength, message = self._apply_policy(policy, final_strength, signal)

            if not passed:
                if policy.action_on_fail == "block":
                    errors.append(f"{policy.name}: {message}")
                    result = ValidationResult.BLOCKED
                    break
                elif policy.action_on_fail == "adjust" and policy.allow_adjustment:
                    final_strength = new_strength
                    warnings.append(f"{policy.name}: Adjusted - {message}")
                    if result != ValidationResult.BLOCKED:
                        result = ValidationResult.ADJUSTED
                elif policy.action_on_fail == "warn":
                    warnings.append(f"{policy.name}: {message}")
                    if result == ValidationResult.PASSED:
                        result = ValidationResult.WARNING
                else:  # "pass"
                    pass

        if result == ValidationResult.BLOCKED:
            final_strength = 0.0

        return ValidationReport(
            signal_id=signal.id,
            result=result,
            original_strength=signal.strength,
            final_strength=final_strength,
            policies_applied=policies_applied,
            warnings=warnings,
            errors=errors,
        )

    def _apply_policy(
        self,
        policy: ThresholdPolicy,
        strength: float,
        signal: Signal,
    ) -> tuple[bool, float, str]:
        """Apply a policy to a signal. Returns (passed, adjusted_strength, message)."""
        if policy.threshold_type == ThresholdType.STRENGTH:
            if strength < policy.min_value:
                adjusted = policy.min_value + policy.adjustment_factor
                return False, adjusted, f"Strength {strength:.2f} below minimum {policy.min_value:.2f}"
            if strength > policy.max_value:
                adjusted = policy.max_value
                return False, adjusted, f"Strength {strength:.2f} above maximum {policy.max_value:.2f}"
            return True, strength, "OK"

        elif policy.threshold_type == ThresholdType.NOISE:
            # Simulate noise level based on strength
            noise_estimate = 1.0 - strength
            if noise_estimate > policy.max_value:
                return False, strength, f"Noise level {noise_estimate:.2f} above threshold {policy.max_value:.2f}"
            return True, strength, "OK"

        # Default pass for unhandled types
        return True, strength, "OK"


class DomainConverter:
    """Converts signals between analog and digital domains."""

    def __init__(self) -> None:
        self._policies: dict[str, ConversionPolicy] = {}
        self._log = logger.bind(component="domain_converter")

        # Add default policies
        self._add_default_policies()

    def _add_default_policies(self) -> None:
        """Add default conversion policies."""
        defaults = [
            ConversionPolicy(
                name="analog_to_digital",
                source_domain=SignalDomain.ANALOG,
                target_domain=SignalDomain.DIGITAL,
                mode=ConversionMode.QUANTIZE,
                quantization_levels=256,
                priority=100,
            ),
            ConversionPolicy(
                name="digital_to_analog",
                source_domain=SignalDomain.DIGITAL,
                target_domain=SignalDomain.ANALOG,
                mode=ConversionMode.INTERPOLATE,
                interpolation_factor=4,
                priority=100,
            ),
            ConversionPolicy(
                name="hybrid_to_digital",
                source_domain=SignalDomain.HYBRID,
                target_domain=SignalDomain.DIGITAL,
                mode=ConversionMode.ADAPTIVE,
                priority=50,
            ),
        ]
        for policy in defaults:
            self._policies[policy.id] = policy

    def add_policy(self, policy: ConversionPolicy) -> None:
        """Add a conversion policy."""
        self._policies[policy.id] = policy

    def get_policy(
        self,
        source: SignalDomain,
        target: SignalDomain,
    ) -> ConversionPolicy | None:
        """Get the best policy for a conversion."""
        candidates = [
            p for p in self._policies.values()
            if p.enabled and p.source_domain == source and p.target_domain == target
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.priority)

    def convert(
        self,
        signal: Signal,
        target_domain: SignalDomain,
    ) -> tuple[Signal | None, ConversionReport]:
        """Convert a signal to a target domain."""
        if signal.domain == target_domain:
            # No conversion needed
            return signal, ConversionReport(
                signal_id=signal.id,
                source_domain=signal.domain,
                target_domain=target_domain,
                success=True,
                original_strength=signal.strength,
                converted_strength=signal.strength,
            )

        policy = self.get_policy(signal.domain, target_domain)
        if not policy:
            return None, ConversionReport(
                signal_id=signal.id,
                source_domain=signal.domain,
                target_domain=target_domain,
                success=False,
                original_strength=signal.strength,
                converted_strength=0.0,
            )

        # Perform conversion
        converted_signal, converted_strength = self._apply_conversion(signal, target_domain, policy)

        return converted_signal, ConversionReport(
            signal_id=signal.id,
            source_domain=signal.domain,
            target_domain=target_domain,
            success=True,
            policy_used=policy.name,
            original_strength=signal.strength,
            converted_strength=converted_strength,
            metadata_preserved=policy.preserve_metadata,
        )

    def _apply_conversion(
        self,
        signal: Signal,
        target_domain: SignalDomain,
        policy: ConversionPolicy,
    ) -> tuple[Signal, float]:
        """Apply conversion to a signal."""
        # Calculate converted strength based on mode
        if policy.mode == ConversionMode.QUANTIZE:
            # Quantization can introduce slight loss
            quantized = round(signal.strength * policy.quantization_levels) / policy.quantization_levels
            converted_strength = max(quantized, policy.min_output_strength)
        elif policy.mode == ConversionMode.INTERPOLATE:
            # Interpolation preserves strength well
            converted_strength = signal.strength
        elif policy.mode == ConversionMode.SAMPLE_HOLD:
            # Sample and hold may introduce slight variance
            converted_strength = signal.strength * 0.99
        else:  # ADAPTIVE
            # Adaptive tries to preserve as much as possible
            converted_strength = signal.strength

        if policy.preserve_strength:
            converted_strength = max(converted_strength, signal.strength * 0.95)

        # Create converted signal
        converted = signal.convert_domain(target_domain)

        # If strength changed, create new signal with updated strength
        if abs(converted_strength - signal.strength) > 0.001:
            converted = Signal(
                domain=target_domain,
                payload=signal.payload,
                payload_type=signal.payload_type,
                source=signal.source,
                target=signal.target,
                correlation_id=signal.correlation_id,
                strength=converted_strength,
                threshold=signal.threshold,
                priority=signal.priority,
                tags=signal.tags | frozenset(["converted"]),
                metadata={
                    **signal.metadata,
                    "converted_from": signal.domain.name,
                    "original_strength": signal.strength,
                    "conversion_policy": policy.name,
                },
            )

        return converted, converted_strength


class SignalThresholdGuard(Subsystem):
    """
    Monitors and validates signals across analog-digital thresholds.

    Process Loop:
    1. Receive: Intake mixed signals (analog or digital)
    2. Validate: Assess whether signals meet threshold criteria
    3. Convert: Transform signals between formats when required
    4. Distribute: Route converted signals to appropriate subsystems
    """

    def __init__(self) -> None:
        metadata = SubsystemMetadata(
            name="signal_threshold_guard",
            display_name="Signal Threshold Guard",
            description="Monitors and validates signals across analog-digital thresholds",
            type=SubsystemType.ROUTING,
            tags=frozenset(["signal", "threshold", "conversion", "validation"]),
            input_types=frozenset(["SIGNAL"]),
            output_types=frozenset(["SIGNAL"]),
            subscribed_topics=frozenset([
                str(SubsystemTopics.ALL),
                "signal.#",
                "threshold.#",
            ]),
            published_topics=frozenset([
                "threshold.validation.passed",
                "threshold.validation.failed",
                "threshold.conversion.complete",
                "threshold.signal.blocked",
            ]),
        )
        super().__init__(metadata)

        self._validator = ThresholdValidator()
        self._converter = DomainConverter()

        # Statistics
        self._total_validated = 0
        self._total_passed = 0
        self._total_failed = 0
        self._total_adjusted = 0
        self._total_blocked = 0
        self._total_converted = 0
        self._conversion_successes = 0

    async def intake(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> SymbolicInput:
        """Phase 1: Receive mixed signals."""
        if not input_data.values:
            self._log.debug("empty_input")
            return input_data

        # Filter to signal types
        valid_values = [v for v in input_data.values if v.type == SymbolicValueType.SIGNAL]

        self._log.debug(
            "intake_complete",
            total=len(input_data.values),
            valid=len(valid_values),
        )

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
    ) -> list[tuple[Signal | None, ValidationReport, ConversionReport | None]]:
        """Phase 2 & 3: Validate and convert signals."""
        results: list[tuple[Signal | None, ValidationReport, ConversionReport | None]] = []

        # Check if conversion is requested
        target_domain_str = ctx.metadata.get("target_domain")
        target_domain = None
        if target_domain_str:
            try:
                target_domain = SignalDomain[target_domain_str.upper()]
            except KeyError:
                pass

        for value in input_data.values:
            signal = self._parse_signal(value)
            if not signal:
                continue

            # Validate
            validation = self._validator.validate(signal)
            self._total_validated += 1

            if validation.result == ValidationResult.BLOCKED:
                self._total_blocked += 1
                results.append((None, validation, None))
                continue

            if validation.result == ValidationResult.PASSED:
                self._total_passed += 1
            elif validation.result == ValidationResult.ADJUSTED:
                self._total_adjusted += 1
                # Create adjusted signal
                signal = Signal(
                    domain=signal.domain,
                    payload=signal.payload,
                    payload_type=signal.payload_type,
                    source=signal.source,
                    target=signal.target,
                    correlation_id=signal.correlation_id,
                    strength=validation.final_strength,
                    threshold=signal.threshold,
                    priority=signal.priority,
                    tags=signal.tags | frozenset(["adjusted"]),
                    metadata={**signal.metadata, "allow_weak": True},
                )
            elif validation.result == ValidationResult.FAILED:
                self._total_failed += 1

            # Convert if requested
            conversion = None
            if target_domain and target_domain != signal.domain:
                signal, conversion = self._converter.convert(signal, target_domain)
                self._total_converted += 1
                if conversion.success:
                    self._conversion_successes += 1

            results.append((signal, validation, conversion))

        return results

    async def evaluate(
        self, intermediate: list[tuple[Signal | None, ValidationReport, ConversionReport | None]],
        ctx: ProcessContext[dict[str, Any]],
    ) -> tuple[SymbolicOutput, bool]:
        """Phase 4: Create output with validated/converted signals."""
        values: list[SymbolicValue] = []

        for signal, validation, conversion in intermediate:
            if signal:
                value = SymbolicValue(
                    type=SymbolicValueType.SIGNAL,
                    content={
                        "signal_id": signal.id,
                        "domain": signal.domain.name,
                        "source": signal.source,
                        "strength": signal.strength,
                        "validation_result": validation.result.name,
                        "was_converted": conversion is not None and conversion.success,
                        "payload": signal.payload,
                    },
                    source_subsystem=self.name,
                    tags=signal.tags | frozenset(["validated"]),
                    meaning=f"Validated signal from {signal.source}",
                    confidence=signal.strength,
                )
            else:
                # Blocked signal
                value = SymbolicValue(
                    type=SymbolicValueType.SCHEMA,
                    content={
                        "signal_id": validation.signal_id,
                        "validation_result": validation.result.name,
                        "errors": validation.errors,
                        "warnings": validation.warnings,
                    },
                    source_subsystem=self.name,
                    tags=frozenset(["blocked", "validation"]),
                    meaning="Signal blocked by threshold guard",
                    confidence=0.0,
                )

            values.append(value)

        output = self.create_output(
            values=values,
            input_id=ctx.metadata.get("input_id"),
        )

        return output, False

    async def integrate(
        self, output: SymbolicOutput, ctx: ProcessContext[dict[str, Any]]
    ) -> SymbolicInput | None:
        """Phase 4: Emit events for validated signals."""
        if self._message_bus and output.values:
            for value in output.values:
                if value.type == SymbolicValueType.SIGNAL:
                    await self.emit_event(
                        "threshold.validation.passed",
                        {
                            "signal_id": value.content.get("signal_id"),
                            "strength": value.content.get("strength"),
                            "was_converted": value.content.get("was_converted"),
                        },
                    )
                else:
                    await self.emit_event(
                        "threshold.signal.blocked",
                        {
                            "signal_id": value.content.get("signal_id"),
                            "errors": value.content.get("errors"),
                        },
                    )

        return None

    def _parse_signal(self, value: SymbolicValue) -> Signal | None:
        """Parse a Signal from a SymbolicValue."""
        content = value.content
        if not isinstance(content, dict):
            return None

        try:
            signal_data = content.get("signal", content)

            domain_str = signal_data.get("domain", "DIGITAL")
            try:
                domain = SignalDomain[domain_str.upper()]
            except KeyError:
                domain = SignalDomain.DIGITAL

            return Signal(
                id=signal_data.get("id", str(ULID())),
                domain=domain,
                payload=signal_data.get("payload", content),
                payload_type=signal_data.get("payload_type", "unknown"),
                source=signal_data.get("source", value.source_subsystem or "unknown"),
                target=signal_data.get("target"),
                correlation_id=signal_data.get("correlation_id"),
                strength=signal_data.get("strength", 1.0),
                threshold=signal_data.get("threshold", 0.5),
                tags=frozenset(signal_data.get("tags", [])) | value.tags,
                metadata={"allow_weak": True},  # Allow weak signals for validation
            )
        except Exception as e:
            self._log.warning("signal_parse_failed", value_id=value.id, error=str(e))
            return None

    # --- Message handlers ---

    async def handle_signal(self, signal: Signal) -> None:
        """Handle incoming signals for validation."""
        validation = self._validator.validate(signal)
        self._log.debug(
            "signal_validated",
            signal_id=signal.id,
            result=validation.result.name,
        )

    async def handle_event(self, message: Message) -> None:
        """Handle incoming events."""
        if message.topic.startswith("threshold."):
            self._log.debug("threshold_event_received", topic=message.topic)

    # --- Public API ---

    def validate(self, signal: Signal) -> ValidationReport:
        """Validate a signal against threshold policies."""
        report = self._validator.validate(signal)
        self._total_validated += 1
        if report.result == ValidationResult.PASSED:
            self._total_passed += 1
        elif report.result == ValidationResult.FAILED:
            self._total_failed += 1
        elif report.result == ValidationResult.ADJUSTED:
            self._total_adjusted += 1
        elif report.result == ValidationResult.BLOCKED:
            self._total_blocked += 1
        return report

    def convert(
        self,
        signal: Signal,
        target_domain: SignalDomain,
    ) -> tuple[Signal | None, ConversionReport]:
        """Convert a signal to a target domain."""
        result = self._converter.convert(signal, target_domain)
        self._total_converted += 1
        if result[1].success:
            self._conversion_successes += 1
        return result

    def validate_and_convert(
        self,
        signal: Signal,
        target_domain: SignalDomain | None = None,
    ) -> tuple[Signal | None, ValidationReport, ConversionReport | None]:
        """Validate and optionally convert a signal."""
        validation = self.validate(signal)

        if validation.result == ValidationResult.BLOCKED:
            return None, validation, None

        # Apply adjusted strength if needed
        if validation.result == ValidationResult.ADJUSTED:
            signal = Signal(
                domain=signal.domain,
                payload=signal.payload,
                payload_type=signal.payload_type,
                source=signal.source,
                target=signal.target,
                correlation_id=signal.correlation_id,
                strength=validation.final_strength,
                threshold=signal.threshold,
                priority=signal.priority,
                tags=signal.tags | frozenset(["adjusted"]),
                metadata={**signal.metadata, "allow_weak": True},
            )

        # Convert if requested
        conversion: ConversionReport | None = None
        result_signal: Signal | None = signal
        if target_domain and target_domain != signal.domain:
            result_signal, conversion = self.convert(signal, target_domain)

        return result_signal, validation, conversion

    def add_threshold_policy(self, policy: ThresholdPolicy) -> None:
        """Add a threshold policy."""
        self._validator.add_policy(policy)

    def add_conversion_policy(self, policy: ConversionPolicy) -> None:
        """Add a conversion policy."""
        self._converter.add_policy(policy)

    def get_stats(self) -> GuardStats:
        """Get guard statistics."""
        conversion_rate = (
            self._conversion_successes / self._total_converted
            if self._total_converted > 0
            else 1.0
        )
        return GuardStats(
            total_validated=self._total_validated,
            total_passed=self._total_passed,
            total_failed=self._total_failed,
            total_adjusted=self._total_adjusted,
            total_blocked=self._total_blocked,
            total_converted=self._total_converted,
            conversion_success_rate=conversion_rate,
        )

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._total_validated = 0
        self._total_passed = 0
        self._total_failed = 0
        self._total_adjusted = 0
        self._total_blocked = 0
        self._total_converted = 0
        self._conversion_successes = 0
