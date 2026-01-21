"""
ProcessMonetizer: Converts processes into monetizable products or services.

Transforms processes into products or services with assigned value,
quantifies and tracks value produced by system workflows.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
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


class ProductType(Enum):
    """Types of monetizable products."""

    SERVICE = auto()  # One-time service
    SUBSCRIPTION = auto()  # Recurring access
    LICENSE = auto()  # Usage license
    CONSUMABLE = auto()  # One-time use item
    ASSET = auto()  # Transferable asset
    ACCESS = auto()  # Access rights


class RevenueModel(Enum):
    """Revenue models for products."""

    FIXED_PRICE = auto()  # Fixed one-time price
    USAGE_BASED = auto()  # Pay per use
    SUBSCRIPTION = auto()  # Recurring payments
    FREEMIUM = auto()  # Free base, paid premium
    COMMISSION = auto()  # Percentage of transaction
    AUCTION = auto()  # Dynamic pricing


class ProcessStatus(Enum):
    """Status of a process in monetization."""

    DRAFT = auto()  # Being defined
    ACTIVE = auto()  # Available for monetization
    PAUSED = auto()  # Temporarily unavailable
    DEPRECATED = auto()  # No longer available
    ARCHIVED = auto()  # Historical record


class PayoutStatus(Enum):
    """Status of a payout."""

    PENDING = auto()  # Awaiting processing
    PROCESSED = auto()  # Completed
    FAILED = auto()  # Failed to process
    CANCELLED = auto()  # Cancelled


class MonetizableProcess(BaseModel):
    """A process that can be monetized."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    name: str
    description: str = ""
    status: ProcessStatus = ProcessStatus.DRAFT

    # Source process info
    source_subsystem: str | None = None
    source_process_id: str | None = None

    # Product info
    product_type: ProductType = ProductType.SERVICE
    revenue_model: RevenueModel = RevenueModel.FIXED_PRICE

    # Pricing
    base_price: Decimal = Decimal("0")
    currency: str = "TOKEN"
    usage_rate: Decimal = Decimal("0")  # For usage-based

    # Metrics
    total_revenue: Decimal = Decimal("0")
    usage_count: int = 0
    payout_count: int = 0

    # Metadata
    owner_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tags: frozenset[str] = Field(default_factory=frozenset)


class Product(BaseModel):
    """A monetizable product derived from a process."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    process_id: str
    name: str
    description: str = ""
    product_type: ProductType = ProductType.SERVICE

    # Pricing
    price: Decimal
    currency: str = "TOKEN"

    # Availability
    available: bool = True
    inventory: int | None = None  # None = unlimited
    sold_count: int = 0

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tags: frozenset[str] = Field(default_factory=frozenset)


class UsageRecord(BaseModel):
    """Record of product usage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    process_id: str
    product_id: str | None = None
    user_id: str

    # Usage info
    units: Decimal = Decimal("1")
    unit_price: Decimal = Decimal("0")
    total_value: Decimal = Decimal("0")

    # Metadata
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class Payout(BaseModel):
    """A payout to a process owner."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    process_id: str
    recipient_id: str
    status: PayoutStatus = PayoutStatus.PENDING

    # Amount
    amount: Decimal
    currency: str = "TOKEN"
    fee: Decimal = Decimal("0")
    net_amount: Decimal = Decimal("0")

    # Period
    period_start: datetime | None = None
    period_end: datetime | None = None

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    processed_at: datetime | None = None
    reference: str | None = None


@dataclass
class ValuationResult:
    """Result of process valuation."""

    process_id: str
    estimated_value: Decimal
    factors: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.5


@dataclass
class MonetizationResult:
    """Result of monetizing a process."""

    process_id: str
    product_id: str | None
    success: bool
    price: Decimal = Decimal("0")
    error: str | None = None


@dataclass
class MonetizationStats:
    """Statistics about monetization."""

    total_processes: int
    active_processes: int
    total_products: int
    total_revenue: Decimal
    total_payouts: int
    pending_payouts: Decimal
    average_price: Decimal


class ProcessRegistry:
    """Registry of monetizable processes."""

    def __init__(self) -> None:
        self._processes: dict[str, MonetizableProcess] = {}
        self._by_owner: dict[str, set[str]] = {}
        self._by_status: dict[ProcessStatus, set[str]] = {}
        self._log = logger.bind(component="process_registry")

    @property
    def process_count(self) -> int:
        return len(self._processes)

    def add(self, process: MonetizableProcess) -> None:
        """Add a process to the registry."""
        self._processes[process.id] = process
        if process.owner_id:
            self._by_owner.setdefault(process.owner_id, set()).add(process.id)
        self._by_status.setdefault(process.status, set()).add(process.id)
        self._log.debug("process_added", process_id=process.id, name=process.name)

    def get(self, process_id: str) -> MonetizableProcess | None:
        """Get a process by ID."""
        return self._processes.get(process_id)

    def get_by_owner(self, owner_id: str) -> list[MonetizableProcess]:
        """Get processes by owner."""
        process_ids = self._by_owner.get(owner_id, set())
        return [self._processes[pid] for pid in process_ids if pid in self._processes]

    def get_by_status(self, status: ProcessStatus) -> list[MonetizableProcess]:
        """Get processes by status."""
        process_ids = self._by_status.get(status, set())
        return [self._processes[pid] for pid in process_ids if pid in self._processes]

    def update(self, process: MonetizableProcess) -> bool:
        """Update a process."""
        if process.id not in self._processes:
            return False

        old = self._processes[process.id]
        self._by_status[old.status].discard(process.id)
        self._by_status.setdefault(process.status, set()).add(process.id)
        self._processes[process.id] = process
        return True

    def get_active_count(self) -> int:
        """Get count of active processes."""
        return len(self._by_status.get(ProcessStatus.ACTIVE, set()))


class ProductCatalog:
    """Catalog of products."""

    def __init__(self) -> None:
        self._products: dict[str, Product] = {}
        self._by_process: dict[str, set[str]] = {}
        self._log = logger.bind(component="product_catalog")

    @property
    def product_count(self) -> int:
        return len(self._products)

    def add(self, product: Product) -> None:
        """Add a product to the catalog."""
        self._products[product.id] = product
        self._by_process.setdefault(product.process_id, set()).add(product.id)
        self._log.debug("product_added", product_id=product.id, name=product.name)

    def get(self, product_id: str) -> Product | None:
        """Get a product by ID."""
        return self._products.get(product_id)

    def get_by_process(self, process_id: str) -> list[Product]:
        """Get products for a process."""
        product_ids = self._by_process.get(process_id, set())
        return [self._products[pid] for pid in product_ids if pid in self._products]

    def update(self, product: Product) -> bool:
        """Update a product."""
        if product.id not in self._products:
            return False
        self._products[product.id] = product
        return True


class PayoutManager:
    """Manages payouts to process owners."""

    def __init__(self, commission_rate: Decimal = Decimal("0.1")) -> None:
        self._payouts: dict[str, Payout] = {}
        self._by_process: dict[str, set[str]] = {}
        self._by_recipient: dict[str, set[str]] = {}
        self._commission_rate = commission_rate
        self._log = logger.bind(component="payout_manager")

    @property
    def payout_count(self) -> int:
        return len(self._payouts)

    def create_payout(
        self,
        process_id: str,
        recipient_id: str,
        amount: Decimal,
        currency: str = "TOKEN",
        **kwargs: Any,
    ) -> Payout:
        """Create a payout."""
        fee = amount * self._commission_rate
        net_amount = amount - fee

        payout = Payout(
            process_id=process_id,
            recipient_id=recipient_id,
            amount=amount,
            currency=currency,
            fee=fee,
            net_amount=net_amount,
            period_start=kwargs.get("period_start"),
            period_end=kwargs.get("period_end"),
            reference=kwargs.get("reference"),
        )

        self._payouts[payout.id] = payout
        self._by_process.setdefault(process_id, set()).add(payout.id)
        self._by_recipient.setdefault(recipient_id, set()).add(payout.id)

        return payout

    def get_payout(self, payout_id: str) -> Payout | None:
        """Get a payout by ID."""
        return self._payouts.get(payout_id)

    def process_payout(self, payout_id: str) -> Payout | None:
        """Mark a payout as processed."""
        payout = self._payouts.get(payout_id)
        if not payout or payout.status != PayoutStatus.PENDING:
            return None

        processed = Payout(
            id=payout.id,
            process_id=payout.process_id,
            recipient_id=payout.recipient_id,
            status=PayoutStatus.PROCESSED,
            amount=payout.amount,
            currency=payout.currency,
            fee=payout.fee,
            net_amount=payout.net_amount,
            period_start=payout.period_start,
            period_end=payout.period_end,
            created_at=payout.created_at,
            processed_at=datetime.now(UTC),
            reference=payout.reference,
        )
        self._payouts[payout_id] = processed
        return processed

    def get_by_status(self, status: PayoutStatus) -> list[Payout]:
        """Get payouts by status."""
        return [p for p in self._payouts.values() if p.status == status]

    def get_pending_total(self) -> Decimal:
        """Get total pending payout amount."""
        return sum((p.net_amount for p in self.get_by_status(PayoutStatus.PENDING)), Decimal("0"))


class ProcessMonetizer(Subsystem):
    """
    Converts processes into monetizable products or services.

    Process Loop:
    1. Ingest: Receive defined processes or workflows
    2. Convert: Translate processes into tangible outputs
    3. Value: Assign values and pricing models
    4. Distribute: Release products and track performance
    """

    def __init__(self, commission_rate: Decimal = Decimal("0.1")) -> None:
        metadata = SubsystemMetadata(
            name="process_monetizer",
            display_name="Process Monetizer",
            description="Converts processes into monetizable outputs",
            type=SubsystemType.VALUE,
            tags=frozenset(["monetization", "value", "pricing", "products"]),
            input_types=frozenset(["REFERENCE", "PATTERN", "PROCESS"]),
            output_types=frozenset(["TOKEN", "CURRENCY", "ASSET", "PRODUCT"]),
            subscribed_topics=frozenset([
                str(SubsystemTopics.ALL),
                "monetization.#",
                "process.#",
            ]),
            published_topics=frozenset([
                "monetization.process.created",
                "monetization.product.created",
                "monetization.usage.recorded",
                "monetization.payout.created",
            ]),
        )
        super().__init__(metadata)

        self._registry = ProcessRegistry()
        self._catalog = ProductCatalog()
        self._payout_manager = PayoutManager(commission_rate)
        self._usage_records: list[UsageRecord] = []

    @property
    def process_count(self) -> int:
        return self._registry.process_count

    @property
    def product_count(self) -> int:
        return self._catalog.product_count

    @property
    def payout_count(self) -> int:
        return self._payout_manager.payout_count

    async def intake(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> SymbolicInput:
        """Phase 1: Receive defined processes or workflows."""
        if not input_data.values:
            self._log.debug("empty_input")
            return input_data

        self._log.debug("intake_complete", value_count=len(input_data.values))
        return input_data

    async def process(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> list[MonetizationResult | ValuationResult]:
        """Phase 2: Convert and value processes."""
        results: list[MonetizationResult | ValuationResult] = []

        for value in input_data.values:
            content = value.content
            if not isinstance(content, dict):
                continue

            action = content.get("action", "register")

            if action == "register":
                process = self._create_process_from_value(value)
                if process:
                    self._registry.add(process)
                    results.append(MonetizationResult(
                        process_id=process.id,
                        product_id=None,
                        success=True,
                    ))

            elif action == "value":
                process_id = content.get("process_id")
                if process_id:
                    valuation = self._valuate_process(process_id)
                    if valuation:
                        results.append(valuation)

            elif action == "create_product":
                result = self._create_product_from_value(value)
                if result:
                    results.append(result)

            elif action == "record_usage":
                record = self._create_usage_record(value)
                if record:
                    self._usage_records.append(record)
                    results.append(MonetizationResult(
                        process_id=record.process_id,
                        product_id=record.product_id,
                        success=True,
                        price=record.total_value,
                    ))

        return results

    async def evaluate(
        self,
        intermediate: list[MonetizationResult | ValuationResult],
        ctx: ProcessContext[dict[str, Any]],
    ) -> tuple[SymbolicOutput, bool]:
        """Phase 3: Create output from results."""
        values: list[SymbolicValue] = []

        for result in intermediate:
            if isinstance(result, ValuationResult):
                value = SymbolicValue(
                    type=SymbolicValueType.REFERENCE,
                    content={
                        "process_id": result.process_id,
                        "estimated_value": str(result.estimated_value),
                        "factors": result.factors,
                        "confidence": result.confidence,
                    },
                    source_subsystem=self.name,
                    tags=frozenset(["valuation"]),
                    meaning=f"Valuation: {result.estimated_value}",
                    confidence=result.confidence,
                )
            else:  # MonetizationResult
                value = SymbolicValue(
                    type=SymbolicValueType.REFERENCE,
                    content={
                        "process_id": result.process_id,
                        "product_id": result.product_id,
                        "success": result.success,
                        "price": str(result.price),
                        "error": result.error,
                    },
                    source_subsystem=self.name,
                    tags=frozenset(["monetization", "success" if result.success else "failed"]),
                    meaning=f"Monetization: {'success' if result.success else 'failed'}",
                    confidence=1.0 if result.success else 0.0,
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
        """Phase 4: Emit events."""
        if self._message_bus and output.values:
            for value in output.values:
                content = value.content
                if not isinstance(content, dict):
                    continue

                if "estimated_value" in content:
                    pass  # Valuation - no event needed
                elif content.get("success"):
                    if content.get("product_id"):
                        await self.emit_event(
                            "monetization.product.created",
                            {
                                "process_id": content.get("process_id"),
                                "product_id": content.get("product_id"),
                            },
                        )
                    else:
                        await self.emit_event(
                            "monetization.process.created",
                            {"process_id": content.get("process_id")},
                        )

        return None

    def _create_process_from_value(self, value: SymbolicValue) -> MonetizableProcess | None:
        """Create a MonetizableProcess from a SymbolicValue."""
        content = value.content
        if not isinstance(content, dict):
            return None

        try:
            product_type_str = content.get("product_type", "SERVICE")
            try:
                product_type = ProductType[product_type_str.upper()]
            except KeyError:
                product_type = ProductType.SERVICE

            revenue_model_str = content.get("revenue_model", "FIXED_PRICE")
            try:
                revenue_model = RevenueModel[revenue_model_str.upper()]
            except KeyError:
                revenue_model = RevenueModel.FIXED_PRICE

            return MonetizableProcess(
                id=content.get("id", str(ULID())),
                name=content.get("name", f"process_{value.id[:8]}"),
                description=content.get("description", ""),
                source_subsystem=content.get("source_subsystem"),
                source_process_id=content.get("source_process_id"),
                product_type=product_type,
                revenue_model=revenue_model,
                base_price=Decimal(str(content.get("base_price", "0"))),
                currency=content.get("currency", "TOKEN"),
                usage_rate=Decimal(str(content.get("usage_rate", "0"))),
                owner_id=content.get("owner_id"),
                tags=frozenset(content.get("tags", [])) | value.tags,
            )
        except Exception as e:
            self._log.warning("process_parse_failed", value_id=value.id, error=str(e))
            return None

    def _create_product_from_value(self, value: SymbolicValue) -> MonetizationResult | None:
        """Create a Product from a SymbolicValue."""
        content = value.content
        if not isinstance(content, dict):
            return None

        process_id = content.get("process_id")
        if not process_id:
            return MonetizationResult(
                process_id="",
                product_id=None,
                success=False,
                error="Process ID required",
            )

        process = self._registry.get(process_id)
        if not process:
            return MonetizationResult(
                process_id=process_id,
                product_id=None,
                success=False,
                error="Process not found",
            )

        try:
            product = Product(
                process_id=process_id,
                name=content.get("name", f"Product: {process.name}"),
                description=content.get("description", process.description),
                product_type=process.product_type,
                price=Decimal(str(content.get("price", process.base_price))),
                currency=content.get("currency", process.currency),
                inventory=content.get("inventory"),
                tags=frozenset(content.get("tags", [])) | value.tags,
            )
            self._catalog.add(product)

            return MonetizationResult(
                process_id=process_id,
                product_id=product.id,
                success=True,
                price=product.price,
            )
        except Exception as e:
            return MonetizationResult(
                process_id=process_id,
                product_id=None,
                success=False,
                error=str(e),
            )

    def _create_usage_record(self, value: SymbolicValue) -> UsageRecord | None:
        """Create a UsageRecord from a SymbolicValue."""
        content = value.content
        if not isinstance(content, dict):
            return None

        process_id = content.get("process_id")
        user_id = content.get("user_id")
        if not process_id or not user_id:
            return None

        process = self._registry.get(process_id)
        if not process:
            return None

        units = Decimal(str(content.get("units", "1")))
        unit_price = process.usage_rate if process.revenue_model == RevenueModel.USAGE_BASED else process.base_price
        total_value = units * unit_price

        return UsageRecord(
            process_id=process_id,
            product_id=content.get("product_id"),
            user_id=user_id,
            units=units,
            unit_price=unit_price,
            total_value=total_value,
            metadata=content.get("metadata", {}),
        )

    def _valuate_process(self, process_id: str) -> ValuationResult | None:
        """Calculate a valuation for a process."""
        process = self._registry.get(process_id)
        if not process:
            return None

        factors: dict[str, float] = {}

        # Base value factor
        base_value = float(process.base_price) if process.base_price > 0 else 10.0
        factors["base_price"] = base_value

        # Usage factor
        usage_factor = 1.0 + (process.usage_count * 0.01)
        factors["usage_multiplier"] = usage_factor

        # Revenue factor
        revenue_factor = 1.0 + (float(process.total_revenue) * 0.001)
        factors["revenue_multiplier"] = revenue_factor

        # Calculate estimated value
        estimated_value = Decimal(str(base_value * usage_factor * revenue_factor))

        # Confidence based on data availability
        confidence = min(0.9, 0.3 + (process.usage_count * 0.01))

        return ValuationResult(
            process_id=process_id,
            estimated_value=estimated_value,
            factors=factors,
            confidence=confidence,
        )

    # --- Message handlers ---

    async def handle_event(self, message: Message) -> None:
        """Handle incoming events."""
        if message.topic.startswith("monetization."):
            self._log.debug("monetization_event_received", topic=message.topic)

    async def handle_signal(self, signal: Any) -> None:
        """Handle incoming signals."""
        self._log.debug("signal_received", signal_id=getattr(signal, "id", "unknown"))

    # --- Public API ---

    def register_process(
        self,
        name: str,
        owner_id: str | None = None,
        product_type: ProductType = ProductType.SERVICE,
        revenue_model: RevenueModel = RevenueModel.FIXED_PRICE,
        base_price: Decimal = Decimal("0"),
        **kwargs: Any,
    ) -> MonetizableProcess:
        """Register a process for monetization."""
        process = MonetizableProcess(
            name=name,
            description=kwargs.get("description", ""),
            status=ProcessStatus.DRAFT,
            source_subsystem=kwargs.get("source_subsystem"),
            source_process_id=kwargs.get("source_process_id"),
            product_type=product_type,
            revenue_model=revenue_model,
            base_price=base_price,
            currency=kwargs.get("currency", "TOKEN"),
            usage_rate=kwargs.get("usage_rate", Decimal("0")),
            owner_id=owner_id,
            tags=frozenset(kwargs.get("tags", [])),
        )
        self._registry.add(process)
        return process

    def get_process(self, process_id: str) -> MonetizableProcess | None:
        """Get a process by ID."""
        return self._registry.get(process_id)

    def activate_process(self, process_id: str) -> MonetizableProcess | None:
        """Activate a process for monetization."""
        process = self._registry.get(process_id)
        if not process:
            return None

        updated = MonetizableProcess(
            id=process.id,
            name=process.name,
            description=process.description,
            status=ProcessStatus.ACTIVE,
            source_subsystem=process.source_subsystem,
            source_process_id=process.source_process_id,
            product_type=process.product_type,
            revenue_model=process.revenue_model,
            base_price=process.base_price,
            currency=process.currency,
            usage_rate=process.usage_rate,
            total_revenue=process.total_revenue,
            usage_count=process.usage_count,
            payout_count=process.payout_count,
            owner_id=process.owner_id,
            created_at=process.created_at,
            updated_at=datetime.now(UTC),
            tags=process.tags,
        )
        self._registry.update(updated)
        return updated

    def create_product(
        self,
        process_id: str,
        name: str | None = None,
        price: Decimal | None = None,
        **kwargs: Any,
    ) -> Product | None:
        """Create a product from a process."""
        process = self._registry.get(process_id)
        if not process:
            return None

        product = Product(
            process_id=process_id,
            name=name or f"Product: {process.name}",
            description=kwargs.get("description", process.description),
            product_type=process.product_type,
            price=price or process.base_price,
            currency=kwargs.get("currency", process.currency),
            inventory=kwargs.get("inventory"),
            tags=frozenset(kwargs.get("tags", [])),
        )
        self._catalog.add(product)
        return product

    def get_product(self, product_id: str) -> Product | None:
        """Get a product by ID."""
        return self._catalog.get(product_id)

    def get_products_for_process(self, process_id: str) -> list[Product]:
        """Get products for a process."""
        return self._catalog.get_by_process(process_id)

    def record_usage(
        self,
        process_id: str,
        user_id: str,
        units: Decimal = Decimal("1"),
        product_id: str | None = None,
        **kwargs: Any,
    ) -> UsageRecord | None:
        """Record usage of a process."""
        process = self._registry.get(process_id)
        if not process:
            return None

        unit_price = (
            process.usage_rate
            if process.revenue_model == RevenueModel.USAGE_BASED
            else process.base_price
        )
        total_value = units * unit_price

        record = UsageRecord(
            process_id=process_id,
            product_id=product_id,
            user_id=user_id,
            units=units,
            unit_price=unit_price,
            total_value=total_value,
            metadata=kwargs.get("metadata", {}),
        )
        self._usage_records.append(record)

        # Update process metrics
        updated = MonetizableProcess(
            id=process.id,
            name=process.name,
            description=process.description,
            status=process.status,
            source_subsystem=process.source_subsystem,
            source_process_id=process.source_process_id,
            product_type=process.product_type,
            revenue_model=process.revenue_model,
            base_price=process.base_price,
            currency=process.currency,
            usage_rate=process.usage_rate,
            total_revenue=process.total_revenue + total_value,
            usage_count=process.usage_count + 1,
            payout_count=process.payout_count,
            owner_id=process.owner_id,
            created_at=process.created_at,
            updated_at=datetime.now(UTC),
            tags=process.tags,
        )
        self._registry.update(updated)

        return record

    def valuate_process(self, process_id: str) -> ValuationResult | None:
        """Calculate a valuation for a process."""
        return self._valuate_process(process_id)

    def create_payout(
        self,
        process_id: str,
        amount: Decimal | None = None,
        **kwargs: Any,
    ) -> Payout | None:
        """Create a payout for a process owner."""
        process = self._registry.get(process_id)
        if not process or not process.owner_id:
            return None

        payout_amount = amount or process.total_revenue

        payout = self._payout_manager.create_payout(
            process_id=process_id,
            recipient_id=process.owner_id,
            amount=payout_amount,
            currency=process.currency,
            **kwargs,
        )

        # Update payout count
        updated = MonetizableProcess(
            id=process.id,
            name=process.name,
            description=process.description,
            status=process.status,
            source_subsystem=process.source_subsystem,
            source_process_id=process.source_process_id,
            product_type=process.product_type,
            revenue_model=process.revenue_model,
            base_price=process.base_price,
            currency=process.currency,
            usage_rate=process.usage_rate,
            total_revenue=process.total_revenue,
            usage_count=process.usage_count,
            payout_count=process.payout_count + 1,
            owner_id=process.owner_id,
            created_at=process.created_at,
            updated_at=datetime.now(UTC),
            tags=process.tags,
        )
        self._registry.update(updated)

        return payout

    def process_payout(self, payout_id: str) -> Payout | None:
        """Process a pending payout."""
        return self._payout_manager.process_payout(payout_id)

    def get_payout(self, payout_id: str) -> Payout | None:
        """Get a payout by ID."""
        return self._payout_manager.get_payout(payout_id)

    def get_stats(self) -> MonetizationStats:
        """Get monetization statistics."""
        total_revenue = Decimal("0")
        for process in self._registry._processes.values():
            total_revenue += process.total_revenue

        prices = [
            p.price for p in self._catalog._products.values()
            if p.price > 0
        ]
        avg_price = Decimal(sum(prices) / len(prices)) if prices else Decimal("0")

        return MonetizationStats(
            total_processes=self._registry.process_count,
            active_processes=self._registry.get_active_count(),
            total_products=self._catalog.product_count,
            total_revenue=total_revenue,
            total_payouts=self._payout_manager.payout_count,
            pending_payouts=self._payout_manager.get_pending_total(),
            average_price=avg_price,
        )

    def clear(self) -> tuple[int, int, int]:
        """Clear all data. Returns (processes, products, usage_records) cleared."""
        processes = self._registry.process_count
        products = self._catalog.product_count
        usage = len(self._usage_records)

        self._registry = ProcessRegistry()
        self._catalog = ProductCatalog()
        self._usage_records.clear()

        return processes, products, usage
