"""
ValueExchangeManager: Manages symbolic trade and value exchange.

Governs trade and exchange of value, ensures balanced distribution and
recognition of symbolic resources, facilitating fair and consistent transactions.
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


class CurrencyType(Enum):
    """Types of symbolic currencies."""

    TOKEN = auto()  # Generic token # allow-secret
    CREDIT = auto()  # System credit
    POINT = auto()  # Reward points
    SHARE = auto()  # Ownership share
    VOUCHER = auto()  # Redeemable voucher
    SYMBOLIC = auto()  # Abstract symbolic value


class TransactionStatus(Enum):
    """Status of a transaction."""

    PENDING = auto()  # Awaiting execution
    COMPLETED = auto()  # Successfully completed
    FAILED = auto()  # Failed to execute
    REVERSED = auto()  # Rolled back
    CANCELLED = auto()  # Cancelled before execution


class TransactionType(Enum):
    """Types of transactions."""

    TRANSFER = auto()  # Transfer between accounts
    DEPOSIT = auto()  # Add value to account
    WITHDRAWAL = auto()  # Remove value from account
    EXCHANGE = auto()  # Currency exchange
    FEE = auto()  # Service fee
    REWARD = auto()  # Reward payout


class Account(BaseModel):
    """A value account."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    name: str
    owner_id: str
    currency: CurrencyType = CurrencyType.TOKEN
    balance: Decimal = Decimal("0")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    is_active: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class Transaction(BaseModel):
    """A value transaction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    transaction_type: TransactionType
    status: TransactionStatus = TransactionStatus.PENDING
    currency: CurrencyType = CurrencyType.TOKEN
    amount: Decimal

    # Parties
    source_account_id: str | None = None  # None for deposits
    target_account_id: str | None = None  # None for withdrawals

    # For exchanges
    source_currency: CurrencyType | None = None
    target_currency: CurrencyType | None = None
    exchange_rate: Decimal | None = None

    # Metadata
    description: str = ""
    reference: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    executed_at: datetime | None = None
    tags: frozenset[str] = Field(default_factory=frozenset)


class ExchangeRate(BaseModel):
    """An exchange rate between currencies."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: CurrencyType
    target: CurrencyType
    rate: Decimal
    valid_from: datetime = Field(default_factory=lambda: datetime.now(UTC))
    valid_until: datetime | None = None


@dataclass
class TransactionResult:
    """Result of a transaction."""

    transaction_id: str
    success: bool
    status: TransactionStatus
    source_balance: Decimal | None = None
    target_balance: Decimal | None = None
    error: str | None = None


@dataclass
class ExchangeStats:
    """Statistics about value exchange."""

    total_accounts: int
    active_accounts: int
    total_transactions: int
    completed_transactions: int
    total_volume: dict[str, Decimal]  # By currency
    pending_transactions: int


class AccountRegistry:
    """Registry of accounts."""

    def __init__(self) -> None:
        self._accounts: dict[str, Account] = {}
        self._by_owner: dict[str, set[str]] = {}  # owner_id -> account IDs
        self._by_currency: dict[CurrencyType, set[str]] = {}
        self._log = logger.bind(component="account_registry")

    @property
    def account_count(self) -> int:
        return len(self._accounts)

    def create_account(self, account: Account) -> None:
        """Create a new account."""
        self._accounts[account.id] = account
        self._by_owner.setdefault(account.owner_id, set()).add(account.id)
        self._by_currency.setdefault(account.currency, set()).add(account.id)
        self._log.debug("account_created", account_id=account.id, owner=account.owner_id)

    def get_account(self, account_id: str) -> Account | None:
        """Get an account by ID."""
        return self._accounts.get(account_id)

    def get_by_owner(self, owner_id: str) -> list[Account]:
        """Get accounts by owner."""
        account_ids = self._by_owner.get(owner_id, set())
        return [self._accounts[aid] for aid in account_ids if aid in self._accounts]

    def get_by_currency(self, currency: CurrencyType) -> list[Account]:
        """Get accounts by currency."""
        account_ids = self._by_currency.get(currency, set())
        return [self._accounts[aid] for aid in account_ids if aid in self._accounts]

    def update_balance(self, account_id: str, new_balance: Decimal) -> Account | None:
        """Update account balance."""
        account = self._accounts.get(account_id)
        if not account:
            return None

        updated = Account(
            id=account.id,
            name=account.name,
            owner_id=account.owner_id,
            currency=account.currency,
            balance=new_balance,
            created_at=account.created_at,
            is_active=account.is_active,
            metadata=account.metadata,
        )
        self._accounts[account_id] = updated
        return updated

    def deactivate(self, account_id: str) -> bool:
        """Deactivate an account."""
        account = self._accounts.get(account_id)
        if not account:
            return False

        updated = Account(
            id=account.id,
            name=account.name,
            owner_id=account.owner_id,
            currency=account.currency,
            balance=account.balance,
            created_at=account.created_at,
            is_active=False,
            metadata=account.metadata,
        )
        self._accounts[account_id] = updated
        return True

    def get_active_count(self) -> int:
        """Get count of active accounts."""
        return len([a for a in self._accounts.values() if a.is_active])


class TransactionProcessor:
    """Processes value transactions."""

    def __init__(self, registry: AccountRegistry) -> None:
        self._registry = registry
        self._transactions: dict[str, Transaction] = {}
        self._exchange_rates: dict[tuple[CurrencyType, CurrencyType], ExchangeRate] = {}
        self._log = logger.bind(component="transaction_processor")

    @property
    def transaction_count(self) -> int:
        return len(self._transactions)

    def set_exchange_rate(self, rate: ExchangeRate) -> None:
        """Set an exchange rate."""
        self._exchange_rates[(rate.source, rate.target)] = rate
        self._log.debug(
            "rate_set",
            source=rate.source.name,
            target=rate.target.name,
            rate=str(rate.rate),
        )

    def get_exchange_rate(
        self,
        source: CurrencyType,
        target: CurrencyType,
    ) -> ExchangeRate | None:
        """Get the exchange rate between currencies."""
        return self._exchange_rates.get((source, target))

    def execute(self, transaction: Transaction) -> TransactionResult:
        """Execute a transaction."""
        self._transactions[transaction.id] = transaction

        # Validate
        validation_error = self._validate(transaction)
        if validation_error:
            return self._fail_transaction(transaction, validation_error)

        # Execute based on type
        if transaction.transaction_type == TransactionType.DEPOSIT:
            return self._execute_deposit(transaction)
        elif transaction.transaction_type == TransactionType.WITHDRAWAL:
            return self._execute_withdrawal(transaction)
        elif transaction.transaction_type == TransactionType.TRANSFER:
            return self._execute_transfer(transaction)
        elif transaction.transaction_type == TransactionType.EXCHANGE:
            return self._execute_exchange(transaction)
        else:
            return self._fail_transaction(transaction, f"Unknown type: {transaction.transaction_type}")

    def _validate(self, transaction: Transaction) -> str | None:
        """Validate a transaction."""
        if transaction.amount <= 0:
            return "Amount must be positive"

        if transaction.transaction_type in (TransactionType.WITHDRAWAL, TransactionType.TRANSFER):
            if not transaction.source_account_id:
                return "Source account required"
            source = self._registry.get_account(transaction.source_account_id)
            if not source:
                return "Source account not found"
            if not source.is_active:
                return "Source account inactive"
            if source.balance < transaction.amount:
                return "Insufficient balance"

        if transaction.transaction_type in (TransactionType.DEPOSIT, TransactionType.TRANSFER):
            if not transaction.target_account_id:
                return "Target account required"
            target = self._registry.get_account(transaction.target_account_id)
            if not target:
                return "Target account not found"
            if not target.is_active:
                return "Target account inactive"

        if transaction.transaction_type == TransactionType.EXCHANGE:
            if not transaction.source_currency or not transaction.target_currency:
                return "Exchange currencies required"
            rate = self.get_exchange_rate(
                transaction.source_currency,
                transaction.target_currency,
            )
            if not rate:
                return "No exchange rate available"

        return None

    def _execute_deposit(self, transaction: Transaction) -> TransactionResult:
        """Execute a deposit."""
        assert transaction.target_account_id is not None  # Validated in _validate
        target = self._registry.get_account(transaction.target_account_id)
        assert target is not None  # Validated in _validate
        new_balance = target.balance + transaction.amount
        self._registry.update_balance(target.id, new_balance)

        return self._complete_transaction(
            transaction,
            target_balance=new_balance,
        )

    def _execute_withdrawal(self, transaction: Transaction) -> TransactionResult:
        """Execute a withdrawal."""
        assert transaction.source_account_id is not None  # Validated in _validate
        source = self._registry.get_account(transaction.source_account_id)
        assert source is not None  # Validated in _validate
        new_balance = source.balance - transaction.amount
        self._registry.update_balance(source.id, new_balance)

        return self._complete_transaction(
            transaction,
            source_balance=new_balance,
        )

    def _execute_transfer(self, transaction: Transaction) -> TransactionResult:
        """Execute a transfer."""
        assert transaction.source_account_id is not None  # Validated in _validate
        assert transaction.target_account_id is not None  # Validated in _validate
        source = self._registry.get_account(transaction.source_account_id)
        target = self._registry.get_account(transaction.target_account_id)
        assert source is not None  # Validated in _validate
        assert target is not None  # Validated in _validate

        source_balance = source.balance - transaction.amount
        target_balance = target.balance + transaction.amount

        self._registry.update_balance(source.id, source_balance)
        self._registry.update_balance(target.id, target_balance)

        return self._complete_transaction(
            transaction,
            source_balance=source_balance,
            target_balance=target_balance,
        )

    def _execute_exchange(self, transaction: Transaction) -> TransactionResult:
        """Execute a currency exchange."""
        assert transaction.source_currency is not None  # Validated in _validate
        assert transaction.target_currency is not None  # Validated in _validate
        rate = self.get_exchange_rate(
            transaction.source_currency,
            transaction.target_currency,
        )
        assert rate is not None  # Validated in _validate

        assert transaction.source_account_id is not None  # Validated in _validate
        assert transaction.target_account_id is not None  # Validated in _validate
        source = self._registry.get_account(transaction.source_account_id)
        target = self._registry.get_account(transaction.target_account_id)
        assert source is not None  # Validated in _validate
        assert target is not None  # Validated in _validate

        # Calculate converted amount
        converted_amount = transaction.amount * rate.rate

        source_balance = source.balance - transaction.amount
        target_balance = target.balance + converted_amount

        self._registry.update_balance(source.id, source_balance)
        self._registry.update_balance(target.id, target_balance)

        # Update transaction with exchange rate
        updated = Transaction(
            id=transaction.id,
            transaction_type=transaction.transaction_type,
            status=TransactionStatus.COMPLETED,
            currency=transaction.currency,
            amount=transaction.amount,
            source_account_id=transaction.source_account_id,
            target_account_id=transaction.target_account_id,
            source_currency=transaction.source_currency,
            target_currency=transaction.target_currency,
            exchange_rate=rate.rate,
            description=transaction.description,
            reference=transaction.reference,
            created_at=transaction.created_at,
            executed_at=datetime.now(UTC),
            tags=transaction.tags,
        )
        self._transactions[transaction.id] = updated

        return TransactionResult(
            transaction_id=transaction.id,
            success=True,
            status=TransactionStatus.COMPLETED,
            source_balance=source_balance,
            target_balance=target_balance,
        )

    def _complete_transaction(
        self,
        transaction: Transaction,
        source_balance: Decimal | None = None,
        target_balance: Decimal | None = None,
    ) -> TransactionResult:
        """Mark a transaction as completed."""
        updated = Transaction(
            id=transaction.id,
            transaction_type=transaction.transaction_type,
            status=TransactionStatus.COMPLETED,
            currency=transaction.currency,
            amount=transaction.amount,
            source_account_id=transaction.source_account_id,
            target_account_id=transaction.target_account_id,
            source_currency=transaction.source_currency,
            target_currency=transaction.target_currency,
            exchange_rate=transaction.exchange_rate,
            description=transaction.description,
            reference=transaction.reference,
            created_at=transaction.created_at,
            executed_at=datetime.now(UTC),
            tags=transaction.tags,
        )
        self._transactions[transaction.id] = updated

        return TransactionResult(
            transaction_id=transaction.id,
            success=True,
            status=TransactionStatus.COMPLETED,
            source_balance=source_balance,
            target_balance=target_balance,
        )

    def _fail_transaction(
        self,
        transaction: Transaction,
        error: str,
    ) -> TransactionResult:
        """Mark a transaction as failed."""
        updated = Transaction(
            id=transaction.id,
            transaction_type=transaction.transaction_type,
            status=TransactionStatus.FAILED,
            currency=transaction.currency,
            amount=transaction.amount,
            source_account_id=transaction.source_account_id,
            target_account_id=transaction.target_account_id,
            source_currency=transaction.source_currency,
            target_currency=transaction.target_currency,
            exchange_rate=transaction.exchange_rate,
            description=transaction.description,
            reference=transaction.reference,
            created_at=transaction.created_at,
            tags=transaction.tags,
        )
        self._transactions[transaction.id] = updated

        return TransactionResult(
            transaction_id=transaction.id,
            success=False,
            status=TransactionStatus.FAILED,
            error=error,
        )

    def reverse_transaction(self, transaction_id: str) -> TransactionResult | None:
        """Reverse a completed transaction."""
        transaction = self._transactions.get(transaction_id)
        if not transaction:
            return None
        if transaction.status != TransactionStatus.COMPLETED:
            return TransactionResult(
                transaction_id=transaction_id,
                success=False,
                status=transaction.status,
                error="Can only reverse completed transactions",
            )

        # Create reverse transaction
        if transaction.transaction_type == TransactionType.TRANSFER:
            reverse = Transaction(
                transaction_type=TransactionType.TRANSFER,
                currency=transaction.currency,
                amount=transaction.amount,
                source_account_id=transaction.target_account_id,
                target_account_id=transaction.source_account_id,
                description=f"Reversal of {transaction_id}",
                reference=transaction_id,
            )
            result = self.execute(reverse)

            if result.success:
                # Mark original as reversed
                updated = Transaction(
                    id=transaction.id,
                    transaction_type=transaction.transaction_type,
                    status=TransactionStatus.REVERSED,
                    currency=transaction.currency,
                    amount=transaction.amount,
                    source_account_id=transaction.source_account_id,
                    target_account_id=transaction.target_account_id,
                    source_currency=transaction.source_currency,
                    target_currency=transaction.target_currency,
                    exchange_rate=transaction.exchange_rate,
                    description=transaction.description,
                    reference=transaction.reference,
                    created_at=transaction.created_at,
                    executed_at=transaction.executed_at,
                    tags=transaction.tags,
                )
                self._transactions[transaction_id] = updated

            return result

        return TransactionResult(
            transaction_id=transaction_id,
            success=False,
            status=transaction.status,
            error="Cannot reverse this transaction type",
        )

    def get_transaction(self, transaction_id: str) -> Transaction | None:
        """Get a transaction by ID."""
        return self._transactions.get(transaction_id)

    def get_by_status(self, status: TransactionStatus) -> list[Transaction]:
        """Get transactions by status."""
        return [t for t in self._transactions.values() if t.status == status]


class ValueExchangeManager(Subsystem):
    """
    Manages symbolic trade and value exchange.

    Process Loop:
    1. Request: Receive exchange or trade requests
    2. Validate: Check input values and applicable rules
    3. Execute: Complete the exchange and update balances
    4. Record: Log transaction outcomes for audit and analysis
    """

    def __init__(self) -> None:
        metadata = SubsystemMetadata(
            name="value_exchange_manager",
            display_name="Value Exchange Manager",
            description="Manages symbolic trade and value exchange",
            type=SubsystemType.VALUE,
            tags=frozenset(["value", "exchange", "trade", "currency"]),
            input_types=frozenset(["TOKEN", "CURRENCY", "ASSET"]),
            output_types=frozenset(["TOKEN", "CURRENCY", "TRANSACTION"]),
            subscribed_topics=frozenset([
                str(SubsystemTopics.ALL),
                "value.#",
                "exchange.#",
            ]),
            published_topics=frozenset([
                "value.transaction.completed",
                "value.transaction.failed",
                "value.account.created",
            ]),
        )
        super().__init__(metadata)

        self._registry = AccountRegistry()
        self._processor = TransactionProcessor(self._registry)

    @property
    def account_count(self) -> int:
        return self._registry.account_count

    @property
    def transaction_count(self) -> int:
        return self._processor.transaction_count

    async def intake(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> SymbolicInput:
        """Phase 1: Receive exchange or trade requests."""
        if not input_data.values:
            self._log.debug("empty_input")
            return input_data

        self._log.debug("intake_complete", value_count=len(input_data.values))
        return input_data

    async def process(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> list[TransactionResult]:
        """Phase 2: Validate and execute exchanges."""
        results: list[TransactionResult] = []

        for value in input_data.values:
            content = value.content
            if not isinstance(content, dict):
                continue

            action = content.get("action", "transfer")

            if action == "create_account":
                account = self._create_account_from_value(value)
                if account:
                    self._registry.create_account(account)
                    # Return a pseudo-result for account creation
                    results.append(TransactionResult(
                        transaction_id=account.id,
                        success=True,
                        status=TransactionStatus.COMPLETED,
                    ))

            elif action in ("transfer", "deposit", "withdrawal", "exchange"):
                transaction = self._create_transaction_from_value(value)
                if transaction:
                    result = self._processor.execute(transaction)
                    results.append(result)

        return results

    async def evaluate(
        self,
        intermediate: list[TransactionResult],
        ctx: ProcessContext[dict[str, Any]],
    ) -> tuple[SymbolicOutput, bool]:
        """Phase 3: Create output from transaction results."""
        values: list[SymbolicValue] = []

        for result in intermediate:
            value = SymbolicValue(
                type=SymbolicValueType.REFERENCE,
                content={
                    "transaction_id": result.transaction_id,
                    "success": result.success,
                    "status": result.status.name,
                    "source_balance": str(result.source_balance) if result.source_balance else None,
                    "target_balance": str(result.target_balance) if result.target_balance else None,
                    "error": result.error,
                },
                source_subsystem=self.name,
                tags=frozenset(["transaction", result.status.name.lower()]),
                meaning=f"Transaction: {result.status.name}",
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
        """Phase 4: Emit events for transactions."""
        if self._message_bus and output.values:
            for value in output.values:
                content = value.content
                if not isinstance(content, dict):
                    continue

                if content.get("success"):
                    await self.emit_event(
                        "value.transaction.completed",
                        {
                            "transaction_id": content.get("transaction_id"),
                            "status": content.get("status"),
                        },
                    )
                else:
                    await self.emit_event(
                        "value.transaction.failed",
                        {
                            "transaction_id": content.get("transaction_id"),
                            "error": content.get("error"),
                        },
                    )

        return None

    def _create_account_from_value(self, value: SymbolicValue) -> Account | None:
        """Create an Account from a SymbolicValue."""
        content = value.content
        if not isinstance(content, dict):
            return None

        try:
            currency_str = content.get("currency", "TOKEN")
            try:
                currency = CurrencyType[currency_str.upper()]
            except KeyError:
                currency = CurrencyType.TOKEN

            balance_str = content.get("balance", "0")
            balance = Decimal(str(balance_str))

            return Account(
                id=content.get("id", str(ULID())),
                name=content.get("name", f"account_{value.id[:8]}"),
                owner_id=content.get("owner_id", "unknown"),
                currency=currency,
                balance=balance,
                metadata=content.get("metadata", {}),
            )
        except Exception as e:
            self._log.warning("account_parse_failed", value_id=value.id, error=str(e))
            return None

    def _create_transaction_from_value(self, value: SymbolicValue) -> Transaction | None:
        """Create a Transaction from a SymbolicValue."""
        content = value.content
        if not isinstance(content, dict):
            return None

        try:
            action = content.get("action", "transfer")
            type_map = {
                "transfer": TransactionType.TRANSFER,
                "deposit": TransactionType.DEPOSIT,
                "withdrawal": TransactionType.WITHDRAWAL,
                "exchange": TransactionType.EXCHANGE,
            }
            transaction_type = type_map.get(action, TransactionType.TRANSFER)

            currency_str = content.get("currency", "TOKEN")
            try:
                currency = CurrencyType[currency_str.upper()]
            except KeyError:
                currency = CurrencyType.TOKEN

            amount = Decimal(str(content.get("amount", "0")))

            source_currency = None
            target_currency = None
            if transaction_type == TransactionType.EXCHANGE:
                if sc := content.get("source_currency"):
                    source_currency = CurrencyType[sc.upper()]
                if tc := content.get("target_currency"):
                    target_currency = CurrencyType[tc.upper()]

            return Transaction(
                id=content.get("id", str(ULID())),
                transaction_type=transaction_type,
                currency=currency,
                amount=amount,
                source_account_id=content.get("source_account_id"),
                target_account_id=content.get("target_account_id"),
                source_currency=source_currency,
                target_currency=target_currency,
                description=content.get("description", ""),
                reference=content.get("reference"),
                tags=frozenset(content.get("tags", [])) | value.tags,
            )
        except Exception as e:
            self._log.warning("transaction_parse_failed", value_id=value.id, error=str(e))
            return None

    # --- Message handlers ---

    async def handle_event(self, message: Message) -> None:
        """Handle incoming events."""
        if message.topic.startswith("value."):
            self._log.debug("value_event_received", topic=message.topic)

    async def handle_signal(self, signal: Any) -> None:
        """Handle incoming signals."""
        self._log.debug("signal_received", signal_id=getattr(signal, "id", "unknown"))

    # --- Public API ---

    def create_account(
        self,
        name: str,
        owner_id: str,
        currency: CurrencyType = CurrencyType.TOKEN,
        initial_balance: Decimal = Decimal("0"),
        **kwargs: Any,
    ) -> Account:
        """Create a new account."""
        account = Account(
            name=name,
            owner_id=owner_id,
            currency=currency,
            balance=initial_balance,
            metadata=kwargs.get("metadata", {}),
        )
        self._registry.create_account(account)
        return account

    def get_account(self, account_id: str) -> Account | None:
        """Get an account by ID."""
        return self._registry.get_account(account_id)

    def get_accounts_by_owner(self, owner_id: str) -> list[Account]:
        """Get accounts by owner."""
        return self._registry.get_by_owner(owner_id)

    def deposit(
        self,
        account_id: str,
        amount: Decimal,
        **kwargs: Any,
    ) -> TransactionResult:
        """Deposit value into an account."""
        transaction = Transaction(
            transaction_type=TransactionType.DEPOSIT,
            target_account_id=account_id,
            amount=amount,
            description=kwargs.get("description", ""),
            reference=kwargs.get("reference"),
        )
        return self._processor.execute(transaction)

    def withdraw(
        self,
        account_id: str,
        amount: Decimal,
        **kwargs: Any,
    ) -> TransactionResult:
        """Withdraw value from an account."""
        transaction = Transaction(
            transaction_type=TransactionType.WITHDRAWAL,
            source_account_id=account_id,
            amount=amount,
            description=kwargs.get("description", ""),
            reference=kwargs.get("reference"),
        )
        return self._processor.execute(transaction)

    def transfer(
        self,
        source_account_id: str,
        target_account_id: str,
        amount: Decimal,
        **kwargs: Any,
    ) -> TransactionResult:
        """Transfer value between accounts."""
        transaction = Transaction(
            transaction_type=TransactionType.TRANSFER,
            source_account_id=source_account_id,
            target_account_id=target_account_id,
            amount=amount,
            description=kwargs.get("description", ""),
            reference=kwargs.get("reference"),
        )
        return self._processor.execute(transaction)

    def exchange(
        self,
        source_account_id: str,
        target_account_id: str,
        amount: Decimal,
        source_currency: CurrencyType,
        target_currency: CurrencyType,
        **kwargs: Any,
    ) -> TransactionResult:
        """Exchange between currencies."""
        transaction = Transaction(
            transaction_type=TransactionType.EXCHANGE,
            source_account_id=source_account_id,
            target_account_id=target_account_id,
            amount=amount,
            source_currency=source_currency,
            target_currency=target_currency,
            description=kwargs.get("description", ""),
            reference=kwargs.get("reference"),
        )
        return self._processor.execute(transaction)

    def set_exchange_rate(
        self,
        source: CurrencyType,
        target: CurrencyType,
        rate: Decimal,
    ) -> ExchangeRate:
        """Set an exchange rate."""
        exchange_rate = ExchangeRate(
            source=source,
            target=target,
            rate=rate,
        )
        self._processor.set_exchange_rate(exchange_rate)
        return exchange_rate

    def get_exchange_rate(
        self,
        source: CurrencyType,
        target: CurrencyType,
    ) -> ExchangeRate | None:
        """Get an exchange rate."""
        return self._processor.get_exchange_rate(source, target)

    def reverse_transaction(self, transaction_id: str) -> TransactionResult | None:
        """Reverse a transaction."""
        return self._processor.reverse_transaction(transaction_id)

    def get_transaction(self, transaction_id: str) -> Transaction | None:
        """Get a transaction by ID."""
        return self._processor.get_transaction(transaction_id)

    def get_stats(self) -> ExchangeStats:
        """Get exchange statistics."""
        completed = self._processor.get_by_status(TransactionStatus.COMPLETED)
        pending = self._processor.get_by_status(TransactionStatus.PENDING)

        # Calculate volume by currency
        volume: dict[str, Decimal] = {}
        for t in completed:
            key = t.currency.name
            volume[key] = volume.get(key, Decimal("0")) + t.amount

        return ExchangeStats(
            total_accounts=self._registry.account_count,
            active_accounts=self._registry.get_active_count(),
            total_transactions=self._processor.transaction_count,
            completed_transactions=len(completed),
            total_volume=volume,
            pending_transactions=len(pending),
        )

    def clear(self) -> tuple[int, int]:
        """Clear all data. Returns (accounts, transactions) cleared."""
        accounts = self._registry.account_count
        transactions = self._processor.transaction_count
        self._registry = AccountRegistry()
        self._processor = TransactionProcessor(self._registry)
        return accounts, transactions
