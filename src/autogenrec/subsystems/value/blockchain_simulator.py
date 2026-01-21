"""
BlockchainSimulator: Models distributed ledger logic for symbolic economies.

Tracks actions with verifiable order and persistence, simulates blockchain
behavior to ensure transparency, immutability, and trust.
"""

import hashlib
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


class ConsensusType(Enum):
    """Types of consensus mechanisms."""

    PROOF_OF_WORK = auto()  # Simulated PoW
    PROOF_OF_STAKE = auto()  # Simulated PoS
    AUTHORITY = auto()  # Single authority
    MAJORITY = auto()  # Simple majority
    UNANIMOUS = auto()  # All must agree


class TransactionState(Enum):
    """State of a blockchain transaction."""

    PENDING = auto()  # In mempool
    CONFIRMED = auto()  # In a block
    REJECTED = auto()  # Rejected by validation
    ORPHANED = auto()  # In orphaned block


class BlockchainTransaction(BaseModel):
    """A transaction on the blockchain."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    state: TransactionState = TransactionState.PENDING
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Transaction data
    sender: str
    recipient: str
    data: dict[str, Any] = Field(default_factory=dict)
    signature: str | None = None

    # Fees and priority
    fee: float = 0.0
    priority: int = 50

    # Block info (set when confirmed)
    block_number: int | None = None
    block_hash: str | None = None


class Block(BaseModel):
    """A block in the chain."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    number: int
    hash: str = ""
    previous_hash: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    transactions: tuple[str, ...] = Field(default_factory=tuple)  # Transaction IDs
    merkle_root: str = ""
    nonce: int = 0

    # Metadata
    miner: str | None = None
    difficulty: int = 1
    size: int = 0


class ConsensusConfig(BaseModel):
    """Configuration for consensus mechanism."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    consensus_type: ConsensusType = ConsensusType.AUTHORITY
    min_confirmations: int = 1
    block_time_seconds: int = 10
    max_transactions_per_block: int = 100
    difficulty: int = 1


@dataclass
class ValidationResult:
    """Result of transaction validation."""

    transaction_id: str
    valid: bool
    errors: list[str] = field(default_factory=list)


@dataclass
class BlockResult:
    """Result of block creation."""

    block_number: int
    block_hash: str
    transaction_count: int
    success: bool
    error: str | None = None


@dataclass
class ChainStats:
    """Statistics about the blockchain."""

    block_height: int
    total_transactions: int
    pending_transactions: int
    confirmed_transactions: int
    total_size: int
    average_block_time: float
    chain_valid: bool


class TransactionPool:
    """Pool of pending transactions (mempool)."""

    def __init__(self, max_size: int = 10000) -> None:
        self._transactions: dict[str, BlockchainTransaction] = {}
        self._max_size = max_size
        self._log = logger.bind(component="transaction_pool")

    @property
    def size(self) -> int:
        return len(self._transactions)

    def add(self, transaction: BlockchainTransaction) -> bool:
        """Add a transaction to the pool."""
        if len(self._transactions) >= self._max_size:
            self._log.warning("pool_full")
            return False

        self._transactions[transaction.id] = transaction
        self._log.debug("transaction_added", tx_id=transaction.id)
        return True

    def get(self, transaction_id: str) -> BlockchainTransaction | None:
        """Get a transaction from the pool."""
        return self._transactions.get(transaction_id)

    def remove(self, transaction_id: str) -> bool:
        """Remove a transaction from the pool."""
        return self._transactions.pop(transaction_id, None) is not None

    def get_pending(self, limit: int = 100) -> list[BlockchainTransaction]:
        """Get pending transactions sorted by priority and fee."""
        pending = list(self._transactions.values())
        pending.sort(key=lambda t: (-t.priority, -t.fee, t.timestamp))
        return pending[:limit]

    def clear(self) -> int:
        """Clear all transactions. Returns count cleared."""
        count = len(self._transactions)
        self._transactions.clear()
        return count


class ChainManager:
    """Manages the blockchain."""

    GENESIS_HASH = "0" * 64

    def __init__(self, config: ConsensusConfig | None = None) -> None:
        self._config = config or ConsensusConfig()
        self._blocks: dict[int, Block] = {}
        self._transactions: dict[str, BlockchainTransaction] = {}
        self._by_hash: dict[str, int] = {}  # hash -> block number
        self._log = logger.bind(component="chain_manager")

        # Create genesis block
        genesis = Block(
            number=0,
            hash=self.GENESIS_HASH,
            previous_hash="",
            transactions=(),
            merkle_root=self._compute_merkle_root([]),
            nonce=0,
            miner="genesis",
        )
        self._blocks[0] = genesis
        self._by_hash[genesis.hash] = 0

    @property
    def height(self) -> int:
        return max(self._blocks.keys()) if self._blocks else 0

    @property
    def block_count(self) -> int:
        return len(self._blocks)

    @property
    def transaction_count(self) -> int:
        return len(self._transactions)

    def get_latest_block(self) -> Block:
        """Get the latest block."""
        return self._blocks[self.height]

    def get_block(self, number: int) -> Block | None:
        """Get a block by number."""
        return self._blocks.get(number)

    def get_block_by_hash(self, block_hash: str) -> Block | None:
        """Get a block by hash."""
        number = self._by_hash.get(block_hash)
        return self._blocks.get(number) if number is not None else None

    def get_transaction(self, transaction_id: str) -> BlockchainTransaction | None:
        """Get a transaction by ID."""
        return self._transactions.get(transaction_id)

    def create_block(
        self,
        transactions: list[BlockchainTransaction],
        miner: str = "system",
    ) -> BlockResult:
        """Create a new block with transactions."""
        if not transactions:
            return BlockResult(
                block_number=-1,
                block_hash="",
                transaction_count=0,
                success=False,
                error="No transactions",
            )

        latest = self.get_latest_block()
        new_number = latest.number + 1
        tx_ids = tuple(t.id for t in transactions)

        # Compute hashes
        merkle_root = self._compute_merkle_root(tx_ids)
        nonce = self._find_nonce(new_number, latest.hash, merkle_root)
        block_hash = self._compute_block_hash(new_number, latest.hash, merkle_root, nonce)

        # Create block
        block = Block(
            number=new_number,
            hash=block_hash,
            previous_hash=latest.hash,
            transactions=tx_ids,
            merkle_root=merkle_root,
            nonce=nonce,
            miner=miner,
            difficulty=self._config.difficulty,
            size=len(str(transactions)),
        )

        self._blocks[new_number] = block
        self._by_hash[block_hash] = new_number

        # Update transactions
        for tx in transactions:
            confirmed = BlockchainTransaction(
                id=tx.id,
                state=TransactionState.CONFIRMED,
                timestamp=tx.timestamp,
                sender=tx.sender,
                recipient=tx.recipient,
                data=tx.data,
                signature=tx.signature,
                fee=tx.fee,
                priority=tx.priority,
                block_number=new_number,
                block_hash=block_hash,
            )
            self._transactions[tx.id] = confirmed

        self._log.info("block_created", number=new_number, tx_count=len(transactions))

        return BlockResult(
            block_number=new_number,
            block_hash=block_hash,
            transaction_count=len(transactions),
            success=True,
        )

    def validate_chain(self) -> bool:
        """Validate the entire chain."""
        for i in range(1, self.height + 1):
            block = self._blocks.get(i)
            prev_block = self._blocks.get(i - 1)

            if not block or not prev_block:
                return False

            # Check previous hash
            if block.previous_hash != prev_block.hash:
                self._log.warning("chain_invalid", block=i, reason="previous_hash_mismatch")
                return False

            # Verify block hash
            expected_hash = self._compute_block_hash(
                block.number,
                block.previous_hash,
                block.merkle_root,
                block.nonce,
            )
            if block.hash != expected_hash:
                self._log.warning("chain_invalid", block=i, reason="hash_mismatch")
                return False

        return True

    def _compute_merkle_root(self, tx_ids: tuple[str, ...] | list[str]) -> str:
        """Compute Merkle root of transactions."""
        if not tx_ids:
            return hashlib.sha256(b"empty").hexdigest()

        hashes = [hashlib.sha256(tid.encode()).hexdigest() for tid in tx_ids]

        while len(hashes) > 1:
            if len(hashes) % 2 == 1:
                hashes.append(hashes[-1])  # Duplicate last
            new_hashes = []
            for i in range(0, len(hashes), 2):
                combined = hashes[i] + hashes[i + 1]
                new_hashes.append(hashlib.sha256(combined.encode()).hexdigest())
            hashes = new_hashes

        return hashes[0]

    def _compute_block_hash(
        self,
        number: int,
        previous_hash: str,
        merkle_root: str,
        nonce: int,
    ) -> str:
        """Compute block hash."""
        data = f"{number}{previous_hash}{merkle_root}{nonce}"
        return hashlib.sha256(data.encode()).hexdigest()

    def _find_nonce(
        self,
        number: int,
        previous_hash: str,
        merkle_root: str,
    ) -> int:
        """Find a valid nonce (simplified PoW simulation)."""
        # For simplicity, just return a sequential nonce
        # In a real implementation, this would find a hash with required leading zeros
        nonce = 0
        target_prefix = "0" * self._config.difficulty

        for _ in range(1000):  # Limit iterations
            block_hash = self._compute_block_hash(number, previous_hash, merkle_root, nonce)
            if block_hash.startswith(target_prefix):
                return nonce
            nonce += 1

        return nonce  # Return last nonce if target not reached


class BlockchainSimulator(Subsystem):
    """
    Models distributed ledger logic for symbolic economies.

    Process Loop:
    1. Submit: Receive symbolic transactions
    2. Validate: Verify transaction validity and consensus criteria
    3. Record: Add transactions to the ledger
    4. Distribute: Propagate updated records to all relevant nodes
    """

    def __init__(self, config: ConsensusConfig | None = None) -> None:
        metadata = SubsystemMetadata(
            name="blockchain_simulator",
            display_name="Blockchain Simulator",
            description="Models distributed ledger logic for symbolic economies",
            type=SubsystemType.VALUE,
            tags=frozenset(["blockchain", "ledger", "consensus", "immutable"]),
            input_types=frozenset(["TOKEN", "MESSAGE", "TRANSACTION"]),
            output_types=frozenset(["TOKEN", "REFERENCE", "BLOCK"]),
            subscribed_topics=frozenset([
                str(SubsystemTopics.ALL),
                "blockchain.#",
                "ledger.#",
            ]),
            published_topics=frozenset([
                "blockchain.transaction.submitted",
                "blockchain.transaction.confirmed",
                "blockchain.block.created",
            ]),
        )
        super().__init__(metadata)

        self._config = config or ConsensusConfig()
        self._pool = TransactionPool()
        self._chain = ChainManager(self._config)

    @property
    def block_height(self) -> int:
        return self._chain.height

    @property
    def pending_count(self) -> int:
        return self._pool.size

    @property
    def transaction_count(self) -> int:
        return self._chain.transaction_count

    async def intake(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> SymbolicInput:
        """Phase 1: Receive symbolic transactions."""
        if not input_data.values:
            self._log.debug("empty_input")
            return input_data

        self._log.debug("intake_complete", value_count=len(input_data.values))
        return input_data

    async def process(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> list[ValidationResult | BlockResult]:
        """Phase 2: Validate transactions and create blocks."""
        results: list[ValidationResult | BlockResult] = []

        # Process incoming transactions
        for value in input_data.values:
            content = value.content
            if not isinstance(content, dict):
                continue

            action = content.get("action", "submit")

            if action == "submit":
                transaction = self._create_transaction_from_value(value)
                if transaction:
                    validation = self._validate_transaction(transaction)
                    if validation.valid:
                        self._pool.add(transaction)
                    results.append(validation)

            elif action == "mine":
                block_result = self.mine_block(content.get("miner", "system"))
                results.append(block_result)

        return results

    async def evaluate(
        self,
        intermediate: list[ValidationResult | BlockResult],
        ctx: ProcessContext[dict[str, Any]],
    ) -> tuple[SymbolicOutput, bool]:
        """Phase 3: Create output from results."""
        values: list[SymbolicValue] = []

        for result in intermediate:
            if isinstance(result, ValidationResult):
                value = SymbolicValue(
                    type=SymbolicValueType.REFERENCE,
                    content={
                        "transaction_id": result.transaction_id,
                        "valid": result.valid,
                        "errors": result.errors,
                    },
                    source_subsystem=self.name,
                    tags=frozenset(["validation", "valid" if result.valid else "invalid"]),
                    meaning=f"Validation: {'valid' if result.valid else 'invalid'}",
                    confidence=1.0 if result.valid else 0.0,
                )
            else:  # BlockResult
                value = SymbolicValue(
                    type=SymbolicValueType.REFERENCE,
                    content={
                        "block_number": result.block_number,
                        "block_hash": result.block_hash,
                        "transaction_count": result.transaction_count,
                        "success": result.success,
                        "error": result.error,
                    },
                    source_subsystem=self.name,
                    tags=frozenset(["block", "mined" if result.success else "failed"]),
                    meaning=f"Block {result.block_number}: {'mined' if result.success else 'failed'}",
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

                if "transaction_id" in content:
                    if content.get("valid"):
                        await self.emit_event(
                            "blockchain.transaction.submitted",
                            {"transaction_id": content.get("transaction_id")},
                        )
                elif "block_number" in content and content.get("success"):
                    await self.emit_event(
                        "blockchain.block.created",
                        {
                            "block_number": content.get("block_number"),
                            "block_hash": content.get("block_hash"),
                        },
                    )

        return None

    def _create_transaction_from_value(
        self,
        value: SymbolicValue,
    ) -> BlockchainTransaction | None:
        """Create a BlockchainTransaction from a SymbolicValue."""
        content = value.content
        if not isinstance(content, dict):
            return None

        try:
            return BlockchainTransaction(
                id=content.get("id", str(ULID())),
                sender=content.get("sender", "unknown"),
                recipient=content.get("recipient", "unknown"),
                data=content.get("data", {}),
                signature=content.get("signature"),
                fee=content.get("fee", 0.0),
                priority=content.get("priority", 50),
            )
        except Exception as e:
            self._log.warning("transaction_parse_failed", value_id=value.id, error=str(e))
            return None

    def _validate_transaction(self, transaction: BlockchainTransaction) -> ValidationResult:
        """Validate a transaction."""
        errors: list[str] = []

        if not transaction.sender:
            errors.append("Missing sender")
        if not transaction.recipient:
            errors.append("Missing recipient")
        if transaction.sender == transaction.recipient:
            errors.append("Sender and recipient cannot be the same")

        return ValidationResult(
            transaction_id=transaction.id,
            valid=len(errors) == 0,
            errors=errors,
        )

    # --- Message handlers ---

    async def handle_event(self, message: Message) -> None:
        """Handle incoming events."""
        if message.topic.startswith("blockchain."):
            self._log.debug("blockchain_event_received", topic=message.topic)

    async def handle_signal(self, signal: Any) -> None:
        """Handle incoming signals."""
        self._log.debug("signal_received", signal_id=getattr(signal, "id", "unknown"))

    # --- Public API ---

    def submit_transaction(
        self,
        sender: str,
        recipient: str,
        data: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Submit a transaction to the pool."""
        transaction = BlockchainTransaction(
            sender=sender,
            recipient=recipient,
            data=data or {},
            signature=kwargs.get("signature"),
            fee=kwargs.get("fee", 0.0),
            priority=kwargs.get("priority", 50),
        )

        validation = self._validate_transaction(transaction)
        if validation.valid:
            self._pool.add(transaction)

        return validation

    def get_pending_transactions(self, limit: int = 100) -> list[BlockchainTransaction]:
        """Get pending transactions from the pool."""
        return self._pool.get_pending(limit)

    def mine_block(self, miner: str = "system") -> BlockResult:
        """Mine a new block with pending transactions."""
        pending = self._pool.get_pending(self._config.max_transactions_per_block)
        if not pending:
            return BlockResult(
                block_number=-1,
                block_hash="",
                transaction_count=0,
                success=False,
                error="No pending transactions",
            )

        result = self._chain.create_block(pending, miner)

        if result.success:
            # Remove confirmed transactions from pool
            for tx in pending:
                self._pool.remove(tx.id)

        return result

    def get_block(self, number: int) -> Block | None:
        """Get a block by number."""
        return self._chain.get_block(number)

    def get_latest_block(self) -> Block:
        """Get the latest block."""
        return self._chain.get_latest_block()

    def get_transaction(self, transaction_id: str) -> BlockchainTransaction | None:
        """Get a transaction by ID."""
        # Check chain first, then pool
        tx = self._chain.get_transaction(transaction_id)
        if tx:
            return tx
        return self._pool.get(transaction_id)

    def validate_chain(self) -> bool:
        """Validate the entire blockchain."""
        return self._chain.validate_chain()

    def get_stats(self) -> ChainStats:
        """Get blockchain statistics."""
        confirmed = len([
            t for t in self._chain._transactions.values()
            if t and t.state == TransactionState.CONFIRMED
        ])

        # Calculate average block time
        avg_time = 0.0
        if self._chain.height > 1:
            times = []
            for i in range(1, min(self._chain.height + 1, 11)):
                block = self._chain.get_block(i)
                prev = self._chain.get_block(i - 1)
                if block and prev:
                    delta = (block.timestamp - prev.timestamp).total_seconds()
                    times.append(delta)
            avg_time = sum(times) / len(times) if times else 0.0

        return ChainStats(
            block_height=self._chain.height,
            total_transactions=self._chain.transaction_count + self._pool.size,
            pending_transactions=self._pool.size,
            confirmed_transactions=self._chain.transaction_count,
            total_size=sum(b.size for b in [self._chain.get_block(i) for i in range(self._chain.height + 1)] if b),
            average_block_time=avg_time,
            chain_valid=self.validate_chain(),
        )

    def clear(self) -> tuple[int, int]:
        """Clear all data. Returns (blocks, pending_transactions) cleared."""
        blocks = self._chain.block_count - 1  # Exclude genesis
        pending = self._pool.clear()
        self._chain = ChainManager(self._config)
        return blocks, pending
