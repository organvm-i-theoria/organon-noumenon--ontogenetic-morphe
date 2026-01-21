"""Tests for value subsystems: ValueExchangeManager, BlockchainSimulator, ProcessMonetizer."""

from decimal import Decimal

import pytest

from autogenrec.subsystems.value.value_exchange_manager import (
    Account,
    CurrencyType,
    TransactionStatus,
    TransactionType,
    ValueExchangeManager,
)
from autogenrec.subsystems.value.blockchain_simulator import (
    BlockchainSimulator,
    ConsensusConfig,
    ConsensusType,
    TransactionState,
)
from autogenrec.subsystems.value.process_monetizer import (
    ProcessMonetizer,
    ProcessStatus,
    ProductType,
    RevenueModel,
)


# ============================================================================
# ValueExchangeManager Tests
# ============================================================================


class TestValueExchangeManager:
    """Tests for ValueExchangeManager subsystem."""

    @pytest.fixture
    def manager(self) -> ValueExchangeManager:
        return ValueExchangeManager()

    def test_initialization(self, manager: ValueExchangeManager) -> None:
        """Test ValueExchangeManager initializes correctly."""
        assert manager.name == "value_exchange_manager"
        assert manager.account_count == 0
        assert manager.transaction_count == 0

    def test_create_account(self, manager: ValueExchangeManager) -> None:
        """Test creating an account."""
        account = manager.create_account(
            name="Test Account",
            owner_id="user_123",
            currency=CurrencyType.TOKEN,
            initial_balance=Decimal("100"),
        )

        assert account.name == "Test Account"
        assert account.owner_id == "user_123"
        assert account.currency == CurrencyType.TOKEN
        assert account.balance == Decimal("100")
        assert manager.account_count == 1

    def test_get_account(self, manager: ValueExchangeManager) -> None:
        """Test retrieving an account."""
        account = manager.create_account(
            name="Retrieve Test",
            owner_id="user_123",
        )

        retrieved = manager.get_account(account.id)
        assert retrieved is not None
        assert retrieved.id == account.id

    def test_deposit(self, manager: ValueExchangeManager) -> None:
        """Test depositing value."""
        account = manager.create_account(
            name="Deposit Test",
            owner_id="user_123",
            initial_balance=Decimal("100"),
        )

        result = manager.deposit(account.id, Decimal("50"))
        assert result.success is True
        assert result.status == TransactionStatus.COMPLETED
        assert result.target_balance == Decimal("150")

        updated = manager.get_account(account.id)
        assert updated.balance == Decimal("150")

    def test_withdraw(self, manager: ValueExchangeManager) -> None:
        """Test withdrawing value."""
        account = manager.create_account(
            name="Withdraw Test",
            owner_id="user_123",
            initial_balance=Decimal("100"),
        )

        result = manager.withdraw(account.id, Decimal("30"))
        assert result.success is True
        assert result.status == TransactionStatus.COMPLETED
        assert result.source_balance == Decimal("70")

    def test_withdraw_insufficient_balance(self, manager: ValueExchangeManager) -> None:
        """Test withdrawing more than available."""
        account = manager.create_account(
            name="Insufficient Test",
            owner_id="user_123",
            initial_balance=Decimal("50"),
        )

        result = manager.withdraw(account.id, Decimal("100"))
        assert result.success is False
        assert result.status == TransactionStatus.FAILED
        assert "Insufficient balance" in result.error

    def test_transfer(self, manager: ValueExchangeManager) -> None:
        """Test transferring value between accounts."""
        source = manager.create_account(
            name="Source",
            owner_id="user_1",
            initial_balance=Decimal("100"),
        )
        target = manager.create_account(
            name="Target",
            owner_id="user_2",
            initial_balance=Decimal("50"),
        )

        result = manager.transfer(source.id, target.id, Decimal("30"))
        assert result.success is True
        assert result.source_balance == Decimal("70")
        assert result.target_balance == Decimal("80")

    def test_exchange(self, manager: ValueExchangeManager) -> None:
        """Test currency exchange."""
        source = manager.create_account(
            name="Token Account",
            owner_id="user_1",
            currency=CurrencyType.TOKEN,
            initial_balance=Decimal("100"),
        )
        target = manager.create_account(
            name="Credit Account",
            owner_id="user_1",
            currency=CurrencyType.CREDIT,
            initial_balance=Decimal("0"),
        )

        # Set exchange rate
        manager.set_exchange_rate(
            CurrencyType.TOKEN,
            CurrencyType.CREDIT,
            Decimal("2.5"),  # 1 TOKEN = 2.5 CREDIT # allow-secret
        )

        result = manager.exchange(
            source.id,
            target.id,
            Decimal("40"),
            CurrencyType.TOKEN,
            CurrencyType.CREDIT,
        )

        assert result.success is True
        assert result.source_balance == Decimal("60")
        assert result.target_balance == Decimal("100")  # 40 * 2.5

    def test_exchange_no_rate(self, manager: ValueExchangeManager) -> None:
        """Test exchange without exchange rate fails."""
        source = manager.create_account(
            name="Token Account",
            owner_id="user_1",
            currency=CurrencyType.TOKEN,
            initial_balance=Decimal("100"),
        )
        target = manager.create_account(
            name="Credit Account",
            owner_id="user_1",
            currency=CurrencyType.CREDIT,
        )

        result = manager.exchange(
            source.id,
            target.id,
            Decimal("40"),
            CurrencyType.TOKEN,
            CurrencyType.CREDIT,
        )

        assert result.success is False
        assert "No exchange rate" in result.error

    def test_reverse_transaction(self, manager: ValueExchangeManager) -> None:
        """Test reversing a transaction."""
        source = manager.create_account(
            name="Source",
            owner_id="user_1",
            initial_balance=Decimal("100"),
        )
        target = manager.create_account(
            name="Target",
            owner_id="user_2",
            initial_balance=Decimal("50"),
        )

        # Make transfer
        result = manager.transfer(source.id, target.id, Decimal("30"))
        assert result.success is True

        # Reverse it
        reverse = manager.reverse_transaction(result.transaction_id)
        assert reverse is not None
        assert reverse.success is True

        # Check balances restored
        source_account = manager.get_account(source.id)
        target_account = manager.get_account(target.id)
        assert source_account.balance == Decimal("100")
        assert target_account.balance == Decimal("50")

    def test_get_accounts_by_owner(self, manager: ValueExchangeManager) -> None:
        """Test getting accounts by owner."""
        manager.create_account(name="Account 1", owner_id="user_1")
        manager.create_account(name="Account 2", owner_id="user_1")
        manager.create_account(name="Account 3", owner_id="user_2")

        accounts = manager.get_accounts_by_owner("user_1")
        assert len(accounts) == 2

    def test_get_stats(self, manager: ValueExchangeManager) -> None:
        """Test getting statistics."""
        account = manager.create_account(
            name="Test",
            owner_id="user_1",
            initial_balance=Decimal("100"),
        )
        manager.deposit(account.id, Decimal("50"))

        stats = manager.get_stats()
        assert stats.total_accounts == 1
        assert stats.active_accounts == 1
        assert stats.completed_transactions == 1

    def test_clear(self, manager: ValueExchangeManager) -> None:
        """Test clearing all data."""
        account = manager.create_account(
            name="Test",
            owner_id="user_1",
            initial_balance=Decimal("100"),
        )
        manager.deposit(account.id, Decimal("50"))

        accounts, transactions = manager.clear()
        assert accounts == 1
        assert transactions == 1
        assert manager.account_count == 0
        assert manager.transaction_count == 0


# ============================================================================
# BlockchainSimulator Tests
# ============================================================================


class TestBlockchainSimulator:
    """Tests for BlockchainSimulator subsystem."""

    @pytest.fixture
    def simulator(self) -> BlockchainSimulator:
        return BlockchainSimulator()

    def test_initialization(self, simulator: BlockchainSimulator) -> None:
        """Test BlockchainSimulator initializes correctly."""
        assert simulator.name == "blockchain_simulator"
        assert simulator.block_height == 0  # Genesis block
        assert simulator.pending_count == 0

    def test_submit_transaction(self, simulator: BlockchainSimulator) -> None:
        """Test submitting a transaction."""
        result = simulator.submit_transaction(
            sender="alice",
            recipient="bob",
            data={"amount": 100},
        )

        assert result.valid is True
        assert simulator.pending_count == 1

    def test_submit_invalid_transaction(self, simulator: BlockchainSimulator) -> None:
        """Test submitting an invalid transaction."""
        result = simulator.submit_transaction(
            sender="alice",
            recipient="alice",  # Same as sender
            data={},
        )

        assert result.valid is False
        assert "cannot be the same" in result.errors[0]

    def test_mine_block(self, simulator: BlockchainSimulator) -> None:
        """Test mining a block."""
        # Submit some transactions
        simulator.submit_transaction("alice", "bob", {"amount": 100})
        simulator.submit_transaction("bob", "charlie", {"amount": 50})

        result = simulator.mine_block("miner1")
        assert result.success is True
        assert result.transaction_count == 2
        assert result.block_number == 1

        # Transactions should be confirmed
        assert simulator.pending_count == 0
        assert simulator.block_height == 1

    def test_mine_empty_pool(self, simulator: BlockchainSimulator) -> None:
        """Test mining with no pending transactions."""
        result = simulator.mine_block()
        assert result.success is False
        assert "No pending transactions" in result.error

    def test_get_block(self, simulator: BlockchainSimulator) -> None:
        """Test getting a block."""
        simulator.submit_transaction("alice", "bob", {"amount": 100})
        simulator.mine_block()

        block = simulator.get_block(1)
        assert block is not None
        assert block.number == 1
        assert len(block.transactions) == 1

    def test_get_latest_block(self, simulator: BlockchainSimulator) -> None:
        """Test getting latest block."""
        latest = simulator.get_latest_block()
        assert latest.number == 0  # Genesis

        simulator.submit_transaction("alice", "bob", {"amount": 100})
        simulator.mine_block()

        latest = simulator.get_latest_block()
        assert latest.number == 1

    def test_validate_chain(self, simulator: BlockchainSimulator) -> None:
        """Test chain validation."""
        # Initial chain should be valid
        assert simulator.validate_chain() is True

        # Add blocks and validate
        simulator.submit_transaction("alice", "bob", {"amount": 100})
        simulator.mine_block()
        simulator.submit_transaction("bob", "charlie", {"amount": 50})
        simulator.mine_block()

        assert simulator.validate_chain() is True

    def test_get_transaction_confirmed(self, simulator: BlockchainSimulator) -> None:
        """Test getting a confirmed transaction."""
        result = simulator.submit_transaction("alice", "bob", {"amount": 100})
        simulator.mine_block()

        tx = simulator.get_transaction(result.transaction_id)
        assert tx is not None
        assert tx.state == TransactionState.CONFIRMED
        assert tx.block_number == 1

    def test_get_transaction_pending(self, simulator: BlockchainSimulator) -> None:
        """Test getting a pending transaction."""
        result = simulator.submit_transaction("alice", "bob", {"amount": 100})

        tx = simulator.get_transaction(result.transaction_id)
        assert tx is not None
        assert tx.state == TransactionState.PENDING

    def test_get_pending_transactions(self, simulator: BlockchainSimulator) -> None:
        """Test getting pending transactions."""
        simulator.submit_transaction("alice", "bob", {"amount": 100}, priority=90)
        simulator.submit_transaction("bob", "charlie", {"amount": 50}, priority=50)
        simulator.submit_transaction("charlie", "dave", {"amount": 25}, priority=100)

        pending = simulator.get_pending_transactions()
        assert len(pending) == 3
        # Should be sorted by priority
        assert pending[0].priority == 100
        assert pending[1].priority == 90
        assert pending[2].priority == 50

    def test_custom_config(self) -> None:
        """Test with custom consensus config."""
        config = ConsensusConfig(
            consensus_type=ConsensusType.MAJORITY,
            max_transactions_per_block=5,
            difficulty=2,
        )
        simulator = BlockchainSimulator(config)

        # Submit more than max
        for i in range(10):
            simulator.submit_transaction(f"sender_{i}", f"recipient_{i}", {})

        result = simulator.mine_block()
        assert result.transaction_count == 5  # Max per block

    def test_get_stats(self, simulator: BlockchainSimulator) -> None:
        """Test getting statistics."""
        simulator.submit_transaction("alice", "bob", {})
        simulator.submit_transaction("bob", "charlie", {})
        simulator.mine_block()
        simulator.submit_transaction("charlie", "dave", {})

        stats = simulator.get_stats()
        assert stats.block_height == 1
        assert stats.confirmed_transactions == 2
        assert stats.pending_transactions == 1
        assert stats.chain_valid is True

    def test_clear(self, simulator: BlockchainSimulator) -> None:
        """Test clearing all data."""
        simulator.submit_transaction("alice", "bob", {})
        simulator.mine_block()
        simulator.submit_transaction("bob", "charlie", {})

        blocks, pending = simulator.clear()
        assert blocks == 1  # Excluding genesis
        assert pending == 1
        assert simulator.block_height == 0  # Only genesis


# ============================================================================
# ProcessMonetizer Tests
# ============================================================================


class TestProcessMonetizer:
    """Tests for ProcessMonetizer subsystem."""

    @pytest.fixture
    def monetizer(self) -> ProcessMonetizer:
        return ProcessMonetizer()

    def test_initialization(self, monetizer: ProcessMonetizer) -> None:
        """Test ProcessMonetizer initializes correctly."""
        assert monetizer.name == "process_monetizer"
        assert monetizer.process_count == 0
        assert monetizer.product_count == 0

    def test_register_process(self, monetizer: ProcessMonetizer) -> None:
        """Test registering a process."""
        process = monetizer.register_process(
            name="Test Process",
            owner_id="user_123",
            product_type=ProductType.SERVICE,
            revenue_model=RevenueModel.FIXED_PRICE,
            base_price=Decimal("100"),
        )

        assert process.name == "Test Process"
        assert process.owner_id == "user_123"
        assert process.status == ProcessStatus.DRAFT
        assert process.base_price == Decimal("100")
        assert monetizer.process_count == 1

    def test_get_process(self, monetizer: ProcessMonetizer) -> None:
        """Test retrieving a process."""
        process = monetizer.register_process(
            name="Retrieve Test",
            owner_id="user_123",
        )

        retrieved = monetizer.get_process(process.id)
        assert retrieved is not None
        assert retrieved.id == process.id

    def test_activate_process(self, monetizer: ProcessMonetizer) -> None:
        """Test activating a process."""
        process = monetizer.register_process(
            name="Activate Test",
            owner_id="user_123",
        )

        activated = monetizer.activate_process(process.id)
        assert activated is not None
        assert activated.status == ProcessStatus.ACTIVE

    def test_create_product(self, monetizer: ProcessMonetizer) -> None:
        """Test creating a product from a process."""
        process = monetizer.register_process(
            name="Test Process",
            owner_id="user_123",
            base_price=Decimal("50"),
        )

        product = monetizer.create_product(
            process_id=process.id,
            name="Test Product",
            price=Decimal("75"),
        )

        assert product is not None
        assert product.name == "Test Product"
        assert product.price == Decimal("75")
        assert product.process_id == process.id
        assert monetizer.product_count == 1

    def test_get_products_for_process(self, monetizer: ProcessMonetizer) -> None:
        """Test getting products for a process."""
        process = monetizer.register_process(
            name="Test Process",
            owner_id="user_123",
        )

        monetizer.create_product(process.id, "Product 1", Decimal("50"))
        monetizer.create_product(process.id, "Product 2", Decimal("75"))

        products = monetizer.get_products_for_process(process.id)
        assert len(products) == 2

    def test_record_usage(self, monetizer: ProcessMonetizer) -> None:
        """Test recording usage."""
        process = monetizer.register_process(
            name="Usage Test",
            owner_id="user_123",
            revenue_model=RevenueModel.USAGE_BASED,
            usage_rate=Decimal("10"),
        )

        record = monetizer.record_usage(
            process_id=process.id,
            user_id="customer_1",
            units=Decimal("5"),
        )

        assert record is not None
        assert record.units == Decimal("5")
        assert record.total_value == Decimal("50")  # 5 * 10

        # Check process metrics updated
        updated = monetizer.get_process(process.id)
        assert updated.usage_count == 1
        assert updated.total_revenue == Decimal("50")

    def test_valuate_process(self, monetizer: ProcessMonetizer) -> None:
        """Test valuating a process."""
        process = monetizer.register_process(
            name="Valuation Test",
            owner_id="user_123",
            base_price=Decimal("100"),
        )

        # Record some usage
        monetizer.record_usage(process.id, "user_1", Decimal("1"))
        monetizer.record_usage(process.id, "user_2", Decimal("1"))

        valuation = monetizer.valuate_process(process.id)
        assert valuation is not None
        assert valuation.estimated_value > Decimal("0")
        assert valuation.confidence > 0
        assert "base_price" in valuation.factors

    def test_create_payout(self, monetizer: ProcessMonetizer) -> None:
        """Test creating a payout."""
        process = monetizer.register_process(
            name="Payout Test",
            owner_id="user_123",
        )

        # Record usage to generate revenue
        monetizer.record_usage(process.id, "customer_1", Decimal("1"))

        payout = monetizer.create_payout(
            process_id=process.id,
            amount=Decimal("100"),
        )

        assert payout is not None
        assert payout.recipient_id == "user_123"
        assert payout.amount == Decimal("100")
        assert payout.fee > Decimal("0")  # Commission taken
        assert payout.net_amount < payout.amount

    def test_process_payout(self, monetizer: ProcessMonetizer) -> None:
        """Test processing a payout."""
        process = monetizer.register_process(
            name="Process Payout Test",
            owner_id="user_123",
        )

        payout = monetizer.create_payout(
            process_id=process.id,
            amount=Decimal("100"),
        )

        processed = monetizer.process_payout(payout.id)
        assert processed is not None
        assert processed.status.name == "PROCESSED"
        assert processed.processed_at is not None

    def test_different_product_types(self, monetizer: ProcessMonetizer) -> None:
        """Test different product types."""
        types = [
            ProductType.SERVICE,
            ProductType.SUBSCRIPTION,
            ProductType.LICENSE,
            ProductType.CONSUMABLE,
            ProductType.ASSET,
            ProductType.ACCESS,
        ]

        for pt in types:
            process = monetizer.register_process(
                name=f"Process {pt.name}",
                owner_id="user_123",
                product_type=pt,
            )
            assert process.product_type == pt

    def test_different_revenue_models(self, monetizer: ProcessMonetizer) -> None:
        """Test different revenue models."""
        models = [
            RevenueModel.FIXED_PRICE,
            RevenueModel.USAGE_BASED,
            RevenueModel.SUBSCRIPTION,
            RevenueModel.FREEMIUM,
            RevenueModel.COMMISSION,
            RevenueModel.AUCTION,
        ]

        for rm in models:
            process = monetizer.register_process(
                name=f"Process {rm.name}",
                owner_id="user_123",
                revenue_model=rm,
            )
            assert process.revenue_model == rm

    def test_get_stats(self, monetizer: ProcessMonetizer) -> None:
        """Test getting statistics."""
        process = monetizer.register_process(
            name="Stats Test",
            owner_id="user_123",
            base_price=Decimal("100"),
        )
        monetizer.activate_process(process.id)
        monetizer.create_product(process.id, "Product", Decimal("50"))
        monetizer.record_usage(process.id, "user_1", Decimal("1"))

        stats = monetizer.get_stats()
        assert stats.total_processes == 1
        assert stats.active_processes == 1
        assert stats.total_products == 1
        assert stats.total_revenue >= Decimal("0")

    def test_clear(self, monetizer: ProcessMonetizer) -> None:
        """Test clearing all data."""
        process = monetizer.register_process(
            name="Clear Test",
            owner_id="user_123",
        )
        monetizer.create_product(process.id, "Product", Decimal("50"))
        monetizer.record_usage(process.id, "user_1", Decimal("1"))

        processes, products, usage = monetizer.clear()
        assert processes == 1
        assert products == 1
        assert usage == 1
        assert monetizer.process_count == 0
        assert monetizer.product_count == 0
