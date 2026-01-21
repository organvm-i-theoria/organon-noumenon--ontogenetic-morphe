#!/usr/bin/env python3
"""
Example: Value Exchange Demo

Demonstrates:
- ValueExchangeManager for account management and transfers
- BlockchainSimulator for immutable transaction recording
- ProcessMonetizer for converting processes to monetizable products

This shows the value economy subsystem:
ACCOUNTS -> TRANSFERS -> BLOCKCHAIN -> MONETIZATION
"""

from decimal import Decimal

from autogenrec.subsystems.value.value_exchange_manager import (
    ValueExchangeManager,
    CurrencyType,
)
from autogenrec.subsystems.value.blockchain_simulator import BlockchainSimulator
from autogenrec.subsystems.value.process_monetizer import (
    ProcessMonetizer,
    ProductType,
    RevenueModel,
)


def main():
    print("=" * 60)
    print("Value Exchange Demo")
    print("=" * 60)
    print()

    # Initialize subsystems
    exchange = ValueExchangeManager()
    blockchain = BlockchainSimulator()
    monetizer = ProcessMonetizer()

    # =========================================================================
    # Step 1: Create accounts
    # =========================================================================
    print("Step 1: Creating Accounts")
    print("-" * 40)

    # Create platform and user accounts
    platform = exchange.create_account(
        name="Platform Treasury",
        owner_id="platform",
        currency=CurrencyType.TOKEN,
        initial_balance=Decimal("10000"),
    )
    print(f"  Created: {platform.name}")
    print(f"    ID: {platform.id}")
    print(f"    Balance: {platform.balance} {platform.currency.name}")

    alice = exchange.create_account(
        name="Alice",
        owner_id="user_alice",
        currency=CurrencyType.TOKEN,
        initial_balance=Decimal("500"),
    )
    print(f"  Created: {alice.name}")
    print(f"    Balance: {alice.balance} {alice.currency.name}")

    bob = exchange.create_account(
        name="Bob",
        owner_id="user_bob",
        currency=CurrencyType.TOKEN,
        initial_balance=Decimal("250"),
    )
    print(f"  Created: {bob.name}")
    print(f"    Balance: {bob.balance} {bob.currency.name}")

    creator = exchange.create_account(
        name="Content Creator",
        owner_id="creator_1",
        currency=CurrencyType.TOKEN,
        initial_balance=Decimal("0"),
    )
    print(f"  Created: {creator.name}")
    print(f"    Balance: {creator.balance} {creator.currency.name}")
    print()

    # =========================================================================
    # Step 2: Perform transfers
    # =========================================================================
    print("Step 2: Performing Transfers")
    print("-" * 40)

    # Alice sends tokens to Bob
    result1 = exchange.transfer(alice.id, bob.id, Decimal("100"))
    print(f"  Transfer: Alice -> Bob: 100 TOKENS")
    print(f"    Success: {result1.success}")
    print(f"    Transaction ID: {result1.transaction_id}")

    # Bob sends some back to Alice
    result2 = exchange.transfer(bob.id, alice.id, Decimal("25"))
    print(f"  Transfer: Bob -> Alice: 25 TOKENS")
    print(f"    Success: {result2.success}")
    print(f"    Transaction ID: {result2.transaction_id}")

    # Check balances
    alice_updated = exchange.get_account(alice.id)
    bob_updated = exchange.get_account(bob.id)
    print(f"\n  Updated Balances:")
    print(f"    Alice: {alice_updated.balance} TOKENS")
    print(f"    Bob: {bob_updated.balance} TOKENS")
    print()

    # =========================================================================
    # Step 3: Record on blockchain
    # =========================================================================
    print("Step 3: Recording on Blockchain")
    print("-" * 40)

    # Submit transactions to blockchain
    tx1 = blockchain.submit_transaction(
        sender=alice.id,
        recipient=bob.id,
        data={
            "type": "transfer",
            "amount": "100",
            "exchange_tx_id": result1.transaction_id,
        },
    )
    print(f"  Submitted TX: {tx1.transaction_id[:16]}...")
    print(f"    Valid: {tx1.valid}")

    tx2 = blockchain.submit_transaction(
        sender=bob.id,
        recipient=alice.id,
        data={
            "type": "transfer",
            "amount": "25",
            "exchange_tx_id": result2.transaction_id,
        },
    )
    print(f"  Submitted TX: {tx2.transaction_id[:16]}...")
    print(f"    Valid: {tx2.valid}")

    # Mine a block
    block_result = blockchain.mine_block()
    print(f"\n  Mined Block:")
    print(f"    Block Number: {block_result.block_number}")
    print(f"    Transactions: {block_result.transaction_count}")
    print(f"    Hash: {block_result.block_hash[:16]}...")

    # Check chain stats
    chain_stats = blockchain.get_stats()
    print(f"\n  Chain Statistics:")
    print(f"    Total Blocks: {chain_stats.block_height}")
    print(f"    Total Transactions: {chain_stats.total_transactions}")
    print()

    # =========================================================================
    # Step 4: Monetize a process
    # =========================================================================
    print("Step 4: Process Monetization")
    print("-" * 40)

    # Register a monetizable process
    api_service = monetizer.register_process(
        name="Premium Data API",
        owner_id="creator_1",
        product_type=ProductType.SERVICE,
        revenue_model=RevenueModel.USAGE_BASED,
        usage_rate=Decimal("5"),  # 5 tokens per API call
        description="High-quality data API with real-time updates",
    )
    print(f"  Registered Process: {api_service.name}")
    print(f"    Type: {api_service.product_type.name}")
    print(f"    Revenue Model: {api_service.revenue_model.name}")
    print(f"    Usage Rate: {api_service.usage_rate} TOKENS/call")

    # Activate the process
    monetizer.activate_process(api_service.id)
    print(f"    Status: ACTIVE")

    # Simulate usage
    print(f"\n  Recording Usage:")
    usage1 = monetizer.record_usage(api_service.id, "user_alice", Decimal("10"))
    print(f"    Alice: 10 API calls = {usage1.total_value} TOKENS")

    usage2 = monetizer.record_usage(api_service.id, "user_bob", Decimal("5"))
    print(f"    Bob: 5 API calls = {usage2.total_value} TOKENS")

    usage3 = monetizer.record_usage(api_service.id, "user_alice", Decimal("3"))
    print(f"    Alice: 3 more calls = {usage3.total_value} TOKENS")

    # Check process revenue
    updated_process = monetizer.get_process(api_service.id)
    print(f"\n  Process Revenue:")
    print(f"    Total Revenue: {updated_process.total_revenue} TOKENS")
    print(f"    Usage Count: {updated_process.usage_count}")

    # Create payout
    payout = monetizer.create_payout(api_service.id)
    print(f"\n  Payout Created:")
    print(f"    Gross Amount: {payout.amount} TOKENS")
    print(f"    Platform Fee: {payout.fee} TOKENS")
    print(f"    Net Amount: {payout.net_amount} TOKENS")
    print()

    # =========================================================================
    # Step 5: Execute payout transfer
    # =========================================================================
    print("Step 5: Executing Payout")
    print("-" * 40)

    # Transfer payout to creator
    payout_result = exchange.transfer(platform.id, creator.id, payout.net_amount)
    print(f"  Platform -> Creator: {payout.net_amount} TOKENS")
    print(f"    Success: {payout_result.success}")

    # Record payout on blockchain
    payout_tx = blockchain.submit_transaction(
        sender=platform.id,
        recipient=creator.id,
        data={
            "type": "payout",
            "process_id": api_service.id,
            "amount": str(payout.net_amount),
            "payout_id": payout.id,
        },
    )
    blockchain.mine_block()
    print(f"  Recorded on blockchain: {payout_tx.transaction_id[:16]}...")

    # Final balances
    creator_final = exchange.get_account(creator.id)
    platform_final = exchange.get_account(platform.id)
    print(f"\n  Final Balances:")
    print(f"    Creator: {creator_final.balance} TOKENS")
    print(f"    Platform: {platform_final.balance} TOKENS")
    print()

    # =========================================================================
    # Step 6: Summary statistics
    # =========================================================================
    print("Step 6: Summary Statistics")
    print("-" * 40)

    exchange_stats = exchange.get_stats()
    monetizer_stats = monetizer.get_stats()
    chain_stats = blockchain.get_stats()

    print(f"  Exchange:")
    print(f"    Total Accounts: {exchange_stats.total_accounts}")
    print(f"    Total Transactions: {exchange_stats.total_transactions}")
    print(f"    Total Volume: {exchange_stats.total_volume} TOKENS")

    print(f"  Monetizer:")
    print(f"    Active Processes: {monetizer_stats.active_processes}")
    print(f"    Total Revenue: {monetizer_stats.total_revenue} TOKENS")
    print(f"    Total Payouts: {monetizer_stats.total_payouts}")

    print(f"  Blockchain:")
    print(f"    Chain Length: {chain_stats.block_height} blocks")
    print(f"    Total Transactions: {chain_stats.total_transactions}")

    print()
    print("=" * 60)
    print("Example Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
