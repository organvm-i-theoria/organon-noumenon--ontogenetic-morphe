---
title: BlockchainSimulator
system: Recursive–Generative Organizational Body
type: subsystem
category: value
tags: [blockchain, ledger, immutable, transactions]
dependencies: []
---

# BlockchainSimulator

The **BlockchainSimulator** provides distributed ledger logic for symbolic economies, offering immutable transaction recording and simple consensus simulation.

## Overview

| Property | Value |
|----------|-------|
| Category | Value & Exchange |
| Module | `autogenrec.subsystems.value.blockchain_simulator` |
| Dependencies | None |

## Domain Models

### Enums

```python
class TransactionState(Enum):
    PENDING = auto()       # In mempool, not yet mined
    CONFIRMED = auto()     # Included in a block
    FAILED = auto()        # Transaction failed

class BlockState(Enum):
    PENDING = auto()       # Being assembled
    MINED = auto()         # Successfully mined
    ORPHANED = auto()      # Not in main chain
```

### Core Models

- **BlockchainTransaction**: Transaction with sender, recipient, data, state
- **Block**: Block with transactions, previous hash, nonce, timestamp
- **ChainStats**: Statistics about the blockchain state

## Process Loop

1. **Intake**: Receive transactions to record
2. **Process**: Validate transactions, add to mempool
3. **Evaluate**: Mine blocks, verify chain integrity
4. **Integrate**: Update chain state, confirm transactions

## Public API

### Transaction Submission

```python
from autogenrec.subsystems.value.blockchain_simulator import BlockchainSimulator

blockchain = BlockchainSimulator()

# Submit a transaction
tx = blockchain.submit_transaction(
    sender="platform_account",
    recipient="user_account",
    data={
        "type": "royalty_payout",
        "product_id": "product_001",
        "amount": "100",
    },
)

print(f"TX ID: {tx.transaction_id}")
print(f"State: {tx.state.name}")  # PENDING
```

### Mining

```python
# Mine a new block (includes pending transactions)
block = blockchain.mine_block()

print(f"Block #{block.block_number}")
print(f"Transactions: {len(block.transactions)}")
print(f"Previous Hash: {block.previous_hash[:16]}...")
```

### Chain Queries

```python
# Get transaction by ID
tx = blockchain.get_transaction(tx_id)

# Get block by number
block = blockchain.get_block(block_number)

# Get latest block
latest = blockchain.get_latest_block()

# Verify chain integrity
is_valid = blockchain.verify_chain()
```

### Statistics

```python
stats = blockchain.get_stats()
# ChainStats with:
#   block_height (number of blocks)
#   pending_transactions
#   total_transactions
#   chain_valid (integrity check)
```

## Chain Structure

```
Genesis Block (0)
    ↓
Block 1 [tx1, tx2, tx3]
    ↓
Block 2 [tx4, tx5]
    ↓
...
```

Each block contains:
- Block number
- Transactions list
- Previous block hash
- Timestamp
- Nonce (for mining simulation)

## Integration

The BlockchainSimulator provides:
- **ValueExchangeManager**: Immutable transaction logs
- **ProcessMonetizer**: Payout verification

## Example

See `examples/value_exchange_demo.py` for blockchain transaction recording and mining.
