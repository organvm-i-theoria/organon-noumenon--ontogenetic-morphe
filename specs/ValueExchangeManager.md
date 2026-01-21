---
title: ValueExchangeManager
system: Recursive–Generative Organizational Body
type: subsystem
category: value
tags: [value, exchange, trade, accounts, transactions]
dependencies: [BlockchainSimulator]
---

# ValueExchangeManager

The **ValueExchangeManager** governs symbolic trade and value exchange, managing multi-currency accounts, transactions, and balance reconciliation.

## Overview

| Property | Value |
|----------|-------|
| Category | Value & Exchange |
| Module | `autogenrec.subsystems.value.value_exchange_manager` |
| Dependencies | BlockchainSimulator |

## Domain Models

### Enums

```python
class CurrencyType(Enum):
    TOKEN = auto()         # System tokens # allow-secret
    CREDIT = auto()        # Credit units
    POINTS = auto()        # Reward points
    CUSTOM = auto()        # Custom currency

class TransactionStatus(Enum):
    PENDING = auto()
    COMPLETED = auto()
    FAILED = auto()
    REVERSED = auto()

class TransactionType(Enum):
    TRANSFER = auto()      # Account to account
    DEPOSIT = auto()       # External deposit
    WITHDRAWAL = auto()    # External withdrawal
    FEE = auto()           # System fee
    REWARD = auto()        # System reward
```

### Core Models

- **Account**: Account with owner, currency type, balance, and transaction history
- **Transaction**: Transfer record with sender, recipient, amount, status
- **ExchangeRate**: Currency conversion rate

## Process Loop

1. **Intake**: Receive exchange requests, deposits, withdrawals
2. **Process**: Validate balances, execute transfers, apply fees
3. **Evaluate**: Verify transaction integrity, check for errors
4. **Integrate**: Update balances, record transactions, emit events

## Public API

### Account Management

```python
from autogenrec.subsystems.value.value_exchange_manager import (
    ValueExchangeManager, CurrencyType
)
from decimal import Decimal

exchange = ValueExchangeManager()

# Create accounts
platform = exchange.create_account(
    name="Platform Treasury",
    owner_id="platform",
    currency_type=CurrencyType.TOKEN,
    initial_balance=Decimal("100000"),
)

user = exchange.create_account(
    name="User Wallet",
    owner_id="user_001",
    currency_type=CurrencyType.TOKEN,
    initial_balance=Decimal("0"),
)

# Get account
account = exchange.get_account(platform.id)
```

### Transfers

```python
# Transfer between accounts
result = exchange.transfer(
    from_account_id=platform.id,
    to_account_id=user.id,
    amount=Decimal("100"),
    memo="Initial allocation",
)

# Check balances
user_account = exchange.get_account(user.id)
print(f"Balance: {user_account.balance}")  # 100
```

### Deposits and Withdrawals

```python
# Deposit (external funds in)
exchange.deposit(
    account_id=user.id,
    amount=Decimal("50"),
    source="external_payment",
)

# Withdraw (funds out)
exchange.withdraw(
    account_id=user.id,
    amount=Decimal("25"),
    destination="bank_account",
)
```

### Transaction History

```python
# Get transaction history
transactions = exchange.get_transaction_history(user.id)
for tx in transactions:
    print(f"{tx.transaction_type.name}: {tx.amount}")
```

### Statistics

```python
stats = exchange.get_stats()
# ExchangeStats with:
#   total_accounts
#   total_transactions
#   total_volume (sum of all transfers)
```

## Integration

The ValueExchangeManager works with:
- **BlockchainSimulator**: For immutable transaction recording
- **ProcessMonetizer**: For revenue distribution

## Example

See `examples/value_exchange_demo.py` for account creation, transfers, and transaction tracking.
