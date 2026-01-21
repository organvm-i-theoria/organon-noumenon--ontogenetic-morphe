---
title: ProcessMonetizer
system: Recursive–Generative Organizational Body
type: subsystem
category: value
tags: [monetization, revenue, products, payouts]
dependencies: [ValueExchangeManager, BlockchainSimulator]
---

# ProcessMonetizer

The **ProcessMonetizer** converts processes into monetizable outputs, managing product registration, revenue models, usage tracking, and payout generation.

## Overview

| Property | Value |
|----------|-------|
| Category | Value & Exchange |
| Module | `autogenrec.subsystems.value.process_monetizer` |
| Dependencies | ValueExchangeManager, BlockchainSimulator |

## Domain Models

### Enums

```python
class ProductType(Enum):
    SERVICE = auto()       # API or service access
    CONTENT = auto()       # Digital content
    LICENSE = auto()       # Software license
    SUBSCRIPTION = auto()  # Recurring subscription

class RevenueModel(Enum):
    FIXED = auto()         # Fixed price per unit
    USAGE_BASED = auto()   # Pay per use
    TIERED = auto()        # Tiered pricing
    SUBSCRIPTION = auto()  # Recurring fee

class ProcessStatus(Enum):
    DRAFT = auto()
    ACTIVE = auto()
    PAUSED = auto()
    RETIRED = auto()
```

### Core Models

- **MonetizedProcess**: Product with type, revenue model, pricing, usage stats
- **UsageRecord**: Record of product usage
- **Payout**: Revenue distribution with fees and net amount

## Process Loop

1. **Intake**: Receive product registrations, usage events, payout requests
2. **Process**: Track usage, calculate revenue, apply pricing
3. **Evaluate**: Validate usage records, compute fees
4. **Integrate**: Generate payouts, update statistics

## Public API

### Product Registration

```python
from autogenrec.subsystems.value.process_monetizer import (
    ProcessMonetizer, ProductType, RevenueModel
)
from decimal import Decimal

monetizer = ProcessMonetizer()

# Register a usage-based product
api_product = monetizer.register_process(
    name="SymboliQ API Access",
    owner_id="researcher_001",
    product_type=ProductType.SERVICE,
    revenue_model=RevenueModel.USAGE_BASED,
    usage_rate=Decimal("10"),  # 10 tokens per API call
    description="API access to pattern recognition engine",
)

# Activate the product
monetizer.activate_process(api_product.id)
```

### Usage Tracking

```python
# Record usage (called when user makes API call)
monetizer.record_usage(
    process_id=api_product.id,
    user_id="user_001",
    quantity=Decimal("1"),  # 1 API call
)

# Check product stats
product = monetizer.get_process(api_product.id)
print(f"Usage Count: {product.usage_count}")
print(f"Total Revenue: {product.total_revenue}")
```

### Payout Generation

```python
# Create payout for accumulated revenue
payout = monetizer.create_payout(api_product.id)

print(f"Gross Amount: {payout.amount}")
print(f"Platform Fee (10%): {payout.fee}")
print(f"Net to Owner: {payout.net_amount}")

# Use with ValueExchangeManager to execute transfer
exchange.transfer(
    platform_account.id,
    owner_account.id,
    payout.net_amount,
)
```

### Product Management

```python
# Pause product
monetizer.pause_process(product_id)

# Resume product
monetizer.resume_process(product_id)

# Retire product
monetizer.retire_process(product_id)
```

### Statistics

```python
stats = monetizer.get_stats()
# MonetizationStats with:
#   total_processes, active_processes
#   total_revenue, total_payouts
```

## Revenue Models

| Model | Pricing | Use Case |
|-------|---------|----------|
| FIXED | `base_price` per unit | One-time purchases |
| USAGE_BASED | `usage_rate` per use | API calls, metered services |
| TIERED | Volume-based rates | Bulk discounts |
| SUBSCRIPTION | `subscription_price` recurring | Monthly/annual access |

## Integration

The ProcessMonetizer works with:
- **ValueExchangeManager**: For payout transfers
- **BlockchainSimulator**: For transaction verification
- **ConsumptionManager**: For usage quota enforcement

## Example

See `examples/value_exchange_demo.py` for complete monetization workflow.
