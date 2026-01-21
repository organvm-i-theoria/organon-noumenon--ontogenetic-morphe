---
title: ArchiveManager
system: Recursive–Generative Organizational Body
type: subsystem
category: data
tags: [archive, preservation, retention, retrieval]
dependencies: [ReferenceManager]
---

# ArchiveManager

The **ArchiveManager** preserves and retrieves records, managing retention policies, archive storage, and search capabilities.

## Overview

| Property | Value |
|----------|-------|
| Category | Data & Records |
| Module | `autogenrec.subsystems.data.archive_manager` |
| Dependencies | ReferenceManager |

## Domain Models

### Enums

```python
class ArchiveStatus(Enum):
    PENDING = auto()       # Awaiting archive
    ARCHIVED = auto()      # Successfully archived
    EXPIRED = auto()       # Past retention
    DELETED = auto()       # Removed

class RetentionPolicy(Enum):
    PERMANENT = auto()     # Keep forever
    YEARLY = auto()        # 1 year retention
    QUARTERLY = auto()     # 90 day retention
    MONTHLY = auto()       # 30 day retention
    TEMPORARY = auto()     # 7 day retention
```

### Core Models

- **ArchiveEntry**: Archived item with content, metadata, status
- **RetentionRule**: Retention policy with duration
- **SearchResult**: Archive search result

## Process Loop

1. **Intake**: Receive archive requests, retrieval queries
2. **Process**: Apply retention policies, index content
3. **Evaluate**: Check retention, validate integrity
4. **Integrate**: Store archives, emit archive events

## Public API

### Archiving

```python
from autogenrec.subsystems.data.archive_manager import (
    ArchiveManager, RetentionPolicy
)

archives = ArchiveManager()

# Archive an item
entry = await archives.archive(
    content={"type": "publication", "data": {...}},
    content_type="publication",
    retention_policy=RetentionPolicy.PERMANENT,
    metadata={"author": "researcher_001"},
)
```

### Retrieval

```python
# Retrieve by ID
entry = await archives.retrieve(archive_id)

# Search archives
results = await archives.search(
    query="symbolic processing",
    content_type="publication",
    limit=10,
)
```

### Retention Management

```python
# Set retention policy
await archives.set_retention(
    archive_id=entry.id,
    policy=RetentionPolicy.YEARLY,
)

# Process expired archives
expired = await archives.process_expired()
```

### Statistics

```python
stats = archives.get_stats()
# ArchiveStats with:
#   total_archives, archives_by_status
#   archives_by_policy
```

## Retention Policies

| Policy | Duration | Use Case |
|--------|----------|----------|
| PERMANENT | Forever | Critical records |
| YEARLY | 365 days | Annual reports |
| QUARTERLY | 90 days | Quarterly data |
| MONTHLY | 30 days | Monthly logs |
| TEMPORARY | 7 days | Transient data |

## Integration

The ArchiveManager provides:
- **AcademiaManager**: Publication archiving
- **ReferenceManager**: Archive references

## Example

See `examples/conflict_resolution_demo.py` for academic archiving workflows.
