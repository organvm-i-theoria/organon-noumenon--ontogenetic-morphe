"""
ArchiveManager: Organizes, preserves, and retrieves symbolic records.

Ensures structured access across the system and prevents data loss
through retention policies and systematic organization.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
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


class ArchiveCategory(Enum):
    """Categories for archived records."""

    SYSTEM = auto()  # System records and logs
    PROCESS = auto()  # Process execution records
    DATA = auto()  # General data records
    SIGNAL = auto()  # Signal history
    INTERPRETATION = auto()  # Interpretation results
    RULE = auto()  # Rule definitions and compilations
    REFERENCE = auto()  # Reference records
    USER = auto()  # User-generated content
    METADATA = auto()  # Metadata records
    AUDIT = auto()  # Audit trails


class ArchiveStatus(Enum):
    """Status of an archived record."""

    ACTIVE = auto()  # Currently accessible
    ARCHIVED = auto()  # In cold storage
    EXPIRED = auto()  # Past retention period
    DELETED = auto()  # Marked for deletion
    LOCKED = auto()  # Cannot be modified or deleted


class RetentionPolicy(BaseModel):
    """Policy for record retention."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    name: str
    category: ArchiveCategory
    retention_days: int = 365  # How long to keep records
    archive_after_days: int = 30  # When to move to cold storage
    auto_delete: bool = False  # Whether to auto-delete after retention
    priority: int = 0  # Higher priority policies override lower


class ArchiveRecord(BaseModel):
    """A record in the archive."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    category: ArchiveCategory
    status: ArchiveStatus = ArchiveStatus.ACTIVE

    # Content
    title: str
    content: Any
    content_type: str = "unknown"  # MIME type or custom type

    # Metadata
    source_subsystem: str = ""
    source_id: str = ""  # Original ID from source
    tags: frozenset[str] = Field(default_factory=frozenset)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Classification
    classification: str = "unclassified"  # Security/access classification
    searchable: bool = True

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    archived_at: datetime | None = None
    expires_at: datetime | None = None

    # Relationships
    parent_id: str | None = None
    related_ids: frozenset[str] = Field(default_factory=frozenset)


@dataclass
class SearchQuery:
    """Query parameters for searching archives."""

    text: str | None = None  # Full-text search
    category: ArchiveCategory | None = None
    status: ArchiveStatus | None = None
    source_subsystem: str | None = None
    tags: set[str] | None = None
    classification: str | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None
    limit: int = 100
    offset: int = 0


@dataclass
class SearchResult:
    """Result of an archive search."""

    records: list[ArchiveRecord]
    total_count: int
    query_time_ms: float


@dataclass
class ArchiveStats:
    """Statistics about the archive."""

    total_records: int
    active_records: int
    archived_records: int
    expired_records: int
    by_category: dict[str, int]
    oldest_record: datetime | None
    newest_record: datetime | None
    total_size_estimate: int  # Rough byte estimate


class ArchiveCatalog:
    """Catalog for organizing and indexing archive records."""

    def __init__(self) -> None:
        self._records: dict[str, ArchiveRecord] = {}
        self._by_category: dict[ArchiveCategory, set[str]] = {}
        self._by_source: dict[str, set[str]] = {}
        self._by_tag: dict[str, set[str]] = {}
        self._by_status: dict[ArchiveStatus, set[str]] = {}
        self._log = logger.bind(component="archive_catalog")

    @property
    def record_count(self) -> int:
        return len(self._records)

    def add(self, record: ArchiveRecord) -> None:
        """Add a record to the catalog."""
        self._records[record.id] = record

        # Index by category
        self._by_category.setdefault(record.category, set()).add(record.id)

        # Index by source
        if record.source_subsystem:
            self._by_source.setdefault(record.source_subsystem, set()).add(record.id)

        # Index by tags
        for tag in record.tags:
            self._by_tag.setdefault(tag, set()).add(record.id)

        # Index by status
        self._by_status.setdefault(record.status, set()).add(record.id)

        self._log.debug("record_added", record_id=record.id, category=record.category.name)

    def get(self, record_id: str) -> ArchiveRecord | None:
        """Get a record by ID."""
        return self._records.get(record_id)

    def search(self, query: SearchQuery) -> SearchResult:
        """Search records with filters."""
        start = datetime.now(UTC)
        results: list[ArchiveRecord] = []

        for record in self._records.values():
            if not record.searchable:
                continue
            if query.category and record.category != query.category:
                continue
            if query.status and record.status != query.status:
                continue
            if query.source_subsystem and record.source_subsystem != query.source_subsystem:
                continue
            if query.classification and record.classification != query.classification:
                continue
            if query.tags and not query.tags <= record.tags:
                continue
            if query.created_after and record.created_at < query.created_after:
                continue
            if query.created_before and record.created_at > query.created_before:
                continue
            if query.text:
                # Simple text search in title and content
                text_lower = query.text.lower()
                if text_lower not in record.title.lower():
                    content_str = str(record.content).lower()
                    if text_lower not in content_str:
                        continue
            results.append(record)

        # Sort by created_at descending
        results.sort(key=lambda r: r.created_at, reverse=True)

        total = len(results)
        results = results[query.offset : query.offset + query.limit]

        elapsed = (datetime.now(UTC) - start).total_seconds() * 1000

        return SearchResult(
            records=results,
            total_count=total,
            query_time_ms=elapsed,
        )

    def remove(self, record_id: str) -> bool:
        """Remove a record from the catalog."""
        if record_id not in self._records:
            return False

        record = self._records.pop(record_id)

        # Remove from indices
        if record.category in self._by_category:
            self._by_category[record.category].discard(record_id)
        if record.source_subsystem in self._by_source:
            self._by_source[record.source_subsystem].discard(record_id)
        for tag in record.tags:
            if tag in self._by_tag:
                self._by_tag[tag].discard(record_id)
        if record.status in self._by_status:
            self._by_status[record.status].discard(record_id)

        self._log.debug("record_removed", record_id=record_id)
        return True

    def get_by_category(self, category: ArchiveCategory) -> list[ArchiveRecord]:
        """Get all records in a category."""
        record_ids = self._by_category.get(category, set())
        return [self._records[rid] for rid in record_ids if rid in self._records]

    def get_by_source(self, source: str) -> list[ArchiveRecord]:
        """Get all records from a source."""
        record_ids = self._by_source.get(source, set())
        return [self._records[rid] for rid in record_ids if rid in self._records]

    def get_stats(self) -> ArchiveStats:
        """Get archive statistics."""
        by_category: dict[str, int] = {}
        for cat, ids in self._by_category.items():
            by_category[cat.name] = len(ids)

        oldest = None
        newest = None
        for record in self._records.values():
            if oldest is None or record.created_at < oldest:
                oldest = record.created_at
            if newest is None or record.created_at > newest:
                newest = record.created_at

        return ArchiveStats(
            total_records=len(self._records),
            active_records=len(self._by_status.get(ArchiveStatus.ACTIVE, set())),
            archived_records=len(self._by_status.get(ArchiveStatus.ARCHIVED, set())),
            expired_records=len(self._by_status.get(ArchiveStatus.EXPIRED, set())),
            by_category=by_category,
            oldest_record=oldest,
            newest_record=newest,
            total_size_estimate=sum(
                len(str(r.content)) for r in self._records.values()
            ),
        )


class RetentionManager:
    """Manages retention policies and enforcement."""

    def __init__(self) -> None:
        self._policies: dict[str, RetentionPolicy] = {}
        self._category_policies: dict[ArchiveCategory, list[str]] = {}
        self._log = logger.bind(component="retention_manager")

        # Register default policies
        self._register_default_policies()

    def _register_default_policies(self) -> None:
        """Register default retention policies."""
        defaults = [
            RetentionPolicy(
                name="system_default",
                category=ArchiveCategory.SYSTEM,
                retention_days=365,
                archive_after_days=30,
                auto_delete=False,
            ),
            RetentionPolicy(
                name="audit_retention",
                category=ArchiveCategory.AUDIT,
                retention_days=365 * 7,  # 7 years for audit
                archive_after_days=90,
                auto_delete=False,
                priority=100,
            ),
            RetentionPolicy(
                name="signal_retention",
                category=ArchiveCategory.SIGNAL,
                retention_days=90,
                archive_after_days=7,
                auto_delete=True,
            ),
        ]
        for policy in defaults:
            self.add_policy(policy)

    def add_policy(self, policy: RetentionPolicy) -> None:
        """Add or update a retention policy."""
        self._policies[policy.id] = policy
        self._category_policies.setdefault(policy.category, []).append(policy.id)

        # Sort by priority
        self._category_policies[policy.category].sort(
            key=lambda pid: self._policies[pid].priority,
            reverse=True,
        )

        self._log.debug(
            "policy_added",
            policy_id=policy.id,
            category=policy.category.name,
        )

    def get_policy_for(self, category: ArchiveCategory) -> RetentionPolicy | None:
        """Get the highest priority policy for a category."""
        policy_ids = self._category_policies.get(category, [])
        if policy_ids:
            return self._policies.get(policy_ids[0])
        return None

    def should_archive(self, record: ArchiveRecord) -> bool:
        """Check if a record should be archived."""
        policy = self.get_policy_for(record.category)
        if not policy:
            return False

        age_days = (datetime.now(UTC) - record.created_at).days
        return age_days >= policy.archive_after_days

    def should_delete(self, record: ArchiveRecord) -> bool:
        """Check if a record should be deleted."""
        policy = self.get_policy_for(record.category)
        if not policy or not policy.auto_delete:
            return False

        age_days = (datetime.now(UTC) - record.created_at).days
        return age_days >= policy.retention_days

    def get_expiration(self, record: ArchiveRecord) -> datetime | None:
        """Get the expiration date for a record."""
        policy = self.get_policy_for(record.category)
        if not policy:
            return None
        return record.created_at + timedelta(days=policy.retention_days)


class ArchiveManager(Subsystem):
    """
    Organizes, preserves, and retrieves symbolic records.

    Process Loop:
    1. Intake: Receive symbolic records and metadata
    2. Process: Index and classify records for retrieval
    3. Evaluate: Apply retention policies and prepare for storage
    4. Integrate: Provide records on demand to other processes
    """

    def __init__(self) -> None:
        metadata = SubsystemMetadata(
            name="archive_manager",
            display_name="Archive Manager",
            description="Organizes, preserves, and retrieves symbolic records",
            type=SubsystemType.DATA,
            tags=frozenset(["archive", "records", "storage", "retrieval"]),
            input_types=frozenset(["ARCHIVE", "REFERENCE", "SCHEMA"]),
            output_types=frozenset(["ARCHIVE", "REFERENCE"]),
            subscribed_topics=frozenset([
                str(SubsystemTopics.ALL),
                "archive.store.#",
                "archive.retrieve.#",
                "archive.search.#",
            ]),
            published_topics=frozenset([
                "archive.stored",
                "archive.retrieved",
                "archive.expired",
                "archive.deleted",
            ]),
        )
        super().__init__(metadata)

        self._catalog = ArchiveCatalog()
        self._retention = RetentionManager()

    @property
    def record_count(self) -> int:
        return self._catalog.record_count

    async def intake(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> SymbolicInput:
        """Phase 1: Receive symbolic records and metadata."""
        if not input_data.values:
            self._log.debug("empty_input")
            return input_data

        self._log.debug("intake_complete", value_count=len(input_data.values))
        return input_data

    async def process(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> list[ArchiveRecord]:
        """Phase 2: Index and classify records."""
        records: list[ArchiveRecord] = []

        for value in input_data.values:
            record = self._create_record(value)
            if record:
                # Apply retention policy
                expires_at = self._retention.get_expiration(record)
                if expires_at:
                    # Create new record with expiration
                    record = ArchiveRecord(
                        id=record.id,
                        category=record.category,
                        status=record.status,
                        title=record.title,
                        content=record.content,
                        content_type=record.content_type,
                        source_subsystem=record.source_subsystem,
                        source_id=record.source_id,
                        tags=record.tags,
                        metadata=record.metadata,
                        classification=record.classification,
                        searchable=record.searchable,
                        created_at=record.created_at,
                        expires_at=expires_at,
                        parent_id=record.parent_id,
                        related_ids=record.related_ids,
                    )

                self._catalog.add(record)
                records.append(record)

                self._log.debug(
                    "record_indexed",
                    record_id=record.id,
                    category=record.category.name,
                )

        return records

    async def evaluate(
        self, intermediate: list[ArchiveRecord], ctx: ProcessContext[dict[str, Any]]
    ) -> tuple[SymbolicOutput, bool]:
        """Phase 3: Create output with archived records."""
        values: list[SymbolicValue] = []

        for record in intermediate:
            value = SymbolicValue(
                type=SymbolicValueType.ARCHIVE,
                content={
                    "record_id": record.id,
                    "title": record.title,
                    "category": record.category.name,
                    "status": record.status.name,
                    "source_subsystem": record.source_subsystem,
                    "expires_at": record.expires_at.isoformat() if record.expires_at else None,
                },
                source_subsystem=self.name,
                tags=record.tags | frozenset(["archive", record.category.name.lower()]),
                meaning=f"Archived: {record.title}",
                confidence=1.0,
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
        """Phase 4: Emit events for archived records."""
        if self._message_bus and output.values:
            for value in output.values:
                await self.emit_event(
                    "archive.stored",
                    {
                        "record_id": value.content.get("record_id"),
                        "title": value.content.get("title"),
                        "category": value.content.get("category"),
                    },
                )

        return None

    def _create_record(self, value: SymbolicValue) -> ArchiveRecord | None:
        """Create an ArchiveRecord from a SymbolicValue."""
        content = value.content
        if not isinstance(content, dict):
            # Wrap non-dict content
            content = {"data": content}

        try:
            # Determine category from value type
            category_map = {
                SymbolicValueType.SIGNAL: ArchiveCategory.SIGNAL,
                SymbolicValueType.RULE: ArchiveCategory.RULE,
                SymbolicValueType.REFERENCE: ArchiveCategory.REFERENCE,
                SymbolicValueType.PATTERN: ArchiveCategory.INTERPRETATION,
                SymbolicValueType.SCHEMA: ArchiveCategory.DATA,
            }
            category = category_map.get(value.type, ArchiveCategory.DATA)

            # Override with explicit category
            if "category" in content:
                try:
                    category = ArchiveCategory[content["category"].upper()]
                except KeyError:
                    pass

            record = ArchiveRecord(
                id=content.get("id", str(ULID())),
                category=category,
                title=content.get("title", value.meaning or f"record_{value.id[:8]}"),
                content=content.get("content", content),
                content_type=content.get("content_type", value.type.name),
                source_subsystem=value.source_subsystem or "",
                source_id=value.id,
                tags=frozenset(content.get("tags", [])) | value.tags,
                metadata=content.get("metadata", {}),
                classification=content.get("classification", "unclassified"),
                searchable=content.get("searchable", True),
                parent_id=content.get("parent_id"),
                related_ids=frozenset(content.get("related_ids", [])),
            )

            return record

        except Exception as e:
            self._log.warning("create_record_failed", value_id=value.id, error=str(e))
            return None

    # --- Message handlers ---

    async def handle_event(self, message: Message) -> None:
        """Handle incoming events."""
        if message.topic.startswith("archive.retrieve"):
            self._log.debug("retrieve_request_received", message_id=message.id)
        elif message.topic.startswith("archive.search"):
            self._log.debug("search_request_received", message_id=message.id)

    async def handle_signal(self, signal: Any) -> None:
        """Handle incoming signals."""
        self._log.debug("signal_received", signal_id=getattr(signal, "id", "unknown"))

    # --- Public API ---

    def archive(
        self,
        title: str,
        content: Any,
        category: ArchiveCategory = ArchiveCategory.DATA,
        **kwargs: Any,
    ) -> ArchiveRecord:
        """Archive content with the given title."""
        record = ArchiveRecord(
            category=category,
            title=title,
            content=content,
            content_type=kwargs.get("content_type", type(content).__name__),
            source_subsystem=kwargs.get("source_subsystem", ""),
            source_id=kwargs.get("source_id", ""),
            tags=frozenset(kwargs.get("tags", [])),
            metadata=kwargs.get("metadata", {}),
            classification=kwargs.get("classification", "unclassified"),
            searchable=kwargs.get("searchable", True),
            parent_id=kwargs.get("parent_id"),
            related_ids=frozenset(kwargs.get("related_ids", [])),
        )

        # Apply retention policy
        expires_at = self._retention.get_expiration(record)
        if expires_at:
            record = ArchiveRecord(
                id=record.id,
                category=record.category,
                status=record.status,
                title=record.title,
                content=record.content,
                content_type=record.content_type,
                source_subsystem=record.source_subsystem,
                source_id=record.source_id,
                tags=record.tags,
                metadata=record.metadata,
                classification=record.classification,
                searchable=record.searchable,
                expires_at=expires_at,
                parent_id=record.parent_id,
                related_ids=record.related_ids,
            )

        self._catalog.add(record)
        return record

    def retrieve(self, record_id: str) -> ArchiveRecord | None:
        """Retrieve a record by ID."""
        return self._catalog.get(record_id)

    def search(self, query: SearchQuery) -> SearchResult:
        """Search the archive."""
        return self._catalog.search(query)

    def search_text(self, text: str, limit: int = 100) -> SearchResult:
        """Simple text search."""
        return self._catalog.search(SearchQuery(text=text, limit=limit))

    def get_by_category(self, category: ArchiveCategory) -> list[ArchiveRecord]:
        """Get all records in a category."""
        return self._catalog.get_by_category(category)

    def get_by_source(self, source: str) -> list[ArchiveRecord]:
        """Get all records from a source subsystem."""
        return self._catalog.get_by_source(source)

    def delete(self, record_id: str) -> bool:
        """Delete a record."""
        return self._catalog.remove(record_id)

    def add_retention_policy(self, policy: RetentionPolicy) -> None:
        """Add a retention policy."""
        self._retention.add_policy(policy)

    def get_stats(self) -> ArchiveStats:
        """Get archive statistics."""
        return self._catalog.get_stats()

    def enforce_retention(self) -> tuple[int, int]:
        """
        Enforce retention policies on all records.

        Returns: (archived_count, deleted_count)
        """
        archived = 0
        deleted = 0

        for record in list(self._catalog._records.values()):
            if record.status == ArchiveStatus.LOCKED:
                continue

            if self._retention.should_delete(record):
                self._catalog.remove(record.id)
                deleted += 1
            elif record.status == ArchiveStatus.ACTIVE and self._retention.should_archive(record):
                # Create new record with archived status
                self._catalog.remove(record.id)
                archived_record = ArchiveRecord(
                    id=record.id,
                    category=record.category,
                    status=ArchiveStatus.ARCHIVED,
                    title=record.title,
                    content=record.content,
                    content_type=record.content_type,
                    source_subsystem=record.source_subsystem,
                    source_id=record.source_id,
                    tags=record.tags,
                    metadata=record.metadata,
                    classification=record.classification,
                    searchable=record.searchable,
                    created_at=record.created_at,
                    archived_at=datetime.now(UTC),
                    expires_at=record.expires_at,
                    parent_id=record.parent_id,
                    related_ids=record.related_ids,
                )
                self._catalog.add(archived_record)
                archived += 1

        return archived, deleted

    def clear(self) -> int:
        """Clear all records. Returns count cleared."""
        count = self._catalog.record_count
        self._catalog = ArchiveCatalog()
        return count
