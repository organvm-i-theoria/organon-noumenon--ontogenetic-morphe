"""
Process and Subsystem registries for the AutoGenRec system.

Registries provide discovery, lookup, and management of processes
and subsystems within the system.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import structlog
from ulid import ULID

from autogenrec.core.process import ProcessResult, ProcessState

if TYPE_CHECKING:
    from autogenrec.core.subsystem import Subsystem, SubsystemMetadata, SubsystemType

logger = structlog.get_logger()

T = TypeVar("T")


@dataclass
class RegistryEntry(Generic[T]):
    """An entry in a registry."""

    id: str
    item: T
    registered_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)


class ProcessRegistry:
    """
    Registry for tracking process executions.

    Maintains a record of all process runs with their results,
    enabling process discovery, status tracking, and history lookup.
    """

    def __init__(self, max_history: int = 1000) -> None:
        self._entries: dict[str, RegistryEntry[ProcessResult[Any]]] = {}
        self._by_process: dict[str, list[str]] = {}  # process_name -> [run_ids]
        self._max_history = max_history
        self._log = logger.bind(component="process_registry")

    def register(
        self,
        process_name: str,
        result: ProcessResult[Any],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Register a process execution result.

        Args:
            process_name: Name of the process
            result: Execution result
            metadata: Optional additional metadata

        Returns:
            Registry entry ID
        """
        entry_id = str(ULID())
        entry = RegistryEntry(
            id=entry_id,
            item=result,
            metadata=metadata or {},
        )

        self._entries[entry_id] = entry

        if process_name not in self._by_process:
            self._by_process[process_name] = []
        self._by_process[process_name].append(entry_id)

        # Trim history if needed
        if len(self._by_process[process_name]) > self._max_history:
            old_id = self._by_process[process_name].pop(0)
            self._entries.pop(old_id, None)

        self._log.debug(
            "process_registered",
            entry_id=entry_id,
            process_name=process_name,
            state=result.state.name,
        )

        return entry_id

    def get(self, entry_id: str) -> ProcessResult[Any] | None:
        """Get a process result by entry ID."""
        entry = self._entries.get(entry_id)
        return entry.item if entry else None

    def get_entry(self, entry_id: str) -> RegistryEntry[ProcessResult[Any]] | None:
        """Get a full registry entry by ID."""
        return self._entries.get(entry_id)

    def get_history(
        self,
        process_name: str,
        limit: int | None = None,
    ) -> list[ProcessResult[Any]]:
        """
        Get execution history for a process.

        Args:
            process_name: Name of the process
            limit: Maximum number of results (most recent first)

        Returns:
            List of process results, most recent first
        """
        entry_ids = self._by_process.get(process_name, [])
        results = [self._entries[eid].item for eid in reversed(entry_ids) if eid in self._entries]
        if limit:
            results = results[:limit]
        return results

    def get_latest(self, process_name: str) -> ProcessResult[Any] | None:
        """Get the most recent result for a process."""
        history = self.get_history(process_name, limit=1)
        return history[0] if history else None

    def get_by_state(self, state: ProcessState) -> list[ProcessResult[Any]]:
        """Get all results with a specific state."""
        return [entry.item for entry in self._entries.values() if entry.item.state == state]

    def list_processes(self) -> list[str]:
        """List all registered process names."""
        return list(self._by_process.keys())

    def clear(self, process_name: str | None = None) -> int:
        """
        Clear registry entries.

        Args:
            process_name: If provided, only clear entries for this process

        Returns:
            Number of entries removed
        """
        if process_name:
            entry_ids = self._by_process.pop(process_name, [])
            for entry_id in entry_ids:
                self._entries.pop(entry_id, None)
            return len(entry_ids)
        else:
            count = len(self._entries)
            self._entries.clear()
            self._by_process.clear()
            return count

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[ProcessResult[Any]]:
        return iter(entry.item for entry in self._entries.values())


class SubsystemRegistry:
    """
    Registry for tracking subsystems.

    Provides subsystem discovery, lookup by various criteria,
    and lifecycle management support.
    """

    def __init__(self) -> None:
        self._entries: dict[str, RegistryEntry["Subsystem"]] = {}
        self._by_type: dict["SubsystemType", list[str]] = {}
        self._by_tag: dict[str, list[str]] = {}
        self._log = logger.bind(component="subsystem_registry")

    def register(
        self,
        subsystem: "Subsystem",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Register a subsystem.

        Args:
            subsystem: Subsystem instance to register
            metadata: Optional additional metadata

        Returns:
            Registry entry ID
        """
        from autogenrec.core.subsystem import Subsystem

        if not isinstance(subsystem, Subsystem):
            raise TypeError(f"Expected Subsystem, got {type(subsystem)}")

        name = subsystem.name
        if name in self._entries:
            raise ValueError(f"Subsystem '{name}' already registered")

        entry = RegistryEntry(
            id=name,  # Use subsystem name as ID
            item=subsystem,
            metadata=metadata or {},
        )

        self._entries[name] = entry

        # Index by type
        sub_type = subsystem.metadata.type
        if sub_type not in self._by_type:
            self._by_type[sub_type] = []
        self._by_type[sub_type].append(name)

        # Index by tags
        for tag in subsystem.metadata.tags:
            if tag not in self._by_tag:
                self._by_tag[tag] = []
            self._by_tag[tag].append(name)

        self._log.debug(
            "subsystem_registered",
            name=name,
            type=sub_type.name,
            tags=list(subsystem.metadata.tags),
        )

        return name

    def unregister(self, name: str) -> bool:
        """
        Unregister a subsystem.

        Args:
            name: Subsystem name

        Returns:
            True if subsystem was unregistered
        """
        entry = self._entries.pop(name, None)
        if not entry:
            return False

        subsystem = entry.item
        sub_type = subsystem.metadata.type

        # Remove from type index
        if sub_type in self._by_type:
            self._by_type[sub_type] = [n for n in self._by_type[sub_type] if n != name]

        # Remove from tag indices
        for tag in subsystem.metadata.tags:
            if tag in self._by_tag:
                self._by_tag[tag] = [n for n in self._by_tag[tag] if n != name]

        self._log.debug("subsystem_unregistered", name=name)
        return True

    def get(self, name: str) -> "Subsystem | None":
        """Get a subsystem by name."""
        entry = self._entries.get(name)
        return entry.item if entry else None

    def get_entry(self, name: str) -> "RegistryEntry[Subsystem] | None":
        """Get a full registry entry by name."""
        return self._entries.get(name)

    def get_metadata(self, name: str) -> "SubsystemMetadata | None":
        """Get subsystem metadata by name."""
        subsystem = self.get(name)
        return subsystem.metadata if subsystem else None

    def get_by_type(self, sub_type: "SubsystemType") -> list["Subsystem"]:
        """Get all subsystems of a specific type."""
        names = self._by_type.get(sub_type, [])
        return [self._entries[n].item for n in names if n in self._entries]

    def get_by_tag(self, tag: str) -> list["Subsystem"]:
        """Get all subsystems with a specific tag."""
        names = self._by_tag.get(tag, [])
        return [self._entries[n].item for n in names if n in self._entries]

    def get_by_tags(self, tags: set[str], match_all: bool = True) -> list["Subsystem"]:
        """
        Get subsystems matching tags.

        Args:
            tags: Set of tags to match
            match_all: If True, subsystem must have all tags; if False, any tag

        Returns:
            List of matching subsystems
        """
        results: list["Subsystem"] = []
        for entry in self._entries.values():
            subsystem_tags = entry.item.metadata.tags
            if match_all:
                if tags <= subsystem_tags:
                    results.append(entry.item)
            else:
                if tags & subsystem_tags:
                    results.append(entry.item)
        return results

    def list_names(self) -> list[str]:
        """List all registered subsystem names."""
        return list(self._entries.keys())

    def list_types(self) -> list["SubsystemType"]:
        """List all registered subsystem types."""
        return list(self._by_type.keys())

    def list_tags(self) -> list[str]:
        """List all registered tags."""
        return list(self._by_tag.keys())

    def get_all(self) -> list["Subsystem"]:
        """Get all registered subsystems."""
        return [entry.item for entry in self._entries.values()]

    def get_running(self) -> list["Subsystem"]:
        """Get all currently running subsystems."""
        from autogenrec.core.subsystem import SubsystemState

        return [
            entry.item
            for entry in self._entries.values()
            if entry.item.subsystem_state == SubsystemState.RUNNING
        ]

    def clear(self) -> int:
        """Clear all registry entries."""
        count = len(self._entries)
        self._entries.clear()
        self._by_type.clear()
        self._by_tag.clear()
        return count

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator["Subsystem"]:
        return iter(entry.item for entry in self._entries.values())

    def __contains__(self, name: str) -> bool:
        return name in self._entries
