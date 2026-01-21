"""
AcademiaManager: Manages learning, research, and publication cycles.

Orchestrates academic processes including research projects, learning cycles,
publications, and scholarly knowledge dissemination.
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


class ResearchStatus(Enum):
    """Status of a research project."""

    PROPOSED = auto()
    APPROVED = auto()
    IN_PROGRESS = auto()
    REVIEW = auto()
    COMPLETED = auto()
    ARCHIVED = auto()
    REJECTED = auto()


class PublicationType(Enum):
    """Types of academic publications."""

    PAPER = auto()
    ARTICLE = auto()
    THESIS = auto()
    REPORT = auto()
    DATASET = auto()
    PRESENTATION = auto()
    NOTE = auto()


class LearningStatus(Enum):
    """Status of a learning cycle."""

    NOT_STARTED = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    PAUSED = auto()
    ABANDONED = auto()


class ResearchProject(BaseModel):
    """A research project."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    title: str
    description: str = ""
    status: ResearchStatus = ResearchStatus.PROPOSED

    # Research details
    methodology: str = ""
    hypothesis: str = ""
    objectives: tuple[str, ...] = Field(default_factory=tuple)
    keywords: frozenset[str] = Field(default_factory=frozenset)

    # Team
    lead_researcher_id: str | None = None
    collaborator_ids: frozenset[str] = Field(default_factory=frozenset)

    # Timeline
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    deadline: datetime | None = None

    # Progress
    progress_percent: float = 0.0
    milestones: tuple[dict[str, Any], ...] = Field(default_factory=tuple)

    # Outputs
    publication_ids: frozenset[str] = Field(default_factory=frozenset)
    dataset_ids: frozenset[str] = Field(default_factory=frozenset)

    # Metadata
    tags: frozenset[str] = Field(default_factory=frozenset)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Publication(BaseModel):
    """An academic publication."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    title: str
    publication_type: PublicationType = PublicationType.PAPER
    abstract: str = ""

    # Content
    content: str = ""
    sections: tuple[dict[str, Any], ...] = Field(default_factory=tuple)

    # Authors
    author_ids: tuple[str, ...] = Field(default_factory=tuple)
    corresponding_author_id: str | None = None

    # References
    citation_ids: frozenset[str] = Field(default_factory=frozenset)
    cited_by_ids: frozenset[str] = Field(default_factory=frozenset)

    # Project link
    project_id: str | None = None

    # Publication details
    venue: str = ""  # Journal, conference, etc.
    doi: str | None = None
    published_at: datetime | None = None
    is_published: bool = False

    # Metadata
    keywords: frozenset[str] = Field(default_factory=frozenset)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tags: frozenset[str] = Field(default_factory=frozenset)


class Citation(BaseModel):
    """A citation/reference."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    title: str
    authors: tuple[str, ...] = Field(default_factory=tuple)
    year: int | None = None
    venue: str = ""
    doi: str | None = None
    url: str | None = None

    # Citation text
    citation_text: str = ""

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tags: frozenset[str] = Field(default_factory=frozenset)


class LearningCycle(BaseModel):
    """A learning cycle/course."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    title: str
    description: str = ""
    status: LearningStatus = LearningStatus.NOT_STARTED

    # Learner
    learner_id: str | None = None

    # Content
    topics: tuple[str, ...] = Field(default_factory=tuple)
    materials: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    assessments: tuple[dict[str, Any], ...] = Field(default_factory=tuple)

    # Progress
    progress_percent: float = 0.0
    completed_topics: frozenset[str] = Field(default_factory=frozenset)
    assessment_scores: dict[str, float] = Field(default_factory=dict)

    # Timeline
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    estimated_duration: timedelta | None = None

    tags: frozenset[str] = Field(default_factory=frozenset)


class AcademicArchive(BaseModel):
    """An archived academic work."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    title: str
    archive_type: str = "publication"  # publication, project, dataset, etc.
    source_id: str  # ID of the archived item

    # Archive details
    content_hash: str = ""
    version: str = "1.0"
    archived_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Access
    is_public: bool = True
    access_count: int = 0

    tags: frozenset[str] = Field(default_factory=frozenset)
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class ResearchResult:
    """Result of a research operation."""

    project_id: str
    success: bool
    status: ResearchStatus | None = None
    error: str | None = None


@dataclass
class PublicationResult:
    """Result of a publication operation."""

    publication_id: str
    success: bool
    is_published: bool = False
    error: str | None = None


@dataclass
class AcademiaStats:
    """Statistics about academic activities."""

    total_projects: int
    active_projects: int
    completed_projects: int
    total_publications: int
    published_count: int
    total_citations: int
    total_learning_cycles: int
    total_archives: int


class ProjectRegistry:
    """Registry of research projects."""

    def __init__(self) -> None:
        self._projects: dict[str, ResearchProject] = {}
        self._by_status: dict[ResearchStatus, set[str]] = {}
        self._by_researcher: dict[str, set[str]] = {}
        self._log = logger.bind(component="project_registry")

    @property
    def project_count(self) -> int:
        return len(self._projects)

    def register(self, project: ResearchProject) -> None:
        """Register a project."""
        self._projects[project.id] = project
        self._by_status.setdefault(project.status, set()).add(project.id)
        if project.lead_researcher_id:
            self._by_researcher.setdefault(project.lead_researcher_id, set()).add(project.id)
        self._log.debug("project_registered", project_id=project.id, title=project.title)

    def get(self, project_id: str) -> ResearchProject | None:
        """Get a project by ID."""
        return self._projects.get(project_id)

    def get_by_status(self, status: ResearchStatus) -> list[ResearchProject]:
        """Get projects by status."""
        ids = self._by_status.get(status, set())
        return [self._projects[pid] for pid in ids if pid in self._projects]

    def get_by_researcher(self, researcher_id: str) -> list[ResearchProject]:
        """Get projects by researcher."""
        ids = self._by_researcher.get(researcher_id, set())
        return [self._projects[pid] for pid in ids if pid in self._projects]

    def update(self, project_id: str, **updates: Any) -> ResearchProject | None:
        """Update a project."""
        project = self._projects.get(project_id)
        if not project:
            return None

        # Build updated project
        old_status = project.status
        new_status = updates.get("status", project.status)

        data = {
            "id": project.id,
            "title": updates.get("title", project.title),
            "description": updates.get("description", project.description),
            "status": new_status,
            "methodology": updates.get("methodology", project.methodology),
            "hypothesis": updates.get("hypothesis", project.hypothesis),
            "objectives": updates.get("objectives", project.objectives),
            "keywords": updates.get("keywords", project.keywords),
            "lead_researcher_id": updates.get("lead_researcher_id", project.lead_researcher_id),
            "collaborator_ids": updates.get("collaborator_ids", project.collaborator_ids),
            "created_at": project.created_at,
            "started_at": updates.get("started_at", project.started_at),
            "completed_at": updates.get("completed_at", project.completed_at),
            "deadline": updates.get("deadline", project.deadline),
            "progress_percent": updates.get("progress_percent", project.progress_percent),
            "milestones": updates.get("milestones", project.milestones),
            "publication_ids": updates.get("publication_ids", project.publication_ids),
            "dataset_ids": updates.get("dataset_ids", project.dataset_ids),
            "tags": updates.get("tags", project.tags),
            "metadata": updates.get("metadata", project.metadata),
        }
        updated = ResearchProject(**data)
        self._projects[project_id] = updated

        # Update status index
        if old_status != new_status:
            if old_status in self._by_status:
                self._by_status[old_status].discard(project_id)
            self._by_status.setdefault(new_status, set()).add(project_id)

        return updated


class PublicationRegistry:
    """Registry of publications."""

    def __init__(self) -> None:
        self._publications: dict[str, Publication] = {}
        self._by_project: dict[str, set[str]] = {}
        self._by_author: dict[str, set[str]] = {}
        self._log = logger.bind(component="publication_registry")

    @property
    def publication_count(self) -> int:
        return len(self._publications)

    def register(self, publication: Publication) -> None:
        """Register a publication."""
        self._publications[publication.id] = publication
        if publication.project_id:
            self._by_project.setdefault(publication.project_id, set()).add(publication.id)
        for author_id in publication.author_ids:
            self._by_author.setdefault(author_id, set()).add(publication.id)
        self._log.debug("publication_registered", pub_id=publication.id, title=publication.title)

    def get(self, publication_id: str) -> Publication | None:
        """Get a publication by ID."""
        return self._publications.get(publication_id)

    def get_by_project(self, project_id: str) -> list[Publication]:
        """Get publications for a project."""
        ids = self._by_project.get(project_id, set())
        return [self._publications[pid] for pid in ids if pid in self._publications]

    def get_by_author(self, author_id: str) -> list[Publication]:
        """Get publications by author."""
        ids = self._by_author.get(author_id, set())
        return [self._publications[pid] for pid in ids if pid in self._publications]

    def get_published(self) -> list[Publication]:
        """Get all published publications."""
        return [p for p in self._publications.values() if p.is_published]

    def update(self, publication_id: str, **updates: Any) -> Publication | None:
        """Update a publication."""
        pub = self._publications.get(publication_id)
        if not pub:
            return None

        data = {
            "id": pub.id,
            "title": updates.get("title", pub.title),
            "publication_type": updates.get("publication_type", pub.publication_type),
            "abstract": updates.get("abstract", pub.abstract),
            "content": updates.get("content", pub.content),
            "sections": updates.get("sections", pub.sections),
            "author_ids": updates.get("author_ids", pub.author_ids),
            "corresponding_author_id": updates.get("corresponding_author_id", pub.corresponding_author_id),
            "citation_ids": updates.get("citation_ids", pub.citation_ids),
            "cited_by_ids": updates.get("cited_by_ids", pub.cited_by_ids),
            "project_id": updates.get("project_id", pub.project_id),
            "venue": updates.get("venue", pub.venue),
            "doi": updates.get("doi", pub.doi),
            "published_at": updates.get("published_at", pub.published_at),
            "is_published": updates.get("is_published", pub.is_published),
            "keywords": updates.get("keywords", pub.keywords),
            "created_at": pub.created_at,
            "tags": updates.get("tags", pub.tags),
        }
        updated = Publication(**data)
        self._publications[publication_id] = updated
        return updated


class CitationManager:
    """Manages citations and references."""

    def __init__(self) -> None:
        self._citations: dict[str, Citation] = {}
        self._log = logger.bind(component="citation_manager")

    @property
    def citation_count(self) -> int:
        return len(self._citations)

    def add(self, citation: Citation) -> None:
        """Add a citation."""
        self._citations[citation.id] = citation
        self._log.debug("citation_added", citation_id=citation.id, title=citation.title)

    def get(self, citation_id: str) -> Citation | None:
        """Get a citation by ID."""
        return self._citations.get(citation_id)

    def search(self, query: str) -> list[Citation]:
        """Search citations by title or author."""
        query_lower = query.lower()
        results = []
        for citation in self._citations.values():
            if query_lower in citation.title.lower():
                results.append(citation)
            elif any(query_lower in author.lower() for author in citation.authors):
                results.append(citation)
        return results


class LearningManager:
    """Manages learning cycles."""

    def __init__(self) -> None:
        self._cycles: dict[str, LearningCycle] = {}
        self._by_learner: dict[str, set[str]] = {}
        self._log = logger.bind(component="learning_manager")

    @property
    def cycle_count(self) -> int:
        return len(self._cycles)

    def create(self, cycle: LearningCycle) -> None:
        """Create a learning cycle."""
        self._cycles[cycle.id] = cycle
        if cycle.learner_id:
            self._by_learner.setdefault(cycle.learner_id, set()).add(cycle.id)
        self._log.debug("cycle_created", cycle_id=cycle.id, title=cycle.title)

    def get(self, cycle_id: str) -> LearningCycle | None:
        """Get a learning cycle by ID."""
        return self._cycles.get(cycle_id)

    def get_by_learner(self, learner_id: str) -> list[LearningCycle]:
        """Get cycles for a learner."""
        ids = self._by_learner.get(learner_id, set())
        return [self._cycles[cid] for cid in ids if cid in self._cycles]

    def update_progress(
        self,
        cycle_id: str,
        progress: float,
        completed_topic: str | None = None,
    ) -> LearningCycle | None:
        """Update learning progress."""
        cycle = self._cycles.get(cycle_id)
        if not cycle:
            return None

        completed = set(cycle.completed_topics)
        if completed_topic:
            completed.add(completed_topic)

        # Determine status
        status = cycle.status
        if progress >= 100:
            status = LearningStatus.COMPLETED
        elif progress > 0 and status == LearningStatus.NOT_STARTED:
            status = LearningStatus.IN_PROGRESS

        updated = LearningCycle(
            id=cycle.id,
            title=cycle.title,
            description=cycle.description,
            status=status,
            learner_id=cycle.learner_id,
            topics=cycle.topics,
            materials=cycle.materials,
            assessments=cycle.assessments,
            progress_percent=min(100, progress),
            completed_topics=frozenset(completed),
            assessment_scores=cycle.assessment_scores,
            created_at=cycle.created_at,
            started_at=cycle.started_at or datetime.now(UTC),
            completed_at=datetime.now(UTC) if progress >= 100 else cycle.completed_at,
            estimated_duration=cycle.estimated_duration,
            tags=cycle.tags,
        )
        self._cycles[cycle_id] = updated
        return updated


class ArchiveManager:
    """Manages academic archives."""

    def __init__(self) -> None:
        self._archives: dict[str, AcademicArchive] = {}
        self._by_source: dict[str, str] = {}  # source_id -> archive_id
        self._log = logger.bind(component="archive_manager")

    @property
    def archive_count(self) -> int:
        return len(self._archives)

    def archive(self, archive: AcademicArchive) -> None:
        """Archive an item."""
        self._archives[archive.id] = archive
        self._by_source[archive.source_id] = archive.id
        self._log.debug("item_archived", archive_id=archive.id, source_id=archive.source_id)

    def get(self, archive_id: str) -> AcademicArchive | None:
        """Get an archive by ID."""
        return self._archives.get(archive_id)

    def get_by_source(self, source_id: str) -> AcademicArchive | None:
        """Get archive for a source item."""
        archive_id = self._by_source.get(source_id)
        if archive_id:
            return self._archives.get(archive_id)
        return None

    def increment_access(self, archive_id: str) -> AcademicArchive | None:
        """Increment access count."""
        archive = self._archives.get(archive_id)
        if not archive:
            return None

        updated = AcademicArchive(
            id=archive.id,
            title=archive.title,
            archive_type=archive.archive_type,
            source_id=archive.source_id,
            content_hash=archive.content_hash,
            version=archive.version,
            archived_at=archive.archived_at,
            is_public=archive.is_public,
            access_count=archive.access_count + 1,
            tags=archive.tags,
            metadata=archive.metadata,
        )
        self._archives[archive_id] = updated
        return updated


class AcademiaManager(Subsystem):
    """
    Manages learning, research, and scholarly cycles.

    Process Loop:
    1. Intake: Gather materials and resources
    2. Apply Methods: Perform analysis and synthesis
    3. Evaluate: Assess results for rigor
    4. Archive: Publish or archive outputs
    """

    def __init__(self) -> None:
        metadata = SubsystemMetadata(
            name="academia_manager",
            display_name="Academia Manager",
            description="Manages learning, research, and publication cycles",
            type=SubsystemType.ACADEMIC,
            tags=frozenset(["academic", "research", "learning", "publication"]),
            input_types=frozenset(["NARRATIVE", "REFERENCE", "PATTERN", "DATA"]),
            output_types=frozenset(["REFERENCE", "ARCHIVE", "PATTERN", "PUBLICATION"]),
            subscribed_topics=frozenset([
                str(SubsystemTopics.ALL),
                "academia.#",
                "research.#",
                "publication.#",
            ]),
            published_topics=frozenset([
                "academia.project.created",
                "academia.project.completed",
                "academia.publication.published",
                "academia.learning.completed",
            ]),
        )
        super().__init__(metadata)

        self._projects = ProjectRegistry()
        self._publications = PublicationRegistry()
        self._citations = CitationManager()
        self._learning = LearningManager()
        self._archives = ArchiveManager()

    @property
    def project_count(self) -> int:
        return self._projects.project_count

    @property
    def publication_count(self) -> int:
        return self._publications.publication_count

    @property
    def citation_count(self) -> int:
        return self._citations.citation_count

    @property
    def learning_cycle_count(self) -> int:
        return self._learning.cycle_count

    @property
    def archive_count(self) -> int:
        return self._archives.archive_count

    async def intake(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> SymbolicInput:
        """Phase 1: Gather materials and resources."""
        if not input_data.values:
            self._log.debug("empty_input")
            return input_data

        self._log.debug("intake_complete", value_count=len(input_data.values))
        return input_data

    async def process(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> list[ResearchResult | PublicationResult]:
        """Phase 2: Process academic operations."""
        results: list[ResearchResult | PublicationResult] = []

        for value in input_data.values:
            content = value.content
            if not isinstance(content, dict):
                continue

            action = content.get("action", "create_project")

            if action == "create_project":
                results.append(self._create_project_from_value(value))
            elif action == "create_publication":
                results.append(self._create_publication_from_value(value))

        return results

    async def evaluate(
        self,
        intermediate: list[ResearchResult | PublicationResult],
        ctx: ProcessContext[dict[str, Any]],
    ) -> tuple[SymbolicOutput, bool]:
        """Phase 3: Assess and synthesize results."""
        values: list[SymbolicValue] = []

        for result in intermediate:
            if isinstance(result, ResearchResult):
                value = SymbolicValue(
                    type=SymbolicValueType.REFERENCE,
                    content={
                        "project_id": result.project_id,
                        "success": result.success,
                        "status": result.status.name if result.status else None,
                        "error": result.error,
                    },
                    source_subsystem=self.name,
                    tags=frozenset(["research", "project"]),
                    meaning="Research project result",
                    confidence=1.0 if result.success else 0.0,
                )
            else:
                value = SymbolicValue(
                    type=SymbolicValueType.REFERENCE,
                    content={
                        "publication_id": result.publication_id,
                        "success": result.success,
                        "is_published": result.is_published,
                        "error": result.error,
                    },
                    source_subsystem=self.name,
                    tags=frozenset(["publication"]),
                    meaning="Publication result",
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
        """Phase 4: Publish and archive outputs."""
        if self._message_bus and output.values:
            for value in output.values:
                content = value.content
                if not isinstance(content, dict):
                    continue

                if "project_id" in content and content.get("success"):
                    await self.emit_event(
                        "academia.project.created",
                        {"project_id": content["project_id"]},
                    )
                elif "publication_id" in content and content.get("is_published"):
                    await self.emit_event(
                        "academia.publication.published",
                        {"publication_id": content["publication_id"]},
                    )

        return None

    def _create_project_from_value(self, value: SymbolicValue) -> ResearchResult:
        """Create a project from a SymbolicValue."""
        content = value.content
        if not isinstance(content, dict):
            return ResearchResult(project_id="", success=False, error="Invalid content")

        try:
            project = ResearchProject(
                title=content.get("title", f"Project {value.id[:8]}"),
                description=content.get("description", ""),
                methodology=content.get("methodology", ""),
                hypothesis=content.get("hypothesis", ""),
                objectives=tuple(content.get("objectives", [])),
                keywords=frozenset(content.get("keywords", [])),
                lead_researcher_id=content.get("lead_researcher_id"),
                collaborator_ids=frozenset(content.get("collaborator_ids", [])),
                tags=frozenset(content.get("tags", [])) | value.tags,
            )
            self._projects.register(project)
            return ResearchResult(
                project_id=project.id,
                success=True,
                status=project.status,
            )

        except Exception as e:
            return ResearchResult(project_id="", success=False, error=str(e))

    def _create_publication_from_value(self, value: SymbolicValue) -> PublicationResult:
        """Create a publication from a SymbolicValue."""
        content = value.content
        if not isinstance(content, dict):
            return PublicationResult(publication_id="", success=False, error="Invalid content")

        try:
            type_str = content.get("publication_type", "PAPER")
            try:
                pub_type = PublicationType[type_str.upper()]
            except KeyError:
                pub_type = PublicationType.PAPER

            publication = Publication(
                title=content.get("title", f"Publication {value.id[:8]}"),
                publication_type=pub_type,
                abstract=content.get("abstract", ""),
                content=content.get("content", ""),
                author_ids=tuple(content.get("author_ids", [])),
                project_id=content.get("project_id"),
                keywords=frozenset(content.get("keywords", [])),
                tags=frozenset(content.get("tags", [])) | value.tags,
            )
            self._publications.register(publication)
            return PublicationResult(
                publication_id=publication.id,
                success=True,
                is_published=publication.is_published,
            )

        except Exception as e:
            return PublicationResult(publication_id="", success=False, error=str(e))

    # --- Message handlers ---

    async def handle_event(self, message: Message) -> None:
        """Handle incoming events."""
        if message.topic.startswith("academia.") or message.topic.startswith("research."):
            self._log.debug("event_received", topic=message.topic)

    async def handle_signal(self, signal: Any) -> None:
        """Handle incoming signals."""
        self._log.debug("signal_received", signal_id=getattr(signal, "id", "unknown"))

    # --- Public API: Research Projects ---

    def create_project(
        self,
        title: str,
        **kwargs: Any,
    ) -> ResearchProject:
        """Create a research project."""
        project = ResearchProject(
            title=title,
            description=kwargs.get("description", ""),
            methodology=kwargs.get("methodology", ""),
            hypothesis=kwargs.get("hypothesis", ""),
            objectives=tuple(kwargs.get("objectives", [])),
            keywords=frozenset(kwargs.get("keywords", [])),
            lead_researcher_id=kwargs.get("lead_researcher_id"),
            collaborator_ids=frozenset(kwargs.get("collaborator_ids", [])),
            deadline=kwargs.get("deadline"),
            tags=frozenset(kwargs.get("tags", [])),
            metadata=kwargs.get("metadata", {}),
        )
        self._projects.register(project)
        return project

    def get_project(self, project_id: str) -> ResearchProject | None:
        """Get a project by ID."""
        return self._projects.get(project_id)

    def start_project(self, project_id: str) -> ResearchProject | None:
        """Start a research project."""
        return self._projects.update(
            project_id,
            status=ResearchStatus.IN_PROGRESS,
            started_at=datetime.now(UTC),
        )

    def complete_project(self, project_id: str) -> ResearchProject | None:
        """Complete a research project."""
        return self._projects.update(
            project_id,
            status=ResearchStatus.COMPLETED,
            completed_at=datetime.now(UTC),
            progress_percent=100.0,
        )

    def update_project_progress(
        self,
        project_id: str,
        progress: float,
    ) -> ResearchProject | None:
        """Update project progress."""
        return self._projects.update(project_id, progress_percent=progress)

    def get_active_projects(self) -> list[ResearchProject]:
        """Get all active (in-progress) projects."""
        return self._projects.get_by_status(ResearchStatus.IN_PROGRESS)

    # --- Public API: Publications ---

    def create_publication(
        self,
        title: str,
        publication_type: PublicationType = PublicationType.PAPER,
        **kwargs: Any,
    ) -> Publication:
        """Create a publication."""
        publication = Publication(
            title=title,
            publication_type=publication_type,
            abstract=kwargs.get("abstract", ""),
            content=kwargs.get("content", ""),
            author_ids=tuple(kwargs.get("author_ids", [])),
            corresponding_author_id=kwargs.get("corresponding_author_id"),
            citation_ids=frozenset(kwargs.get("citation_ids", [])),
            project_id=kwargs.get("project_id"),
            venue=kwargs.get("venue", ""),
            keywords=frozenset(kwargs.get("keywords", [])),
            tags=frozenset(kwargs.get("tags", [])),
        )
        self._publications.register(publication)
        return publication

    def get_publication(self, publication_id: str) -> Publication | None:
        """Get a publication by ID."""
        return self._publications.get(publication_id)

    def publish_publication(
        self,
        publication_id: str,
        venue: str = "",
        doi: str | None = None,
    ) -> Publication | None:
        """Publish a publication."""
        return self._publications.update(
            publication_id,
            is_published=True,
            published_at=datetime.now(UTC),
            venue=venue,
            doi=doi,
        )

    def get_publications_for_project(self, project_id: str) -> list[Publication]:
        """Get publications for a project."""
        return self._publications.get_by_project(project_id)

    def get_published_publications(self) -> list[Publication]:
        """Get all published publications."""
        return self._publications.get_published()

    # --- Public API: Citations ---

    def add_citation(
        self,
        title: str,
        authors: list[str] | None = None,
        **kwargs: Any,
    ) -> Citation:
        """Add a citation/reference."""
        citation = Citation(
            title=title,
            authors=tuple(authors or []),
            year=kwargs.get("year"),
            venue=kwargs.get("venue", ""),
            doi=kwargs.get("doi"),
            url=kwargs.get("url"),
            citation_text=kwargs.get("citation_text", ""),
            tags=frozenset(kwargs.get("tags", [])),
        )
        self._citations.add(citation)
        return citation

    def get_citation(self, citation_id: str) -> Citation | None:
        """Get a citation by ID."""
        return self._citations.get(citation_id)

    def search_citations(self, query: str) -> list[Citation]:
        """Search citations."""
        return self._citations.search(query)

    # --- Public API: Learning ---

    def create_learning_cycle(
        self,
        title: str,
        learner_id: str | None = None,
        **kwargs: Any,
    ) -> LearningCycle:
        """Create a learning cycle."""
        cycle = LearningCycle(
            title=title,
            description=kwargs.get("description", ""),
            learner_id=learner_id,
            topics=tuple(kwargs.get("topics", [])),
            materials=tuple(kwargs.get("materials", [])),
            assessments=tuple(kwargs.get("assessments", [])),
            estimated_duration=kwargs.get("estimated_duration"),
            tags=frozenset(kwargs.get("tags", [])),
        )
        self._learning.create(cycle)
        return cycle

    def get_learning_cycle(self, cycle_id: str) -> LearningCycle | None:
        """Get a learning cycle by ID."""
        return self._learning.get(cycle_id)

    def update_learning_progress(
        self,
        cycle_id: str,
        progress: float,
        completed_topic: str | None = None,
    ) -> LearningCycle | None:
        """Update learning progress."""
        return self._learning.update_progress(cycle_id, progress, completed_topic)

    def get_learner_cycles(self, learner_id: str) -> list[LearningCycle]:
        """Get learning cycles for a learner."""
        return self._learning.get_by_learner(learner_id)

    # --- Public API: Archives ---

    def archive_publication(self, publication_id: str) -> AcademicArchive | None:
        """Archive a publication."""
        pub = self._publications.get(publication_id)
        if not pub:
            return None

        import hashlib
        content_hash = hashlib.sha256(pub.content.encode()).hexdigest()[:16]

        archive = AcademicArchive(
            title=pub.title,
            archive_type="publication",
            source_id=publication_id,
            content_hash=content_hash,
            is_public=pub.is_published,
            tags=pub.tags,
        )
        self._archives.archive(archive)
        return archive

    def archive_project(self, project_id: str) -> AcademicArchive | None:
        """Archive a completed project."""
        project = self._projects.get(project_id)
        if not project or project.status != ResearchStatus.COMPLETED:
            return None

        import hashlib
        content_hash = hashlib.sha256(project.title.encode()).hexdigest()[:16]

        archive = AcademicArchive(
            title=project.title,
            archive_type="project",
            source_id=project_id,
            content_hash=content_hash,
            tags=project.tags,
            metadata={"completion_date": project.completed_at.isoformat() if project.completed_at else None},
        )
        self._archives.archive(archive)
        return archive

    def get_archive(self, archive_id: str) -> AcademicArchive | None:
        """Get an archive by ID."""
        return self._archives.increment_access(archive_id)

    # --- Statistics ---

    def get_stats(self) -> AcademiaStats:
        """Get academia statistics."""
        active = len(self._projects.get_by_status(ResearchStatus.IN_PROGRESS))
        completed = len(self._projects.get_by_status(ResearchStatus.COMPLETED))
        published = len(self._publications.get_published())

        return AcademiaStats(
            total_projects=self._projects.project_count,
            active_projects=active,
            completed_projects=completed,
            total_publications=self._publications.publication_count,
            published_count=published,
            total_citations=self._citations.citation_count,
            total_learning_cycles=self._learning.cycle_count,
            total_archives=self._archives.archive_count,
        )

    def clear(self) -> tuple[int, int, int, int, int]:
        """Clear all data. Returns (projects, publications, citations, cycles, archives)."""
        projects = self._projects.project_count
        publications = self._publications.publication_count
        citations = self._citations.citation_count
        cycles = self._learning.cycle_count
        archives = self._archives.archive_count

        self._projects = ProjectRegistry()
        self._publications = PublicationRegistry()
        self._citations = CitationManager()
        self._learning = LearningManager()
        self._archives = ArchiveManager()

        return projects, publications, citations, cycles, archives
