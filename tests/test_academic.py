"""Tests for Week 9: Academic subsystem."""

from datetime import timedelta

import pytest

from autogenrec.subsystems.academic.academia_manager import (
    AcademiaManager,
    ResearchStatus,
    PublicationType,
    LearningStatus,
)


# ============================================================================
# AcademiaManager Tests
# ============================================================================

class TestAcademiaManagerBasics:
    """Basic AcademiaManager tests."""

    def test_initialization(self):
        """Test AcademiaManager initializes correctly."""
        am = AcademiaManager()
        assert am.name == "academia_manager"
        assert am.project_count == 0
        assert am.publication_count == 0
        assert am.citation_count == 0
        assert am.learning_cycle_count == 0
        assert am.archive_count == 0


class TestResearchProjects:
    """Test research project management."""

    def test_create_project(self):
        """Test creating a research project."""
        am = AcademiaManager()
        project = am.create_project(
            title="Test Research",
            description="A test research project",
            methodology="Qualitative analysis",
            hypothesis="Testing improves quality",
            objectives=["Objective 1", "Objective 2"],
            keywords=["testing", "research"],
            lead_researcher_id="researcher_1",
        )
        assert project is not None
        assert project.title == "Test Research"
        assert project.status == ResearchStatus.PROPOSED
        assert len(project.objectives) == 2
        assert am.project_count == 1

    def test_get_project(self):
        """Test getting a project by ID."""
        am = AcademiaManager()
        project = am.create_project(title="Test")
        retrieved = am.get_project(project.id)
        assert retrieved is not None
        assert retrieved.id == project.id

    def test_start_project(self):
        """Test starting a project."""
        am = AcademiaManager()
        project = am.create_project(title="Test")
        started = am.start_project(project.id)
        assert started is not None
        assert started.status == ResearchStatus.IN_PROGRESS
        assert started.started_at is not None

    def test_complete_project(self):
        """Test completing a project."""
        am = AcademiaManager()
        project = am.create_project(title="Test")
        am.start_project(project.id)
        completed = am.complete_project(project.id)
        assert completed is not None
        assert completed.status == ResearchStatus.COMPLETED
        assert completed.completed_at is not None
        assert completed.progress_percent == 100.0

    def test_update_project_progress(self):
        """Test updating project progress."""
        am = AcademiaManager()
        project = am.create_project(title="Test")
        am.start_project(project.id)
        updated = am.update_project_progress(project.id, 50.0)
        assert updated is not None
        assert updated.progress_percent == 50.0

    def test_get_active_projects(self):
        """Test getting active projects."""
        am = AcademiaManager()
        p1 = am.create_project(title="Active 1")
        p2 = am.create_project(title="Active 2")
        p3 = am.create_project(title="Not started")
        
        am.start_project(p1.id)
        am.start_project(p2.id)
        
        active = am.get_active_projects()
        assert len(active) == 2

    def test_project_with_deadline(self):
        """Test project with deadline."""
        am = AcademiaManager()
        from datetime import datetime, UTC, timedelta
        deadline = datetime.now(UTC) + timedelta(days=30)
        
        project = am.create_project(
            title="Deadline Project",
            deadline=deadline,
        )
        assert project.deadline == deadline


class TestPublications:
    """Test publication management."""

    def test_create_publication(self):
        """Test creating a publication."""
        am = AcademiaManager()
        pub = am.create_publication(
            title="Test Paper",
            publication_type=PublicationType.PAPER,
            abstract="This is a test abstract",
            author_ids=["author_1", "author_2"],
            keywords=["test", "paper"],
        )
        assert pub is not None
        assert pub.title == "Test Paper"
        assert pub.publication_type == PublicationType.PAPER
        assert pub.is_published is False
        assert am.publication_count == 1

    def test_get_publication(self):
        """Test getting a publication by ID."""
        am = AcademiaManager()
        pub = am.create_publication(title="Test")
        retrieved = am.get_publication(pub.id)
        assert retrieved is not None
        assert retrieved.id == pub.id

    def test_publish(self):
        """Test publishing a publication."""
        am = AcademiaManager()
        pub = am.create_publication(title="Test Paper")
        published = am.publish_publication(pub.id, venue="Test Journal", doi="10.1234/test")
        assert published is not None
        assert published.is_published is True
        assert published.venue == "Test Journal"
        assert published.doi == "10.1234/test"
        assert published.published_at is not None

    def test_different_publication_types(self):
        """Test different publication types."""
        am = AcademiaManager()
        
        paper = am.create_publication("Paper", PublicationType.PAPER)
        thesis = am.create_publication("Thesis", PublicationType.THESIS)
        report = am.create_publication("Report", PublicationType.REPORT)
        
        assert paper.publication_type == PublicationType.PAPER
        assert thesis.publication_type == PublicationType.THESIS
        assert report.publication_type == PublicationType.REPORT

    def test_publication_linked_to_project(self):
        """Test publication linked to project."""
        am = AcademiaManager()
        project = am.create_project(title="Research Project")
        pub = am.create_publication(
            title="Project Paper",
            project_id=project.id,
        )
        
        pubs = am.get_publications_for_project(project.id)
        assert len(pubs) == 1
        assert pubs[0].id == pub.id

    def test_get_published_publications(self):
        """Test getting published publications."""
        am = AcademiaManager()
        p1 = am.create_publication(title="Published")
        p2 = am.create_publication(title="Draft")
        
        am.publish_publication(p1.id)
        
        published = am.get_published_publications()
        assert len(published) == 1
        assert published[0].id == p1.id


class TestCitations:
    """Test citation management."""

    def test_add_citation(self):
        """Test adding a citation."""
        am = AcademiaManager()
        citation = am.add_citation(
            title="Referenced Paper",
            authors=["Author A", "Author B"],
            year=2024,
            venue="Test Conference",
            doi="10.1234/ref",
        )
        assert citation is not None
        assert citation.title == "Referenced Paper"
        assert len(citation.authors) == 2
        assert am.citation_count == 1

    def test_get_citation(self):
        """Test getting a citation by ID."""
        am = AcademiaManager()
        citation = am.add_citation(title="Test Citation")
        retrieved = am.get_citation(citation.id)
        assert retrieved is not None
        assert retrieved.id == citation.id

    def test_search_citations_by_title(self):
        """Test searching citations by title."""
        am = AcademiaManager()
        am.add_citation(title="Machine Learning Basics")
        am.add_citation(title="Deep Learning Advances")
        am.add_citation(title="Web Development Guide")
        
        results = am.search_citations("Learning")
        assert len(results) == 2

    def test_search_citations_by_author(self):
        """Test searching citations by author."""
        am = AcademiaManager()
        am.add_citation(title="Paper 1", authors=["John Smith"])
        am.add_citation(title="Paper 2", authors=["Jane Doe"])
        am.add_citation(title="Paper 3", authors=["John Doe"])
        
        results = am.search_citations("John")
        assert len(results) == 2


class TestLearningCycles:
    """Test learning cycle management."""

    def test_create_learning_cycle(self):
        """Test creating a learning cycle."""
        am = AcademiaManager()
        cycle = am.create_learning_cycle(
            title="Python Basics",
            learner_id="learner_1",
            description="Introduction to Python",
            topics=["Variables", "Functions", "Classes"],
        )
        assert cycle is not None
        assert cycle.title == "Python Basics"
        assert cycle.status == LearningStatus.NOT_STARTED
        assert len(cycle.topics) == 3
        assert am.learning_cycle_count == 1

    def test_get_learning_cycle(self):
        """Test getting a learning cycle by ID."""
        am = AcademiaManager()
        cycle = am.create_learning_cycle(title="Test")
        retrieved = am.get_learning_cycle(cycle.id)
        assert retrieved is not None
        assert retrieved.id == cycle.id

    def test_update_learning_progress(self):
        """Test updating learning progress."""
        am = AcademiaManager()
        cycle = am.create_learning_cycle(
            title="Test Course",
            topics=["Topic 1", "Topic 2"],
        )
        
        updated = am.update_learning_progress(cycle.id, 50.0, "Topic 1")
        assert updated is not None
        assert updated.progress_percent == 50.0
        assert updated.status == LearningStatus.IN_PROGRESS
        assert "Topic 1" in updated.completed_topics

    def test_complete_learning_cycle(self):
        """Test completing a learning cycle."""
        am = AcademiaManager()
        cycle = am.create_learning_cycle(title="Test")
        
        completed = am.update_learning_progress(cycle.id, 100.0)
        assert completed is not None
        assert completed.status == LearningStatus.COMPLETED
        assert completed.completed_at is not None

    def test_get_learner_cycles(self):
        """Test getting cycles for a learner."""
        am = AcademiaManager()
        am.create_learning_cycle(title="Course 1", learner_id="learner_1")
        am.create_learning_cycle(title="Course 2", learner_id="learner_1")
        am.create_learning_cycle(title="Course 3", learner_id="learner_2")
        
        cycles = am.get_learner_cycles("learner_1")
        assert len(cycles) == 2


class TestArchives:
    """Test archive management."""

    def test_archive_publication(self):
        """Test archiving a publication."""
        am = AcademiaManager()
        pub = am.create_publication(title="Paper to Archive", content="Some content")
        
        archive = am.archive_publication(pub.id)
        assert archive is not None
        assert archive.source_id == pub.id
        assert archive.archive_type == "publication"
        assert am.archive_count == 1

    def test_archive_completed_project(self):
        """Test archiving a completed project."""
        am = AcademiaManager()
        project = am.create_project(title="Project to Archive")
        am.start_project(project.id)
        am.complete_project(project.id)
        
        archive = am.archive_project(project.id)
        assert archive is not None
        assert archive.source_id == project.id
        assert archive.archive_type == "project"

    def test_cannot_archive_incomplete_project(self):
        """Test that incomplete projects cannot be archived."""
        am = AcademiaManager()
        project = am.create_project(title="Incomplete Project")
        
        archive = am.archive_project(project.id)
        assert archive is None

    def test_get_archive_increments_access(self):
        """Test that getting an archive increments access count."""
        am = AcademiaManager()
        pub = am.create_publication(title="Test")
        archive = am.archive_publication(pub.id)
        
        # First access
        retrieved1 = am.get_archive(archive.id)
        assert retrieved1.access_count == 1
        
        # Second access
        retrieved2 = am.get_archive(archive.id)
        assert retrieved2.access_count == 2


class TestAcademiaStats:
    """Test academia statistics."""

    def test_get_stats(self):
        """Test getting academia statistics."""
        am = AcademiaManager()
        
        # Create some data
        p1 = am.create_project(title="Project 1")
        am.create_project(title="Project 2")
        am.start_project(p1.id)
        am.complete_project(p1.id)
        
        pub = am.create_publication(title="Pub 1")
        am.create_publication(title="Pub 2")
        am.publish_publication(pub.id)
        
        am.add_citation(title="Citation 1")
        am.create_learning_cycle(title="Course 1")
        am.archive_publication(pub.id)
        
        stats = am.get_stats()
        assert stats.total_projects == 2
        assert stats.completed_projects == 1
        assert stats.total_publications == 2
        assert stats.published_count == 1
        assert stats.total_citations == 1
        assert stats.total_learning_cycles == 1
        assert stats.total_archives == 1

    def test_clear(self):
        """Test clearing all data."""
        am = AcademiaManager()
        
        am.create_project(title="Project")
        am.create_publication(title="Pub")
        am.add_citation(title="Citation")
        am.create_learning_cycle(title="Course")
        pub = am.create_publication(title="Archive this")
        am.archive_publication(pub.id)
        
        projects, pubs, citations, cycles, archives = am.clear()
        assert projects == 1
        assert pubs == 2
        assert citations == 1
        assert cycles == 1
        assert archives == 1
        assert am.project_count == 0
