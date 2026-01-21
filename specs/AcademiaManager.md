---
title: AcademiaManager
system: Recursive–Generative Organizational Body
type: subsystem
category: academic
tags: [academia, research, learning, publication]
dependencies: [ReferenceManager, ArchiveManager]
---

# AcademiaManager

The **AcademiaManager** manages academic processes including research projects, learning cycles, publications, citations, and scholarly knowledge dissemination.

## Overview

| Property | Value |
|----------|-------|
| Category | Academic |
| Module | `autogenrec.subsystems.academic.academia_manager` |
| Dependencies | ReferenceManager, ArchiveManager |

## Domain Models

### Enums

```python
class ResearchStatus(Enum):
    PROPOSED = auto()      # Initial proposal
    APPROVED = auto()      # Approved for execution
    IN_PROGRESS = auto()   # Currently active
    REVIEW = auto()        # Under review
    COMPLETED = auto()     # Successfully finished
    ARCHIVED = auto()      # Moved to archive
    REJECTED = auto()      # Proposal rejected

class PublicationType(Enum):
    PAPER = auto()         # Academic paper
    ARTICLE = auto()       # Journal article
    THESIS = auto()        # Thesis/dissertation
    REPORT = auto()        # Technical report
    DATASET = auto()       # Published dataset
    PRESENTATION = auto()  # Conference presentation
    NOTE = auto()          # Research note

class LearningStatus(Enum):
    NOT_STARTED = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    PAUSED = auto()
    ABANDONED = auto()
```

### Core Models

- **ResearchProject**: Research project with title, methodology, objectives, progress tracking
- **Publication**: Academic publication with type, abstract, authors, venue, DOI
- **Citation**: Reference to external work with authors, year, venue
- **LearningCycle**: Structured learning path with topics and progress
- **Archive**: Archived academic artifact with access tracking

## Process Loop

1. **Intake**: Receive research proposals, publications, learning materials, or archive requests
2. **Process**: Execute research workflows, compile publications, track learning progress
3. **Evaluate**: Assess research quality, validate publications, measure learning outcomes
4. **Integrate**: Archive completed work, update references, emit completion events

## Public API

### Project Management

```python
# Create a new research project
project = academia.create_project(
    title="Pattern Recognition in Symbolic Systems",
    description="Investigating emergent patterns...",
    methodology="Mixed methods",
    objectives=["Develop framework", "Validate through experiments"],
    lead_researcher_id="researcher_001",
    keywords=["symbolic", "patterns"],
)

# Progress through project lifecycle
academia.start_project(project.id)
academia.update_project_progress(project.id, 50.0)  # 50% complete
completed = academia.complete_project(project.id)
```

### Publications

```python
# Create and publish
paper = academia.create_publication(
    title="RecursiQ: A Framework for Symbolic Processing",
    publication_type=PublicationType.PAPER,
    abstract="This paper presents...",
    content="[Full paper content]",
    author_ids=["researcher_001", "researcher_002"],
    project_id=project.id,
)

published = academia.publish(
    paper.id,
    venue="International Conference on Symbolic Computing",
    doi="10.1234/icsc.2024.001",
)
```

### Citations

```python
citation = academia.add_citation(
    title="Foundations of Symbolic AI",
    authors=["Smith, J.", "Johnson, K."],
    year=2023,
    venue="AI Journal",
)
```

### Learning Cycles

```python
cycle = academia.create_learning_cycle(
    title="Understanding Recursive Systems",
    learner_id="student_001",
    topics=["Basics", "Advanced patterns", "Applications"],
)

academia.update_learning_progress(
    cycle.id,
    progress=50.0,
    completed_topic="Basics",
)
```

### Archiving

```python
archive = academia.archive_publication(paper.id)
retrieved = academia.get_archive(archive.id)  # Increments access count
```

### Statistics

```python
stats = academia.get_stats()
# AcademiaStats with:
#   total_projects, completed_projects
#   total_publications, published_count
#   total_citations, total_learning_cycles
#   total_archives
```

## Integration

The AcademiaManager integrates with:
- **ReferenceManager**: For citation resolution and canonical references
- **ArchiveManager**: For long-term preservation of research artifacts

## Example

See `examples/conflict_resolution_demo.py` for a complete academic research lifecycle demonstration.
