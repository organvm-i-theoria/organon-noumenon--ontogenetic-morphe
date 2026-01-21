#!/usr/bin/env python3
"""
Example: Academic Research Lifecycle

Demonstrates:
- AcademiaManager managing research projects
- Publications and citations
- Learning cycles
- Archive management

This shows the academic subsystem:
PROJECT -> PUBLICATION -> CITATION -> ARCHIVE
"""

from datetime import datetime, UTC, timedelta

from autogenrec.subsystems.academic.academia_manager import (
    AcademiaManager,
    PublicationType,
    ResearchStatus,
)


def main():
    print("=" * 60)
    print("Academic Research Lifecycle Example")
    print("=" * 60)
    print()

    # Initialize subsystem
    academia = AcademiaManager()

    # =========================================================================
    # Step 1: Create a research project
    # =========================================================================
    print("Step 1: Creating Research Project")
    print("-" * 40)

    project = academia.create_project(
        title="Symbolic Processing in Recursive Systems",
        description="Investigating emergent patterns in recursive symbolic architectures",
        methodology="Mixed methods: theoretical modeling + empirical simulation",
        hypothesis="Recursive feedback enhances pattern recognition accuracy",
        objectives=[
            "Develop theoretical framework",
            "Build prototype system",
            "Validate through experiments",
            "Publish findings",
        ],
        lead_researcher_id="researcher_001",
        keywords=["symbolic", "recursion", "patterns"],
    )

    print(f"  Project: {project.title}")
    print(f"  Status: {project.status.name}")
    print(f"  Objectives: {len(project.objectives)}")
    print()

    # =========================================================================
    # Step 2: Progress through research phases
    # =========================================================================
    print("Step 2: Research Progress")
    print("-" * 40)

    # Start the project
    academia.start_project(project.id)
    print(f"  [Started] Status: IN_PROGRESS")

    # Update progress
    milestones = [
        (25, "Completed literature review"),
        (50, "Framework design complete"),
        (75, "Prototype implemented"),
        (100, "Experiments concluded"),
    ]

    for progress, milestone in milestones:
        academia.update_project_progress(project.id, progress)
        print(f"  [{progress}%] {milestone}")

    # Complete the project
    completed = academia.complete_project(project.id)
    print(f"  [Complete] Status: {completed.status.name}")
    print()

    # =========================================================================
    # Step 3: Create publications
    # =========================================================================
    print("Step 3: Creating Publications")
    print("-" * 40)

    # Main paper
    paper = academia.create_publication(
        title="RecursiQ: Emergent Patterns in Recursive Symbolic Systems",
        publication_type=PublicationType.PAPER,
        abstract="This paper presents RecursiQ, a framework for discovering "
                 "emergent patterns through recursive symbolic processing...",
        content="[Full paper content]",
        author_ids=["researcher_001", "researcher_002"],
        project_id=project.id,
        keywords=["symbolic processing", "recursion", "emergence"],
    )
    print(f"  Created: {paper.title}")
    print(f"  Type: {paper.publication_type.name}")

    # Technical report
    report = academia.create_publication(
        title="RecursiQ Technical Report: Implementation Details",
        publication_type=PublicationType.REPORT,
        abstract="Technical implementation details for the RecursiQ framework",
        content="[Technical report content]",
        author_ids=["researcher_001"],
        project_id=project.id,
    )
    print(f"  Created: {report.title}")
    print(f"  Type: {report.publication_type.name}")
    print()

    # =========================================================================
    # Step 4: Publish and add citations
    # =========================================================================
    print("Step 4: Publishing and Citations")
    print("-" * 40)

    # Publish the main paper
    published = academia.publish_publication(
        paper.id,
        venue="International Conference on Symbolic Computing",
        doi="10.1234/icsc.2024.recursiq",
    )
    print(f"  Published: {published.title}")
    print(f"  Venue: {published.venue}")
    print(f"  DOI: {published.doi}")

    # Add citations (references used in the paper)
    citations = [
        ("Foundations of Symbolic AI", ["Smith, J."], 2020, "AI Journal"),
        ("Recursive Systems Theory", ["Johnson, K.", "Williams, R."], 2022, "ICML"),
        ("Pattern Recognition Methods", ["Brown, A."], 2023, "NeurIPS"),
    ]

    print(f"\n  Adding citations:")
    for title, authors, year, venue in citations:
        citation = academia.add_citation(
            title=title,
            authors=authors,
            year=year,
            venue=venue,
        )
        print(f"    - {title} ({year})")

    print()

    # =========================================================================
    # Step 5: Create learning cycles
    # =========================================================================
    print("Step 5: Creating Learning Cycles")
    print("-" * 40)

    # Create a learning cycle for someone studying the research
    cycle = academia.create_learning_cycle(
        title="Understanding RecursiQ",
        learner_id="student_001",
        description="Learn the fundamentals of recursive symbolic processing",
        topics=[
            "Basic symbolic processing",
            "Recursion fundamentals",
            "Pattern recognition",
            "RecursiQ framework",
            "Practical applications",
        ],
    )
    print(f"  Learning Cycle: {cycle.title}")
    print(f"  Topics: {len(cycle.topics)}")
    print(f"  Status: {cycle.status.name}")

    # Progress through learning
    for i, topic in enumerate(cycle.topics, 1):
        progress = (i / len(cycle.topics)) * 100
        updated = academia.update_learning_progress(
            cycle.id,
            progress,
            completed_topic=topic,
        )
        print(f"    [{progress:.0f}%] Completed: {topic}")

    print(f"  Final Status: {updated.status.name}")
    print()

    # =========================================================================
    # Step 6: Archive completed work
    # =========================================================================
    print("Step 6: Archiving")
    print("-" * 40)

    # Archive the publication
    pub_archive = academia.archive_publication(paper.id)
    print(f"  Archived: {paper.title}")
    print(f"    Archive ID: {pub_archive.id[:16]}...")
    print(f"    Type: {pub_archive.archive_type}")

    # Archive the project
    proj_archive = academia.archive_project(project.id)
    print(f"  Archived: {project.title}")
    print(f"    Archive ID: {proj_archive.id[:16]}...")
    print(f"    Type: {proj_archive.archive_type}")

    # Access archived item (increments access count)
    retrieved = academia.get_archive(pub_archive.id)
    retrieved = academia.get_archive(pub_archive.id)
    print(f"\n  Archive Access Count: {retrieved.access_count}")
    print()

    # =========================================================================
    # Step 7: Statistics
    # =========================================================================
    print("Step 7: Academia Statistics")
    print("-" * 40)

    stats = academia.get_stats()

    print(f"  Projects:")
    print(f"    Total: {stats.total_projects}")
    print(f"    Completed: {stats.completed_projects}")

    print(f"  Publications:")
    print(f"    Total: {stats.total_publications}")
    print(f"    Published: {stats.published_count}")

    print(f"  Citations: {stats.total_citations}")
    print(f"  Learning Cycles: {stats.total_learning_cycles}")
    print(f"  Archives: {stats.total_archives}")

    print()
    print("=" * 60)
    print("Example Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
