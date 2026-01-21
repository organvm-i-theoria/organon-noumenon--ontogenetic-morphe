"""Tests for temporal subsystems: TimeManager, EvolutionScheduler, LocationResolver."""

from datetime import UTC, datetime, timedelta

import pytest

from autogenrec.subsystems.temporal.time_manager import (
    Cycle,
    CycleType,
    EventStatus,
    TimeEvent,
    TimeManager,
)
from autogenrec.subsystems.temporal.evolution_scheduler import (
    EvolutionScheduler,
    GrowthPhase,
    MutationType,
    StabilityLevel,
)
from autogenrec.subsystems.temporal.location_resolver import (
    LocationResolver,
    Place,
    PlaceType,
    ResolutionStatus,
    SpatialRelation,
)


# ============================================================================
# TimeManager Tests
# ============================================================================


class TestTimeManager:
    """Tests for TimeManager subsystem."""

    @pytest.fixture
    def time_manager(self) -> TimeManager:
        return TimeManager()

    def test_initialization(self, time_manager: TimeManager) -> None:
        """Test TimeManager initializes correctly."""
        assert time_manager.name == "time_manager"
        assert time_manager.event_count == 0
        assert time_manager.cycle_count == 0

    def test_schedule_event(self, time_manager: TimeManager) -> None:
        """Test scheduling a one-time event."""
        scheduled_at = datetime.now(UTC) + timedelta(hours=1)
        event = time_manager.schedule_event(
            name="Test Event",
            scheduled_at=scheduled_at,
            description="A test event",
        )

        assert event.name == "Test Event"
        assert event.scheduled_at == scheduled_at
        assert event.status == EventStatus.PENDING
        assert event.cycle_type == CycleType.ONE_TIME
        assert time_manager.event_count == 1

    def test_schedule_recurring_event(self, time_manager: TimeManager) -> None:
        """Test scheduling a recurring event."""
        event = time_manager.schedule_recurring(
            name="Recurring Event",
            interval_seconds=3600,
            max_occurrences=5,
        )

        assert event.name == "Recurring Event"
        assert event.cycle_type == CycleType.RECURRING
        assert event.interval == timedelta(seconds=3600)
        assert event.max_occurrences == 5

    def test_get_event(self, time_manager: TimeManager) -> None:
        """Test retrieving an event by ID."""
        event = time_manager.schedule_event(
            name="Retrieve Test",
            scheduled_at=datetime.now(UTC),
        )

        retrieved = time_manager.get_event(event.id)
        assert retrieved is not None
        assert retrieved.id == event.id
        assert retrieved.name == "Retrieve Test"

    def test_get_due_events(self, time_manager: TimeManager) -> None:
        """Test getting due events."""
        # Schedule past event (due)
        past_event = time_manager.schedule_event(
            name="Past Event",
            scheduled_at=datetime.now(UTC) - timedelta(minutes=5),
        )

        # Schedule future event (not due)
        time_manager.schedule_event(
            name="Future Event",
            scheduled_at=datetime.now(UTC) + timedelta(hours=1),
        )

        due = time_manager.get_due_events()
        assert len(due) == 1
        assert due[0].id == past_event.id

    @pytest.mark.asyncio
    async def test_trigger_due_events(self, time_manager: TimeManager) -> None:
        """Test triggering due events."""
        event = time_manager.schedule_event(
            name="Trigger Test",
            scheduled_at=datetime.now(UTC) - timedelta(minutes=1),
        )

        results = await time_manager.trigger_due_events()
        assert len(results) == 1
        assert results[0].event_id == event.id
        assert results[0].success is True

        # Event should be executed
        updated = time_manager.get_event(event.id)
        assert updated is not None
        assert updated.status == EventStatus.EXECUTED

    def test_cancel_event(self, time_manager: TimeManager) -> None:
        """Test cancelling an event."""
        event = time_manager.schedule_event(
            name="Cancel Test",
            scheduled_at=datetime.now(UTC) + timedelta(hours=1),
        )

        result = time_manager.cancel_event(event.id)
        assert result is True

        cancelled = time_manager.get_event(event.id)
        assert cancelled is not None
        assert cancelled.status == EventStatus.CANCELLED

    def test_create_cycle(self, time_manager: TimeManager) -> None:
        """Test creating a cycle."""
        cycle = time_manager.create_cycle(
            name="Daily Cycle",
            period_seconds=86400,
            total_phases=4,
        )

        assert cycle.name == "Daily Cycle"
        assert cycle.period == timedelta(seconds=86400)
        assert cycle.total_phases == 4
        assert cycle.current_phase == 0
        assert time_manager.cycle_count == 1

    def test_start_and_advance_cycle(self, time_manager: TimeManager) -> None:
        """Test starting and advancing a cycle."""
        cycle = time_manager.create_cycle(
            name="Test Cycle",
            period_seconds=3600,
            total_phases=3,
        )

        # Start cycle
        started = time_manager.start_cycle(cycle.id)
        assert started is not None
        assert started.started_at is not None
        assert started.current_phase == 0

        # Advance cycle
        advanced = time_manager.advance_cycle(cycle.id)
        assert advanced is not None
        assert advanced.current_phase == 1
        assert advanced.iteration == 0

        # Advance to wrap around
        time_manager.advance_cycle(cycle.id)
        wrapped = time_manager.advance_cycle(cycle.id)
        assert wrapped is not None
        assert wrapped.current_phase == 0
        assert wrapped.iteration == 1

    def test_get_active_cycles(self, time_manager: TimeManager) -> None:
        """Test getting active cycles."""
        cycle1 = time_manager.create_cycle(name="Active", period_seconds=3600)
        time_manager.create_cycle(name="Inactive", period_seconds=3600)

        time_manager.start_cycle(cycle1.id)

        active = time_manager.get_active_cycles()
        assert len(active) == 1
        assert active[0].name == "Active"

    def test_get_stats(self, time_manager: TimeManager) -> None:
        """Test getting statistics."""
        time_manager.schedule_event(
            name="Event 1",
            scheduled_at=datetime.now(UTC) + timedelta(hours=1),
        )
        time_manager.schedule_event(
            name="Event 2",
            scheduled_at=datetime.now(UTC) + timedelta(hours=2),
        )
        cycle = time_manager.create_cycle(name="Cycle 1", period_seconds=3600)
        time_manager.start_cycle(cycle.id)

        stats = time_manager.get_stats()
        assert stats.total_events == 2
        assert stats.pending_events == 2
        assert stats.total_cycles == 1
        assert stats.active_cycles == 1

    def test_clear(self, time_manager: TimeManager) -> None:
        """Test clearing all data."""
        time_manager.schedule_event(
            name="Event",
            scheduled_at=datetime.now(UTC),
        )
        time_manager.create_cycle(name="Cycle", period_seconds=3600)

        events, cycles = time_manager.clear()
        assert events == 1
        assert cycles == 1
        assert time_manager.event_count == 0
        assert time_manager.cycle_count == 0


# ============================================================================
# EvolutionScheduler Tests
# ============================================================================


class TestEvolutionScheduler:
    """Tests for EvolutionScheduler subsystem."""

    @pytest.fixture
    def scheduler(self) -> EvolutionScheduler:
        return EvolutionScheduler(seed=42)  # Fixed seed for reproducibility

    def test_initialization(self, scheduler: EvolutionScheduler) -> None:
        """Test EvolutionScheduler initializes correctly."""
        assert scheduler.name == "evolution_scheduler"
        assert scheduler.pattern_count == 0
        assert scheduler.cycle_count == 0

    def test_create_pattern(self, scheduler: EvolutionScheduler) -> None:
        """Test creating a growth pattern."""
        pattern = scheduler.create_pattern(
            name="Test Pattern",
            content={"key": "value"},
            description="A test pattern",
        )

        assert pattern.name == "Test Pattern"
        assert pattern.phase == GrowthPhase.DORMANT
        assert pattern.content == {"key": "value"}
        assert scheduler.pattern_count == 1

    def test_get_pattern(self, scheduler: EvolutionScheduler) -> None:
        """Test retrieving a pattern by ID."""
        pattern = scheduler.create_pattern(
            name="Retrieve Test",
            content={"data": 1},
        )

        retrieved = scheduler.get_pattern(pattern.id)
        assert retrieved is not None
        assert retrieved.id == pattern.id

    def test_advance_phase(self, scheduler: EvolutionScheduler) -> None:
        """Test advancing growth phase."""
        pattern = scheduler.create_pattern(
            name="Phase Test",
            content={"data": 1},
        )

        # Advance through phases
        advanced = scheduler.advance_phase(pattern.id)
        assert advanced is not None
        assert advanced.phase == GrowthPhase.GERMINATION

        advanced = scheduler.advance_phase(pattern.id)
        assert advanced.phase == GrowthPhase.GROWTH

        advanced = scheduler.advance_phase(pattern.id)
        assert advanced.phase == GrowthPhase.BLOOM

    def test_mutate_pattern(self, scheduler: EvolutionScheduler) -> None:
        """Test mutating a pattern."""
        pattern = scheduler.create_pattern(
            name="Mutation Test",
            content={"speed": 5},
        )

        evolved, mutation = scheduler.mutate_pattern(
            pattern_id=pattern.id,
            mutation_type=MutationType.ADDITION,
        )

        assert evolved is not None
        assert mutation is not None
        assert mutation.mutation_type == MutationType.ADDITION
        assert evolved.generation == 1
        assert evolved.parent_id == pattern.id

    def test_mutate_pattern_different_types(self, scheduler: EvolutionScheduler) -> None:
        """Test different mutation types."""
        mutation_types = [
            MutationType.ADDITION,
            MutationType.REMOVAL,
            MutationType.MODIFICATION,
            MutationType.DUPLICATION,
        ]

        for mt in mutation_types:
            pattern = scheduler.create_pattern(
                name=f"Test {mt.name}",
                content={"key": "value"},
            )
            evolved, mutation = scheduler.mutate_pattern(pattern.id, mt)
            assert mutation is not None
            assert mutation.mutation_type == mt

    def test_get_patterns_by_phase(self, scheduler: EvolutionScheduler) -> None:
        """Test getting patterns by phase."""
        pattern1 = scheduler.create_pattern(name="Pattern 1", content={})
        pattern2 = scheduler.create_pattern(name="Pattern 2", content={})
        scheduler.create_pattern(name="Pattern 3", content={})

        # Advance pattern1 and pattern2 to GERMINATION
        scheduler.advance_phase(pattern1.id)
        scheduler.advance_phase(pattern2.id)

        germinating = scheduler.get_patterns_by_phase(GrowthPhase.GERMINATION)
        assert len(germinating) == 2

        dormant = scheduler.get_patterns_by_phase(GrowthPhase.DORMANT)
        assert len(dormant) == 1

    def test_full_lifecycle(self, scheduler: EvolutionScheduler) -> None:
        """Test full lifecycle through all phases."""
        pattern = scheduler.create_pattern(
            name="Lifecycle Test",
            content={},
        )

        phases = [
            GrowthPhase.GERMINATION,
            GrowthPhase.GROWTH,
            GrowthPhase.BLOOM,
            GrowthPhase.HARVEST,
            GrowthPhase.DECAY,
            GrowthPhase.RENEWAL,
            GrowthPhase.DORMANT,  # Cycles back
        ]

        for expected_phase in phases:
            advanced = scheduler.advance_phase(pattern.id)
            assert advanced is not None
            assert advanced.phase == expected_phase

    def test_create_cycle(self, scheduler: EvolutionScheduler) -> None:
        """Test creating an evolution cycle."""
        pattern = scheduler.create_pattern(name="Pattern", content={})
        cycle = scheduler.create_cycle(
            name="Test Cycle",
            pattern_ids=[pattern.id],
        )

        assert cycle.name == "Test Cycle"
        assert pattern.id in cycle.pattern_ids
        assert scheduler.cycle_count == 1

    def test_start_cycle(self, scheduler: EvolutionScheduler) -> None:
        """Test starting an evolution cycle."""
        pattern = scheduler.create_pattern(name="Pattern", content={})
        cycle = scheduler.create_cycle(
            name="Test Cycle",
            pattern_ids=[pattern.id],
        )

        started = scheduler.start_cycle(cycle.id)
        assert started is not None
        assert started.started_at is not None
        assert started.current_phase == GrowthPhase.GERMINATION

    def test_get_viable_patterns(self, scheduler: EvolutionScheduler) -> None:
        """Test getting viable patterns."""
        # All new patterns start with default fitness (0.5) and are viable
        scheduler.create_pattern(name="Viable 1", content={})
        scheduler.create_pattern(name="Viable 2", content={})

        viable = scheduler.get_viable_patterns()
        assert len(viable) == 2

    def test_get_stats(self, scheduler: EvolutionScheduler) -> None:
        """Test getting statistics."""
        pattern1 = scheduler.create_pattern(name="Pattern 1", content={})
        scheduler.create_pattern(name="Pattern 2", content={})

        scheduler.advance_phase(pattern1.id)
        scheduler.mutate_pattern(pattern1.id, MutationType.ADDITION)

        stats = scheduler.get_stats()
        assert stats.total_patterns == 3  # Original 2 + 1 evolved
        assert stats.total_mutations == 1
        assert GrowthPhase.DORMANT.name in stats.by_phase
        assert GrowthPhase.GERMINATION.name in stats.by_phase

    def test_clear(self, scheduler: EvolutionScheduler) -> None:
        """Test clearing all data."""
        pattern = scheduler.create_pattern(name="Pattern", content={})
        scheduler.mutate_pattern(pattern.id, MutationType.ADDITION)
        scheduler.create_cycle(name="Cycle", pattern_ids=[pattern.id])

        patterns, cycles, mutations = scheduler.clear()
        assert patterns == 2  # Original + evolved
        assert cycles == 1
        assert mutations == 1
        assert scheduler.pattern_count == 0
        assert scheduler.cycle_count == 0

    def test_stability_levels(self, scheduler: EvolutionScheduler) -> None:
        """Test different stability levels."""
        pattern = scheduler.create_pattern(
            name="Stable Pattern",
            content={"data": 1},
            stability=StabilityLevel.ROBUST,
        )

        assert pattern.stability == StabilityLevel.ROBUST


# ============================================================================
# LocationResolver Tests
# ============================================================================


class TestLocationResolver:
    """Tests for LocationResolver subsystem."""

    @pytest.fixture
    def resolver(self) -> LocationResolver:
        return LocationResolver()

    def test_initialization(self, resolver: LocationResolver) -> None:
        """Test LocationResolver initializes correctly."""
        assert resolver.name == "location_resolver"
        assert resolver.place_count == 0
        assert resolver.link_count == 0
        assert resolver.assignment_count == 0

    def test_create_place(self, resolver: LocationResolver) -> None:
        """Test creating a place."""
        place = resolver.create_place(
            name="Test Place",
            place_type=PlaceType.PHYSICAL,
            description="A test location",
            aliases=["TP", "Test"],
        )

        assert place.name == "Test Place"
        assert place.place_type == PlaceType.PHYSICAL
        assert place.description == "A test location"
        assert "tp" in [a.lower() for a in place.aliases]
        assert resolver.place_count == 1

    def test_get_place(self, resolver: LocationResolver) -> None:
        """Test retrieving a place by ID."""
        place = resolver.create_place(name="Retrieve Test")

        retrieved = resolver.get_place(place.id)
        assert retrieved is not None
        assert retrieved.id == place.id

    def test_find_by_name(self, resolver: LocationResolver) -> None:
        """Test finding places by name."""
        resolver.create_place(name="Paris", place_type=PlaceType.PHYSICAL)
        resolver.create_place(name="London", place_type=PlaceType.PHYSICAL)

        results = resolver.find_by_name("Paris")
        assert len(results) == 1
        assert results[0].name == "Paris"

    def test_find_by_alias(self, resolver: LocationResolver) -> None:
        """Test finding places by alias."""
        resolver.create_place(
            name="New York City",
            aliases=["NYC", "Big Apple"],
        )

        results = resolver.find_by_name("NYC")
        assert len(results) == 1
        assert results[0].name == "New York City"

    def test_resolve_exact_match(self, resolver: LocationResolver) -> None:
        """Test resolving an exact match."""
        resolver.create_place(name="Tokyo")

        result = resolver.resolve("Tokyo")
        assert result.status == ResolutionStatus.RESOLVED
        assert result.place is not None
        assert result.place.name == "Tokyo"
        assert result.confidence == 1.0

    def test_resolve_ambiguous(self, resolver: LocationResolver) -> None:
        """Test resolving ambiguous references."""
        resolver.create_place(name="Springfield", place_type=PlaceType.PHYSICAL)
        resolver.create_place(name="Springfield", place_type=PlaceType.VIRTUAL)

        result = resolver.resolve("Springfield")
        assert result.status == ResolutionStatus.AMBIGUOUS
        assert result.candidates is not None
        assert len(result.candidates) == 2

    def test_resolve_with_context(self, resolver: LocationResolver) -> None:
        """Test resolving with context for disambiguation."""
        resolver.create_place(name="Springfield", place_type=PlaceType.PHYSICAL)
        resolver.create_place(name="Springfield", place_type=PlaceType.VIRTUAL)

        result = resolver.resolve("Springfield", place_type="VIRTUAL")
        assert result.status == ResolutionStatus.RESOLVED
        assert result.place is not None
        assert result.place.place_type == PlaceType.VIRTUAL

    def test_resolve_unresolved(self, resolver: LocationResolver) -> None:
        """Test resolving a non-existent place."""
        result = resolver.resolve("Nonexistent Place")
        assert result.status == ResolutionStatus.UNRESOLVED
        assert result.place is None
        assert result.error is not None

    def test_link_places(self, resolver: LocationResolver) -> None:
        """Test creating spatial links."""
        place1 = resolver.create_place(name="Place A")
        place2 = resolver.create_place(name="Place B")

        link = resolver.link_places(
            source_id=place1.id,
            target_id=place2.id,
            relation=SpatialRelation.ADJACENT,
            bidirectional=True,
        )

        assert link.source_id == place1.id
        assert link.target_id == place2.id
        assert link.relation == SpatialRelation.ADJACENT
        assert link.bidirectional is True
        assert resolver.link_count == 1

    def test_get_related(self, resolver: LocationResolver) -> None:
        """Test getting related places."""
        place1 = resolver.create_place(name="Center")
        place2 = resolver.create_place(name="North")
        place3 = resolver.create_place(name="South")

        resolver.link_places(place1.id, place2.id, SpatialRelation.ADJACENT)
        resolver.link_places(place1.id, place3.id, SpatialRelation.CONTAINS)

        related = resolver.get_related(place1.id)
        assert len(related) == 2

        # Filter by relation
        adjacent = resolver.get_related(place1.id, relation=SpatialRelation.ADJACENT)
        assert len(adjacent) == 1
        assert adjacent[0][0].name == "North"

    def test_assign_entity(self, resolver: LocationResolver) -> None:
        """Test assigning an entity to a place."""
        place = resolver.create_place(name="Office")

        assignment = resolver.assign_entity(
            entity_id="user_123",
            place_id=place.id,
            entity_type="user",
        )

        assert assignment.entity_id == "user_123"
        assert assignment.place_id == place.id
        assert assignment.entity_type == "user"
        assert resolver.assignment_count == 1

    def test_get_entity_locations(self, resolver: LocationResolver) -> None:
        """Test getting entity locations."""
        place1 = resolver.create_place(name="Office")
        place2 = resolver.create_place(name="Home")

        resolver.assign_entity("user_123", place1.id, "user")
        resolver.assign_entity("user_123", place2.id, "user")

        locations = resolver.get_entity_locations("user_123")
        assert len(locations) == 2
        assert {p.name for p in locations} == {"Office", "Home"}

    def test_get_entities_at_place(self, resolver: LocationResolver) -> None:
        """Test getting entities at a place."""
        place = resolver.create_place(name="Meeting Room")

        resolver.assign_entity("user_1", place.id, "user")
        resolver.assign_entity("user_2", place.id, "user")

        entities = resolver.get_entities_at_place(place.id)
        assert len(entities) == 2
        assert {e.entity_id for e in entities} == {"user_1", "user_2"}

    def test_find_path(self, resolver: LocationResolver) -> None:
        """Test finding a path between places."""
        # Create a chain: A -> B -> C -> D
        place_a = resolver.create_place(name="A")
        place_b = resolver.create_place(name="B")
        place_c = resolver.create_place(name="C")
        place_d = resolver.create_place(name="D")

        resolver.link_places(place_a.id, place_b.id, SpatialRelation.CONNECTED, bidirectional=True)
        resolver.link_places(place_b.id, place_c.id, SpatialRelation.CONNECTED, bidirectional=True)
        resolver.link_places(place_c.id, place_d.id, SpatialRelation.CONNECTED, bidirectional=True)

        path = resolver.find_path(place_a.id, place_d.id)
        assert path is not None
        assert len(path) == 4
        assert path[0].name == "A"
        assert path[-1].name == "D"

    def test_find_path_no_path(self, resolver: LocationResolver) -> None:
        """Test finding path when none exists."""
        place_a = resolver.create_place(name="Isolated A")
        place_b = resolver.create_place(name="Isolated B")

        path = resolver.find_path(place_a.id, place_b.id)
        assert path is None

    def test_search_places(self, resolver: LocationResolver) -> None:
        """Test searching for places."""
        resolver.create_place(name="New York", description="City in USA")
        resolver.create_place(name="New Orleans", description="City in Louisiana")
        resolver.create_place(name="Los Angeles", description="City in California")

        results = resolver.search_places("New")
        assert len(results) == 2
        assert all("New" in p.name for p in results)

    def test_hierarchical_places(self, resolver: LocationResolver) -> None:
        """Test hierarchical place structure."""
        country = resolver.create_place(name="USA", place_type=PlaceType.PHYSICAL)
        state = resolver.create_place(
            name="California",
            place_type=PlaceType.PHYSICAL,
            parent_id=country.id,
        )
        city = resolver.create_place(
            name="Los Angeles",
            place_type=PlaceType.PHYSICAL,
            parent_id=state.id,
        )

        # Verify hierarchy
        assert city.parent_id == state.id
        assert state.parent_id == country.id

    def test_get_stats(self, resolver: LocationResolver) -> None:
        """Test getting statistics."""
        resolver.create_place(name="Physical Place", place_type=PlaceType.PHYSICAL)
        resolver.create_place(name="Virtual Place", place_type=PlaceType.VIRTUAL)
        place1 = resolver.create_place(name="Place 1")
        place2 = resolver.create_place(name="Place 2")
        resolver.link_places(place1.id, place2.id, SpatialRelation.CONNECTED)
        resolver.assign_entity("user_1", place1.id, "user")

        stats = resolver.get_stats()
        assert stats.total_places == 4
        assert stats.total_links == 1
        assert stats.total_assignments == 1
        assert PlaceType.PHYSICAL.name in stats.places_by_type
        assert PlaceType.VIRTUAL.name in stats.places_by_type

    def test_clear(self, resolver: LocationResolver) -> None:
        """Test clearing all data."""
        place1 = resolver.create_place(name="Place 1")
        place2 = resolver.create_place(name="Place 2")
        resolver.link_places(place1.id, place2.id, SpatialRelation.CONNECTED)
        resolver.assign_entity("user_1", place1.id, "user")

        places, links, assignments = resolver.clear()
        assert places == 2
        assert links == 1
        assert assignments == 1
        assert resolver.place_count == 0
        assert resolver.link_count == 0
        assert resolver.assignment_count == 0

    def test_different_place_types(self, resolver: LocationResolver) -> None:
        """Test creating different types of places."""
        physical = resolver.create_place(name="Building", place_type=PlaceType.PHYSICAL)
        virtual = resolver.create_place(name="VR Space", place_type=PlaceType.VIRTUAL)
        conceptual = resolver.create_place(name="Dream Realm", place_type=PlaceType.CONCEPTUAL)
        mythic = resolver.create_place(name="Valhalla", place_type=PlaceType.MYTHIC)
        liminal = resolver.create_place(name="Threshold", place_type=PlaceType.LIMINAL)

        assert physical.place_type == PlaceType.PHYSICAL
        assert virtual.place_type == PlaceType.VIRTUAL
        assert conceptual.place_type == PlaceType.CONCEPTUAL
        assert mythic.place_type == PlaceType.MYTHIC
        assert liminal.place_type == PlaceType.LIMINAL

    def test_different_spatial_relations(self, resolver: LocationResolver) -> None:
        """Test different spatial relationships."""
        place1 = resolver.create_place(name="Place 1")
        place2 = resolver.create_place(name="Place 2")

        relations = [
            SpatialRelation.CONTAINS,
            SpatialRelation.CONTAINED_BY,
            SpatialRelation.ADJACENT,
            SpatialRelation.CONNECTED,
            SpatialRelation.DISTANT,
            SpatialRelation.OVERLAPS,
            SpatialRelation.OPPOSITE,
            SpatialRelation.PARALLEL,
        ]

        for relation in relations:
            link = resolver.link_places(place1.id, place2.id, relation)
            assert link.relation == relation
