#!/usr/bin/env python3
"""
Example: Recursive Process Demo

Demonstrates:
- The 4-phase ProcessLoop pattern (intake, process, evaluate, integrate)
- Output -> Input feedback loop
- Evolution and growth patterns over iterations

This shows the recursive-generative core:
INTAKE -> PROCESS -> EVALUATE -> INTEGRATE -> (feedback) -> INTAKE...
"""

import asyncio

from autogenrec.subsystems.temporal.evolution_scheduler import (
    EvolutionScheduler,
    GrowthPhase,
    MutationType,
)
from autogenrec.subsystems.temporal.time_manager import (
    TimeManager,
    CycleType,
)
from autogenrec.subsystems.transformation.process_converter import (
    ProcessConverter,
    ConversionFormat,
)
from autogenrec.subsystems.transformation.consumption_manager import (
    ConsumptionManager,
    ResourceType,
)
from decimal import Decimal


def main():
    print("=" * 60)
    print("Recursive Process Demo")
    print("=" * 60)
    print()

    # Initialize subsystems
    evolution = EvolutionScheduler()
    time_mgr = TimeManager()
    converter = ProcessConverter()
    consumption = ConsumptionManager()

    # =========================================================================
    # Step 1: Create an evolvable pattern
    # =========================================================================
    print("Step 1: Creating Evolvable Pattern")
    print("-" * 40)

    # Create a growth pattern that will evolve
    pattern = evolution.create_pattern(
        name="data_processor",
        content={
            "type": "processor",
            "version": "1.0",
            "capabilities": ["parse", "validate"],
            "efficiency": 0.7,
        },
        phase=GrowthPhase.DORMANT,
        description="A data processing pattern that evolves over time",
    )

    print(f"  Pattern Created: {pattern.name}")
    print(f"    ID: {pattern.id}")
    print(f"    Phase: {pattern.phase.name}")
    print(f"    Generation: {pattern.generation}")
    print(f"    Content: {pattern.content}")
    print()

    # =========================================================================
    # Step 2: Run recursive evolution cycles
    # =========================================================================
    print("Step 2: Recursive Evolution Cycles")
    print("-" * 40)

    # Track resource consumption during evolution
    consumption.add_quota(
        name="evolution_compute",
        resource_type=ResourceType.COMPUTE,
        max_amount=Decimal("1000"),
    )

    current_pattern = pattern
    for cycle in range(1, 6):
        print(f"\n  === Cycle {cycle} ===")

        # INTAKE PHASE: Receive pattern state
        print(f"  [Intake] Receiving pattern (gen {current_pattern.generation})")

        # PROCESS PHASE: Apply mutation
        event = consumption.create_event(
            consumer_id="evolution_system",
            resource_type=ResourceType.COMPUTE,
            amount=Decimal("50"),
            context=f"evolution_cycle_{cycle}",
        )
        consumption.consume(event)

        # Mutate the pattern
        mutation_type = MutationType.ADDITION if cycle % 2 == 1 else MutationType.MODIFICATION
        mutated, mutation = evolution.mutate_pattern(
            current_pattern.id,
            mutation_type=mutation_type,
        )

        print(f"  [Process] Applied {mutation.mutation_type.name} mutation")
        print(f"    Mutation ID: {mutation.id[:16]}...")

        # EVALUATE PHASE: Assess fitness
        fitness_delta = mutated.fitness - current_pattern.fitness
        should_continue = mutated.generation < 5

        print(f"  [Evaluate] Fitness: {mutated.fitness:.3f} (delta: {fitness_delta:+.3f})")
        print(f"    Continue evolution: {should_continue}")

        # INTEGRATE PHASE: Update state and feedback
        print(f"  [Integrate] Pattern evolved to generation {mutated.generation}")

        # Advance phase if conditions met
        if mutated.generation >= 2 and mutated.phase == GrowthPhase.DORMANT:
            advanced = evolution.advance_phase(mutated.id)
            if advanced:
                mutated = advanced
                print(f"    Phase advanced to: {mutated.phase.name}")

        # Feedback: The output becomes the next input
        current_pattern = mutated

    print()

    # =========================================================================
    # Step 3: Convert final pattern to different formats
    # =========================================================================
    print("Step 3: Converting Evolved Pattern")
    print("-" * 40)

    # Register the evolved pattern as a process
    process = converter.register_process(
        name=f"evolved_{current_pattern.name}",
        steps=[
            {"name": "intake", "action": "receive_data"},
            {"name": "process", "action": "transform_data"},
            {"name": "evaluate", "action": "assess_quality"},
            {"name": "integrate", "action": "emit_results"},
        ],
        inputs=["raw_data"],
        outputs=["processed_data", "quality_metrics"],
        metadata={
            "generation": current_pattern.generation,
            "fitness": current_pattern.fitness,
            "phase": current_pattern.phase.name,
        },
    )

    print(f"  Registered Process: {process.name}")

    # Convert to different formats
    json_result = converter.convert(process.id, ConversionFormat.JSON)
    print(f"\n  JSON Format:")
    print(f"    Success: {json_result.success}")
    if json_result.success:
        content = json_result.output.content
        if isinstance(content, dict):
            print(f"    Keys: {list(content.keys())}")

    yaml_result = converter.convert(process.id, ConversionFormat.YAML)
    print(f"\n  YAML Format:")
    print(f"    Success: {yaml_result.success}")

    schema_result = converter.convert(process.id, ConversionFormat.SCHEMA)
    print(f"\n  Schema Format:")
    print(f"    Success: {schema_result.success}")
    print()

    # =========================================================================
    # Step 4: Resource consumption summary
    # =========================================================================
    print("Step 4: Resource Consumption")
    print("-" * 40)

    allowed, remaining = consumption.check_quota(
        "evolution_system",
        ResourceType.COMPUTE,
        Decimal("1"),
    )

    print(f"  Compute Resources:")
    print(f"    Total Allocated: 1000 units")
    print(f"    Remaining: {remaining} units")
    print(f"    Used: {1000 - int(remaining)} units")
    print(f"    Per Cycle: ~50 units")

    consumption_stats = consumption.get_stats()
    print(f"\n  Consumption Stats:")
    print(f"    Total Events: {consumption_stats.total_events}")
    print(f"    Approved: {consumption_stats.consumed_count}")
    print(f"    Denied: {consumption_stats.rejected_count}")

    print()

    # =========================================================================
    # Step 6: Final pattern state
    # =========================================================================
    print("Step 5: Final Pattern State")
    print("-" * 40)

    final = evolution.get_pattern(current_pattern.id)
    print(f"  Name: {final.name}")
    print(f"  Generation: {final.generation}")
    print(f"  Phase: {final.phase.name}")
    print(f"  Fitness: {final.fitness:.3f}")
    print(f"  Stability: {final.stability.name}")
    print(f"  Content: {final.content}")

    print()
    print("=" * 60)
    print("Example Complete!")
    print("=" * 60)
    print()
    print("Key Concepts Demonstrated:")
    print("  1. 4-phase loop: intake -> process -> evaluate -> integrate")
    print("  2. Output -> Input feedback: each cycle's output feeds the next")
    print("  3. Evolution: patterns grow and adapt through mutations")
    print("  4. Resource tracking: compute usage monitored per cycle")


if __name__ == "__main__":
    main()
