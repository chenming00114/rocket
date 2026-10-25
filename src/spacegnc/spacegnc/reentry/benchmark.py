"""Implement the reversal planner benchmark of the EXAMPLE scenario and synthetic zone counts.

Every planner shares the arc primitives and the 3DOF shooting, so the comparison counts what differs:
reduced model path evaluations, LP interior point iterations, full 3DOF integrations and wall time.
The predictor-corrector guidance needs no shooting, its 3DOF closed-loop flight is counted in segments.

    python -m spacegnc.reentry.benchmark --output-dir reentry_benchmark
"""

import argparse
import contextlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from spacegnc.logconfig import create_logger
from spacegnc.reentry import planner as reduced_model
from spacegnc.reentry.lp_planner import LinearProgramReversalPlanner
from spacegnc.reentry.planner import ArcPrimitives, ArcProfile, BankReversalPlanner, side_label
from spacegnc.reentry.predictor_corrector import PredictorCorrectorGuidance
from spacegnc.reentry.run_reentry import RANGE_MARGIN
from spacegnc.reentry.scenario import KeepOutZone, ReentryScenario
from spacegnc.reentry.shooting import MEGAJOULE, shoot
from spacegnc.reentry.simulation import BankReversalSimulator

LOG = create_logger(__name__)

KILOMETER = 1e3


@dataclass
class EvaluationCounter:
    """Counts of reduced path evaluations and full 3DOF integrations."""

    reduced_paths: int = 0
    integrations: int = 0


@contextlib.contextmanager
def count_evaluations(simulator: BankReversalSimulator | None = None):
    """Count reduced arc chains, relaxed LP paths and 3DOF integrations made inside the context.

    A full run delegates to one segment, so counting segments counts both shooting runs and closed-loop segments.
    """
    counter = EvaluationCounter()
    arc_chain = reduced_model.reduced_arc_chain
    relaxed_path = LinearProgramReversalPlanner.relaxed_path
    run_segment = simulator.run_segment if simulator is not None else None

    def counted_arc_chain(*args, **kwargs):
        counter.reduced_paths += 1
        return arc_chain(*args, **kwargs)

    def counted_relaxed_path(self, *args, **kwargs):
        counter.reduced_paths += 1
        return relaxed_path(self, *args, **kwargs)

    def counted_run_segment(*args, **kwargs):
        counter.integrations += 1
        return run_segment(*args, **kwargs)

    reduced_model.reduced_arc_chain = counted_arc_chain
    LinearProgramReversalPlanner.relaxed_path = counted_relaxed_path
    if simulator is not None:
        simulator.run_segment = counted_run_segment
    try:
        yield counter
    finally:
        reduced_model.reduced_arc_chain = arc_chain
        LinearProgramReversalPlanner.relaxed_path = relaxed_path
        if simulator is not None:
            del simulator.run_segment


@dataclass
class PlannerRecord:
    """Outcome, cost and evaluation counts of one planner on the example scenario."""

    planner: str
    sides: str
    reversal_count: int
    bank_magnitude_deg: float
    reversal_energies_mj_per_kg: list[float]
    miss_km: float
    min_clearance_km: float
    works: bool
    planning_time_s: float
    shooting_time_s: float
    planning_reduced_paths: int
    planning_integrations: int
    shooting_integrations: int
    extra: dict = field(default_factory=dict)


def shoot_and_record(name, simulator, scenario, plan, planning_time, planning_counter, extra=None) -> PlannerRecord:
    """Shoot a reduced plan in the full 3DOF dynamics and record the outcome."""
    start_time = time.time()
    with count_evaluations(simulator) as counter:
        shot = shoot(simulator, plan, scenario.keep_out_zones)
    min_clearance = min(passage.clearance for passage in shot.passages)
    return PlannerRecord(
        planner=name,
        sides=plan.label,
        reversal_count=plan.reversal_count,
        bank_magnitude_deg=shot.bank_magnitude_deg,
        reversal_energies_mj_per_kg=list(shot.reversal_energies / MEGAJOULE),
        miss_km=float(np.linalg.norm(shot.miss)),
        min_clearance_km=min_clearance / KILOMETER,
        works=bool(np.linalg.norm(shot.miss) < 0.2 and min_clearance >= 0.0),
        planning_time_s=planning_time,
        shooting_time_s=time.time() - start_time,
        planning_reduced_paths=planning_counter.reduced_paths,
        planning_integrations=planning_counter.integrations,
        shooting_integrations=counter.integrations,
        extra=extra or {},
    )


def fly_and_record(name, simulator, scenario, primitives, initial_bank_deg, prediction_interval) -> PlannerRecord:
    """Fly the predictor-corrector in closed loop, with the 3DOF bias correction every M cycles unless M is None."""
    start_time = time.time()
    with count_evaluations(simulator) as counter:
        guidance = PredictorCorrectorGuidance(
            primitives,
            scenario.keep_out_zones,
            scenario.clearance_margin,
            bias_correction=prediction_interval is not None,
            prediction_interval=prediction_interval or 1,
        )
        result = guidance.fly(simulator, initial_bank_deg)
    guidance_time = time.time() - start_time
    # Guidance and 3DOF flight are interleaved, so the closed-loop time is booked as planning
    min_clearance = min(passage.clearance for passage in result.passages)
    return PlannerRecord(
        planner=name,
        sides=side_label([passage.side for passage in result.passages]),
        reversal_count=int(result.reversal_energies.size),
        bank_magnitude_deg=result.commands[-1].bank_magnitude_deg,
        reversal_energies_mj_per_kg=list(result.reversal_energies / MEGAJOULE),
        miss_km=float(np.linalg.norm(result.miss)),
        min_clearance_km=min_clearance / KILOMETER,
        works=bool(np.linalg.norm(result.miss) < 0.2 and min_clearance >= 0.0),
        planning_time_s=guidance_time,
        shooting_time_s=0.0,
        planning_reduced_paths=counter.reduced_paths,
        planning_integrations=counter.integrations,
        shooting_integrations=0,
        extra={
            "guidance_cycles": len(result.commands),
            "segments": result.integrations,
            "predictions_3dof": result.predictions,
            "prediction_flight_time_s": result.prediction_flight_time_s,
            "prediction_wall_time_s": result.prediction_wall_time_s,
            "status": result.trajectory.status,
        },
    )


def benchmark_example(scenario: ReentryScenario, prediction_intervals=(1,)) -> tuple[list[PlannerRecord], float]:
    """Run every planner on the example scenario from shared arc primitives.

    The predictor-corrector flies once without the 3DOF bias correction and once per prediction interval M.
    """
    frame = scenario.build_frame()
    initial_state = scenario.initial_state(frame)
    simulator = BankReversalSimulator(
        scenario.build_dynamics(frame), initial_state, scenario.terminal_energy, scenario.bank_rate_deg
    )
    _, heading0 = frame.flight_path_angle_and_heading(initial_state[:3], initial_state[3:])
    start_time = time.time()
    primitives = ArcPrimitives(simulator, scenario.bank_grid_deg)
    primitives_time = time.time() - start_time
    initial_bank_deg = primitives.bank_magnitude_for_range(RANGE_MARGIN * abs(scenario.entry_downrange))
    entry = (scenario.entry_downrange, scenario.entry_crossrange, heading0, scenario.clearance_margin)
    records = []

    start_time = time.time()
    with count_evaluations(simulator) as counter:
        enumeration = BankReversalPlanner(primitives, scenario.keep_out_zones, *entry)
        plans = enumeration.plan_all_assignments(initial_bank_deg)
        plan = enumeration.refine_bank_magnitude(plans[0])
    records.append(
        shoot_and_record(
            "enumeration",
            simulator,
            scenario,
            plan,
            time.time() - start_time,
            counter,
            {"assignments": 2 ** len(enumeration.zones)},
        )
    )

    start_time = time.time()
    with count_evaluations(simulator) as counter:
        lp_plan = LinearProgramReversalPlanner(primitives, scenario.keep_out_zones, *entry).plan(initial_bank_deg)
    extra = {
        "start_sign": lp_plan.start_sign,
        "rounds": len(lp_plan.rounds),
        "interior_point_iterations": sum(round_result.interior_point_iterations for round_result in lp_plan.rounds),
        "bang_bang_fraction": lp_plan.bang_bang_fraction,
    }
    records.append(shoot_and_record("lp", simulator, scenario, lp_plan.plan, time.time() - start_time, counter, extra))

    for prediction_interval in [None, *prediction_intervals]:
        name = "predictor-corrector" + ("" if prediction_interval is None else f" bias M={prediction_interval}")
        records.append(fly_and_record(name, simulator, scenario, primitives, initial_bank_deg, prediction_interval))
    return records, primitives_time


class SyntheticPrimitives:
    """Primitives stand-in returning one synthetic profile for every bank magnitude."""

    def __init__(self, profile: ArcProfile):
        """Initialize with the profile returned for every bank magnitude."""
        self.profile = profile
        self.bank_grid_deg = np.array([0.0, 90.0])

    def at(self, bank_magnitude_deg: float) -> ArcProfile:
        """Synthetic profile, independent of the bank magnitude."""
        return self.profile


def synthetic_profile(energy_samples: int, total_range: float = 2.0e6, curvature: float = 6e-7) -> ArcProfile:
    """Profile with arc length linear in energy and constant lift curvature."""
    energy = np.linspace(10e6, 60e6, energy_samples)
    arc_length = (energy - energy[0]) / (energy[-1] - energy[0]) * total_range
    return ArcProfile(
        bank_magnitude_deg=45.0,
        energy=energy,
        arc_length=arc_length,
        curvature=np.full_like(energy, curvature),
        bias_curvature=np.zeros_like(energy),
        energy_rate=np.full_like(energy, 1e4),
        heading_change=curvature * arc_length,
        bank_rate=np.radians(12.0),
    )


def synthetic_zones(zone_count: int, total_range: float = 2.0e6) -> list[KeepOutZone]:
    """Zones evenly spaced along the track, alternating above and below the reference line."""
    downranges = -total_range + total_range * (np.arange(zone_count) + 1) / (zone_count + 1)
    return [KeepOutZone(x, 25e3 * (-1) ** idx, 20e3) for idx, x in enumerate(downranges)]


def planner_scaling(zone_counts, energy_samples, cell_counts) -> list[dict]:
    """Planning wall time and reduced path count against m zones, samples and K cells.

    The predictor-corrector row is the cost of one reduced model guidance cycle at entry, a closed-loop
    flight repeats it once per cycle. The 3DOF bias prediction needs the full dynamics and is not included.
    """
    rows = []
    for zone_count in zone_counts:
        for samples in energy_samples:
            primitives = SyntheticPrimitives(synthetic_profile(samples))
            zones = synthetic_zones(zone_count)
            entry = (-2.0e6, 0.0, 0.0, 5e3)
            start_time = time.time()
            with count_evaluations() as counter:
                plans = BankReversalPlanner(primitives, zones, *entry).plan_all_assignments(45.0)
            rows.append(
                {
                    "planner": "enumeration",
                    "zones": zone_count,
                    "samples": samples,
                    "cells": None,
                    "time_s": time.time() - start_time,
                    "reduced_paths": counter.reduced_paths,
                    "feasible": bool(plans),
                    "reversals": plans[0].reversal_count if plans else None,
                }
            )
            for cell_count in cell_counts:
                start_time = time.time()
                with count_evaluations() as counter:
                    lp_plan = LinearProgramReversalPlanner(primitives, zones, *entry, cell_count=cell_count).plan(45.0)
                rows.append(
                    {
                        "planner": "lp",
                        "zones": zone_count,
                        "samples": samples,
                        "cells": cell_count,
                        "time_s": time.time() - start_time,
                        "reduced_paths": counter.reduced_paths,
                        "feasible": lp_plan.plan.feasible,
                        "reversals": lp_plan.plan.reversal_count,
                    }
                )
            guidance = PredictorCorrectorGuidance(primitives, zones, entry[3], bias_correction=False)
            start_time = time.time()
            with count_evaluations() as counter:
                command = guidance.command(entry[:3] + (primitives.profile.energy[0],), 1, 45.0)
            rows.append(
                {
                    "planner": "predictor-corrector cycle",
                    "zones": zone_count,
                    "samples": samples,
                    "cells": None,
                    "time_s": time.time() - start_time,
                    "reduced_paths": counter.reduced_paths,
                    "feasible": command.predicted_clearance_km * KILOMETER >= entry[3],
                    "reversals": None if command.next_reversal is None else 1,
                }
            )
            LOG.info(f"Scaling m={zone_count} samples={samples}: {rows[-len(cell_counts) - 2 :]}")
    return rows


def main():
    """Parse command line arguments and run the benchmark."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("reentry_benchmark"), help="Directory for the results")
    parser.add_argument("--skip-example", action="store_true", help="Only run the synthetic scaling study")
    parser.add_argument("--zones", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="Synthetic zone counts m")
    parser.add_argument("--samples", type=int, nargs="+", default=[1500], help="Synthetic energy samples")
    parser.add_argument("--cells", type=int, nargs="+", default=[20, 40, 80], help="LP energy cells K")
    parser.add_argument(
        "--prediction-intervals",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Guidance cycles M between predictor-corrector 3DOF bias predictions",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    if not args.skip_example:
        records, primitives_time = benchmark_example(ReentryScenario(), args.prediction_intervals)
        results["arc_primitives_time_s"] = primitives_time
        results["example"] = [asdict(record) for record in records]
        for record in records:
            LOG.info(f"Example {record}")
    results["scaling"] = planner_scaling(args.zones, args.samples, args.cells)
    with open(args.output_dir / "benchmark.json", "w") as file:
        json.dump(results, file, indent=1, default=float)
    LOG.info(f"Benchmark written to {args.output_dir / 'benchmark.json'}")


if __name__ == "__main__":
    main()
