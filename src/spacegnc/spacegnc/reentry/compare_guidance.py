"""Run selected guidance strategies on the EXAMPLE scenario and write comparison plots and tables.

All strategies share one set of arc primitives. Each strategy is a function returning a GuidanceRun,
registered in STRATEGIES, so a new one needs a runner here and an adapter in evaluation.

    python -m spacegnc.reentry.compare_guidance --strategies enumeration lp pc pc-bias --output-dir compare
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from spacegnc.logconfig import create_logger
from spacegnc.reentry import comparison_plots
from spacegnc.reentry.benchmark import count_evaluations
from spacegnc.reentry.evaluation import GuidanceRun, from_closed_loop, from_shooting, summary_rows, write_summary
from spacegnc.reentry.lp_planner import LinearProgramReversalPlanner
from spacegnc.reentry.planner import ArcPrimitives, BankReversalPlanner
from spacegnc.reentry.predictor_corrector import PredictorCorrectorGuidance
from spacegnc.reentry.run_reentry import RANGE_MARGIN
from spacegnc.reentry.scenario import ReentryScenario
from spacegnc.reentry.shooting import shoot
from spacegnc.reentry.simulation import BankReversalSimulator

LOG = create_logger(__name__)


@dataclass
class ComparisonCase:
    """Scenario, simulator and shared arc primitives every strategy runs on."""

    scenario: ReentryScenario
    simulator: BankReversalSimulator
    primitives: ArcPrimitives
    heading0: float
    initial_bank_deg: float
    primitives_time_s: float

    @classmethod
    def build(cls, scenario: ReentryScenario) -> "ComparisonCase":
        """Build the simulator and the arc primitives of the scenario."""
        frame = scenario.build_frame()
        initial_state = scenario.initial_state(frame)
        simulator = BankReversalSimulator(
            scenario.build_dynamics(frame), initial_state, scenario.terminal_energy, scenario.bank_rate_deg
        )
        _, heading0 = frame.flight_path_angle_and_heading(initial_state[:3], initial_state[3:])
        start_time = time.time()
        primitives = ArcPrimitives(simulator, scenario.bank_grid_deg)
        initial_bank_deg = primitives.bank_magnitude_for_range(RANGE_MARGIN * abs(scenario.entry_downrange))
        return cls(scenario, simulator, primitives, heading0, initial_bank_deg, time.time() - start_time)

    @property
    def entry(self) -> tuple:
        """Entry downrange, crossrange, heading and clearance margin passed to the planners."""
        scenario = self.scenario
        return scenario.entry_downrange, scenario.entry_crossrange, self.heading0, scenario.clearance_margin


def shoot_plan(name: str, case: ComparisonCase, make_plan) -> GuidanceRun:
    """Time and count a reduced planner and the 3DOF shooting of its plan."""
    start_time = time.time()
    with count_evaluations(case.simulator) as planning_counter:
        plan = make_plan()
    planning_time = time.time() - start_time
    start_time = time.time()
    with count_evaluations(case.simulator) as shooting_counter:
        shot = shoot(case.simulator, plan, case.scenario.keep_out_zones)
    timing = {"primitives": case.primitives_time_s, "planning": planning_time, "shooting": time.time() - start_time}
    integrations = planning_counter.integrations + shooting_counter.integrations
    return from_shooting(name, shot, timing, planning_counter.reduced_paths, integrations)


def run_enumeration(case: ComparisonCase, options) -> GuidanceRun:
    """Side assignment enumeration with the refined best plan shot in 3DOF."""
    planner = BankReversalPlanner(case.primitives, case.scenario.keep_out_zones, *case.entry)
    return shoot_plan(
        "enumeration",
        case,
        lambda: planner.refine_bank_magnitude(planner.plan_all_assignments(case.initial_bank_deg)[0]),
    )


def run_linear_program(case: ComparisonCase, options) -> GuidanceRun:
    """Sequential LP on the relaxed bank sign with its polished plan shot in 3DOF."""
    planner = LinearProgramReversalPlanner(case.primitives, case.scenario.keep_out_zones, *case.entry)
    return shoot_plan("lp", case, lambda: planner.plan(case.initial_bank_deg).plan)


def fly_predictor_corrector(name: str, case: ComparisonCase, bias_correction: bool, prediction_interval: int):
    """Time and count the closed-loop predictor-corrector flight."""
    scenario = case.scenario
    guidance = PredictorCorrectorGuidance(
        case.primitives,
        scenario.keep_out_zones,
        scenario.clearance_margin,
        bias_correction=bias_correction,
        prediction_interval=prediction_interval,
    )
    start_time = time.time()
    with count_evaluations(case.simulator) as counter:
        result = guidance.fly(case.simulator, case.initial_bank_deg)
    # Guidance and the flight are interleaved, the closed loop is booked as planning
    timing = {"primitives": case.primitives_time_s, "planning": time.time() - start_time}
    return from_closed_loop(name, result, timing, counter.reduced_paths)


def run_predictor_corrector(case: ComparisonCase, options) -> GuidanceRun:
    """Closed-loop predictor-corrector on reduced model predictions only."""
    return fly_predictor_corrector("pc", case, False, 1)


def run_bias_corrected(case: ComparisonCase, options) -> GuidanceRun:
    """Closed-loop predictor-corrector with the 3DOF terminal bias every M cycles."""
    interval = options.prediction_interval
    return fly_predictor_corrector(f"pc-bias M={interval}", case, True, interval)


STRATEGIES = {
    "enumeration": run_enumeration,
    "lp": run_linear_program,
    "pc": run_predictor_corrector,
    "pc-bias": run_bias_corrected,
}


def load_scaling_rows(paths: list[Path]) -> list[dict]:
    """Synthetic scaling rows of benchmark.json files, later files replace rows of the same series and m."""
    rows = {}
    for path in paths:
        for row in json.loads(Path(path).read_text()).get("scaling", []):
            rows[(row["planner"], row.get("cells"), row["zones"], row["samples"])] = row
    return list(rows.values())


def write_plots(runs: list[GuidanceRun], scenario: ReentryScenario, output_dir: Path, scaling_rows: list[dict]):
    """Write every comparison plot that has data and close the figures."""
    zones = sorted(scenario.keep_out_zones, key=lambda zone: zone.downrange)
    entry_point = (scenario.entry_downrange, scenario.entry_crossrange)
    figures = [
        comparison_plots.plot_ground_tracks(
            runs, zones, scenario.clearance_margin, entry_point, output_dir / "ground_tracks.png"
        ),
        comparison_plots.plot_bank_vs_energy(runs, output_dir / "bank_vs_energy.png"),
        comparison_plots.plot_state_vs_energy(runs, output_dir / "state_vs_energy.png"),
        comparison_plots.plot_performance(runs, output_dir / "performance.png"),
        comparison_plots.plot_cycle_diagnostics(runs, output_dir / "cycle_diagnostics.png"),
    ]
    if scaling_rows:
        figures.append(comparison_plots.plot_scaling(scaling_rows, output_dir / "scaling.png"))
    for figure in figures:
        if figure is not None:
            plt.close(figure)


def main():
    """Parse command line arguments, run the strategies and write the comparison."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("guidance_comparison"), help="Directory for the outputs")
    parser.add_argument(
        "--strategies", nargs="+", choices=tuple(STRATEGIES), default=list(STRATEGIES), help="Strategies to compare"
    )
    parser.add_argument("--prediction-interval", type=int, default=1, help="Cycles M between pc-bias 3DOF predictions")
    parser.add_argument(
        "--scaling-json", type=Path, nargs="*", default=[], help="benchmark.json files whose scaling rows are plotted"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scenario = ReentryScenario()
    case = ComparisonCase.build(scenario)
    LOG.info(f"Arc primitives built in {case.primitives_time_s:.1f} s, |sigma|_0={case.initial_bank_deg:.2f} deg")
    runs = []
    for strategy in args.strategies:
        runs.append(STRATEGIES[strategy](case, args))
        LOG.info(f"{runs[-1].name}: miss {runs[-1].miss_norm_km:.3f} km, timing {runs[-1].timing_s}")
    write_summary(runs, args.output_dir)
    write_plots(runs, scenario, args.output_dir, load_scaling_rows(args.scaling_json))
    LOG.info("\n" + "\n".join(" | ".join(str(cell) for cell in row) for row in summary_rows(runs)))
    LOG.info(f"Comparison written to {args.output_dir}")


if __name__ == "__main__":
    main()
