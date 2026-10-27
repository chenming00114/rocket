"""Fly guidance strategies in closed loop on the EXAMPLE scenario through truth atmosphere cases.

The guidance keeps the nominal atmosphere while the truth plant flies the nominal one with the density
scaled by f above an altitude, one case per --density-scale. Each strategy is a function returning a
Guidance, registered in STRATEGIES, so a new one needs a Guidance and a factory here.

    python -m spacegnc.reentry.simulate --strategies lp-open-loop lp-closed-loop pc-bias \\
        --density-scale 0.8 1.0 1.2 --scale-above-km 35 --guidance-period 5 --output-dir closed_loop
"""

import argparse
import functools
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spacegnc.logconfig import create_logger
from spacegnc.reentry import comparison_plots
from spacegnc.reentry.closed_loop import ClosedLoopSimulator
from spacegnc.reentry.compare_guidance import ComparisonCase
from spacegnc.reentry.evaluation import GuidanceRun, from_flight, write_summary
from spacegnc.reentry.guidance_adapters import ArcLinearProgramGuidance, OpenLoopPlanGuidance, PredictorCorrectorAdapter
from spacegnc.reentry.lp_planner import LinearProgramReversalPlanner
from spacegnc.reentry.predictor_corrector import PredictorCorrectorGuidance
from spacegnc.reentry.scenario import ReentryScenario
from spacegnc.reentry.shooting import MEGAJOULE, shoot
from spacegnc.reentry.simulation import BankReversalSimulator
from spacegnc.reentry.truth_atmosphere import ScaledDensityAtmosphere, with_atmosphere
from spacegnc.reentry.vertical_lift import VerticalLiftGuidance

LOG = create_logger(__name__)

KILOMETER = 1e3


def nominal_shot(case: ComparisonCase):
    """Arc LP plan shot in nominal 3DOF before entry, computed once per case for the open-loop reference."""
    if not hasattr(case, "nominal_shot"):
        planner = LinearProgramReversalPlanner(case.primitives, case.scenario.keep_out_zones, *case.entry)
        case.nominal_shot = shoot(case.simulator, planner.plan(case.initial_bank_deg).plan, case.scenario.keep_out_zones)
    return case.nominal_shot


def open_loop_plan(case: ComparisonCase, options) -> OpenLoopPlanGuidance:
    """Nominal arc LP plan shot in 3DOF, flown without feedback."""
    return OpenLoopPlanGuidance.from_shooting(nominal_shot(case))


def closed_loop_linear_program(case: ComparisonCase, options) -> ArcLinearProgramGuidance:
    """Arc LP re-planned from the navigated state and re-shot in nominal 3DOF every update."""
    scenario = case.scenario
    return ArcLinearProgramGuidance(
        case.primitives,
        scenario.keep_out_zones,
        scenario.clearance_margin,
        case.simulator,
        case.initial_bank_deg,
        replan_interval=options.replan_interval,
        shooting=not options.no_reshoot,
    )


def predictor_corrector(case: ComparisonCase, options, bias_correction: bool = True) -> PredictorCorrectorAdapter:
    """Predictor-corrector, its 3DOF bias every update on the nominal model."""
    scenario = case.scenario
    guidance = PredictorCorrectorGuidance(
        case.primitives, scenario.keep_out_zones, scenario.clearance_margin, bias_correction=bias_correction
    )
    return PredictorCorrectorAdapter(guidance, case.simulator, case.initial_bank_deg)


def vertical_lift(case: ComparisonCase, options) -> VerticalLiftGuidance:
    """Vertical L/D rooted for the downrange in nominal 3DOF every update, reversals at a crossrange deadband."""
    return VerticalLiftGuidance(case.simulator, case.initial_bank_deg)


STRATEGIES = {
    "lp-open-loop": open_loop_plan,
    "lp-closed-loop": closed_loop_linear_program,
    "pc-bias": predictor_corrector,
    "pc": functools.partial(predictor_corrector, bias_correction=False),
    "vertical-ld": vertical_lift,
}


def peak_load(run: GuidanceRun) -> float:
    """Peak aerodynamic load |L + D| / g0 of the flown trajectory."""
    return float(np.max(run.trajectory.load_factor))


def compute_stat(run: GuidanceRun, reduce) -> float:
    """Reduced guidance compute time per update (s)."""
    return float(reduce(run.cycle_diagnostics["compute_s"]))


# Columns of the closed-loop summary table, header and the GuidanceRun row getter
CLOSED_LOOP_COLUMNS = (
    ("strategy", lambda run: run.name),
    ("sides", lambda run: run.sides),
    ("reversals [MJ/kg]", lambda run: " / ".join(f"{energy / MEGAJOULE:.3f}" for energy in run.reversal_energies)),
    ("sigma_f [deg]", lambda run: f"{run.bank_magnitude_deg:.2f}"),
    ("X_f [km]", lambda run: f"{run.miss[0]:.3f}"),
    ("Y_f [km]", lambda run: f"{run.miss[1]:.3f}"),
    ("miss [km]", lambda run: f"{run.miss_norm_km:.3f}"),
    ("min clearance [km]", lambda run: f"{run.min_clearance_km:.1f}"),
    ("violated", lambda run: any(passage.clearance < 0 for passage in run.passages)),
    ("h_f [km]", lambda run: f"{run.trajectory.altitude[-1] / KILOMETER:.2f}"),
    ("V_f [m/s]", lambda run: f"{run.trajectory.speed[-1]:.0f}"),
    ("psi_f [deg]", lambda run: f"{np.degrees(run.trajectory.heading[-1]):.2f}"),
    ("peak load [g]", lambda run: f"{peak_load(run):.2f}"),
    ("peak q [kPa]", lambda run: f"{np.max(run.trajectory.dynamic_pressure) / 1e3:.1f}"),
    ("peak heat rate [kW/m^2]", lambda run: f"{np.max(run.trajectory.heat_rate) / 1e3:.0f}"),
    ("heat load [MJ/m^2]", lambda run: f"{run.trajectory.heat_load / 1e6:.1f}"),
    ("cycles", lambda run: run.cycle_diagnostics["compute_s"].size),
    ("mean cycle [s]", lambda run: f"{compute_stat(run, np.mean):.3f}"),
    ("max cycle [s]", lambda run: f"{compute_stat(run, np.max):.3f}"),
    ("wall [s]", lambda run: f"{run.total_time_s:.1f}"),
    ("status", lambda run: run.trajectory.status),
)


def truth_plant(case: ComparisonCase, density_scale: float, scale_above: float) -> BankReversalSimulator:
    """Plant flying the nominal atmosphere with the density scaled by f above the altitude (m)."""
    nominal = case.simulator
    atmosphere = ScaledDensityAtmosphere(nominal.dynamics.atmosphere, density_scale, scale_above)
    return BankReversalSimulator(
        with_atmosphere(nominal.dynamics, atmosphere),
        nominal.initial_state,
        nominal.terminal_energy,
        np.degrees(nominal.bank_rate),
        nominal.max_step,
    )


def write_plots(runs: list[GuidanceRun], scenario: ReentryScenario, output_dir: Path, suffix: str):
    """Write the comparison plots of a set of runs with a file name suffix and close the figures."""
    zones = sorted(scenario.keep_out_zones, key=lambda zone: zone.downrange)
    entry_point = (scenario.entry_downrange, scenario.entry_crossrange)
    figures = [
        comparison_plots.plot_ground_tracks(
            runs, zones, scenario.clearance_margin, entry_point, output_dir / f"ground_tracks_{suffix}.png"
        ),
        comparison_plots.plot_bank_vs_energy(runs, output_dir / f"bank_vs_energy_{suffix}.png"),
        comparison_plots.plot_bank_vs_time(runs, output_dir / f"bank_vs_time_{suffix}.png"),
        comparison_plots.plot_state_vs_energy(runs, output_dir / f"state_vs_energy_{suffix}.png"),
        comparison_plots.plot_performance(runs, output_dir / f"performance_{suffix}.png"),
        comparison_plots.plot_constraint_history(runs, output_dir / f"constraints_{suffix}.png"),
        comparison_plots.plot_cycle_diagnostics(runs, output_dir / f"cycle_diagnostics_{suffix}.png"),
    ]
    for figure in figures:
        if figure is not None:
            plt.close(figure)


def main():
    """Parse command line arguments, fly every strategy in every atmosphere case and write the comparison."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("closed_loop"), help="Directory for the outputs")
    parser.add_argument(
        "--strategies", nargs="+", choices=tuple(STRATEGIES), default=list(STRATEGIES), help="Strategies to fly"
    )
    parser.add_argument("--density-scale", type=float, nargs="+", default=[1.0], help="Truth density scales f")
    parser.add_argument("--scale-above-km", type=float, default=35.0, help="Altitude above which density is scaled")
    parser.add_argument("--guidance-period", type=float, default=5.0, help="Simulated seconds between updates")
    parser.add_argument("--replan-interval", type=int, default=12, help="Updates between closed-loop arc LP re-plans")
    parser.add_argument("--no-reshoot", action="store_true", help="Fly the closed-loop arc LP without 3DOF re-shooting")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scenario = ReentryScenario()
    case = ComparisonCase.build(scenario)
    runs = []
    for density_scale in args.density_scale:
        plant = truth_plant(case, density_scale, args.scale_above_km * KILOMETER)
        simulator = ClosedLoopSimulator(plant, scenario.keep_out_zones, args.guidance_period)
        case_runs = []
        for strategy in args.strategies:
            start_time = time.time()
            guidance = STRATEGIES[strategy](case, args)
            setup_time = time.time() - start_time
            flight = simulator.fly(guidance)
            timing = {"setup": setup_time, "closed loop": time.time() - start_time - setup_time}
            case_runs.append(from_flight(f"{strategy} f={density_scale:g}", flight, timing))
            LOG.info(f"{case_runs[-1].name}: miss {case_runs[-1].miss_norm_km:.3f} km, timing {timing}")
        write_plots(case_runs, scenario, args.output_dir, f"f{density_scale:g}")
        runs.extend(case_runs)
    for strategy in args.strategies:
        strategy_runs = [run for run in runs if run.name.startswith(f"{strategy} ")]
        write_plots(strategy_runs, scenario, args.output_dir, strategy)
    write_summary(runs, args.output_dir, CLOSED_LOOP_COLUMNS)
    LOG.info("\n" + (args.output_dir / "summary.md").read_text())


if __name__ == "__main__":
    main()
