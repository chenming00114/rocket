"""Run the EXAMPLE reentry bank-reversal planning case end to end.

Arc primitives -> side assignments -> reduced reversal plans -> full 3DOF shooting -> plots and summary.
The lp planner replaces the side assignment enumeration with sequential LPs on a relaxed bank sign, and
the predictor-corrector guidance replaces planning and shooting with closed-loop next reversal solves,
corrected by a 3DOF terminal bias prediction every --prediction-interval cycles unless --no-bias-correction.

    python -m spacegnc.reentry.run_reentry --output-dir reentry_output [--planner lp|predictor-corrector]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from spacegnc.logconfig import create_logger
from spacegnc.reentry.lp_planner import LinearProgramReversalPlanner
from spacegnc.reentry.planner import ArcPrimitives, BankReversalPlanner, side_label
from spacegnc.reentry.plotting import (
    plot_arc_primitives,
    plot_closed_loop,
    plot_convergence,
    plot_energy_windows,
    plot_ground_track,
    plot_profiles,
    plot_relaxed_bank_sign,
)
from spacegnc.reentry.predictor_corrector import PredictorCorrectorGuidance
from spacegnc.reentry.scenario import ReentryScenario
from spacegnc.reentry.shooting import MEGAJOULE, shoot
from spacegnc.reentry.simulation import BankReversalSimulator

LOG = create_logger(__name__)

# Margin of required arc length over the straight downrange to pay for weaving
RANGE_MARGIN = 1.03
# Planner choices, enumeration of side assignments is the default
ENUMERATION_PLANNER = "enumeration"
LINEAR_PROGRAM_PLANNER = "lp"
PREDICTOR_CORRECTOR_PLANNER = "predictor-corrector"


def run_case(
    scenario: ReentryScenario,
    output_dir: Path,
    planner_kind: str = ENUMERATION_PLANNER,
    bias_correction: bool = True,
    prediction_interval: int = 1,
) -> dict:
    """Plan, shoot and plot the scenario with the chosen planner, returning the run summary."""
    start_time = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = scenario.build_frame()
    dynamics = scenario.build_dynamics(frame)
    initial_state = scenario.initial_state(frame)
    simulator = BankReversalSimulator(dynamics, initial_state, scenario.terminal_energy, scenario.bank_rate_deg)
    flight_path_angle0, heading0 = frame.flight_path_angle_and_heading(initial_state[:3], initial_state[3:])
    inertial_speed, inertial_flight_path_angle = scenario.entry_inertial_speed_and_flight_path_angle()
    LOG.info(
        f"Entry: inertial V={inertial_speed:.1f} m/s FPA={np.degrees(inertial_flight_path_angle):.3f} deg; "
        f"Earth-relative V={np.linalg.norm(initial_state[3:]):.1f} m/s FPA={np.degrees(flight_path_angle0):.3f} deg "
        f"psi={np.degrees(heading0):.2f} deg; e_f={scenario.terminal_energy / MEGAJOULE:.3f} MJ/kg"
    )

    primitives = ArcPrimitives(simulator, scenario.bank_grid_deg)
    LOG.info(f"Arc primitives built for {primitives.bank_grid_deg.size} bank magnitudes ({time.time() - start_time:.0f} s)")
    planner = BankReversalPlanner(
        primitives,
        scenario.keep_out_zones,
        scenario.entry_downrange,
        scenario.entry_crossrange,
        heading0,
        scenario.clearance_margin,
    )
    initial_bank_deg = primitives.bank_magnitude_for_range(RANGE_MARGIN * abs(scenario.entry_downrange))
    LOG.info(f"|sigma| from S(e_f) = {RANGE_MARGIN}|X0|: {initial_bank_deg:.2f} deg")

    if planner_kind == PREDICTOR_CORRECTOR_PLANNER:
        return fly_closed_loop(
            scenario, simulator, primitives, initial_bank_deg, output_dir, start_time, bias_correction, prediction_interval
        )

    planning_start_time = time.time()
    if planner_kind == LINEAR_PROGRAM_PLANNER:
        lp_planner = LinearProgramReversalPlanner(
            primitives,
            scenario.keep_out_zones,
            scenario.entry_downrange,
            scenario.entry_crossrange,
            heading0,
            scenario.clearance_margin,
        )
        lp_plan = lp_planner.plan(initial_bank_deg)
        if not lp_plan.plan.feasible:
            LOG.warning("LP reversal plan misses the reduced tolerances, shooting it anyway")
        plans, refined_plans = [lp_plan.plan], [lp_plan.plan]
        plot_relaxed_bank_sign(output_dir / "relaxed_bank_sign.png", lp_plan)
    else:
        plans = planner.plan_all_assignments(initial_bank_deg)
        if not plans:
            raise RuntimeError("No side assignment is feasible in the reduced model.")
        # Shoot the best plan and the runner up for comparison
        refined_plans = [planner.refine_bank_magnitude(plan) for plan in plans[:2]]
    planning_time = time.time() - planning_start_time
    LOG.info(f"Reduced planning with the {planner_kind} planner took {planning_time:.1f} s")

    shots = []
    for name, color, refined in zip(("chosen", "alternate"), ("tab:blue", "tab:gray"), refined_plans):
        shot = shoot(simulator, refined, scenario.keep_out_zones)
        LOG.info(
            f"Shooting {name} {refined.label}: miss history {np.round(shot.miss_history, 3)} km, "
            f"|sigma|={shot.bank_magnitude_deg:.3f} deg, e_k={np.round(shot.reversal_energies / MEGAJOULE, 3)} MJ/kg, "
            f"clearances={[round(float(passage.clearance) / 1e3, 1) for passage in shot.passages]} km, "
            f"status={shot.trajectory.status}"
        )
        shots.append((name, color, refined, shot))

    zones = sorted(scenario.keep_out_zones, key=lambda zone: zone.downrange)
    entry_point = (scenario.entry_downrange, scenario.entry_crossrange)
    _, _, chosen_plan, chosen_shot = shots[0]
    plot_ground_track(output_dir / "ground_track.png", zones, scenario.clearance_margin, entry_point, shots[::-1])
    plot_energy_windows(
        output_dir / "energy_windows.png",
        zones,
        chosen_plan,
        chosen_shot,
        primitives.at(chosen_plan.bank_magnitude_deg),
        scenario.entry_downrange,
    )
    plot_profiles(output_dir / "profiles.png", chosen_shot, scenario.bank_rate_deg)
    plot_arc_primitives(output_dir / "arc_primitives.png", primitives, [20, 35, 50, 65])
    plot_convergence(
        output_dir / "convergence.png", [(f"{name} ({plan.label})", shot.miss_history) for name, _, plan, shot in shots]
    )

    trajectory = chosen_shot.trajectory
    summary = {
        "candidates": [
            {
                "sides": plan.label,
                "N": plan.reversal_count,
                "d0": plan.initial_sign,
                "gaps": plan.gaps,
                "min_clearance_km": plan.min_clearance / 1e3,
            }
            for plan in plans
        ],
        "planner": planner_kind,
        "planning_time_s": planning_time,
        "initial_bank_deg": initial_bank_deg,
        "chosen": {
            "sides": chosen_plan.label,
            "N": chosen_plan.reversal_count,
            "bank_magnitude_deg": chosen_shot.bank_magnitude_deg,
            "reversal_energies_mj_per_kg": list(chosen_shot.reversal_energies / MEGAJOULE),
            "downrange_at_reversals_km": list(
                trajectory.value_at_energy(chosen_shot.reversal_energies, trajectory.downrange) / 1e3
            ),
            "miss_km": list(chosen_shot.miss),
            "miss_history_km": chosen_shot.miss_history,
            "clearance_km": [passage.clearance / 1e3 for passage in chosen_shot.passages],
            "final_time_s": trajectory.time[-1],
            "final_altitude_km": trajectory.altitude[-1] / 1e3,
            "final_speed_m_per_s": trajectory.speed[-1],
            "final_heading_deg": np.degrees(trajectory.heading[-1]),
            "status": trajectory.status,
        },
    }
    summary["total_time_s"] = time.time() - start_time
    with open(output_dir / "summary.json", "w") as file:
        json.dump(summary, file, indent=1, default=float)
    LOG.info(f"Case finished in {summary['total_time_s']:.0f} s, outputs in {output_dir}")
    return summary


def fly_closed_loop(
    scenario: ReentryScenario,
    simulator: BankReversalSimulator,
    primitives: ArcPrimitives,
    initial_bank_deg: float,
    output_dir: Path,
    start_time: float,
    bias_correction: bool = True,
    prediction_interval: int = 1,
) -> dict:
    """Fly the predictor-corrector guidance in closed loop, plot and summarize it."""
    guidance_start_time = time.time()
    guidance = PredictorCorrectorGuidance(
        primitives,
        scenario.keep_out_zones,
        scenario.clearance_margin,
        bias_correction=bias_correction,
        prediction_interval=prediction_interval,
    )
    result = guidance.fly(simulator, initial_bank_deg)
    guidance_time = time.time() - guidance_start_time
    trajectory = result.trajectory
    LOG.info(
        f"Predictor-corrector: {len(result.commands)} cycles, d0={result.initial_sign:+d}, "
        f"e_k={np.round(result.reversal_energies / MEGAJOULE, 3)} MJ/kg, miss={np.round(result.miss, 3)} km, "
        f"clearances={[round(float(passage.clearance) / 1e3, 1) for passage in result.passages]} km, "
        f"status={trajectory.status}, 3DOF predictions={result.predictions} ({guidance_time:.1f} s)"
    )

    zones = sorted(scenario.keep_out_zones, key=lambda zone: zone.downrange)
    entry_point = (scenario.entry_downrange, scenario.entry_crossrange)
    plot_closed_loop(output_dir / "ground_track.png", zones, scenario.clearance_margin, entry_point, result)
    plot_profiles(output_dir / "profiles.png", result, scenario.bank_rate_deg)
    plot_arc_primitives(output_dir / "arc_primitives.png", primitives, [20, 35, 50, 65])

    summary = {
        "planner": PREDICTOR_CORRECTOR_PLANNER,
        "planning_time_s": guidance_time,
        "initial_bank_deg": initial_bank_deg,
        "chosen": {
            "sides": side_label([passage.side for passage in result.passages]),
            "N": int(result.reversal_energies.size),
            "d0": result.initial_sign,
            "guidance_cycles": len(result.commands),
            "bias_correction": bias_correction,
            "prediction_interval": prediction_interval,
            "predictions_3dof": result.predictions,
            "prediction_flight_time_s": result.prediction_flight_time_s,
            "prediction_wall_time_s": result.prediction_wall_time_s,
            "final_bank_magnitude_deg": result.commands[-1].bank_magnitude_deg,
            "reversal_energies_mj_per_kg": list(result.reversal_energies / MEGAJOULE),
            "miss_km": list(result.miss),
            "clearance_km": [passage.clearance / 1e3 for passage in result.passages],
            "final_time_s": trajectory.time[-1],
            "final_altitude_km": trajectory.altitude[-1] / 1e3,
            "final_speed_m_per_s": trajectory.speed[-1],
            "final_heading_deg": np.degrees(trajectory.heading[-1]),
            "status": trajectory.status,
        },
        "total_time_s": time.time() - start_time,
    }
    with open(output_dir / "summary.json", "w") as file:
        json.dump(summary, file, indent=1, default=float)
    LOG.info(f"Case finished in {summary['total_time_s']:.0f} s, outputs in {output_dir}")
    return summary


def main():
    """Parse command line arguments and run the example case."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("reentry_output"), help="Directory for plots and summary")
    parser.add_argument(
        "--planner",
        choices=(ENUMERATION_PLANNER, LINEAR_PROGRAM_PLANNER, PREDICTOR_CORRECTOR_PLANNER),
        default=ENUMERATION_PLANNER,
        help="Reversal planner, side assignment enumeration, sequential LP on a relaxed bank sign or closed-loop "
        "predictor-corrector guidance",
    )
    parser.add_argument(
        "--no-bias-correction",
        action="store_true",
        help="Predictor-corrector without the 3DOF terminal bias correction, reduced model predictions only",
    )
    parser.add_argument(
        "--prediction-interval", type=int, default=1, help="Guidance cycles M between predictor-corrector 3DOF predictions"
    )
    args = parser.parse_args()
    run_case(ReentryScenario(), args.output_dir, args.planner, not args.no_bias_correction, args.prediction_interval)


if __name__ == "__main__":
    main()
