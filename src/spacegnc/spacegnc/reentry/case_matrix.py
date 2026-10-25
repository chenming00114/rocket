"""Run named truth model cases and a seeded Monte Carlo of the closed-loop guidance strategies.

The guidance always keeps its nominal model, the arc primitives and the spherical, US76, Mach table
3DOF simulator, only the truth plant changes per case. Suites:

    ladder        the fidelity ladder, one truth model upgrade at a time and all of them together
    dispersions   one-at-a-time dispersions on the nominal truth model
    monte-carlo   seeded combined dispersions, with miss percentiles and the keep-out violation rate

Every run is pickled under <output-dir>/runs/ as it completes and skipped when rerun, and the summary
tables and plots are rewritten from the pickles, so an interrupted suite resumes where it stopped.

    python -m spacegnc.reentry.case_matrix --suite ladder dispersions --strategies pc-bias lp-open-loop \\
        --output-dir case_matrix
"""

import argparse
import functools
import pickle
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np

from spacegnc.logconfig import create_logger
from spacegnc.reentry.closed_loop import ClosedLoopSimulator, ErroneousNavigation, NavigationErrors, PerfectNavigation
from spacegnc.reentry.compare_guidance import ComparisonCase
from spacegnc.reentry.evaluation import GuidanceRun, from_flight, write_summary
from spacegnc.reentry.scenario import ReentryScenario
from spacegnc.reentry.simulate import CLOSED_LOOP_COLUMNS, STRATEGIES, write_plots
from spacegnc.reentry.truth_atmosphere import ScaledDensityAtmosphere, SinusoidalDensityAtmosphere
from spacegnc.reentry.truth_model import (
    BankActuator,
    EntryDispersion,
    GaussianJetWind,
    TruthAeroModel,
    TruthModelConfig,
    build_truth_plant,
)

LOG = create_logger(__name__)

KILOMETER = 1e3
# Altitude above which the density dispersions act (m)
DENSITY_ALTITUDE = 35e3
# Core altitude and thickness of the dispersed wind jets (m), PLACEHOLDER profile inside the flown altitudes
JET_ALTITUDE = 45e3
JET_WIDTH = 8e3


def density_scale(scale: float) -> Callable:
    """Density scaled by f above 35 km with a consistent temperature."""
    return functools.partial(
        ScaledDensityAtmosphere, density_scale=scale, scale_above=DENSITY_ALTITUDE, temperature_consistent=True
    )


def density_wave(amplitude: float) -> Callable:
    """Density wave of relative amplitude a and 10 km wavelength above 35 km with a consistent temperature."""
    return functools.partial(
        SinusoidalDensityAtmosphere, amplitude=amplitude, scale_above=DENSITY_ALTITUDE, temperature_consistent=True
    )


def aero(**dispersions) -> Callable:
    """Trim aero model with dispersions."""
    return functools.partial(TruthAeroModel, **dispersions)


def jet(downrange: float = 0.0, crossrange: float = 0.0) -> GaussianJetWind:
    """Wind jet with peak downrange (tail positive) and crossrange components (m/s)."""
    return GaussianJetWind(downrange, crossrange, JET_ALTITUDE, JET_WIDTH)


HIGH_FIDELITY = TruthModelConfig(
    gravity="j2", altitude="geodetic", aero=aero(rarefied=True), actuator=BankActuator(acceleration_limit_deg=6.0)
)

# Fidelity ladder, each upgrade on its own and then all of them
LADDER = {
    "nominal": TruthModelConfig(),
    "j2-gravity": TruthModelConfig(gravity="j2"),
    "geodetic-altitude": TruthModelConfig(altitude="geodetic"),
    "trim-aero": TruthModelConfig(aero=aero()),
    "rarefied-aero": TruthModelConfig(aero=aero(rarefied=True)),
    "bank-actuator": TruthModelConfig(actuator=BankActuator()),
    "high-fidelity": HIGH_FIDELITY,
}

# One-at-a-time dispersions on the nominal truth model
DISPERSIONS = {
    "density+20%>35km": TruthModelConfig(atmosphere=density_scale(1.2)),
    "density-20%>35km": TruthModelConfig(atmosphere=density_scale(0.8)),
    "density-wave+10%": TruthModelConfig(atmosphere=density_wave(0.1)),
    "density-wave-10%": TruthModelConfig(atmosphere=density_wave(-0.1)),
    "head-wind-50": TruthModelConfig(wind=jet(downrange=-50.0)),
    "tail-wind-50": TruthModelConfig(wind=jet(downrange=50.0)),
    "cross-wind-50": TruthModelConfig(wind=jet(crossrange=50.0)),
    "C_L+10%": TruthModelConfig(aero=aero(lift_scale=1.1)),
    "C_L-10%": TruthModelConfig(aero=aero(lift_scale=0.9)),
    "C_D+10%": TruthModelConfig(aero=aero(drag_scale=1.1)),
    "C_D-10%": TruthModelConfig(aero=aero(drag_scale=0.9)),
    "L/D+10%": TruthModelConfig(aero=aero(lift_to_drag_scale=1.1)),
    "L/D-10%": TruthModelConfig(aero=aero(lift_to_drag_scale=0.9)),
    "gamma+0.1deg": TruthModelConfig(entry=EntryDispersion(flight_path_angle_deg=0.1)),
    "gamma-0.1deg": TruthModelConfig(entry=EntryDispersion(flight_path_angle_deg=-0.1)),
    "V+20m/s": TruthModelConfig(entry=EntryDispersion(speed=20.0)),
    "V-20m/s": TruthModelConfig(entry=EntryDispersion(speed=-20.0)),
    "Y+5km": TruthModelConfig(entry=EntryDispersion(crossrange=5e3)),
    "Y-5km": TruthModelConfig(entry=EntryDispersion(crossrange=-5e3)),
    "bank-rate/2": TruthModelConfig(bank_rate_deg=6.0),
    "nav-h+500m": TruthModelConfig(navigation=NavigationErrors(altitude_bias=500.0)),
    "nav-h-500m": TruthModelConfig(navigation=NavigationErrors(altitude_bias=-500.0)),
}


def monte_carlo_cases(count: int, seed: int) -> dict[str, TruthModelConfig]:
    """Combined Gaussian dispersions with the one-at-a-time magnitudes as 3 sigma, on the high fidelity model."""
    rng = np.random.default_rng(seed)

    def sigma3(value):
        return value / 3 * rng.standard_normal()

    cases = {}
    for index in range(count):
        # Density scale and wave combined as one profile: the wave rides on the scaled density
        scale, amplitude = 1 + sigma3(0.2), sigma3(0.1)
        phase = rng.uniform(0, 2 * np.pi)

        def atmosphere(nominal, scale=scale, amplitude=amplitude, phase=phase):
            scaled = density_scale(scale)(nominal)
            return SinusoidalDensityAtmosphere(
                scaled, amplitude, phase=phase, scale_above=DENSITY_ALTITUDE, temperature_consistent=True
            )

        cases[f"mc-{seed}-{index:03d}"] = replace(
            HIGH_FIDELITY,
            atmosphere=atmosphere,
            wind=jet(sigma3(50.0), sigma3(50.0)),
            aero=aero(
                rarefied=True,
                lift_scale=1 + sigma3(0.1),
                drag_scale=1 + sigma3(0.1),
                lift_to_drag_scale=1 + sigma3(0.1),
            ),
            entry=EntryDispersion(sigma3(0.1), sigma3(20.0), sigma3(5e3)),
            navigation=NavigationErrors(altitude_bias=sigma3(500.0), seed=int(rng.integers(2**31))),
        )
    return cases


def fly_case(case: ComparisonCase, config: TruthModelConfig, strategy: str, options) -> GuidanceRun:
    """Fly a strategy on the truth plant of a configuration, navigation and guidance on the nominal model."""
    scenario = case.scenario
    plant = build_truth_plant(scenario, case.simulator, config)
    nominal_dynamics = case.simulator.dynamics
    navigation = (
        ErroneousNavigation(nominal_dynamics, config.navigation)
        if config.navigation is not None
        else PerfectNavigation(nominal_dynamics)
    )
    simulator = ClosedLoopSimulator(plant, scenario.keep_out_zones, options.guidance_period, navigation)
    start_time = time.time()
    guidance = STRATEGIES[strategy](case, options)
    setup_time = time.time() - start_time
    flight = simulator.fly(guidance)
    timing = {"setup": setup_time, "closed loop": time.time() - start_time - setup_time}
    return from_flight(strategy, flight, timing)


def run_path(output_dir: Path, suite: str, name: str, strategy: str) -> Path:
    """Pickle path of one run."""
    safe_name = name.replace("/", "_per_").replace("%", "pct").replace(">", "_above_")
    return output_dir / "runs" / suite / f"{safe_name}__{strategy}.pkl"


def run_suite(case: ComparisonCase, suite: str, cases: dict, options) -> list[tuple[str, GuidanceRun]]:
    """Fly or load every case and strategy of a suite, pickling each run as it completes."""
    results = []
    for name, config in cases.items():
        for strategy in options.strategies:
            path = run_path(options.output_dir, suite, name, strategy)
            if path.exists():
                run = pickle.loads(path.read_bytes())
            else:
                LOG.info(f"{suite}: flying {name} with {strategy}")
                run = fly_case(case, config, strategy, options)
                run.name = f"{strategy} {name}"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(pickle.dumps(run))
                LOG.info(f"{run.name}: miss {run.miss_norm_km:.3f} km in {run.total_time_s:.0f} s")
            results.append((name, run))
            write_suite_summary(suite, results, options.output_dir)
    return results


def write_suite_summary(suite: str, results: list[tuple[str, GuidanceRun]], output_dir: Path):
    """Write the suite's table, one row per case and strategy."""
    suite_dir = output_dir / suite
    suite_dir.mkdir(parents=True, exist_ok=True)
    write_summary([run for _, run in results], suite_dir, CLOSED_LOOP_COLUMNS)


def monte_carlo_statistics(runs: list[GuidanceRun]) -> dict:
    """Miss percentiles, keep-out violation probability and constraint percentiles of Monte Carlo runs."""
    miss = np.array([run.miss_norm_km for run in runs])
    violated = np.array([any(passage.clearance < 0 for passage in run.passages) for run in runs])
    peaks = {
        "peak load [g]": [np.max(run.trajectory.load_factor) for run in runs],
        "peak heat rate [kW/m^2]": [np.max(run.trajectory.heat_rate) / 1e3 for run in runs],
        "heat load [MJ/m^2]": [run.trajectory.heat_load / 1e6 for run in runs],
    }
    statistics = {
        "runs": len(runs),
        "miss 50% [km]": float(np.percentile(miss, 50)),
        "miss 99% [km]": float(np.percentile(miss, 99)),
        "miss max [km]": float(np.max(miss)),
        "violation probability": float(np.mean(violated)),
    }
    for label, values in peaks.items():
        statistics[f"{label} 50%"] = float(np.percentile(values, 50))
        statistics[f"{label} 99%"] = float(np.percentile(values, 99))
    return statistics


def main():
    """Parse command line arguments and run the requested suites."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", nargs="+", choices=("ladder", "dispersions", "monte-carlo"), default=["ladder"])
    parser.add_argument("--strategies", nargs="+", choices=tuple(STRATEGIES), default=["pc-bias", "lp-open-loop"])
    parser.add_argument("--cases", nargs="+", help="Only these named cases of the ladder and dispersion suites")
    parser.add_argument("--output-dir", type=Path, default=Path("case_matrix"), help="Directory for the outputs")
    parser.add_argument("--guidance-period", type=float, default=5.0, help="Simulated seconds between updates")
    parser.add_argument("--replan-interval", type=int, default=12, help="Updates between closed-loop arc LP re-plans")
    parser.add_argument("--no-reshoot", action="store_true", help="Fly the closed-loop arc LP without 3DOF re-shooting")
    parser.add_argument("--runs", type=int, default=10, help="Monte Carlo runs")
    parser.add_argument("--seed", type=int, default=0, help="Monte Carlo seed")
    options = parser.parse_args()

    scenario = ReentryScenario()
    case = ComparisonCase.build(scenario)
    for suite in options.suite:
        cases = {"ladder": LADDER, "dispersions": DISPERSIONS}.get(suite) or monte_carlo_cases(options.runs, options.seed)
        if options.cases and suite != "monte-carlo":
            cases = {name: config for name, config in cases.items() if name in options.cases}
        results = run_suite(case, suite, cases, options)
        for strategy in options.strategies:
            runs = [run for _, run in results if run.name.startswith(f"{strategy} ")]
            if suite == "monte-carlo":
                statistics = monte_carlo_statistics(runs)
                lines = [f"| {key} | {value:.3f} |" for key, value in statistics.items()]
                text = f"## {strategy}\n\n| statistic | value |\n|---|---|\n" + "\n".join(lines) + "\n"
                (options.output_dir / suite / f"statistics_{strategy}.md").write_text(text)
                LOG.info("\n" + text)
            elif runs:
                write_plots(runs, scenario, options.output_dir / suite, strategy)


if __name__ == "__main__":
    main()
