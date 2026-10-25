"""Define the strategy-agnostic guidance run record, its adapters and the comparison summary table.

Every guidance strategy ends in a flown full 3DOF trajectory, so the comparison only needs that
trajectory, the reversals, the terminal miss, the zone passages, a timing breakdown and evaluation
counts. A new strategy needs only an adapter filling a GuidanceRun from its own result type, the
plots and tables in compare_guidance and comparison_plots then work unchanged.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from spacegnc.reentry.closed_loop import ClosedLoopFlight
from spacegnc.reentry.planner import ZonePassage, side_label
from spacegnc.reentry.predictor_corrector import ClosedLoopResult
from spacegnc.reentry.shooting import MEGAJOULE, ShootingResult
from spacegnc.reentry.simulation import EntryTrajectory

KILOMETER = 1e3

# Columns of the summary table, header and the GuidanceRun row getter
SUMMARY_COLUMNS = (
    ("strategy", lambda run: run.name),
    ("sides", lambda run: run.sides),
    ("N", lambda run: run.reversal_energies.size),
    ("e_k [MJ/kg]", lambda run: " / ".join(f"{energy / MEGAJOULE:.3f}" for energy in run.reversal_energies)),
    ("sigma [deg]", lambda run: f"{run.bank_magnitude_deg:.2f}"),
    ("plan miss [km]", lambda run: f"{run.plan_miss_km:.3f}"),
    ("miss [km]", lambda run: f"{run.miss_norm_km:.3f}"),
    ("min clearance [km]", lambda run: f"{run.min_clearance_km:.1f}"),
    ("planning [s]", lambda run: f"{run.timing_s.get('planning', 0.0):.1f}"),
    ("shooting [s]", lambda run: f"{run.timing_s.get('shooting', 0.0):.1f}"),
    ("total [s]", lambda run: f"{run.total_time_s:.1f}"),
    ("reduced paths", lambda run: run.reduced_paths),
    ("3DOF integrations", lambda run: run.integrations),
    ("status", lambda run: run.trajectory.status),
)


@dataclass
class GuidanceRun:
    """Flown 3DOF outcome, cost and optional per cycle diagnostics of one guidance strategy."""

    name: str
    trajectory: EntryTrajectory
    reversal_energies: np.ndarray
    # Final (or constant) bank magnitude (deg)
    bank_magnitude_deg: float
    # Terminal downrange/crossrange miss [X_f, Y_f] (km)
    miss: np.ndarray
    passages: list[ZonePassage]
    # Wall time per phase (s), e.g. primitives, planning, shooting
    timing_s: dict[str, float] = field(default_factory=dict)
    reduced_paths: int = 0
    integrations: int = 0
    # Per guidance cycle arrays keyed by name, energy first, empty for open-loop strategies
    cycle_diagnostics: dict[str, np.ndarray] = field(default_factory=dict)
    # 3DOF terminal miss of the open-loop plan before the shooting (km), NaN for closed-loop strategies
    plan_miss_km: float = np.nan

    @property
    def miss_norm_km(self) -> float:
        """Terminal miss distance (km)."""
        return float(np.linalg.norm(self.miss))

    @property
    def min_clearance_km(self) -> float:
        """Smallest zone clearance of the flown track (km)."""
        return min([passage.clearance for passage in self.passages] or [np.inf]) / KILOMETER

    @property
    def sides(self) -> str:
        """Passing side label of the flown track, A above the zone and B below."""
        return side_label([passage.side for passage in self.passages])

    @property
    def total_time_s(self) -> float:
        """Sum of the timed phases (s)."""
        return float(sum(self.timing_s.values()))


def from_shooting(name: str, shot: ShootingResult, timing_s: dict, reduced_paths: int = 0, integrations: int = 0):
    """GuidanceRun of an open-loop reduced plan corrected by 3DOF shooting (enumeration, LP)."""
    return GuidanceRun(
        name=name,
        trajectory=shot.trajectory,
        reversal_energies=np.asarray(shot.reversal_energies, dtype=float),
        bank_magnitude_deg=shot.bank_magnitude_deg,
        miss=np.asarray(shot.miss, dtype=float),
        passages=shot.passages,
        timing_s=dict(timing_s),
        reduced_paths=reduced_paths,
        integrations=integrations,
        plan_miss_km=float(shot.miss_history[0]),
    )


def from_closed_loop(name: str, result: ClosedLoopResult, timing_s: dict, reduced_paths: int = 0):
    """GuidanceRun of the closed-loop predictor-corrector, with its per cycle commands as diagnostics."""
    commands = result.commands
    diagnostics = {
        "energy": np.array([command.energy for command in commands]),
        "bank_magnitude_deg": np.array([command.bank_magnitude_deg for command in commands]),
        "next_reversal": np.array(
            [np.nan if command.next_reversal is None else command.next_reversal for command in commands]
        ),
        "predicted_miss_x_km": np.array([command.predicted_miss_km[0] for command in commands]),
        "predicted_miss_y_km": np.array([command.predicted_miss_km[1] for command in commands]),
        "bias_x_km": np.array([command.bias_km[0] for command in commands]),
        "bias_y_km": np.array([command.bias_km[1] for command in commands]),
    }
    return GuidanceRun(
        name=name,
        trajectory=result.trajectory,
        reversal_energies=np.asarray(result.reversal_energies, dtype=float),
        bank_magnitude_deg=commands[-1].bank_magnitude_deg,
        miss=np.asarray(result.miss, dtype=float),
        passages=result.passages,
        timing_s=dict(timing_s),
        reduced_paths=reduced_paths,
        integrations=result.integrations + result.predictions,
        cycle_diagnostics=diagnostics,
    )


def from_flight(name: str, flight: ClosedLoopFlight, timing_s: dict) -> GuidanceRun:
    """GuidanceRun of a ClosedLoopSimulator flight, with its per update schedules and compute time as diagnostics."""
    cycles = flight.cycles
    diagnostics = {
        "energy": np.array([cycle.energy for cycle in cycles]),
        "time": np.array([cycle.time for cycle in cycles]),
        "bank_magnitude_deg": np.array([cycle.schedule.bank_magnitude_deg for cycle in cycles]),
        "next_reversal": np.array([cycle.schedule.next_reversal for cycle in cycles]),
        "predicted_miss_x_km": np.array([cycle.schedule.predicted_miss_km[0] for cycle in cycles]),
        "predicted_miss_y_km": np.array([cycle.schedule.predicted_miss_km[1] for cycle in cycles]),
        "bias_x_km": np.array([cycle.schedule.bias_km[0] for cycle in cycles]),
        "bias_y_km": np.array([cycle.schedule.bias_km[1] for cycle in cycles]),
        "compute_s": flight.compute_s,
    }
    return GuidanceRun(
        name=name,
        trajectory=flight.trajectory,
        reversal_energies=np.asarray(flight.reversal_energies, dtype=float),
        bank_magnitude_deg=cycles[-1].schedule.bank_magnitude_deg,
        miss=np.asarray(flight.miss, dtype=float),
        passages=flight.passages,
        timing_s=dict(timing_s),
        cycle_diagnostics=diagnostics,
    )


def summary_rows(runs: list[GuidanceRun], columns=SUMMARY_COLUMNS) -> list[list]:
    """Header and one row per run of the comparison table."""
    return [[header for header, _ in columns]] + [[getter(run) for _, getter in columns] for run in runs]


def write_summary(runs: list[GuidanceRun], output_dir: Path, columns=SUMMARY_COLUMNS):
    """Write the comparison table as summary.csv and summary.md."""
    rows = summary_rows(runs, columns)
    with open(output_dir / "summary.csv", "w", newline="") as file:
        csv.writer(file).writerows(rows)
    lines = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    lines.insert(1, "|" + "---|" * len(rows[0]))
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
