"""Implement comparison plots of guidance strategies from GuidanceRun records and benchmark rows.

Every function takes a list of runs, draws one consistent color and line style per strategy, returns
the figure and saves it when a path is given.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spacegnc.reentry.evaluation import GuidanceRun
from spacegnc.reentry.plotting import draw_zones
from spacegnc.reentry.scenario import KeepOutZone
from spacegnc.reentry.shooting import MEGAJOULE

KILOMETER = 1e3
DPI = 150

# Color and line style cycled over the runs in order
STYLES = (
    ("tab:blue", "-"),
    ("tab:orange", "--"),
    ("tab:green", "-."),
    ("tab:red", ":"),
    ("tab:purple", "-"),
    ("tab:brown", "--"),
)


def style(index: int) -> dict:
    """Color and line style of the index-th run."""
    color, linestyle = STYLES[index % len(STYLES)]
    return {"color": color, "ls": linestyle}


def _finish(fig, path: Path | None):
    """Lay out the figure and save it when a path is given."""
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=DPI)
    return fig


def _reversal_values(run: GuidanceRun, values: np.ndarray) -> np.ndarray:
    """Trajectory values at the run's reversal energies."""
    return run.trajectory.value_at_energy(run.reversal_energies, values)


def plot_ground_tracks(
    runs: list[GuidanceRun],
    zones: list[KeepOutZone],
    margin: float,
    entry_point: tuple[float, float],
    path: Path | None = None,
    inset_half_width_km: float = 15.0,
):
    """Flown ground tracks with keep-out zones, reversals, entry, target and a near-target inset."""
    fig, ax = plt.subplots(figsize=(14, 6.5))
    draw_zones(ax, zones, margin)
    ax.axhline(0, color="k", lw=0.8, ls="-.", label="reference line Y=0")

    def draw_runs(axis, with_labels: bool):
        for idx, run in enumerate(runs):
            trajectory = run.trajectory
            label = f"{run.name}: N={run.reversal_energies.size}, miss {run.miss_norm_km:.3g} km"
            axis.plot(
                trajectory.downrange / KILOMETER,
                trajectory.crossrange / KILOMETER,
                lw=2,
                label=label if with_labels else None,
                **style(idx),
            )
            axis.plot(
                _reversal_values(run, trajectory.downrange) / KILOMETER,
                _reversal_values(run, trajectory.crossrange) / KILOMETER,
                "o",
                ms=7,
                mfc="w",
                mec=style(idx)["color"],
                mew=2,
            )
        axis.plot([0], [0], "*", color="gold", mec="k", ms=16, label="target (origin)" if with_labels else None)
        axis.grid(alpha=0.3)

    draw_runs(ax, True)
    ax.plot([entry_point[0] / KILOMETER], [entry_point[1] / KILOMETER], "s", color="g", ms=9, label="entry")
    ax.plot([], [], "ko", mfc="w", label="bank reversal")
    ax.set_xlim(entry_point[0] / KILOMETER - 150, 150)
    ax.set(xlabel="Downrange X [km] (target at 0)", ylabel="Crossrange Y [km]", title="Flown 3DOF ground tracks")
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    # The inset is added after the layout, tight_layout does not handle inset axes
    fig.tight_layout()
    inset = ax.inset_axes((0.13, 0.60, 0.24, 0.33))  # empty upper-left area, clear of the entry point
    draw_runs(inset, False)
    inset.set_xlim(-inset_half_width_km, inset_half_width_km)
    inset.set_ylim(-inset_half_width_km, inset_half_width_km)
    inset.set_title("near the target (X, Y in km)", fontsize=9)
    inset.tick_params(labelsize=8)
    if path is not None:
        fig.savefig(path, dpi=DPI)
    return fig


def plot_bank_vs_energy(runs: list[GuidanceRun], path: Path | None = None):
    """Flown bank angle against specific energy, one panel per run with its reversals marked."""
    fig, axs = plt.subplots(len(runs), 1, figsize=(12, 2.4 * len(runs) + 0.6), sharex=True, squeeze=False)
    for idx, (ax, run) in enumerate(zip(axs[:, 0], runs)):
        trajectory = run.trajectory
        ax.plot(trajectory.energy / MEGAJOULE, np.degrees(trajectory.bank), lw=1.8, **style(idx))
        for reversal in run.reversal_energies:
            ax.axvline(reversal / MEGAJOULE, color="gray", ls=":")
        reversals = ", ".join(f"{energy / MEGAJOULE:.2f}" for energy in run.reversal_energies) or "none"
        ax.set_title(f"{run.name}: |σ|={run.bank_magnitude_deg:.2f} deg, reversals at {reversals} MJ/kg", fontsize=10)
        ax.set_ylabel("σ [deg]")
        ax.grid(alpha=0.3)
    axs[-1, 0].set_xlabel("Specific energy e [MJ/kg] (dotted = reversal energies)")
    return _finish(fig, path)


def plot_bank_vs_time(runs: list[GuidanceRun], path: Path | None = None):
    """Flown bank angle and, for closed-loop runs, the commanded |σ| per guidance update against time."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for idx, run in enumerate(runs):
        trajectory = run.trajectory
        ax.plot(trajectory.time, np.degrees(trajectory.bank), lw=1.4, label=run.name, **style(idx))
        if "time" in run.cycle_diagnostics:
            diagnostics = run.cycle_diagnostics
            ax.plot(diagnostics["time"], diagnostics["bank_magnitude_deg"], ".", ms=3, color=style(idx)["color"])
    ax.set(xlabel="Time t [s] (dots = commanded |σ| per guidance update)", ylabel="σ [deg]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    return _finish(fig, path)


def plot_state_vs_energy(runs: list[GuidanceRun], path: Path | None = None):
    """Altitude, speed, heading and crossrange of the flown trajectories against specific energy."""
    fig, axs = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    quantities = (
        (axs[0, 0], "altitude", KILOMETER, "Altitude h [km]"),
        (axs[0, 1], "speed", KILOMETER, "Earth-relative V [km/s]"),
        (axs[1, 0], "heading", np.pi / 180, "Heading ψ [deg]"),
        (axs[1, 1], "crossrange", KILOMETER, "Crossrange Y [km]"),
    )
    for idx, run in enumerate(runs):
        energy = run.trajectory.energy / MEGAJOULE
        for ax, key, scale, _ in quantities:
            ax.plot(energy, getattr(run.trajectory, key) / scale, lw=1.6, label=run.name, **style(idx))
    for ax, _, _, ylabel in quantities:
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    for ax in axs[1]:
        ax.set_xlabel("Specific energy e [MJ/kg]")
    axs[0, 0].legend(fontsize=9)
    return _finish(fig, path)


def plot_performance(runs: list[GuidanceRun], path: Path | None = None):
    """Bars of terminal miss (log), minimum clearance, timing breakdown and 3DOF integrations per run."""
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    names = [run.name for run in runs]
    positions = np.arange(len(runs))
    colors = [style(idx)["color"] for idx in range(len(runs))]

    def bars(ax, values, ylabel, fmt):
        rects = ax.bar(positions, values, color=colors)
        ax.bar_label(rects, labels=[fmt.format(value) for value in values], fontsize=9, padding=2)
        ax.set_ylabel(ylabel)

    bars(axs[0, 0], [max(run.miss_norm_km, 1e-4) for run in runs], "Terminal miss [km] (log)", "{:.3g}")
    axs[0, 0].set_yscale("log")
    axs[0, 0].set_ylim(top=axs[0, 0].get_ylim()[1] * 5)
    bars(axs[0, 1], [run.min_clearance_km for run in runs], "Minimum zone clearance [km]", "{:.1f}")
    bars(axs[1, 1], [run.integrations for run in runs], "3DOF integrations", "{:d}")

    ax = axs[1, 0]
    phases = sorted({phase for run in runs for phase in run.timing_s})
    bottom = np.zeros(len(runs))
    for phase, hatch in zip(phases, ("", "//", "..", "xx", "\\\\")):
        values = np.array([run.timing_s.get(phase, 0.0) for run in runs])
        ax.bar(positions, values, bottom=bottom, color=colors, hatch=hatch, edgecolor="k", lw=0.5, label=phase)
        bottom += values
    for position, total in zip(positions, bottom):
        ax.text(position, total, f"{total:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Wall time [s]")
    ax.legend(fontsize=8, title="phase (hatch)", title_fontsize=8)
    for ax in axs.flat:
        ax.set_xticks(positions, names, rotation=15, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        ax.margins(y=0.15)
    return _finish(fig, path)


def plot_cycle_diagnostics(runs: list[GuidanceRun], path: Path | None = None):
    """Per cycle bias, commands and predicted terminal miss of the closed-loop runs, None when there are none."""
    closed_loop = [(idx, run) for idx, run in enumerate(runs) if run.cycle_diagnostics]
    if not closed_loop:
        return None
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for idx, run in closed_loop:
        diagnostics = run.cycle_diagnostics
        energy = diagnostics["energy"] / MEGAJOULE
        line = style(idx)
        axs[0].plot(energy, diagnostics["bias_x_km"], marker="o", ms=3, label=f"{run.name} b_x", **line)
        axs[0].plot(energy, diagnostics["bias_y_km"], marker="s", ms=3, alpha=0.6, label=f"{run.name} b_y", **line)
        axs[1].step(energy, diagnostics["bank_magnitude_deg"], where="post", label=f"{run.name} |σ|", **line)
        # Corrected prediction of the terminal miss, reduced model plus the held 3DOF bias
        predicted = np.hypot(
            diagnostics["predicted_miss_x_km"] + diagnostics["bias_x_km"],
            diagnostics["predicted_miss_y_km"] + diagnostics["bias_y_km"],
        )
        axs[2].semilogy(energy, np.maximum(predicted, 1e-3), marker="o", ms=3, label=f"{run.name} predicted", **line)
        axs[2].axhline(max(run.miss_norm_km, 1e-3), lw=1, color=line["color"], alpha=0.7, label=f"{run.name} flown")
    reversal_ax = axs[1].twinx()
    for idx, run in closed_loop:
        energy = run.cycle_diagnostics["energy"] / MEGAJOULE
        reversal_ax.plot(energy, run.cycle_diagnostics["next_reversal"] / MEGAJOULE, "x", color=style(idx)["color"])
    reversal_ax.set_ylabel("Next reversal e_1 [MJ/kg] (x)")
    axs[0].set(ylabel="3DOF bias b [km]", title="Terminal bias z_f^3DOF - z_f^reduced held per cycle")
    axs[1].set(ylabel="|σ| command [deg]", title="Commands per guidance cycle")
    axs[2].set(ylabel="|z_f| [km] (log)", title="Predicted terminal miss per cycle vs flown miss")
    axs[2].set_xlabel("Specific energy at the guidance cycle [MJ/kg]")
    for ax in axs:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2, loc="best")
    return _finish(fig, path)


def plot_scaling(rows: list[dict], path: Path | None = None):
    """Planning time against zone count per planner and LP cell count, infeasible points as crosses."""
    fig, ax = plt.subplots(figsize=(11, 6))
    series = {}
    for row in rows:
        label = row["planner"] + ("" if row.get("cells") is None else f" K={row['cells']}")
        series.setdefault(label, []).append(row)
    for idx, (label, points) in enumerate(sorted(series.items())):
        points = sorted(points, key=lambda row: row["zones"])
        zones = np.array([row["zones"] for row in points])
        times = np.array([row["time_s"] for row in points])
        feasible = np.array([bool(row["feasible"]) for row in points])
        line = style(idx)
        ax.semilogy(zones, times, lw=1.6, label=label, **line)
        ax.semilogy(zones[feasible], times[feasible], "o", color=line["color"])
        ax.semilogy(zones[~feasible], times[~feasible], "X", ms=11, color=line["color"], mec="k")
    ax.plot([], [], "ko", label="feasible")
    ax.plot([], [], "X", color="w", mec="k", ms=11, label="infeasible")
    ax.set(xlabel="Number of keep-out zones m (synthetic)", ylabel="Planning wall time [s] (log)")
    ax.set_title("Planner scaling (predictor-corrector: one reduced guidance cycle)")
    ax.set_xticks(sorted({row["zones"] for row in rows}))
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.01, 0.5))
    return _finish(fig, path)
