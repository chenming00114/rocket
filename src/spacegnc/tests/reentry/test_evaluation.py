"""Test the guidance run adapters, summary table and comparison plots on synthetic records."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from spacegnc.reentry import comparison_plots  # noqa: E402
from spacegnc.reentry.compare_guidance import STRATEGIES, load_scaling_rows  # noqa: E402
from spacegnc.reentry.evaluation import from_closed_loop, from_shooting, summary_rows, write_summary  # noqa: E402
from spacegnc.reentry.planner import ZonePassage  # noqa: E402
from spacegnc.reentry.predictor_corrector import ClosedLoopResult, GuidanceCommand  # noqa: E402
from spacegnc.reentry.scenario import KeepOutZone  # noqa: E402
from spacegnc.reentry.shooting import ShootingResult  # noqa: E402
from spacegnc.reentry.simulation import EntryTrajectory  # noqa: E402

ZONES = [KeepOutZone(-500e3, 30e3, 20e3)]


def make_trajectory(crossrange_scale: float = 20e3, samples: int = 200) -> EntryTrajectory:
    """Straight-in synthetic trajectory with a bank reversal at its middle energy."""
    time = np.linspace(0.0, 1000.0, samples)
    energy = np.linspace(30e6, 60e6, samples)
    downrange = np.linspace(-1.0e6, 0.0, samples)
    bank = np.radians(np.where(energy < 45e6, 50.0, -50.0))
    zeros = np.zeros(samples)
    return EntryTrajectory(
        time=time,
        position=np.zeros((samples, 3)),
        velocity=np.zeros((samples, 3)),
        bank=bank,
        altitude=np.linspace(100e3, 30e3, samples),
        speed=np.linspace(7e3, 0.7e3, samples),
        mach=np.linspace(25.0, 2.5, samples),
        energy=energy,
        downrange=downrange,
        crossrange=crossrange_scale * np.sin(np.pi * time / time[-1]),
        flight_path_angle=zeros,
        heading=np.radians(np.linspace(5.0, -5.0, samples)),
        lift=zeros,
        drag=zeros,
        arc_length=downrange - downrange[0],
        status="terminal_energy",
    )


def make_runs():
    """One open-loop and one closed-loop synthetic run through the adapters."""
    passages = [ZonePassage(clearance=12e3, side=-1, crossrange_at_zone=0.0)]
    shot = ShootingResult(
        bank_magnitude_deg=50.0,
        reversal_energies=np.array([45e6]),
        initial_sign=1,
        miss=np.array([0.003, -0.002]),
        miss_history=[10.0, 0.004],
        step_lengths=[1.0],
        trajectory=make_trajectory(),
        passages=passages,
    )
    commands = [
        GuidanceCommand(
            energy=energy,
            bank_magnitude_deg=50.0 + idx,
            next_reversal=45e6 if energy < 45e6 else None,
            predicted_miss_km=(1.0, -0.5),
            predicted_clearance_km=12.0,
            bias_km=(-0.2 * idx, 0.1 * idx),
        )
        for idx, energy in enumerate(np.linspace(30e6, 58e6, 6))
    ]
    closed_loop = ClosedLoopResult(
        initial_sign=1,
        reversal_energies=np.array([45e6]),
        commands=commands,
        miss=np.array([0.15, 0.1]),
        trajectory=make_trajectory(15e3),
        passages=passages,
        integrations=6,
        predictions=10,
    )
    return [
        from_shooting("enumeration", shot, {"planning": 3.0, "shooting": 2.0}, reduced_paths=100, integrations=9),
        from_closed_loop("pc-bias M=1", closed_loop, {"planning": 4.0}, reduced_paths=50),
    ]


def test_adapters_keep_the_outcome_and_counts():
    """Test both adapters carry the trajectory, reversals, miss, clearance, timing and counts over."""
    open_loop, closed_loop = make_runs()
    assert open_loop.sides == "B" and np.isclose(open_loop.min_clearance_km, 12.0)
    assert np.isclose(open_loop.miss_norm_km, np.hypot(0.003, 0.002))
    assert open_loop.total_time_s == 5.0 and open_loop.integrations == 9 and not open_loop.cycle_diagnostics
    # Closed-loop integrations count the flown segments and the 3DOF bias predictions
    assert closed_loop.integrations == 16 and closed_loop.bank_magnitude_deg == 55.0
    diagnostics = closed_loop.cycle_diagnostics
    assert diagnostics["energy"].size == 6 and np.isnan(diagnostics["next_reversal"][-1])
    assert np.allclose(diagnostics["bias_x_km"], -0.2 * np.arange(6))


def test_summary_table_files(tmp_path):
    """Test the CSV and markdown tables hold a header and one row per run."""
    runs = make_runs()
    rows = summary_rows(runs)
    assert len(rows) == 3 and rows[1][0] == "enumeration" and rows[1][2] == 1
    write_summary(runs, tmp_path)
    assert len((tmp_path / "summary.csv").read_text().strip().splitlines()) == 3
    markdown = (tmp_path / "summary.md").read_text().splitlines()
    assert len(markdown) == 4 and markdown[1].startswith("|---|")


@pytest.mark.parametrize(
    "plot, expected_axes",
    [
        (lambda runs, path: comparison_plots.plot_ground_tracks(runs, ZONES, 5e3, (-1.0e6, 0.0), path), 2),
        (comparison_plots.plot_bank_vs_energy, 2),
        (comparison_plots.plot_state_vs_energy, 4),
        (comparison_plots.plot_performance, 4),
        (comparison_plots.plot_cycle_diagnostics, 4),
    ],
)
def test_comparison_plots_draw_every_run(plot, expected_axes, tmp_path):
    """Test each comparison plot saves a figure with lines or bars for the runs."""
    path = tmp_path / "plot.png"
    figure = plot(make_runs(), path)
    assert path.stat().st_size > 0
    axes = figure.axes + [child for ax in figure.axes for child in ax.child_axes]  # include the ground-track inset
    assert len(axes) == expected_axes
    assert all(ax.lines or ax.patches for ax in axes)
    plt.close(figure)


def test_cycle_diagnostics_skipped_without_closed_loop_runs():
    """Test the per cycle plot returns None when no run has diagnostics."""
    assert comparison_plots.plot_cycle_diagnostics(make_runs()[:1]) is None


def test_scaling_plot_marks_infeasible_points(tmp_path):
    """Test merged benchmark rows draw one series per planner and cell count with infeasible crosses."""
    rows = [
        {"planner": "enumeration", "zones": m, "samples": 10, "cells": None, "time_s": 2.0**m, "feasible": True}
        for m in (1, 2)
    ] + [{"planner": "lp", "zones": m, "samples": 10, "cells": 20, "time_s": 1.0, "feasible": m < 2} for m in (1, 2)]
    path = tmp_path / "benchmark.json"
    path.write_text('{"scaling": ' + __import__("json").dumps(rows) + "}")
    loaded = load_scaling_rows([path, path])
    assert len(loaded) == 4
    figure = comparison_plots.plot_scaling(loaded)
    markers = [line.get_marker() for line in figure.axes[0].lines if line.get_xdata().size]
    assert markers.count("X") == 1
    plt.close(figure)


def test_every_strategy_is_registered():
    """Test the runner registry covers the four compared strategies."""
    assert set(STRATEGIES) == {"enumeration", "lp", "pc", "pc-bias"}
