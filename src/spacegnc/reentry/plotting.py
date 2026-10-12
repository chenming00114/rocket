"""Implement plots of the bank-reversal planning case."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from reentry.planner import ArcPrimitives, ArcProfile, ReversalPlan
from reentry.scenario import KeepOutZone
from reentry.shooting import MEGAJOULE, ShootingResult

KILOMETER = 1e3


def _save(fig, path: Path):
    """Save and close a figure."""
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_ground_track(
    path: Path,
    zones: list[KeepOutZone],
    margin: float,
    entry_point: tuple[float, float],
    shots: list[tuple[str, str, ReversalPlan, ShootingResult]],
):
    """Ground track of reduced arc chains and shot 3DOF trajectories with keep-out zones."""
    fig, ax = plt.subplots(figsize=(13, 6))
    for idx, zone in enumerate(zones):
        center = (zone.downrange / KILOMETER, zone.crossrange / KILOMETER)
        ax.add_patch(
            Circle(center, zone.radius / KILOMETER, color="tab:red", alpha=0.25, label="keep-out zone" if idx == 0 else None)
        )
        ax.add_patch(
            Circle(
                center,
                (zone.radius + margin) / KILOMETER,
                fill=False,
                ls=":",
                color="tab:red",
                label=f"zone + {margin / KILOMETER:.0f} km margin" if idx == 0 else None,
            )
        )
        ax.text(*center, f"Z{idx + 1}", ha="center", va="center", fontsize=11, weight="bold")
    ax.axhline(0, color="k", lw=0.8, ls="-.", label="reference line Y=0")

    for name, color, plan, shot in shots:
        ax.plot(
            plan.downrange / KILOMETER,
            plan.crossrange / KILOMETER,
            "--",
            color=color,
            lw=1.5,
            label=f"{name} plan sides {plan.label}: reduced arc chain",
        )
        trajectory = shot.trajectory
        ax.plot(
            trajectory.downrange / KILOMETER,
            trajectory.crossrange / KILOMETER,
            "-",
            color=color,
            lw=2.2,
            label=f"{name}: full 3DOF (shot, N={plan.reversal_count})",
        )
        ax.plot(
            trajectory.value_at_energy(shot.reversal_energies, trajectory.downrange) / KILOMETER,
            trajectory.value_at_energy(shot.reversal_energies, trajectory.crossrange) / KILOMETER,
            "o",
            ms=8,
            mfc="w",
            mec=color,
            mew=2,
        )
        reversal_idx = [
            min(np.searchsorted(plan.energy, reversal), plan.energy.size - 1) for reversal in plan.reversal_energies
        ]
        ax.plot(
            plan.downrange[reversal_idx] / KILOMETER,
            plan.crossrange[reversal_idx] / KILOMETER,
            "x",
            color=color,
            ms=8,
            mew=2,
        )

    ax.plot([entry_point[0] / KILOMETER], [entry_point[1] / KILOMETER], "s", color="g", ms=9, label="entry (100 km alt)")
    ax.plot([0], [0], "*", color="gold", mec="k", ms=18, label="target (origin)")
    ax.plot([], [], "ko", mfc="w", label="reversal (3DOF)")
    ax.plot([], [], "kx", label="reversal (reduced plan)")
    ax.set_xlabel("Downrange X [km] (target at 0)")
    ax.set_ylabel("Crossrange Y [km] (+ = right of approach)")
    ax.set_title("Ground track in target-centred frame (crossrange axis exaggerated)")
    ax.set_xlim(entry_point[0] / KILOMETER - 150, 150)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    _save(fig, path)


def plot_energy_windows(
    path: Path, zones: list[KeepOutZone], plan: ReversalPlan, shot: ShootingResult, profile: ArcProfile, downrange0: float
):
    """Reversal energy windows against the downrange to energy map."""
    fig, ax = plt.subplots(figsize=(12, 6))
    energy = profile.energy / MEGAJOULE
    planner_downrange = downrange0 + profile.arc_length / profile.arc_length[-1] * (-downrange0)
    ax.plot(energy, planner_downrange / KILOMETER, "k--", label="planner map X(e) = X0 + S(e)·|X0|/S_f")
    ax.plot(
        shot.trajectory.energy / MEGAJOULE, shot.trajectory.downrange / KILOMETER, "tab:blue", lw=2, label="full 3DOF X(e)"
    )
    for idx, zone in enumerate(zones):
        ax.axhspan(
            (zone.downrange - zone.radius) / KILOMETER,
            (zone.downrange + zone.radius) / KILOMETER,
            color="tab:red",
            alpha=0.2,
            label="zone downrange span" if idx == 0 else None,
        )
        ax.text(energy[0] + 0.3, zone.downrange / KILOMETER, f"Z{idx + 1}", va="center", color="tab:red", weight="bold")
    for idx, gap in enumerate(sorted(set(plan.gaps))):
        lower, upper = plan.windows[gap]
        ax.axvspan(
            lower / MEGAJOULE,
            upper / MEGAJOULE,
            color="tab:green",
            alpha=0.15,
            label="reversal energy window (gap)" if idx == 0 else None,
        )
        ax.text((lower + upper) / 2 / MEGAJOULE, downrange0 / KILOMETER + 200, f"gap {gap}", ha="center", color="darkgreen")
    for idx, (planned, shot_energy) in enumerate(zip(plan.reversal_energies, shot.reversal_energies)):
        ax.axvline(planned / MEGAJOULE, color="tab:orange", ls="--", label="e_k planned (reduced)" if idx == 0 else None)
        ax.axvline(
            shot_energy / MEGAJOULE, color="tab:purple", ls="-", label="e_k after 3DOF shooting" if idx == 0 else None
        )
        ax.text(shot_energy / MEGAJOULE, -300, f"e{idx + 1}", color="tab:purple", ha="left")
    ax.set_xlabel("Specific energy e = μ/|ρ| − V²/2 [MJ/kg]")
    ax.set_ylabel("Downrange X [km]")
    ax.set_title("Reversal energy windows vs downrange")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, path)


def plot_profiles(path: Path, shot: ShootingResult, bank_rate_deg: float):
    """Altitude, speed and Mach, bank and heading histories of the shot trajectory."""
    trajectory = shot.trajectory
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    axs[0, 0].plot(trajectory.downrange / KILOMETER, trajectory.altitude / KILOMETER)
    axs[0, 0].set(xlabel="Downrange X [km]", ylabel="Altitude h [km]", title="Altitude vs downrange")

    ax = axs[0, 1]
    ax.plot(trajectory.time, trajectory.speed / KILOMETER, label="Earth-relative V")
    ax.set(xlabel="Time [s]", ylabel="V [km/s]", title="Velocity and Mach vs time")
    mach_ax = ax.twinx()
    mach_ax.plot(trajectory.time, trajectory.mach, "tab:orange", label="Mach")
    mach_ax.set_ylabel("Mach [-]", color="tab:orange")
    ax.legend(loc="upper right")

    ax = axs[1, 0]
    ax.plot(trajectory.time, np.degrees(trajectory.bank))
    for reversal_time in trajectory.value_at_energy(shot.reversal_energies, trajectory.time):
        ax.axvline(reversal_time, color="gray", ls=":")
    ax.set(
        xlabel="Time [s]",
        ylabel="Bank σ [deg]",
        title=f"Bank angle (reversals slew at {bank_rate_deg:.0f} deg/s; dotted = energy triggers)",
    )

    axs[1, 1].plot(trajectory.downrange / KILOMETER, np.degrees(trajectory.heading))
    axs[1, 1].set(xlabel="Downrange X [km]", ylabel="Heading ψ [deg] (rel. downrange)", title="Heading vs downrange")
    for ax in axs.flat:
        ax.grid(alpha=0.3)
    _save(fig, path)


def plot_arc_primitives(path: Path, primitives: ArcPrimitives, banks_deg: list[float]):
    """Arc length, curvature and heading change primitives for a few bank magnitudes."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
    for bank_deg in banks_deg:
        profile = primitives.at(bank_deg)
        energy = profile.energy / MEGAJOULE
        axs[0].plot(energy, profile.arc_length / KILOMETER, label=f"|σ|={bank_deg:.0f}°")
        axs[1].semilogy(energy, np.maximum(profile.curvature, 1e-12) * 1e6, label=f"|σ|={bank_deg:.0f}°")
        axs[2].plot(energy, np.degrees(profile.heading_change), label=f"|σ|={bank_deg:.0f}°")
    axs[1].semilogy(energy, np.abs(profile.bias_curvature) * 1e6, "k:", label="|κ_bias| (Coriolis+sphere)")
    axs[0].set_ylabel("S(e) ground range flown [km]")
    axs[1].set_ylabel("κ(e) ground curvature [1/(1000 km)]")
    axs[2].set_ylabel("Ψ(e) cumulative heading change [deg]")
    for ax, title in zip(axs, ["Range S(e)", "Curvature κ(e)", "Heading Ψ(e) (no reversal)"]):
        ax.set(xlabel="Energy e [MJ/kg]", title=title)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    _save(fig, path)


def plot_convergence(path: Path, histories: list[tuple[str, list[float]]]):
    """Terminal miss per shooting iteration."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, history in histories:
        ax.semilogy(range(len(history)), history, "o-", label=label)
    ax.set(xlabel="Shooting iteration", ylabel="Terminal miss ||(X_f, Y_f)|| [km]", title="Full-3DOF shooting convergence")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    _save(fig, path)
