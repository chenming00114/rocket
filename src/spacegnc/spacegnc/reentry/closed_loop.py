"""Implement the closed-loop entry simulation platform for testing reentry guidance strategies.

The truth plant is the 3DOF entry vehicle of a BankReversalSimulator flying its own truth atmosphere,
integrated in time with the rate limited bank. Every guidance period of simulated time the navigation
turns the truth state into a Navigation, the guidance answers with a BankSchedule from its own nominal
model, and the plant flies that schedule, the bank magnitude, the current sign and the upcoming reversal
energies, until the next update. A bank command change is flown through the bank rate limit like a
reversal. A new strategy only implements Guidance.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from spacegnc.logconfig import create_logger
from spacegnc.reentry.planner import ZonePassage, zone_passages
from spacegnc.reentry.scenario import KeepOutZone
from spacegnc.reentry.simulation import BankReversalSimulator, EntryTrajectory

LOG = create_logger(__name__)


@dataclass
class Navigation:
    """Navigated vehicle state handed to the guidance at one update."""

    time: float
    # Position and Earth-relative velocity in the target-centred frame [r, v]
    state: np.ndarray
    # Flown bank angle (rad) and the commanded bank sign, 0 before the first update
    bank: float
    bank_sign: int
    downrange: float
    crossrange: float
    heading: float
    flight_path_angle: float
    energy: float
    altitude: float
    mach: float
    speed: float

    @property
    def planar(self) -> tuple[float, float, float, float]:
        """Reduced model state (X, Y, psi, e) of the planners."""
        return self.downrange, self.crossrange, self.heading, self.energy

    @property
    def augmented_state(self) -> np.ndarray:
        """[r, v, sigma] state of the BankReversalSimulator segments."""
        return np.append(self.state, self.bank)


class PerfectNavigation:
    """Navigation from the truth state, with Mach from the navigation's own nominal speed of sound.

    A navigation error model overrides navigate and perturbs the returned Navigation.
    """

    def __init__(self, dynamics):
        """Initialize with the nominal entry dynamics providing the frame and speed of sound."""
        self.dynamics = dynamics

    def navigate(self, time: float, augmented_state: np.ndarray, bank_sign: int) -> Navigation:
        """Navigation of a [r, v, sigma] truth state."""
        frame, state = self.dynamics.frame, np.asarray(augmented_state[:6], dtype=float)
        outputs = self.dynamics.navigation_outputs(state)
        downrange, crossrange = frame.surface_range(state[:3][None, :])
        flight_path_angle, heading = frame.flight_path_angle_and_heading(state[:3], state[3:6])
        return Navigation(
            time=time,
            state=state,
            bank=float(augmented_state[6]),
            bank_sign=bank_sign,
            downrange=float(downrange[0]),
            crossrange=float(crossrange[0]),
            heading=float(heading),
            flight_path_angle=float(flight_path_angle),
            energy=float(self.dynamics.specific_energy(state)),
            altitude=float(outputs.altitude),
            mach=float(outputs.mach),
            speed=float(outputs.speed),
        )


@dataclass
class BankSchedule:
    """Bank command of one guidance update, flown until the next one."""

    bank_magnitude_deg: float
    # Bank sign from now on, a change from the flown sign is a reversal right away
    bank_sign: int
    # Upcoming reversal energies (J/kg), the sign flips at each one passed
    reversal_energies: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # Terminal miss [X_f, Y_f] (km) predicted by the guidance and its 3DOF bias over its reduced model (km)
    predicted_miss_km: tuple[float, float] = (np.nan, np.nan)
    bias_km: tuple[float, float] = (0.0, 0.0)

    @property
    def next_reversal(self) -> float:
        """First upcoming reversal energy, NaN when the sign is held."""
        return float(self.reversal_energies[0]) if len(self.reversal_energies) else np.nan


class Guidance(ABC):
    """Bank guidance updated from the navigated state with its own nominal model."""

    @abstractmethod
    def update(self, navigation: Navigation) -> BankSchedule:
        """Bank schedule from the navigated state, the first call has navigation.bank_sign 0 and picks the sign."""


@dataclass
class GuidanceCycle:
    """Navigated time, energy, schedule and compute time of one guidance update."""

    time: float
    energy: float
    schedule: BankSchedule
    compute_s: float


@dataclass
class ClosedLoopFlight:
    """Truth trajectory and guidance history of one closed-loop simulation."""

    initial_sign: int
    reversal_energies: np.ndarray
    cycles: list[GuidanceCycle]
    # Terminal miss [X_f, Y_f] (km)
    miss: np.ndarray
    trajectory: EntryTrajectory
    passages: list[ZonePassage]

    @property
    def compute_s(self) -> np.ndarray:
        """Guidance compute time per update (s)."""
        return np.array([cycle.compute_s for cycle in self.cycles])


class ClosedLoopSimulator:
    """Fly a guidance in closed loop on a truth plant with one update per guidance period of simulated time."""

    def __init__(
        self,
        plant: BankReversalSimulator,
        zones: list[KeepOutZone],
        guidance_period: float = 5.0,
        navigation: PerfectNavigation | None = None,
    ):
        """Initialize with the truth plant, the keep-out zones, the guidance period (s) and the navigation."""
        self.plant = plant
        self.zones = sorted(zones, key=lambda zone: zone.downrange)
        self.guidance_period = guidance_period
        self.navigation = navigation or PerfectNavigation(plant.dynamics)

    def _update(self, guidance: Guidance, navigation: Navigation, cycles: list[GuidanceCycle]) -> BankSchedule:
        """Timed guidance update logged as one cycle."""
        start_time = perf_counter()
        schedule = guidance.update(navigation)
        cycles.append(GuidanceCycle(navigation.time, navigation.energy, schedule, perf_counter() - start_time))
        return schedule

    def fly(self, guidance: Guidance) -> ClosedLoopFlight:
        """Fly from the plant's entry state to the terminal energy or a limit event."""
        plant, cycles, reversals = self.plant, [], []
        time, augmented_state = 0.0, np.append(plant.initial_state, 0.0)
        schedule = self._update(guidance, self.navigation.navigate(time, augmented_state, 0), cycles)
        bank_sign = initial_sign = schedule.bank_sign
        # The entry starts at the first commanded bank
        augmented_state[6] = bank_sign * np.radians(schedule.bank_magnitude_deg)
        times, states = [], []
        while True:
            energy = plant.dynamics.specific_energy(augmented_state[:6])
            if schedule.bank_sign != bank_sign:
                reversals.append(energy)
                bank_sign = schedule.bank_sign
            upcoming = np.sort(np.asarray(schedule.reversal_energies, dtype=float))
            upcoming = upcoming[upcoming > energy]
            segment_time, segment_states, status = plant.run_segment(
                augmented_state,
                time,
                schedule.bank_magnitude_deg,
                upcoming,
                bank_sign,
                plant.terminal_energy,
                stop_time=time + self.guidance_period,
            )
            times.append(segment_time if not times else segment_time[1:])
            states.append(segment_states if not states else segment_states[1:])
            time, augmented_state = segment_time[-1], segment_states[-1]
            passed = upcoming[upcoming < plant.dynamics.specific_energy(augmented_state[:6])]
            reversals.extend(passed)
            bank_sign *= (-1) ** passed.size
            if status is not None:
                break
            schedule = self._update(guidance, self.navigation.navigate(time, augmented_state, bank_sign), cycles)

        trajectory = plant.post_process(np.concatenate(times), np.vstack(states), status)
        LOG.info(
            f"Closed loop {status}: {len(cycles)} guidance cycles, reversals at {np.round(np.asarray(reversals) / 1e6, 3)} "
            f"MJ/kg, miss [{trajectory.downrange[-1] / 1e3:.2f}, {trajectory.crossrange[-1] / 1e3:.2f}] km"
        )
        return ClosedLoopFlight(
            initial_sign=initial_sign,
            reversal_energies=np.asarray(reversals),
            cycles=cycles,
            miss=np.array([trajectory.downrange[-1], trajectory.crossrange[-1]]) / 1e3,
            trajectory=trajectory,
            passages=zone_passages(trajectory.downrange, trajectory.crossrange, self.zones),
        )
