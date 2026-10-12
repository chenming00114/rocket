"""Test the reentry bank-reversal planner logic with synthetic arc primitives."""

import numpy as np
import pytest

from reentry.planner import ArcProfile, BankReversalPlanner, reduced_arc_chain, side_label, zone_passages
from reentry.scenario import KeepOutZone, ReentryScenario


def make_profile(curvature: float = 2e-7, total_range: float = 1.0e6) -> ArcProfile:
    """Synthetic profile with arc length linear in energy and constant curvature."""
    energy = np.linspace(10e6, 60e6, 2001)
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


class StubPrimitives:
    """Primitives stand-in returning the synthetic profile."""

    def __init__(self, profile: ArcProfile):
        self.profile = profile
        self.bank_grid_deg = np.array([0.0, 90.0])

    def at(self, bank_magnitude_deg):
        return self.profile


@pytest.fixture
def planner():
    """Planner with two zones ahead of an entry point on the reference line."""
    zones = [KeepOutZone(-300e3, 20e3, 30e3), KeepOutZone(-700e3, -20e3, 30e3)]
    return BankReversalPlanner(StubPrimitives(make_profile()), zones, -1.0e6, 0.0, 0.0, clearance_margin=5e3)


def test_zones_sorted_by_downrange(planner):
    """Test zones are ordered from the entry point toward the target."""
    assert [zone.downrange for zone in planner.zones] == [-700e3, -300e3]


@pytest.mark.parametrize("sides, expected_gaps", [((1, 1, 1), []), ((1, -1, 1), [1, 2]), ((-1, -1, 1), [2])])
def test_switch_gaps_count_side_switches(sides, expected_gaps):
    """Test a reversal is forced in every gap where the passing side switches."""
    assert BankReversalPlanner.switch_gaps(sides) == expected_gaps
    assert side_label(sides).count("A") == sides.count(1)


def test_energy_windows_ordered_and_between_zones(planner):
    """Test the J + 1 gap windows are increasing and do not overlap the zones."""
    profile = make_profile()
    windows = planner.energy_windows(profile)
    assert len(windows) == len(planner.zones) + 1
    edges = np.ravel(windows)
    assert np.all(np.diff(edges) > 0)
    # Gap 1 lies between the first zone exit (X = -670 km) and the second zone entry (X = -330 km)
    energy_per_meter = (profile.energy[-1] - profile.energy[0]) / 1.0e6
    assert windows[1][0] > profile.energy[0] + 330e3 * energy_per_meter
    assert windows[1][1] < profile.energy[0] + 670e3 * energy_per_meter


def test_reversal_flips_curvature_direction():
    """Test a single reversal turns the reduced path back the other way."""
    profile = make_profile()
    _, crossrange_no_reversal, heading_no_reversal = reduced_arc_chain(profile, [], 1, -1.0e6, 0.0, 0.0)
    _, crossrange_reversal, heading_reversal = reduced_arc_chain(profile, [35e6], 1, -1.0e6, 0.0, 0.0)
    assert heading_no_reversal[-1] > 0 and crossrange_no_reversal[-1] > 0
    assert heading_reversal[-1] < heading_no_reversal[-1]
    assert crossrange_reversal[-1] < crossrange_no_reversal[-1]


def test_zone_passages_side_and_clearance():
    """Test passing side and clearance of a straight track past two zones."""
    downrange = np.linspace(-1.0e6, 0.0, 1001)
    crossrange = np.zeros_like(downrange)
    above, below = zone_passages(downrange, crossrange, [KeepOutZone(-5e5, -5e4, 2e4), KeepOutZone(-2e5, 6e4, 1e4)])
    assert above.side == 1 and np.isclose(above.clearance, 3e4)
    assert below.side == -1 and np.isclose(below.clearance, 5e4)


def test_solve_single_reversal_nulls_crossrange():
    """Test one reversal solves zero terminal crossrange with no zones in the way."""
    planner = BankReversalPlanner(StubPrimitives(make_profile()), [], -1.0e6, 0.0, 0.0, clearance_margin=5e3)
    plan = planner.solve_reversals(make_profile(), sides=[], gaps=[0], initial_sign=1)
    assert plan.feasible and plan.reversal_count == 1
    assert abs(plan.crossrange[-1]) < planner.CROSSRANGE_TOLERANCE


def test_example_scenario_is_consistent():
    """Test the example scenario builds an entry state at 100 km near circular speed."""
    scenario = ReentryScenario()
    frame = scenario.build_frame()
    state = scenario.initial_state(frame)
    dynamics = scenario.build_dynamics(frame)
    navigation = dynamics.navigation_outputs(state)
    assert np.isclose(navigation.altitude, scenario.entry_altitude, atol=1e-3)
    assert 7000 < navigation.speed < 8000 and navigation.mach > 20
    assert dynamics.specific_energy(state) < scenario.terminal_energy
