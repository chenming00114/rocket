"""Test the vertical L/D bank guidance with its crossrange deadband on the coarse example simulator."""

import numpy as np
import pytest

from spacegnc.reentry.closed_loop import PerfectNavigation
from spacegnc.reentry.simulate import STRATEGIES
from spacegnc.reentry.vertical_lift import VerticalLiftGuidance, crossrange_error

from .test_closed_loop import build_simulator


def entry_navigation(simulator, bank_sign: int = 1):
    """Navigation of the entry state banked at 40 deg."""
    augmented_state = simulator.initial_augmented_state(bank_sign * np.radians(40.0))
    return PerfectNavigation(simulator.dynamics).navigate(0.0, augmented_state, bank_sign)


def test_crossrange_error_is_the_target_distance_from_the_velocity_plane():
    """Test a heading on the bearing has no error and a heading turned by +psi has R_go sin(psi)."""
    assert np.isclose(crossrange_error(-1.0e6, 0.0, 0.0), 0.0)
    assert np.isclose(crossrange_error(-1.0e6, 0.0, 0.1), 1.0e6 * np.sin(0.1))
    assert np.isclose(crossrange_error(-1.0e6, 1.0e6, -np.pi / 4), 0.0, atol=1e-6)
    # Flying parallel to the reference line 20 km to its -Y side, the target bears at +psi
    assert crossrange_error(-1.0e6, -20e3, 0.0) < 0


def test_bank_magnitude_flies_the_vertical_lift_to_drag():
    """Test cos|sigma| L/D(M) is the commanded (L/D)_v and a ratio over L/D clips to full lift up or down."""
    guidance = VerticalLiftGuidance(build_simulator(), 40.0)
    lift_to_drag = guidance.lift_to_drag(np.array([20.0]))[0]
    assert np.isclose(np.cos(guidance.bank_magnitude(0.5 * lift_to_drag, 20.0)) * lift_to_drag, 0.5 * lift_to_drag)
    assert guidance.bank_magnitude(2.0 * lift_to_drag, 20.0) == 0.0
    assert np.isclose(guidance.bank_magnitude(-2.0 * lift_to_drag, 20.0), np.pi)


def test_deadband_is_quadratic_in_speed():
    """Test Y_db(V) = c0 + c2 V^2."""
    guidance = VerticalLiftGuidance(build_simulator(), 40.0, deadband_constant=2e3, deadband_quadratic=1e-3)
    assert np.isclose(guidance.deadband(0.0), 2e3)
    assert np.isclose(guidance.deadband(5e3), 2e3 + 25e3)


def test_more_vertical_lift_to_drag_flies_further():
    """Test the predicted X_f grows with (L/D)_v, the monotonicity the root find relies on."""
    simulator = build_simulator()
    guidance = VerticalLiftGuidance(simulator, 40.0)
    navigation = entry_navigation(simulator)
    ranges = [guidance.predict_downrange(navigation, value, 1) for value in (0.4, 0.6, 0.8)]
    assert np.all(np.diff(ranges) > 0)


def test_vertical_lift_to_drag_roots_the_downrange():
    """Test the rooted (L/D)_v lands the nominal 3DOF prediction on the target downrange."""
    simulator = build_simulator()
    guidance = VerticalLiftGuidance(simulator, 40.0)
    navigation = entry_navigation(simulator)
    root = guidance.solve_vertical_lift_to_drag(navigation, 1)
    assert 0.0 < root < guidance.max_lift_to_drag
    assert abs(guidance.predict_downrange(navigation, root, 1)) < 10e3


@pytest.mark.parametrize("bank_sign", [1, -1])
def test_bank_sign_reverses_only_outside_the_deadband(bank_sign):
    """Test the sign is held inside the deadband and reversed once the error grows past it on the turning side."""
    simulator = build_simulator()
    guidance = VerticalLiftGuidance(simulator, 40.0, deadband_constant=1e3, deadband_quadratic=0.0)
    guidance.solve_vertical_lift_to_drag = lambda navigation, sign: guidance.vertical_lift_to_drag
    navigation = entry_navigation(simulator, bank_sign)
    navigation.downrange, navigation.crossrange = -1.0e6, 0.0
    navigation.heading = bank_sign * 0.5e-3
    assert guidance.update(navigation).bank_sign == bank_sign
    navigation.heading = bank_sign * 2e-3
    assert guidance.update(navigation).bank_sign == -bank_sign
    navigation.heading = -bank_sign * 2e-3
    assert guidance.update(navigation).bank_sign == bank_sign


def test_vertical_lift_is_registered():
    """Test the closed-loop registry offers the vertical L/D guidance."""
    assert "vertical-ld" in STRATEGIES
