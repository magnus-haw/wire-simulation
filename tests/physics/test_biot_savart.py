import numpy as np
import pytest

from wireflux.physics.biot_savart import biot_savart, getBField


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def make_unit_circle(n=200, radius=1.0):
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros_like(x)
    return np.column_stack((x, y, z))


class DummyWire:
    """Minimal stand-in for Wire with only required attributes."""
    def __init__(self, p, I=1.0, r=0.01):
        self.p = p
        self.I = I
        self.r = r


# ---------------------------------------------------------------------
# biot_savart tests
# ---------------------------------------------------------------------

def test_biot_savart_center_of_circular_loop_direction():
    """
    Field at the center of a circular loop must point along +z
    (right-hand rule, normalized units).
    """
    path = make_unit_circle()
    B = biot_savart(np.array([0.0, 0.0, 0.0]), I=1.0, path=path)

    assert B[2] > 0.0
    assert np.isclose(B[0], 0.0, atol=1e-6)
    assert np.isclose(B[1], 0.0, atol=1e-6)


def test_biot_savart_center_of_circular_loop_magnitude():
    """
    Normalized magnitude for unit-radius circular loop
    should be approximately pi.
    """
    path = make_unit_circle()
    B = biot_savart(np.array([0.0, 0.0, 0.0]), I=1.0, path=path)

    assert np.isclose(np.linalg.norm(B), np.pi, rtol=2e-2)


def test_biot_savart_scales_linearly_with_current():
    """
    B-field must scale linearly with current I.
    """
    path = make_unit_circle()

    B1 = biot_savart(np.zeros(3), I=1.0, path=path)
    B2 = biot_savart(np.zeros(3), I=2.5, path=path)

    assert np.allclose(B2, 2.5 * B1, rtol=1e-6)


def test_biot_savart_along_axis():
    """
    Field evaluated along the axis should remain axial.
    """
    path = make_unit_circle()
    path = np.vstack([path, path[0]])  # closed loop

    for z in [0.2, 0.5, 1.0]:
        B = biot_savart(np.array([0.0, 0.0, z]), I=1.0, path=path)
        assert np.isclose(B[0], 0.0, atol=1e-5)
        assert np.isclose(B[1], 0.0, atol=1e-5)


def test_biot_savart_far_field_decay():
    """
    Field magnitude should decrease with distance from loop.
    """
    path = make_unit_circle()
    path = np.vstack([path, path[0]])  # closed loop

    B_near = np.linalg.norm(
        biot_savart(np.array([0.0, 0.0, 0.5]), I=1.0, path=path)
    )
    B_far = np.linalg.norm(
        biot_savart(np.array([0.0, 0.0, 2.0]), I=1.0, path=path)
    )

    assert B_far < B_near


# ---------------------------------------------------------------------
# getBField tests
# ---------------------------------------------------------------------

def test_biot_savart_loop_center_field():
    R = 1.0
    I = 1.0

    p_loop = make_unit_circle(n=600, radius=R)
    p_loop = np.vstack([p_loop, p_loop[0]])  # explicit closure

    wire = DummyWire(p_loop, I=I, r=0.02)

    B = biot_savart(
        p=np.array([0.0, 0.0, 0.0]),
        I=wire.I,
        path=wire.p,
        delta=wire.r
    )

    # Direction
    assert np.allclose(B[:2], 0.0, atol=1e-6)
    assert B[2] > 0.0

    # Magnitude (normalized)
    assert np.isclose(B[2], np.pi, rtol=0.05)


def test_getBField_single_wire_matches_biot_savart():
    """
    getBField for a single wire must match direct biot_savart calls.
    """
    path = make_unit_circle()
    wire = DummyWire(make_unit_circle())

    probe = make_unit_circle(n=50)
    B1 = getBField(probe, [wire])

    B2 = np.array([
        biot_savart(p, wire.I, wire.p, delta=wire.r)
        for p in probe
    ])

    assert np.allclose(B1, B2, rtol=1e-6)


def test_getBField_superposition():
    """
    getBField must satisfy linear superposition.
    """
    path = make_unit_circle()
    wire1 = DummyWire(make_unit_circle(), I=1.0)
    wire2 = DummyWire(make_unit_circle(), I=2.0)

    probe = np.array([[0.0, 0.0, 0.0]])

    B_sum = getBField(probe, [wire1, wire2])[0]
    B_sep = (
        biot_savart(probe[0], wire1.I, wire1.p)
        + biot_savart(probe[0], wire2.I, wire2.p)
    )

    assert np.allclose(B_sum, B_sep, rtol=1e-6)
