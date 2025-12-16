import numpy as np
import pytest

from wireflux.physics.forces import JxB_force
from wireflux.physics.biot_savart import getBField
from wireflux.utils.constants import mu0, pi

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def make_straight_wire(n=50, length=1.0, axis="z"):
    s = np.linspace(-length/2, length/2, n)

    if axis == "x":
        return np.column_stack((s, np.zeros_like(s), np.zeros_like(s)))
    if axis == "y":
        return np.column_stack((np.zeros_like(s), s, np.zeros_like(s)))
    if axis == "z":
        return np.column_stack((np.zeros_like(s), np.zeros_like(s), s))

    raise ValueError("axis must be x, y, or z")

def make_circle(n=200, radius=1.0):
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack((
        radius*np.cos(theta),
        radius*np.sin(theta),
        np.zeros_like(theta)
    ))

class DummyWire:
    """Minimal stand-in for Wire with only required attributes."""
    def __init__(self, p, I=1.0, r=0.01):
        self.p = p
        self.I = I
        self.r = r

# ---------------------------------------------------------------------
# Segment-based JxB_force tests
# ---------------------------------------------------------------------

def test_zero_current_gives_zero_force():
    p = make_straight_wire()
    B = np.random.randn(*p.shape)

    F = JxB_force(p, current=0.0, B=B)

    assert np.allclose(F, 0.0)


def test_parallel_B_and_segments_gives_zero_force():
    """
    If B is parallel to dl on every segment, force must vanish.
    """
    p = make_straight_wire(axis="z")
    B = np.tile([0.0, 0.0, 1.0], (len(p), 1))

    F = JxB_force(p, current=1.0, B=B)

    assert np.allclose(F, 0.0, atol=1e-12)


def test_uniform_B_perpendicular_to_straight_wire():
    """
    Straight wire along z, uniform B along x.
    Segment force should point along +y everywhere.
    """
    p = make_straight_wire(axis="z")
    B = np.tile([1.0, 0.0, 0.0], (len(p), 1))

    F = JxB_force(p, current=1.0, B=B)

    # Force should be purely y-directed
    assert np.allclose(F[:, 0], 0.0, atol=1e-12)
    assert np.allclose(F[:, 2], 0.0, atol=1e-12)

    # Interior nodes must have positive y-force
    assert np.all(F[1:-1, 1] > 0.0)


def test_force_scales_linearly_with_current():
    p = make_straight_wire(axis="z")
    B = np.tile([0.0, 1.0, 0.0], (len(p), 1))

    F1 = JxB_force(p, current=1.0, B=B)
    F2 = JxB_force(p, current=3.0, B=B)

    assert np.allclose(F2, 3.0 * F1, rtol=1e-12)


def test_force_superposition_in_B():
    """
    J x (B1 + B2) = J x B1 + J x B2
    """
    p = make_straight_wire(axis="z")

    B1 = np.tile([1.0, 0.0, 0.0], (len(p), 1))
    B2 = np.tile([0.0, 1.0, 0.0], (len(p), 1))

    F_sum = JxB_force(p, current=1.0, B=B1 + B2)
    F_sep = (
        JxB_force(p, current=1.0, B=B1)
        + JxB_force(p, current=1.0, B=B2)
    )

    assert np.allclose(F_sum, F_sep, rtol=1e-12)


def test_closed_loop_uniform_B_has_zero_net_force():
    """
    A closed current loop in a uniform magnetic field
    must experience zero net force.
    """
    p = make_circle()
    p = np.vstack([p, p[0]])  # explicit closure
    B = np.tile([0.0, 0.0, 1.0], (len(p), 1))

    F = JxB_force(p, current=1.0, B=B)

    F_net = np.sum(F, axis=0)

    assert np.allclose(F_net, 0.0, atol=1e-5)


def test_reparameterization_invariance():
    p_uniform = make_straight_wire(n=50, axis="z")

    # Same endpoints, non-uniform spacing
    s = np.linspace(-0.5, 0.5, 50)
    s_nonuniform = s**3 / np.max(np.abs(s**3)) * 0.5

    p_nonuniform = np.column_stack((
        np.zeros_like(s_nonuniform),
        np.zeros_like(s_nonuniform),
        s_nonuniform
    ))

    B = np.tile([1.0, 0.0, 0.0], (50, 1))

    F1 = np.sum(JxB_force(p_uniform, 1.0, B), axis=0)
    F2 = np.sum(JxB_force(p_nonuniform, 1.0, B), axis=0)

    assert np.allclose(F1, F2, rtol=1e-2)


def test_force_shape_and_finiteness():
    p = make_straight_wire(n=40)
    B = np.random.randn(*p.shape)

    F = JxB_force(p, current=1.0, B=B)

    assert F.shape == p.shape
    assert np.all(np.isfinite(F))


# ---------------------------------------------------------------------
# Segment-based JxB_force in dimensional units
# ---------------------------------------------------------------------

def test_current_loop_hoop_force_direction_and_scale():
    """
    A current loop must experience an outward hoop force.
    Force density should be radial and scale with I^2 / R.
    """

    R = 1.0
    I = 1.5

    p = make_circle(n=400, radius=R)
    p = np.vstack([p, p[0]])  # closed loop

    wire = DummyWire(p, I=I, r=0.02)

    B = getBField(p, [wire])
    F_kernel = JxB_force(p, current=I, B=B)

    # Convert to physical units
    F = (mu0 / (4*pi)) * F_kernel

    # Radial unit vectors
    r_vec = p[:, :2]
    r_hat = r_vec / np.linalg.norm(r_vec, axis=1, keepdims=True)

    # Project force onto radial direction
    Fr = F[:, 0]*r_hat[:, 0] + F[:, 1]*r_hat[:, 1]

    # Hoop force must be outward on average
    assert np.mean(Fr) > 0.0

    # Scale check: order-of-magnitude only
    expected_scale = mu0 * I**2 / (8 * pi**2 * R)
    assert np.isclose(np.mean(Fr), expected_scale, rtol=0.5)

def test_parallel_wires_physical_force_density():
    """
    Two finite-length parallel wires carrying currents in the same direction
    should attract. In the central region, the force per unit length should
    agree (to order unity) with the infinite-wire analytic expression.

    This test:
    - uses physical units (mu0)
    - accounts for finite-length end effects
    - compares force per unit length in the central region only

    # NOTE:
    # These tests include both kernel-level invariants (direction, symmetry)
    # and finite-geometry physical benchmarks (with μ0 scaling).
    # Tolerances are intentionally loose to reflect finite-length effects.
    """

    I1 = I2 = 1.0
    d = 0.5
    n = 4000

    # Geometry: two parallel straight wires along z
    p1 = make_straight_wire(n=n, axis="z")
    p2 = make_straight_wire(n=n, axis="z") + np.array([d, 0.0, 0.0])

    wire1 = DummyWire(p1, I=I1)
    wire2 = DummyWire(p2, I=I2)

    # Magnetic field from wire2 evaluated on wire1
    B_on_1 = getBField(p1, [wire2])

    # Segment-based JxB force (kernel units)
    F_kernel = JxB_force(p1, current=I1, B=B_on_1)

    # Convert to physical units
    F = (mu0 / (2 * pi)) * F_kernel

    # ------------------------------------------------------------------
    # Direction checks (global)
    # ------------------------------------------------------------------
    assert np.mean(F[:, 0]) > 0.0          # attraction toward +x
    assert np.allclose(F[:, 1], 0.0, atol=1e-8)
    assert np.allclose(F[:, 2], 0.0, atol=1e-8)

    # ------------------------------------------------------------------
    # Force-per-unit-length check (central region only)
    # ------------------------------------------------------------------
    # Segment lengths
    segment_lengths = np.linalg.norm(p1[1:] - p1[:-1], axis=1)

    # Use central half of the wire to suppress end effects
    i0 = n // 4
    i1 = 3 * n // 4

    # Total force in central region
    F_central = np.sum(F[i0:i1, 0])

    # Length of central region
    L_central = np.sum(segment_lengths[i0:i1])

    # Force per unit length (numerical)
    F_per_length = F_central / L_central

    # Infinite-wire analytic result
    expected = mu0 * I1 * I2 / (2 * pi * d)

    # Order-unity agreement is all that is physically meaningful here
    assert np.isclose(F_per_length, expected, rtol=0.3)

def test_parallel_current_loops_physical_attraction():

    R = 1.0
    dz = 0.5
    I = 1.0

    loop1 = make_circle(n=400, radius=R)
    loop2 = make_circle(n=400, radius=R) + np.array([0.0, 0.0, dz])

    loop1 = np.vstack([loop1, loop1[0]])
    loop2 = np.vstack([loop2, loop2[0]])

    wire1 = DummyWire(loop1, I=I)
    wire2 = DummyWire(loop2, I=I)

    B_on_1 = getBField(loop1, [wire2])
    F_kernel = JxB_force(loop1, current=I, B=B_on_1)
    F = (mu0 / (2*pi)) * F_kernel

    # Axial attraction
    assert np.mean(F[:, 2]) > 0.0
    assert np.allclose(F[:, 0], 0.0, atol=1e-5)
    assert np.allclose(F[:, 1], 0.0, atol=1e-5)


