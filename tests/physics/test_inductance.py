import numpy as np
from scipy.special import ellipk, ellipe

from wireflux.physics.inductance import inductance
from wireflux.utils.constants import mu0, pi

def make_straight_wire(n=400, length=1.0, axis="z"):
    s = np.linspace(-length/2, length/2, n)
    if axis == "z":
        return np.column_stack((np.zeros_like(s), np.zeros_like(s), s))
    if axis == "x":
        return np.column_stack((s, np.zeros_like(s), np.zeros_like(s)))
    if axis == "y":
        return np.column_stack((np.zeros_like(s), s, np.zeros_like(s)))
    raise ValueError


def make_circle(n=400, radius=1.0, z=0.0):
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack((
        radius*np.cos(theta),
        radius*np.sin(theta),
        z*np.ones_like(theta)
    ))


def test_inductance_parallel_wires_asymptotic():

    L = 10.0
    d = 0.5

    p1 = make_straight_wire(n=600, length=L, axis="z")
    p2 = make_straight_wire(n=600, length=L, axis="z") + np.array([d, 0.0, 0.0])

    M_num = inductance(p1, p2, rwire=1e-3)

    # Correct finite-length analytic expression (Grover)
    M_expected = (mu0 / (2 * pi)) * (
        L * np.log((L + np.sqrt(L**2 + d**2)) / d)
        - np.sqrt(L**2 + d**2)
        + d
    )

    assert np.isclose(M_num, M_expected, rtol=0.15)

def test_inductance_coaxial_loops():
    R = 1.0
    z = 0.5

    p1 = make_circle(n=600, radius=R, z=0.0)
    p2 = make_circle(n=600, radius=R, z=z)

    M_num = inductance(p1, p2, rwire=1e-3)

    k2 = (4*R*R) / ((2*R)**2 + z*z)
    k = np.sqrt(k2)

    K = ellipk(k2)
    E = ellipe(k2)

    M_expected = mu0 * R * ((2/k - k)*K - (2/k)*E)

    assert np.isclose(M_num, M_expected, rtol=0.05)

def test_inductance_symmetry():

    p1 = make_circle(radius=1.0)
    p2 = make_circle(radius=0.5, z=0.3)

    M12 = inductance(p1, p2)
    M21 = inductance(p2, p1)

    assert np.isclose(M12, M21, rtol=1e-10)

def test_inductance_scales_with_length():
    d = 0.5
    L1 = 5.0
    L2 = 10.0

    p1a = make_straight_wire(length=L1)
    p2a = make_straight_wire(length=L1) + np.array([d, 0.0, 0.0])

    p1b = make_straight_wire(length=L2)
    p2b = make_straight_wire(length=L2) + np.array([d, 0.0, 0.0])

    M1 = inductance(p1a, p2a)
    M2 = inductance(p1b, p2b)

    # Analytic finite-length ratio (Grover)
    def M_analytic(L):
        return (
            L * np.log((L + np.sqrt(L**2 + d**2)) / d)
            - np.sqrt(L**2 + d**2)
            + d
        )

    expected_ratio = M_analytic(L2) / M_analytic(L1)

    assert np.isclose(M2 / M1, expected_ratio, rtol=0.2)

