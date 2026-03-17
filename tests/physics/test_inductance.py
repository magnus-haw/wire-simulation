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


def make_arc(n=400, radius=1.0,theta_f=2*np.pi, z=0.0):
    theta = np.linspace(0, theta_f, n, endpoint=False)
    return np.column_stack((
        radius*np.cos(theta),
        radius*np.sin(theta),
        z*np.ones_like(theta)
    ))


def make_helix(n=400,radius=1.0,pitch=np.pi/4,curve_length=1.0):
    a = radius
    c = pitch/2*np.pi
    b = np.sqrt(a**2 + c**2)
    
    s = np.linspace(0, curve_length, n, endpoint=False)
    return np.column_stack((
        a*np.sin(s/b),
        a*np.cos(s/b),
        c*s/b
    ))

    

def test_mutual_inductance_parallel_wires_asymptotic():

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

def test_mutual_inductance_coaxial_loops():
    R = 1.0
    z = 0.5

    p1 = make_arc(n=600, radius=R, z=0.0)
    p2 = make_arc(n=600, radius=R, z=z)

    M_num = inductance(p1, p2, rwire=1e-3)

    k2 = (4*R*R) / ((2*R)**2 + z*z)
    k = np.sqrt(k2)

    K = ellipk(k2)
    E = ellipe(k2)

    M_expected = mu0 * R * ((2/k - k)*K - (2/k)*E)

    assert np.isclose(M_num, M_expected, rtol=0.05)

def test_mutual_inductance_symmetry():

    p1 = make_arc(radius=1.0)
    p2 = make_arc(radius=0.5, z=0.3)

    M12 = inductance(p1, p2)
    M21 = inductance(p2, p1)

    assert np.isclose(M12, M21, rtol=1e-10)

def test_mutual_inductance_scales_with_length():
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
    def M_analytic(l):
        return (
            l * np.log((l + np.sqrt(l**2 + d**2)) / d)
            - np.sqrt(l**2 + d**2)
            + d
        )

    expected_ratio = M_analytic(L2) / M_analytic(L1)

    assert np.isclose(M2 / M1, expected_ratio, rtol=0.2)

def test_self_inductance_scales_with_length():
    l1 = 5.0
    l2 = 10.0
    d = 0.01

    p1a = make_straight_wire(length=l1)
    p1b = make_straight_wire(length=l2)

    L1 = inductance(p1a,rwire=d,part='self')
    L2 = inductance(p1b,rwire=d,part='self')

    def L_analytic(l):
        return (l*np.log((l + np.sqrt(l**2 + d**2)) / d)
            - np.sqrt(l**2 + d**2)
            + d )


    expected_ratio = L_analytic(L2) / L_analytic(L1)
    
    assert np.isclose(L2/L1, expected_ratio, rtol=0.02)

def test_self_inductance_scales_with_radius():
    l = 1.0
    d1 = 0.01
    d2 = 0.1

    p1 = make_straight_wire(length=L1)

    L1 = inductance(p1,rwire=d1,part='self')
    L2 = inductance(p1,rwire=d2,part='self')

    def L_analytic(d):
        return ( l*np.log((l + np.sqrt(l**2 + d**2)) / d)
            - np.sqrt(l**2 + d**2)
            + d)


    expected_ratio = L_analytic(d2) / L_analytic(d1)
    
    assert np.isclose(L2/L1, expected_ratio, rtol=0.02)

def test_self_inductance_of_arc():
    angle1 = np.pi/2
    angle2 = np.pi
    angle3 = 3*np.pi/2
    angle4 = 6.24018
    angle5 = 6.28
    d=0.001

    c1 = make_arc(radius=1/(angle1), theta_f=angle1)      # All arcs have a length of 1m
    c2 = make_arc(radius=1/(angle2), theta_f=angle2)
    c3 = make_arc(radius=1/(angle3), theta_f=angle3)
    c4 = make_arc(radius=1/(angle4), theta_f=angle4)
    c5 = make_arc(radius=1/(angle5), theta_f=angle5)

    # From Majic, 2024, 'An Integral for the Self-inductance of thin wires' (all estimated from plot digitizer)
    analyticVals = {np.pi/2:-0.09014,np.pi:-0.64665,3*np.pi/2:-1.79887,6.24018:-2.91797,6.28:-2.90692}

    L1 = inductance(c1, rwire=d, part='self')
    L2 = inductance(c2, rwire=d, part='self')
    L3 = inductance(c3, rwire=d, part='self')
    L4 = inductance(c4, rwire=d, part='self')
    L5 = inductance(c5, rwire=d, part='self')
    
    assert (np.isclose(L1, analyticVals[angle1],rtol=.1) and 
            np.isclose(L2, analyticVals[angle2],rtol=.1) and 
            np.isclose(L3, analyticVals[angle3],rtol=.1) and
            np.isclose(L4, analyticVals[angle4],rtol=.1) and
            np.isclose(L5, analyticVals[angle5],rtol=.1))

def test_self_inductance_of_helix(plot=True):
    pitch1 = np.pi/36
    pitch2 = np.pi/12
    pitch3 = np.pi/4
    d=0.001

    h1 = make_helix(radius=0.025, curve_length=0.5, pitch=pitch1)
    h2 = make_helix(radius=0.025, curve_length=0.5, pitch=pitch2)
    h3 = make_helix(radius=0.025, curve_length=0.5, pitch=pitch3)

    # From Weaver, 2011, 'The Inductance of a Helix of Any Pitch', pg. 18 (last one is estimated from plot digitizer)
    analyticVals = {np.pi/36:6.8808e-6,np.pi/12:5.0111e-6,np.pi/4:6.9784e-6}

    L1 = inductance(h1, rwire=d, part='self')
    L2 = inductance(h2, rwire=d, part='self')
    L3 = inductance(h3, rwire=d, part='self')
    
    assert (np.isclose(L1, analyticVals[pitch1],rtol=.1) and np.isclose(L2, analyticVals[pitch2],rtol=.1) and np.isclose(L3, analyticVals[pitch3],rtol=.1))