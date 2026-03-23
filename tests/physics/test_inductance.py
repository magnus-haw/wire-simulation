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


def make_arc(n=1000, radius=1.0,theta_f=2*np.pi, z=0.0):
    theta = np.linspace(0, theta_f, n, endpoint=False)
    return np.column_stack((
        radius*np.cos(theta),
        radius*np.sin(theta),
        z*np.ones_like(theta)
    ))


def make_helix(n=1000,radius=1.0,pitch=np.pi/4,curve_length=1.0):
    a = radius
    c = pitch/(2*np.pi)
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
    l2 = 50.0
    r = 0.001

    p1a = make_straight_wire(length=l1)
    p1b = make_straight_wire(length=l2)

    L1 = inductance(p1a, rwire=r, norm_mag=mu0, part='self')
    L2 = inductance(p1b, rwire=r, norm_mag=mu0, part='self')
    print("Computed:")
    print(L1)
    print(L2)
    print("ratio: " +str(L2/L1))

    # From Edward B. Rosa, "The Self and Mutual Inductance of Linear Conductors," 1894, p. 305
    # NOTE: We are assuming the relative permeability of the wire is near unity, for simplicity. This should constitute a negligible contribution for all but the smallest wires.
    def L_analytic(l):
        return 2*(l*np.log( (1 / r)*(l + np.sqrt(l**2 + r**2))) - np.sqrt(l**2 + r**2) + l/4 + r )
    print("analytic")
    print(L_analytic(l1))
    print(L_analytic(l2))
    print("ratio: " + str(L_analytic(l2)/L_analytic(l1)))
    print("error: " + str( (L_analytic(l2)/L_analytic(l1) - L2/L1)/(L_analytic(l2)/L_analytic(l1)) ))
    
    expected_ratio = L_analytic(l2) / L_analytic(l1)
    
    assert np.isclose(L2/L1, expected_ratio, rtol=0.02)

def test_self_inductance_scales_with_radius():
    l = 1.0
    r1 = 0.01
    r2 = 0.1

    p1 = make_straight_wire(length=l)

    L1 = inductance(p1, rwire=r1, norm_mag=mu0, part='self')
    L2 = inductance(p1, rwire=r2, norm_mag=mu0, part='self')
    print("Computed:")
    print(L1)
    print(L2)
    print("ratio: " +str(L2/L1))

    # From Rosa
    def L_analytic(r):
        return 2*(l*np.log( (1 / r)*(l + np.sqrt(l**2 + r**2))) - np.sqrt(l**2 + r**2) + l/4 + r )

    print("analytic")
    print(L_analytic(r1))
    print(L_analytic(r2))
    print("ratio: " + str(L_analytic(r2)/L_analytic(r1)))
    print("error: " + str( (L_analytic(r2)/L_analytic(r1) - L2/L1)/(L_analytic(r2)/L_analytic(r1)) ))

    expected_ratio = L_analytic(r2) / L_analytic(r1)
    
    assert np.isclose(L2/L1, expected_ratio, rtol=0.02)

def test_self_inductance_of_arc():
    angle1 = np.pi/2
    angle2 = np.pi
    angle3 = 3*np.pi/2
    angle4 = 6.24018
    angle5 = 6.28
    r = 0.001
    l = 1

    c1 = make_arc(radius=l/(angle1), theta_f=angle1)      # All arcs have a length of 1m
    c2 = make_arc(radius=l/(angle2), theta_f=angle2)
    c3 = make_arc(radius=l/(angle3), theta_f=angle3)
    c4 = make_arc(radius=l/(angle4), theta_f=angle4)
    c5 = make_arc(radius=l/(angle5), theta_f=angle5)


    def Ti2(x,gran=1000):
        tspace = np.linspace(0,x,gran)
        
        invTanIntegral = (np.arctan(tspace)/tspace)*(tspace[1]-tspace[0])
        
        res = np.nansum(invTanIntegral)
        
        return res
    
    # From Majic, 2024
    def analytic_arc_L(xspace,l,gran=1000):
        vals = np.zeros(xspace.shape)
    
        for i in range(len(vals)):
            theta = xspace[i]
            vals[i] = 2*l*((4/theta)*np.sin(theta/2) + np.log((4/theta)*np.tan(theta/4)) - (4/theta)*Ti2(np.tan(theta/4),gran=gran) - 1)
    
        return vals

    analyticVals = analytic_arc_L([angle1,angle2,angle3,angle4,angle5])
    
    L1 = inductance(c1, rwire=r, part='self')
    L2 = inductance(c2, rwire=r, part='self')
    L3 = inductance(c3, rwire=r, part='self')
    L4 = inductance(c4, rwire=r, part='self')
    L5 = inductance(c5, rwire=r, part='self')
    L_parr = 2*mu0*(l*np.log( (1 / r)*(l + np.sqrt(l**2 + r**2))) - np.sqrt(l**2 + r**2) + l/4 + r )
    
    assert (np.isclose(L1-L_parr, mu0*analyticVals[0],rtol=.1) and 
            np.isclose(L2-L_parr, mu0*analyticVals[1],rtol=.1) and 
            np.isclose(L3-L_parr, mu0*analyticVals[2],rtol=.1) and
            np.isclose(L4-L_parr, mu0*analyticVals[3],rtol=.1) and
            np.isclose(L5-L_parr, mu0*analyticVals[4],rtol=.1))

def test_self_inductance_of_helix(plot=True):
    pitch1 = np.pi/36
    pitch2 = np.pi/12
    pitch3 = np.pi/4
    d=0.001

    h1 = make_helix(radius=0.0125, curve_length=50, pitch=pitch1)
    h2 = make_helix(radius=0.0125, curve_length=50, pitch=pitch2)
    h3 = make_helix(radius=0.0125, curve_length=50, pitch=pitch3)

    # From Weaver, 2011, 'The Inductance of a Helix of Any Pitch', pg. 18 (last one is estimated from plot digitizer, values in H)
    analyticVals = {np.pi/36:6.8808e-6,np.pi/12:5.0111e-6,np.pi/4:6.9784e-6}

    L1 = inductance(h1, rwire=d, part='self')
    L2 = inductance(h2, rwire=d, part='self')
    L3 = inductance(h3, rwire=d, part='self')
    print("Computed inductance of helix with pitch " + str(pitch1) + ": " + str(L1))
    print("Computed inductance of helix with pitch " + str(pitch2) + ": " + str(L2))
    print("Computed inductance of helix with pitch " + str(pitch3) + ": " + str(L3))

    print("Analytic values: " + str(analyticVals))
    
    assert (np.isclose(L1, analyticVals[pitch1],rtol=.1) and np.isclose(L2, analyticVals[pitch2],rtol=.1) and np.isclose(L3, analyticVals[pitch3],rtol=.1))