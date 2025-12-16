import numpy as np
import pytest

from wireflux.models.wires import Wire
from wireflux.models.newwires import NewWire
from wireflux.core.engine import MultiWireEngine
from wireflux.physics.biot_savart import getBField, biot_savart
from wireflux.utils.smooth import smooth3DVectors


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def make_circle(R=1.0, N=200):
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    return np.column_stack([R*np.cos(t), R*np.sin(t), np.zeros_like(t)])

def total_mass(w):
    return float(np.sum(w.m))


# -------------------------------------------------------------------
# 1. B-FIELD COMPARISON TEST
# -------------------------------------------------------------------

@pytest.mark.parametrize("WireClass", [Wire, NewWire])
def test_b_field_from_engine(WireClass):
    """Ensure both Wire and NewWire produce physically correct B-fields from engine perspective."""

    # Circular loop
    p = make_circle()
    v = np.zeros_like(p)
    m = np.ones((len(p),1))
    I = 1.0

    w = WireClass(p, v, m, I)

    # Point at center
    x0 = np.array([0.0, 0.0, 0.0])

    # Compute B-field using Utility (same used by Engine)
    B = biot_savart(x0, w.I, w.p)

    # Analytical magnetic field for center of loop
    mu0 = 4e-7 * np.pi
    R = 1.0
    B_expected = mu0 * I / (2*R)

    assert np.isclose(np.linalg.norm(B), B_expected, rtol=1e-2)


# -------------------------------------------------------------------
# 2. MASS CONSERVATION TEST
# -------------------------------------------------------------------

@pytest.mark.parametrize("WireClass", [Wire, NewWire])
def test_mass_conservation(WireClass):
    """Mass must remain unchanged after interpolation/remeshing."""

    p = make_circle()
    v = np.zeros_like(p)
    m = np.ones((len(p),1))
    I = 1.0

    w = WireClass(p, v, m, I)
    M0 = total_mass(w)

    # Perform multiple interpolate/remesh operations
    for _ in range(10):
        w.interpolate()
        assert np.isclose(total_mass(w), M0, rtol=1e-6), "Mass changed!"

    # Perform dummy updates
    forces = np.zeros_like(w.p)
    for _ in range(10):
        w.update(0.01, forces)
        w.interpolate()
        assert np.isclose(total_mass(w), M0, rtol=1e-6)


# -------------------------------------------------------------------
# 3. COMPARABLE MOTION UNDER SAME FORCES
# -------------------------------------------------------------------

def test_wire_vs_newwire_motion_comparison():
    """
    Apply the same synthetic force field to both Wire and NewWire,
    and ensure they move in comparable ways (not identical, but
    same trend, same COM motion direction, same energy trend).
    """
    # initial geometry
    p = make_circle()
    v = 0.01 * np.random.randn(*p.shape)
    m = np.ones((len(p),1))
    I = 1.0
    
    w1 = Wire(p.copy(), v.copy(), m.copy(), I)
    w2 = NewWire(p.copy(), v.copy(), m.copy(), I)

    dt = 0.01
    steps = 20

    # Synthetic force field: outward radial force
    forces = np.zeros_like(p)
    forces[:,0] = p[:,0]
    forces[:,1] = p[:,1]

    for _ in range(steps):
        w1.update(dt, forces)
        w1.interpolate()
        w2.update(dt, forces)
        w2.interpolate()

    # Compare center of mass motion
    com1 = np.mean(w1.p, axis=0)
    com2 = np.mean(w2.p, axis=0)

    # They should not diverge significantly
    assert np.linalg.norm(com1 - com2) < 0.1

    # Compare radial expansion
    r1 = np.mean(np.linalg.norm(w1.p[:,:2], axis=1))
    r2 = np.mean(np.linalg.norm(w2.p[:,:2], axis=1))

    # Expansion rate should be similar
    assert np.isclose(r1, r2, rtol=0.1)


# -------------------------------------------------------------------
# 4. ENGINE-LEVEL TEST (actual Engine force pipeline)
# -------------------------------------------------------------------

def test_engine_wire_vs_newwire():
    """
    Run a few engine steps and verify that using Wire vs NewWire
    yields comparable energy and motion, even though meshes differ.
    """
    from wireflux.core.Engine import MultiWireEngine
    from wireflux.core.State import State

    p = make_circle()
    v = np.zeros_like(p)
    m = np.ones((len(p),1))
    I = 1.0

    # Create states
    s1 = State("wireA", [Wire(p.copy(), v.copy(), m.copy(), I)], time=0, load=0)
    s2 = State("wireB", [NewWire(p.copy(), v.copy(), m.copy(), I)], time=0, load=0)

    eng1 = MultiWireEngine(s1)
    eng2 = MultiWireEngine(s2)

    # Run ~10 steps
    for _ in range(10):
        eng1.step()
        eng2.step()

    # Compare energies
    E1 = eng1.getEnergy()
    E2 = eng2.getEnergy()

    assert np.isclose(E1, E2, rtol=0.1)

    # Compare COM motion
    com1 = np.mean(eng1.state.items[0].p, axis=0)
    com2 = np.mean(eng2.state.items[0].p, axis=0)

    assert np.linalg.norm(com1 - com2) < 0.2
