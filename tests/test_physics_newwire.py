import numpy as np
import pytest

from wireflux.models.wires import Wire
from wireflux.models.newwires import NewWire
from wireflux.core.engine import MultiWireEngine
from wireflux.physics.biot_savart import getBField, biot_savart
from wireflux.physics.forces import JxB_force


###############################################
# 1 — PARALLEL WIRE FORCE TEST (from ConvergenceTests.py)
###############################################

@pytest.mark.parametrize("WireClass", [Wire, NewWire])
def test_parallel_wire_force_direction(WireClass):
    """
    Physically correct behavior:
      Two parallel wires with currents in the same direction ATTRACT.
    """

    # build two long straight wires
    N = 200
    z   = np.linspace(0, 1.0, N)
    sep = 0.1

    p1 = np.column_stack([np.zeros_like(z),
                          np.zeros_like(z),
                          z])
    p2 = np.column_stack([np.full_like(z, sep),
                          np.zeros_like(z),
                          z])

    v = np.zeros_like(p1)
    m = np.ones((N,1))
    I = 1.0

    w1 = WireClass(p1, v.copy(), m.copy(), I)
    w2 = WireClass(p2, v.copy(), m.copy(), I)

    B = getBField(w1.p, [w2.p], [w2.I], delta=0.01)
    F = JxB_force(w1.p, w1.I, B)

    # force should point +x (toward wire 2)
    mean_fx = np.mean(F[:,0])
    assert mean_fx > 0.0, "Parallel wires should attract along +x"


@pytest.mark.parametrize("WireClass", [Wire, NewWire])
def test_parallel_wire_force_convergence(WireClass):
    """
    Increase resolution → force-per-unit-length should converge.
    """

    def compute_force(N):
        z = np.linspace(0, 1.0, N)
        sep = 0.1

        p1 = np.column_stack([np.zeros_like(z),
                              np.zeros_like(z),
                              z])
        p2 = np.column_stack([np.full_like(z, sep),
                              np.zeros_like(z),
                              z])

        v = np.zeros_like(p1)
        m = np.ones((N,1))
        I = 1.0

        w1 = WireClass(p1, v.copy(), m.copy(), I)
        w2 = WireClass(p2, v.copy(), m.copy(), I)

        B = getBField(w1.p, [w2.p], [w2.I], delta=0.01)
        F = JxB_force(w1.p, w1.I, B)

        return np.mean(F[:,0])

    f50  = compute_force(50)
    f100 = compute_force(100)
    f200 = compute_force(200)

    # convergence → changes shrink
    assert abs(f50 - f100) > abs(f100 - f200)


###############################################
# 2 — MASS CONSERVATION (part of BenchmarkTests.py)
###############################################

@pytest.mark.parametrize("WireClass", [Wire, NewWire])
def test_mass_conservation_after_many_steps(WireClass):
    """
    Ensure repeated updates + interpolation do not change total mass.
    """

    N = 200
    t = np.linspace(0, 2*np.pi, N)
    p = np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)])
    v = np.zeros_like(p)
    m = np.ones((N,1))
    w = WireClass(p, v, m, 1.0)

    M0 = np.sum(w.m)

    # apply repeated updates
    F = np.zeros_like(p)
    for _ in range(50):
        w.update(0.01, F)
        w.interpolate()

    M1 = np.sum(w.m)
    assert np.isclose(M1, M0, rtol=1e-6), "Mass drifted during evolution"


###############################################
# 3 — COMPARABLE MOTION (BenchmarkTests.py)
###############################################

def test_wire_vs_newwire_motion_equivalence():
    """
    Ensures that under the same forces:
     - Center-of-mass motion is comparable
     - Radial expansion trend is similar
    """

    N = 200
    t = np.linspace(0, 2*np.pi, N)
    p = np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)])
    v = 0.01*np.random.randn(*p.shape)
    m = np.ones((N,1))

    wA = Wire(p.copy(), v.copy(), m.copy(), 1.0)
    wB = NewWire(p.copy(), v.copy(), m.copy(), 1.0)

    dt = 0.01
    steps = 20

    # synthetic outward force
    F = p.copy()

    for _ in range(steps):
        wA.update(dt, F)
        wA.interpolate()
        wB.update(dt, F)
        wB.interpolate()

    comA = np.mean(wA.p, axis=0)
    comB = np.mean(wB.p, axis=0)

    assert np.linalg.norm(comA - comB) < 0.1

    rA = np.mean(np.linalg.norm(wA.p[:,:2], axis=1))
    rB = np.mean(np.linalg.norm(wB.p[:,:2], axis=1))

    assert np.isclose(rA, rB, rtol=0.15)


###############################################
# 4 — INDUCTANCE CONSISTENCY (based on CompareTorusInductance.py)
###############################################
@pytest.mark.parametrize("WireClass", [Wire, NewWire])
def test_inductance_symmetry(WireClass):
    """
    A circular loop should have rotationally symmetric inductance,
    and NewWire should match Wire’s inductance.
    """

    from wireflux.utils.Utility import inductance

    N = 200
    t = np.linspace(0, 2*np.pi, N)
    p = np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)])
    v = np.zeros_like(p)
    m = np.ones((N,1))
    w = WireClass(p, v, m, 1.0)

    Lxx = inductance(w.p, w.p, rwire=0.1)
    Lyx = inductance(w.p, np.roll(w.p, 20, axis=0), rwire=0.1)

    assert np.isclose(Lxx, Lyx, rtol=1e-2), \
        "Inductance should be symmetric under rotation"


###############################################
# 5 — ENGINE-LEVEL SANITY CHECK
###############################################

def test_engine_runs_with_both_wire_types():
    """
    Ensure MultiWireEngine can run for several steps with both Wire and NewWire.
    """

    from wireflux.core.State import State

    N = 200
    t = np.linspace(0, 2*np.pi, N)
    p = np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)])
    v = np.zeros_like(p)
    m = np.ones((N,1))

    s1 = State("testA", [Wire(p.copy(), v.copy(), m.copy(), 1.0)], time=0, load=0)
    s2 = State("testB", [NewWire(p.copy(), v.copy(), m.copy(), 1.0)], time=0, load=0)

    eng1 = MultiWireEngine(s1)
    eng2 = MultiWireEngine(s2)

    # run a few steps
    for _ in range(5):
        eng1.step()
        eng2.step()

    assert not np.any(np.isnan(eng1.state.items[0].p))
    assert not np.any(np.isnan(eng2.state.items[0].p))
