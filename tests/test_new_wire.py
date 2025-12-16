import numpy as np
import pytest

from wireflux.models.wires import Wire
from wireflux.models.newwires import NewWire
from wireflux.core.engine import MultiWireEngine
from wireflux.physics.biot_savart import getBField, biot_savart
from wireflux.utils.smooth import smooth3DVectors
from wireflux.utils.constants import mu0


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
def test_biot_savart_loop(WireClass):
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
    B_phys = (mu0 / (4*np.pi)) * 2 * B

    # Analytical magnetic field for center of loop
    R = 1.0
    B_expected = mu0 * I / (2*R)

    assert np.isclose(np.linalg.norm(B_phys), B_expected, rtol=1e-2)


# -------------------------------------------------------------------
# 2. MASS CONSERVATION TEST
# -------------------------------------------------------------------

@pytest.mark.parametrize("WireClass", [Wire, NewWire])
def test_mass_conservation(WireClass):
    """
    Total mass must remain unchanged under repeated interpolation / remeshing.
    This is a model-level invariant, independent of engine time stepping.
    """

    p = make_circle()
    v = np.zeros_like(p)
    m = np.ones((len(p), 1))
    I = 1.0

    w = WireClass(p, v, m, I)
    M0 = total_mass(w)

    # Repeated remeshing / interpolation
    for _ in range(20):
        w.interpolate()
        assert np.isclose(
            total_mass(w),
            M0,
            rtol=1e-6,
            atol=0.0
        ), "Mass changed during interpolation/remeshing"

