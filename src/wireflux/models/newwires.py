import numpy as np
from scipy.interpolate import CubicSpline
from ..utils.smooth import smooth3DVectors
from .wires import Wire


class NewWire(Wire):
    """
    Adaptive mesh version of the Wire class.
    Drop-in replacement:
      - identical constructor signature
      - identical public interface
      - Engine can use this class with zero modifications

    Enhances:
      - curvature computation
      - force-gradient computation
      - density-based adaptive remeshing
      - optional smoothing (using Utility.smooth3DVectors)
    """

    # -------------------------------------------------------------------------
    # Constructor (same signature as Wire)
    # -------------------------------------------------------------------------
    def __init__(self, p, v, m, I, is_fixed=False, r=.25,
                 alpha=1.0, beta=1.0, smoothing=0.5):
        super().__init__(p, v, m, I, is_fixed=is_fixed, r=r)
        self.alpha = alpha        # curvature weight
        self.beta = beta          # force-gradient weight
        self.smoothing = smoothing

        # Adaptive geometry internal state
        self.last_force = None    # saved from update()
        self.ds = None            # segment lengths
        self.s = None             # cumulative arclength
        self.tangent = None
        self.curvature = None
        self.force_grad = None
        self.rho = None           # density metric

    # -------------------------------------------------------------------------
    # Geometry fundamentals
    # -------------------------------------------------------------------------
    def compute_arclength(self):
        x = self.p
        N = len(x)

        diffs = np.roll(x, -1, axis=0) - x
        ds = np.linalg.norm(diffs, axis=1)

        self.ds = ds
        self.s = np.concatenate([[0.0], np.cumsum(ds)[:-1]])
        return self.s

    def compute_tangent(self):
        x = self.p
        forward = np.roll(x, -1, axis=0)
        backward = np.roll(x, 1, axis=0)
        tang = forward - backward
        norm = np.linalg.norm(tang, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        self.tangent = tang / norm
        return self.tangent

    def compute_curvature(self):
        if self.tangent is None:
            self.compute_tangent()
        if self.ds is None:
            self.compute_arclength()

        tang = self.tangent
        dt = np.roll(tang, -1, axis=0) - np.roll(tang, 1, axis=0)
        dt_norm = np.linalg.norm(dt, axis=1)

        curvature = dt_norm / np.maximum(self.ds, 1e-12)

        # mild smoothing
        curvature = (
            0.25*np.roll(curvature,1)
            +0.50*curvature
            +0.25*np.roll(curvature,-1)
        )

        self.curvature = curvature
        return curvature

    # -------------------------------------------------------------------------
    # Force gradient along the filament
    # -------------------------------------------------------------------------
    def compute_force_gradient(self):
        if self.last_force is None:
            return None
        F = self.last_force
        if self.ds is None:
            self.compute_arclength()

        dF = np.roll(F,-1,axis=0) - np.roll(F,1,axis=0)
        dF_norm = np.linalg.norm(dF, axis=1)
        gradF = dF_norm / np.maximum(self.ds, 1e-12)

        # smoothing
        gradF = (
            0.25*np.roll(gradF,1)
            +0.50*gradF
            +0.25*np.roll(gradF,-1)
        )

        self.force_grad = gradF
        return gradF

    # -------------------------------------------------------------------------
    # Density metric (curvature + force gradient)
    # -------------------------------------------------------------------------
    def compute_density_metric(self):
        if self.curvature is None:
            self.compute_curvature()
        if self.force_grad is None:
            self.compute_force_gradient()

        self.rho = np.sqrt(
            1.0
            + self.alpha * (self.curvature**2)
            + self.beta  * (self.force_grad**2)
        )
        return self.rho

    # -------------------------------------------------------------------------
    # Invert cumulative density to get new arc positions
    # -------------------------------------------------------------------------
    def generate_new_arclength_positions(self, N_new):
        rho = self.rho
        s = self.s

        cumulative = np.cumsum(rho * self.ds)
        cumulative = cumulative - cumulative[0]
        cumulative /= cumulative[-1]  # normalize to [0,1]

        T = np.linspace(0, 1, N_new)
        new_s = np.interp(T, cumulative, s)
        return new_s

    # -------------------------------------------------------------------------
    # Interpolation helpers
    # -------------------------------------------------------------------------
    def interpolate_vector(self, arr, new_s):
        """Component-wise spline interpolation for vector-valued arrays."""
        s = self.s
        out = np.column_stack([
            CubicSpline(s, arr[:,k], bc_type="periodic")(new_s)
            for k in range(arr.shape[1])
        ])
        return out

    def interpolate_scalar(self, arr, new_s):
        """Spline interpolation for scalar array (e.g. mass)."""
        s = self.s
        return CubicSpline(s, arr.flatten(), bc_type="periodic")(new_s)

    # -------------------------------------------------------------------------
    # ENGINE-COMPATIBLE REMESHING HOOK: interpolate()
    # -------------------------------------------------------------------------
    def interpolate(self):
        """
        Engine calls this after update().
        This override performs curvature- and force-gradient–driven
        adaptive remeshing *without changing the Engine interface*.
        """
        if self.is_fixed:
            return  # do nothing

        # On first few steps, no forces yet → fallback
        if self.last_force is None:
            return super().interpolate()

        # ----- Compute geometry and metrics -----
        self.compute_arclength()
        self.compute_tangent()
        self.compute_curvature()
        self.compute_force_gradient()
        self.compute_density_metric()

        N_old = len(self.p)
        N_new = N_old  # keep same number of points (Engine expects this)

        # ----- Generate new arc positions -----
        new_s = self.generate_new_arclength_positions(N_new)

        # ----- Interpolate p, v, m -----
        p_new = self.interpolate_vector(self.p, new_s)
        v_new = self.interpolate_vector(self.v, new_s)
        m_new = self.interpolate_scalar(self.m, new_s).reshape(-1,1)

        # ----- Smoothing -----
        p_new = smooth3DVectors(p_new, n=3)
        v_new = smooth3DVectors(v_new, n=3)

        # ----- Mass conservation -----
        # m_new is a density function distributed along the filament.
        # Ensure total mass matches the original:
        total_mass = np.sum(self.m)
        m_new *= total_mass / np.sum(m_new)

        # ----- Update state -----
        self.p = p_new
        self.v = v_new
        self.m = m_new
