import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator
from ..utils.smooth import smooth3DVectors, smooth
from .wires import Wire
import matplotlib.pyplot as plt

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
                 alpha=.0001, beta=1.0, Bp=1.0, smoothing=0.5, L_init=None,
                 color="red", transparency=0):
        super().__init__(p, v, m, I, Bp=Bp, is_fixed=is_fixed, r=r, L_init=L_init, color=color, transparency=transparency)
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

        # Visualization Info
        self.color = color
        self.transparency = transparency

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

        curvature = dt_norm / np.maximum(self.ds, 1e-3)

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
        gradF = dF_norm / np.maximum(self.ds, 1e-5)

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
        rho = np.clip(self.rho, 0.5, 3.0)
        rho = smooth(rho, window_len=5)

        s = self.s
        ds = self.ds

        cumulative = np.cumsum(rho * ds)
        cumulative -= cumulative[0]
        cumulative /= cumulative[-1]

        T = np.linspace(0, 1, N_new)
        new_s = np.interp(T, cumulative, s)

        # --------------------------------------------------
        # HARD MINIMUM SPACING ENFORCEMENT
        # --------------------------------------------------
        s_min = self.r   # minimum allowed spacing

        for i in range(1, len(new_s)):
            if new_s[i] - new_s[i-1] < s_min:
                new_s[i] = new_s[i-1] + s_min

        # Prevent overshoot at end
        L_total = s[-1]
        if new_s[-1] > L_total:
            new_s = new_s * (L_total / new_s[-1])

        return new_s

    # -------------------------------------------------------------------------
    # Interpolation helpers
    # -------------------------------------------------------------------------
    def interpolate_vector(self, arr, new_s):
        """Component-wise spline interpolation for vector-valued arrays."""
        s = self.s
        out = np.column_stack([
            CubicSpline(s, arr[:,k], bc_type="natural")(new_s)
            for k in range(arr.shape[1])
        ])
        return out

    def _interpolate_scalar(self, arr, new_s):
        """Spline interpolation for scalar array (e.g. mass)."""
        s = self.s
        return CubicSpline(s, arr.flatten(), bc_type="natural")(new_s)

    def interpolate_scalar(self, arr, new_s):
        """
        Positivity-preserving interpolation for scalar fields (e.g., mass).
        Uses log-space PCHIP interpolation.

        Guarantees:
        - No negative values
        - No spline overshoot
        - Stable for 1-2 orders of magnitude variation
        """

        s = self.s
        y = arr.flatten()

        # Safety floor to avoid log(0)
        eps = 1e-14
        y_safe = np.maximum(y, eps)

        # Interpolate in log space
        log_interp = PchipInterpolator(s, np.log(y_safe))
        log_y_new = log_interp(new_s)

        y_new = np.exp(log_y_new)

        return y_new.reshape(-1,1)
    
    # -------------------------------------------------------------------------
    # ENGINE-COMPATIBLE REMESHING HOOK: interpolate()
    # -------------------------------------------------------------------------
    def interpolate(self):
        """
        Engine calls this after update().
        This override performs curvature- and force-gradient-driven
        adaptive remeshing without changing the Engine interface*.
        """
        
        if self.is_fixed:
            return  # do nothing

        # On first step, no forces yet
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
        # ----- Mass positivity -----
        # Ensure total mass stays positive:
        m_new[m_new <= 9e-4] = 9e-4

        # ----- Update state -----
        self.p[2:-2] = p_new[2:-2]
        self.v[2:-2] = v_new[2:-2]
        self.m[2:-2] = m_new[2:-2]
