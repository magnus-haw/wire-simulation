import numpy as np
import pyvista as pv
from scipy.interpolate import interp1d,splprep,splev
import matplotlib.pyplot as plt

from wireflux.utils.smooth import smooth3DVectors

class Wire(object):
    """
    Class describing moving wire in 3D, with arrays for path, velocity, and mass

    class attributes:
      npaths   -> tracks number of instances
      _min_len -> defines minimum path length, needed for interpolation

    instance attributes:
      p -> position array, shape(n,3)
      v -> velocity array, shape(n,3)
      m -> mass array, shape(n,1)
      ind -> unique instance index
      fields -> dictionary for additional fields
      scalars-> dictionary for additional scalars (i.e. current mag, fixed boolean)

    """
    npaths = 0
    _min_len = 5
    _max_len = 200
    
    def __init__(self, p, v, m, I, r=1.0, Bp=1.0, 
                 L_init=None,
                 is_fixed=False, 
                 params=None,
                 color='red',
                 transparency=0):
        """
        Initialize moving wire in 3D.

        p : (N,3) positions
        v : (N,3) velocities
        m : (N,1) mass per node
        """

        # ---- Convert to arrays ----
        p = np.asarray(p, dtype=float)
        v = np.asarray(v, dtype=float)
        m = np.asarray(m, dtype=float)

        # ---- Shape checks ----
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError(f"p must have shape (N,3), got {p.shape}")

        N = p.shape[0]

        if N < Wire._min_len:
            raise ValueError(f"Wire must have ≥ {Wire._min_len} nodes, got {N}")

        if v.shape != (N, 3):
            raise ValueError(f"v must have shape {(N,3)}, got {v.shape}")

        if m.shape != (N, 1):
            raise ValueError(f"m must have shape {(N,1)}, got {m.shape}")

        if not np.all(np.isfinite(p)):
            raise ValueError("Positions contain non-finite values")

        if not np.all(np.isfinite(v)):
            raise ValueError("Velocities contain non-finite values")

        if not np.all(np.isfinite(m)):
            raise ValueError("Mass contains non-finite values")

        if np.any(m <= 0):
            raise ValueError("Mass values must be positive")

        # ---- Assign core state ----
        self.p = p.copy()
        self.v = v.copy()
        self.m = m.copy()

        self.I = float(I)
        self.r = float(r)
        self.Bp = float(Bp)

        self.is_fixed = bool(is_fixed)
        self.last_force = None

        self.color = color
        self.transparency = transparency
        
        self.params = {} if params is None else dict(params)

        self.ind = Wire.npaths
        Wire.npaths += 1

        # ---- Derived quantities ----
        self.total_mass = float(self.m.sum())

        # Compute geometric properties safely
        _, CumLen, *_ = self.get_3D_curve_params()

        if L_init is None:
            self.L_init = float(CumLen[-1])
        else:
            self.L_init = float(L_init)

    def smooth(self):
        newp = smooth3DVectors(self.p,n=5)
        newv = smooth3DVectors(self.v,n=5)
        self.p[2:-2,:] = newp[2:-2,:]
        self.v[2:-2,:] = newv[2:-2,:]
        
    def interpolate(self,constant_density=None,smooth=False):
        # calculate length
        length_param = np.cumsum(np.linalg.norm(np.diff(self.p,axis=0),axis=1))
        length_param = np.append(0,length_param)
        L = length_param[-1]
        num_points = min(int(L/self.r) +1,Wire._max_len)
        lp = np.linspace(0,L,num_points)
        dl = np.linalg.norm(np.gradient(self.p, axis=0),axis=1)
        dl = np.array([dl]).T
        
        # reinterpolate positions
        fl = interp1d(length_param, self.p, kind='cubic',axis=0)
        self.p = fl(lp)
        new_dl = np.linalg.norm(np.gradient(self.p, axis=0),axis=1)
        new_dl = np.array([new_dl]).T

        # reinterpolate mass
        if constant_density is None:
            density = self.m/dl
        else:
            density = constant_density*np.ones(np.shape(dl))
        fd = interp1d(length_param, density, kind='cubic',axis=0)
        new_m = fd(lp)*new_dl        
        self.m = new_m*self.total_mass/new_m.sum()

        # reinterpolate velocity
        fv = interp1d(length_param, self.v, kind='cubic',axis=0)
        self.v = fv(lp)
        
    def get_3D_curve_params(self):
        """ Uses cubic splines to calculate path derivatives, length"""
        x,y,z = self.p[:,0],self.p[:,1],self.p[:,2]
        tck,s = splprep([x,y,z],s=0)
        ds = np.zeros(len(s))
        ds[1:] = np.diff(s)
        dx,dy,dz = splev(s,tck,der=1)
        dr = np.sqrt(dx*dx + dy*dy + dz*dz)
        dl = dr*ds
        L = np.cumsum(dl)
        T = np.array([dx/dr,dy/dr,dz/dr]).T

        p,u = splprep([dx/dr,dy/dr,dz/dr],u=L,s=0)
        dTx,dTy,dTz = splev(u,p,der=1)
        kurv = np.sqrt(dTx*dTx + dTy*dTy + dTz*dTz)
        kurv[kurv==0] = 1e-14
        N = np.array([dTx/kurv,dTy/kurv,dTz/kurv]).T
        R = 1./kurv

        # return tangent vector, length, normal vector, radius of curvature, spline_params, normed parameterization
        return T,L,dl,N,R,tck,s
    
    def show(self, forces=None, velocity=False, plotter=None):

        if plotter is None:
            plotter = pv.Plotter()

        # Set color per wire type
        if self.is_fixed:
            cl = 'sienna'  # copper
        else:
            cl = self.color  # user-defined color, default red

        # Create smoothed line and tube
        line = pv.Spline(self.p, len(self.p)*10)
        tube = line.tube(radius=self.r)

        # Add mesh with controlled shading
        plotter.add_mesh(
            tube,
            color=cl,
            opacity=np.clip(1-self.transparency,0,1),
            smooth_shading=False,   # IMPORTANT
            ambient=0.5,
            diffuse=0.5
        )

        if forces:
            plotter.add_arrows(self.p, self.last_force, mag=1.0, color='blue')

        if velocity:
            plotter.add_arrows(self.p, self.v, mag=2.0, color='green')

        return plotter

    def __repr__(self):
        T,L,dl,N,R,tck,s = self.get_3D_curve_params()
        return "initial length {0}\nCurrent length {1}\nMax Rcurv {2}\nMin Rcurv {3}".format(self.L_init,L,R.max(),R.min())

    def __len__(self):
        return len(self.m)


if __name__ == "__main__":
    n=100
    L=1.
    phi = np.linspace(0.,2*np.pi,n)
    mass = np.ones((n,1))
    path0 = np.array([np.cos(phi),np.sin(phi),0*phi]).T
    w = Wire(path0,path0,mass,-1,is_fixed=False,r=.05)
    w.interpolate()
    plotter = w.show()
    plotter.show()

