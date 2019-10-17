import numpy as np
import mayavi.mlab as mlab
from scipy.interpolate import interp1d,splprep,splev
import matplotlib.pyplot as plt

from Utility import smooth3DVectors,inductance

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
    
    def __init__(self,p,v,m,I,r=1,Bp=1.,L_init=0,is_fixed=0,params={}):
        """Initializes wire, p: array(n,3), v: array(n,3), m: array(n,1)"""
        sp,sv,sm = np.shape(p),np.shape(v),np.shape(m)
        
        try:
            if sp[1] == 3 and sp[0] >= Wire._min_len and sv == sp and sm[0] == sp[0] and sm[1] ==1:
                self.p = np.array(p) # position
                self.v = np.array(v) # velocity
                self.m = np.array(m) # mass
                self.Bp = float(Bp)  # axial flux
                self.I = float(I)    # current
                self.r = float(r)    # wire minor radius
                self.is_fixed = bool(is_fixed) # boolean for stationary wires
                
                self.params = dict(params)
                self.ind = Wire.npaths
                Wire.npaths += 1

                T,CumLen,dl,N,R,tck,s = self.get_3D_curve_params()
                if L_init == 0:
                    self.L_init = CumLen[-1] # initial length
                else:
                    self.L_init = L_init

                self.total_mass = self.m.sum()
        except:
            print(sp,sv,sm)
            print("Wire initialization error: incorrect shape of input arrays")

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

##    def sparse_interpolate(self,rfactor=20):
##        #print(len(self.m), self.m.sum(), (self.m * self.v).sum(axis=0))
##        T,L,dl,N,R,tck,s = self.get_3D_curve_params()
##        new_s = []
##        new_mass = []
##        new_vel = []
##
##        # loops through intervals 
##        for i in range(0,len(self.p)):
##            new_s.append(s[i])
##            new_mass.append(self.m[i])
##            new_vel.append(self.v[i])
##
##            # adds new point if spacing interval larger than
##            #  some factor of the radius of curvature
##            if i < len(self.p)-1:
##                #dl[0] == 0 for integration purposes, hence index offset for dl array 
##                if (dl[i+1] > R[i]/rfactor or dl[i+1] > 3*self.r) and dl[i+1]/2. > self.r:
##                    new_s.append((s[i]+s[i+1])/2.)
##                    new_mass.append(0.)
##                    new_vel.append(0.)
##                    
##        nn = len(new_s)
##
##        # interpolate positions
##        self.p = np.zeros((nn,3))
##        self.p[:,0],self.p[:,1],self.p[:,2] = splev(new_s,tck)
##
##        # interpolate mass and velocity (conserves mass/momentum)
##        self.m,self.v = np.zeros((nn,1)),np.zeros((nn,3))
##        for i in range(0,nn):
##            self.m[i] += new_mass[i]
##            self.v[i] = new_vel[i]
##            if new_mass[i] ==0:
##                self.v[i] = (new_mass[i-1]*new_vel[i-1] + new_mass[i+1]*new_vel[i+1])/(new_mass[i-1]+new_mass[i+1])
##                self.m[i] = (new_mass[i-1] + new_mass[i+1])/4.
##                
##                self.m[i-1] -= new_mass[i-1]/4. 
##                self.m[i+1] -= new_mass[i+1]/4.
##
##        #print(len(self.m), self.m.sum(), (self.m * self.v).sum(axis=0))
####        T,L,dl,N,R,tck,s = self.get_3D_curve_params()
####        print(dl[1:-1].max())
        
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

    def show(self,forces=None,velocity=False):
        if self.is_fixed:
            cl = (0.84765625,0.5625,0.34375) #copper color for stationary coils
            mlab.plot3d(self.p[:,0], self.p[:,1], self.p[:,2], tube_radius=self.r, color=cl)
        else:
            cl = (1,0,0.) # red color for flux ropes
            # 3D tube representation of path
            #
            mlab.plot3d(self.p[:,0], self.p[:,1], self.p[:,2], tube_radius=self.r, color=cl)

##            line = mlab.pipeline.line_source(self.p[:,0], self.p[:,1], self.p[:,2], self.m[:,0])
##            tube = mlab.pipeline.tube(line, tube_radius=self.r, tube_sides=10)
##            #tube.filter.vary_radius = 'vary_radius_by_scalar'
##            mlab.pipeline.surface(tube)

        if forces is not None:
            print(np.shape(forces),np.shape(self.p))
            mlab.quiver3d(self.p[:,0], self.p[:,1], self.p[:,2],forces[:,0], forces[:,1], forces[:,2])
        if velocity:
            vecs= mlab.quiver3d(self.p[:,0], self.p[:,1], self.p[:,2],self.v[:,0], self.v[:,1], self.v[:,2])
            vecs.glyph.glyph.scale_factor = 2.0
    
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
    w = Wire(path0,path0,mass,-1,is_fixed=False,r=.25)
    w.interpolate()
    w.show()
    mlab.show()

