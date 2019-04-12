import sys
import os
import pickle

import numpy as np
import mayavi.mlab as mlab
from scipy.interpolate import interp1d,splprep,splev

from Utility import smooth
import matplotlib.pyplot as plt

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
    
    def __init__(self,p,v,m,I,r=1,Bp=1.,is_fixed=0,params={}):
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
                self.L_init = CumLen[-1] # initial length
        except:
            print(sp,sv,sm)
            print("Wire initialization error: incorrect shape of input arrays")
            sys.exit()

##    def interpolate(self):
##        # reinterpolate spline
##        length_param = np.cumsum(np.linalg.norm(np.diff(self.p,axis=0),axis=1))
##        length_param = np.append(0,length_param)
##        f = interp1d(length_param, self.p, kind='cubic',axis=0)
##        lp = np.linspace(0,length_param[-1],len(length_param))
##        self.p = f(lp)        

    def interpolate(self,rfactor=10):
        #print(len(self.m), self.m.sum(), (self.m * self.v).sum(axis=0))
        T,L,dl,N,R,tck,s = self.get_3D_curve_params()
        new_s = []
        new_mass = []
        new_vel = []

        # loops through intervals 
        for i in range(0,len(self.p)):
            new_s.append(s[i])
            new_mass.append(self.m[i])
            new_vel.append(self.v[i])

            # adds new point if spacing interval larger than
            #  some factor of the radius of curvature
            if i < len(self.p)-1:
                #dl[0] == 0 for integration purposes, hence index offset for dl array 
                if (dl[i+1] > R[i]/rfactor or dl[i+1] > 10*self.r) and dl[i+1]/2. > self.r:
                    new_s.append((s[i]+s[i+1])/2.)
                    new_mass.append(0.)
                    new_vel.append(0.)
                    
        nn = len(new_s)

        # interpolate positions
        self.p = np.zeros((nn,3))
        self.p[:,0],self.p[:,1],self.p[:,2] = splev(new_s,tck)

        # interpolate mass and velocity (conserves mass/momentum)
        self.m,self.v = np.zeros((nn,1)),np.zeros((nn,3))
        for i in range(0,nn):
            self.m[i] += new_mass[i]
            self.v[i] = new_vel[i]
            if new_mass[i] ==0:
                self.v[i] = (new_mass[i-1]*new_vel[i-1] + new_mass[i+1]*new_vel[i+1])/(new_mass[i-1]+new_mass[i+1])
                self.m[i] = (new_mass[i-1] + new_mass[i+1])/4.
                
                self.m[i-1] -= new_mass[i-1]/4. 
                self.m[i+1] -= new_mass[i+1]/4.

        #print(len(self.m), self.m.sum(), (self.m * self.v).sum(axis=0))
##        T,L,dl,N,R,tck,s = self.get_3D_curve_params()
##        print(dl[1:-1].max())
        
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

    def show(self):
        if self.is_fixed:
            cl = (0.84765625,0.5625,0.34375) #copper color for stationary coils
            mlab.plot3d(self.p[:,0], self.p[:,1], self.p[:,2], tube_radius=self.r, color=cl)
        else:
            cl = (1,0,0.) # red color for flux ropes
            # 3D tube representation of path
            #
            #mlab.points3d(self.p[:,0], self.p[:,1], self.p[:,2], color=cl)

            line = mlab.pipeline.line_source(self.p[:,0], self.p[:,1], self.p[:,2], self.m[:,0])
            tube = mlab.pipeline.tube(line, tube_radius=self.r, tube_sides=10)
            #tube.filter.vary_radius = 'vary_radius_by_scalar'
            mlab.pipeline.surface(tube)
            
    def __repr__(self):
        T,L,dl,N,R,tck,s = self.get_3D_curve_params()
        return "initial length {0}\nCurrent length {1}\nMax Rcurv {2}\nMin Rcurv {3}".format(self.L_init,L,R.max(),R.min())

    def __len__(self):
        return len(self.m)
    

class State(object):
    """Stores simulation state info, can load/write to pickle file"""
    
    def __init__(self,name,items=[],time=0,load=1):
        """Initializes state and loads file if it exists"""
        self.name = name
        self.items = items
        self.time = time
        self.fname = "{0}_{1}.pickle".format(self.name,self.time)

        if os.path.isfile(self.fname) and load:
            with open(self.fname,'rb') as fin:
                state_tuple = pickle.load(fin)
                self.name = state_tuple[0]
                self.items = state_tuple[1]
                self.time  = state_tuple[2]
                self.fname = "{0}_{1}.pickle".format(self.name,self.time)
            
    def save(self):
        """Writes state to pickle file"""
        with open(self.fname,'wb') as fout:
            pickle.dump((self.name,self.items,self.time),fout)

    def show(self):
        """Plot items"""
        for item in self.items:
            item.show()

    def __repr__(self):
        return "{0}, time: {1}\n  nItems: {2}".format(self.name,self.time,len(self.items))

    def __len__(self):
        return len(self.items)
    
class OctTreeNode:
    def __init__(self, corner, width):
        '''
        Initializes an OctTreeNode that represents a region of space.

        corner: one corner (x0, y0, z0) of the region
        width: width of region, i.e. the region contains all (x, y, z) with
               x0 <= x <= x0 + width, etc.
        '''
        self.corner = np.array(corner)
        self.width = np.array(width)

        # Number of particles
        self.n = 0

        # Sum of particle mass
        self.total_mass = 0

        # Center of mass multiplied by sum of particle mass add
        self.mass_weighted_positions = np.zeros(3)

        # If the region contains more than 1 particle, then it's recursively
        # subdivided into octants
        self.children = []

    def center_of_mass(self):
        if self.total_mass is not 0:
            return self.mass_weighted_positions / self.total_mass
        else:
            return None

    def approx_F(self, m, p, theta=0.5):
        '''
        Approximates the gravitational force on a point particle due to this
        region of space.

        m: mass of point particle
        p: position of point particle
        theta: we consider smaller regions for a better approximation whenever

                width / |p - center_of_mass| >= theta
        '''

        # An empty region exerts no gravitational force
        if self.n == 0:
            return np.zeros(3)

        delta = self.center_of_mass() - p
        r = norm(delta)

        # Particles don't exert force on themselves; we assume that no two
        # particles extremely close to each other
        if r < 1e-9:
            return np.zeros(3)

        # We can approximate the region as a single particle located at its
        # center of mass if either:
        # - The region only has a single particle
        # - width / r < theta
        if self.n == 1 or self.width / r < theta:
            return G * self.total_mass * m * delta / r**3
        else:
            total_force = np.zeros(3)
            for child in self.children:
                total_force += child.approx_F(m, p, theta=theta)
            return total_force

    def in_region(self, p):
        '''
        Checks whether a point particle at p would be within the region.
        '''
        return all(self.corner <= p) and all(p <= self.corner + self.width)

    def add_particle(self, m, p):
        '''
        Attempts to add a particle to the region; ignores particles that would
        be outside the region.
        '''
        if not self.in_region(p):
            return

        # "Leaf" nodes can have at most one particle, so adding a second
        # particle requires us to subdivide the region first.
        if self.n == 1:
            self.subdivide()

        self.n += 1
        self.total_mass += m
        self.mass_weighted_positions += m * p

        for child in self.children:
            child.add_particle(m, p)

    def subdivide(self):
        '''
        Attempts to subdivide the region into eight octants; does nothing if
        the region is already subdivided.
        '''
        if len(self.children) > 0:
            return

        for i in range(8):
            child = OctTreeNode(self.corner, self.width / 2)

            # We could manually set the coordinates for each child, but using
            # the binary representation of i lets us do this more easily
            for k in [0, 1, 2]:
                if i & (1 << k):
                    child.corner[k] += child.width[k]

            # Pass particle information to child; note that for all but one
            # child, this call will be ignored
            child.add_particle(
                    self.total_mass,
                    self.center_of_mass())

            self.children.append(child)

    def show(self):
        origin = self.corner
        w= self.width
        mlab.points3d(origin[0:1]+w[0]/2,
                      origin[1:2]+w[1]/2,
                      origin[2:3]+w[2]/2,
                      mode='cube',opacity=.1,scale_factor=w[0])

        if len(self.children) >0:
            for child in self.children:
                child.show()
                


if __name__ == "__main__":
    n=10
    L=1.
    phi = np.linspace(0.,np.pi,n)
    mass = np.ones((n,1))
    path0 = np.array([np.cos(phi),np.sin(phi),0*phi]).T
    w = Wire(path0,path0,mass,-1,is_fixed=False,r=.1)

