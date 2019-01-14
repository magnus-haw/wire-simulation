import sys
import os
import pickle

import numpy as np
import mayavi.mlab as mlab
from scipy.interpolate import interp1d,splprep,splev

from Utility import smooth

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
    
    def __init__(self,p,v,m,scalars={},fields={},is_fixed=0,I=1.):
        """Initializes wire, p: array(n,3), v: array(n,3), m: array(n,1)"""
        sp,sv,sm = np.shape(p),np.shape(v),np.shape(m)
        
        if sp[1] == 3 and sp[0] >= Wire._min_len and sv == sp and sm[0] == sp[0] and sm[1] ==1:
            self.p = np.array(p)
            self.v = np.array(v)
            self.m = np.array(m)
            
            self.fields = dict(fields)
            self.scalars = dict(scalars)
            self.scalars['is_fixed'] = is_fixed
            self.scalars['I'] = I
            self.ind = Wire.npaths
            Wire.npaths += 1
        else:
            print("Wire initialization error: incorrect shape of input arrays")
            sys.exit()

    def boundary_conditions(self,time,r0=2.):
        # impervious lower boundary
        self.v[2:-2,2][self.p[2:-2,2] < r0] = 0
        self.p[2:-2,2][self.p[2:-2,2] < r0] = r0

    def interpolate(self):
        # TODO: Optimize interpolation and re-parametrize here?
        # TODO: variable mass density/ adaptive time steps
        

##        # path smoothing
##        for j in range(0,3):
##            self.p[2:-2,j] = smooth(self.p[:,j],window_len=5)[2:-2]
        
        # reinterpolate spline
        length_param = np.cumsum(np.linalg.norm(np.diff(self.p,axis=0),axis=1))
        length_param = np.append(0,length_param)
        f = interp1d(length_param, self.p, kind='cubic',axis=0)
        lp = np.linspace(0,length_param[-1],len(length_param))
        self.p = f(lp)

    def show(self,r0=1.):
        mlab.plot3d(self.p[:,0], self.p[:,1], self.p[:,2], tube_radius=r0, color=(1,0,0))
            
    def __repr__(self):
        return "scalars {0}, fields {1}\n pos: {2}".format(self.scalars,self.fields,self.p)

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

        # Center of mass multiplied by sum of particle massadd
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
                
