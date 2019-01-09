from numpy.linalg import norm
import numpy as np
from mayavi import mlab

G = 6.67e-11

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
                
