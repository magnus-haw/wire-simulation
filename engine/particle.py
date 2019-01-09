from numpy.linalg import norm
from numpy.random import rand
import numpy as np

from barnes_hut import OctTreeNode

class Particle:
    def __init__(self, m, p, v):
        '''
        Initializes a point particle with the given mass, position, and velocity
        '''
        self.m = m
        self.p = p
        self.v = v

    def __str__(self):
        return "m = {}, p = {}, v = {}".format(self.m, self.p, self.v)

def advance_simulation(particles, dt=0.001):
    # Find region boundaries
    lo = float('inf')
    hi = float('-inf')
    for particle in particles:
        lo = min(lo, min(particle.p))
        hi = max(hi, max(particle.p))

    root = OctTreeNode(lo * np.ones(3), hi - lo)
    for particle in particles:
        root.add_particle(particle.m, particle.p)

    result = []
    for particle in particles:
        F = root.approx_F(particle.m, particle.p)
        result.append(Particle(
            particle.m,
            particle.p + particle.v * dt,
            particle.v + F / particle.m * dt))
    return result
