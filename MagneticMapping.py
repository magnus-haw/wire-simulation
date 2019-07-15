import numpy as np
import math
import mayavi.mlab as mlab
import matplotlib.pyplot as plt
import sys,os
cwd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cwd + "/classes/")

from Constants import mu0, pi, Kb,amu,mass_elec,elec
from Engine import MultiWireEngine
from Wires import Wire,State
from Utility import biot_savart_SI

import matplotlib.pyplot as plt
from sympy import Symbol, cos, sin, solve
from sympy.vector import CoordSys3D, Vector
N = CoordSys3D('N')

### Dimensional scales
L = .03 #m
I = 1. #A
B0 = mu0*I/(2*pi*L) #Tesla
v0 = 150 #m/s
tau = L/v0 #s
dt = .02*tau
n=1000
mu = pi*4e-7

dr = 1.
dm = pi*dr

print("L (m)", L)
print("I (A)", B0)
print("tau (s)", tau)

######################## General Functions #####################################

# Calculate the magnitude of magnetic field w/ its components
def mag(vector):
    magnitude = (vector.dot(vector))**(0.5)
    return magnitude

# Calculate the magnetic field's angle phi
def phi(vector):
    x = vector[0]
    y = vector[1]
    z = vector[2]
    adj = ((x**2) + (y**2))**(0.5)
    phi = np.arctan(z/adj)
    return phi

# Calculate the magnetic field's angle theta
def theta(vector):
    x = vector[0]
    y = vector[1]
    theta = np.arctan(y/x)
    return theta

# Determine the distance between sensor and current using ideal (straight) wire scenario
def radialConstraint(vector, current):
    bmag = mag(vector)
    R = mu*I/(bmag*2.*np.pi)
    return R

def intersection(R, position, Bvec, n_vec):
    #n_vec is an array (e.g. [0, 0, 1]) that represents the direction of the current
    x = position[0]
    y = position[1]
    z = position[2]
    t = Symbol('t')

    if n_vec[0] == 0 and n_vec[1] == 0:
        """ It is the R*cos(t)*N.i + R*sin(t)*N.k that needs modifying """
        c = Bvec.dot(position)
        v1 = c/Bvec[0]*N.i
        Bvec = Bvec[0]*N.i + Bvec[1]*N.j + Bvec[2]*N.k
        v2 = v1.cross(Bvec)
        v1, v2 = scale_vec(v1, R), scale_vec(v2, R)
        v = minDistance_Z(position, v1, v2)
        # v = x*N.i + y*N.j + z*N.k + v1*cos(t) + v2*sin(t)
        # p = x*N.i + y*N.j + z*N.k
        # n_vec = n_vec[0]*N.i + n_vec[1]*N.j + n_vec[2]*N.k
        # eq = (v - p).dot(n_vec)
        # sol = solve(eq, t)
        int_pnt = [v.dot(N.i), v.dot(N.j), v.dot(N.k)]
    return int_pnt

def minDistance_Z(pos, v1, v2):
    min_vec = None
    min_z = 1000.
    for t in range(0, 1000):
        v = pos[0]*N.i + pos[1]*N.j + pos[2]*N.k + v1*np.cos(t) + v2*np.sin(t)
        if v.dot(N.k) < min_z:
            min_vec = v
    return min_vec

def scale_vec(vector, length = 1):
    mag_vec = mag(vector)
    scaled_vec = vector*length/mag_vec
    return scaled_vec

def angle(points_array):
    #points_array is an array that represents all the positions of the
    # helix at each instance
    theta_array = []
    phi_array = []
    for i in range(len(points_array)):
        x0 = points_array[i][0]
        y0 = points_array[i][1]
        z0 = points_array[i][2]
        x1 = points_array[i+1][0]
        y1 = points_array[i+1][1]
        z1 = points_array[i+1][2]

        x = x1 - x0
        y = y1 - y0
        z = z1 - z0

        vec = [x, y, z]
        theta = theta(vec)
        phi = phi(vec)

        theta_array.append(theta)
        phi_array.append(phi)
    return theta_array, phi_array

# Finds the closest distance from the dictionary of all distances
def current_pos(position, path):
    path_list = path.tolist()
    for i in path_list:
        if percent(0., i[2]) < 100:
            print('YEAH')
    return None

# Calculates the percent of deviation from a selected value
def percent(obs_val, act_val):
    percent = abs(100*(obs_val - act_val)/act_val)
    return percent

################################################################################

""" Remember to reconstruct a timeseries of the helix and test the function
above"""

# Initialize probes
R = 0.07
probes = np.array([[R, 0. , 0], [0, R, 0], [-R, 0, 0],
                   [0, -R, 0]])
n_vec = [0, 0, 1]

Bvec_array = []
pnt_array = []
current_array = []
t_rng = range(0, 10)
for i in t_rng:

    #Initialize path/wire
    phi = np.linspace(-4.*np.pi, 4.*np.pi, n) + 2.*np.pi*i/100 # DISCUSS WITH MAGNUS THE LENGTH OF WIRE
    path = np.array([0.02*np.cos(phi), 0.02*np.sin(phi), (phi/pi)/1.5]).T
    mass = np.ones((n,1))
    wr = Wire(path, path*0., mass, I, r=0.01, Bp=1)

    #Calculate the magnetic field vector at probes[0]
    # All values in SI units
    Bvec = biot_savart_SI(probes[0], I, wr.p, delta = 0.01)
    Bvec_array.append(Bvec)
    r_dist = radialConstraint(Bvec, I)
    pnt = intersection(r_dist, probes[0], Bvec, n_vec)
    pnt_array.append(pnt)
    current_pnt = current_pos(probes[0], path)
    current_array.append(current_pnt)

print('CALCULATED POINTS OF CURRENT: ' + str(pnt_array))
print('ACTUAL POINTS OF CURRENT: ' + str(current_array))
