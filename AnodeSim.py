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

from scipy import interpolate

### Dimensional scales
L = .03 #m
I = 1. #A
B0 = mu0*I/(2*pi*L) #Tesla
v0 = 150 #m/s
tau = L/v0 #s
dt = .02*tau
n=101
mu = pi*4e-7

print("L (m)", L)
print("I (A)", B0)
print("tau (s)", tau)

############################ GENERAL FUNCTIONS #################################

def array_const(num, len):
    array = []
    for i in range(0, len):
        array.append(num)
    return array

def scale(array, range):
    array = np.array(array)
    return np.interp(array, (array.min(), array.max()), range)

################################################################################
# Initialize probes
R = 0.35
z = 0.
probes = np.array([[R*np.cos((5./4.)*pi), R*np.cos((5./4.)*pi), z],
                   [-R*np.cos((5./4.)*pi), R*np.cos((5./4.)*pi), z],
                   [R*np.cos((5./4.)*pi), -R*np.cos((5./4.)*pi), z],
                   [-R*np.cos((5./4.)*pi), -R*np.cos((5./4.)*pi), z]])
wire_rad = 0.01

#-----------------------------------PATH 1-------------------------------------#

z0 = [-75.496, -53.535, -20.844, 1.267, 19.684, 21.458, 20.646, 20.305, 19.918, 2.708]
y0 = [0., 0.1, 5.346, 13.575, 24.602, 27.916, 31.515, 31.773, 32.029, 41.452]

z0 = scale(z0, (-0.75, 0.25))
y0 = scale(y0, (0., 0.4))

f = interpolate.interp1d(y0, z0)
ynew = np.linspace(y0.min(), y0.max(), num=101, endpoint=True)
znew = f(ynew)
x = array_const(0., len(znew))

path = np.array([x, ynew, znew]).T
mass = np.ones((n,1))
wr = Wire(path, path*0, mass, I, r=wire_rad, Bp=1)

#-----------------------------------PATH 2-------------------------------------#
z1 = [-75.496, -53.535, -20.844, 1.267, 19.684, 21.458, 20.646, 20.305, 19.918, 2.708]
y1 = [0., 0.1, 1.0, 6.7875, 13.396, 24.176, 31.515, 31.773, 32.029, 41.452]
x1 =[0., 0.1, 5.252, 11.756, 17.396, 13.958, 1., 0.1, 0.05, 0.]

z1 = scale(z1, (-0.75, 0.25))
y1 = scale(y1, (0., 0.4))
x1 = scale(x1, (0., 0.17))

f1 = interpolate.interp1d(y1, z1)
f1x = interpolate.interp1d(y1, x1)
ynew1 = np.linspace(y1.min(), y1.max(), num=101, endpoint=True)
znew1 = f1(ynew1)
xnew1 = f1x(ynew1)

path1 = np.array([xnew1, ynew1, znew1]).T
mass = np.ones((n,1))
wr1 = Wire(path1, path1*0, mass, I, r=wire_rad, Bp=1)

#-----------------------------------PATH 3-------------------------------------#
z2 = [-75.496, -53.535, -20.844, 1.267, 19.684, 21.458, 20.646, 20.305, 19.918, 2.708]
y2 = [0., 0.1, 1.0, 6.7875, 13.396, 24.176, 31.515, 31.773, 32.029, 41.452]
x2 = [0., -0.1, -5.252, -11.756, -17.396, -13.958, -1., -0.1, -0.05, 0.]

z2 = scale(z2, (-0.75, 0.25))
y2 = scale(y2, (0., 0.4))
x2 = scale(x2, (-0.17, 0.))

f2 = interpolate.interp1d(y2, z2)
f2x = interpolate.interp1d(y2, x2)
ynew2 = np.linspace(y2.min(), y2.max(), num=101, endpoint=True)
znew2 = f2(ynew2)
xnew2 = f2x(ynew2)

path2 = np.array([xnew2, ynew2, znew2]).T
mass = np.ones((n,1))
wr2 = Wire(path2, path2*0, mass, I, r=wire_rad, Bp=1)

#-----------------------------------PATH 4-------------------------------------#
z3 = [-75.496, -53.535, -20.844, 1.267, 19.684, 21.458] #20.646, 20.305, 19.918, 2.708]
y3 = [0.05, -0.1, -5.346, -10.7875, -1., 13.396] #31.515, 31.773, 32.029, 41.452]
x3 = [0.05, -0.08, -0.1, 1., -11.756, -17.396] #-1., -0.1, -0.05, 0.]

z3 = scale(z3, (-0.75, 0.25))
y3 = scale(y3, (-0.1, 0.13))
x3 = scale(x3, (-.17, 0.))

f3 = interpolate.interp1d(z3, y3)
f3x = interpolate.interp1d(z3, x3)
znew3 = np.linspace(z3.min(), z3.max(), num=101, endpoint=True)
ynew3 = f3(znew3)
xnew3 = f3x(znew3)

path3 = np.array([xnew3, ynew3, znew3]).T
mass = np.ones((n,1))
wr3 = Wire(path3, path3*0, mass, I, r=wire_rad, Bp=1)

z4 = [21.458, 21., 20.646, 20.305, 19.918, 2.708]
y4 = [13.396, 24., 31.515, 31.773, 32.029, 41.452]
x4 = [-17.396, -13., -0.5, -0.1, -0.05, 0.]

z4 = scale(z4, (0.03, 0.25))
y4 = scale(y4, (.13, 0.4))
x4 = scale(x4, (-0.17, 0.))

f4 = interpolate.interp1d(y4, z4)
f4x = interpolate.interp1d(y4, x4)
ynew4 = np.linspace(y4.min(), y4.max(), num=101, endpoint=True)
znew4 = f4(ynew4)
xnew4 = f4x(ynew4)

path4 = np.array([xnew4, ynew4, znew4]).T
mass = np.ones((n,1))
wr4 = Wire(path4, path4*0, mass, I, r=wire_rad, Bp=1)

################################################################################

Bvec0 = biot_savart_SI(probes[0], I, wr.p, delta = 0.01)
Bvec1 = biot_savart_SI(probes[0], I, wr1.p, delta = 0.01)
Bvec2 = biot_savart_SI(probes[0], I, wr2.p, delta = 0.01)
Bvec3 = biot_savart_SI(probes[0], I, wr3.p, delta = 0.01)
Bvec4 = biot_savart_SI(probes[0], I, wr4.p, delta = 0.01)

Bvec = Bvec0 + Bvec1 + Bvec2 + Bvec3 + Bvec4
print(Bvec)

################################################################################

wr.show()
wr1.show()
wr2.show()
wr3.show()
wr4.show()
mlab.points3d(probes[:,0], probes[:,1], probes[:,2], scale_factor=.05)
mlab.show()
