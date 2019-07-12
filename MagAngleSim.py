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

### Dimensional scales
L = .03 #m
I = 1. #A
B0 = mu0*I/(2*pi*L) #Tesla
v0 = 150 #m/s
tau = L/v0 #s
dt = .02*tau
n=1000
mu = pi*4e-7

print("L (m)", L)
print("I (A)", B0)
print("tau (s)", tau)
######################## General Functions #####################################
def mag(components):
    x = components[0]
    y = components[1]
    z = components[2]
    magnitude = ((x**2) + (y**2) + (z**2))**(0.5)
    return magnitude

# Calculate the magnetic field's angle phi
def mag_phi(components):
    x = components[0]
    y = components[1]
    z = components[2]
    adj = ((x**2) + (y**2))**(0.5)
    phi = np.arctan(z/adj)
    return phi

# Calculate the magnetic field's angle theta
def mag_theta(components):
    x = components[0]
    y = components[1]
    theta = np.arctan(y/x)
    return theta

#Calculates the magnetic field from a straight wire
def magStraightWire(I, radial_dist):
    B = mu*I/(2.*pi*radial_dist)
    return B
################################################################################
# Initialize probes
l = 15*2*pi
rad = 4.67
probes = np.array([[L*rad, 0. , 0], [0, 0 , rad*L], [-rad*L, 0, 0],
                   [0, 0 , -L*rad], [rad*L, l*2 , 0], [0, l*2, rad*L],
                   [-rad*L, l*2 , 0], [0, l*2, -rad*L]])

#Initialize path/wire
phi = np.linspace(-48.*pi, 48.*pi, n)
const1 = 10. #slope of x-direction
path = np.array([const1*0.*phi, .01*phi,0.005*phi]).T
mass = np.ones((n,1))
wr = Wire(path, path*0., mass, I, r=0.3, Bp=1)

#Calculate the magnetic field vector at probes[0]
# All values in SI units
Bvec = biot_savart_SI(probes[0], I, wr.p, delta = 0.01)

print(Bvec)#print vector
print(mag(Bvec))#print magnitude of vector
print(magStraightWire(I, probes[0][0]))#print magnitude of wire assuming ideal straight wire formula

Bz = Bvec[2]
Bphi = magStraightWire(I, probes[0][0])
pred_theta = np.arcsin(Bz/Bphi)#predicted theta from Bz=Bphi*sin(x) method (assuming ideal wire condition)
theta = mag_theta(Bvec)#actual theta from components

#print('Predicted: ' + str(pred_theta))
#print('Simulated: ' + str(theta))

wr.show()
mlab.points3d(probes[:,0], probes[:,1], probes[:,2], scale_factor=4)
mlab.show()
