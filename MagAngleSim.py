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
R = 0.07
probes = np.array([[R, 0. , 0], [0, R, 0], [-R, 0, 0],
                   [0, -R, 0]])

#Initialize path/wire
phi = np.linspace(-pi/2, pi/2, n)
path = np.array([0.*phi, 0.02*np.sin(phi), (phi/pi)/1.5]).T
mass = np.ones((n,1))
wr = Wire(path, path*0., mass, I, r=0.01, Bp=1)

#Calculate the magnetic field vector at probes[0]
# All values in SI units
Bvec = biot_savart_SI(probes[0], I, wr.p, delta = 0.01)

print(Bvec)#print vector
print(mag(Bvec))#print magnitude of vector
print(magStraightWire(I, probes[0][0]))#print magnitude of wire assuming ideal straight wire formula

Bz = Bvec[2]
Bphi = magStraightWire(I, probes[0][0])
pred_theta = np.arcsin(Bz/Bphi)#predicted theta from Bz=Bphi*sin(x) method (assuming ideal wire condition)
theta = np.pi/2 - mag_theta(Bvec)#actual theta from components

print('Predicted: ' + str(pred_theta*180/pi))
print('Simulated: ' + str(theta))

wr.show()
mlab.points3d(probes[:,0], probes[:,1], probes[:,2], scale_factor=0.05)
mlab.show()

################################################################################
#Helical Test Case

phi = np.linspace(-pi/2, pi/2, n)
path = np.array([0.02*np.cos(phi), 0.02*np.sin(phi), (phi/pi)/1.5]).T
mass = np.ones((n, 1))
wr = Wire(path, path*0, mass, I, r=0.01, Bp=1)

Bvec = biot_savart_SI(probes[0], I, wr.p, delta = 0.01)
Bz = Bvec[2]
Bphi = magStraightWire(I, probes[0][0])
pred_theta = np.arcsin(Bz/Bphi)#predicted theta from Bz=Bphi*sin(x) method (assuming ideal wire condition)
print('For helical test case ' + str(pred_theta*180/pi))

wr.show()
mlab.points3d(probes[:,0], probes[:,1], probes[:,2], scale_factor=0.05)
mlab.show()
