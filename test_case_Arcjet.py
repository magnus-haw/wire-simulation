import numpy as np
import mayavi.mlab as mlab
import sys,os
cwd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cwd + "/classes/")

from Constants import mu0, pi, Kb,amu,mass_elec,elec
from Wires import Wire,State
from Utility import biot_savart,getBField, get_R, get_normal

import matplotlib.pyplot as plt

### Dimensional scales
r0 = 0.01 #m
L0 = .04 #m
I0 = 10000. #Amps
nden0 = 1e21 #m^-3
rho0 = nden0*amu*40. #kg/m^3
B0 = I0*mu0/(2*pi*r0) #tesla
vA0 = B0/np.sqrt(mu0*rho0)
tau0 = L0/vA0 #s
m0 = rho0*pi*r0*r0*L0
n = 110

print("L0 (m)", L0)
print("B0 (T)", B0)
print("rho0 (kg/m^3)", rho0)
print("tau (s)", tau0)
print("vA (m/s)", vA0)
print("m (kg)", m0)

### Non-dimensional parameters
L = L0/r0
dr = 1.
dt = .02
I = 1.
rho = 1.
dm = pi*dr

################ Single loop wire ################
### Initialize path
phi = np.linspace(0.,2*pi,n)
path = np.array([L*np.cos(phi),0*phi,L*np.sin(phi)]).T
#path[:,1] -= path[0,1]
### Initialize mass
mass = np.ones((n,1))*dm
### Create wire 
wr = Wire(path,path*0,mass,I,r=.3,Bp=1)
##################################################

probes = np.array([[0,.1,0],[0,.3,0],[0,.5,0]])

for i in range(0,len(probes)):
    B = biot_savart(probes[i], I, wr.p, delta=.01)
    print(B,probes[i])
mlab.points3d(probes.T)
wr.show()
mlab.axes()
mlab.show()
