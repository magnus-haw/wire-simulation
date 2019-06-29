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
from Utility import biot_savart

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
n = 1000

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

###### Initialize Probe Locations and Scalars ########

lmda = 15*2*pi
rad = 4.67
time = range(0, 100)

# Create a series of points
probes = np.array([[rad*L, lmda*2 , 0], [0, lmda*2 , rad*L], [-rad*L, lmda*2, 0]])
probes0 = np.array([[0, lmda*2 , -L*rad], [rad*L, lmda*2.33 , 0], [0, lmda*3, rad*L]])
probes1 = np.array([[-rad*L, lmda*3 , 0], [0, lmda*3, -rad*L], [rad*L, lmda*4, 0]])
probes2 = np.array([[0, lmda*4, rad*L], [-rad*L, lmda*4, 0], [0, lmda*4, -rad*L]])

###### General functions ######
# Calculate the magnitude of magnetic field w/ its components
def mag(components):
    x = components[0]
    y = components[1]
    z = components[2]
    magnitude = ((x**2) + (y**2) + (z**2))**(0.5)
    return magnitude

################ Single loop wire ################
### Initialize path
phi = np.linspace(0.,36*pi,n) + 0.5
path = np.array([L*np.cos(phi),15*phi,L*np.sin(phi)]).T
path[:,1] -= path[0,1]
### Initialize mass
mass = np.ones((n,1))*dm
### Create wire
wr = Wire(path,path*0,mass,I,r=0.3,Bp=1)
wr.show()
mlab.show()
##################################################

# magnitude of magnetic field of five probes in a time-series
B_mag = []
B_mag_off_negx = []
B_mag_off_posz = []

# Create a time series with no offset
for i in time:
    phi = np.linspace(0.,36*pi,n) + np.pi*i/50.
    path = np.array([L*np.cos(phi),15*phi,L*np.sin(phi)]).T
    path[:,1] -= path[0,1]
    ### Initialize mass
    mass = np.ones((n,1))*dm
    ### Create wire
    wr = Wire(path,path*0,mass,I,r=0.3,Bp=1)

    B = biot_savart(probes[0], I, wr.p, delta = 0.1)
    B_mag.append(mag(B))

for j in time:
    phi = np.linspace(0.,36*pi,n) + np.pi*j/50.
    path_off_negx = np.array([L*np.cos(phi),15*phi,L*np.sin(phi)]).T

    # Add an offset to the path in the -x direction
    for i in range(len(path_off_negx)):
        path_off_negx[i][0]= path_off_negx[i][0] - 1

    path_off_negx[:,1] -= path_off_negx[0,1]
    ### Initialize mass
    mass = np.ones((n,1))*dm
    ### Create wire
    wr_off_negx = Wire(path_off_negx,path_off_negx*0,mass,I,r=0.3,Bp=1)

    # Calculate the magnetic field at each instance
    B_off_negx = biot_savart(probes[0], I, wr_off_negx.p, delta = 0.1)
    B_mag_off_negx.append(mag(B_off_negx))

for k in time:
    phi = np.linspace(0.,36*pi,n) + np.pi*k/50.
    path_off_posz = np.array([L*np.cos(phi),15*phi,L*np.sin(phi)]).T

    # Add an offset to the path in the -x direction
    for i in range(len(path_off_posz)):
        path_off_posz[i][2]= path_off_posz[i][2] + 1

    path_off_posz[:,1] -= path_off_posz[0,1]
    ### Initialize mass
    mass = np.ones((n,1))*dm
    ### Create wire
    wr_off_posz = Wire(path_off_posz,path_off_posz*0,mass,I,r=0.3,Bp=1)

    # Calculate the magnetic field at each instance
    B_off_posz = biot_savart(probes[0], I, wr_off_posz.p, delta = 0.1)
    B_mag_off_posz.append(mag(B_off_posz))

plt.figure(1)
plt.plot(time, B_mag_off_negx, 'bo', label="with offset in -x dir")
plt.plot(time, B_mag_off_posz, 'go', label="with offset in z dir")
plt.plot(time, B_mag, color = (0.278791, 0.062145, 0.386592, 1.), label="with no offset")
plt.ylabel('Magnitude of Magnetic Field [T]')
plt.xlabel('Time')
plt.legend()
plt.title('B Field vs. Time')
plt.show()
