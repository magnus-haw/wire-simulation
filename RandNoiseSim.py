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
import random

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
time = range(0, 500)

# Create a series of points, NOTE that probes[0] was place at mid point of the current
probes = np.array([[rad*L, lmda*9 , 0], [0, lmda*2 , rad*L], [-rad*L, lmda*2, 0]])
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

def avgPlot(noisy_data):
    avg_plot = []
    avg_temp = []
    for i in noisy_data:
        if len(avg_temp) == 9:
            avg_temp.append(i)
            avg = 0
            for j in avg_temp:
                avg += j
            avg = avg/10.
            avg_plot.append(avg)
            avg_temp = []
        else:
            avg_temp.append(i)
    return avg_plot

def changeTimeScale(axis, length):
    scaled_axis = []
    count = 0
    for i in axis:
        scaled_axis.append(10*i)
        count += 1
        if count >= length:
            return scaled_axis

############# Random Noise in both x, y, z directions #################

# magnitude of magnetic field of five probes in a time-series
B_mag = []
B_mag_noise = []
B_mag_noise1 = []

# Create a time series with no offset
for i in time:
    phi = np.linspace(0.,36*pi,n) + np.pi*i/250.
    path = np.array([L*np.cos(phi),15*phi,L*np.sin(phi)]).T
    path[:,1] -= path[0,1]
    ### Initialize mass
    mass = np.ones((n,1))*dm
    ### Create wire
    wr = Wire(path,path*0,mass,I,r=0.3,Bp=1)

    B = biot_savart(probes[0], I, wr.p, delta = 0.1)
    B_mag.append(mag(B))

for j in time:
    phi = np.linspace(0.,36*pi,n) + np.pi*j/250.
    path_noise = np.array([L*np.cos(phi),15*phi,L*np.sin(phi)]).T

    # Add random noise
    for i in range(len(path_noise)):
        path_noise[i][0]= path_noise[i][0] + random.uniform(-1., 1.)
        path_noise[i][1]= path_noise[i][1] + random.uniform(-1., 1.)
        path_noise[i][2]= path_noise[i][2] + random.uniform(-1., 1.)

    path_noise[:,1] -= path_noise[0,1]
    ### Initialize mass
    mass = np.ones((n,1))*dm
    ### Create wire
    wr_noise = Wire(path_noise,path_noise*0,mass,I,r=0.3,Bp=1)

    # Calculate the magnetic field at each instance
    B_noise = biot_savart(probes[0], I, wr_noise.p, delta = 0.1)
    B_mag_noise.append(mag(B_noise))

for k in time:
    phi = np.linspace(0.,36*pi,n) + np.pi*k/250.
    path_noise1 = np.array([L*np.cos(phi),15*phi,L*np.sin(phi)]).T

    # Add random noise
    for i in range(len(path_noise1)):
        path_noise1[i][0]= path_noise1[i][0] + random.uniform(-3., 3.)
        path_noise1[i][1]= path_noise1[i][1] + random.uniform(-3., 3.)
        path_noise1[i][2]= path_noise1[i][2] + random.uniform(-3., 3.)

    path_noise1[:,1] -= path_noise1[0,1]
    ### Initialize mass
    mass = np.ones((n,1))*dm
    ### Create wire
    wr_noise1 = Wire(path_noise1,path_noise1*0,mass,I,r=0.3,Bp=1)

    # Calculate the magnetic field at each instance
    B_noise1 = biot_savart(probes[0], I, wr_noise1.p, delta = 0.1)
    B_mag_noise1.append(mag(B_noise1))

# Plot simulation and data
wr.show()
wr_noise.show()
wr_noise1.show()
mlab.show()

plt.figure(1)
plt.plot(time, B_mag, 'r-', label="ideal helix")
plt.plot(time, B_mag_noise, 'b-', label="with random noise; delta = 0.0-1.0")
plt.plot(time, B_mag_noise1, 'g-', label="with randome noise; delta = 0.0-3.0")
plt.ylabel('Magnitude of Magnetic Field [T]')
plt.xlabel('Time')
plt.legend()
plt.title('B Field vs. Time (Randome 3D Noise)')

############## Random Noise (varying lambda) #################

# magnitude of magnetic field of five probes in a time-series
B_mag = []
B_mag_lmda_ns = []
B_mag_lmda_ns1 = []

# Create a time series with no noise
for i in time:
    phi = np.linspace(0.,36*pi,n) + np.pi*i/250.
    path = np.array([L*np.cos(phi),15*phi,L*np.sin(phi)]).T
    path[:,1] -= path[0,1]
    ### Initialize mass
    mass = np.ones((n,1))*dm
    ### Create wire
    wr = Wire(path,path*0,mass,I,r=0.3,Bp=1)

    B = biot_savart(probes[0], I, wr.p, delta = 0.1)
    B_mag.append(mag(B))

for j in time:
    phi = np.linspace(0.,36*pi,n) + np.pi*j/250.
    path_noise = np.array([L*np.cos(phi),15*phi*random.uniform(0.99, 1.01),L*np.sin(phi)]).T
    path_noise[:,1] -= path_noise[0,1]
    ### Initialize mass
    mass = np.ones((n,1))*dm
    ### Create wire
    wr_noise = Wire(path_noise,path_noise*0,mass,I,r=0.3,Bp=1)

    # Calculate the magnetic field at each instance
    B_noise = biot_savart(probes[0], I, wr_noise.p, delta = 0.1)
    B_mag_lmda_ns.append(mag(B_noise))

B_noise_avg = avgPlot(B_mag_noise)
time_5 = changeTimeScale(time, len(B_noise_avg))

for k in time:
    phi = np.linspace(0.,36*pi,n) + np.pi*k/250.
    path_noise1 = np.array([L*np.cos(phi),15*phi*random.uniform(0.97, 1.03),L*np.sin(phi)]).T
    path_noise1[:,1] -= path_noise1[0,1]
    ### Initialize mass
    mass = np.ones((n,1))*dm
    ### Create wire
    wr_noise1 = Wire(path_noise1,path_noise1*0,mass,I,r=0.3,Bp=1)

    # Calculate the magnetic field at each instance
    B_noise1 = biot_savart(probes[0], I, wr_noise1.p, delta = 0.1)
    B_mag_lmda_ns1.append(mag(B_noise1))

# Plot simulation and data
wr.show()
wr_noise.show()
wr_noise1.show()
mlab.show()

plt.figure(2)
plt.plot(time, B_mag, 'r-', label="ideal helix")
plt.plot(time, B_mag_lmda_ns, 'b-', label="with random lambda noise; delta = 0.99-1.01")
plt.plot(time, B_mag_lmda_ns1, 'y-', label="with randome lambda noise; delta = 0.97-1.03")
plt.ylabel('Magnitude of Magnetic Field [T]')
plt.xlabel('Time')
plt.legend()
plt.title('B Field vs. Time (Random Lambda Noise)')
plt.show()
