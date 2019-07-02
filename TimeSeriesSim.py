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
import scipy.fftpack
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

### Non-dimensional parameters
L = L0/r0
dr = 1.
dt = .02
I = 1.
rho = 1.
dm = pi*dr

###### General functions ######

# Calculate the magnitude of magnetic field w/ its components
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

# Calculate the angle relative to the column
def angleCol(B, coordinate):
    if coordinate[0] == 0:
        angle = mag_phi(B)
    if coordinate[2] == 0:
        angle = mag_theta(B)
    return angle

#Calculate the angle relative to cross section plane
def angleCrossSection(B, coordinate):
    if coordinate[0] == 0:
        angle = mag_theta(B)
    if coordinate[2] == 0:
        angle = mag_phi(B)
    return angle

def timeSeries(lmbda, radius, time = 100, noise_x = 0., noise_y = 0., noise_z = 0.):

    #Lambda is measured in centimeters
    #Radius is measured in centimeters
    rad = 4.67
    probes = np.array([[4., lmbda*9 , 0], [0, lmbda*9 , 4.]])
    lmbda = lmbda/(2.*np.pi)
    t = range(0, time)


    B0_mag = []
    B0_angleCol = []
    B0_angleCross = []
    B1_mag = []
    B1_angleCol = []
    B1_angleCross = []

    for i in t:
        phi = np.linspace(0.,36*pi,n) + np.pi*3.*i/(time)
        path_noise = np.array([radius*np.cos(phi),lmbda*phi,radius*np.sin(phi)]).T

        # Add random noise
        for i in range(len(path_noise)):
            #unecessary v
            randFloatx = random.uniform(-noise_x, noise_x)
            randFloaty = random.uniform(-noise_y, noise_y)
            randFloatz = random.uniform(-noise_z, noise_z)
                #unecessary ^
            path_noise[i][0]= path_noise[i][0] + randFloatx
            path_noise[i][1]= path_noise[i][1] + randFloaty
            path_noise[i][2]= path_noise[i][2] + randFloatz

        path_noise[:,1] -= path_noise[0,1]
        ### Initialize mass
        mass = np.ones((n,1))*dm
        ### Create wire
        wr_noise = Wire(path_noise,path_noise*0,mass,I,r=0.3,Bp=1)

        # Calculate the magnetic field at each instance
        B0 = biot_savart(probes[0], I, wr_noise.p, delta = 0.1)
        B1 = biot_savart(probes[1], I, wr_noise.p, delta = 0.1)

        B0_mag.append(mag(B0))
        B1_mag.append(mag(B1))
        B0_angleCol.append(angleCol(B0, probes[0]))
        B1_angleCol.append(angleCol(B1, probes[1]))
        B0_angleCross.append(angleCrossSection(B0, probes[0]))
        B1_angleCross.append(angleCrossSection(B1, probes[1]))

    plt.figure(1)
    plt.plot(t, B0_mag, 'b-', label="Magnitude")
    plt.plot(t, B0_angleCol, 'g-', label="Angle relative to column")
    plt.plot(t, B0_angleCross, 'r-', label="Angle relative to cross section plane")
    plt.plot(t, B1_mag, 'b-', label="Magnitude")
    plt.plot(t, B1_angleCol, 'g-', label="Angle relative to column")
    plt.plot(t, B1_angleCross, 'r-', label="Angle relative to cross section plane")
    plt.ylabel('Magnitude of Magnetic Field [T]')
    plt.xlabel('Time')
    plt.legend()
    plt.title('B Field vs. Time')

    plt.show()

    return [B0_mag, B0_angleCol, B0_angleCross, B1_mag, B1_angleCol, B1_angleCross]

timeSeries(50., 3., 100, .1, .1, .1)
