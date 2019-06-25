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

# returns a list of all the angles of the magnetic field based on the position
def angleList(B_array, coordinate):
    ang_list = []
    if coordinate[0] == 0:
        for i in range(len(B_array)):
            angle = mag_phi(B_array[i])
            ang_list.append(angle)
    if coordinate[2] == 0:
        for i in range(len(B_array)):
            angle = mag_theta(B_array[i])
            #print(angle)
            #print(B_array[i][0])
            ang_list.append(angle)
    #print(ang_dict)
    return ang_list

# returns the max angle of the magnetic field using angleList()
def maxAngle(B_array, coordinate):
    ang_dict = angleDict(B_array, coordinate)
    max = list(ang_dict.values())[0]
    for i in ang_dict.values():
        if i >max:
            max = i
    B = list(ang_dict.keys())[list(ang_dict.values()).index(max)]
    #print('The max angle is ' + str(max) + ' at the magnetic vector ' + str(B))
    return [B, max];

################ Single loop wire ################
### Initialize path
phi = np.linspace(0.,36*pi,n) + 0.5
path = np.array([L*np.cos(phi),15*phi,L*np.sin(phi)]).T
path[:,1] -= path[0,1]
### Initialize mass
mass = np.ones((n,1))*dm
### Create wire
wr = Wire(path,path*0,mass,I,r=.3,Bp=1)
##################################################
# Initialize length of grid arrays
lmda = 15*2*pi
rad = 4.67
lmda_scl = range(1, 26) #NOTE: both ranges must be the same size
L_scl = range(1, 26)
time = range(0, 100)

# Create a series of points
probes = np.array([[rad*L, lmda*2 , 0], [0, lmda*2 , rad*L], [-rad*L, lmda*2, 0]])
probes0 = np.array([[0, lmda*2 , -L*rad], [rad*L, lmda*2.33 , 0], [0, lmda*3, rad*L]])
probes1 = np.array([[-rad*L, lmda*3 , 0], [0, lmda*3, -rad*L], [rad*L, lmda*4, 0]])
probes2 = np.array([[0, lmda*4, rad*L], [-rad*L, lmda*4, 0], [0, lmda*4, -rad*L]])

# Initialize the grid arrays for the contour
lambda_array = []
radius_array = []
B_array = []
angleCol_array = []
angleCross_array = []

# Iterate and produce grid array for helix's radius
for k in L_scl:
    phi = np.linspace(0.,36*pi,n) + np.pi
    path = np.array([L*np.cos(phi)/k,15*phi,L*np.sin(phi)/k]).T
    path[:,1] -= path[0,1]
    wr = Wire(path,path*0,mass,I,r=.3,Bp=1)

    radius_array.append(L/k)

# Iterate through the grid arrays and calculate magnetic field and phi and theta
for j in lmda_scl:
    B_L = []
    angleCol_L = []
    angleCross_L = []
    for k in L_scl:
        phi = np.linspace(0.,36*pi,n) + np.pi
        path = np.array([L*np.cos(phi)/k,15*phi/j,L*np.sin(phi)/k]).T
        path[:,1] -= path[0,1]
        wr = Wire(path,path*0,mass,I,r=.3,Bp=1)

        B = biot_savart(probes[0], I, wr.p, delta = 0.1)
        B_L.append(mag(B))
        angleCol_L.append(angleCol(B, probes[0]))
        angleCross_L.append(angleCrossSection(B, probes[0]))

    lambda_array.append(lmda/j) # Produce gird array for the helix's lambda
    B_array.append(B_L)
    angleCol_array.append(angleCol_L)
    angleCross_array.append(angleCross_L)

#print(B_array)
#print(angleCol_array)
#print(angleCross_array)

# Plot different contour maps
fig = plt.figure(1)
ax = fig.add_subplot(111)
CS = ax.contourf(lambda_array, radius_array, angleCol_array)
CB = fig.colorbar(CS, shrink=0.8, extend='both')
plt.ylabel('Radius of Helix Current [cm]')
plt.xlabel('Lambda [cm]')
plt.title('Angle of Magnetic Field Vector Relative to the Column')

fig = plt.figure(2)
ax = fig.add_subplot(111)
ES = ax.contourf(lambda_array, radius_array, angleCross_array)
EB = fig.colorbar(ES, shrink=0.8, extend='both')
plt.ylabel('Radius of Helix Current [cm]')
plt.xlabel('Lambda [cm]')
plt.title('Angle of Magnetic Field Vector Relative to the Cross Section Plane')

fig = plt.figure(3)
ax = fig.add_subplot(111)
DS = ax.contourf(lambda_array, radius_array, B_array)
DB = fig.colorbar(DS, shrink=0.8, extend='both')
plt.ylabel('Radius of Helix Current [cm]')
plt.xlabel('Lambda [cm]')
plt.title('Magnitude of Magnetic Field')
plt.show()
