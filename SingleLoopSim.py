#NOTE: Max angle project begins at line 228

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

# returns a dictionary of all the angles of the magnetic field based on the position
def angleDict(B_array, coordinate):
    ang_dict = {}
    if coordinate[0] == 0:
        for i in range(len(B_array)):
            angle = mag_phi(B_array[i])
            ang_dict.update({B_array[i] : angle})
    if coordinate[2] == 0:
        for i in range(len(B_array)):
            angle = mag_theta(B_array[i])
            #print(angle)
            #print(B_array[i][0])
            ang_dict.update({tuple(B_array[i]) : angle})
    #print(ang_dict)
    return ang_dict

# returns the max angle of the magnetic field
def maxAngle(B_array, coordinate):
    ang_dict = angleDict(B_array, coordinate)
    max = list(ang_dict.values())[0]
    for i in ang_dict.values():
        if i >max:
            max = i
    B = list(ang_dict.keys())[list(ang_dict.values()).index(max)]
    #print('The max angle is ' + str(max) + ' at the magnetic vector ' + str(B))
    return [B, max];

# Calculate and produce a dictionary w/ the probe's distance from each path point
def distance(position, path):
    x = position[0]
    y = position[1]
    z = position[2]
    path_list = path.tolist()
    dist_dict = {}
    for i in range(len(path_list)):
        dist = math.sqrt((path_list[i][0] - x)**2 + (path_list[i][1] - y)**2 + (path_list[i][2] - z)**2)
        dist_dict.update({tuple(path_list[i]) : dist})
    return dist_dict

# Finds the closest distance from the dictionary of all distances
def closest(position, path):
    dist_dict = distance(position, path)
    min = list(dist_dict.values())[0]
    for i in dist_dict.values():
        if i < min:
            min = i
    return min

################ Single loop wire ################
#L = 8*L
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
lmda = 15*2*pi
rad = 4.67
time = range(0, 100)

# Create a series of points
probes = np.array([[rad*L, lmda*2 , 0], [0, lmda*2 , rad*L], [-rad*L, lmda*2, 0]])
probes0 = np.array([[0, lmda*2 , -L*rad], [rad*L, lmda*2.33 , 0], [0, lmda*3, rad*L]])
probes1 = np.array([[-rad*L, lmda*3 , 0], [0, lmda*3, -rad*L], [rad*L, lmda*4, 0]])
probes2 = np.array([[0, lmda*4, rad*L], [-rad*L, lmda*4, 0], [0, lmda*4, -rad*L]])

# magnitude of magnetic field of five probes in a time-series
B_mag = []
B_mag1 = []
B_mag2 = []
B_mag3 = []
B_mag4 = []

# Create a time series with wavelength:
for j in time:
    phi = np.linspace(0.,36*pi,n) + np.pi*j/50.
    path = np.array([L*np.cos(phi),15*phi,L*np.sin(phi)]).T
    path[:,1] -= path[0,1]
    wr = Wire(path,path*0,mass,I,r=.3,Bp=1)

    # Calculate the magnetic field at each instance
    B = biot_savart(probes[0], I, wr.p, delta = 0.1)
    B1 = biot_savart(probes[1], I, wr.p, delta = 0.1)
    B2 = biot_savart(probes[2], I, wr.p, delta = 0.1)
    B3 = biot_savart(probes0[0], I, wr.p, delta = 0.1)
    B4 = biot_savart(probes0[1], I, wr.p, delta = 0.1)
    B_mag.append(mag(B))
    B_mag1.append(mag(B1))
    B_mag2.append(mag(B2))
    B_mag3.append(mag(B3))
    B_mag4.append(mag(B4))

plt.figure(1)
plt.plot(time, B_mag, 'ro', label="(x, 0, 0)")
plt.plot(time, B_mag1, 'bo', label="(0, 0, z)")
plt.plot(time, B_mag2, 'go', label="(-x, 0, 0)")
plt.plot(time, B_mag3, 'yo', label="(0, 0, -z)")
plt.plot(time, B_mag4, 'mo', label="(x, y, 0)")
plt.ylabel('Magnitude of Magnetic Field [T]')
plt.xlabel('Time')
plt.legend()
plt.title('B Field vs. Time')

# varying the wavelength of the helix and plotting the magnitude of the magnetic field
B1_lam = []
B2_lam = []
B3_lam = []
B4_lam = []

# Producing a time series for four different wavelengths
for i in time:
    # wavelength:
    phi = np.linspace(0.,36*pi,n) + np.pi*i/50.
    path = np.array([L*np.cos(phi),15*phi,L*np.sin(phi)]).T
    path[:,1] -= path[0,1]
    wr = Wire(path,path*0,mass,I,r=.3,Bp=1)

    # wavelength:
    phi1 = np.linspace(0.,36*pi,n) + np.pi*i/50.
    path1 = np.array([L*np.cos(phi1),5*phi1,L*np.sin(phi1)]).T
    path1[:,1] -= path1[0,1]
    wr1 = Wire(path1,path1*0,mass,I,r=.3,Bp=1)

    # wavelength:
    phi2 = np.linspace(0.,36*pi,n) + np.pi*i/50.
    path2 = np.array([L*np.cos(phi2),phi2,L*np.sin(phi2)]).T
    path2[:,1] -= path2[0,1]
    wr2 = Wire(path2,path2*0,mass,I,r=.3,Bp=1)

    # wavelength:
    phi3 = np.linspace(0.,36*pi,n) + np.pi*i/50.
    path3 = np.array([L*np.cos(phi3),phi3/2,L*np.sin(phi3)]).T
    path3[:,1] -= path3[0,1]
    wr3 = Wire(path3,path3*0,mass,I,r=.3,Bp=1)

    # Calculate the magnetic field at each instance
    B = biot_savart(probes[0], I, wr.p, delta = 0.1)
    B1 = biot_savart(probes[0], I, wr1.p, delta = 0.1)
    B2 = biot_savart(probes[0], I, wr2.p, delta = 0.1)
    B3 = biot_savart(probes[0], I, wr3.p, delta = 0.1)
    B1_lam.append(mag(B))
    B2_lam.append(mag(B1))
    B3_lam.append(mag(B2))
    B4_lam.append(mag(B3))

plt.figure(2)
plt.plot(time, B1_lam, 'ro', label="lambda")
plt.plot(time, B2_lam, 'bo', label="lambda/3")
plt.plot(time, B3_lam, 'go', label="lambda/15")
plt.plot(time, B4_lam, 'mo', label="lambda/30")
plt.ylabel('Magnitude of Magnetic Field [T]')
plt.xlabel('Time')
plt.title('B Field vs. Time')
plt.legend()

################# Max Angle Analysis ##########################
B_array = []

# Producing a time series at wavelength:
for l in time:
    phi = np.linspace(0.,36*pi,n) + np.pi*l/50.
    path = np.array([L*np.cos(phi),5*phi,L*np.sin(phi)]).T
    path[:,1] -= path[0,1]
    wr = Wire(path,path*0,mass,I,r=.3,Bp=1)

    # Calculate magnetic field and its components
    B = biot_savart(probes[0], I, wr.p, delta = 0.1)
    B_array.append(B)

max_angle = maxAngle(B_array, probes[0])

# Calculating the magnitude of each vector and indexing the instance of max angle
B_array_mag = []
t = 0.
B_max_angle_mag = 0.
for m in B_array:
    if mag(m) == mag(max_angle[0]):
        t = B_array.index(m)
        B_max_angle_mag = mag(m)
    B_array_mag.append(mag(m))

fig = plt.figure(3)
ax = fig.add_subplot(111)
plt.plot(time, B_array_mag, 'ro', label = 'Magnetic Field')
plt.ylabel('Magnitude of Magnetic Field [T]')
plt.xlabel('Time')
plt.title('B Field vs. Time')
string = 'max angle = ' + str(max_angle[1]) + ' rad'
ax.annotate(string, xy = (t, B_max_angle_mag))
plt.legend()
plt.show()
#########################################################################

# plotting the components of the magnetic field at one probe in a time series
Bx = []
By = []
Bz = []

# Producing a time series at wavelength:
for k in time:
    phi = np.linspace(0.,36*pi,n) + np.pi*k/50.
    path = np.array([L*np.cos(phi),5*phi,L*np.sin(phi)]).T
    path[:,1] -= path[0,1]
    wr = Wire(path,path*0,mass,I,r=.3,Bp=1)

    # Calculate magnetic field and its components
    B = biot_savart(probes[0], I, wr.p, delta = 0.1)
    x = B[0]
    y = B[1]
    z = B[2]
    Bx.append(x)
    By.append(y)
    Bz.append(z)

plt.figure(4)
plt.plot(time, Bx, 'ro', label = 'x-component')
plt.plot(time, By, 'bo', label = 'y-component')
plt.plot(time, Bz, 'go', label = 'z-component')
plt.ylabel('Magnitude of Magnetic Field [T]')
plt.xlabel('Time')
plt.title('B Field vs. Time')
plt.legend()
plt.show()

## plotting the time series of the vector of the magnetic field
mlab.points3d(Bx, By, Bz)
mlab.points3d(0., 0., 0., name = 'Origin', color = (1., 0., 0.), scale_factor = 0.001)
mlab.title('Magnetic Field Vector at Probe')
mlab.axes(ranges = [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5], x_axis_visibility=True, y_axis_visibility=True, z_axis_visibility=True)
mlab.show()

# ################ Footpoint coils #################
# ### Initialize path
# phi = np.linspace(0.,2*pi,50)
# path0 = np.array([(L/4)*np.cos(phi)-L,(L/4)*np.sin(phi),0*phi-1]).T
# path1 = np.array([(L/4)*np.cos(phi)+L,(L/4)*np.sin(phi),0*phi-1]).T
# ### Initialize mass
# mass = np.ones((len(path0),1))
# ### Create coils
# coil0 = Wire(path0,path0*0,mass,-1,is_fixed=True,r=.1)
# coil1 = Wire(path1,path1*0,mass,1,is_fixed=True,r=.1)
# ##################################################



# ############### Create intial state ##############
# st = State('single_loop_test',load=0)
# st.items.append(wr)
# st.items.append(coil0)
# st.items.append(coil1)
# #st.show()
# #mlab.show()
# #st.save()
# ##################################################
#
#
# ############## Run simulation engine #############
# #sim = MultiWireEngine(st,dt)
# #for i in range(0,500):
#     #new_st = sim.advance()
#
#     #if i%10 == 0:
#         #new_st.show()
#         #mlab.show()
#         #forces = sim.forceScheme()[0]
#         #plt.plot(forces[:,0],forces[:,2])
#         #plt.show()
# ##################################################
#
#
# ################# Plot Results ###################
# plt.figure(0)
# plt.title("forces")
# forces = sim.forceScheme()[0]
# plt.plot(forces[:,0],forces[:,2])
#
# plt.figure(1)
# plt.title("position")
# wire = sim.state.items[0]
# plt.plot(wire.p[:,0],wire.p[:,2],'bo')
# plt.show()
#
# #new_st.show()
# mlab.show()
# ##################################################
