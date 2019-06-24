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
n = 250

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

######## Magnitude and Anuglar Components #########
def mag(components):
    x = components[0]
    y = components[1]
    z = components[2]
    magnitude = ((x**2) + (y**2) + (z**2))**(0.5)
    return magnitude

def mag_phi(components):
    x = components[0]
    y = components[1]
    z = components[2]
    adj = ((x**2) + (y**2))**(0.5)
    phi = np.arctan(z/adj)
    return phi

def mag_theta(components):
    x = components[0]
    y = components[1]
    theta = np.arctan(y/x)
    return theta

def distance(position, path):
    x = position[0]
    y = position[1]
    z = position[2]
    path_list = path.tolist()
    #print(path_list)
    dist_dict = {}
    for i in range(len(path_list)):
        #for j in range(3):
            #print(path_list[i][j])
        dist = math.sqrt((path_list[i][0] - x)**2 + (path_list[i][1] - y)**2 + (path_list[i][2] - z)**2)
        dist_dict.update({tuple(path_list[i]) : dist})
    return dist_dict

def closest(position, path):
    dist_dict = distance(position, path)
    min = list(dist_dict.values())[0]
    for i in dist_dict.values():
        if i < min:
            min = i
    return min

################ Single loop wire ################
### Initialize path
phi = np.linspace(0.,12*pi,n) + 0.5
path = np.array([L*np.cos(phi),15*phi,L*np.sin(phi)]).T
path[:,1] -= path[0,1]
### Initialize mass
mass = np.ones((n,1))*dm
### Create wire
wr = Wire(path,path*0,mass,I,r=.3,Bp=1)
##################################################
lmda = 15*2*pi
rad = 4.67
#print(L/lmda)
probes = np.array([[rad*L, lmda*2 , 0], [0, lmda*2 , rad*L], [-rad*L, lmda*2, 0]])
probes0 = np.array([[0, lmda*2 , -L*rad], [rad*L, lmda*2.33 , 0], [0, lmda*3, rad*L]])
probes1 = np.array([[-rad*L, lmda*3 , 0], [0, lmda*3, -rad*L], [rad*L, lmda*4, 0]])
probes2 = np.array([[0, lmda*4, rad*L], [-rad*L, lmda*4, 0], [0, lmda*4, -rad*L]])
count = 1
Blist = []
Dlist = []

myB = []
myB1 = []
myB2 = []
myB3 = []
myB4 = []
time = range(0, 100)
for j in time:
    phi = np.linspace(0.,12*pi,n) + np.pi*j/50.
    path = np.array([L*np.cos(phi),15*phi,L*np.sin(phi)]).T
    path[:,1] -= path[0,1]
    wr = Wire(path,path*0,mass,I,r=.3,Bp=1)

    B = biot_savart(probes[0], I, wr.p, delta = 0.1)
    B1 = biot_savart(probes[1], I, wr.p, delta = 0.1)
    B2 = biot_savart(probes[2], I, wr.p, delta = 0.1)
    B3 = biot_savart(probes0[0], I, wr.p, delta = 0.1)
    B4 = biot_savart(probes0[1], I, wr.p, delta = 0.1)
    myB.append(mag(B))
    myB1.append(mag(B1))
    myB2.append(mag(B2))
    myB3.append(mag(B3))
    myB4.append(mag(B4))
#Red is (x, 0, 0), blue is (0, 0, z), green is (-x, 0, 0), yellow is (0, 0, -z), and purple is (x, y, 0)
plt.figure(1)
plt.plot(time, myB, 'ro', time, myB1, 'bo', time, myB2, 'go', time, myB3, 'yo', time, myB4, 'mo')
plt.ylabel('Magnitude of Magnetic Field [T]')
plt.xlabel('Time')
plt.title('B Field vs. Time (*lambda = pi*(1.2))')

myB_a = []
myB1_a = []
for i in time:
    phi1 = np.linspace(0.,12*pi,n) + np.pi*j/50.
    path1 = np.array([L*np.cos(phi1),5*phi1,L*np.sin(phi1)]).T
    path1[:,1] -= path1[0,1]
    wr = Wire(path1,path1*0,mass,I,r=.3,Bp=1)

    B_a = biot_savart(probes[0], I, wr.p, delta = 0.1)
    B1_a = biot_savart(probes[1], I, wr.p, delta = 0.1)
    myB_a.append(mag(B_a))
    myB1_a.append(mag(B1_a))
#Red is (x, 0, 0), blue is (0, 0, z), green is (-x, 0, 0), yellow is (0, 0, -z), and purple is (x, y, 0)
plt.figure(2)
plt.plot(time, myB, 'ro', time, myB1, 'bo')
plt.ylabel('Magnitude of Magnetic Field [T]')
plt.xlabel('Time')
plt.title('B Field vs. Time (*lambda = pi*(0.4))')

#for i in range(len(probes)):
    #B = biot_savart(probes[i], I, wr.p, delta = 0.1)
    #B0 = biot_savart(probes0[i], I, wr.p, delta = 0.1)
    #B1 = biot_savart(probes1[i], I, wr.p, delta = 0.1)
    #B2 = biot_savart(probes2[i], I, wr.p, delta = 0.1)
    #d = closest(probes[i], path)
    #d0 = closest(probes0[i], path)
    #d1 = closest(probes1[i], path)
    #d2 = closest(probes2[i], path)
    #Blist.append(mag(B))
    #Blist.append(mag(B0))
    #Blist.append(mag(B1))
    #Blist.append(mag(B2))
    #Dlist.append(d)
    #Dlist.append(d0)
    #Dlist.append(d1)
    #Dlist.append(d2)
    #print('Coordinate ' + str(i + count) + ': ' + str(B) + ', MAG = ' + str(mag(B)) + ', PHI = ' + str(mag_phi(B)) + ', THETA = ' + str(mag_theta(B)))
    #print('Coordinate ' + str(i + count + 1) + ': ' + str(B0) + ', MAG = ' + str(mag(B0)) + ', PHI = ' + str(mag_phi(B0)) + ', THETA = ' + str(mag_theta(B0)))
    #print('Coordinate ' + str(i + count + 2) + ': ' + str(B1) + ', MAG = ' + str(mag(B1)) + ', PHI = ' + str(mag_phi(B1)) + ', THETA = ' + str(mag_theta(B1)))
    #rint('Coordinate ' + str(i + count + 3) + ': ' + str(B2) + ', MAG = ' + str(mag(B2)) + ', PHI = ' + str(mag_phi(B2)) + ', THETA = ' + str(mag_theta(B2)))
    #count = count + 1
mlab.points3d(probes.T[0], probes.T[1], probes.T[2])
mlab.points3d(probes0.T[0], probes0.T[1], probes0.T[2])
mlab.points3d(probes1.T[0], probes1.T[1], probes1.T[2])
mlab.points3d(probes2.T[0], probes2.T[1], probes2.T[2])
wr.show()
mlab.show()
#plt.plot(Dlist, Blist, 'ro')
#plt.ylabel('Magnitude of Magnetic Field [T]')
#plt.xlabel('Closest Distance from Wire [m]')
#plt.title('B Field vs. Distance (lambda = pi*(1.2))')
plt.show()

################ Footpoint coils #################
### Initialize path
phi = np.linspace(0.,2*pi,50)
path0 = np.array([(L/4)*np.cos(phi)-L,(L/4)*np.sin(phi),0*phi-1]).T
path1 = np.array([(L/4)*np.cos(phi)+L,(L/4)*np.sin(phi),0*phi-1]).T
### Initialize mass
mass = np.ones((len(path0),1))
### Create coils
coil0 = Wire(path0,path0*0,mass,-1,is_fixed=True,r=.1)
coil1 = Wire(path1,path1*0,mass,1,is_fixed=True,r=.1)
##################################################



############### Create intial state ##############
st = State('single_loop_test',load=0)
st.items.append(wr)
st.items.append(coil0)
st.items.append(coil1)
#st.show()
#mlab.show()
#st.save()
##################################################


############## Run simulation engine #############
#sim = MultiWireEngine(st,dt)
#for i in range(0,500):
    #new_st = sim.advance()

    #if i%10 == 0:
        #new_st.show()
        #mlab.show()
        #forces = sim.forceScheme()[0]
        #plt.plot(forces[:,0],forces[:,2])
        #plt.show()
##################################################


################# Plot Results ###################
plt.figure(0)
plt.title("forces")
forces = sim.forceScheme()[0]
plt.plot(forces[:,0],forces[:,2])

plt.figure(1)
plt.title("position")
wire = sim.state.items[0]
plt.plot(wire.p[:,0],wire.p[:,2],'bo')
plt.show()

#new_st.show()
mlab.show()
##################################################
