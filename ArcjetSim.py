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
L = .03 #m
I = 1. #A
B0 = mu0*I/(2*pi*L) #Tesla
v0 = 150 #m/s
tau = L/v0 #s
dt = .02*tau

print("L (m)", L)
print("I (A)", B0)
print("tau (s)", tau)


################### Single wire ##################
### Initialize path
phi = np.linspace(0.,12*pi,n)
path = np.array([L*np.cos(phi),15*phi,L*np.sin(phi)]).T
### Initialize mass
mass = np.ones((n,1))
### Create wire
wr = Wire(path,path*0,mass,I,r=.3,Bp=1)
##################################################
lmda = 15*2*pi
rad = 4.67


########## Define probe positions ################
probes = np.array([[rad*L, lmda*2 , 0],
                   [0, lmda*2 , rad*L],
                   [-rad*L, lmda*2, 0],
                   [0, lmda*2 , -L*rad],
                   [rad*L, lmda*2.33 , 0],
                   [0, lmda*3, rad*L],
                   [-rad*L, lmda*3 , 0],
                   [0, lmda*3, -rad*L],
                   [rad*L, lmda*4, 0],
                   [0, lmda*4, rad*L],
                   [-rad*L, lmda*4, 0],
                   [0, lmda*4, -rad*L]])
###################################################


################ Calculate time series ############
timesteps = 100
Bx = np.zeros((timesteps,len(probes)))
By = np.zeros((timesteps,len(probes)))
Bz = np.zeros((timesteps,len(probes)))

time = range(0, timesteps)
for j in time:
    phi = np.linspace(0.,12*pi,n) + np.pi*j/50.
    path = np.array([L*np.cos(phi),15*phi,L*np.sin(phi)]).T
    wr = Wire(path,path*0,mass,I,r=.3)

    for p in range(0,len(probes)):        
        Bvec = biot_savart_SI(probes[p], I, wr.p, delta = 0.01)
        Bx[j,p] = Bvec[0]
        By[j,p] = Bvec[1]
        Bz[j,p] = Bvec[2]
Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)*B0
###################################################


############### Plot B-field Magnitude ############
plt.figure(1)

for p in range(0,1):
    plt.plot(Bmag[:,p])
plt.xlabel('Time')
plt.ylabel('Magnitude of Magnetic Field [T]')
plt.title('B Field vs. Time (*lambda = pi*(1.2))')
plt.legend()
plt.show()
###################################################


####### Plot probes and current channel ###########
mlab.points3d(probes[:,0], probes[:,1], probes[:,2],scale_factor=4)
wr.show()
mlab.show()
###################################################



