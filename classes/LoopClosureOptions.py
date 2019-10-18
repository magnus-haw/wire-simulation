#############################################################
'''
Script demonstrating different loop closure options:
   Shows the different forces and inductances from different
   footpoint closures
'''
#############################################################

import numpy as np
import mayavi.mlab as mlab
import sys,os
cwd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cwd + "/classes/")

from Utility import get_rect_grid
from Constants import mu0, pi, Kb,amu,mass_elec,elec
from Engine import MultiWireEngine
from Wires import Wire
from State import State

import matplotlib.pyplot as plt

### Dimensional scales
L0 = 0.04 #m
r0 = 0.008 #m
I0 = 30000. #Amps
nden0 = 1e22 #m^-3
n = 100

### Derived scales
rho0 = nden0*amu*40. #kg/m^3
B0 = I0*mu0/(2*pi*L0) #tesla
vA0 = B0/np.sqrt(mu0*rho0)
tau0 = L0/vA0 #s

### Total mass
loop_len = 2*pi*L0 #m
m0 = rho0*pi*r0*r0*loop_len

print("L0 (m)", L0)
print("B0 (T)", B0)
print("rho0 (kg/m^3)", rho0)
print("tau (s)", tau0)
print("vA (m/s)", vA0)
print("m (kg)", m0)

### Non-dimensional parameters
dt = .001
L = 1.
I = 1.
rho = 1.5
Bp = 1.

r = r0/L0
dm = 1.

################## No Closure ###################
### Initialize base path
phi0 = np.linspace(-pi/6.,7*pi/6,n)
path0 = np.array([L*np.cos(phi0),0*phi0,L*np.sin(phi0)+.5]).T

### Create wire
path = path0
mass = np.ones((len(path),1))
wr = Wire(path,path*0,mass,I,r=r,is_fixed=False)
##################################################

################## Circle Closure ##################
##### Initialize base path
##phi0 = np.linspace(-pi/6.,7*pi/6,n)
##path0 = np.array([L*np.cos(phi0),0*phi0,L*np.sin(phi0)+.5]).T
##
##### Regular circle
##phi = np.linspace(7*pi/6,11*pi/6.,n)
##pathcirc = np.array([L*np.cos(phi),0*phi,L*np.sin(phi)+.501]).T
##
##### Create wire
##path = np.append(path0,pathcirc,axis=0)
##mass = np.ones((len(path),1))
##wr = Wire(path,path*0,mass,I,r=r,is_fixed=False)
####################################################

################## Mirror Closure ##################
##### Initialize base path
##phi0 = np.linspace(-pi/6.,7*pi/6,n)
##path0 = np.array([L*np.cos(phi0),0*phi0,L*np.sin(phi0)+.5]).T
##
##### Mirror closure
##pathmirror = np.array([L*np.cos(phi0[::-1]),0*phi0,-L*np.sin(phi0[::-1])-.500001]).T
##
##### Create wire
##path = np.append(path0,pathmirror,axis=0)
##mass = np.ones((len(path),1))
##wr = Wire(path,path*0,mass,I,r=r,is_fixed=False)
####################################################

################# Vertical Closure #################
##### Initialize base path
##phi0 = np.linspace(-pi/6.,7*pi/6,n)
##path0 = np.array([L*np.cos(phi0),0*phi0,L*np.sin(phi0)+.5]).T
##
##### Vertical closure
##z = np.arange(-2,0,.1)
##pathvert1 = np.array([0*z,0*z,z]).T + path0[0]
##pathvert2 = np.array([0*z,0*z,z[::-1]]).T + path0[-1]
##
##### Create wire
##path = np.append(pathvert1,path0,axis=0)
##path = np.append(path,pathvert2,axis=0)
##mass = np.ones((len(path),1))
##wr = Wire(path,path*0,mass,I,r=r,is_fixed=False)
####################################################

################# Horizontal Closure ###############
##### Initialize base path
##phi0 = np.linspace(-pi/6.,7*pi/6,n)
##path0 = np.array([L*np.cos(phi0),0*phi0,L*np.sin(phi0)+.5]).T
##
##### Horizontal closure
##z = np.arange(0,10,1)
##pathhoriz1 = np.array([z[::-1]/6,0*z,-np.arctan(z[::-1])/2]).T + path0[0]*.99
##pathhoriz2 = np.array([-z/6,0*z,-np.arctan(z)/2]).T + path0[-1]*.99
##
##### Create wire
##path = np.append(pathhoriz1,path0,axis=0)
##path = np.append(path,pathhoriz2,axis=0)
##mass = np.ones((len(path),1))
##wr = Wire(path[::-1],path*0,mass,I,r=r,is_fixed=False)
####################################################

wr.interpolate()
wr.show()

###################### Surface ###################
blue = (0.34765625,0.5625,0.84375)
[x,y] = np.mgrid[-2:2:.2,-2:2:.2]
z = x*0
mlab.mesh(x, y, z, color=blue)
##################################################
fig = mlab.gcf()
fig.scene.camera.position = [-0.5286122421224984, 7.073519564703222, 0.41797665332279077]
fig.scene.camera.focal_point = [-0.023579024481236895, -0.11013972148999236, -0.008308510934655955]
fig.scene.camera.view_angle = 30.0
fig.scene.camera.view_up = [-0.0037900187195243585, -0.05950183212861701, 0.9982210014477975]
fig.scene.camera.clipping_range = [2.9170787976767643, 12.55754248035167]
fig.scene.camera.compute_view_plane_normal()
fig.scene.render()

mlab.show()

############### Create intial state ##############
st = State('closure_test',load=0)
st.items.append(wr)
#st.save()
##################################################


############## Run simulation engine #############
sim = MultiWireEngine(st,dt)

f = sim.forceScheme()[0]
print(f[:,2].argmax(),f[:,2].max())
##################################################


##################################################
