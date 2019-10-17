#############################################################
'''
Script demonstrating the Bfield plotting capability:
   Shows the transitioning magnetic field for a stationary
   loop with increasing current in a dipole background field
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
dm = pi*r*r*(loop_len/L0)*rho/n

################ Single loop wire ################
### Initialize path
phi = np.linspace(0.,pi,n)
path = np.array([L*np.cos(phi),0*phi,L*np.sin(phi)]).T
path[:,1] -= path[0,1]
### Initialize mass
height = L*np.sin(phi)
mass = np.ones((n,1))*np.vstack(np.exp(-height/(0.75*L)))*2*dm

### Create wire 
wr = Wire(path,path*0,mass,I,r=r,Bp=1,is_fixed=False)
##################################################




############### Create intial state ##############
st = State('single_loop_test',load=0)
st.items.append(wr)
##################################################


############## Run simulation engine #############
sim = MultiWireEngine(st,dt)
X,Y,Z,dx,dy,dz = get_rect_grid([-2,2],[-1,1],[0.3,2],10)


for i in [0,0,0]:
    st.items[2].I = np.sin(i*pi/600.)
    sim.state = st
    bx,by,bz = sim.getB(X,Y,Z)
    st.show()
    Bsrc = mlab.pipeline.vector_field(X,Y,Z,bx,by,bz)
    Bmag = mlab.pipeline.extract_vector_norm(Bsrc)
    ##iso = mlab.pipeline.iso_surface(Bmag, opacity=0.3)
    ##vec = mlab.pipeline.vectors(Bsrc,scale_factor=1)
    streamline = mlab.pipeline.streamline(Bmag, seedtype='plane',
                                        seed_visible=False,
                                        seed_resolution=5,
                                        integration_direction = 'both')
    streamline.seed.widget.center = np.array([ 1.33226763e-15, -4.17973529e-02,  1.11280360e+00])
    streamline.seed.widget.normal = np.array([1., 0., 0.])
    streamline.seed.widget.origin = np.array([ 1.33226763e-15, -9.04779744e-01,  2.96643737e-01])
    streamline.seed.widget.point1 = np.array([1.33226763e-15, 8.21185039e-01, 2.96643737e-01])
    streamline.seed.widget.point2 = np.array([ 1.33226763e-15, -9.04779744e-01,  1.92896347e+00])
    streamline.seed.widget.enabled = True

    fig = mlab.gcf()
    fig.scene.z_plus_view()
    streamline.seed.widget.enabled = False
    fig.scene.render()
    mlab.savefig("shear-{:0>4}.png".format(i),size=(500,500),figure=fig)
##    mlab.show()
    mlab.clf()
    
##################################################


##################################################
