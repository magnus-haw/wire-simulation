import time
import numpy as np

from Constants import mu0, pi, Kb,amu,mass_elec,elec
from Engine import MultiWireEngine
from Wires import Wire
from State import State
from Utility import getBField,biot_savart,JxB_force

import matplotlib.pyplot as plt

### Non-dimensional parameters
n = 500
L = 1000
r = 1.
I = 1.
rho = 1.

for n in [250,500,1000]:
    ############## Single straight wire ##############
    ### Initialize path
    z = np.linspace(-L/2.,L/2.,n)
    path = np.array([0*z,0*z,z]).T
    ### Initialize mass
    mass = np.ones((n,1))

    ### Create wire 
    wr = Wire(path,path*0,mass,I)
    ##################################################

    ############### B-field calculation ##############
    x = np.linspace(r,10*r,n)
    points = np.array([x,0*x,0*x]).T
    Bsim = getBField(points,[path],[I],.01)

    Beq = I/x

    ##plt.plot(r,Beq,'k-')
    ##plt.plot(r,Bsim[:,1],'ro')
    plt.plot(x,Bsim[:,1]/Beq,'-',label="Nodes per length=%.2f"%(n/1000.))
    ##################################################
plt.legend(loc=0)
plt.ylabel(r'|B$_{s}$/B|')
plt.xlabel('Distance from Wire')
plt.title('Straight wire convergence test')
plt.show()





### Non-dimensional parameters
n = 500
R = 100
L = 2*pi*R
r = 1.
I = 1.
rho = 1.

for n in [50,100,250]:
    ################ Single loop wire ################
    ### Initialize path
    phi = np.linspace(0.,2*pi,n)
    path = np.array([(R)*np.cos(phi),(R)*np.sin(phi),0*phi]).T

    ### Initialize mass
    mass = np.ones((n,1))

    ### Create wire 
    wr = Wire(path,path*0,mass,I)
    ##################################################

    ############### B-field calculation ##############
    z = np.linspace(0,3*R,n)
    points = np.array([0*z,0*z,z]).T
    Bsim = getBField(points,[path],[I],.01)

    Bz = (I/2.)*(2*pi*R*R)/((z*z + R*R)**1.5)
    ##plt.plot(r,Beq,'k-')
    ##plt.plot(r,Bsim[:,1],'ro')
    plt.plot(z,Bsim[:,2]/Bz,'-',label="Nodes per length=%.2f"%(n/L))
    ##################################################
plt.legend(loc=0)
plt.ylabel(r'|B$_{s}$/B|')
plt.xlabel('Distance from Loop Plane')
plt.title('Loop Bz convergence test')
plt.show()




### Non-dimensional parameters
n=1000
L = 1.
I = 1.
rho = 1.

############### Two straight wires ##############
### Initialize path
z = np.linspace(-50*L,50*L,n)
path1 = np.array([-.5+0*z,0*z,z]).T
path2 = np.array([.5+0*z,0*z,z]).T
### Initialize mass
mass = np.ones((n,1))

### Create wire 
wr1 = Wire(path1,path1*0,mass,I)
wr2 = Wire(path2,path2*0,mass,I)

### Calculate forces on wire1
B = getBField(wr1.p,[wr2.p],[wr2.I],delta=.01)
forces = JxB_force(wr1.p,wr1.I,B)
force_per_length = forces/(z[1]-z[0])
##################################################

plt.plot(z,force_per_length[:,0],'-',label="(JxB)_x")
##################################################
plt.legend(loc=0)
plt.ylabel(r'Normalized Force [$\mu_0I_0^2/(2\pi L_0)$]')
plt.xlabel(r'Distance along wire [$L_0$]')
plt.title('Parallel wire force test')
plt.show()
