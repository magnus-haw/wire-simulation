
""" This code constructs a up-to-date model of the solenoid used for calibration
 and calculates the magnetic field at the center"""

import numpy as np
import math
import mayavi.mlab as mlab 
import sys,os
cwd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cwd + "/classes/")

from Constants import mu0, pi, Kb,amu,mass_elec,elec
from Engine import MultiWireEngine
from Wires import Wire,State
from Utility import biot_savart_SI

import matplotlib.pyplot as plt
import random
from math import atan, sqrt
from scipy import interpolate
from sympy import Symbol, cos, sin, solve
from sympy.vector import CoordSys3D, Vector
N = CoordSys3D('N')

### Dimensional scales
L = .04 #m
L1 = 0.042 #m
I = 3. #A
B0 = mu0*I/(2*pi*L) #Tesla
v0 = 150 #m/s
tau = L/v0 #s
dt = .02*tau
n=10000
mu = pi*4e-7

dr = 1.
dm = pi*dr

print("L (m)", L)
print("I (A)", B0)
print("tau (s)", tau)

######################### General Function(s) ##################################

def magnitude(vector_array):
    x = vector_array[0]
    y = vector_array[1]
    z = vector_array[2]
    mag = sqrt(x**2. + y**2. + z**2.)
    return mag

################################################################################
#Probe located at the origin
probes = np.array([[0., 0., 0.]])

pi_to_len_ratio = 2.*29.360679005512086340 #determines the spacing between turns
n_turns = 53 #number of turns

#Path 1 -> first layer of turns
phi0 = np.linspace(-pi,pi,n)
path0 = np.array([L*np.cos(n_turns*phi0),phi0/pi_to_len_ratio,L*np.sin(n_turns*phi0)]).T
mass = np.ones((n,1))
wr0 = Wire(path0,path0*0,mass,I,r=.001)

#Path 2 -> second layer of turns
phi1 = np.linspace(-pi,pi,n)
path1 = np.array([L1*np.cos(53.*phi1),phi1/pi_to_len_ratio,L1*np.sin(53.*phi1)]).T
mass = np.ones((n,1))
wr1 = Wire(path1,path1*0,mass,I,r=.001)

#Calculate the magnetic field at the origin
B0 = biot_savart_SI(probes[0], I, wr0.p, delta = 0.01)
B1 = biot_savart_SI(probes[0], I, wr1.p, delta = 0.01)
B = [B0[0] + B1[0], B0[1] + B1[1], B0[2] + B1[2]]
mag = magnitude(B)
print(mag) #Teslas
print(mag*10000.) #Gauss

#Display the modeled solenoid and the probe
wr0.show()
wr1.show()
mlab.points3d(0., 0., 0., scale_factor=0.01)
mlab.show()
