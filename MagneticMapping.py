
"""This file to meant to simulate the algorithm that maps magnetic data to a
three-dimensional model and can be applied to real data by commenting out
simulation code and uncommenting real data code"""

import numpy as np
import math
import mayavi.mlab as mlab
import scipy.io
mat = scipy.io.loadmat('c11_027_5.mat')
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
from scipy import interpolate, signal
from sympy import Symbol, cos, sin, solve
from sympy.vector import CoordSys3D, Vector
N = CoordSys3D('N')

### Dimensional scales
L = .03 #m
I = 1500. #A
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

#Retrieve the data from each channel and certain parameters
channel_A = mat.get('A')
channel_B = mat.get('B')
channel_C = mat.get('C')
Tinterval = mat.get('Tinterval')[0][0]
sample_rate = 1/(Tinterval*100.)
print("Tinterval (sec)", Tinterval)
print("Sample Rate (Hz)", sample_rate)

######################## General Functions #####################################

# Calculate the magnitude of magnetic field w/ its components
def mag(vector):
    magnitude = (vector.dot(vector))**(0.5)
    return magnitude

# Calculate the magnetic field's angle phi
def phi_angle(vector, sympy_vec = False):
    if sympy_vec:
        x = vector.dot(N.i)
        y = vector.dot(N.j)
        z = vector.dot(N.k)
        adj = ((x**2) + (y**2))**(0.5)
        phi = atan(z/adj)
        return phi
    x = vector[0]
    y = vector[1]
    z = vector[2]
    adj = ((x**2) + (y**2))**(0.5)
    phi = atan(z/adj)
    return phi

# Calculate the magnetic field's angle theta
def theta_angle(vector):
    x = vector[0]
    y = vector[1]
    theta = atan(y/x)
    return theta

# Determine the distance between sensor and current using ideal (straight) wire scenario
def radialConstraint(vector, current):
    bmag = mag(vector)
    R = mu*I/(bmag*2.*np.pi)
    return R

# Finds where the current is most likely located
def intersection(R, position, B, n_vec):
    #n_vec is an array (e.g. [0, 0, 1]) that represents the direction of the current
    x = position[0]
    y = position[1]
    z = position[2]


    if n_vec[0] == 0 and n_vec[1] == 0: #if current is travelling in this direction
        #vecotrize position of probe
        pos_vec = position[0]*N.i + position[1]*N.j + position[2]*N.k
        #vectorize the magnetic field at the probe
        Bvec = B[0]*N.i + B[1]*N.j + B[2]*N.k
        #Normalize the B vector
        Bvec_norm = scale_vec(Bvec)
        #Find vector perpendicular to B vector
        v1 = -B[1]*N.i + B[0]*N.j + 0.*N.k
        #Cross B vector and v1 to find second vector that is perpendicular to B vector
        v2 = v1.cross(Bvec)
        #Scale vector v1 and v2 to be length R
        v1, v2 = scale_vec(v1, R), scale_vec(v2, R)
        #finds the vector that points to the location of the current
        v, vec_array = minDistance_Z(position, v1, v2)
        #puts the vector as an point array
        int_pnt = [v.dot(N.i), v.dot(N.j), v.dot(N.k)]

    # THIS METHOD SHOULD FUNCTION FOR ARBITRARY DIRECTION OF CURRENT

    return int_pnt, vec_array

#Buils an array that represents a circle from the inputs
#Finds the point that intersects on the plane perpendicular to the z axis
def minDistance_Z(pos, v1, v2):
    min_vec = None
    vec_array = []
    min_z = 1000. #Can be any large number that is surely larger the expected min
    for t in range(0, 200):
        #Construct circle
        v = pos[0]*N.i + pos[1]*N.j + pos[2]*N.k + v1*np.cos(t*0.005*2*np.pi) + v2*np.sin(t*0.005*2*np.pi)
        #represent the vec as a point array
        vec = [v.dot(N.i), v.dot(N.j), v.dot(N.k)]
        vec_array.append(vec)
        #Finds the point with the min distance from the xy plane
        if abs(v.dot(N.k)-pos[2]) < min_z and abs(v.dot(N.i)) < R and abs(v.dot(N.j)) < R:
            min_vec = v
            min_z = abs(v.dot(N.k)-pos[2])
    return min_vec, vec_array

#Scale any vector to the length desired
def scale_vec(vector, length = 1):
    mag_vec = mag(vector)
    scaled_vec = vector*length/mag_vec
    return scaled_vec

#Iterates through a constructed path and returns an array that represents the angle of that path
def angle(points_array):
    #points_array is an array that represents all the positions of the
    # helix at each instance
    theta_array = []
    phi_array = []
    for i in range(len(points_array) - 1):
        x0 = points_array[i][0]
        y0 = points_array[i][1]
        z0 = points_array[i][2]
        x1 = points_array[i+1][0]
        y1 = points_array[i+1][1]
        z1 = points_array[i+1][2]

        x = x1 - x0
        y = y1 - y0
        z = z1 - z0

        vec = [x, y, z]
        theta = theta_angle(vec)
        phi = phi_angle(vec)

        theta_array.append(theta)
        phi_array.append(phi)

    return theta_array, phi_array

#Finds a point from a path that is closest to the inputted point
def findCommonPoint(point, path):
    z = point[2]
    for i in range(len(path) - 1):
        if (path[i][2] <= z) and (path[i + 1][2] >= z):
            return i
#Calculate percent error between two vectors
def percentErrorVec(obs_vec, act_vec):
    obs_x = obs_vec[0]
    obs_y = obs_vec[1]
    obs_z = obs_vec[2]

    act_x = act_vec[0]
    act_y = act_vec[1]
    act_z = act_vec[2]

    pEVx = 100.*(act_x - obs_x)/act_x
    pEVy = 100.*(act_y - obs_y)/act_y
    pEVz = 100.*(act_z - obs_z)/act_z
    return np.mean([pEVx, pEVy, pEVz])

#Calculate distance between two vector endpoints
def dist(vec0, vec1):
    # does NOT account for z-deviation because it is the most accurate it can be
    # where z-deviation purely depends on resolution
    dist = sqrt((vec1[0]-vec0[0])**2. + (vec1[1]-vec0[1])**2.)
    return dist

#Converts the data from voltage to magnetic field
def cleanData(A, B, C):
    num_samp = round((10e-3)/Tinterval)#Numebr of samples in 10ms
    for i in range(1): #Only looks at a section of the array
        x1 = int(125000*i)
        x2 = int(x1 + round(num_samp/2))
        print(x1, x2)
        x = np.linspace(x1, x2, (x2-x1+1))
        y = np.linspace(1, 1000, 1000)

        t1 = round(x1*Tinterval)
        t2 = round(x2*Tinterval)
        t = np.linspace(t1, t2, (x2-x1+1))

        #Calibration variables
        B_sol = 29.63920288354406/10000 #Teslas
        Va = 98.81e-3
        Va_off = 36.8e-3
        Vb = -22.71e-3
        Vb_off = -52.43e-3
        Vc = -54.33e-3
        Vc_off = -14.87e-3
        Vc_off = Vc - (Vc_off - Vc) #With CURRENT calibration data, the hall
        #sensor for channel C was inserted the wrong direction. This is
        #redundant, where one can simply find the difference and take the absolute

        #Calculate T per V from calibration
        BpVa = B_sol/(Va - Va_off)
        BpVb = B_sol/(Vb - Vb_off)
        BpVc = B_sol/(Vc - Vc_off)

        #Grab the mean of each channel's sample
        mean_a = np.mean(A[x1:x2])
        mean_b = np.mean(B[x1:x2])
        mean_c = np.mean(C[x1:x2])
        A2 = A[x1:x2]
        B2 = B[x1:x2]
        C2 = C[x1:x2]

        for j in range(len(A2)):
        #Calculate the offset mean and subtract that from the channel
            A2[j] = A[j] - mean_a
            B2[j] = B[j] - mean_b
            C2[j] = C[j] - mean_c

        Am, Bm, Cm = list(range(len(A2))), list(range(len(B2))), list(range(len(C2)))
        #Calculate the offset mean and subtract that from the channel
        Di = np.mean(np.mean(A2)+np.mean(B2)+np.mean(C2))
        for j in range(len(A2)):
            #Removing a common node noise and converting to magnetic units
            Am[j] = ((-Di+A2[j])-Va_off)*BpVa
            Bm[j] = ((-Di+B2[j])-Vb_off)*BpVb
            Cm[j] = ((-Di+C2[j])-Vc_off)*BpVc

        #Manually adjusting channel's A and B so it is centered around 0
        mean_B = np.mean(Bm)
        mean_A = np.mean(Am)
        for k in range(len(Bm)):
            Am[k] = Am[k] - mean_A
            Bm[k] = Bm[k] - mean_B

        plt.figure(10)
        plt.plot(Am, label='Channel A')
        plt.plot(Bm, label='Channel B')
        plt.plot(Cm, label='Channel C')
        plt.title('Magnetic Field vs. Time-Series (Samples)')
        plt.ylabel('Magnetic Field [Teslas]')
        plt.xlabel('Samples')
        plt.legend()
        plt.show()

    return Am, Bm, Cm
################################################################################

#Produce modified channel data for analysis
channel_A, channel_B, channel_C = cleanData(channel_A, channel_B, channel_C)

Bvec_array0 = []
pnt_array0 = []
Bvec_array1 = []
pnt_array1 = []
Bvec_array2 = []
pnt_array2 = []
Bvec_array3 = []
pnt_array3 = []

dist_error_array = []

# t_rang = range(len(channel_A))
t_rng = range(300, 3000)
for i in t_rng:
    # Initialize probes
    R = 0.06
    k = i*100 #Look at every 100 points rather than each one
    time_int = k*Tinterval
    dist = time_int*v0
    print(dist)
    const = dist
    probes = np.array([[R, 0. , const], [0, R, const], [-R, 0, const],
                       [0, -R, const]])
    n_vec = [0, 0, 1]

    #Initialize path/wire
    phi = np.linspace(-8.*np.pi, 8.*np.pi, n)
    path = np.array([0.003*np.cos(phi), 0.003*np.sin(phi), (phi/pi)/1.5]).T

#-----------------------Add Random Noise to Path-------------------------------#

    for i in range(len(path)):
        path[i][0] = random.uniform(0.9*path[i][0], 1.1*path[i][0])
        path[i][1] = random.uniform(0.9*path[i][1], 1.1*path[i][1])

#------------------------------------------------------------------------------#

    mass = np.ones((n,1))

#------------------------Add Random Noise to Current---------------------------#
    I_noise = I
    I_noise = random.uniform(1.*I, 1.*I)
#------------------------------------------------------------------------------#

    wr = Wire(path, path*0., mass, I_noise, r=0.001, Bp=1)

    Calculate the magnetic field vector at probes[0]
    All values in SI units
    Bvec0 = biot_savart_SI(probes[0], I_noise, wr.p, delta = 0.01)
    Bvec1 = biot_savart_SI(probes[1], I_noise, wr.p, delta = 0.01)
    Bvec2 = biot_savart_SI(probes[2], I_noise, wr.p, delta = 0.01)
    Bvec3 = biot_savart_SI(probes[3], I_noise, wr.p, delta = 0.01)
    Bvec_array0.append(Bvec0)
    Bvec_array1.append(Bvec1)
    Bvec_array2.append(Bvec2)
    Bvec_array3.append(Bvec3)

    # Bvec0 = np.array([channel_A[i][0], channel_B[i][0], channel_C[i][0]]) #Uncomment if using actual data

    r_dist0 = radialConstraint(Bvec0, I)
    r_dist1 = radialConstraint(Bvec1, I_noise)
    r_dist2 = radialConstraint(Bvec2, I_noise)
    r_dist3 = radialConstraint(Bvec3, I_noise)
    pnt0, vec_array0 = intersection(r_dist0, probes[0], Bvec0, n_vec)
    pnt1, vec_array1 = intersection(r_dist1, probes[1], Bvec1, n_vec)
    pnt2, vec_array2 = intersection(r_dist2, probes[2], Bvec2, n_vec)
    pnt3, vec_array3 = intersection(r_dist3, probes[3], Bvec3, n_vec)

    wr.show()
    mlab.points3d(R, 0., 0., scale_factor=0.01)
    mlab.points3d(vec_array0[:,0], vec_array0[:,1], vec_array0[:,2], scale_factor=0.01, color=(0,1,0))
    mlab.show()

# ########################## For Actual Test Data ################################

    #Verify that the components are in the right place

    # Bvec0 = np.array([channel_A[k][0], channel_C[k][0], channel_B[k][0]])
    # print('Count: ' + str(i - 300))
    # print(Bvec0)
    # Bvec_array0.append(Bvec0)
    # r_dist0 = radialConstraint(Bvec0, I)
    # print(r_dist0)
    # pnt0, vec_array0 = intersection(r_dist0, probes[0], Bvec0, n_vec)
    # vec_array0 = np.array(vec_array0, dtype='float')
    #
    # mlab.points3d(vec_array0[:,0], vec_array0[:,1], vec_array0[:,2], scale_factor=0.01, color=(0,1,0))
    # mlab.points3d(R, 0., 0., scale_factor=0.01)
    # mlab.points3d(0., 0., 0., scale_factor=0.01, color=(1,0,0))
    # mlab.show()

################################################################################

#-------------------Calculate Deviation from True Value------------------------#

    ind = findCommonPoint(pnt0, path)
    path_vec = path[i]
    distance = dist(pnt0, path_vec)
    print('Distance between vectors = ' + str(distance))
    print('\n')

    dist_error_array.append(distance)

#------------------------------------------------------------------------------#
    pnt_array0.append(pnt0)
    pnt_array1.append(pnt1)
    pnt_array2.append(pnt2)
    pnt_array3.append(pnt3)

pnt_array0 = np.array(pnt_array0, dtype='float')
pnt_array1 = np.array(pnt_array1, dtype='float')
pnt_array2 = np.array(pnt_array2, dtype='float')
pnt_array3 = np.array(pnt_array3, dtype='float')

#---------------------------Interpolate Points---------------------------------#

x, y, z = pnt_array0[:,0], pnt_array0[:,1], pnt_array0[:,2]

fx = interpolate.interp1d(z, x)
fy = interpolate.interp1d(z, y)
znew = np.linspace(z.min(), z.max(), num = len(pnt_array0), endpoint=True)
xnew = fx(znew)
ynew = fy(znew)
path0 = np.array([xnew, ynew, znew]).T
mass = np.ones((len(pnt_array0),1))
wr0 = Wire(path0, path0*0, mass, I, r=0.0001, Bp=1)

#------------------------------------------------------------------------------#

#-------------------------Display Distance Deviation---------------------------#
mean_dist = np.mean(dist_error_array)
print('Average Distance Error = ' + str(mean_dist))
#------------------------------------------------------------------------------#

mlab.points3d(pnt_array0[:,0], pnt_array0[:,1], pnt_array0[:,2], scale_factor=0.002)
mlab.points3d(pnt_array1[:,0], pnt_array1[:,1], pnt_array1[:,2], scale_factor=0.002, color=(1,0,0))
mlab.points3d(pnt_array2[:,0], pnt_array2[:,1], pnt_array2[:,2], scale_factor=0.002, color=(0,0,1))
mlab.points3d(pnt_array3[:,0], pnt_array3[:,1], pnt_array3[:,2], scale_factor=0.002, color=(0,1,0))
mlab.points3d(R, 0., 0., scale_factor=0.01)
mlab.points3d(0., 0., 0., scale_factor=0.01, color=(1,0,0))
mlab.points3d(-R, 0., 0., scale_factor=0.01, color=(0,0,1))
mlab.points3d(0., -R, 0., scale_factor=0.01, color=(0,1,0))
wr.show()
wr0.show()
mlab.show()

#-------------------------Calcualte Average Amplitude--------------------------#
amplitude = []
for i in pnt_array0:
    dist = sqrt(i[0]**2 + i[1]**2)
    amplitude.append(dist)

avg_offset = np.mean(amplitude)
amp = []
for i in amplitude:
    amp.append(amplitude - avg_offset)

mean_amp = np.mean(abs(amp))
amp = np.array(amp)
max_a = amp.max()
min_a = amp.min()
print('AVERAGE AMPLITUDE = ' + str(mean_amp))
print(max_a, min_a)
#------------------------------------------------------------------------------#

#Call angle() to see how the angle progresses with time
angle_array0 = angle(pnt_array0)
angle_array1 = angle(pnt_array1)
angle_array2 = angle(pnt_array2)
angle_array3 = angle(pnt_array3)

plt.figure(0)
plt.plot(angle_array0[0], 'r-', label='theta of probes[0]')
plt.plot(angle_array1[0], 'b-', label=' theta of probes[1]')
plt.plot(angle_array2[0], 'g-', label='theta of probes[2]')
plt.plot(angle_array3[0], 'm-', label='theta of probes[3]')
plt.title('Theta vs. Time-Series (Samples)')
plt.xlabel('Samples')
plt.ylabel('Theta (xz-plane)')
plt.legend()

plt.figure(1)
plt.plot(angle_array0[1], 'r-', label='phi of probes[0]')
plt.plot(angle_array1[1], 'b-', label='phi of probes[1]')
plt.plot(angle_array2[1], 'g-', label='phi of probes[2]')
plt.plot(angle_array3[1], 'm-', label='phi of probes[3]')
plt.title('Phi vs. Time-Series (Samples)')
plt.xlabel('Samples')
plt.ylabel('Phi (yz-plane)')
plt.legend()

Bvec_array0 = np.array(Bvec_array0)

plt.figure(2)
plt.plot(Bvec_array0[:,0], label='x-coordinate')
plt.plot(Bvec_array0[:,2], label='y-corrdinate')
plt.plot(Bvec_array0[:,1], label='z-coordinate')
plt.title('Arcjet Simulation: Magnetic Field vs. Time-Series (Samples)')
plt.ylabel('Magnetic Field [Teslas]')
plt.xlabel('Samples')
plt.legend()

plt.show()
