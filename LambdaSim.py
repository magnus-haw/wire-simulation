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

from TimeSeriesSim import timeSeries

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
v0 = 150 #m/s
n = 1000

### Non-dimensional parameters
L = L0/r0
dr = 1.
dt = .02
I = 1.
rho = 1.
dm = pi*dr

#Set font size for plots
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}
plt.rc('font', **font)

########################### HELPER FUNCTIONS ###################################

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

def mag(components):
    x = components[0]
    y = components[1]
    z = components[2]
    magnitude = ((x**2) + (y**2) + (z**2))**(0.5)
    return magnitude

# Calculates the percent of deviation from a selected value
def percent(obs_val, act_val):
    percent = abs(100*(obs_val - act_val)/act_val)
    return percent

# Removes any points/values from a list that are close in value to neighboring* values
def removePoints(array):
    if np.var(array) < 80000:
        fin_array = array
        i = 0
        while not (i == len(fin_array) - 1):
            if percent(array[i], array[i + 1]) < 50/(i + 1.):
                fin_array.remove(array[i + 1])
            else:
                i = i + 1
        return fin_array
    else:
        fin_array = []
        dif_array = []
        for i in range(len(array)):
            if not (i + 1 == len(array)):
                dif = array[i+1] - array[i]
                dif_array.append(dif)
        dif_avg = np.mean(dif_array)
        for i in range(len(dif_array)):
            if dif_array[i] > dif_avg:
                fin_array.append(array[i])
            if not (i == len(dif_array) - 1):
                if dif_array[i] < dif_avg and dif_array[i+1] < dif_avg:
                    fin_array.append(array[i])
            if i == len(dif_array) - 1:
                if dif_array[i-1] < dif_avg and dif_array[i] < dif_avg:
                    fin_array.append(array[i + 1])
        return fin_array

################################################################################
#Generate ideal data of a moving helix of current
fig, [ax1, ax2, ax3] = plt.subplots(1, 3, sharex = True)
B_data10 = timeSeries(45., 3., 1000, 0.5, 0.5, 0.5, cartesian=True, n_period = 10)

################################################################################
#Calculate the average and find near intersections

# Calculate average
avg_x = np.mean(B_data10[0])
avg_y = np.mean(B_data10[1])
avg_z = np.mean(B_data10[2])

# Calculates f - g and the corresponding signs using np.sign. Applying
# np.diff reveals all the positions, where the sign changes (e.g. the lines
# cross). Using np.argwhere gives us the exact indices.
idx = np.argwhere(np.diff(np.sign(B_data10[0] - avg_x))).flatten()
idy = np.argwhere(np.diff(np.sign(B_data10[1] - avg_y))).flatten()
idz = np.argwhere(np.diff(np.sign(B_data10[2] - avg_z))).flatten()

even_idx = []
even_idy = []
even_idz = []
for i in range(len(idx)):
    if not i%2 == 0:
        even_idx.append(idx[i])
for i in range(len(idy)):
    if not i%2 == 0:
        even_idy.append(idy[i])
for i in range(len(idz)):
    if not i%2 == 0:
        even_idz.append(idz[i])
idx = even_idx
idy = even_idy
idz = even_idz

print('Before: ' + str(idx) + ' and variance is ' + str(np.var(idx)))
idx = removePoints(idx)
print('After: ' + str(idx))
print('\n')
print('Before: ' + str(idy) + ' and variance is ' + str(np.var(idy)))
idy = removePoints(idy)
print('After: ' + str(idy))
print('\n')
print('Before: ' + str(idz) + ' and variance is ' + str(np.var(idz)))
idz = removePoints(idz)
print('After: ' + str(idz))

################################################################################

b_x = []
for i in range(len(idx)):
    b_x.append(B_data10[0][i])

b_y = []
for i in range(len(idy)):
    b_y.append(B_data10[1][i])

b_z = []
for i in range(len(idz)):
    b_z.append(B_data10[2][i])

ax1.plot(B_data10[0], 'b-', idx, b_x, 'ro')
ax1.set_ylabel('x Component of the Magnetic Field Vector')
ax1.set_xlabel('Time (Samples)')
ax2.plot(B_data10[1], 'b-', idy, b_y, 'ro')
ax2.set_ylabel('y Component of Magnetic Field Vector')
ax2.set_xlabel('Time (Samples)')
ax3.plot(B_data10[2], 'b-', idz, b_z, 'ro')
ax3.set_ylabel('z Component of Magnetic Field Vector')
ax3.set_xlabel('Time (Samples)')
fig.suptitle('Magnetic Field Vector in Time-Series', fontsize=16)

################################################################################

# Determine the time interval between each intersection
time_intervals_x = []
time_intervals_y = []
time_intervals_z = []

for i in range(len(idx)):
    if i < len(idx) - 1:
        t1 = B_data10[3][idx[i]]
        t2 = B_data10[3][idx[i + 1]]
        t_diff = t2 - t1
        time_intervals_x.append(t_diff)

for i in range(len(idy)):
    if i < len(idy) - 1:
        t1 = B_data10[3][idy[i]]
        t2 = B_data10[3][idy[i + 1]]
        t_diff = t2 - t1
        time_intervals_y.append(t_diff)

for i in range(len(idz)):
    if i < len(idz) - 1:
        t1 = B_data10[3][idz[i]]
        t2 = B_data10[3][idz[i + 1]]
        t_diff = t2 - t1
        time_intervals_z.append(t_diff)

# Determine the average period and thus lambda
avg_period_x = np.mean(time_intervals_x)
avg_period_y = np.mean(time_intervals_y)
avg_period_z = np.mean(time_intervals_z)
lambda_x = avg_period_x*v0
lambda_y = avg_period_y*v0
lambda_z = avg_period_z*v0
print(lambda_x)
print(lambda_y)
print(lambda_z)

#plt.show()
