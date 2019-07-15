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
def removePoints(array, bool):
    if bool:
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

l_rng = range(1, 21)
r_rng = l_rng

var_per = []
var_avg = []

for i in l_rng:
    const_l = 5.*i
    B_data10 = timeSeries(const_l, 4., 1000, 0.5, 0.5, 0.5,
    cartesian = True, n_period = 10)

    avg_x = np.mean(B_data10[0])
    avg_y = np.mean(B_data10[1])
    avg_z = np.mean(B_data10[2])

    idx = np.argwhere(np.diff(np.sign(B_data10[0] - avg_x))).flatten().tolist()
    idy = np.argwhere(np.diff(np.sign(B_data10[1] - avg_y))).flatten().tolist()
    idz = np.argwhere(np.diff(np.sign(B_data10[2] - avg_z))).flatten().tolist()

    idx_per = removePoints(idx, True)
    idx_avg = removePoints(idx, False)
    idy_per = removePoints(idy, True)
    idy_avg = removePoints(idy, False)
    idz_per = removePoints(idz, True)
    idz_avg = removePoints(idz, False)

    # Determine the time interval between each intersection
    time_intervals_x_per = []
    time_intervals_x_avg = []
    time_intervals_y_per = []
    time_intervals_y_avg = []
    time_intervals_z_per = []
    time_intervals_z_avg = []

    for i in range(len(idx_per)):
        if i < len(idx_per) - 1:
            t1 = B_data10[3][idx_per[i]]
            t2 = B_data10[3][idx_per[i + 1]]
            t_diff = t2 - t1
            time_intervals_x_per.append(t_diff)

    for i in range(len(idx_avg)):
        if i < len(idx_avg) - 1:
            t1 = B_data10[3][idx_avg[i]]
            t2 = B_data10[3][idx_avg[i + 1]]
            t_diff = t2 - t1
            time_intervals_x_avg.append(t_diff)

    for i in range(len(idy_per)):
        if i < len(idy_per) - 1:
            t1 = B_data10[3][idy_per[i]]
            t2 = B_data10[3][idy_per[i + 1]]
            t_diff = t2 - t1
            time_intervals_y_per.append(t_diff)

    for i in range(len(idy_avg)):
        if i < len(idy_avg) - 1:
            t1 = B_data10[3][idy_avg[i]]
            t2 = B_data10[3][idy_avg[i + 1]]
            t_diff = t2 - t1
            time_intervals_y_avg.append(t_diff)

    for i in range(len(idz_per)):
        if i < len(idz_per) - 1:
            t1 = B_data10[3][idz_per[i]]
            t2 = B_data10[3][idz_per[i + 1]]
            t_diff = t2 - t1
            time_intervals_z_per.append(t_diff)

    for i in range(len(idz_avg)):
        if i < len(idz_avg) - 1:
            t1 = B_data10[3][idz_avg[i]]
            t2 = B_data10[3][idz_avg[i + 1]]
            t_diff = t2 - t1
            time_intervals_z_avg.append(t_diff)

    # Determine the average period and thus lambda
    avg_period_x_per = np.mean(time_intervals_x_per)
    avg_period_x_avg = np.mean(time_intervals_x_avg)
    avg_period_y_per = np.mean(time_intervals_y_per)
    avg_period_y_avg = np.mean(time_intervals_y_avg)
    avg_period_z_per = np.mean(time_intervals_z_per)
    avg_period_z_avg = np.mean(time_intervals_z_avg)
    lambda_x_per = avg_period_x_per*v0
    lambda_x_avg = avg_period_x_avg*v0
    lambda_y_per = avg_period_y_per*v0
    lambda_y_avg = avg_period_y_avg*v0
    lambda_z_per = avg_period_z_per*v0
    lambda_z_avg = avg_period_z_avg*v0

    l_x_per = percent(lambda_x_per, const_l)
    l_x_avg = percent(lambda_x_avg, const_l)
    l_y_per = percent(lambda_y_per, const_l)
    l_y_avg = percent(lambda_y_avg, const_l)
    l_z_per = percent(lambda_z_per, const_l)
    l_z_avg = percent(lambda_z_avg, const_l)

    if l_x_per < l_x_avg:
        var_per.append(np.var(idx))
    else:
        var_avg.append(np.var(idx))
    if l_y_per < l_y_avg:
        var_per.append(np.var(idy))
    else:
        var_avg.append(np.var(idy))
    if l_z_per < l_z_avg:
        var_per.append(np.var(idz))
    else:
        var_avg.append(np.var(idz))

print(np.mean(var_per))
print(np.mean(var_avg))
