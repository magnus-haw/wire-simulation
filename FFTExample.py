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
# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api

S = 1000; # number of samples
n_period = 10; # number of oscillations
x_inc = 2.*pi*n_period/S; # increment distance between each point
v0 = 150; #m/s
t_inc = x_inc/v0; # interval
t = []
count = 0.
for i in range(0, S):
    t.append(count)
    count = count + t_inc

y = timeSeries(45., 3., 1000, 0.5, 0.5, 0.5, cartesian=True, n_period = 10)[0]

n = len(y) # length of the signal
k = np.arange(n)
T = n/count
frq = k/T # two sides frequency range
frq = frq[range(n//2)] # one side frequency range

Y = np.fft.fft(y)/n # fft computing and normalization
Y = Y[range(n//2)]
i_max = abs(Y).tolist().index(max(abs(Y).tolist()))
max_freq = frq[i_max]
print('The frequency of the signal is ' + str(max_freq))

Period = 1/max_freq
lmbda = v0*Period
print('The lambda of the signal is ' + str(lmbda))

fig, ax = plt.subplots(2, 1)
ax[0].plot(t,y)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[1].plot(frq,abs(Y),'r') # plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Y(freq)|')

plt.show()
