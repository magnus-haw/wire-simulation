### Utility functions
from numpy import array,zeros,gradient as grad,linalg,cross
from numpy import newaxis,shape
import numpy as np

def biot_savart(p, I, path, delta=.01):
    '''Given a point, a current and its path, calculates the magnetic field at that point
       This function uses normalized units:
        e.g. positions are normalized by a radial length scale r~(r'/L0)
             current is normalized to a characteristic value I~(I'/I0)
             Bfield is normalized to B~(B'/B0)
        These non-dimensional scalings are defined in Dimensions.py
    '''
    dl = grad(path, axis=0)
    r = path-p
    rmag = linalg.norm(r, axis=1)
    rmag[rmag<= delta] = 1e6

    B = sum(cross(r,dl) / (rmag**3.)[:,newaxis])
    B *= I/2.
    return B

def getBField(path, coil_paths, currents, delta=.01):
    '''
    Given the path of the loop, a list of coil paths (current paths), and
    a list of current magnitudes, returns B field at points along path
    '''
    n = len(path)
    B = zeros((n, 3))
    for p in range(n):
        for c in range(len(coil_paths)):
            B[p,:] += biot_savart(path[p], currents[c], coil_paths[c],delta=delta)
    return B


def JxB_force(path,I,B):
    '''
    Given a wire path, a current I and magnetic field B,
    calculates the JxB force at each point
    '''
    dl = grad(path, axis=0)
    return I*cross(dl,B)

def smooth(x,window_len=10,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print len(s), len(x)
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    ret = y[(int(window_len/2)-1):-int(window_len/2)]
    #print(len(x),len(ret))
    return ret
