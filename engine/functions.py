#Magnus Haw Sept.24 2010
# Function Library
import math
import numpy as np
from scipy import optimize
from numpy import sqrt,pi,diff,sign,array,mean,log,exp, linalg,cross,dot,zeros
from numpy import arccos, arange , cos, sin, shape, meshgrid, linspace,reshape
from numpy import mgrid,cumsum
from scipy.interpolate import griddata, splprep,splev,splrep
from numpy.random import uniform 
from numpy import gradient as grad
#from plotting import plot_fit
#from jlinfit import linfit
import matplotlib.pyplot as plt

def local_maxes(x):
    return (diff(sign(diff(x))) < 0).nonzero()[0] + 1

def local_mins(x):
    return (diff(sign(diff(x))) > 0).nonzero()[0] + 1

def excise(x,a):
    for val in a:
        while val in x:
            x.remove(val)
    return x

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

def rms(a):
    b = np.array(a)
    return (np.mean(b**2.))**.5

def rms_diff(a,b):
    a= np.array(a)
    b= np.array(b)
    return rms(a-b)

def r_fft(d,f_s):
    '''
    inputs: data and its sampling frequency
    returns: freq, fft(data)
    '''
    from scipy.fftpack import fft,fftshift, fftfreq
    n = len(d)
    P=fft(d)
    M = fftfreq(n)

    y = fftshift(P)
    w = fftshift(M)

    return f_s*w,abs(y)

def myfft(d,f_s):
    from scipy.fftpack import fft,fftshift, fftfreq
    n = len(d)
    P=fft(d)
    M = fftfreq(n,d=1./f_s)

    y = fftshift(P)
    w = fftshift(M)

    return w,y

def butter_bandpass(lowcut, highcut, fs, order=8):
    from scipy.signal import butter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    from scipy.signal import lfilter
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def exp_fit(x,y, yerr=None):
    if yerr==None:
        myerr = None
    else:
        myerr = log(yerr)
    p,perr,chisq,logfit = linfit(x,log(y),err=myerr)
    yfit = exp(logfit)
    return p,perr,chisq,yfit

def mylin(x,y,yerr=None):
    F = np.zeros((len(x),2))
    F[:,0] = np.ones(len(x))
    F[:,1] = np.array(x)
    F = np.matrix(F)
    y = np.matrix(y).T
    if yerr != None:
        M = np.diag(1./yerr) ##weighting matrix
    else:
        M = np.diag(np.ones(len(x)))
    #print M
    M = np.matrix(M)
    Minv = M.getI()
    
    H = F.T*Minv*F
    Hinv = H.getI()
    theta = Hinv*F.T*Minv*y
    vartheta = np.diag(Hinv)
    return theta.getA1(),(vartheta/len(x))**.5

def LeastSqEst(F,y,M=None):
    y = np.matrix(y)
    if len(input[0]) > 1:
        y = y.T
    if M == None:
        M = np.diag(np.ones(len(x)))
    #print M
    M = np.matrix(M)
    Minv = M.getI()
    
    H = F.T*Minv*F
    Hinv = H.getI()
    theta = Hinv*F.T*Minv*y
    vartheta = np.diag(Hinv)
    return theta.getA1(),(vartheta/len(x))**.5

### Use cubic splines to calculate path derivatives
def get_3D_curve_params(x,y,z):
    tck,s = splprep([x,y,z],s=0)
    ds = s[1]-s[0]
    dx,dy,dz = splev(s,tck,der=1)
    dr = sqrt(dx*dx + dy*dy + dz*dz)
    dl = dr*ds
    L = cumsum(dl)
    T = array([dx/dr,dy/dr,dz/dr]).T

    p,u = splprep([dx/dr,dy/dr,dz/dr],u=L,s=0)
    dTx,dTy,dTz = splev(u,p,der=1)
    kurv = sqrt(dTx*dTx + dTy*dTy + dTz*dTz)
    kurv[kurv==0] = 1e-14
    N = array([dTx/kurv,dTy/kurv,dTz/kurv]).T
    R = 1./kurv

    # return tangent, length, normal vector, radius of curvature
    return T,L,dl,N,R

def get_R(r):
    '''Solve for curvature in 3d (assume x(s), y(s), z(s)).
       endpoints and penultimate points will give
       incorrect values due to boundary effects
    '''
    xp = grad(r[0])
    xpp= grad(xp)

    yp = grad(r[1])
    ypp= grad(yp)

    zp = grad(r[2])
    zpp= grad(zp)

    numer = ( (zpp*yp - ypp*zp)**2. + (xpp*zp - zpp*xp)**2. + (ypp*xp - xpp*yp)**2. )**.5
    denom = (xp**2. + yp**2. + zp**2.)**1.5
    kappa = numer/denom

    return 1./kappa
    
def get_normal(r):
    '''
    Solve for normal unit vector of 3D curve, r=(x,y,z)
    '''
    xp = grad(r[0])
    yp = grad(r[1])
    zp = grad(r[2])
    
    v = (xp**2 + yp**2 + zp**2)**.5
    
    tx = xp/v
    ty = yp/v
    tz = zp/v
    
    nx = grad(tx)
    ny = grad(ty)
    nz = grad(tz)
    norm = (nx**2 + ny**2 + nz**2)**.5
    N = array([nx,ny,nz])/norm

    return N.T

def curl(Bx,By,Bz,dx,dy,dz):
    '''
    Take numerical curl in 3D regular cartesian coords
    '''
    dBx,dBy,dBz = grad(Bx), grad(By), grad(Bz)
    Rx = dBz[1]/dy - dBy[2]/dz
    Ry = dBx[2]/dz - dBz[0]/dx
    Rz = dBy[0]/dx - dBx[1]/dy
    
    return Rx,Ry,Rz

def divergence(Bx,By,Bz,dx,dy,dz):
    dBx,dBy,dBz = grad(Bx), grad(By), grad(Bz)
    ret = dBx[0]/dx + dBy[1]/dy + dBz[2]/dz
    
    return ret


def relax3D(B,f,h,niters=100):
    '''
    Solves Poisson Eq in 3D via relaxation
    (del^2 B = f), does not affect boundary values
    '''
    l,m,n = shape(B)
    for iters in range(0,niters):
        for i in range(1,l-1):
            for j in range(1,m-1):
                for k in range(1,k-1):
                    B[i][j][k] = (sum_neighbors(B,i,j,k) - h*h*f[i][j][k])/6.
    return B


def sum_neighbors(B,i,j,k):
    '''
    Sums neighbors in 3D grid
    '''
    nsum =0
    for a in [i-1,i+1]:
        nsum += B[a][j][k]
    for b in [j-1,j+1]:
        nsum += B[i][b][k]
    for c in [k-1,k+1]:
        nsum += B[i][j][c]
                
    return nsum

def reduce_array(data, n_new):
    bins = n_new
    slices = np.linspace(0, len(data), bins+1, True).astype(np.int)
    counts = np.diff(slices)
    res = np.add.reduceat(data, slices[:-1]) / counts[0]
    return res

def get_disk(pos, norm, radius=1, n=10):
    norm /= 1.0*linalg.norm(norm)
    if linalg.norm(cross(norm,norm+1.)) != 0:
        perp1 = cross(norm,norm+1.)
        #print 'go fish'
    else:
        perp1 = cross(norm,norm+array([1.,0,0]))
        #print 'hi there'
    perp1 /= linalg.norm(perp1)
    perp2 = cross(norm,perp1)
    perp2 /= linalg.norm(perp2)

    positions = zeros((n,3))
    for i in range(0,n):
        theta = uniform(0,2*pi)
        r = radius*(uniform(0,1)**.5)
        x,y = r*cos(theta), r*sin(theta)
        positions[i] = x*perp1 + y*perp2 + pos
    return positions

def get_rect_grid(x,y,z,n):
    '''
    Inputs: x,y,z -> initial positions
            n -> minimum number of points/dimension

    Returns: X,Y,Z -> 3D rectangular arrays covering full space of inital values 
    '''
    ###resample randomly spaced points to regular grid
    dx,dy,dz = x.max()-x.min(),y.max()-y.min(),z.max()-z.min()
    dmin = min(dx,dy,dz)
    
    nx = complex(0, int(n*dx/dmin) )
    ny = complex(0, int(n*dy/dmin) )
    nz = complex(0, int(n*dz/dmin) ) ### imaginary part lets mgrid interpolate between max,min
    X,Y,Z = mgrid[x.min():x.max():nx,y.min():y.max():ny,z.min():z.max():nz]
    dx,dy,dz = X[1,0,0]-X[0,0,0],Y[0,1,0]-Y[0,0,0],Z[0,0,1]-Z[0,0,0]

    return X,Y,Z,dx,dy,dz

def interpolate3D(X,Y,Z,Jx,Jy,Jz,xp,yp,zp,sf=1,fill_value=0.):
    '''
    Inputs: X,Y,Z -> arrays of input positions, arbitrary shape
            Jx,Jy,Jz -> vector values, must have same shape as X,Y,Z
            xp,yp,zp -> positions at which to interpolate
            sf  -> scale factor: use every sf'th point for interpolation

    Returns: 3D interpolated values at X,Y,Z positions
    '''
    Jx_inter = griddata((X.ravel()[::sf],Y.ravel()[::sf],Z.ravel()[::sf]),Jx.ravel()[::sf],(xp,yp,zp),fill_value=0.)
    Jy_inter = griddata((X.ravel()[::sf],Y.ravel()[::sf],Z.ravel()[::sf]),Jy.ravel()[::sf],(xp,yp,zp),fill_value=0.)
    Jz_inter = griddata((X.ravel()[::sf],Y.ravel()[::sf],Z.ravel()[::sf]),Jz.ravel()[::sf],(xp,yp,zp),fill_value=0.)
    
    return Jx_inter,Jy_inter,Jz_inter

def get_bounding_box(x,y,z,mins,deltas,sizes,pad=2):
    '''
    Inputs: x,y,z -> positions inside box, arbitrary shape
            X,Y,Z -> arrays of grid positions
            pad   -> extra cells around minimum bounding box

    Returns: indices of bounding box
    '''
    x,y,z = array(x),array(y),array(z)
    xmin,xmax = x.min(),x.max()
    ymin,ymax = y.min(),y.max()
    zmin,zmax = z.min(),z.max()

    xlow,xhigh = int((xmin - mins[0])/deltas[0]), int((xmax - mins[0])/deltas[0])
    ylow,yhigh = int((ymin - mins[1])/deltas[1]), int((ymax - mins[1])/deltas[1])
    zlow,zhigh = int((zmin - mins[2])/deltas[2]), int((zmax - mins[2])/deltas[2])

    lows = array([xlow,ylow,zlow])-pad
    highs= array([xhigh,yhigh,zhigh])+pad

    flag=0
    for i in range(0,3):
        if lows[i] < 0:
            lows[i]=0
            flag = 1
        if highs[i] >= sizes[i]:
            highs[i] = sizes[i]-1
            flag=1
    return lows,highs,flag
