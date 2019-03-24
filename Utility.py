### Utility functions
from numpy import array,zeros,gradient as grad,linalg,cross
from numpy import newaxis,shape,sqrt
from scipy.interpolate import splprep,splev,splrep
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

    B = sum(np.cross(r,dl) / (rmag**3.)[:,newaxis])
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
    return I*np.cross(dl,B)

def tension_force(path,B):
    '''
    Given a path, shape(n,3), and a B-field magnitude,
    calculates magnetic tension force from path curvature
    '''
    ## Radius of curvature
    R = get_R(path.T)
    rhat= get_normal(path.T)
    
    ## tension force
    ft = B*B*(1/R)*rhat.T/mu0
    return ft.T

def tension_force1(R,N,L,L0,Phi,a,dl):
    ## tension force
    ft = dl*(Phi*Phi/(pi*a*a*2*mu0*R))*((L*L-L0*L0)/L/L)*N.T
    return ft.T

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
    kurv = np.sqrt(dTx*dTx + dTy*dTy + dTz*dTz)
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
