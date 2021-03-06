### Utility functions
from numpy import array,zeros,gradient as grad,linalg,cross
from numpy import newaxis,shape,sqrt,pi
from scipy.interpolate import splprep,splev,splrep,griddata
import numpy as np
mu0=4e-7*pi

def biot_savart(p, I, path, delta=.01):
    '''Given a point, a current and its path, calculates the magnetic field at that point
       This function uses normalized units:
        e.g. positions are normalized by a radial length scale r~(r'/L0)
             current is normalized to a characteristic value I~(I'/I0)
             Bfield is normalized to B~(B'/B0)
    '''
    dl = grad(path, axis=0) #centered differences
    r = path-p
    rmag = linalg.norm(r, axis=1)
    rmag[rmag<= delta] = 1e6

    B = sum(np.cross(r,dl) / (rmag**3.)[:,newaxis])
    B *= I/2.
    return B

##def getBField(path, coil_paths, currents, delta=.01):
##    '''
##    Given the path of the loop, a list of coil paths (current paths), and
##    a list of current magnitudes, returns B field at points along path
##    '''
##    n = len(path)
##    B = zeros((n, 3))
##    for p in range(n):
##        for c in range(len(coil_paths)):
##            B[p,:] += biot_savart(path[p], currents[c], coil_paths[c],delta=delta)
##    return B

def getBField(path, wires):
    '''
    Given a path & a list of wires, returns B field at points along path
    '''
    n = len(path)
    B = zeros((n, 3))
    for p in range(n):
        for wire in wires:
            B[p,:] += biot_savart(path[p], wire.I, wire.p,delta=wire.r)
    return B

def JxB_force(path,I,B):
    '''
    Given a wire path, a current I and magnetic field B,
    calculates the JxB force at each point
    '''
    dl = grad(path, axis=0) #centered differences
    return I*np.cross(dl,B)

def inductance(path1,path2,rwire=0.001,norm=None):
    '''
    Given two wire paths, calculates the mutual inductance in SI units
    taken from: Advanced Electromagnetics. 2016;5(1)
    DOI 10.7716/aem.v5i1.331 
    '''
    dl1 = grad(path1, axis=0) #centered differences
    dl2 = grad(path2, axis=0) #centered differences
    L = 0
    for i in range(0,len(path1)):        
        for j in range(0,len(path2)):
            dd = sqrt( ((path1[i] - path2[j])**2).sum() )
            if dd > rwire/2.:
                L+=np.dot(dl1[i],dl2[j])/dd
    if norm is None:
        return mu0*L/(4*pi)
    else:
        return mu0*L/(4*pi)/norm
      
def tension_force(wire):
    '''
    Calculates tension force from 3D curve properties
    '''
    T,CumLen,dl,N,R,tck,s = wire.get_3D_curve_params()
    vol = np.pi*wire.r*wire.r*dl
    Lsq = CumLen[-1]**2
    ft = vol*(wire.Bp*wire.Bp/R)*((Lsq - wire.L_init**2)/Lsq)*N.T
    return ft.T

def tension_force1(R,N,L,L0,Phi,a,dl):
    ## tension force
    ft = dl*(Phi*Phi/(pi*a*a*2*mu0*R))*((L*L-L0*L0)/L/L)*N.T
    return ft.T

def tension_force0(path,B):
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

def get_rect_grid(x,y,z,n):
    '''
    Inputs: x,y,z -> initial positions
            n -> minimum number of points/dimension

    Returns: X,Y,Z -> 3D rectangular arrays covering full space of inital values 
    '''
    x,y,z = np.array(x),np.array(y),np.array(z)
    ###resample input point volume to regular grid
    dx,dy,dz = x.max()-x.min(),y.max()-y.min(),z.max()-z.min()
    dmin = min(dx,dy,dz)
    
    nx = complex(0, int(n*dx/dmin) )
    ny = complex(0, int(n*dy/dmin) )
    nz = complex(0, int(n*dz/dmin) ) ### imaginary part lets mgrid interpolate between max,min
    X,Y,Z = np.mgrid[x.min():x.max():nx,y.min():y.max():ny,z.min():z.max():nz]
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

def smooth(x,window_len=11,window='hanning'):
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
    #print(len(x),len(y),window_len)
    ret = y[int((window_len-1)/2):-int(window_len/2)]
    #print(len(x),len(ret))
    return ret

def smooth3DVectors(vc,n=5):
    #print(np.shape(vc))
    x,y,z = vc[:,0],vc[:,1],vc[:,2]
    xs = smooth(x,window_len=n)
    ys = smooth(y,window_len=n)
    zs = smooth(z,window_len=n)
    vcs = vc.copy()
    vcs[:,0],vcs[:,1],vcs[:,2] = xs,ys,zs
    return vcs
