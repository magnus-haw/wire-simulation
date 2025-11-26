import numpy as np
from numpy import gradient as grad
from scipy.interpolate import griddata

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
