import numpy as np
from numpy.linalg import norm
from ..utils.constants import mu0, pi


def inductance(path1,path2,rwire=0.001,norm=None):
    '''
    Given two wire paths, calculates the mutual inductance in SI units
    taken from: Advanced Electromagnetics. 2016;5(1)
    DOI 10.7716/aem.v5i1.331 
    '''
    dl1 = np.gradient(path1, axis=0) #centered differences
    dl2 = np.gradient(path2, axis=0) #centered differences
    L = 0
    for i in range(0,len(path1)):        
        for j in range(0,len(path2)):
            dd = np.sqrt( ((path1[i] - path2[j])**2).sum() )
            if dd > rwire/2.:
                L+=np.dot(dl1[i],dl2[j])/dd
    if norm is None:
        return mu0*L/(4*pi)
    else:
        return mu0*L/(4*pi)/norm