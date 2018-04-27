import numpy as np
import itertools

from polvect6C import *

class structtype():
    pass

def covar6C(data,test_param,wtype='P',W=200,number_source=1,stab=1.e-9,l_auto_perc=1,method="POWER"):
    ''' 
    Six-component MUSIC algorithm after Sollberger et al. (2018)
    
    data: Nx6 Matrix containing 6-C data (N samples) ordered as 
          [v_x,v_y,v_z,omega_x,omega_y,omega_z]. Acceleration and
          rotation rate can be used instead of particle velocity and
          rotation angle

    test_param: STRUCTURE ARRAY CONTAINING PARAMETER SPACE TO BE SEARCHED
                ONLY THE PARAMETERS OF THE SPECIFIED WAVE TYPE ARE REQUIRED
                test_param.vp   : vector containing P-wave velocities (ms)
                test_param.vs   : vector containnig S-wave velocities
                test_param.vR   : vector containing Rayleigh wave velocities
                test_param.vL   : vector containing Love wave velocities
                test_param.theta: vector containing incidence angles (degree)
                test_param.phi  : vector containing azimuth angles (degree)
                test_param.xi   : vector containing ellipticity angles (radians)

    wtype: WAVE TYPE
           'P' : P-wave
           'SV': SV-wave
           'SH': SH-wave
           'L' : Love-wave
           'R' : Rayleigh-wave


    W: WINDOW LENGTH (IN seconds)

    number_source: DETERMINES THE DIMENSION OF THE NULL-SPACE OF THE
                   COHERENCY MATRIX (see Eq.45 in Sollberger et al., 2018)
                   l=4: isolated wave, one dominant eigenvalue
                   l=3: two waves, two dominant eigenvalues
                   l='auto': automatically determine the size of the null space from 
                 the eigenvalue range, determines the number of eigenvalues 
                 that are smaller than l_auto_perc*lambda_max

    stab: OPTIONAL STABILISATION PARAMETER TO AVOID DIVISION BY 0.
          DEFAULT VALUE IS stab=1e-9;

    l_auto_perc: OPTIONAL PARAMETER FOR THE AUTOMATIC DETERMINATION OF THE
                 DIMENSION OF THE NULL-SPACE. THE DIMENSION OF THE NULL
                 SPACE IS ESTIMATED BY DETERMINING THE NUMBER OF EIGENVALUES
                 THAT ARE SMALLER THAN l_auto_perc*lambda_max. DEFAULT VALUE
                 IS l_auto_perc=0.01 (one percent)

    method: 'MUSIC'
            'CAPON'
            'POWER'
    ''' 
    param = structtype()
    l_auto_perc /= 100.
    samp = W/2
    v_scal = test_param.vp[0] + (test_param.vp[-1] - test_param.vp[0])/2.
    if number_source != 'auto':
        l = 5-number_source
    else:
        l = "auto"

    C = np.matrix.getH(data[int(samp-W/2):int(samp+W/2),:]) @ data[int(samp-W/2):int(samp+W/2),:] # compute covariance matrix
    C = C / W # average over window length

    if method == 'CAPON':
        invC = np.linalg.pinv(C, rcond=1e-15)

    if method == 'MUSIC':
        Cprime,Q = np.linalg.eigh(C,UPLO='U')     # eigenvalue decomposition (Q: eigenvectors, Cprime: eigenvalues)
        lambda_  = np.sort(Cprime)[::-1] # sort eigenvalues in descending order
        loc =   np.argsort(Cprime)[::-1]
        Q   =   Q[:,loc]                 # sort eigenvectors
        # determination of the size of the null space
        if l == 'auto':
            I = np.nonzero(lambda_[1:]/lambda_[0] < l_auto_perc)
            I = (list(itertools.chain.from_iterable(I)))[0] + 1
            l = (5 - I) - 1
        Q = Q[:,5-l:5] @ np.matrix.getH(Q[:,5-l:5]) # null space
     

    ## P-wave
    if wtype == 'P':
        L = np.zeros((len(test_param.theta),len(test_param.phi),len(test_param.vp),len(test_param.vs)))
        for it1 in range(0,len(test_param.theta)):
            for it2 in range(0,len(test_param.phi)):
                for it3 in range(0,len(test_param.vp)):
                    for it4 in range(0,len(test_param.vs)):
                        param.theta = test_param.theta[it1]
                        param.phi   = test_param.phi[it2]
                        param.vp    = test_param.vp[it3]
                        param.vs    = test_param.vs[it4]
                        v = polvect6C(param,v_scal,'P')                           # calculate test polarization vector
                        v = v / np.linalg.norm(v)                                 # convert to unit vector
                        if method == 'MUSIC':
                            L[it1,it2,it3,it4] = 1/(np.abs(np.matrix.getH(v) @ Q @ v) + stab) # MUSIC estimator
                        #    L[it1,it2,it3,it4] = 1/(np.abs(np.matrix.getH(v) @ Q + stab))**2 # MUSIC estimator after Kirlin
                        if method == 'CAPON':
                            L[it1,it2,it3,it4] = 1/(np.abs(np.matrix.getH(v) @ invC @ v) + stab) # Capon estimator
                        if method == 'POWER':
                            L[it1,it2,it3,it4] = np.abs(np.matrix.getH(v) @ C @ v)  # classical Power Spectrum
        return L
    
    ## SV-wave
    if wtype == 'SV':
        L = np.zeros((len(test_param.theta),len(test_param.phi),len(test_param.vp),len(test_param.vs)))
        for it1 in range(0,len(test_param.theta)):
            for it2 in range(0,len(test_param.phi)):
                for it3 in range(0,len(test_param.vp)):
                    for it4 in range(0,len(test_param.vs)):
                        param.theta = test_param.theta[it1]
                        param.phi   = test_param.phi[it2]
                        param.vp    = test_param.vp[it3]
                        param.vs    = test_param.vs[it4]
                        v = polvect6C(param,v_scal,'SV')                          # calculate test polarization vector
                        v = v / np.linalg.norm(v)                                 # convert to unit vector
                        if method == 'MUSIC':
                            L[it1,it2,it3,it4] = 1/(np.abs(np.matrix.getH(v) @ Q @ v) + stab) # MUSIC estimator
                            #L[it1,it2,it3,it4] = 1/(np.abs(np.matrix.getH(v) @ Q + stab))**2 # MUSIC estimator after Kirlin
                        if method == 'CAPON':
                            L[it1,it2,it3,it4] = 1/(np.abs(np.matrix.getH(v) @ invC @ v) + stab) # Capon estimator
                        if method == 'POWER':
                            L[it1,it2,it3,it4] = np.abs(np.matrix.getH(v) @ C @ v)  # classical Power Spectrum
        return L
                                             
    ## SH-wave
    if wtype == 'SH':
        L = np.zeros((len(test_param.theta),len(test_param.phi),len(test_param.vs)))
        for it1 in range(0,len(test_param.theta)):
            for it2 in range(0,len(test_param.phi)):
                for it3 in range(0,len(test_param.vs)):
                    param.theta = test_param.theta[it1]
                    param.phi   = test_param.phi[it2]
                    param.vs    = test_param.vs[it3]
                    v = polvect6C(param,v_scal,'SH')                      # calculate test polarization vector
                    v = v / np.linalg.norm(v)                             # convert to unit vector
                    if method == 'MUSIC':
                        L[it1,it2,it3] = 1/(np.abs(np.matrix.getH(v) @ Q @ v) + stab) # MUSIC estimator
                        #L[it1,it2,it3] = 1/(np.abs(np.matrix.getH(v) @ Q + stab))**2 # MUSIC estimator after Kirlin
                    if method == 'CAPON':
                        L[it1,it2,it3] = 1/(np.abs(np.matrix.getH(v) @ invC @ v) + stab) # Capon estimator
                    if method == 'POWER':
                        L[it1,it2,it3] = np.abs(np.matrix.getH(v) @ C @ v)  # classical Power Spectrum
        return L
    
    ## Rayleigh-wave
    if wtype == 'R':
        L = np.zeros((len(test_param.phi),len(test_param.xi),len(test_param.vs)))
        for it1 in range(0,len(test_param.phi)):
            for it2 in range(0,len(test_param.xi)):
                for it3 in range(0,len(test_param.vs)):
                    param.phi  = test_param.phi[it1]
                    param.xi = test_param.xi[it2]
                    param.vs  = test_param.vs[it3]
                    v = polvect6C(param,v_scal,'R')                       # calculate test polarization vector
                    v = v / np.linalg.norm(v)                             # convert to unit vector
                    if method == 'MUSIC':
                        L[it1,it2,it3] = 1/(np.abs(np.matrix.getH(v) @ Q @ v) + stab) # MUSIC estimator
                    if method == 'CAPON':
                        L[it1,it2,it3] = 1/(np.abs(np.matrix.getH(v) @ invC @ v) + stab) # Capon estimator
                    if method == 'POWER':
                        L[it1,it2,it3] = np.abs(np.matrix.getH(v) @ C @ v)  # classical Power Spectrum
        return L
    
    ## Love-wave
    if wtype == 'L':
        L = np.zeros((len(test_param.phi),len(test_param.vs)))
        for it1 in range(0,len(test_param.phi)):
            for it2 in range(0,len(test_param.vs)):
                param.phi = test_param.phi[it1]
                param.vs  = test_param.vs[it2]
                v = polvect6C(param,v_scal,'L')                   # calculate test polarization vector
                v = v / np.linalg.norm(v)                         # convert to unit vector
                if method == 'MUSIC':
                    L[it1,it2] = 1/(np.abs(np.matrix.getH(v) @ Q @ v) + stab) # MUSIC estimator
                    #L[it1,it2] = 1/(np.abs(np.matrix.getH(v) @ Q + stab))**2 # MUSIC estimator after Kirlin
                if method == 'CAPON':
                    L[it1,it2] = 1/(np.abs(np.matrix.getH(v) @ invC @ v) + stab) # Capon estimator
                if method == 'POWER':
                    L[it1,it2] = np.abs(np.matrix.getH(v) @ C @ v)  # classical Power Spectrum
        return L
