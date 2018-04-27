import numpy as np

def polvect6C(param, v_scal, wtype):
    ''' 
    Computes the six-component polarization vector for a specified wave-type
    at the free surface according to Equations 40 in Sollberger et al. (2018)
    
    param: STRUCTURE ARRAY CONTAINING WAVE PARAMETERS
           ONLY THE PARAMETERS OF THE SPECIFIED WAVE TYPE ARE REQUIRED
           param.vp   : P-wave velocity (m/s) at the receiver location
           param.vs   : S-wave velocity (m/s) at the receiver location
           param.vL   : Love-wave velocity (m/s)
           param.vR   : Rayleigh-wave velocity (m/s)
           param.theta: Inclination (degree), only for body waves
           param.phi  : Azimuth (degree)
           param.xi   : Ellipticity (rad) for Rayleigh waves
    
    v_scal: SCALING VELOCITY (m/s)
    
    w_type: WAVE TYPE
            'P' : P-wave
            'SV': SV-wave
            'SH': SH-wave
            'L' : Love-wave
            'R' : Rayleigh-wave
    '''


    
    ## P-wave
    if wtype == 'P':
        theta = param.theta
        phi   = param.phi
        vp    = param.vp
        vs    = param.vs

        deg = 1./180.*np.pi # convert degree to radian
        if vs == vp:
            v = 0.2
        else:
            v     = (vp**2 - 2*vs**2) / (2*(vp**2 - vs**2)) # poisson's ratio
        kappa = (2*(1 - v) / (1 - 2*v))**(1/2.)
        theta_S = np.arcsin((1/kappa)*np.sin(theta*deg)) / deg # angle of reflected S-wave
    
        alpha_PP =         (np.sin(2*theta*deg)*np.sin(2*theta_S*deg)-kappa**2*(np.cos(2*theta_S*deg))**2) \
                          /(np.sin(2*theta*deg)*np.sin(2*theta_S*deg)+kappa**2*(np.cos(2*theta_S*deg))**2) # amplitude of reflected P-wave
        alpha_PS = (2*kappa*np.sin(2*theta*deg)*np.cos(2*theta_S*deg)) \
                          /(np.sin(2*theta*deg)*np.cos(2*theta_S*deg)+kappa**2*(np.cos(2*theta_S*deg))**2) # amplitude of reflected S-wave
    
        v_x =         -(np.sin(theta  *deg)*np.cos(phi*deg) \
              +alpha_PP*np.sin(theta  *deg)*np.cos(phi*deg) \
              +alpha_PS*np.cos(theta_S*deg)*np.cos(phi*deg)) / v_scal
        v_y =         -(np.sin(theta  *deg)*np.sin(phi*deg) \
              +alpha_PP*np.sin(theta  *deg)*np.sin(phi*deg) \
              +alpha_PS*np.cos(theta_S*deg)*np.sin(phi*deg)) / v_scal
        v_z =         -(np.cos(theta  *deg) \
              -alpha_PP*np.cos(theta  *deg) \
              +alpha_PS*np.sin(theta_S*deg)) / v_scal
    
        w_x =  (1/2.)*alpha_PS*np.sin(phi*deg) / vs
        w_y = -(1/2.)*alpha_PS*np.cos(phi*deg) / vs
        w_z =  0. * w_x
    
        return np.asarray([v_x,v_y,v_z,w_x,w_y,w_z])
    
    
    ## SV-wave
    elif wtype == 'SV':
        theta = param.theta
        phi   = param.phi
        vp    = param.vp
        vs    = param.vs
        
        deg = 1/180.*np.pi # convert degree to radian
    
        if vs == vp:
            v = 0.2
        else:
            v     = (vp**2 - 2*vs**2) / (2*(vp**2 - vs**2)) # poisson's ratio
        kappa = (2*(1 - v) / (1 - 2*v))**(1/2.)
        theta_P = np.arcsin(np.sin(theta*deg)*vp/vs) / deg
        
        alpha_SS =   (np.sin(2*theta*deg)*np.sin(2*theta_P*deg)-kappa**2*(np.cos(2*theta_P*deg))**2) \
                    /(np.sin(2*theta*deg)*np.sin(2*theta_P*deg)+kappa**2*(np.cos(2*theta  *deg))**2)
        alpha_SP =  -(kappa*np.sin(4*theta*deg)) \
                    /(np.sin(2*theta*deg)*np.sin(2*theta_P*deg)+kappa**2*(np.cos(2*theta  *deg))**2)
        
        v_x =        (np.cos(theta  *deg)*np.cos(phi*deg) \
            -alpha_SS*np.cos(theta  *deg)*np.cos(phi*deg) \
            -alpha_SP*np.sin(theta_P*deg)*np.cos(phi*deg)) / v_scal
        v_y =        (np.cos(theta  *deg)*np.sin(phi*deg) \
            -alpha_SS*np.cos(theta  *deg)*np.sin(phi*deg) \
            -alpha_SP*np.sin(theta_P*deg)*np.sin(phi*deg)) / v_scal
        v_z=        -(np.sin(theta  *deg) \
            +alpha_SS*np.sin(theta  *deg) \
            -alpha_SP*np.cos(theta_P*deg)) / v_scal
        
        w_x =  (1/2.) * (1+alpha_SS) * np.sin(phi*deg) / vs
        w_y = -(1/2.) * (1+alpha_SS) * np.cos(phi*deg) / vs
        w_z =  0. * w_x
    
        return np.asarray([v_x,v_y,v_z,w_x,w_y,w_z])
    
    ## SH-wave
    elif wtype == 'SH':
        phi   = param.phi
        theta = param.theta
        vs = param.vs
        
        deg = 1/180.*np.pi # convert degree to radian
        
        v_x =  2 / v_scal * np.sin(phi*deg)
        v_y = -2 / v_scal * np.cos(phi*deg)
        v_z =  0. * v_x
        
        w_z =  1 / vs * np.sin(theta*deg)
        w_x =  0. * w_z
        w_y =  0. * w_z
        
        return np.asarray([v_x,v_y,v_z,w_x,w_y,w_z])
    
    ## Rayleigh-wave
    elif wtype == 'R':
        phi = param.phi
        vR  = param.vs
        xi  = param.xi
        
        deg = 1/180.*np.pi # convert degree to radian
        
        v_x =  1j * 1 / v_scal * np.sin(xi) * np.cos(phi*deg)
        v_y =  1j * 1 / v_scal * np.sin(xi) * np.sin(phi*deg)
        v_z =      -1 / v_scal * np.cos(xi)
        
        w_x =  1 / vR * np.sin(phi*deg) * np.cos(xi)
        w_y = -1 / vR * np.cos(phi*deg) * np.cos(xi)
        w_z =  0. * w_x
    
        return np.asarray([v_x,v_y,v_z,w_x,w_y,w_z])
    
    ## Love-wave
    elif wtype == 'L':
        phi = param.phi
        vL  = param.vs
        
        deg = 1/180.*np.pi # convert degree to radian
        
        v_x =  1 / v_scal * np.sin(phi*deg)
        v_y = -1 / v_scal * np.cos(phi*deg)
        v_z =  0. * v_x
        
        w_z =  1 / (2 * vL)
        w_x =  0. * w_z
        w_y =  0. * w_z
        
        return np.asarray([v_x,v_y,v_z,w_x,w_y,w_z])
