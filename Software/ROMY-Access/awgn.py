import numpy as np
 
def awgn(x,SNR_dB):
    '''
    adds AWGN noise vector to signal 'x' to generate a
    resulting signal vector of specified SNR in dB
    '''
    L = x.size
    SNR = 10**(SNR_dB/10)                  # SNR to linear scale
    Esym = np.sum(abs(x)**2,axis=None)/(L) # calculate actual symbol energy
    N0 = Esym / SNR                        # find the noise spectral density
    noiseSigma = np.sqrt(N0)               # standard deviation for AWGN Noise
    n = noiseSigma * np.random.randn(x[:,0].size,x[0,:].size) # computed noise
    return x + n                           # received signal