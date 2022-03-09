from scipy.io import wavfile
import numpy as np

def xcorr_norm(xp, lmin, lmax, Nblock):
    '''
    Computes the normalized autocorrelation

    Inputs:
        xp          : input block
        lmin        : min. of tested lag range
        lmax        : max. of tested lag range
        Nblock      : block size
    Outputs:
        rxx_norm    : normalized autocorr. sequence
        rxx         : autocorr. sequence
        rxx0        : energy of delayed blocks
    '''

    # --- initializations ---
    x = xp[lmax:Nblock + lmax] # input block without pre-samples
    #print(x.shape)
    lags = np.arange(lmin, lmax, 1) # tested lag range
    Nlag = len(lags) #no. of lags

    # --- empty output variables ---
    rxx = np.zeros(Nlag)
    rxx0 = np.zeros(Nlag)
    rxx_norm = np.zeros(Nlag)

    # --- computes autocorrelation(s) ---
    for l in range(Nlag):
        ii = lags[l] # tested lag
        start = lmax-lags[l]
        end = start + Nblock
        rxx0[l] = np.sum(xp[start:end]**2)
        # --- energy of delayed block ---
        rxx[l] = np.sum(x*xp[start:end])
    
    ii = np.argwhere(rxx0 != 0)
    rxx_norm[ii] = (rxx[ii]**2)/rxx0[ii]

    return rxx_norm, rxx, rxx0

def find_loc_max(x, nargout):
    N = len(x)
    dx = np.diff(x)
    #print(dx.shape, N)

    dx1 = dx[1:N-1]
    dx2 = dx[0:N-2]

    prod = dx1*dx2

    idx1 = np.argwhere(prod < 0) # sign change in dx1
    #print(idx1.shape)
    idx2 = np.argwhere(dx1[idx1] < 0) # only change from + to -
    idx2 = idx2[:,0]
    #print(idx2.shape)
    idx = idx1[idx2] # positions of single maxima
    #print(idx.shape)
    
    idx3 = np.argwhere(dx==0)
    idx4 = np.argwhere(x[idx3]>0) # only maxima
    idx4 = idx4[:,0]
    idx0 = idx3[idx4]
    idx0 = idx0[:,0]
    #print(idx0)
    #----- positions of double maxima, same values at idx3(idx4)+1 -----
    if nargout == 1: # output 1 vector
    # with positions of all maxima
        idx = np.sort(np.append(idx, idx0))  # (for double max. only 1st position)
        #print(idx.shape)
        return idx

    #print(idx.shape)

    return idx, idx0

def find_pitch_ltp(xp, lmin, lmax, Nblock, Fs, b0_thresh):
    '''
    Computes the pitch lag candidates from a signal block

    Inputs:
        xp          : input block including lmax pre-samples 
                      for correct autocorreation
        lmin        : min. checked pitch lag
        lmax        : max. checked pitch lag
        Nblock      : block length without pre-samples
        Fs          : sampling freq.
        b0_thresh   : max. b0 deviation from 1
    Outputs:
        M           : pitch lags
        Fp          : pitch frequencies
    '''

    lags = np.arange(lmin, lmax, 1) # tested lag range
    Nlag = len(lags) # no. of lags
    rxx_norm, rxx, rxx0 = xcorr_norm(xp, lmin, lmax, Nblock)

    # --- calc. autocorr. sequences ---
    ii = np.argwhere(rxx0 != 0)
    B0 = np.zeros(rxx0.shape)
    B0[ii] = rxx[ii]/rxx0[ii] # LTP coeffs for all lags
    idx = find_loc_max(rxx_norm, 1)
    #print(idx.shape)
    i = np.argwhere(rxx[idx] > 0) # only max. where rxx > 0
    i = i[:,0]
    #print(i.shape)
    idx = idx[i]
    #print(B0[idx].shape)
    i = np.argwhere(np.abs(B0[idx] - 1) < b0_thresh)
    i = i[:,0]
    #print(np.abs(B0[idx] - 1))

    # --- only max. where LTP coeff. is close to 1 ---
    idx = idx[i]

    # --- vectors for all pitch candidates: ---
    M = lags[idx]
    M = M[:] # pitch lags
    Fp = Fs/M
    Fp = Fp[:] # pitch frequs.

    return M, Fp

def segmentation(voiced, M, pitches):
    '''
    Performs pitch segmentation

    Inputs:
        voiced      : original voiced/unvoiced detection
        M           : minimum number of consecutive blocks with same voiced flag
        pitches     : original pitches
    Outputs:
        V           : changed voiced flag
        pitches2    : changed pitches
    '''

    blocks = len(voiced)
    pitches2 = pitches
    V = voiced
    Nv = len(V)

    # ---step1: eliminate too short voiced segments:
    V[Nv - 1] = not V[Nv - 2]# change at the end to get length of last segment
    dv = np.append(np.array([0]), np.diff(V)) # derivate
    idx = np.argwhere(dv != 0) # changes in voiced
    di = np.append(idx[0]-1, np.diff(idx)) # segment lengths
    v0 = V[0] # status of 1st segment
    k0 = 1
    ii = 0 # counter for segments, idx[ii]-1 is end of segment
    if v0 == 0:
        k0 = idx[0] # start of voiced
        ii += 1 # first change voiced to unvoiced

    while ii < len(idx):
        L = di[ii]
        k1 = idx[ii] - 1 # end of voiced segment
        if L < M:
            V[k0:k1] = np.zeros(k1 - k0 + 1)
        if ii < len(idx) - 1:
            k0 = idx[ii+1] # star of next voiced segment
        ii += 2

    # ---step2: eliminate too short unvoiced segments:
    V[Nv - 1] = not V[Nv - 2]# one more change at the end to get length of last segment
    dv = np.append(np.array([0]), np.diff(V)) # derivate
    idx = np.argwhere(dv != 0) # changes in voiced
    di = np.append(idx[0] - 1, np.diff(idx)) # segment lengths
    if len(idx) > 1: # changes in V
        v0 = V[0]
        k0 = 1
        ii = 0 # counter for segments, idx[ii]-1 is end of segment
        if v0 == 0:
            k0 = idx[1] # start of voiced
            ii += 2 # first change voiced to unvoiced

        while ii < len(idx):
            L = di[ii]
            k1 = idx[ii] - 1 # end of unvoiced segment
            if L < M:
                if k1 < blocks: # NOT last unvoiced segment
                    V[k0:k1] = np.ones(k1 - k0 + 1)
                    # linear pitch interpolation:
                    p0 = pitches[k0 - 1]
                    p1 = pitches[k1 + 1]
                    N = k1 - k0 + 1
                    pitches2[k0:k1] = np.arange(1, N, 1)*(p1-p0)/(N+1) + p0
            if ii < len(idx) - 1:
                k0 = idx[ii+1] # start of next unvoiced segment
            ii += 2
    
    V = V[0:Nv] # cut last element

    return V, pitches2

def Pitch_Tracker_LTP(fname, n0, n1, K, N, b0_thresh, fmin, fmax):
    '''
    Performs block-wise pitch detection for input signal in wav file fname
    and determines a pitch estimation for the signal slice

    Inputs:
        fname       : wav file containing audio signal
        n0, n1      : first and final sample indexes, determines 
                      signal slice to be analysed
        K           : hop size for time resolution of pitch estimation
        N           : block length
        b0_thresh   : threshold for LTP coeff
        fmin, fmax  : determines pitch range in Hz
    Output:
        p           : pitch estimation for signal slice
    '''

    p_fac_thresh = 1.05 # threshold for voiced detection
                        # deviation of pitch from mean value

    Fs, xin = wavfile.read(fname) # get Fs
    # lag range in samples
    lmin = np.floor(Fs/fmax).astype(int)
    lmax = np.ceil(Fs/fmin).astype(int)
    pre = lmax # number of pre-samples
    if n0 - pre < 1:
        n0 = pre + 1
    Nx = n1 - n0 + 1 # signal length
    blocks = np.floor(Nx/K).astype(int)
    Nx = int((blocks - 1)*K + N)
    X = xin[n0-pre:n0+Nx]

    pitches = np.zeros(blocks)
    for b in range(blocks):
        start = b*K
        end = start + N + pre
        x = X[start:end]
        M, F0 = find_pitch_ltp(x, lmin, lmax, N, Fs, b0_thresh)
        #print(M.shape)
        if M.size > 0:
            #print('here')
            pitches[b] = Fs/M[0] # take candidates with lowest pitch
        else:
            pitches[b] = 0

    #print(pitches)

    # --- post-processing (may change) ---
    L = 9 # moving-average filter size
    if L % 2 == 0:  # L has to be odd
        L += 1
    D = (L - 1)//2 # delay
    h = np.ones(L)/L # moving-average filter impulse response
    # mirror start and end for non-causal filtering (that is, treat the boarders)
    p = np.append(pitches[D+1:1:-1], pitches)
    p = np.append(p, pitches[blocks - 1: blocks - D - 1:-1])
    y = np.convolve(p, h) # length: blocks + 2*D + 2*D
    pm = y[2*D:blocks + 2*D] # cut result

    Fac = np.zeros(blocks)
    idx = np.argwhere(pm != 0) # avoid dividing by zero
    Fac[idx] = pitches[idx]/pm[idx]
    ii = np.array([], dtype = int)
    for i in range(Fac.size):
        if 0 < Fac[i] < 1:
            ii = np.append(ii, i)
    #ii = np.argwhere(Fac < 1)
    #ii = np.argwhere(Fac[ii] != 0)
    Fac[ii] = 1/Fac[ii] # all non-zero elements are now > 1
    # voiced/unvoiced detection:
    voiced = Fac != 0
    voiced = voiced < p_fac_thresh

    T = 40 # time for segment lengths
    M = np.round(T/10000*Fs/K) # min. number of blocks
    V, p2 = segmentation(voiced, M, pitches)
    p2 = V*p2

    pitch = p2[idx]
    p = np.mean(pitch)

    return p # estimated pitch of interval

def compare(p_target, p_test):
    '''
    Checks if p_test is close to p_target withing a given geometric
    tolerance in cents.
    
    Inputs:
        p_target    : centroid of interval
        p_test      : tested pitch
        
    Output:
        checker     : boolean representing comparison result. True
                      if p_test is acceptable, False otherwise
    '''
    
    tolc = 50 # tolerance in cents:
              # 100 cents - difference between 2 consecutive notes
              # 50 cents - halfway between 2 consecutive notes
    
    # define symmetrical tolerance interval
    min_lim = p_target/(2**(tolc/1200))
    max_lim = p_target*(2**(tolc/1200))
    
    checker = min_lim < p_test < max_lim
    
    return checker

def compare_file(p_target, p_file):
    '''
    Checks if p_test is close to p_target withing a given geometric
    tolerance in cents.
    
    Inputs:
        p_target    : centroid of interval
        p_file      : file to be evaluated
        
    Output:
        checker     : boolean representing comparison result. True
                      if p_test is acceptable, False otherwise
    '''
    
    tolc = 50 # tolerance in cents:
              # 100 cents - difference between 2 consecutive notes
              # 50 cents - halfway between 2 consecutive notes
    
    # define symmetrical tolerance interval
    min_lim = p_target/(2**(tolc/1200))
    max_lim = p_target*(2**(tolc/1200))
    
    p_test = Pitch_Tracker_LTP(p_file, 40000, 70000, 200, 1024, .2, 50, 800)
    
    checker = min_lim < p_test < max_lim
    
    return checker