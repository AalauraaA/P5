# -*- coding: utf-8 -*-
"""
This Python file is been made by the project group Mattek5 C4-202

This script contains algorithms for linear prediction specifically for speech
signals. Both algorithms for computing parameters and predicting by using 
parameters are used. 
"""
from __future__ import division
import LP_algorithms as ma
import windowfunctions as win
import numpy as np

# =============================================================================
# Algorithms for calulating parameters
# =============================================================================

def LP_parameters(data,N_arg,order_arg,overlap,window_arg = win.Rectangular,\
                  fastarg = False,zeta1 = 0.63):
    """
    Calculated parameters for a sound file of some length divided into
    sections of length N with some overlap. Returns the parameters as nested 
    dictionary.
    
    Input:
        data   : Is the the soundfile
        N      : Is length of the sections of which the parameters are calculated
                 for. (Keep around 10-20ms)
        order  : Is order of AR-filter moddeling the sound 
        overlap: Is degree of overlap ex: 0 is 0% 12overlap and .5 is 50% overlap
        window : Is set to be the rectangular window
        fastarg: If True calculating the pitch will be done with FFT (Not the 
                 real results by make the calculating faster)
        zeta1  : Is the end point for the linear function used in pitch estimation.
                 It set to be 0.63.
    
    Output:
        parameters: Nedsted dictionary with parameters of each individual segment
    """
    # Declare global variables used multible times in algorithm
    global order, N, idx,bias_arr, lastslice, fast, mean, cov,firstslice,window
    
    # Load traning data
    mean, cov = load_training()
    
    # The global variables
    order = order_arg
    N = N_arg
    fast = fastarg
    window = window_arg
#    data = data.copy() #Copy data in order not to mess with it when windowd
    idx = np.arange(N) # Used for setting up autocorrelation 
    
    if overlap >= 1 or overlap < 0:
        raise ValueError("Let overlap be in the interval [0,1)")
    
    if len(data) < N:
        raise ValueError("Let N < len(data)")
    
    N_o = int(np.floor(N*overlap))             # Number of samples that overlap in each segment
    N_d = N-N_o                                # Delay between each window
    N_sec = int(np.floor((len(data)-N)/(N_d))) # Number of parameter evaluations

    parameters = dict.fromkeys(range(N_sec))   # Dictunary for holding parameters

    parameters['order'] = order
    parameters['N'] = N
    parameters['delay'] = N_d
    
    # Compute bias array for later use
    bias_arr = bias_array(1,zeta1)
    lastslice = False
    
    # Setting the parameters up for each slide
    parameters[0] = LP_parameters_slice(data[0:2*N],firstslice =True)
    for para in range(1,N_sec):
        if para*(N_d) + 2*N <= len(data): 
            # Calculates parameters for each slice
            # (2N length slices are passed to estimate pitch with more cirtanty
            parameters[para] = LP_parameters_slice(data[para*(N_d)-1:para*(N_d) + 2*N])
        else:
            lastslice = True
            # Only pass length N slice for end slices in order for pitch 
            # detection not to evaluate undefined samples
            parameters[para] = LP_parameters_slice(data[para*(N_d)-1:\
                                  para*(N_d) + N])
    return parameters

def LP_parameters_slice(dat,firstslice = False):
    """
    Takes in data of some size and applies linear prediction algorithms in  
    order to optain parameters of model for the sound. The paramaters are
    coefficients of AR-filter, gain of excatition signal, and
    weather signal is voiced or unvoiced. 
    
    The functions utilizes:
        Autocorrelation method: in order to obtain coefficients of AR-filter
                                here the Levinson Durbin algorithm is used.
        Autocorrelation method: to estimate gain (output from Levinson Durbin)
        Autocorrelation: to estimate pitch
        Serveral methods: to detect voiced/unvoiced
        
    Input: 
        data : sound file (as numpy array float 64) of some size. (Make this 
               size small prefearibly in order 10-20 ms when sample rate is taken
               into account)
        order: is order of AR-filter 
        
    Output:
        a    : coefficients of AR-filter
        sigma: gain
        voi  : voiced/unvoiced (0 if silence, 1 if unvoiced, 2 if voiced, 
               4 if packetloss )
        
    Is returned as dictionary with keys:
        - "coef", "gain", "voi", "pitch" and "first_imp"
    """
      
    # Data shall only be used for pitch estimation and rest only half length

    # Detect voiced/unvoiced
    dat = dat.copy()
    if firstslice:
        dat0 = 0
        datshort = dat[:N]*window(range(N),N)   # Window segment
        r_energy = acf_fft(dat)                 # Windowing will affect gain computation
                                                # Therefore r_energy is calculated before windowing to avoid this
        dat *= window(range(2*N),2*N)
        r = acf_fft(dat)
    elif lastslice:
        dat0 = dat[0]
        datshort = dat[1:N+1]*window(range(N),N)
        r_energy = acf_fft(dat[1:])             # See firstslice comment
        dat[1:] *= window(range(N),N)           # Window segment
        r = acf_fft(dat[1:])
    else:
        dat0 = dat[0]
        datshort = dat[1:N+1]*window(range(N),N)
        r_energy = acf_fft(dat[1:])             # See firstslice comment
        dat[1:] *= window(range(2*N),2*N)       # Window segment
        r = acf_fft(dat[1:])
        
    # Check for packet loss
    if np.all(datshort == 0): # Voi = 4 is code for packetloss
        return {"coef":None,"gain":None,"voi":4,"pitch":None,"first_imp":None}
    
    # Get filter coefficients with autocorrelation method and lev.durb algorithm
    a_r, E = ma.lev_durb(r,order) # The error here will not be used because:
                                  # a: For unvoiced the autocorrelation is scaled wrong leading to wrong E
                                  # b: For voiced the error cal. is based on only a single impulse not 
                                  #    an impulse train. So autokorrelation is wrong based on this cal
    
    # Detect if silence, voiced or unvoiced
    voi = vu_detector(datshort,dat0,r[1],a_r[0],E) 
#    voi = 2 # Either silence, voiced  or unvoiced
    if voi == 1: #If unvoiced the autocorrelation must be devided by N
        r /= len(dat)
        r_energy /= len(dat)
    
    # Calculate the pitch - Computes autocorrelation for various functions
    if not fast:    # Under assumption signal is not perfectly stationary
                    # (affects pitch estimation)
        if firstslice:
            r_normalised = acf_normalized_fft(dat,r)
        else:
            r_normalised = acf_normalized_fft(dat[1:],r)
    else: # Under assumption signal is almost perfectly stationary (FAST! 10x)
        r_normalised = r
    if voi == 2: # Voiced
        pitch = identify_pitch(r_normalised)
        # Identify placement of first impulse
        peak0 = first_impulse(datshort,pitch)
        delta = delta_impulse(a_r)
        imp0 = peak0 - delta
    elif voi == 1: # Unvoiced
        pitch = None
        imp0 = None
    else: # Silence
        E = (np.linalg.norm(dat)**2)/(len(dat))
        gain = np.sqrt(E)
        return {"coef":None,"gain":gain,"voi":voi,"pitch":None,"first_imp":None}
   
    # Compute gain
    if voi == 1:                            # Unvoiced
        E = E_p(r_energy,a_r)               # Recalculate error (r have been updated)
    elif voi == 2:                          # Voiced
        rtemp = (pitch/len(dat))*r_energy   # Calulates acf using information about pitch
        E = E_p(rtemp,a_r)
    if E < 0:                               # For silence miss-detected negative error can occour
        E = 0                               # set to zero in this case
    gain = np.sqrt(E)  
    return {"coef":a_r,"gain":gain,"voi":voi,"pitch":pitch,"first_imp":imp0}

def acf_normalized_fft(data,r):
    """
    Computes ACF for pitch estimation but where the denominator 
    is not considerd stationary    
    """
    global N
    if lastslice:
        Ncopy = N
        N = int(N/2)
    r_nonsta = acf_non_stationary(data) # ACF not considered stationary
    if lastslice:
        r_normalized = r[:N]/r_nonsta
        N = Ncopy
    else:
        r_normalized = r/r_nonsta
    return r_normalized

def acf_fft(data):
    R = np.fft.fft(data)                    # Fourier transform of data
    R_a2 = R*np.conj(R)                     # Absolut value of squared |R|^2
    return np.real(np.fft.ifft(R_a2))[:N]   # ACF by inverse fourier transform

def E_p(r,coef): 
    """
    Calculated total squared error E_p based on autocorrelation coefficients
    and coefficients for AR process of some order. 
    
    Input:
        r   : ACF coefficients
        coef: Coefficients for ar process
   
    Output:
        E   : Total squared error
    """
    E = r[0]
    for k in range(order):
        E += coef[k]*r[k+1]
    return E


# The following 3 algorithms performs the acf calculation of the nonstationary 
# process (acf_diagonals, acf_matrix, acf_non_stationary)
def acf_non_stationary(data):
    """
    Estiamates acf, sqrt(r[n]), for nonstationary signal of length N
    Calculates by setting up the sums as a matrix vector product which makes
    it faster.
    
    Input:
        data: Data array of length 2N-1
        N   : Length of acf sequence
    Output:
        Autocorrelation sequence sqrt(r[n,n]) for n = 0,1,...,N-1
        
    Performance analysis:
    
    The algorithm mainly takes computational power by:
        A: Specifiyng diagonals of matrix - Mdiagonals
        B: Initialise matrix              - spa.diags
        C: Computing the actual matrix vector product
    
    By analasys it is found that:
        A: takes about 86% of the time
        B: takes about 10% of the time
        C: takes about 2% of the time
    
    """
    acf_diag = acf_diagonals(data)      # Values for acf diagonals
    acf_M = acf_matrix(acf_diag)        # Autocorrelation Matrix
    acf_arr = data[:-1]                 # Array for muliplication
    return np.sqrt(acf_M.dot(acf_arr))  # The multiplication itself is fast

def acf_diagonals(a):
    diag = np.zeros((N,N))
    for i in range(N):
        diag[i] = a[i:i+N]
    return diag

def acf_matrix(diags):
    """
    Sets up autocorrelation sequences as a matrix so the multilication 
    can be performed in the function acf_non_stationary
    """
    matrix = np.zeros((N,2*N-1))
    for n in range(N):
        matrix[idx[:N],idx[:N]+n] = diags[n]
    return matrix

def bias_array(y0,y1,n0 = 15):
    """
    Returns linear line y og length N-n0 with y(0) = y0 and y(1) = y1
    """
    a = (y1-y0)/(N-n0)
    array = [a*n+y0 for n in range(N-n0)]
    return np.array(array)

def identify_pitch(R,n0 = 15):
    """
    Identify pitch given autocorrelation sequence and lower bound n0
    
    Input:
        R : Normalised autocorrelation sequence
        n0: Lower bound for pitch
    
    Output:
        pitch (Formant 1)
    """
    if not lastslice:
        R2 = R[n0:]*bias_arr
    else:
        R2 = R[n0:]
    return n0 + R2.argmax()

def first_impulse(data,pitch):
    """
    Identify placement of first impulse in voiced signal. The first peak
    is identified but typicially first impulse is a few samples before
    therefore the delta value. This algorithms uses the fact that the first
    impulse must occur in the interval [0:pitch] - hence not after a whole
    pitch period
    
    Input:
        data : data array of length N
        pitch: pitch (Formant 1) 
        delta: the typical value the first impulse is before the first peak
    Output:
        The index of the first impulse
    """
    X = data[:pitch]
    imp0 = X.argmax()
    return imp0

def delta_impulse(a_r,MAX = 12):
    """
    Computes the number of samples the first impulse occurs before peak of 
    impulse response, calculate by coefficients a_r. 
    
    Input:
        a_r: Filter coefficients
        MAX: Maximum number of iterations (usually delta is about 4)
    
    Output:
        The number of samples the first peak arrrives after impulse
    """
    h = np.zeros(MAX)
    h[0] = 1
    for n in range(1,MAX):
        for k in range(1,order+1):
            if n-k >= 0:
                h[n] -= a_r[k-1]*h[n-k]
    return np.argmax(h)


# =============================================================================
# Algorithms for voiced unvoiced detection
# =============================================================================
def LogEnergy(frames):
    """
    The log energy for a block defined as 10*log(epsilon + 1/N sum_{n=1}^N s^2(n)).
    N is the frame size.
    This takes a dictionary or a numpy array as an input
    Epsilon is a very small positive constant being << mean-squared value of the speach sampl 1 * 10^(-5)
    And s(n) is the signal in the block (So we are just calculating the power in the given block plus a small constant and then looking at it logorithmicly)
    """
    epsilon1 = 1*10**(-5)
    if type(frames) == dict:
        EnergyS = {key: 10* np.log(epsilon1 + (1./len(frames[key]))*np.sum(frames[key]**2)) for key in frames}
    else:
        EnergyS = 10* np.log(epsilon1 + (1./len(frames))*np.sum(frames**2))
    return EnergyS

def ZCRate(frame): # Zero cross rate
    """
    The Zero crossing for a framed data set.
    This functions takes a framed signal in a dictionary as an input or a numpy array
    Returns a numpy array
    """
    if type(frame) == dict:
        zcrframe =  np.array([1 *(np.diff(np.sign(frame[key])) != 0).sum() for key in frame])
    else:
        zcrframe = 1 *(np.diff(np.sign(frame)) != 0).sum()
    return zcrframe

def NormAcfCoef(frame, acf_coef,dat0): # Nomalized Autocorrelation coefficient
    energysumframe = np.sum(frame**2)
    energysumframeshift = np.sum(frame**2) + dat0**2
    return (acf_coef/(np.sqrt(energysumframe*energysumframeshift)))

def NPE(logenergy, error):
    epsilon2 = 1*10**-6
    return logenergy - (10 * np.log(epsilon2 + np.abs(error)))

def di(x,i):
    """
    Calculates the dispertion
    """
    di = 1/2 * np.dot( np.dot((x-mean[i]).T,np.linalg.inv(cov[i])), (x-mean[i]))
    return di

def P(i,d1,d2,d3):
    dummy = 1/(d1*d2+d2*d3 + d1*d3)
    if i == 1:
        return d2*d3*dummy
    elif i == 2:
        return d1*d3*dummy
    else:
        return d1*d2*dummy

def section(P1,P2,P3):
    if P1 > P2 and P1 > P3:
        return 0
    elif P2 > P3 and P2 > P1:
        return 1
    else:
        return 2
    
def vu_dec_inner(zcrframe,energy, normr_1, a_0_coef, NPE):
    X = (zcrframe,energy, normr_1, a_0_coef, NPE)
    d1 = np.array([di(X,1)])
    d2 = np.array([di(X,2)])
    d3 = np.array([di(X,3)])

    P1 = np.array([P(1,d1,d2,d3)])
    P2 = np.array([P(2,d1,d2,d3)])
    P3 = np.array([P(3,d1,d2,d3)])
    return section(P1,P2,P3)

def vu_detector(data, lastpreviousdata, acf_1, pridictioncoef, error):
    zcrframe = ZCRate(data)
    energy   = LogEnergy(data)
    normr_1  = NormAcfCoef(data,acf_1, lastpreviousdata)
    npe     = NPE(energy,error)
    return vu_dec_inner(zcrframe, energy, normr_1, pridictioncoef, npe)


# =============================================================================
# Algorithms for predicting signals
# =============================================================================
def LP_predict(parameters):
    """
    Models/predicts sound signal from parameters and preditcs signal with 
    segments of given overlap using a window
    
    Input:
        parameters:  Dictionary containing length of each section "N", order of
        used filter "order" and a delay parameter "delay" that determines the
        overlap used in the parameters. It also contains the parameters for
        each segment:
              voi      : 0 if silence, 1 if unvoiced, 2 if voiced and 4 if packetloss
              coef     : Coefficients of AR-filter (of length order)
              pitch    : Pitch if signal is voiced (== None for unvoiced)
              first_imp: Index where first impulse is placed in each slice
            
    Output:
        signal: Predicted signal with the given parameters
    
    """
    global N, signal, N_d, para
    N = parameters["N"]
        
    global order
    order = parameters["order"]
    
    N_d = parameters["delay"]
    N_parameters = len(parameters) - 3
    data_len = N_parameters*N_d + N     # Length of final data array
    
    # Data and window structure
    signal = np.zeros(data_len)         # Is global variable
    para_shift = 0
    if parameters[0]['voi'] == 4:       # If the first packet is lost - create silence
        parameters[0]['voi'] = 0
        parameters[0]['gain'] = 10
    for para in range(N_parameters):
        while parameters[para-para_shift]['voi'] == 4:  # Packet loss handeling
            para_shift += 1                             # Call earlier packet if packet loss is found
        paratemp = para - para_shift
        LP_predict_slice(parameters[paratemp]['voi'],\
                parameters[paratemp]['coef'],parameters[paratemp]['gain'],\
                parameters[paratemp]['pitch'],parameters[paratemp]['first_imp'])        
        para_shift = 0
    return signal

def LP_predict_slice(voi,AR_coef,gain,pitch,imp0):
    """
    Preditcs sound signal of lenght N given parameters as AR-process with 
    either white noice or impulse train input depending if the signal is 
    unvoiced or voiced
    
    Input: 
        N      : Lenght of desired signal (keep low about 20 ms)
        voi    : 0 if silence, 1 if unvoiced, 2 if voiced and 4 if packetloss
        AR_coef: Coefficients of AR-filter (of length order)
        pitch  : Pitch if signal is voiced (== None for unvoiced)
        order  : Order of AR_filter
        
    Output:
        shat   : Signal of length N with given parameters
    """
    global signal 
    signal_temp = signal.copy()
    signal_temp[para*N_d:] = 0 
    if voi == 2:                # Voiced
        n0 = pitch - imp0       # Specify placement of first impulse for voiced signals number of extra samples calculated
    elif voi == 1:              # Unvoiced
        n0 = 0
    else:                       # Silence
        signal[para*N_d:para*N_d + N] =  np.random.normal(0,gain**(.5),N)
        return None
    u = exitation(N+n0,voi,pitch,gain)[n0:] # Compute excitation signal
    for n in range(N):
        AR_sum = 0
        for k in range(1,order+1):
            if para*N_d + n-k >= 0:
                AR_sum += AR_coef[k-1]*signal_temp[para*N_d + n - k]
        signal_temp[para*N_d + n] += -AR_sum + u[n]
    signal[para*N_d:para*N_d+N] += signal_temp[para*N_d:para*N_d+N]
    if para != 0:
        signal[N_d*para: N_d*(para-1) + N] /= 2 # Take average where there is overlap

def exitation(N,voi,pitch,gain): # Leave N as parameter as it varies here
    """
    Creates excitaiton signal of length N. If voiced the signal will be an
    impulse tain. If unvoiced the signal with be white noice. 
    
    Input:
        N    : Length of signal
        voi  : Parameter for determining if voiced or unvoiced
        pitch: Period for impulse train for voiced signal. None for unvoiced
               signal
        gain : Gain for exitation signal
        
    Output:
        u: Exitation signal of length N
    
    """
    if voi == 2:            # Voiced
        u = np.zeros(N)
        for n in range(N):
            if n%pitch == 0:
                u[n] = gain
    else:                   # Unvoiced
        u = gain*np.random.normal(0,1,N)
    return u

# =============================================================================
# Some extra functions
# =============================================================================       
def acf_normalized_stationary(data):
    """
    This functions computes acf where (in pitch calculation) it is assumed
    that the signal is totally stationary. It can give some errors but 
    computes way faster (about 10 times faster)
    """
    R = np.fft.fft(data)                 # Fourier transform of data
    R_a2 = R*np.conj(R)                  # Absolut value of squared |R|^2
    r = (1/N)*np.real(np.fft.ifft(R_a2)) # ACF by inverse fourier transform
    r_normalized = r/np.sqrt(r[0])
    if not lastslice:
        return r,r_normalized[:N]
    else:
        return r,r_normalized[:int(N/2.)]

# =============================================================================
# Traning data
# =============================================================================
def load_training(): # Silence
    mean = {
            3: np.array([  18.23175586,  130.94787753,    0.90586369,   -1.37565401,
         -2.12955145]),
            2: np.array([  7.71744186e+01,   1.07949007e+02,   6.00377773e-02,
        -1.41348726e-01,  -3.82792344e+01]),
            1: np.array([  7.5211626 ,  14.7204753 ,   0.97313362,  -1.20991747,  -1.98691267])}
    
    cov  = {
            3: np.array(
        [[ 1.        ,  0.12658239, -0.90958435,  0.35086631, -0.73782621],
         [ 0.12658239,  1.        , -0.12396601, -0.20439514, -0.03435325],
         [-0.90958435, -0.12396601,  1.        , -0.42966741,  0.70615914],
         [ 0.35086631, -0.20439514, -0.42966741,  1.        , -0.57631987],
         [-0.73782621, -0.03435325,  0.70615914, -0.57631987,  1.        ]]),
            2: np.array(
        [[ 1.        , -0.3474003 , -0.96841622,  0.82872069, -0.29189177],
         [-0.3474003 ,  1.        ,  0.34370121, -0.3560751 ,  0.35451654],
         [-0.96841622,  0.34370121,  1.        , -0.86644825,  0.28595854],
         [ 0.82872069, -0.3560751 , -0.86644825,  1.        , -0.36690416],
         [-0.29189177,  0.35451654,  0.28595854, -0.36690416,  1.        ]]),
            1: np.array(
        [[ 1.        , -0.00563084, -0.89764   ,  0.428191  , -0.7502164 ],
         [-0.00563084,  1.        , -0.06899344, -0.13287582,  0.31625235],
         [-0.89764   , -0.06899344,  1.        , -0.44524101,  0.64268221],
         [ 0.428191  , -0.13287582, -0.44524101,  1.        , -0.45632161],
         [-0.7502164 ,  0.31625235,  0.64268221, -0.45632161,  1.        ]])}
    return mean,cov

def load_training2():
    mean = {1: np.array([25.663, 10.781, 0.649, -0.935, 4.976]),
            2: np.array([49.914, 23.439, 0.007, -0.107, 3.661]), #0
            3: np.array([12.775, 50.608, 0.881, -2.256, 18.944])} #1

    cov = {1: np.array([[ 1.000, -0.032, -0.842,  0.386, -0.629],
                        [-0.032,  1.000, -0.098, -0.558,  0.580],
                        [-0.842, -0.098,  1.000, -0.442,  0.596],
                        [ 0.386,  -0.558, -0.442,  1.000, -0.710],
                        [-0.629,  0.580,  0.596, -0.710,  1.000]]),
    
           2: np.array([[ 1.000,  0.471, -0.959,  0.909, -0.019],
                        [ 0.471,  1.000, -0.454,  0.437,  0.447],
                        [-0.959, -0.454,  1.000, -0.947,  0.028],
                        [ 0.909,  0.437, -0.947,  1.000, -0.044],
                        [-0.019,  0.447,  0.028, -0.044,  1.000]]),
    
           3: np.array([[ 1.000,  0.471, -0.959,  0.909, -0.019],
                        [ 0.471,  1.000, -0.454,  0.437,  0.447],
                        [-0.959, -0.454,  1.000, -0.947,  0.028],
                        [ 0.909,  0.437, -0.947,  1.000, -0.044],
                        [-0.019,  0.447,  0.028, -0.044,  1.000]])}
    return mean,cov
    
