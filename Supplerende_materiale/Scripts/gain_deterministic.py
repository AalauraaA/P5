# -*- coding: utf-8 -*-
"""
This Python file is been made by the project group Mattek5 C4-202

This is the analysis of the gain computation voiced signals.
"""
from __future__ import division
import os
import sys

lib_path = '\\Scripts\\libs'
cwd = os.getcwd()[:-8]
sys.path.insert(0, cwd + lib_path)

import LP_algorithms as lpa
import scipy.io.wavfile as wav

import numpy as np
import matplotlib.pyplot as plt

""" Functions which are used to predict data """
def acf_normalized(data,N):
    R = np.zeros(N)
    for tau in range(N):
         sum_num = 0
         sum_den = 0
         for n in range(N-tau): # Window on datas
             sum_num += data[n]*data[n+tau]
             sum_den += data[n + tau]**2
         R[tau] = sum_num/np.sqrt(sum_den)
    return R
 
def bias(n,N,y0,y1):
    a = (y1-y0)/N
    return a*n + y0

def identify_pitch(R,initial_bias,end_bias):
    N = len(R)
    n0 = 0
    while R[n0] > 0: # Get formant estiamte out of inital peak
        n0 += 1
    bias_array = [bias(n,N-n0,initial_bias,end_bias) for n in range(N-n0)]
    R2 = R[n0:]*bias_array
    return n0 + R2.argmax()

def E_check(r,p,coef):
    E = r[0]
    for k in range(p):
        E += coef[k]*r[k+1]
    return E

def LP_parameters(data,order):
    """
    Takes in data of some size and applies linear prediction algorithms in  
    order to optain parameters of model of the sound. The paramaters are
    coefficients of AR-filter, gain of exatition signal, and
    weather signal is voiced or unvoised. 
    
    The functions utilizes:
        autocorrelation method: in order to obtain coefficients of AR-filter
        here the Levinson Durbin algorithm is used.
        autocorrelation method: to estimate gain (output from Levinson Durbin)
        autocorrelation: to estimate pitch
        xxxxxx: to detect voiced/unvoiced
        
    Input: 
        data: sound file (as numpy array float 64) of some size. (Make this 
        size small prefearibly in order 10-20 ms when sample rate is taken
        into account)
        order: is order of AR-filter 
    Output:
        a: coefficients of AR-filter
        sigma: gain
        voi: voiced/unvoiced (0 if voiced, 1 if unvoiced)
    
    """
    
    # Get filter coefficients with autocorrelation method and lev.durb algo.
    r = acf_fft(data)
    a_r,E = lpa.lev_durb(r,order)
    
    # Compute gain
    gain = np.sqrt(E)
    
    # Detect voiced/unvoiced
    voi = 0 #For now the signal is voiced (detection algorithm comming later)
    
    # Calculate pitch
    acf_norm = acf_normalized(data,len(data))
    pitch = identify_pitch(acf_norm,1,1)
    return a_r,gain,voi,pitch

def exitation(N,voi,pitch,gain):
    if voi == 0: # Voiced
        u = np.zeros(N)
        for n in range(N):
            if n%pitch == 0:
                u[n] = gain
    else:
        raise NameError("Implement unvoiced later on")
    return u

def exitation2(N,voi,pitch,gain):
    if voi == 0: # Voiced
        u = np.zeros(N)
        u[0] = gain
    return u

def LP_predict(N,voi,AR_coef,gain,pitch,n0 = 40):
    """
    predicts signal with given coefficients
    Write more here :) 
    """
    shat = np.zeros(N+n0)
    u = exitation(N+n0,0,pitch,gain)
    for n in range(N+n0):
        shat[n] += u[n]
        for k in range(p):
           if n-(k+1) >= 0:
               shat[n] -= AR_coef[k]*shat[n-(k+1)]
    return shat[n0:],(u/gain)[n0:]
   
def inv_filter(s,N,AR_coef):
    u = np.zeros(N)
    p = len(AR_coef)
    for n in range(N):
        u[n] += s[n]
        for k in range(1,p+1):
            if n-k >= 0:
                u[n] += AR_coef[k-1]*s[n-k]
    return u

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[int(result.size/2):]

def autocorr_biased(X):
    N = len(X)
    R = np.zeros(N)
    for k in range(N):
        for n in range(N-k):
            R[k] += X[n]*X[n+k]
    return R

def autocorr_unbiased(X):
    N = len(X)
    R = np.zeros(N)
    for k in range(N):
        for n in range(N-k):
            R[k] += X[n]*X[n+k]
        R[k] *= 1/(N-k)
    return R

def acf_fft(X):
    """
    X is data signal assumed WSS (wide-sence stationary)
    
    Calculates the estimate of the autocorrelation function from a data set X.
    The calculation is done by the fast Fourier transformation.
    """
    fourier = np.fft.fft(X) 
    S = fourier * np.conj(fourier) # Square root of Periodigram
    return np.real(np.fft.ifft(S))

#==============================================================================
# One voiced file
#==============================================================================
""" Import data """
data = {}
data = wav.read(cwd + "/Lydfiler/Sound/Stemt/jo_a.wav")[1]
data = np.array(data,dtype = "float64")

""" Generate prediction parameters from data signal """
N = 160
N2 = int(N/2)
p = 12 # Order

delay = 399 # Used to slice data signal
d_slice = data[delay:N+delay] # Slice data signal by delay and N

a_r, gain, voi, pitch = LP_parameters(d_slice, p) # Parameters from data signal

s_predict, u = LP_predict(N, voi, a_r, gain, pitch) # Predicted data signal which will be used as test signal

""" Generate prediction parameters from test signal """
ar2, gain2, voi, pitch2 = LP_parameters(s_predict, p) # Parameters from test signal

r = acf_fft(s_predict) # ACF divided by N
a_rdum, Edum = lpa.lev_durb((r*pitch)/N,p) # Calculated filter coefficients and error Ep from r
gaindum = np.sqrt(Edum) # Calculate new gain from Edum
                           
r2 = acf_fft(s_predict[:pitch2]) # ACF not divided by N
a_rgood, Egood = lpa.lev_durb(r2,p) # Calculated filter coefficients and error Ep from r2
gaingood = np.sqrt(Egood) # Calculate new gain from Egood

s_predict2, u2 = LP_predict(N,voi,ar2,gaingood,pitch2) # Predicted test signal with new gain


""" Plot of prediction """
plt.subplot(211)
plt.plot(s_predict)
plt.title("Stemt lyd")
plt.xlabel(r"$n$")
#plt.savefig("figures/stemtgain.png",dpi = 500)
plt.show()

""" Caclculating the deviance """
def deviance(est, teo):
    return 100*(est - teo)/teo

print("Actual gain: %.2f" %(gain))
print("Stupid gain: %.2f" %(gain2))
print("Smart gain: %.2f" %(gaingood))
print("Deviance: %.2f" %(deviance(gaingood,gain)))

#==============================================================================
# All voiced file
#==============================================================================
""" Import data """
dat = {}
names = os.listdir(cwd + "/Lydfiler/Sound/Stemt/")
names.pop(14)
names.pop(-1)
names.pop(36)
for name in names:
    dat[name] = np.array(wav.read(cwd + "/Lydfiler/Sound/Stemt/" + name)\
        [1],dtype = "float64")

""" Predict all the voiced data """
N = 160
p = 12
np.random.seed(1)
M = len(names)
gainerr1 = {}
for name in names:
    K = int(np.floor(len(dat[name])/N))
    nameerr1 = {}
    nameerr2 = {}
    for k in range(K):
        d_slice =  dat[name][k*N:k*N+N] # Slice the data signal
        a_r,gain,voi,pitch = LP_parameters(d_slice,p) # Parameters
        s_pretict, u = LP_predict(N,voi,a_r,gain,pitch) # Prediction
        r = acf_fft(s_pretict) # ACF divided by N
        a_rdum,Edum = lpa.lev_durb((r*pitch)/N,p)
        r2 = acf_fft(s_pretict[:pitch]) # ACF not divided by N
        a_rgood,Egood = lpa.lev_durb(r2,p)
        nameerr1[k] = abs(((gain - np.sqrt(Egood))/gain)*100) # Calculating error
    gainerr1[name] = nameerr1 # Gain error

""" Calculate the error for gain"""
errorplot = []
for name in names:
    for key in gainerr1[name].keys():
        errorplot.append(gainerr1[name][key])
       
print("Average error: %.2f %%" %np.average(errorplot))
print("Variance error: %.2f %%" %np.sqrt(np.var(errorplot)))
