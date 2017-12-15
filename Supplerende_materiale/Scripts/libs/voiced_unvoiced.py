# -*- coding: utf-8 -*-
"""
This Python file is been made by the project group Mattek5 C4-202

The Python file consist of:
    - Algorithms behind the voiced/unvoiced detector
    - The means and covariance which are used in the detector
"""
import os
import numpy as np
cwd = os.getcwd()

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


""" Functions for the detector """
def ha(n,M,a = 0.54): # Hann window, if a = 0.5. Hamming window, if a = 0.54.
    w = np.zeros(len(n))
    for i in range(len(n)):
        if n[i] >= 0 and n[i] <= M:
            w[i] = a - (1 - a)*np.cos((2*np.pi*n[i])/M)
        else:
            w[i] = 0
    return w

def framing(data, p, w, wlen):
    wlenHalf = np.int(wlen/2)
    frameamount = int(np.floor(len(data)/(wlenHalf)))
    frame = {i: data[(wlenHalf)*i:wlen + (wlenHalf)*i]*w for i in range(frameamount-1)}
    return frame

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
        zcrframe = 1 *(np.diff(np.sign(frame[key])) != 0).sum()
    return zcrframe

def NormAcfCoef(frame, acf_coef): # Nomalized Autocorrelation coefficient
    energysumframe = np.array([np.sum(frame[i]**2) for i in range(len(frame))])
    energysumframeshift = np.array([np.sum(frame[i][:-1]**2) for i in range(len(frame))])
    for i in range(1,len(frame)):
        energysumframeshift[i] = energysumframeshift[i] + frame[i-1][-1]**2
    normr_1 = (acf_coef/(np.sqrt(energysumframe*energysumframeshift)))
    return normr_1 

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
    
def vu_detector(zcrframe,energy, normr_1, a_0_coef, NPE):
    X = (zcrframe,energy, normr_1, a_0_coef, NPE)
    d1 = np.array([di(X,1)])
    d2 = np.array([di(X,2)])
    d3 = np.array([di(X,3)])

    P1 = np.array([P(1,d1,d2,d3)])
    P2 = np.array([P(2,d1,d2,d3)])
    P3 = np.array([P(3,d1,d2,d3)])
    return section(P1,P2,P3)

def voi(data, lastpreviousdata, acf_1, pridictioncoef, error):
    zcrframe = ZCRate(data)
    energy   = LogEnergy(data)
    normr_1  = NormAcfCoef(data,acf_1)
    npe     = NPE(energy,error)
    return vu_detector(zcrframe, energy, normr_1, pridictioncoef, npe)