# -*- coding: utf-8 -*-
"""
This Python file is been made by the project group Mattek5 C4-202

The Python file consist of the algorithms for the window functions. The functions
used the window order M and the length of the x-axis n as input.
"""
import numpy as np

def Rectangular(n,M):
    w = np.zeros(len(n))
    for i in range(len(n)):
        if n[i] >= 0 and n[i] <= M:     
            w[i]= 1
        else:
            w[i] = 0
    return w

def Bartlett(n,M):
    w = np.zeros(len(n))
    for i in range(len(n)):
        if n[i] <= M/2:
            w[i]=(2*n[i])/M
        if n[i] > M/2 and n[i] <= M:
            w[i]=2-(2*n[i]/M)
    return w
    
def Hanning(n,M):
    w = np.zeros(len(n))
    for i in range(len(n)):
        if n[i] >= 0 and n[i] <= M:
            w[i]=0.5-0.5*np.cos((2*np.pi*n[i])/M)
        else:
            w[i] = 0
    return w
 
def Hamming(n,M):
    w = np.zeros(len(n))
    for i in range(len(n)):
        if n[i] >= 0 and n[i] <= M:
            w[i]=0.54-0.46*np.cos((2*np.pi*n[i])/M)
        else:
            w[i] = 0
    return w

def Blackman(n,M):
    w = np.zeros(len(n))
    for i in range(len(n)):
        if n[i] >= 0 and n[i] <= M:
            w[i]=0.42-0.5*np.cos((2*np.pi*n[i])/M)+0.08*np.cos(4*np.pi*n[i]/M)
        else:
            w[i] = 0
    return w
