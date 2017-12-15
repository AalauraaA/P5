# -*- coding: utf-8 -*-
"""
This Python file is been made by the project group Mattek5 C4-202

This consist of algorithms which is used to finding residuals for model check
"""
import numpy as np

def Smatrix(data,N,order,delay):
    S = np.zeros((N,order))
    if delay - order < 0:
        raise ValueError("Let delay > order")
    for n in range(N):
        for k in range(order):
            S[n,k] = data[delay + n-k]
    return S

def Hatmatrix(S,Shat):
    return Shat.dot(np.linalg.inv((S.T).dot(S))).dot(S.T)

def cov_res(H,varsighat,sighat,voi):
    N = np.shape(H)[0]
    I = np.identity(N)
    cov_a = varsighat*((I - H).T).dot(I-H)
    if voi == 1: # Unvoiced
        return cov_a + sighat*I
    elif voi == 2: # Voiced
        return cov_a
    elif voi == 0:
        raise ValueError("Silence in not handled")
    
def var_redsiduals(H,varsighat,sighat,voi):
    cov = cov_res(H,varsighat,sighat,voi)
    return np.diag(cov)

def var_error(r,order):
    N = len(r)
    return (r.dot(r.T))/(N-order)

def var_error_student(r,order):
    N = len(r)
    r2 = r**2 # Ssquare values
    r_st = np.zeros(N)
    for n in range(N):
        r_st[n] = np.sum(np.delete(r2,n))
    return r_st/(N-order-1)

def res_standard(data,predict,delay,N,order,gain,voi):
    s = data[delay:delay+N]
    shat = predict[delay-3:delay+N-3]
    r = s - shat                                    # Residuals
    var_sighat = var_error(r,order)                 # Variance of error
    S = Smatrix(data,N,order,delay)
    Shat = Smatrix(predict,N,order,delay)
    H = Hatmatrix(S,Shat)
    var = var_redsiduals(H,var_sighat,gain**2,voi)  # Variance for residuals
    r_s = r/np.sqrt(var)
    return r_s

def res_student(data,predict,delay,N,order,gain,voi):
    s = data[delay:delay+N]
    shat = predict[delay:delay+N]
    r = s - shat                                    # Residuals
    var_sighat = var_error_student(r,order)         # Variance of error
    S = Smatrix(data,N,order,delay)
    Shat = Smatrix(predict,N,order,delay)
    H = Hatmatrix(S,Shat)
    var = var_redsiduals(H,var_sighat,gain**2,voi)  # Variance for residuals
    r_s = r/np.sqrt(var)
    return r_s, var_sighat

def normal_pdf(x,mean,var):
    return (1/(np.sqrt(2*np.pi*var)))*np.exp((-(x-mean)**2)/(2*var))

def middlehist(histbins):
    N = len(histbins) -1
    middlevals = np.zeros(N)
    for n in range(N):
        middlevals[n] = (histbins[n+1] + histbins[n])/2
    return middlevals

def difference(X,P):
    N = len(X)
    Xdif = np.zeros(N-P)
    for n in range(N-P):
        Xdif[n] = X[P + n] - X[n]
    return Xdif

def acf_direct(X):
    N = len(X)
    R = np.zeros(N)
    for k in range(N):
        for n in range(N-k):
            R[k] += X[n]*X[n+k]
    return R

def acf_np(x):
    result = np.correlate(x, x, mode='full')
    return result[int(result.size/2):]

def avg_energy(data,order):
    N = len(data)
    data_avg = np.zeros(len(data))
    for n in range(N):
        for k in range(order):
            if n-k >= 0:
                data_avg[n] += data[n-k]**2
    return np.sqrt(data_avg)    