# -*- coding: utf-8 -*-
"""
This Python file is been made by the project group Mattek5 C4-202

The Python file consist of:
    - General algorithms for linear prediction including the autocorrelation 
      and the covariance method. 
    - Algorithms for solving systems
"""
from __future__ import division
import numpy as np
import scipy.linalg as lin

# =============================================================================
# Algotithms for the autocorrelation method
# =============================================================================
def acf_fft(X):
    """
    X is data signal assumed WSS (wide-sence stationary)
    
    Calculates the estimate of the autocorrelation function from a data set X.
    The calculation is done by the fast Fourier transformation.
    """
    fourier = np.fft.fft(X) 
    S = fourier * np.conj(fourier) # Square root of periodigram
    return np.real(np.fft.ifft(S))

def acf_direct(X):
    N = len(X)
    R = np.zeros(N)
    for k in range(N):
        for n in range(N-k):
            R[k] += X[n]*X[n+k]
    return R

def lev_durb(r,p):
    """    
    Levinson Durbin algorithm for solving systems Rx = r
    where R is p x p toepliz matrix and r is the the 1 to p-1 values in
    first columb of R + an extra value
    
    Computes with complexity O(N^2)
    
    r: Is the (typically autocorrelation) vector the values in R and r comes from
    p: Is number of parameters for the matrix system
      (determines the size of R and r)
    
    """
    E = r[0] # Set error 0
    a = np.zeros(p+1)
    #  a[0] = -1
    for i in range(1,p+1):
        # Calculate the sum for k_i
        ki_sum = r[i] # Start value
        for j in range(1,i):
            ki_sum += a[j]*r[i-j]
        k = -ki_sum/E # The k_i direction value
        a[i] = k 
        if i > 1: # Update the other a's for when they exist
            a_temp = a.copy()
            for j in range(1,i+1):
                a[j] = a_temp[j] + k*a_temp[i-j]
        E = (1-k**2)*E # Update the error
    return a[1:],E # First index in a is zero, return rest

# =============================================================================
# Algorithms for the covariance method including cross correlation methods
# =============================================================================
def cross_corr_val(X,i,k,p,N):
    """
    X is data array of length N
    p is the final range on which phi_{i,k} will be calulated over eventually
    and affects how many values of X phi_{i,k} can be evaluated over
    
    Calculates cross correlation  phi_{i,k} for data by direct implementation
    
    Returns phi_{i,k}
    """
    cross_sum = 0
    # p is added to indexation so no negative values are evaluated for exampel phi_{p,p}
    for n in range(p,N): 
        cross_sum += X[n-i]*X[n-k]
    return cross_sum

def cross_corr_mat(X,p):
    """
    X is data array of length N
    p is dimension of Phi_{i,k} matrix
    returns pxp cross correlation matrix for system a Phi = -phi
    """
    N = len(X)
    if p >= N/2:
        raise ValueError("Choose p < len(X)/2")
    Phi = np.zeros((p,p))
    for k in range(p):
        for i in range(p):
            # Matrix is symmetric so k > i means that the values have
            # already been calculated so this value is just put in
            if k > i:
                Phi[k,i] = Phi[i,k] #Put in value for upper half matrix
            else:
                Phi[k,i] = cross_corr_val(X,i+1,k+1,p,N)
    return Phi

def cross_corr_arr(X,p):
    """
    X is data array of length N
    p is dimension of phi_{i,k} array
    Returns p length cross correlation array for system a Phi = -phi
    """
    N = len(X)
    if p >= N/2:
        raise ValueError("Choose p < len(X)/2")
    phi = np.zeros(p)
    for i in range(p):
        phi[i] = cross_corr_val(X,0,i+1,p,N)
    return phi

def cross_corr_sys(X,p):
    """
    X is data array of length N
    p is dimension of parameter vector
    Returns
     - pxp dim cross correlation matrix Phi
     - p dim cross correlation array phi
    """
    return cross_corr_mat(X,p), cross_corr_arr(X,p)

def Kov(phimat,phiarr):
    """
    Solve system (Φa=φ) that arises using covariance method.
    Returns coefficient vector (a)
    
    """    
    # Creating an identity matrix V and d with zeros
    k, p = np.shape(phimat)
    V = np.identity(p)
    d = np.zeros(p) # Let's just make d a vector instead of a diagonal matrix
    # Computation of V and d in Cholesky decomposition
    d[0] = phimat[0][0] # (4.23) (Please notice that numbers of equations are subject to change)
    for i in range(1,p):
        V[i][0] = phimat[i][0]/d[0] # (4.22)
    for j in range(1,p-1):
        Vsum = 0
        for k in range(j):
            Vsum += (V[j][k]**2)*d[k]
        d[j] = phimat[j][j] - Vsum # (4.23)
        for i in range(j+1,p):
            V2sum = 0
            for k in range(j):
                V2sum += V[i][k]*d[k]*V[j][k]
            V[i][j] = (phimat[i][j] - V2sum)/d[j] # (4.22)
    V3sum = 0
    for k in range(p-1):
        V3sum += (V[-1][k]**2)*d[k]
    d[-1] = phimat[-1][-1] - V3sum
    # Computation of Y
    Y = np.zeros(p)
    Y[0] = phiarr[0] # (4.25)
    for i in range(1,p):
        Ysum = 0
        for j in range(i):
            Ysum += V[i][j]*Y[j]
        Y[i] = phiarr[i] - Ysum # (4.25)
    # Computation of a
    avec = np.zeros(p)
    avec[-1] = Y[-1]/d[-1] # (4.26)
    for i in range(p-2,-1,-1):
        asum = 0
        for j in range(i+1,p):
            asum += V[j][i]*avec[j]
        avec[i] = Y[i]/d[i] - asum
    return avec

# =============================================================================
# Various algotithms
# =============================================================================
def AR_design_matrix(x,N,p):
    """
    Compute the Ar toeplitz design matrix

    Input:
    x     A length Nx1 real numpy array
    N     Row size integer
    p     A starting point from the data set x(n)

    Output:
    H     A size NxM numpy array
    
    """
    return lin.toeplitz(x[p-1:p+N-1],np.flipud(x[:p]))