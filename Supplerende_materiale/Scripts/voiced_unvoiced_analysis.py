# -*- coding: utf-8 -*-
"""
This Python file is been made by the project group Mattek5 C4-202

This is the analysis of the voiced/unvoiced detector for difference signals
"""
from __future__ import division
import os
import sys

cwd = os.getcwd()[:-8]
lib_path = '\\Scripts\\libs'
data_path = '\\Lydfiler\\Sound\\Data_saet'
sys.path.insert(0, cwd + lib_path)

#export_path = '\data\predicted_sound'

import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
import LP_algorithms as lpa
import voiced_unvoiced as vu

os.chdir(cwd + data_path)

""" Finding a data file """
number = '04' # Change from 01 to 14
#fill = number + '-ustemt' # A data file consist of known unvoiced
#fill = number + '-stemt' # A data file consist of known voiced
#fill = number + '-ustemt(engelsk)' # A data file consist of unknown unvoiced
#fill = number + '-stemt(engelsk)' # A data file consist of unknown voiced
#fill = number + '-kontrol-stemt' # A data file consist of controlled voiced
#fill = number + '-kontrol-ustemt' # A data file consist of controlled unvoiced
fill = 'sentences' # A data file consist of a sentence

downsampling = 0
downsampleconst = 6

""" Function to classification of sounds """
def di(x,i):
    di = 1/2 * np.dot( np.dot((x-vu.mean[i]).T,np.linalg.inv(vu.cov[i])), (x-vu.mean[i]))
    return di

def P(i,frame):
    dummy = 1/(d1[frame]*d2[frame]+d2[frame]*d3[frame] + d1[frame]*d3[frame])
    if i == 1:
        return d2[frame]*d3[frame]*dummy
    elif i == 2:
        return d1[frame]*d3[frame]*dummy
    else:
        return d1[frame]*d2[frame]*dummy

""" Importing data to classify """
if downsampling == False:
    f, data = wav.read(fill + '.wav')
else:
    fullf, fulldata = wav.read(fill + '.wav')
    dummy = np.copy(fulldata)
    data = np.array([fulldata[i] for i in range(0,len(fulldata),downsampleconst)])
    f = int(fullf/downsampleconst)

if np.max(np.abs(data)) < 2:
    while np.max(np.abs(data)) < 2000:
        data = 10*data

#normplotdata = data/(10*np.max(np.abs(data)))
#normplotdata1 = data/(200*np.max(np.abs(data)))
#normplotdata2 = data/(np.max(np.abs(data)/100))
#normplotdata3 = data/(4*np.max(np.abs(data)/100))

p = 12
wlen = int(f * 0.02)    # 20 ms window
w = vu.ha(np.arange(wlen),wlen)

frame = vu.framing(data,p,w,wlen) #wlen kan fjernes fra def blot ved at tage len af (w) hvor det er nødvendigt med wlen. Optimering jeg ikke lige gider roder med.

""" Testing """
# Using lp functions to generate a dictionary of acf and
R = {key: lpa.AR_design_matrix(frame[key],wlen-p,p) for key in frame} #The design matrix
r = {key: lpa.acf_fft(frame[key][:wlen]) for key in frame} #The autocorrelation function
a_acf = {key: lpa.lev_durb(r[key],p)  for key in frame} #Acf A values
EstAcf = {key: np.dot(R[key], a_acf[key][0]) for key in frame} #The estimat of the acf


"""
The order of the x vector is; 
The Energy, The ZCR, The normalized Autocorrelation coefficient, 
the a_0 coefficient and the nomalized predicition error (NPE)
"""

""" Energy """
energyf = vu.LogEnergy(frame)
energy = np.array([energyf[i] for i in range(len(energyf))])

""" ZCR """
zcrframe = vu.ZCRate(frame)

""" Nomalized autocorrelation coefficient """
r_1_coef = np.array([r[i][1] for i in range(len(r))])

energysumframe = np.array([np.sum(frame[i]**2) for i in range(len(frame))])
energysumframeshift = np.array([np.sum(frame[i][:-1]**2) for i in range(len(frame))])

for i in range(1,len(frame)):
        energysumframeshift[i] = energysumframeshift[i] + frame[i-1][-1]**2

normr_1 = (r_1_coef/(np.sqrt(energysumframe*energysumframeshift)))

""" a0 coefficient """
a_0_coef = np.empty(len(frame))
for i in range(len(frame)):
   a_0_coef[i] = a_acf[i][0][0]

timeaxis1 = np.linspace(0,len(data)/f,len(a_0_coef))
timeaxis2 = np.linspace(0,len(data)/f,len(data))
normplotdata = data/(np.max(np.abs(data)))

""" Normalized prediction error (NPE) """
epsilon2 = 1*10**-6
NPE = np.array([energy[i] - (10 * np.log(epsilon2 + np.abs(a_acf[i][1]))) for i in range(len(a_acf))])


""" Creating the X vector, with the information from the 5 test """
X = np.vstack((zcrframe,energy, normr_1, a_0_coef, NPE))
X = X.T

#m = np.sum(X, axis=1)*1./len(X.T)

#w = np.zeros([5,5])
#for i in range(len(X.T)):
#    w += np.outer(X.T[i],X.T[i])
#
#w = (1./len(X.T)) * w - np.outer(m,m)
#
#normw = np.zeros([5,5])
#for i in range(5):
#    for j in range(5):
#        normw[i][j] = w[i][j] / np.sqrt(w[i][i]*w[j][j])

""" Silence/voiced/unvoiced """
d1 = np.array([di(X[i],1) for i in range(len(frame))])
d2 = np.array([di(X[i],2) for i in range(len(frame))])
d3 = np.array([di(X[i],3) for i in range(len(frame))])

P1 = np.array([P(1,i) for i in range(len(frame))])
P2 = np.array([P(2,i) for i in range(len(frame))])
P3 = np.array([P(3,i) for i in range(len(frame))])

sections = np.array([vu.section(P1[i],P2[i],P3[i]) for i in range(len(frame))])

os.chdir(cwd + "\Scripts")

""" Plots """
plt.title("Stemt/ustemt analyse af ustemt signal")
plt.plot(timeaxis2, normplotdata, label = 'Skaleret signal')
plt.plot(timeaxis1, sections, label = 'Stemt (2) / Ustemt (1) / Stilhed (0)')
plt.xlabel("Tid [s]")
plt.legend()
plt.savefig("figures/voi_unvoi_analysis_%s.png" % fill, dpi = 500)
plt.show()

#print(len(sections),f)