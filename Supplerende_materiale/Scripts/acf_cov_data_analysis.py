# -*- coding: utf-8 -*-
"""
This Python file is been made by the project group Mattek5 C4-202

This is the analysis of the filter coefficients for autocorrelation method and
the covariance method.
"""
from __future__ import division
import os
import sys

lib_path = '\\Scripts\\libs'
data_path = '\\Lydfiler\\Sound'
cwd = os.getcwd()[:-8]
sys.path.insert(0, cwd + lib_path)
os.chdir(cwd + data_path)

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import LP_algorithms as lpa 

Voiced = True # If anything else than True it is unvoiced 
p = 12
N = 160
LenSim = N - p # 160 - 12
Antal_data_filer = 42

""" Import data files """
Names = np.array(["da_","fr_","jo_","ma_","to_","tr_"]);
if Voiced == True:
    Sound = np.array(["a","e","i","o","oe","u","y"])
    DataLocation = "Stemt"
    Person = 'da'
    sound = 'i'
else:
    Sound = np.array(["p","t","k","f","v","h","s"])
    DataLocation = "Ustemt"
    Person = 'ma'
    sound = 's'

os.chdir(DataLocation)

FileName = np.zeros(Antal_data_filer , dtype='|S5')
for i in range(len(Names)):
    for j in range(len(Sound)):
        FileName[j + (len(Sound))*i] = Names[i] + Sound[j]

dict = {FileName[i]: 1 for i in range(Antal_data_filer)}

for i in range(Antal_data_filer):
    if os.path.exists(FileName[i]+'.wav') == False:
        dict[FileName[i]] = 'File not there'
        del dict[FileName[i]]
    else:
        f, dict[FileName[i]] =  wav.read(FileName[i]+'.wav')

for key in dict:
    dict[key] = dict[key] / (np.max(np.abs(dict[key])))

""" Calculation errors for the two methods """
R = {key: lpa.AR_design_matrix(dict[key],N-p,p) for key in dict} # The design matrix

# The autocorrelation method 
r = {key: lpa.acf_fft(dict[key][:N]) for key in dict}        # The autocorrelation function
a_acf = {key: -lpa.lev_durb(r[key],p)[0]  for key in dict}   # The ACF method's a-coefficients
EstAcf = {key: np.dot(R[key], a_acf[key]) for key in dict}   # The estimate with the ACF method

ErrorAcf = {key: np.linalg.norm(EstAcf[key] - dict[key][p: p + len(EstAcf[key])] ,ord = 2) for key in dict} # The error of the acf estimate

# The covariance method 
Phi = {key: lpa.cross_corr_mat(dict[key],p) for key in dict} # Cross-correlation matrix
phi = {key: lpa.cross_corr_arr(dict[key],p) for key in dict} # Cross-correlation array
a_cov = {key: lpa.Kov(Phi[key],phi[key]) for key in dict}    # The covariance method's a-coefficients
EstCov = {key: np.dot(R[key], a_cov[key]) for key in dict}   # The estimate with the covariance method

ErrorCov = {key: np.linalg.norm(EstCov[key] - dict[key][p: p + len(EstCov[key])] ,ord = 2) for key in dict} # The error of the covariance method estimate

#  The average error between the two methods
Average_errorAcf = np.sum(ErrorAcf.values())/ len(ErrorAcf)  # Average error for the ACF coefficients
Average_errorCov = np.sum(ErrorCov.values())/ len(ErrorCov)  # Average error for the covariance coefficients
Difference = np.abs(Average_errorAcf - Average_errorCov)     # The difference

ADifference = {key: np.linalg.norm(a_cov[key] - a_acf[key] ,ord = 2) for key in dict} # The difference between the a-coefficients
AAverageDifference = np.sum(ADifference.values())/(len(ADifference))


""" Plots of the estimates """
os.chdir(cwd + "\Scripts") # Saving the figures in the right place
plotKey = Person + '_' + sound
time = np.linspace(0.0015,N/f,LenSim)

plt.figure(1)
plt.title("Estimat af signal med ACF-metoden for %s" % plotKey )
plt.plot(time,EstAcf[plotKey], label = r"Syntetiseret signal $\hats[n]$")
plt.plot(time,dict[plotKey][p:p+LenSim], label = r"Observeret signal $s[n]$")
plt.legend()
plt.xlabel('Tid [s]')
plt.ylabel('Amplitude')
plt.savefig("figures/ACF_%s.png" % plotKey, dpi = 500)
plt.show()

plt.figure(2)
plt.title("Estimat af signal med kovariansmetoden for %s" % plotKey)
plt.plot(time,EstCov[plotKey], label = r"Syntetiseret signal $\hats[n]$")
plt.plot(time,dict[plotKey][p:p+LenSim], label = r"Observeret signal $s[n]$")
plt.legend()
plt.xlabel('Tid [s]')
plt.ylabel('Amplitude')
plt.savefig("figures/COV_%s.png" % plotKey, dpi = 500)
plt.show()

""" Calculate the deviance of the estimates """
def deviance(est1, est2):
    d = 100*np.abs(est1 - est2)/np.max([est1,est2])
    return d, np.max([est1,est2]), np.min([est1,est2])

d, upper, lower = deviance(ErrorAcf[plotKey], ErrorCov[plotKey])
d_ave, upper_ave, lower_ave = deviance(Average_errorAcf, Average_errorCov)

print "File: %s.wav" % plotKey
print "Error of the estimated signal for the ACF method:", ErrorAcf[plotKey]
print "Error of the estimated signal for the covariance method:", ErrorCov[plotKey]
print "The average error for the ACF method:", Average_errorAcf
print "The average error for the covariance method:", Average_errorCov
print "The difference between the a-coefficients:", ADifference[plotKey]
print "The difference between the errors:", Difference
print "The average difference between the a-coefficients for the ACF and covariance method:",AAverageDifference
print "Percentual deviance between the errors: \
%.5f is %.5f %% lower than %.5f." %(lower,d,upper)
print "Percentual deviance between the average errors: \
%.5f is %.5f %% lower than %.5f." %(lower_ave,d_ave,upper_ave)

""" Plot the a-cofficients for the two methods """
plt.figure(3)
plt.title("a-koefficienterne for ACF- og kovariansmetoderne")
plt.stem(a_acf[plotKey], linefmt="r-", markerfmt="ro", basefmt="r-", label = "ACF")
plt.stem(a_cov[plotKey], label = "Kovarians")
plt.legend()
plt.savefig("figures/a-koefficienter_%s.png" % plotKey, dpi = 500)
plt.show()

""" Magnitude response for the filter coefficients """
N = 160
N2 = 80 # N/2 aka 160/2 aka 80
omega = np.linspace(0,np.pi, N+1)
freqaxis = np.linspace(0,np.pi,80)
ze = np.exp(-1j * omega)

MeasuredAmplitude = {key: np.abs(np.fft.fft(dict[plotKey][:N])) for key in dict}
AmplitudeResponseAcf = {key: (np.abs(1/(-np.polyval(np.append(np.flipud(a_acf[key]),-1),ze)))) for key in dict}

AmplitudeResponseCov = {key: (np.abs(1/(-np.polyval(np.append(np.flipud(a_cov[key]),-1),ze)))) for key in dict}

""" Normalization of the amplitudes """
for key in dict:
    AmplitudeResponseAcf[key] = AmplitudeResponseAcf[key]/np.max(AmplitudeResponseAcf[key])    
    AmplitudeResponseCov[key] = AmplitudeResponseCov[key]/np.max(AmplitudeResponseCov[key]) 
    MeasuredAmplitude[key] = MeasuredAmplitude[key]/np.max(MeasuredAmplitude[key])

plt.figure(4)
plt.title("Amplituderespons af koefficienterne for %s" % plotKey)
plt.plot(freqaxis,MeasuredAmplitude[plotKey][:N2], label = "Det observerede signal")
plt.plot(omega,AmplitudeResponseAcf[plotKey], label = "ACF")
plt.plot(omega,AmplitudeResponseCov[plotKey], "r-" , label = "Kovarians")
plt.legend()
if Voiced == True:
    plt.axis([0,1,-0.1,1.1])
else:
    plt.axis([0,np.pi,-0.1,1.1])
plt.xlabel(r"Frekvens, $\omega$")
plt.ylabel(r"Ampltituderespons, $G(\omega)$")
plt.savefig("figures/Amplituderespons_%s.png" % plotKey, dpi = 500)
plt.show()