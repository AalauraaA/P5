# -*- coding: utf-8 -*-
"""
This Python file is been made by the project group Mattek5 C4-202

This is the analysis of the model order p and lenght N.
"""
from __future__ import division
import os
import sys

lib_path = '\\Scripts\\libs'
cwd = os.getcwd()[:-8]
sys.path.insert(0, cwd + lib_path)

Stemt = 1
if Stemt == True:
    data_path = "\\Lydfiler\\Sound\\Stemt\\"
else:
    data_path = "\\Lydfiler\\Sound\\Ustemt\\"

import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
import voiced_unvoiced as vu
import LP_speech as lps
import windowfunctions as win

#==============================================================================
# Optimized p and N for voiced files
#==============================================================================
""" Import data """
def downsample(d, dc, filename, data_path):
    """
    Input:
        d: if equal to 1 downsampling will happen. Othervise, the data
        will only be imported.
        dc: downsamplingconstant, which the data will be downsampled by.
        filename: the filename of the data as a string.
    Returns:
        The (possibly downsampled) data in a numpy-array with dtype = float64.
    """
    if d == False:
        fs, data = wav.read(cwd + data_path + filename)
        data = np.array(data, dtype = "float64")
        return fs, data
    else:
        fullfs, fulldata = wav.read(cwd + data_path + filename)
        data = np.array([fulldata[i] for i in range(0,len(fulldata),dc)], dtype = "float64")
        fs = int(fullfs/dc)
        return fs, data

data = {}
for filename in os.listdir(cwd + data_path):
    f, data[filename] = downsample(0,1,filename, data_path)

N = int(0.02*f)
p_max = 101

""" Run the for loop if you have time enough """
E_list_total = np.zeros(p_max)
for key in data.keys(): # Run only if you have a lot of time
    M = len(lps.LP_parameters(data[key],N,5,.5,window_arg = win.Hamming))-3
    E_list = np.zeros((p_max,M))
    for p in range(1,p_max):
        if p % 10 == 0:
            print "First loop. p = %d, key: %s." % (p,key)
        parameters = lps.LP_parameters(data[key],N,p,.5,window_arg = win.Hamming)
        for i in range(M):
            E_list[p][i] = parameters[i]['gain']**2
        
    E_list = E_list/E_list[1]
    E_list_total += np.average(E_list,axis=1)
np.save("npy-files/E_list_total", E_list_total) # Save file for later use

""" Plot the error for p """
plt.figure(1)
plt.plot(E_list_total[1:])
plt.xlabel(r"$p$")
plt.ylabel(r"$\barE_p$")
plt.savefig("figures/E_p.png", dpi = 500)

Error_time = np.zeros(p_max)
Error_freq = np.zeros(p_max)

""" Run if you have enough time """
for p in range(1,p_max): # Run only if you have a lot of time
    if p % 10 == 0:
        print "Second loop. p = %d." % p
    for key in data.keys():
        Parameters = lps.LP_parameters(data[key],N,p,0.5,window_arg = win.Hamming)
        Prediction = lps.LP_predict(Parameters)
        
        amp_data = np.abs(np.fft.fft(data[key]))
        amp_pred = np.abs(np.fft.fft(Prediction))
        
        length = int(len(Prediction))
        Error_time[p] += np.linalg.norm(Prediction - data[key][:length])
        Error_freq[p] += np.linalg.norm(amp_pred[:int(length/2)] - amp_data[:int(length/2)])
np.save("npy-files/Error_time.npy", Error_time)
np.save("npy-files/Error_freq.npy", Error_freq)
np.save("npy-files/amp_data.npy", amp_data)
np.save("npy-files/amp_pred.npy", amp_pred)

""" Calculate the normalized amplitudes of the errors in time and frequency """
for p in range(1,p_max):
    Error_freq[p] /= len(data.keys())
    Error_time[p] /= len(data.keys())

amp_data_norm = amp_data/np.max([amp_data[:len(amp_pred)],amp_pred])
amp_pred_norm = amp_pred/np.max([amp_data[:len(amp_pred)],amp_pred])

""" Plot the errors for p """
plt.figure(2)
plt.plot(amp_data_norm[:int(len(amp_data)/2)], label = r"$|\mathcal{F}[s[n]](e^{j\omega)})|$")
plt.plot(amp_pred_norm[:int(len(amp_pred)/2)], label = r"$|\mathcal{F}[\hat{s}[n]](e^{j\omega)})|$")
plt.xlabel(r"$\omega$")
plt.title("Amplituderespons for de originale og praedikterede data")
plt.legend()
plt.savefig("figures/amp_data.png", dpi = 500)

plt.figure(3)
plt.plot(Error_time[1:95])
plt.xlabel(r"$p$")
plt.title("Gennemsnitlige fejl i tidsdomaenet")
plt.savefig("figures/resi_p_time.png", dpi = 500)

plt.figure(4)
plt.plot(Error_freq[1:95])
plt.xlabel(r"$p$")
plt.title("Gennemsnitlige fejl i frekvensdomaenet")
plt.savefig("figures/resi_p_freq.png", dpi = 500)

""" Finding errors for N """
Nlist = np.arange(50,400,10)
datF = {key: np.abs(np.fft.fft(data[key])) for key in data.keys()}
E_list_time = np.zeros(len(Nlist))
E_list_freq = np.zeros(len(Nlist))

p = 39 # Optimized p 

""" Run if you have enough time """
for i in range(len(Nlist)):
    if Nlist[i] % 10 == 0:
        print "Third loop. N = %d." % Nlist[i]
    for key in data.keys():
        N = Nlist[i]
        p = 40
        parameters = lps.LP_parameters(data[key],N,p,.5,window_arg = win.Rectangular)
        prediction = lps.LP_predict(parameters)
        M = len(prediction)
        M2 = int(M/2.)
        predictF = np.abs(np.fft.fft(prediction))
        E_list_time[i] += np.linalg.norm(prediction-data[key][:M])
        E_list_freq[i] += np.linalg.norm(predictF[:M2]-datF[key][:M2])

for i in range(1,len(Nlist)):
    E_list_freq[i] /= len(data.keys())
    E_list_time[i] /= len(data.keys())

np.save("npy-files/E_list_time.npy", E_list_time)
np.save("npy-files/E_list_freq.npy", E_list_freq)

""" Plots of the errors for N in time and frequency """
plt.figure(5)
plt.title("Gennemsnitlige fejl i tidsdomaenet")
plt.xlabel(r"$N$")
plt.plot(Nlist[1:],E_list_time[1:])
plt.savefig("figures/resi_N_time.png", dpi = 500)

plt.figure(6)
plt.title("Gennemsnitlige fejl i frekvensdomaenet")
plt.xlabel(r"$N$")
plt.plot(Nlist[1:],E_list_freq[1:])
plt.savefig("figures/resi_N_freq.png", dpi = 500)

#==============================================================================
# Optimized p and N for all voiced, unvoiced and sentence files
#==============================================================================
""" Setting path up """
data_path_stemt = "\\Lydfiler\\Sound\\Stemt\\"
data_path_ustemt = "\\Lydfiler\\Sound\\Ustemt\\"
data_path_blandet = "\\Lydfiler\\Sound\\Saetning\\"

""" Data import - manual choose between voiced, unvoiced and sentences """
dat = {} # Voiced 
for filename in os.listdir(cwd + data_path_stemt):
    f, dat[filename] = downsample(0,1,filename,data_path_stemt)

#dat = {} # Unvoiced 
#for filename in os.listdir(cwd + data_path_ustemt):
#    f, dat[filename] = downsample(0,1,filename,data_path_ustemt)

#dat = {} # Sentence (Mixed)
#for filename in os.listdir(cwd + data_path_blandet):
#    if os.path.isdir(cwd + data_path_blandet + filename) == False:
#        f, dat[filename] = downsample(1,2,filename,data_path_blandet)

""" The found p values """
msek = 0.01 # Number of milliseconds
N = int(msek*f)
p1 = 12
p2 = 39

""" Run if you have enough time, else load the npy files """
#Error_time = np.zeros(2)
#Error_freq = np.zeros(2)
#i = 0
#Parameters = {}
#Prediction = {}
#amp_data = {}
#amp_pred = {}
#for p in [p1,p2]:
#    for key in dat.keys():
#        if len(dat[key]) >= 2*N:
#            print "Starting %s." % key
#            Parameters = lps.LP_parameters(dat[key],N,p,0.5)
#            Prediction = lps.LP_predict(Parameters)
#            print "Prediction done."
#            amp_data = np.abs(np.fft.fft(dat[key]))
#            amp_pred = np.abs(np.fft.fft(Prediction))
#            print "FFT done."
#            length = int(len(Prediction))
#            Error_time[i] += np.linalg.norm(Prediction - dat[key][:length])/len(Prediction)
#            Error_freq[i] += np.linalg.norm(amp_pred[:int(length/2)] - amp_dat[:int(length/2)])/len(amp_pred)
#    i += 1
#
#for p in range(2):
#    Error_freq[p] /= len(data.keys())
#    Error_time[p] /= len(data.keys())
#
#""" Save the errors """
#os.chdir(cwd + "\\Scripts\\npy-files") # The right path to save the npy files
#savedir = os.getcwd()
#np.save("Error_time_%d_mix" % N, Error_time_gen)
#np.save("Error_freq_%d_mix" % N, Error_freq_gen)

""" Calculate the error in time and frequency """
os.chdir(cwd + "\\Scripts\\npy-files\\Time_errors_gen")
loaddir1 = os.getcwd()
Time_errors = {}
for filename in os.listdir(loaddir1):
    Time_errors[filename] = np.load(filename)

os.chdir(cwd + "\\Scripts\\npy-files\\Freq_errors_gen")
loaddir2 = os.getcwd()
Freq_errors = {}
for filename in os.listdir(loaddir2):
    Freq_errors[filename] = np.load(filename)

""" Normalized the errors in time and frequency """
normal_time = Time_errors['Error_time_160_voi.npy'][0]
normal_freq = Freq_errors['Error_freq_160_voi.npy'][0]

""" Caluclate the errors """
for key in Time_errors.keys():
    Time_errors[key] /= normal_time

for key in Freq_errors.keys():
    Freq_errors[key] /= normal_freq