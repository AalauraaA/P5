# -*- coding: utf-8 -*-
"""
This Python file is been made by the project group Mattek5 C4-202

This is the analysis of the windows and overlaps which been used in the prediction
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

lib_path = '\\Scripts\\libs'
data_path = '\\Lydfiler\\Sound'
cwd = os.getcwd()[:-8]
sys.path.insert(0, cwd + lib_path)

import scipy.io.wavfile as wav
import LP_speech as lps
import windowfunctions as win

N = 160
p = 12

""" Data import """
filename = 'Tobias_saet'
#filename = 'Laura_saet'
#filename = 'Jonas_saet'
#filename = 'Danny_saet'
fs, data = wav.read(cwd + "/Lydfiler/Sound/Saeting" + filename + ".wav")
data = np.array(data,dtype = "float64")

""" Difference between windows and overlaps"""
overlaplist = np.linspace(0,0.9,10) # Generate a list of overlap from 0 to 0.9
window1 = win.Hamming
window2 = win.Blackman
window3 = win.Rectangular

# Calculate the errors
Error = np.zeros(len(overlaplist))
Error2 = np.zeros(len(overlaplist))
Error3 = np.zeros(len(overlaplist))
for i in range(len(overlaplist)):
    parameters1 = lps.LP_parameters(data, N, p, overlaplist[i], window1,fastarg = True)
    print overlaplist[i], 1
    parameters2 = lps.LP_parameters(data, N, p, overlaplist[i], window2,fastarg = True)
    print overlaplist[i], 2
    parameters3 = lps.LP_parameters(data, N, p, overlaplist[i], window3,fastarg = True)
    print overlaplist[i], 3
    predict1 = lps.LP_predict(parameters1)
    predict2 = lps.LP_predict(parameters2)
    predict3 = lps.LP_predict(parameters3)
    Error[i] = np.linalg.norm(data[:len(predict1)] - predict1)
    Error2[i] = np.linalg.norm(data[:len(predict2)] - predict2)
    Error3[i] = np.linalg.norm(data[:len(predict3)] - predict3)

E_max = max((Error[0],Error2[0],Error3[0])) # Normalize the errors with all the calculated maximum value
Error /= E_max
Error2 /= E_max
Error3 /= E_max

""" Plots of errors for the windows and overlaps """
plt.figure(1)
plt.subplot(211)
plt.plot(overlaplist, Error, label='Hamming')
plt.plot(overlaplist, Error2,label='Blackman')
plt.plot(overlaplist, Error3,label='Rektangulaert')
plt.xticks([])
plt.ylabel("Normaliseret fejl")
plt.title("Normaliseret fejl ved vinduer")
plt.legend()
#plt.savefig("figures/Win_overlap1.png", dpi = 500)
plt.show()
