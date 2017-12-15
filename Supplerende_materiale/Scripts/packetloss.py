# -*- coding: utf-8 -*-
"""
This Python file is been made by the project group Mattek5 C4-202

This is a test of how much packetloss the prediction of a sound file 
can have and still be intelligibly
"""
from __future__ import division
import os
import sys

lib_path = '\\Scripts\\libs'
data_path = '\\Lydfiler\\Sound'
export_path = '\\Lydfiler\Predict'
cwd = os.getcwd()[:-8]
sys.path.insert(0, cwd + lib_path)

import scipy.io.wavfile as wav
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import LP_speech as lps
import scipy.signal as sig

""" Import data """
filename = 'Laura_en_saet'
fs, data= wav.read(cwd + data_path + "/Saetning/" + filename + ".wav")
data = np.array(data,dtype = "float64")

""" Function for packetloss """
def packetlooser(paramters,P):
    count = 0
    for packet in range(len(parameters)-3):
        if np.random.random() <= P:
            count += 1
            parameters[packet] = \
            {"coef":None,"gain":None,"voi":4,"pitch":None,"first_imp":None}
    print("Number of packet losses: %d" %count)
    print("Packet losses precent  : %.1f %%" %((100*count)/(len(parameters)-3)))
    return paramters

""" Predict signal with packetloss """
N = 160
p = 12

P_packetloss = .9 # The probability of packet loss

parameters = lps.LP_parameters(data, N, p, .5)
parameters_lossy = packetlooser(parameters, P_packetloss)
predict = lps.LP_predict(parameters_lossy)

""" Plot of data and predicted data """
plt.subplot(211)
plt.plot(data)
plt.subplot(212)
plt.plot(predict)
plt.show()

""" Save and play the predict packetloss file """
#wav.write(cwd + export_path + "/Packetloss/packetloss_90_" + filename + ".wav",\
#          fs,np.int16(predict))

#sd.play(np.int16(data),fs)
#sd.play(np.int16(predict),fs)
