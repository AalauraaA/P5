# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 08:59:14 2017

@author: Tobias
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

""" Data import """
filename = 'Tobias_saet' # Choose between voiced, unvoiced and sentences
fs, data = wav.read(cwd + data_path + "/Saetning/" + filename + ".wav")
data = np.array(data,dtype = "float64")

N = 160
p = 12

""" Predict data """
parameters = lps.LP_parameters(data, N, p, .5)

predict = lps.LP_predict(parameters)

""" Plot data and predict data """
plt.subplot(211)
plt.plot(data)

plt.subplot(212)
plt.plot(predict)
plt.show()

""" Play the sound and save predict """
#sd.play(np.int16(predict),fs)
#sd.play(np.int16(data),fs)

#wav.write(cwd + export_path + "/Praedikteret/pre_" + filename + ".wav",\
#          fs,np.int16(predict))
