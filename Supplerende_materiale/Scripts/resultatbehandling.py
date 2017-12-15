# -*- coding: utf-8 -*-
"""
This Python file is been made by the project group Mattek5 C4-202

This is the a comparison of a original sound signal and the predict sound signal
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
import P4 as spectrogram

""" Data import """
filename = 'Tobias_saet'
#filename = 'Laura_saet'
#filename = 'Sentences'

fs1, originalsignal = wav.read(cwd + data_path + "/Saetning/" + filename + ".wav")
fs2, predictedsignal = wav.read(cwd + export_path  + "/pre_" + filename +".wav")

if fs1 == fs2:
    Samefreq = True
else:
    Samefreq = False

""" Normalized signals """
normo = originalsignal / np.max(originalsignal)
normp = predictedsignal / (np.max(predictedsignal))

""" The difference in the energy for the two signals """
energydifference = np.sum(normo**2) - np.sum(normp**2)
energydiffnorm = energydifference/max((np.sum(normo**2),np.sum(normp**2)))

# If energydifference is negative the predicted signal has more energy than the original

print('The difference in the energy for the two normalised signals is %f.' % energydifference)
print('The difference in the energy for the two normalised signals divided by the maximum energy is %f.' % energydiffnorm)

""" The L2 norm of the two normalised signals """
timecomparison = np.linalg.norm((normo[:len(normp)] - normp), ord = 2)
print('The difference in L2-norm is %f' % timecomparison)

""" The difference in L2 norm of the two normalised signals in frequency domain """
fftoriginalsignal = np.fft.fft(originalsignal)#/np.max(np.fft.fft(originalsignal))
fftpredictedsignal = np.fft.fft(predictedsignal)#/np.max(np.fft.fft(predictedsignal))

fftoriginalsignal[0] = 0
fftpredictedsignal[0] = 0

""" Normalized amplitude """
fftnormo = fftoriginalsignal/np.max(np.abs(fftoriginalsignal))
fftnormp = fftpredictedsignal/np.max(np.abs(fftpredictedsignal))

""" The difference in L2 norm of the normalized amplitude """
ampdif = np.linalg.norm((np.abs(fftnormo)[:len(np.abs(fftnormp))] - np.abs(fftnormp)), ord = 2)
freqaxis = np.linspace(0,np.pi, len(fftnormo))

print('The difference in L2-norm for the two signals\' magnitude responce is %f.' % ampdif)
print('\n')

""" The spectograms of the signals """
STFTnormo, timebins1, freqbins1 = spectrogram.spectrogram(normo, 80, 160, window = spectrogram.Hann, fs = fs1)
STFTnormp, timebins2, freqbins2 = spectrogram.spectrogram(normp, 80, 160, window = spectrogram.Hann, fs = fs2)

spekdif = np.zeros(len(STFTnormp.T))
for i in range(len(spekdif)):
    spekdif[i] = np.linalg.norm((STFTnormo.T[i],STFTnormp.T[i]), ord = 2)
averagespekdif = np.sum(spekdif)/len(spekdif)
print('The difference in L2-norm for time freq with window 160 and 50 percent overlap is %f' % averagespekdif)

""" Plots """
plt.figure(1)
plt.specgram(originalsignal, Fs = fs1, NFFT = 160, noverlap = 80, cmap=plt.cm.terrain, scale = 'dB')
plt.title('Spektogram for det originale signal')
plt.xlabel('Tid [s]')
plt.ylabel('Frekvens [Hz]')
plt.colorbar()
plt.savefig("figures/spekgram_original_signal.png", dpi = 500)
plt.show()

plt.figure(2)
plt.specgram(predictedsignal, Fs = fs2, NFFT = 160, noverlap = 80, cmap=plt.cm.terrain, scale = 'dB')
plt.title('Spektogram for det praedikterede signal')
plt.xlabel('Tid [s]')
plt.ylabel('Frekvens [Hz]')
plt.colorbar()
plt.savefig("figures/spekgram_praedikteret_signal.png", dpi = 500)
plt.show()

plt.figure(3)
plt.plot(freqaxis[:int(len(freqaxis)/2)],np.abs(fftnormo[:int(len(freqaxis)/2)]), label = 'Originalt signal')
plt.plot(freqaxis[:int(len(freqaxis)/2)],np.abs(fftnormp[:int(len(freqaxis)/2)]), label = 'Praedikteret signal')
plt.xlabel('Frekvens [rad/s]')
plt.ylabel('Amplitude')
plt.title('Amplituderespons')
plt.legend()
plt.savefig("figures/amp_data_pred_signal.png", dpi = 500)
plt.show()

""" Play the sounds """
#sd.play(np.int16(originalsignal),fs1)
#sd.play(np.int16(predictedsignal),fs2)