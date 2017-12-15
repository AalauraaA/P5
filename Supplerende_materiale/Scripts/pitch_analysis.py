# -*- coding: utf-8 -*-
"""
This Python file is been made by the project group Mattek5 C4-202

This is the analysis of the pitch estimation for two voiced signals
"""
from __future__ import division
import os
import sys
lib_path = '\\Scripts\\libs'
cwd = os.getcwd()[:-8]
sys.path.insert(0, cwd + lib_path)

import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
import LP_speech as lps

#==============================================================================
# Two voiced files
#==============================================================================
""" Data import """
# Signal 1
filename = 'fr_y'
fs, data= wav.read(cwd + "/Lydfiler/Sound/Stemt/" + filename + ".wav")
data = np.array(data,dtype = "float64")

# Signal 2
filename2 = 'da_a'
fs2, data2= wav.read(cwd + "/Lydfiler/Sound/Stemt/" + filename2 + ".wav")
data2 = np.array(data2,dtype = "float64")

N = 160
p = 12
K = 15 # How big the delay is

delay = 160*K

""" Signal 1 """
# Getting parameters and prediction
parameters = lps.LP_parameters(data, N, p, 0)
predict = lps.LP_predict(parameters)

parameters2 = lps.LP_parameters(predict, N, p, 0)
predict2 = lps.LP_predict(parameters2)

# Slice the signal with delay 160*15
p_slice = predict[delay : delay + N]
p2_slice = predict2[delay : delay + N]

# Plot
plt.figure(1)
plt.plot(p_slice, label = "Testsignal")
plt.plot(p2_slice, label = "Praedikteret testsignal")
plt.xlabel(r"$n$")
plt.title("Sammenligning - eksempel 1")
plt.legend()
#plt.savefig("figures/test1.png",dpi = 500)
plt.show()

# Calculated the difference
p_parameter = parameters[K]['pitch']
p_parameter2 = parameters2[K]['pitch']

i_parameter = parameters[K]['first_imp']
i_parameter2 = parameters2[K]['first_imp']

print("tau* observed: %.2f" %p_parameter)
print("tau* predict: %.2f" %p_parameter2)
print("tau0 observed: %.2f" %i_parameter)
print("tau0 predict: %.2f" %i_parameter2)
print("Difference between tau*: %.2f " %abs(p_parameter2 - p_parameter))
print("Difference between tau0: %.2f " %abs(i_parameter2 - i_parameter))

""" Signal 2 """
# Getting parameters and prediction
para = lps.LP_parameters(data2, N, p, 0)
pred = lps.LP_predict(para)

para2 = lps.LP_parameters(pred, N, p, 0)
pred2 = lps.LP_predict(para2)

# Slice the signal with delay = 160
p_slice2 = pred[delay : delay + N]
p2_slice2 = pred2[delay : delay + N]

# Plot
plt.figure(2)
plt.plot(p_slice2, label = "Testsignal")
plt.plot(p2_slice2, label = "Praedikteret testsignal")
plt.xlabel(r"$n$")
plt.title("Sammenligning - eksempel 2")
plt.legend()
#plt.savefig("figures/test2.png",dpi = 500)
plt.show()

# Calculate the difference
p_para = para[K]['pitch'] 
p_para2 = para2[K]['pitch']

i_para = para[K]['first_imp']
i_para2 = para2[K]['first_imp']

print("tau* observed: %.2f" %p_para)
print("tau* predict: %.2f" %p_para2)
print("tau0 observed: %.2f" %i_para)
print("tau0 predict: %.2f" %i_para2)
print(r"Difference between tau*: %.2f " %abs(p_para2 - p_para))
print(r"Difference between tau0: %.2f " %abs(i_para2 - i_para))

#==============================================================================
# All voiced files
#==============================================================================
""" Data import """
dat = {}
for filename in os.listdir(cwd + "/Lydfiler/Sound/Stemt/"):
    dat[filename] = np.array(wav.read(cwd + "/Lydfiler/Sound/Stemt/" \
        + filename)[1],dtype = "float64")

N = 160
p = 12

""" Finding difference for tau* and tau0 """
M = len(dat.keys())
snip = M
All_para = {}
All_pre = {}
All_pre_para = {}

for key in dat.keys()[:snip]:
    All_para[key] = lps.LP_parameters(dat[key],N,p,0)
    All_pre[key] = lps.LP_predict(All_para[key])
    All_pre_para[key] = lps.LP_parameters(All_pre[key],N,p,0)

""" Calculate errors """
pitch_err = np.zeros(M)
imp_err = np.zeros(M)
m = 0
for key in dat.keys()[:snip]:
    D = len(All_para[key].keys())-3
    for nestedkey in range(D):
        pitch_err[m] += abs(All_para[key][nestedkey]['pitch'] \
                        - All_pre_para[key][nestedkey]['pitch'])
        imp_err[m] +=  abs(All_para[key][nestedkey]['first_imp'] \
                        - All_pre_para[key][nestedkey]['first_imp'])
    pitch_err[m] /= D
    imp_err[m] /= D
    m += 1

print("Imp err, avg: %.2f and standard deviation: %.2f" %(np.average(imp_err),\
                                                np.sqrt(np.var(imp_err))))
print("")
print("Pitch err, avg: %.2f and standard deviation: %.2f" %(np.average(pitch_err),\
                                                  np.sqrt(np.var(pitch_err))))
