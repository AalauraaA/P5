# -*- coding: utf-8 -*-
"""
This Python file is been made by the project group Mattek5 C4-202

This is a analysis and model check of the model for speech signal. There will
seen on all voiced, all unvoiced signal and all sentences files.
"""
from __future__ import division
import os
import sys

lib_path = '\\Scripts\\libs'
data_path = '\\Lydfiler\\Sound'
cwd = os.getcwd()[:-8]
sys.path.insert(0, cwd + lib_path)

import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
import LP_speech as lps
import scipy.signal as sig
import modelcheck as mc

N = 160
p = 12

# =============================================================================
# Resiudals for voiced  
# =============================================================================
""" Voiced data import and prediction of the data """
filenames = os.listdir(cwd + data_path + "/Stemt")
data = {}
parameters = {}
predict = {}
np.random.seed(1)
cut = 50
vlist = [41,3,10,13,22,34] # Choose some voiced files
for i in vlist:
    data[i] = np.array(wav.read(cwd + data_path + "/Stemt/" \
        + filenames[i])[1],dtype= "float64")
    parameters[i] = lps.LP_parameters(data[i],N,p,0)
    predict[i] = lps.LP_predict(parameters[i])

""" Calculate the residuals for voiced """
R_total = {}
r_st_total = {}
for i in vlist:
    r_st = np.zeros(len(predict[i])-N)
    k = 1
    for j in range(1,len(parameters[i])-3):
        delay = j*N
        gain = parameters[i][j]['gain']
        voi = parameters[i][j]['voi']
        if voi != 0 and not np.all(data[i][delay:delay+N]) == 0: 
            r_st[k*N-N:k*N], var2 = mc.res_student(data[i],predict[i],delay,N,p,gain,voi) # Don't use var2 in this script
            k += 1
    r_st = np.trim_zeros(r_st)
    r_st_total[i] = r_st
    M = len(r_st)
    R = mc.acf_np(r_st)
    win2 = sig.triang(2*M+1)[M:-1]
    R /= win2
    R /= M
    R_total[i] = R
 
""" Plots of residuals """
plt.figure(1)
plt.subplot(311)
plt.title("Autokorrelation for residualer ved stemte lyde",fontsize = 15)
plt.plot(R_total[41][:N],"b.",label = "tr_y")
plt.xticks([])
plt.ylabel(r"$r\check_{r^{st}}[k]$",fontsize = 15)
plt.legend(loc= "upper right", fontsize = 15)
plt.subplot(312)
plt.ylabel(r"$r\check_{r^{st}}[k]$",fontsize = 15)
plt.plot(R_total[3][:N],"b.",label = "da_o")
plt.legend(loc= "upper right", fontsize = 15)
plt.xticks([])
plt.subplot(313)
plt.ylabel(r"$r\check_{r^{st}}[k]$",fontsize = 15)
plt.plot(R_total[10][:N],"b.",label = "fr_o")
plt.xticks([])
plt.legend(loc= "upper right", fontsize = 15)
#plt.savefig("figures/acfmany_voiced1.png",dpi = 500)
plt.show()

plt.figure(2)
plt.subplot(311)
plt.ylabel(r"$r\check_{r^{st}}[k]$",fontsize = 15)
plt.plot(R_total[13][:N],"b.",label = "fr_y")
plt.xticks([])
plt.legend(loc= "upper right", fontsize = 15)
plt.subplot(312)
plt.ylabel(r"$r\check_{r^{st}}[k]$",fontsize = 15)
plt.plot(R_total[22][:N],"b.",label = "ma_e")
plt.legend(loc= "upper right", fontsize = 15)
plt.xticks([])
plt.subplot(313)
plt.ylabel(r"$r\check_{r^{st}}[k]$",fontsize = 15)
plt.plot(R_total[34][:N],"b.",label = "to_y")
plt.xlabel(r"$n$",fontsize = 15)
plt.legend(loc= "upper right", fontsize = 15)
#plt.savefig("figures/acfmany_voiced2.png",dpi = 500)
plt.show()

""" The average energy for residuals of voiced """
u = 40
Energy = mc.avg_energy(r_st_total[41],u)/4
E_avg = np.average(Energy)
plt.figure(3)
plt.title("Residualer for stemt lyd",fontsize = 15)
plt.plot(r_st_total[41])
plt.plot(Energy,"r-",label = r"$\pm\alpha\sqrt{\bar{E}_p}$")
plt.plot(-Energy,"r-")
plt.plot([0,len(r_st_total[41])],[E_avg,E_avg],"g-")
plt.plot([0,len(r_st_total[41])],[-E_avg,-E_avg],"g-")
plt.xlabel(r"$n$",fontsize = 15)
plt.ylabel(r"$r^{st}[n]$",fontsize = 15)
plt.legend(loc = "upper right",fontsize = 15)
#plt.savefig("figures/residualvarians_stemt.png",dpi = 500)
plt.show()

# =============================================================================
# Residuals for unvoiced
# =============================================================================
""" Import data and predict data """
filenames = os.listdir(cwd + data_path + "/Ustemt")
data = {}
parameters = {}
predict = {}
np.random.seed(1)
cut = 50
ulist = [5,6,12,20,22,29] # Choose some unvoiced files
for i in ulist:
    temp =  np.array(wav.read(cwd + data_path + "/Ustemt/" \
        + filenames[i])[1],dtype= "float64")
    if len(temp) >= 2*N:
        data[i] = temp
        parameters[i] = lps.LP_parameters(data[i],N,p,0)
        predict[i] = np.trim_zeros(lps.LP_predict(parameters[i]))

""" Calculate the residuals for unvoiced """
R_total1 = {}
for i in ulist:
    r_st = np.zeros(len(predict[i])-N)
    k = 1
    for j in range(1,len(parameters[i])-3):
        delay = j*N
        gain = parameters[i][j]['gain']
        voi = parameters[i][j]['voi'] 
        if voi != 0 and not np.all(data[i][delay:delay+N]) == 0: 
            if delay + N < len(predict[i]):
                r_st[k*N-N:k*N], var2 = mc.res_student(data[i],predict[i],delay,N,p,gain,voi) # var2 is not used in this script
                k += 1
    if not np.all(r_st == 0):
        r_st = np.trim_zeros(r_st)
        r_st_total[i] = r_st
        M = len(r_st)
        R = mc.acf_np(r_st)
        win2 = sig.triang(2*M+1)[M:-1]
        R /= win2
        R /= M
        R_total[i] = R

" Plots of residuals """      
plt.figure(4)
plt.subplot(311)
plt.title("Autokorrelation for residualer ved ustemte lyde",fontsize = 15)
plt.plot(R_total[5][:N],"b.",label = "fr_f")
plt.xticks([])
plt.legend(loc= "upper right", fontsize = 15)
plt.subplot(312)
plt.plot(R_total[6][:N],"b.",label = "fr_h")
plt.legend(loc= "upper right", fontsize = 15)
plt.xticks([])
plt.subplot(313)
plt.plot(R_total[12][:N],"b.",label = "jo_k")
plt.xticks([])
plt.legend(loc= "upper right", fontsize = 15)
#plt.savefig("figures/acfmany_unvoiced1.png",dpi = 500)
plt.show()

plt.figure(5)
plt.subplot(311)
plt.plot(R_total[20][:N],"b.",label = "ma_p")
plt.xticks([])
plt.legend(loc= "upper right", fontsize = 15)
plt.subplot(312)
plt.plot(R_total[22][:N],"b.",label = "ma_t")
plt.legend(loc= "upper right", fontsize = 15)
plt.xticks([])
plt.subplot(313)
plt.plot(R_total[29][:N],"b.",label = "to_t")
plt.xlabel(r"$n$",fontsize = 15)
#plt.legend(loc= "upper right", fontsize = 15)
plt.savefig("figures/acfmany_unvoiced2.png",dpi = 500)
plt.show()

""" The average energy for residuals of unvoiced """
Energy = mc.avg_energy(r_st_total[12],u)/4
E_avg = np.average(Energy)
plt.figure(6)
plt.title("Residualer for ustemt lyd",fontsize = 15)
plt.plot(r_st_total[12])
plt.plot(Energy,"r-",label = r"$\pm\alpha\sqrt{\bar{E}_p}$")
plt.plot(-Energy,"r-")
plt.plot([0,len(r_st_total[12])],[E_avg,E_avg],"g-")
plt.plot([0,len(r_st_total[12])],[-E_avg,-E_avg],"g-")
plt.xlabel(r"$n$",fontsize = 15)
plt.ylabel(r"$r^{st}[n]$",fontsize = 15)
plt.legend(fontsize = 15)
plt.legend(loc = "upper right",fontsize = 15)
#plt.savefig("figures/residualvarians_ustemt.png",dpi = 500)
plt.show()

# =============================================================================
# Residuals for sentences
# =============================================================================
""" Import data and predict data """
filenames = os.listdir(cwd + data_path + "/Saetning")
filenames.pop(2)
data = {}
parameters = {}
predict = {}
np.random.seed(1)
cut = 50
for i in range(len(filenames)):
    data[i] = np.array(wav.read(cwd + data_path + "/Saetning/" \
        + filenames[i])[1],dtype= "float64")
    parameters[i] = lps.LP_parameters(data[i],N,p,0)
    predict[i] = lps.LP_predict(parameters[i])

""" Calculate residuals for sentences """
R_total = {}
r_st_total = {}
for i in parameters.keys():
    r_st = np.zeros(len(predict[i])-N)
    k = 1
    for j in range(1,len(parameters[i])-3):
        delay = j*N
        gain = parameters[i][j]['gain']
        voi = parameters[i][j]['voi']
        if voi != 0 and not np.all(data[i][delay:delay+N]) == 0: 
            r_st[k*N-N:k*N], var2 = mc.res_student(data[i],predict[i],delay,N,p,gain,voi) # Var2 is not used in this script
            k += 1
    r_st = np.trim_zeros(r_st)
    r_st_total[i] = r_st
    M = len(r_st)
    R = mc.acf_np(r_st)
    window = sig.triang(2*M+1)[M:-1]
    R /= window
    R /= M
    R_total[i] = R

""" Plot of residuals """
plt.figure(7)
plt.subplot(311)
plt.title("Autokorrelation for residualer ved saetninger",fontsize = 15)
plt.plot(R_total[3][:N],"b.",label = "Danny_saet")
plt.xticks([])
plt.legend(loc= "upper right", fontsize = 15)
plt.subplot(312)
plt.plot(R_total[2][:N],"b.",label = "Jonas_saet")
plt.legend(loc= "upper right", fontsize = 15)
plt.xticks([])
plt.subplot(313)
plt.plot(R_total[1][:N],"b.",label = "Laura_saet")
plt.legend(loc= "upper right", fontsize = 15)
plt.xticks([])
#plt.savefig("figures/acfmany_sentence1.png",dpi = 500)
plt.show()

plt.figure(8)
plt.subplot(211)
plt.plot(R_total[4][:N],"b.",label = "sentence")
plt.legend(loc= "upper right", fontsize = 15)
plt.xticks([])
plt.subplot(212)
plt.plot(R_total[0][:N],"b.",label = "tobias_saet")
plt.xlabel(r"$n$",fontsize = 15)
plt.legend(loc= "upper right", fontsize = 15)
#plt.savefig("figures/acfmany_sentence2.png",dpi = 500)
plt.show()

""" The average energy for residuals of sentences """
u = 100
Energy = mc.avg_energy(r_st_total[3],u)/4
E_avg = np.average(Energy)
plt.figure(9)
plt.title("Residualer for saetninger",fontsize = 15)
plt.plot(r_st_total[3])
plt.plot(Energy,"r-",label = r"$\pm \alpha\sqrt{\bar{E}_p}$")
plt.plot(-Energy,"r-")
plt.plot([0,len(r_st_total[3])],[E_avg,E_avg],"g-")
plt.plot([0,len(r_st_total[3])],[-E_avg,-E_avg],"g-")
plt.xlabel(r"$n$",fontsize = 15)
plt.ylabel(r"$r^{st}[n]$",fontsize = 15)
plt.legend(loc = "upper right",fontsize = 15)
#plt.savefig("figures/residualvarians_saetning.png",dpi = 500)
plt.show()
