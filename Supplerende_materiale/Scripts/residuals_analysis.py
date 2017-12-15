# -*- coding: utf-8 -*-
"""
This Python file is been made by the project group Mattek5 C4-202

This is a analysis and model check of the model for speech signal. There will
seen on one voiced and one unvoiced signal.
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
import scipy.stats as st
import scipy.signal as sig
import modelcheck as mc

""" Import data """
filename = 'ma_o'
filename2 = "jo_h"
fs,data= wav.read(cwd + data_path + "/Stemt/" + filename + ".wav")
fs2,data2= wav.read(cwd + data_path + "/Ustemt/" + filename2 + ".wav")
data = np.array(data,dtype = "float64")
data2 = np.array(data2,dtype = "float64")

#==============================================================================
# Prediction of voiced data 
#==============================================================================
N = 160
N2 = int(N/2)
p = 12
np.random.seed(1)

parameters = lps.LP_parameters(data,N,p,0)
predict = lps.LP_predict(parameters)    # Voiced 

scale = np.linalg.norm(data)/np.linalg.norm(predict)
#predict = predict*scale
print("Energy in data is    %.2f" %np.linalg.norm(data)**2)
print("Energy in predict is %.2f" %np.linalg.norm(predict)**2)

""" Plot for voiced with predict as hat matrix """
# sec = 4 for to_a is very good
# sec = 6 for tr_i is very cleer example
sec = 10
delay = sec*N

plt.figure(1)
plt.title("Stemt talesignal")
plt.plot(data[delay:delay+N],label = r"$s[n]$")
plt.plot(predict[delay:delay+N],label = r"$\hat{s}[n]$")
plt.xlabel(r"$n$")
plt.legend(fontsize = 13,loc = "lower right")
#plt.savefig("figures/compare_voiced.png",dpi = 500)
plt.show()

""" Compare the distributions for voiced"""
Slice = predict[delay:delay+N]
gain = parameters[sec]['gain']
voi = parameters[sec]['voi']
r_st,var2 = mc.res_student(data,predict,delay,N,p,gain,voi)
var = np.var(r_st)
mean = np.average(r_st)
xes = np.linspace(r_st.min(),r_st.max(),N)
norm = mc.normal_pdf(xes,mean,var)

histf, histbins = np.histogram(r_st,density = True,bins = 15)
hist_midddle = mc.middlehist(histbins)
norm_hist = mc.normal_pdf(hist_midddle,mean,var)
diff = histf - norm_hist
dist_avg = np.average(np.abs(diff))
dist_stand = np.sqrt(np.var(diff))

# Plots
plt.figure(2)
plt.hist(r_st, bins = 15,normed = True)
plt.plot(xes,norm)
plt.plot(mean,mc.normal_pdf(mean,mean,var),"ro")
plt.text(mean,mc.normal_pdf(mean,mean,var),"%.2e" %mean)
plt.xlabel("Residualer")
plt.ylabel("Normaliseret frekvens")
plt.title("Fordeling af residualer for stemt lyd")
#plt.savefig("figures/voiced_distribuion.png",dpi = 500)
plt.show()

plt.figure(3)
a = st.probplot(r_st, dist = "norm",plot = plt)[1][2]
plt.title("QQ-plot for stemt lyd. " + r"$R^2$" + "= %.3f." %(a))
plt.xlabel("Teoretiske kvantiler")
plt.ylabel("Observerede kvantiler")
#plt.savefig("figures/QQstemt.png",dpi = 500)
plt.show()

""" Plots for the residuals for voiced """
plt.figure(4)
plt.title("Residualplot for stemt lyd")
plt.scatter(range(N),r_st)
plt.xlabel(r"$n$",fontsize = 14)
plt.ylabel(r"$r^{st}[n]$",fontsize = 15)
#plt.savefig("figures/residualplotstemt.png",dpi = 500)
plt.show()

freq = np.abs(np.fft.fft(r_st))[:N2]
f_max = freq.argmax()/N             # Dominant frequency
T = int(1/f_max)                    # Period

plt.figure(5)
r_st_noperiod = mc.difference(r_st,T)
plt.title("Residualplot for stemt lyd - Differens")
plt.scatter(range(N-T),r_st_noperiod)
plt.xlabel(r"$n$",fontsize = 14)
plt.ylabel(r"$r^{st}[n]$",fontsize = 15)
#plt.savefig("figures/residualplotstemt.png",dpi = 500)
plt.show()

""" Calculate the autocorrelation function and its plot """
R = mc.acf_np(r_st)
window = sig.triang(2*N+1)[N:-1]
R /= window
R /=N

plt.figure(6)
plt.title("Autokorrelation af residualer for stemt lyd")
plt.scatter(range(N),R)
plt.xlabel(r"$k$")
plt.ylabel(r"$r\check_{r^{st}}[k]$",fontsize = 15)
#plt.savefig("figures/autokorrelationstemt.png",dpi = 500)
plt.show()

""" Calculate the autocorrelation function for whole voiced file """
r_st= np.zeros(len(predict)-N)
for sec in range(1,len(parameters)-3):
    delay = sec*N
    gain = parameters[sec]['gain']
    voi = parameters[sec]['voi']
    r_st[delay-N:delay], var2 = mc.res_student(data,predict,delay,N,p,gain,voi)
    
M = len(r_st)
R = mc.acf_np(r_st)
win = sig.triang(2*M+1)[M:-1]
R /= win
R /= M

plt.figure(7)
plt.title("Autokorrelation af residualer for stemt lyd")
plt.scatter(range(N),R[:N])
plt.xlabel(r"$k$")
plt.ylabel(r"$r\check_{r^{st}}[k]$",fontsize = 15)
plt.savefig("figures/autokorrelationstemt2.png",dpi = 500)
plt.show()

# =============================================================================
# Prediction of unvoiced data
# =============================================================================
N = 160
N2 = int(N/2)
p = 12
np.random.seed(1)

parameters2 = lps.LP_parameters(data2,N,p,0)

predict2 = lps.LP_predict(parameters2)  # Unvoiced
                         
""" Plot for unvoiced with predict as hat matrix """
sec = 3
delay = sec*N

plt.figure(8)
plt.title("Ustemt talesignal")
plt.plot(data2[delay:delay+N],label = r"$s[n]$")
plt.plot(predict2[delay:delay+N],label = r"$\hat{s}[n]$")
plt.legend(fontsize = 13,loc = "lower right")
#plt.savefig("figures/compare_unvoiced.png",dpi = 500)
plt.xlabel(r"$n$")
plt.show()

""" Compare the distributions for unvoiced"""
gain2 = parameters2[sec]['gain']
voi2 = parameters2[sec]['voi']
r_st,var2 = mc.res_student(data2,predict2,delay,N,p,gain2,voi2)
var = np.var(r_st)
mean = np.average(r_st)
xes = np.linspace(r_st.min(),r_st.max(),N)
norm = mc.normal_pdf(xes,mean,var)

histf, histbins = np.histogram(r_st,density = True,bins = 15)
hist_midddle = mc.middlehist(histbins)
norm_hist = mc.normal_pdf(hist_midddle,mean,var)
diff = histf - norm_hist
dist_avg2 = np.average(np.abs(diff))
dist_stand2= np.sqrt(np.var(diff))

# Plots
plt.figure(9)
plt.hist(r_st, bins = 15,normed = True)
plt.plot(xes,norm)
plt.plot(mean,mc.normal_pdf(mean,mean,var),"ro")
plt.text(mean,mc.normal_pdf(mean,mean,var),"%.2e" %mean)
plt.xlabel("Residualer")
plt.ylabel("Normaliseret frekvens")
plt.title("Fordeling af residualer for ustemt lyd")
#plt.savefig("figures/unvoiced_distribuion.png",dpi = 500)
plt.show()

plt.figure(10)
a = st.probplot(r_st, dist = "norm",plot = plt)[1][2]
plt.title("QQ-plot for ustemt lyd. " + r"$R^2$" + "= %.3f." %(a))
plt.xlabel("Teoretiske kvantiler")
plt.ylabel("Observerede kvantiler")
plt.savefig("figures/QQustemt.png",dpi = 500)
plt.show()

""" Plots for the residuals for unvoiced """
plt.figure(11)
plt.title("Residualplot for ustemt lyd")
plt.scatter(range(N),r_st)
plt.xlabel(r"$n$",fontsize = 14)
plt.ylabel(r"$r^{st}[n]$",fontsize = 15)
#plt.savefig("figures/residualplotustemt.png",dpi = 500)
plt.show()

""" Calculate the autocorrelation function and its plot """
R = mc.acf_np(r_st)
R /= window
R /=N
plt.figure(12)
plt.title("Autokorrelation af residualer for ustemt lyd")
plt.scatter(range(N),R)
plt.xlabel(r"$k$")
plt.ylabel(r"$r\check_{r^{st}}[k]$",fontsize = 15)
#plt.savefig("figures/autokorrelationustemt.png",dpi = 500)
plt.show()

""" Calculate the autocorrelation function for whole unvoiced file """
r_st = np.zeros(len(predict2)-N)
for sec in range(1,len(parameters2)-3):
    delay = sec*N
    gain = parameters2[sec]['gain']
    voi = parameters2[sec]['voi']
    r_st[delay-N:delay], var2 = mc.res_student(data2,predict2,delay,N,p,gain,voi)
    
M = len(r_st)
R = mc.acf_np(r_st)
win = sig.triang(2*M+1)[M:-1]
R /= win
R /= M

plt.figure(13)
plt.title("Autokorrelation af residualer for ustemt lyd")
plt.scatter(range(N),R[:N])
plt.xlabel(r"$k$")
plt.ylabel(r"$r\check_{r^{st}}[k]$",fontsize = 15)
plt.savefig("figures/autokorrelationustemt2.png",dpi = 500)
plt.show()

#==============================================================================
# Prediction of sentence
#==============================================================================
N = 160
p = 12
filenames = os.listdir(cwd + data_path + "/Saetning")
filenames.pop(2)
data = {}
parameters = {}
predict = {}
np.random.seed(1)
cut = 5
for i in range(len(filenames))[:cut]:
    data[i] = np.array(wav.read(cwd + data_path + "/Saetning/" \
        + filenames[i])[1],dtype= "float64")
    parameters[i] = lps.LP_parameters(data[i],N,p,0)
    predict[i] = lps.LP_predict(parameters[i])

""" Calculate the residuals """
R_coef = {}
a_max = 0
for i in range(len(filenames)):
    R = []
    for j in range(1,len(parameters[i])-3):
        delay = j*N
        gain = parameters[i][j]['gain']
        voi = parameters[i][j]['voi']
        if voi != 0 and not np.all(data[i][delay:delay+N]) == 0: 
            r_st, var2 = mc.res_student(data[i],predict[i],delay,N,p,gain,voi)
            a = st.probplot(r_st, dist = "norm",plot = None)[1][2]
            R.append(a)
            if a >= a_max:
                a_max = a
                i_max = i
                j_max = j
    R_coef[i] = R

R_total = np.hstack((R_coef[0],R_coef[1],R_coef[2],R_coef[3]))

print("The average of r_st: %.3f" %np.average(R_coef[0]))
print("The average of r_st: %.3f" %np.average(R_coef[1]))
print("The average of r_st: %.3f" %np.average(R_coef[2]))
print("The average of r_st: %.3f" %np.average(R_coef[3]))
print("The average of r_st: %.3f" %np.average(R_coef[4]))
print("The average of all r_st: %.3f" %np.average(R_total))
print("")
print("The standard deviation of r_st: %.3f" %np.sqrt(np.var(R_coef[0])))
print("The standard deviation of r_st: %.3f" %np.sqrt(np.var(R_coef[1])))
print("The standard deviation of r_st: %.3f" %np.sqrt(np.var(R_coef[2])))
print("The standard deviation of r_st: %.3f" %np.sqrt(np.var(R_coef[3])))
print("The standard deviation of r_st: %.3f" %np.sqrt(np.var(R_coef[4])))
print("The standard deviation of all r_st: %.3f" %np.sqrt(np.var(R_total)))