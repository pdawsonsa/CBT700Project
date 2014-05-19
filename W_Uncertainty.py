# -*- coding: utf-8 -*-
"""
Created on Thu May 15 16:36:23 2014

@author: pedawson
"""

print('')
print('============================ Started ================================')
print('')

import numpy as np
import matplotlib.pyplot as plt


Kp = np.array([[63., 39.2, 8.], 
               [-27., 49., 12.], 
               [-18., -21., 16.]])  
taup = np.array([[52., 68., 60.],
                [61., 47., 50.],
                [40., 41., 45.]])
Dp = np.array([[10., 15., 20.],
               [15., 10., 15.],
               [15., 15., 10.]])
               
#Uncertainty discription
Kp_e = np.array([[0.05, 0.05, 0.05], 
                   [0.05, 0.05, 0.05], 
                   [0.05, 0.05, 0.05]]) 
taup_e = np.array([[0.05, 0.05, 0.05], 
                   [0.05, 0.05, 0.05], 
                   [0.05, 0.05, 0.05]]) 
Dp_e = np.array([[0.05, 0.05, 0.05], 
                   [0.05, 0.05, 0.05], 
                   [0.05, 0.05, 0.05]]) 




def G(s):
    dim = np.shape(Kp)
    G = np.zeros((dim))
    G = Kp*np.exp(-Dp*s)/(taup*s + 1)
    return(G)
    


def Gp(s, row, col, n):
    K = np.linspace(Kp[row, col]*(1-Kp_e[row, col]), Kp[row, col]*(1+Kp_e[row, col]), n)
    tau = np.linspace(taup[row, col]*(1-taup_e[row, col]), taup[row, col]*(1+taup_e[row, col]), n)
    D = np.linspace(Dp[row, col]*(1-Dp_e[row, col]), Dp[row, col]*(1+Dp_e[row, col]), n)  
    Gp = np.zeros((n**3),dtype=complex)
    c = 0
    for k in K:
        for t in tau:
            for d in D:
                Gp[c] = k*np.exp(-d*s)/(t*s + 1)
                c += 1
    Gp_G = np.abs(Gp - G(s)[row, col])
    return(Gp_G)

def GpPlot(row, col, figNum):
    n = 6
    Gps = np.zeros((n**3,1000))
    w = np.logspace(-3,2,1000)
    for i in range(1000):
        Gps[:,i] = Gp(w[i]*1j, row, col, n)
    
    plt.figure(100)
    plt.clf()
    for i in range(n**3):
        plt.loglog(w, Gps[i,],'b-', alpha=0.2)
        plt.show

K = 4
t1 = 1./3e-2
t2 = 1./8e-2
t3 = 1./6
zeta = 0.3

def W(s):
    Wu = K*(t2*s+ 1)**2/((t1**2*s**2 + 2*t1*zeta*s+ 1)*(t3*s + 1))
    return(Wu)
    

GpPlot(0, 0, 100)
w = np.logspace(-3,2,1000)
plt.loglog(w, W(w), 'r-')
plt.axvline(1./t1)
plt.axvline(1./t2)
plt.axvline(1./t3)


print('============================== END ==================================')