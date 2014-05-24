# -*- coding: utf-8 -*-
"""
Created on Thu May 15 16:36:23 2014

@author: pedawson
"""

def W():
    print('')
    print('=================== Running uncertainty fnunction ====================')
    print('')
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    #Transfer function parameters
    Kp = np.array([[63., 39.2, 8.], 
                   [-27., 49., 12.], 
                   [-18., -21., 16.]])  
    taup = np.array([[52., 68., 60.],
                    [61., 47., 50.],
                    [40., 41., 45.]])
    Dp = np.array([[10., 15., 20.],
                   [15., 10., 15.],
                   [15., 15., 10.]])
                   
    #Uncertainty discription parameters
    Kp_e = np.array([[0.090, 0.070, 0.040], 
                       [0.036, 0.028, 0.016], 
                       [0.180, 0.140, 0.080]]) 
    taup_e = np.array([[0.200, 0.147, 0.167], 
                       [0.167, 0.213, 0.200], 
                       [0.250, 0.250, 0.222]]) 
    Dp_e = np.array([[0.500, 0.333, 0.250], 
                     [0.333, 0.500, 0.333], 
                     [0.333, 0.333, 0.500]])
    
    
    
    def G(s):
        return(Kp*np.exp(-Dp*s)/(taup*s + 1))
    
    
    
    def Gp(s, row, col, n):
        K = np.linspace(Kp[row, col]*(1-Kp_e[row, col]), Kp[row, col]*(1+Kp_e[row, col]), n)
        tau = np.linspace(taup[row, col]*(1-taup_e[row, col]), taup[row, col]*(1+taup_e[row, col]), n)
        D = np.linspace(Dp[row, col]*(1-Dp_e[row, col]), Dp[row, col]*(1+Dp_e[row, col]), n)  
        Gp = np.zeros((n**3), dtype=complex)
        c = 0
        for k in K:
            for t in tau:
                for d in D:
                    Gp[c] = k*np.exp(-d*s)/(t*s + 1)
                    c += 1
        Gp_Gadd = np.abs(Gp - G(s)[row, col])                  #Aditive uncertainty
        Gp_Gmult = np.abs((Gp - G(s)[row, col])/G(s)[row, col]) #Multiplicative uncertainty
        return(Gp_Gadd, Gp_Gmult)
    
    def GpPlot(row, col):
        n = 4
        GpsAdd = np.zeros((n**3,1000), dtype=complex)
        GpsMult = np.zeros((n**3,1000), dtype=complex)
        w = np.logspace(-3,1,1000)
        for i in range(1000):
            GpsAdd[:,i], GpsMult[:,i] = Gp(w[i]*1j, row, col, n)
        plt.figure(9)
    #    plt.clf()
        plt.subplot(211)
        for i in range(n**3):
            plt.loglog(w, GpsAdd[i,],'-', color = ([row*0.3, col*0.3, 1]), alpha=0.2)
            plt.grid(True)
            plt.ylabel('|Additive Uncertainty|')
            plt.xlabel('Frequency [rad/s)]')
        plt.subplot(212)
        for i in range(n**3):
            plt.loglog(w, GpsMult[i,],'-', color = ([row*0.3, col*0.3, 1]), alpha=0.2)
            plt.grid(True)
            plt.ylabel('|Multiplicative Uncertainty|')
            plt.xlabel('Frequency [rad/s)]')
        fig = plt.gcf()
        BG = fig.patch
        BG.set_facecolor('white')
        fig.subplots_adjust(bottom=0.2) 
        fig.subplots_adjust(top=0.9) 
        fig.subplots_adjust(left=0.2) 
        fig.subplots_adjust(right=0.9)
    
    plt.clf()
    def errorPlot():
        for i in range(3):
            for j in range(3):
                GpPlot(i, j)
    
    errorPlot()
            
    #Aditive uncertainty Function ==============================        
    K = 9
    ta1 = 1./0.35
    def WA(s):
        Wa = K/((ta1*s + 1))
        return(np.abs(Wa))
    #=========================================================== 
    #Multiplicative uncertainty Function =======================        
    tm1 = 1./0.09
    tm2 = 1./0.25
    def WM(s):
        Wm = (tm1*s + 0.2)/((tm2*s + 1))
        return(np.abs(Wm))
    #===========================================================   
    plt.subplot(211)    
    w = np.logspace(-3,1,1000)
    plt.loglog(w, WA(w*1j), 'r-', lw=2)
    plt.axvline(1./ta1)
    
    plt.subplot(212)    
    w = np.logspace(-3,1,1000)
    plt.loglog(w, WM(w*1j), 'r-', lw=2)
    plt.axvline(1./tm1)
    plt.axvline(1./tm2)



