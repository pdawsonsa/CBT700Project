# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 15:12:17 2014

@author: peterDawson

CBT700 Project: Controlability analysis of MIMO System
"""
print('')
print('============================ Started ================================')
print('')

import control as cn
import numpy as np
import scipy.linalg as la
import scipy
import matplotlib.pyplot as plt



Kp = np.array([[35., 28., 10.], 
               [-15., 35., 15.], 
               [-10, -15., 20]])  
taup = np.array([[50., 68., 60.],
                [60., 47., 50.],
                [40., 40., 45.]])
Dp = np.array([[10., 15., 20.],
               [15., 10., 15.],
               [15., 15., 10.]])#*0.0  #multiply by 0.0 to remove effects of dead time.
Kd = np.array([10., 6., 2.])
taud = np.array([50., 70., 100.])
Dd = np.array([5., 5., 5.])


def _Gp(s):
    dim = np.shape(Kp)
    G = np.zeros((dim))
    G = Kp*np.exp(-Dp*s)/(taup*s + 1)
    return(G)
    

def _Gd(s):
    dim = np.shape(Kd)
    G = np.zeros((dim))
    G = Kd*np.exp(-Dd*s)/(taud*s + 1)
    return(G)
    
    
def _SVD(s):
    K = 1
    G = _Gp(s)
    L = K*G
    S = 1/(np.eye(3) + L)
    T = L/(np.eye(3) + L)
    [U, S, V] = np.linalg.svd(L)
    return(U, S, V)
    
    
def _SVD_S(s):
    K = 1
    G = _Gp(s)
    L = K*G
    S = 1/(np.eye(3) + L)
    T = L/(np.eye(3) + L)
    [U, S, V] = np.linalg.svd(S)
    return(U, S, V)
    
    
def _poles():  
    dim = np.shape(Kp)    
    poles = np.zeros((dim))
    poles = -1/taup
    
    
    return(poles) 
      

def _bode():
    omega = np.logspace(-3,2,1000)
    magPlot1 = np.zeros((len(omega)))
    magPlot2 = np.zeros((len(omega)))
    magPlot3 = np.zeros((len(omega)))
    magPlot1dB = np.zeros((len(omega)))
    magPlot2dB = np.zeros((len(omega)))
    magPlot3dB = np.zeros((len(omega)))
    condNum = np.zeros((len(omega)))
    for i in range(len(omega)):
        U, S, V = _SVD(omega[i]*1j)
        magPlot1[i] = (S[0])
        magPlot2[i] = (S[1])
        magPlot3[i] = (S[2])
        magPlot1dB[i] = 20*np.log(S[0])
        magPlot2dB[i] = 20*np.log(S[1])
        magPlot3dB[i] = 20*np.log(S[2])
        condNum[i] = S[0]/S[2]        
     
    plt.figure(11)
    plt.clf()
    plt.subplot(211)
    plt.loglog(omega, magPlot1, 'r-')
    plt.loglog(omega, magPlot2, 'b-')
    plt.loglog(omega, magPlot3, 'k-')
    plt.loglog(omega, np.ones((1000)), 'g-')
    plt.xlabel('Frequency [rad/s]')
    plt.ylabel('Singular value [dB]')
    plt.grid(True)
    plt.subplot(212)
    plt.semilogx(omega, condNum, 'r-')
    plt.xlabel('Frequency [rad/s]')
    plt.ylabel('Condition number')
    plt.grid(True) 
    
    plt.figure(12)
    plt.clf()
    plt.subplot(211)
    plt.semilogx(omega, magPlot1dB, 'r-')
    plt.semilogx(omega, magPlot2dB, 'b-')
    plt.semilogx(omega, magPlot3dB, 'k-')
    plt.semilogx(omega, np.ones((1000)), 'g-')
    plt.xlabel('Frequency [rad/s]')
    plt.ylabel('Singular value [dB]')
    plt.grid(True)
    plt.subplot(212)
    plt.semilogx(omega, condNum, 'r-')
    plt.xlabel('Frequency [rad/s]')
    plt.ylabel('Condition number')
    plt.grid(True)
    magPlot1dB = np.zeros((len(omega)))
    magPlot2dB = np.zeros((len(omega)))
    magPlot3dB = np.zeros((len(omega)))
    condNum = np.zeros((len(omega)))
    for i in range(len(omega)):
        U, S, V = _SVD_S(omega[i]*1j)
        magPlot1[i] = (S[0])
        magPlot2[i] = (S[1])
        magPlot3[i] = (S[2])
        magPlot1dB[i] = 20*np.log(S[0])
        magPlot2dB[i] = 20*np.log(S[1])
        magPlot3dB[i] = 20*np.log(S[2])
        condNum[i] = S[0]/S[2] 
    plt.subplot(211)
    plt.semilogx(omega, magPlot1dB, 'r:')
    plt.semilogx(omega, magPlot2dB, 'b:')
    plt.semilogx(omega, magPlot3dB, 'k:')
    plt.semilogx(omega, np.ones((1000)), 'g-')
    plt.xlabel('Frequency [rad/s]')
    plt.ylabel('Singular value [dB]')
    plt.grid(True)
    plt.subplot(212)
    plt.semilogx(omega, condNum, 'b:')
    plt.xlabel('Frequency [rad/s]')
    plt.ylabel('Condition number')
    plt.grid(True)





    
  
    
#def _zeros():
#    dim = np.shape(Kp)
#    zeros = np.eye((dim[0]))
#    for i in range(dim[0]):
#        for j in range(dim[1]):
#            z0 = 2/Dp[i,j]   #Pade approx 1st order (2-Ds) = 0
#            def _Gz(s):
#                _Gz = Kp[i,j]*np.exp(-Dp[i,j]*s)/(taup[i,j]*s + 1)
#                return(_Gz)
#            zeros[i,j] = scipy.optimize.fsolve(_Gz,z0)
#            
#    return(zeros)
  

#Frequency response of elements in matrix Gp:  Bode plots

#plt.figure(1)
#plt.clf()
#rows, cols = np.shape(Kp)
#Kc = 1
#for i in range(rows):
#    for j in range(cols):
#        plt.subplot(rows,cols,(3*j) + (i+1))
#        omega = np.linspace(0.001,10,10000)
#        GpMag = np.zeros((len(omega)))
#        def _Gp1(s):
#            G = Kp[i,j]*np.exp(-Dp[i,j]*s)/(taup[i,j]*s + 1)
#            return(G)
#        GpMagL = np.abs(Kc*_Gp1(omega*1j))
#        GpMagS = np.abs(1/(1 + (Kc*_Gp1(omega*1j))))
#        GpMagT = np.abs((Kc*_Gp1(omega*1j))/(1 + (Kc*_Gp1(omega*1j))))
##        GpMag = 20*np.log(np.abs(_Gp(omega*1j,i,j)))
#        plt.loglog(omega,GpMagL,'r-')
#        plt.loglog(omega,GpMagS,'b-')
#        plt.loglog(omega,GpMagT,'g-')
#        plt.axis([np.min(omega), np.max(omega), 0, 100])
#        plt.grid(True)
#        plt.xlabel('Freq')
#        plt.ylabel('Mag')
#fig = plt.gcf()
#fig.subplots_adjust(bottom=0.05) 
#fig.subplots_adjust(top=0.95) 
#fig.subplots_adjust(left=0.05) 
#fig.subplots_adjust(right=0.99)  
#plt.suptitle('Magnitude of frequency responce')      



    
[U, S, T] = np.linalg.svd(_Gp(0.1))
#[U, S, T] = np.linalg.svd([[5, 4],[3, 2]])  #Example 3.3 in Skogestad give me the correct SVD elements


print('Gp matrix:')
print(_Gp(0.1))
print('')
print('Gd matrix:')
print(_Gd(0.1))
print('')
print('Poles of Gp:')
print(_poles())
print('')
#print('Zeros of Gp:')
#print(_zeros())
#print('')

U, S, V = _SVD(0.1)
print('U values')
print(U)
print('')
print('S values')
print(S)
print('')
print('T values')
print(T)
print('')

_bode()
