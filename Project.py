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



Kp = np.array([[-30., -15., -5.], 
               [25., -25., -12.], 
               [12.5, 10., -7.50]])  
taup = np.array([[120., 150., 100.],
                [80., 110., 200.],
                [60., 20., 70.]])
Dp = np.array([[20., 40., 30.],
               [25., 10., 18.],
               [35., 15., 20.]])
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
    
    
def _poles():  
    dim = np.shape(Kp)    
    poles = np.zeros((dim))
    poles = -1/taup
    return(poles) 
    
      

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

plt.figure(1)
plt.clf()
rows, cols = np.shape(Kp)
Kc = 1
for i in range(rows):
    for j in range(cols):
        plt.subplot(rows,cols,(3*j) + (i+1))
        omega = np.linspace(0.001,10,10000)
        GpMag = np.zeros((len(omega)))
        def _Gp1(s):
            G = Kp[i,j]*np.exp(-Dp[i,j]*s)/(taup[i,j]*s + 1)
            return(G)
        GpMagL = np.abs(Kc*_Gp1(omega*1j))
        GpMagS = np.abs(1/(1 + (Kc*_Gp1(omega*1j))))
        GpMagT = np.abs((Kc*_Gp1(omega*1j))/(1 + (Kc*_Gp1(omega*1j))))
#        GpMag = 20*np.log(np.abs(_Gp(omega*1j,i,j)))
        plt.loglog(omega,GpMagL,'r-')
        plt.loglog(omega,GpMagS,'b-')
        plt.loglog(omega,GpMagT,'g-')
        plt.axis([np.min(omega), np.max(omega), 0, 100])
        plt.grid(True)
        plt.xlabel('Freq')
        plt.ylabel('Mag')
fig = plt.gcf()
fig.subplots_adjust(bottom=0.05) 
fig.subplots_adjust(top=0.95) 
fig.subplots_adjust(left=0.05) 
fig.subplots_adjust(right=0.99)  
plt.suptitle('Magnitude of frequency responce')      
plt.show()

plt.figure(2)
plt.clf()
rows, cols = np.shape(Kp)
for i in range(rows):
    for j in range(cols):
        plt.subplot(rows,cols,(3*j) + (i+1))
        omega = np.linspace(0.001,10,10000)
        GpPhase = np.zeros((len(omega)))
        def _Gp1(s):
            G = Kp[i,j]*np.exp(-Dp[i,j]*s)/(taup[i,j]*s + 1)
            return(G)
        GpPhase = np.angle(_Gp1(omega*1j), deg=True)
#        GpMag = 20*np.log(np.abs(_Gp(omega*1j,i,j)))
        plt.semilogx(omega,GpPhase,'r-')
        plt.axis([np.min(omega), np.max(omega), -200, 200])
        plt.grid(True)
        plt.xlabel('Freq')
        plt.ylabel('Phase shift')
fig = plt.gcf()
fig.subplots_adjust(bottom=0.05) 
fig.subplots_adjust(top=0.95) 
fig.subplots_adjust(left=0.05) 
fig.subplots_adjust(right=0.99)  
plt.suptitle('Magnitude of frequency responce')      
plt.show()

    
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

print('U values')
print(U)
print('')
print('S values')
print(S)
print('')
print('T values')
print(T)
print('')


