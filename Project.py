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



def matrixValues():  
    Kp = np.matrix('-300. -150. -50.; 250. -250. -120.; 125. 100. -75.')
    taup = np.matrix('120. 150. 100.; 80. 110. 200.; 60. 20. 70.')
    Dp = np.matrix('20. 40. 30.; 25. 10. 18.; 35. 15. 20.')
    Kd = np.matrix('10. 6., 2.')
    taud = np.matrix('50. 70., 100.')
    Dd = np.matrix('5., 5., 5.')
    return(Kp, taup, Dp, Kd, taud, Dd)  

Kp, taup, Dp, Kd, taud, Dd = matrixValues()

def _Gp(s, In, Out): 
    Kp, taup, Dp, Kd, taud, Dd = matrixValues()
    G = Kp[In,Out]*np.exp(-Dp[In,Out]*s)/(taup[In,Out]*s + 1)
    return(G)  
    
    
def _Gd(s, In, Out):  
    G = Kd[In,Out]*np.exp(-Dd[In,Out]*s)/(taud[In,Out]*s + 1)
    return(G) 
    

def _GpMatrix(s):
    dim = np.shape(Kp)
    G = np.zeros((dim))
    for i in range(dim[0]):             #Input number
        for j in range(dim[1]):         #Output number
            G[i,j] = _Gp(s, i, j)
    return(G) 
 

def _GdMatrix(s):
    dim = np.shape(Kd)
    G = np.zeros((dim))
    for i in range(dim[0]):
        for j in range(dim[1]):
            G[i,j] = _Gd(s, i, j)
    return(G)
    
    
def _poles():  
    dim = np.shape(Kp)    
    poles = np.zeros((dim))
    for i in range(dim[0]):
        for j in range(dim[1]):
            poles[i,j] = -1/taup[i,j]
    return(poles) 
    
      

def _zeros():
    dim = np.shape(Kp)
    zeros = np.eye((dim[0]))
    for i in range(dim[0]):
        for j in range(dim[1]):
            z0 = 2/Dp[i,j]   #Pade approx 1st order (2-Ds) = 0
            def _Gz(s):
                _Gz = Kp[i,j]*np.exp(-Dp[i,j]*s)/(taup[i,j]*s + 1)
                return(_Gz)
            zeros[i,j] = scipy.optimize.fsolve(_Gz,z0)
            
    return(zeros)
  

#Frequency response of elements in matrix Gp:  Bode plots

plt.figure(1)
plt.clf()
rows, cols = np.shape(Kp)
for i in range(rows):
    for j in range(cols):
        plt.subplot(rows,cols,(3*j) + (i+1))
        omega = np.linspace(0.001,10,10000)
        GpMag = np.zeros((len(omega)))
        GpMag = np.abs(_Gp(omega*1j,i,j))
#        GpMag = 20*np.log(np.abs(_Gp(omega*1j,i,j)))
        plt.loglog(omega,GpMag,'b-')
        plt.axis([np.min(omega), np.max(omega), 0, 1000])
        plt.grid(True)
        plt.xlabel('Freq')
        plt.ylabel('Mag [dB]')
fig = plt.gcf()
fig.subplots_adjust(bottom=0.05) 
fig.subplots_adjust(top=0.95) 
fig.subplots_adjust(left=0.05) 
fig.subplots_adjust(right=0.99)  
plt.suptitle('Magnitude of frequency responce')      
plt.show()


    
[U, S, T] = np.linalg.svd(_GpMatrix(0.1))
#[U, S, T] = np.linalg.svd([[5, 4],[3, 2]])  #Example 3.3 in Skogestad give me the correct SVD elements


print('Gp matrix:')
print(_GpMatrix(0.1))
print('')
print('Gd matrix:')
print(_GdMatrix(0.1))
print('')
print('Poles of Gp:')
print(_poles())
print('')
print('Zeros of Gp:')
print(_zeros())
print('')

print('U values')
print(U)
print('')
print('S values')
print(S)
print('')
print('T values')
print(T)
print('')


