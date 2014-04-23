# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 15:12:17 2014

@author: PeterDawson

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
    
    
def _SVD(s):   #SVD of L = KGp
    K = 1.
    G = _Gp(s)
    L = K*G
    S = 1/(np.eye(3) + L)
    T = L/(np.eye(3) + L)
    [U, S, V] = np.linalg.svd(L)
    return(U, S, V)
    
    
def _SVD_S(s):   #SVD of S = 1/(I + L)
    K = 1.
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
    poleValues = np.zeros((9))
#    poleDirs = np.zeros((9))
#    poleValues = np.empty(9,dtype=complex)
    poleDirs = np.empty((3,9),dtype=complex)
    m = 0
    for i in poles:
        for j in range(3):
            poleValues[m] = i[j]
            m = m + 1
    m = 0
    for pole in poleValues:
        U, S, V = _SVD(pole*1j)
        poleDirs[:,m] = U[:,0]
        m = m + 1
    return(poleValues, poleDirs, U) 
poleValues, poleDirs, U = _poles()
for i in range(9):
    print('Pole => %s'%(round(poleValues[i], 4)))
    print('Output1 direction => %s'%(poleDirs[0,i]))
    print('Output2 direction => %s'%(poleDirs[1,i]))
    print('Output3 direction => %s'%(poleDirs[2,i]))
#    print(S)
#    print(V)
    print('')

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
    

    
#[U, S, V] = np.linalg.svd(_Gp(0.1))
#[U, S, V] = np.linalg.svd([[5, 4],[3, 2]])  #Example 3.3 in Skogestad give me the correct SVD elements


#print('Gp matrix:')
#print(_Gp(0.1))
#print('')
#print('Gd matrix:')
#print(_Gd(0.1))
#print('')
#print('Poles of Gp:')
#print(_poles())
#print('')
##print('Zeros of Gp:')
##print(_zeros())
##print('')
#
#U, S, V = _SVD(0.1)
#print('U values')
#print(U)
#print('')
#print('S values')
#print(S)
#print('')
#print('T values')
#print(V)
#print('')

_bode()



#def minsigma(s):
#    U, Sigma, V = np.linalg.svd(_Gp(s))
#    return Sigma.min()
#
#plt.
#complex_plot(minsigma, (-5, 5), (-5, 5))

#zeros1 = np.zeros((100),dtype=complex)
#a = linspace(-5,1,100)
#m = 0
#for i in range(len(a)):
##    for k in range(len(a)):
#        U,S,V = la.svd(_Gp(a[i]*1j))
#        zeros1[m] = S.min()
#        m += 1
#
#print(zeros1)



        