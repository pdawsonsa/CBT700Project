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
from utils import RGA



#=============================================================================
#======================= ASSIGNING OF GLOBAL VARIABLES =======================


Kp = np.array([[28., 22.4, 8.], 
               [-12., 28., 12.], 
               [-8, -12., 16]])  
taup = np.array([[52., 68., 60.],
                [61., 47., 50.],
                [40., 41., 45.]])
Dp = np.array([[10., 15., 20.],
               [15., 10., 15.],
               [15., 15., 10.]])*1  #multiply by 0.0 to remove effects of dead time.
Kd = np.array([-30., -30., -24])
taud = np.array([40., 35., 30.])
Dd = np.array([30., 40., 50.])
Kc = np.array([[1., 0., 0.], 
               [0., 1., 0.], 
               [0., 0., 1.]])
#Kc = 1

#=============================================================================
#========================== DEFINING OF FUNCTIONS ============================


def Gp(s):
    dim = np.shape(Kp)
    G = np.zeros((dim))
    G = Kp*np.exp(-Dp*s)/(taup*s + 1)
    return(G)
    
    

def Gd(s):
    dim = np.shape(Kd)
    G = np.zeros((dim))
    G = Kd*np.exp(-Dd*s)/(taud*s + 1)
    return(G)
    
    
    
def SVD(s):   #SVD of L = KGp
    G = Gp(s)
    L = Kc*G
    S1 = la.inv((np.eye(3) + L))
#    T = L/S1
    [U_L, S_L, V_L] = np.linalg.svd(G)
    [U_S, S_S, V_S] = np.linalg.svd(S1)
    return(U_L, S_L, V_L, U_S, S_S, V_S)
    
  
    
def poles():  
    dim = np.shape(Kp)    
    poles = np.zeros((dim))
    poles = -1/taup
    poleValues = np.zeros((9))
    poleDirsIn = np.empty((3,9),dtype=complex)
    poleDirsOut = np.empty((3,9),dtype=complex)
    c = 0
    for i in poles:
        for j in range(3):
            poleValues[c] = i[j]
            c = c + 1
    c = 0
    for pole in poleValues:
        U, S, V = SVD(pole*1j)
        poleDirsIn[:,c] = V[:,0]
        poleDirsOut[:,c] = U[:,0]
        c = c + 1
    return(poleValues, poleDirsIn, poleDirsOut) 



def bode():
    omega = np.logspace(-3,2,1000)
    magPlotL = np.zeros((len(omega)))
    magPlotS = np.zeros((len(omega)))
    magPlotT = np.zeros((len(omega)))
    wB = np.zeros((3,3))
    plt.figure(11)
    plt.clf()    
    for i in range(3):
        for j in range(3):            
            f = 0                                           #f for flag
            for k in range(len(omega)):
                magPlotL[k] = 20*np.log(np.abs(Gp(omega[k]*1j)[i,j]))
                magPlotS[k] = 20*np.log(np.abs(1/(1 + Gp(omega[k]*1j)[i,j])))
                magPlotT[k] = 20*np.log(np.abs(Gp(omega[k]*1j)[i,j]/(1 + Gp(omega[k]*1j)[i,j])))
                if (f < 1 and magPlotS[k] > -3):
                    wB[i,j] = omega[k]
                    f = 1     
            lineX = np.ones(1000)*wB[i,j]
            lineY = np.linspace(-100, 100, 1000)
            plt.subplot(3,3,3*i + j+1)
            plt.semilogx(omega, magPlotL, 'r-', label='L')
            plt.semilogx(omega, magPlotS, 'b-', label='S')
            plt.semilogx(omega, magPlotT, 'k-', label='T')
            plt.semilogx(omega, np.ones((1000))*-3, 'g-')
            plt.semilogx(lineX, lineY, 'g-')
            plt.text(0.002,-90,'wB = %s rad/s'%(np.round(wB[i,j],3)), color='green')
#            plt.xlabel('Frequency [rad/s]')
            plt.ylabel('Magnitude [dB]')
            plt.title('Valve%s  =>  Level%s'%(j+1,i+1), size=12)
            plt.legend(fontsize=12)
            plt.axis([None,None,-100,100])
            plt.grid(True)
    plt.suptitle('Bode plots of each element of Gp', size=16)
    fig = plt.gcf()
    BG = fig.patch
    BG.set_facecolor('white')
    fig.subplots_adjust(bottom=0.04) 
    fig.subplots_adjust(top=0.93) 
    fig.subplots_adjust(left=0.04) 
    fig.subplots_adjust(right=0.99)
    



def bodeSVD():
    omega = np.logspace(-3,2,1000)
    magPlotL1 = np.zeros((len(omega)))
    magPlotL2 = np.zeros((len(omega)))
    magPlotL3 = np.zeros((len(omega)))
    magPlotS1 = np.zeros((len(omega)))
    magPlotS2 = np.zeros((len(omega)))
    magPlotS3 = np.zeros((len(omega)))
    condNum = np.zeros((len(omega)))
    f = 0
    ff = 0                                                    #f for flag
    for i in range(len(omega)):
        U_L, S_L, V_L, U_S, S_S, V_S = SVD(omega[i]*1j)
        magPlotL1[i] = S_L[0]
        magPlotL2[i] = S_L[1]
        magPlotL3[i] = S_L[2]
        magPlotS1[i] = S_S[0]
        magPlotS2[i] = S_S[1]
        magPlotS3[i] = S_S[2]
        condNum[i] = S_L[0]/S_L[2]  
        if (f < 1 and magPlotL3[i] < 1):
            wC = omega[i]
            f = 1
        if (ff < 1 and magPlotS1[i] > 0.707):
            wB = omega[i]
            ff = 1                                                     
    lineX = np.ones(1000)*wB
    lineY = np.linspace(0.001, 100, 1000)
    lineX1 = np.ones(1000)*wC
    lineY1 = np.linspace(0.001, 100, 1000)
    plt.figure(12)
    plt.clf()
    plt.subplot(211)
    plt.loglog(omega, magPlotL1, 'r-', label = 'G Max $\sigma$')
    plt.loglog(omega, magPlotL3, 'k-', label = 'G Min $\sigma$')
    plt.loglog(omega, magPlotS1, 'r:', label = 'S Max $\sigma$', lw=2)
    plt.loglog(omega, magPlotS3, 'k:', label = 'S Min $\sigma$', lw=2)
    plt.loglog(omega, np.ones((1000))*0.707, 'g-')
    plt.loglog(omega, np.ones((1000))*1, 'b-')
    plt.loglog(lineX, lineY, 'g-')
    plt.loglog(lineX1, lineY1, 'b-')
    plt.text(0.0015,0.3,'wB = %s rad/s'%(np.round(wB,3)), color='green')
    plt.text(0.0015,1.5,'wC = %s rad/s'%(np.round(wC,3)), color='blue')
    plt.xlabel('Frequency [rad/s]')
    plt.ylabel('Singular value [dB]')
    plt.axis([None,None,0.01,100])
    plt.legend(fontsize=12)
    fig = plt.gcf()
    BG = fig.patch
    BG.set_facecolor('white')
    plt.grid(True)
    plt.subplot(212)
    lineX = np.ones(1000)*wB
    lineY = np.linspace(0, 10, 1000)
    plt.semilogx(omega, condNum, 'r-')
    plt.semilogx(lineX, lineY, 'g-')
    plt.text(0.0015,0.3,'wB = %s rad/s'%(np.round(wB,3)), color='green')
    plt.xlabel('Frequency [rad/s]')
    plt.ylabel('Condition number')
    plt.grid(True)
    fig = plt.gcf()
    BG = fig.patch
    BG.set_facecolor('white')
    return(wB,wC)
    
    
def RGAw():
    '''
    Computes the RGA for diagonal pairing at varying frequencies
    '''
    omega = np.logspace(-3,1,1000)
    RGAvalues = np.zeros((len(omega),9))
    RGAnum = np.zeros((len(omega)))
    for i in range(len(omega)):
        G = np.matrix(Gp(omega[i]*1j))
        RGAm = np.abs(np.array(G)*np.array(G.I).T)
        RGAvalues[i,0] = (RGAm[0,0])
        RGAvalues[i,1] = (RGAm[1,0])
        RGAvalues[i,2] = (RGAm[2,0])
        RGAvalues[i,3] = (RGAm[0,1])
        RGAvalues[i,4] = (RGAm[1,1])
        RGAvalues[i,5] = (RGAm[2,1])
        RGAvalues[i,6] = (RGAm[0,2])
        RGAvalues[i,7] = (RGAm[1,2])
        RGAvalues[i,8] = (RGAm[2,2])
        RGAnum[i] = np.sum(RGAm - np.identity(3))
    plt.figure(21)
    plt.clf()
    for i in range(3):
        for j in range(3):
            n = 3*i+j
            plt.subplot(3,3,n+1)
            plt.semilogx(omega, RGAvalues[:,n],'b-', lw=2)
            plt.semilogx(omega, np.ones((1000)), 'r:', lw=3)
            lineX = np.ones(1000)*wB
            lineY = np.linspace(-1, 2, 1000)
            plt.semilogx(lineX, lineY, 'g-')
            plt.title('Valve%s  =>  Level%s'%(j+1,i+1), fontsize=10)
            plt.text(0.002,1.8,'wB = %s rad/s'%(np.round(wB,3)), color='green', fontsize=10)
            plt.text(0.002, 1.1,'|$\lambda$$_i$$_j$| = 1',color='red', fontsize=10)
            plt.ylabel('RGA value |$\lambda$$_i$$_j$|')
            plt.axis([None,None,0,2])
            plt.grid(True)
    plt.suptitle('RGA elements at varying frequencies |$\lambda$$_i$$_j$|', size=16)    
    fig = plt.gcf()
    BG = fig.patch
    BG.set_facecolor('white')
    fig.subplots_adjust(bottom=0.04) 
    fig.subplots_adjust(top=0.91) 
    fig.subplots_adjust(left=0.08) 
    fig.subplots_adjust(right=0.98)
    plt.figure(22)
    plt.clf()
    plt.semilogx(omega, RGAnum, 'b-')
    BG = fig.patch
    BG.set_facecolor('white')
    lineX = np.ones(1000)*np.min(wB)
    lineY = np.linspace(0, max(RGAnum), 1000)
    plt.semilogx(lineX, lineY, 'g-')
    plt.text(0.002,3.6,'min wB = %s rad/s'%(np.round(np.min(wB),3)), color='green', fontsize=10)
    plt.title('RGA number at varying frequencies', size=16)
    plt.ylabel('RGA number')
    plt.grid(True)
    
    




#=============================================================================
#=========================== OUTPUTS AND FIGURES =============================
    

#print('Gp matrix:')
#print(Gp(0))
#print('')
#print('Gd matrix:')
#print(Gd(0))
#print('')

#poleValues, poleDirsIn, poleDirsOut = poles()
#for i in range(9):
#    print('Pole => %s'%(round(poleValues[i], 4)))
#    print('Input1 direction => %s    Output1 direction => %s'%(poleDirsIn[0,i], poleDirsOut[0,i]))
#    print('Input2 direction => %s    Output2 direction => %s'%(poleDirsIn[1,i], poleDirsOut[1,i]))
#    print('Input3 direction => %s    Output3 direction => %s'%(poleDirsIn[2,i], poleDirsOut[2,i]))
#    print('')

bode()
wB, wC = bodeSVD()
RGAw()
U_L, S_L, V_L, U_S, S_S, V_S = SVD(0)

print('The steady state SVD of the system is:')
print('')
print('Input directions:')
print(np.round(V_L,3))
print('')
print('Singular values:')
print(np.round(S_L,3))
print('')
print('Output directions:')
print(np.round(U_L,3))
print('')
print('Condition number:')
print(np.round(S_L[0]/S_L[2],3))



print('')
print('The bandwidth is: %s rad/s'%(np.round(bodeSVD()[0],3)))
print('')


print('')
print('The crossover frequency is: %s rad/s'%(np.round(bodeSVD()[1],3)))
print('')



print('============================== END ==================================')        