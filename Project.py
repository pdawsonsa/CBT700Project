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
    K = 1.
    G = Gp(s)
    L = K*G
    S = 1/(np.eye(3) + L)
    T = L/(np.eye(3) + L)
    [U, S, V] = np.linalg.svd(L)
    return(U, S, V)
    
    
    
def SVD_S(s):   #SVD of S = 1/(I + L)
    K = 1.
    G = Gp(s)
    L = K*G
    S = 1/(np.eye(3) + L)
    T = L/(np.eye(3) + L)
    [U, S, V] = np.linalg.svd(S)
    return(U, S, V)
    
    
    
def poles():  
    dim = np.shape(Kp)    
    poles = np.zeros((dim))
    poles = -1/taup
    poleValues = np.zeros((9))
    poleDirs = np.empty((3,9),dtype=complex)
    c = 0
    for i in poles:
        for j in range(3):
            poleValues[c] = i[j]
            c = c + 1
    c = 0
    for pole in poleValues:
        U, S, V = SVD(pole*1j)
        poleDirs[:,c] = U[:,0]
        c = c + 1
    return(poleValues, poleDirs, U) 



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
            plt.title('Valve%s  =>  Level%s'%(j+1,i+1))
            plt.legend()
            plt.axis([None,None,-100,100])
            plt.grid(True)
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.04) 
    fig.subplots_adjust(top=0.97) 
    fig.subplots_adjust(left=0.04) 
    fig.subplots_adjust(right=0.99)
    return(wB)
    
    

def bodeSVD():
    omega = np.logspace(-3,2,1000)
    magPlot1 = np.zeros((len(omega)))
    magPlot2 = np.zeros((len(omega)))
    magPlot3 = np.zeros((len(omega)))
    condNum = np.zeros((len(omega)))
    f = 0                                                    #f for flag
    for i in range(len(omega)):
        U, S, V = SVD(omega[i]*1j)
        magPlot1[i] = 20*np.log(S[0])
        magPlot2[i] = 20*np.log(S[1])
        magPlot3[i] = 20*np.log(S[2])
        condNum[i] = S[0]/S[2]  
        if (f < 1 and magPlot3[i] < 0):
            crossOver = omega[i]
            f = 1                                        
    lineX = np.ones(1000)*crossOver
    lineY = np.linspace(-100, 100, 1000)
    plt.figure(12)
    plt.clf()
    plt.subplot(211)
    plt.semilogx(omega, magPlot1, 'r-')
    plt.semilogx(omega, magPlot2, 'b-')
    plt.semilogx(omega, magPlot3, 'k-')
    plt.semilogx(omega, np.zeros((1000)), 'g-')
    plt.semilogx(lineX, lineY, 'g-')
    plt.text(0.002,-90,'CrossOver = %s rad/s'%(np.round(crossOver,3)), color='green')
    plt.xlabel('Frequency [rad/s]')
    plt.ylabel('Singular value [dB]')
    plt.axis([None,None,-100,100])
    plt.grid(True)
    plt.subplot(212)
    plt.semilogx(omega, condNum, 'r-')
    plt.xlabel('Frequency [rad/s]')
    plt.ylabel('Condition number')
    plt.grid(True)
    return(crossOver)
    
    
    
def RGAw():
    '''Computes the RGA for diagonal pairing at varying frequencies'''
    omega = np.logspace(-3,3,1000)
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
    wB = bode()
    plt.figure(21)
    plt.clf()
    for i in range(3):
        for j in range(3):
            n = 3*i+j
            plt.subplot(3,3,n+1)
            plt.semilogx(omega, RGAvalues[:,n],'b-', lw=2)
            plt.semilogx(omega, np.ones((1000)), 'r:', lw=3)
            lineX = np.ones(1000)*wB[i,j]
            lineY = np.linspace(-1, 2, 1000)
            plt.semilogx(lineX, lineY, 'g-')
            plt.title('Valve%s  =>  Level%s'%(j+1,i+1), fontsize=12)
            plt.text(0.002,-0.8,'wB = %s rad/s'%(np.round(wB[i,j],3)), color='green')
            plt.text(0.002, 1.1,'|$\lambda$$_i$$_j$| = 1',color='red')
#            plt.xlabel('Frequency [rad/s]')
            plt.ylabel('RGA value |$\lambda$$_i$$_j$|')
            plt.axis([None,None,-1,2])
            plt.grid(True)
    plt.suptitle('RGA elements at varying frequencies |$\lambda$$_i$$_j$|', size=16)    
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.04) 
    fig.subplots_adjust(top=0.93) 
    fig.subplots_adjust(left=0.04) 
    fig.subplots_adjust(right=0.99)
    plt.figure(22)
    plt.clf()
    plt.semilogx(omega, RGAnum, 'r-')
    lineX = np.ones(1000)*np.min(wB)
    lineY = np.linspace(0, max(RGAnum), 1000)
    plt.semilogx(lineX, lineY, 'g-')
    plt.text(0.002,0.25,'min wB = %s rad/s'%(np.round(np.min(wB),3)), color='green')
    plt.title('RGA number at varying frequencies', size=16)
    plt.ylabel('RGA number')
    plt.grid(True)
    
    




#=============================================================================
#=========================== OUTPUTS AND FIGURES =============================
    

print('Gp matrix:')
print(Gp(0))
print('')
print('Gd matrix:')
print(Gd(0))
print('')

#poleValues, poleDirs, U = poles()
#for i in range(9):
#    print('Pole => %s'%(round(poleValues[i], 4)))
#    print('Output1 direction => %s'%(poleDirs[0,i]))
#    print('Output2 direction => %s'%(poleDirs[1,i]))
#    print('Output3 direction => %s'%(poleDirs[2,i]))
#    print('')

print('')
print('The bandwidth frequencies in rad/s are:')
print((np.round(bode(),3)))
print('')


print('')
print('The crossover frequency is: %s rad/s'%(np.round(bodeSVD(),3)))
print('')


RGAw()

print('============================== END ==================================')        