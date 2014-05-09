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



#=============================================================================
#======================= ASSIGNING OF GLOBAL VARIABLES =======================


Kp = np.array([[63., 39.2, 8.], 
               [-27., 49., 12.], 
               [-18, -21., 16]])  
taup = np.array([[52., 68., 60.],
                [61., 47., 50.],
                [40., 41., 45.]])
Dp = np.array([[10., 15., 20.],
               [15., 10., 15.],
               [15., 15., 10.]])*1  #multiply by 0.0 to remove effects of dead time.
Kd = np.array([-12., -12., -9.6])
taud = np.array([40., 35., 30.])
Dd = np.array([30., 40., 50.])
Kc = np.array([[1., 0., 0.], 
               [0., 1., 0.], 
               [0., 0., 1.]])*1
#Kc = 1

#=============================================================================
#========================== DEFINING OF FUNCTIONS ============================


def G(s):
    dim = np.shape(Kp)
    G = np.zeros((dim))
    G = Kp*np.exp(-Dp*s)/(taup*s + 1)
    return(G)
    
    

def Gd(s):
    dim = np.shape(Kd)
    G = np.zeros((dim))
    G = Kd*np.exp(-Dd*s)/(taud*s + 1)
    return(G)
    
    
    
def SVD_G(s):   
    [U, S, V] = np.linalg.svd(G(s))
    return(U, S, V)   
    
def SVD_L(s):   
    L = Kc*G(s)   #SVD of L = KG
    [U, S, V] = np.linalg.svd(L)
    return(U, S, V)   

def SVD_S(s):   #SVD of L = KG
    L = Kc*G(s)   #SVD of L = KG
    S = la.inv((np.eye(3) + L))
    [U, S, V] = np.linalg.svd(S)
    return(U, S, V)

def SVD_T(s):   #SVD of L = KG
    L = Kc*G(s)   #SVD of L = KG
    S = la.inv((np.eye(3) + L))
    T = L*S
    [U, S, V] = np.linalg.svd(T)
    return(U, S, V)    
    
    
#  def SVD(s):   #SVD of L = KG
#    L = Kc*G(s)
#    S = la.inv((np.eye(3) + L))
#    T = L*S
#    [U_L, S_L, V_L] = np.linalg.svd(G(s))
#    [U_S, S_S, V_S] = np.linalg.svd(S)
#    [U_T, S_T, V_T] = np.linalg.svd(T)
#    return(U_L, S_L, V_L, U_S, S_S, V_S, U_T, S_T, V
    
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
        U, S, V = SVD_G(pole*1j)
        poleDirsIn[:,c] = V[:,0]
        poleDirsOut[:,c] = U[:,0]
        c = c + 1
    return(poleValues, poleDirsIn, poleDirsOut) 



def bodeSVD():
    w = np.logspace(-3,2,1000)
    magPlotL1 = np.zeros((len(w)))
    magPlotL2 = np.zeros((len(w)))
    magPlotL3 = np.zeros((len(w)))
    magPlotS1 = np.zeros((len(w)))
    magPlotS2 = np.zeros((len(w)))
    magPlotS3 = np.zeros((len(w)))
    magPlotT1 = np.zeros((len(w)))
    magPlotT2 = np.zeros((len(w)))
    magPlotT3 = np.zeros((len(w)))
    condNum = np.zeros((len(w)))
    f = 0
    ff = 0                                                    #f for flag
    for i in range(len(w)):
        U_G, S_G, V_G = SVD_G(w[i]*1j)
        U_L, S_L, V_L = SVD_L(w[i]*1j)
        U_S, S_S, V_S = SVD_S(w[i]*1j)
        U_T, S_T, V_T = SVD_T(w[i]*1j)   # Not optimal, revise!
        magPlotL1[i] = S_L[0]
        magPlotL2[i] = S_L[1]
        magPlotL3[i] = S_L[2]
        magPlotS1[i] = S_S[0]
        magPlotS2[i] = S_S[1]
        magPlotS3[i] = S_S[2]
        magPlotT1[i] = S_T[0]
        magPlotT2[i] = S_T[1]
        magPlotT3[i] = S_T[2]
        condNum[i] = S_G[0]/S_G[2]  
        if (f < 1 and magPlotL3[i] < 1):
            wC = w[i]
            f = 1
        if (ff < 1 and magPlotS1[i] > 0.707):
            wB = w[i]
            ff = 1                                                     
    lineX = np.ones(len(w))*wB
    lineY = np.linspace(0.001, 100, len(w))
    lineX1 = np.ones(len(w))*wC
    lineY1 = np.linspace(0.001, 100, len(w))
    plt.figure(1)
    plt.clf()
    plt.subplot(211)
    plt.loglog(w, magPlotL1, 'r-', label = 'G Max $\sigma$')
    plt.loglog(w, magPlotL3, 'r:', label = 'G Min $\sigma$', lw=2)
    plt.loglog(w, magPlotS1, 'k-', label = 'S Max $\sigma$')
    plt.loglog(w, magPlotS3, 'k:', label = 'S Min $\sigma$', lw=2)
    plt.loglog(w, magPlotT1, 'b-', label = 'T Max $\sigma$')
    plt.loglog(w, magPlotT3, 'b:', label = 'T Min $\sigma$', lw=2)    
    plt.loglog(w, np.ones((len(w)))*0.707, 'g-')
    plt.loglog(w, np.ones((len(w)))*1, 'b-')
    plt.loglog(lineX, lineY, 'g-')
    plt.loglog(lineX1, lineY1, 'b-')
    plt.text(0.0015,0.3,'wB = %s rad/s'%(np.round(wB,3)), color='green')
    plt.text(0.0015,1.5,'wC = %s rad/s'%(np.round(wC,3)), color='blue')
    plt.xlabel('Frequency [rad/s]')
    plt.ylabel('Singular value [dB]')
    plt.axis([None,None,0.001,100])
    plt.legend(fontsize=12)
    fig = plt.gcf()
    BG = fig.patch
    BG.set_facecolor('white')
    plt.grid(True)
    plt.subplot(212)
    lineX = np.ones(len(w))*wB
    lineY = np.linspace(0, 10, len(w))
    plt.semilogx(w, condNum, 'r-')
    plt.semilogx(lineX, lineY, 'g-')
    plt.text(0.0015,0.3,'wB = %s rad/s'%(np.round(wB,3)), color='green')
    plt.xlabel('Frequency [rad/s]')
    plt.ylabel('Condition number')
    plt.grid(True)
    fig = plt.gcf()
    BG = fig.patch
    BG.set_facecolor('white')
    fig.subplots_adjust(bottom=0.2) 
    fig.subplots_adjust(top=0.9) 
    fig.subplots_adjust(left=0.2) 
    fig.subplots_adjust(right=0.9)
    plt.grid(True)
    return(wB,wC)
    
    
    
def RGAw():
    '''
    Computes the RGA for diagonal pairing at varying frequencies
    '''
    w = np.logspace(-3,1,1000)
    RGAvalues = np.zeros((len(w),9))
    RGAnum = np.zeros((len(w)))
    for i in range(len(w)):
        Gt = np.matrix(G(w[i]*1j))  #Gt is a temp assignment of G
        RGAm = np.abs(np.array(Gt)*np.array(Gt.I).T)
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
    plt.figure(2)
    plt.clf()
    for i in range(3):
        for j in range(3):
            n = 3*i+j
            plt.subplot(3,3,n+1)
            plt.semilogx(w, RGAvalues[:,n],'b-', lw=2)
            plt.semilogx(w, np.ones((1000)), 'r:', lw=3)
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
    plt.figure(3)
    plt.clf()
    plt.subplot(212)
    plt.semilogx(w, RGAnum, 'b-')
    BG = fig.patch
    BG.set_facecolor('white')
    lineX = np.ones(1000)*np.min(wB)
    lineY = np.linspace(0, max(RGAnum), 1000)
    plt.semilogx(lineX, lineY, 'g-')
    plt.text(0.002,3.6,'min wB = %s rad/s'%(np.round(np.min(wB),3)), color='green', fontsize=10)
    plt.title('RGA number at varying frequencies', size=16)
    plt.ylabel('RGA number')
    plt.xlabel('Frequency [rad/s)]')
    fig = plt.gcf()
    BG = fig.patch
    BG.set_facecolor('white')
    fig.subplots_adjust(bottom=0.2) 
    fig.subplots_adjust(top=0.9) 
    fig.subplots_adjust(left=0.2) 
    fig.subplots_adjust(right=0.9)
    plt.grid(True)
    
    

def distRej():
    w = np.logspace(-3,1,1000)
    S1 = np.zeros((len(w)))
    S2 = np.zeros((len(w)))
    Gd1 = np.zeros((len(w)))
    for i in range(len(w)):
        U, S, V = SVD_S(w[i]*1j)
        S1[i] = S[2]                      #S = 1/|L + 1| 
        S2[i] = S[0]
        Gd1[i] = 1/la.norm(Gd(w[i]*1j),2)   #Returns largest sing value of Gd(wj)
    plt.figure(4)
    plt.clf()
    plt.subplot(211)
    plt.loglog(w, S1, 'r-', label = 'min $\sigma$S')
    plt.loglog(w, S2, 'r:', lw=2, label = 'max $\sigma$S')
    plt.loglog(w, Gd1, 'k-', label = '1/||Gd||$_2$')
    plt.ylabel('Magnitude')
    plt.xlabel('Frequency [rad/s)]')
    plt.axis([None, None, None, None])
    plt.grid(True)
    plt.legend(fontsize = 12)
    fig = plt.gcf()
    BG = fig.patch
    BG.set_facecolor('white')
    fig.subplots_adjust(bottom=0.2) 
    fig.subplots_adjust(top=0.9) 
    fig.subplots_adjust(left=0.2) 
    fig.subplots_adjust(right=0.9)
    


def perfectControl():
    #For perfect cotnrol
    w = np.logspace(-3,1,1000)
    Gd1 = np.zeros((len(w)))
    Gd2 = np.zeros((len(w)))
    Gd3 = np.zeros((len(w)))
    for i in range(len(w)):
        Gt = G(w[i]*1j)                 #Gt just a temp assignmnet for G
        Gdt = Gd(w[i]*1j)               #Gdt just a temp assignmnet for Gd
        Gd1[i] = la.norm(la.inv(Gt)*Gdt[0], ord=inf)
        Gd2[i] = la.norm(la.inv(Gt)*Gdt[1], ord=inf)
        Gd3[i] = la.norm(la.inv(Gt)*Gdt[2], ord=inf)
    plt.figure(5)
    plt.clf()
    plt.subplot(211)
    plt.semilogx(w, Gd1, 'r-', label = '||G$^{-1}$g$_d$1||$_{max}$')
    plt.semilogx(w, Gd2, 'k-', label = '||G$^{-1}$g$_d$2||$_{max}$')
    plt.semilogx(w, Gd3, 'b-', label = '||G$^{-1}$g$_d$3||$_{max}$')
    plt.ylabel('Magnitude')
    plt.xlabel('Frequency [rad/s)]')
    plt.axis([None, None, None, None])
    plt.grid(True)
    plt.legend(fontsize = 12)
    fig = plt.gcf()
    BG = fig.patch
    BG.set_facecolor('white')
    fig.subplots_adjust(bottom=0.2) 
    fig.subplots_adjust(top=0.9) 
    fig.subplots_adjust(left=0.2) 
    fig.subplots_adjust(right=0.9)
    #For acceptable control
    S1 = np.zeros((len(w)))
    S2 = np.zeros((len(w)))
    S3 = np.zeros((len(w)))
    Gd1 = np.zeros((len(w)))
    Gd2 = np.zeros((len(w)))
    Gd3 = np.zeros((len(w)))
    for i in range(len(w)):
        U,S,V = la.svd(G(w[i]*1j))
        Gdt = Gd(w[i]*1j)               #Gdt just a temp assignmnet for Gd
        S1[i] = S[0]
        S2[i] = S[1]
        S3[i] = S[2]
        Gd1[i] = np.max(np.abs(np.transpose(np.conj(U[0]))*Gdt[0]) - 1)
        Gd2[i] = np.max(np.abs(np.transpose(np.conj(U[1]))*Gdt[1]) - 1)
        Gd3[i] = np.max(np.abs(np.transpose(np.conj(U[2]))*Gdt[2]) - 1)
    plt.subplot(212)
    plt.semilogx(w, S1, 'r-', label = '$\sigma$$_1$(G)')
    plt.semilogx(w, Gd1, 'r:', label = '|u$_1$$^H$g$_d$|')
    plt.semilogx(w, S2, 'k-', label = '$\sigma$$_2$(G)')
    plt.semilogx(w, Gd2, 'k:', label = '|u$_2$$^H$g$_d$|')
    plt.semilogx(w, S3, 'b-', label = '$\sigma$$_3$(G)')
    plt.semilogx(w, Gd3, 'b:', label = '|u$_3$$^H$g$_d$|')
    plt.ylabel('Magnitude')
    plt.xlabel('Frequency [rad/s)]')
    plt.axis([None, None, None, None])
    plt.grid(True)
    plt.legend(fontsize = 12)
    fig = plt.gcf()
    BG = fig.patch
    BG.set_facecolor('white')
    fig.subplots_adjust(bottom=0.2) 
    fig.subplots_adjust(top=0.9) 
    fig.subplots_adjust(left=0.2) 
    fig.subplots_adjust(right=0.9)        
        
    
    
def nyqPlot():
    """w_start and w_end is the number that would be 10**(number)"""
#    def mod(w):
#        return np.abs(G(w))-1
#
#    w_start_n = sc.optimize.fsolve(mod, 0.001)
#    print w_start_n

#    plt.plot(np.real(G(w_start_n)), np.imag(G(w_start_n)), 'rD')
#    w_start = np.log(w_start_n)

    
    w = np.logspace(-3, 2, 10000)    
    GL = np.zeros((len(w)), dtype=complex)
    x = np.zeros((len(w)))
    y = np.zeros((len(w)))
    for i in range(len(w)):
        L = Kc*G(w[i]*1j)
        GL[i] = la.det(np.eye(3) + L)
        x[i] = np.real(GL[i])
        y[i] = np.imag(GL[i])
        
    plt.figure(6)
    plt.clf()
    plt.plot(x, y, 'b+')
    plt.xlabel('Re G(wj)')
    plt.ylabel('Im G(wj)')

    # plotting a unit circle
    x = np.linspace(-1, 1, 200)

    y_upper = np.sqrt(1-(x)**2)
    y_down = -1*np.sqrt(1-(x)**2)
    plt.plot(x, y_upper, 'r-', x, y_down, 'r-')
    plt.grid(True)
    n = 2
    plt.axis([-n,n,-n,n])

    print "finished"




#=============================================================================
#=========================== OUTPUTS AND FIGURES =============================
    

#print('G matrix:')
#print(G(0))
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

wB, wC = bodeSVD()
RGAw()
distRej()
perfectControl()
nyqPlot()
U, S, V = SVD_G(0)


print('The steady state SVD of the system is:')
print('')
print('Input directions:')
print(np.round(V,3))
print('')
print('Singular values:')
print(np.round(S,3))
print('')
print('Output directions:')
print(np.round(U,3))
print('')
print('Condition number:')
print(np.round(S[0]/S[2],3))



print('')
print('The bandwidth is: %s rad/s'%(np.round(bodeSVD()[0],3)))
print('')


print('')
print('The crossover frequency is: %s rad/s'%(np.round(bodeSVD()[1],3)))
print('')

#for i in range(len(poleValues)):
#    for j in range(len(poleDirsIn)):
#        print('%s'%(np.round(poleDirsOut[j,i],4)))



print('============================== END ==================================')        