# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 15:12:17 2014

@author: PeterDawson

CBT700 Project: Controlability analysis of MIMO System
"""
print('')
print('============================= Running ================================')
print('')

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import utils
import uncertainty



#=============================================================================
#======================= ASSIGNING OF GLOBAL VARIABLES =======================

#Transfer function parameters
Kp = np.array([[63., 39.2, 8.], 
               [-27., 49., 12.], 
               [-18., -21., 16.]])  
taup = np.array([[52., 68., 60.],
                [61., 47., 50.],
                [40., 41., 45.]])
Dp = np.array([[10., 15., 20.],
               [15., 10., 15.],
               [15., 15., 10.]])*1
Kd = np.array([[-2.25],[-1.75],[-0.8]])
taud = np.array([[40.],[35.],[30.]])
Dd = np.array([[30.],[40.],[50.]])


#Controller
#Kc = np.array([[0.08, 0., 0.], 
#               [0., 0.06, 0.],            #Plant as found controller
#               [0., 0., 0.15]])
KcP = np.array([[0.08, 0., 0.], 
                [0., 0.06, 0.],     
                [0., 0., 0.15]])
KcI = np.array([[400., 1., 1.], 
                [1., 400., 1.],     
                [1., 1., 400.]])
KcD = np.array([[2., 0., 0.], 
                [0., 2., 0.],     
                [0., 0., 2.]])               
               
wB_ = 0.05 

#=============================================================================
#========================== DEFINING OF FUNCTIONS ============================

def Kc(s):
    return(KcP*(1 + 1/(KcI*s) + KcD*s))   # PI Controller

def G(s):
    return(Kp*np.exp(-Dp*s)/(taup*s + 1))
    
def L(s):
    return(Kc(s)*G(s))                         #SVD of L = KG)
    
def S(s):
    return(la.inv((np.eye(3) + L(s))))      #SVD of S = 1/(I + L)
    
def T(s):
    return(L(s)*S(s))                       #SVD of T = L/(I + L)
 
def Gp(s):
    Wo = (20*s + 0.2)/(7.7*s + 1)
    return(-Wo*T(s))   

def Gd(s):
    return(Kd*np.exp(-Dd*s)/(taud*s + 1))
    
    
    
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
        U, Sv, V = utils.SVD(G(pole*1j))
        poleDirsIn[:,c] = V[:,0]
        poleDirsOut[:,c] = U[:,0]
        c = c + 1
    return(poleValues, poleDirsIn, poleDirsOut) 



def bodeSVD():
    w = np.logspace(-3,0,1000)
    magPlotL1 = np.zeros((len(w)))
    magPlotL3 = np.zeros((len(w)))
    magPlotS1 = np.zeros((len(w)))
    magPlotS3 = np.zeros((len(w)))
    magPlotT1 = np.zeros((len(w)))
    magPlotT3 = np.zeros((len(w)))
    condNum = np.zeros((len(w)))
    f = 0
    ff = 0                                                    #f for flag
    for i in range(len(w)):
        U_G, Sv_G, V_G = utils.SVD(G(w[i]*1j))
        U_L, Sv_L, V_L = utils.SVD(L(w[i]*1j))
        U_S, Sv_S, V_S = utils.SVD(S(w[i]*1j))
        U_T, Sv_T, V_T = utils.SVD(T(w[i]*1j))  
        magPlotL1[i] = Sv_G[0]
        magPlotL3[i] = Sv_G[2]
        magPlotS1[i] = Sv_S[0]
        magPlotS3[i] = Sv_S[2]
        magPlotT1[i] = Sv_T[0]
        magPlotT3[i] = Sv_T[2]
        condNum[i] = Sv_G[0]/Sv_G[2]  
        if (f < 1 and magPlotL3[i] < 1):
            wC = w[i]
            f = 1
        if (ff < 1 and magPlotS1[i] > 0.707):
            wB = w[i]
            ff = 1                                                     
    lineX = np.ones(10)*wB
    lineY = np.linspace(0.001, 100, 10)
    lineX1 = np.ones(10)*wC
    lineY1 = np.linspace(0.001, 100, 10)
    plt.figure(1)
    plt.clf()
    plt.subplot(211)
    plt.loglog(w, magPlotL1, 'r-', label = 'G Max $\sigma$')
    plt.loglog(w, magPlotL3, 'r-', label = 'G Min $\sigma$', alpha = 0.4)
    plt.loglog(w, magPlotS1, 'k-', label = 'S Max $\sigma$')
    plt.loglog(w, magPlotS3, 'k-', label = 'S Min $\sigma$', alpha = 0.4)
    plt.loglog(w, np.ones((len(w)))*0.707, 'g-')
    plt.loglog(w, np.ones((len(w)))*1, 'b-')
    plt.loglog(lineX, lineY, 'g-')
    plt.loglog(lineX1, lineY1, 'b-')
    plt.text(0.0015,0.3,'wB = %s rad/s'%(np.round(wB,3)), color='green')
    plt.text(0.0015,1.5,'wC = %s rad/s'%(np.round(wC,3)), color='blue')
    plt.xlabel('Frequency [rad/s]')
    plt.ylabel('Singular value [dB]')
    plt.axis([None,None,0.01,100])
    plt.legend(loc='lower left', fontsize=12, ncol=5)
    fig = plt.gcf()
    BG = fig.patch
    BG.set_facecolor('white')
    plt.grid(True)
    plt.subplot(212)
    lineX = np.ones(10)*wB
    lineY = np.linspace(0, 12, 10)
    plt.semilogx(w, condNum, 'r-')
    plt.semilogx(lineX, lineY, 'g-')
    plt.text(wB*1.1, 0.3, 'wB = %s rad/s'%(np.round(wB,3)), color='green')
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
#    plt.show()
    return(wC)



def perf_Wp():
    w = np.logspace(-3,0,1000)
    magPlotS1 = np.zeros((len(w)))
    magPlotS3 = np.zeros((len(w)))
    Wp = np.zeros((len(w)))
    f = 0                                    #f for flag
    for i in range(len(w)):
        U, Sv, V = utils.SVD(S(w[i]*1j))
        magPlotS1[i] = Sv[0]
        magPlotS3[i] = Sv[2]
        if (f < 1 and magPlotS1[i] > 0.707):
            wB = w[i]
            f = 1
    wB_ = 0.05      #20 sec 
#    M = 2
    A = 0.3
    for i in range(len(w)):
        Wp[i] = utils.Wp(wB_, A, w[i]*1j)                                      
    plt.figure(2)
    plt.clf()
    plt.subplot(211)
    plt.loglog(w, magPlotS1, 'r-', label = 'Max $\sigma$(S)')
    plt.loglog(w, 1./Wp, 'k:', label = '|1/W$_P$|', lw=2)
    plt.axhline(0.707, color='green')
    plt.axvline(wB, color='green')
    plt.axvline(wB_, color='blue', ls=':', lw='2')
    plt.text(wB_*1.1, 7, 'req wB', color='blue')
    plt.text(wB*1.1, 0.12, 'wB = %s rad/s'%(np.round(wB,3)), color='green')
    plt.xlabel('Frequency [rad/s]')
    plt.ylabel('Magnitude')
    plt.axis([None,None,0.1,10])
    plt.legend(loc='upper left', fontsize=12, ncol=5)
    plt.grid(True)
    plt.subplot(212)
    plt.semilogx(w, magPlotS1*Wp, 'r-')
    plt.axhline(1, color='blue', ls=':', lw=2)
    plt.text(0.06, np.max(magPlotS1*Wp)*0.95, '||W$_P$S||$_{inf}$')
    plt.axvline(wB_, color='blue', ls=':', lw='2')
    plt.text(wB_*0.5, np.max(magPlotS1*Wp)*0.95, 'req wB', color='blue')
    plt.xlabel('Frequency [rad/s]')
    plt.ylabel('Magnitude')
    fig = plt.gcf()
    BG = fig.patch
    BG.set_facecolor('white')
    fig.subplots_adjust(bottom=0.2) 
    fig.subplots_adjust(top=0.9) 
    fig.subplots_adjust(left=0.2) 
    fig.subplots_adjust(right=0.9)
    plt.grid(True)
#    plt.show()
    return(wB)    
    
    
def RGAw():
    '''
    Computes the RGA for diagonal pairing at varying frequencies
    '''
    w = np.logspace(-3,0,1000)
    RGAvalues = np.zeros((len(w),9))
    RGAnum = np.zeros((len(w)))
    for i in range(len(w)):
        Gt = np.matrix(G(w[i]*1j))  #Gt is a temp assignment of G
        RGAm = np.abs(utils.RGA(Gt))
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
    plt.figure(3)
    plt.clf()
    for i in range(3):
        for j in range(3):
            n = 3*i+j
            plt.subplot(3,3,n+1)
            plt.semilogx(w, RGAvalues[:,n],'b-', lw=2)
            plt.semilogx(w, np.ones((1000)), 'r:', lw=3)
            plt.axvline(wB, color='green')
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
#    plt.show()
    plt.figure(4)
    plt.clf()
    plt.subplot(211)
    plt.semilogx(w, RGAnum, 'b-')
    BG = fig.patch
    BG.set_facecolor('white')
    plt.axvline(wB, color='green')
    plt.text(0.002,3.6,'min wB = %s rad/s'%(np.round(np.min(wB),3)), color='green', fontsize=10)
    plt.title('RGA number at varying frequencies', size=16)
    plt.ylabel('RGA number')
    plt.xlabel('Frequency [rad/s]')
    fig = plt.gcf()
    BG = fig.patch
    BG.set_facecolor('white')
    fig.subplots_adjust(bottom=0.2) 
    fig.subplots_adjust(top=0.9) 
    fig.subplots_adjust(left=0.2) 
    fig.subplots_adjust(right=0.9)
    plt.grid(True)
#    plt.show()

    

def distRej():
    w = np.logspace(-3,0,1000)
    S1 = np.zeros((len(w)))
    S2 = np.zeros((len(w)))
    Gd1 = np.zeros((len(w)))
    distCondNum = np.zeros((len(w)))
    condNum = np.zeros((len(w)))
    for i in range(len(w)):
        U, Sv, V = utils.SVD(S(w[i]*1j))
        S1[i] = Sv[0]                      #S = 1/|L + 1| 
        S2[i] = Sv[2]
        Gd1[i], distCondNum[i] = utils.distRej(G(w[i]*1j), Gd(w[i]*1j))
        condNum[i] = utils.sigmas(G(w[i]*1j),)[0]*utils.sigmas(la.inv(G(w[i]*1j)))[0]
    plt.figure(5)
    plt.clf()
    plt.subplot(211)
    plt.loglog(w, S1, 'r-', label = 'max $\sigma$S')
    plt.loglog(w, S2, 'r-', alpha = 0.4, label = 'min $\sigma$S')
    plt.loglog(w, Gd1, 'k-', label = '1/||Gd||$_2$')
    plt.axvline(wB, color='green')
    plt.ylabel('Magnitude')
    plt.xlabel('Frequency [rad/s)]')
    plt.text(wB*1.1, 0.015, 'wB = %s rad/s'%(np.round(wB,3)), color='green')
    plt.axis([None, None, None, 10])
    plt.grid(True)
    plt.legend(loc='upper left', fontsize = 12, ncol=5)
    fig = plt.gcf()
    BG = fig.patch
    BG.set_facecolor('white')
    plt.subplot(212)
    plt.semilogx(w, distCondNum, 'r-', label = 'Dist CondNum')
    plt.semilogx(w, condNum, 'k-', label = 'CondNum')
    plt.axvline(wB, color='green')
    plt.ylabel('Disturbance condtion number')
    plt.xlabel('Frequency [rad/s)]')
    plt.text(wB*1.1, 0.2, 'wB = %s rad/s'%(np.round(wB,3)), color='green')
    plt.axis([None, None, 0, None])
    plt.legend(loc='upper left', fontsize = 12, ncol=5)
    plt.grid(True)
    fig = plt.gcf()
    BG = fig.patch
    BG.set_facecolor('white')
    fig.subplots_adjust(bottom=0.2) 
    fig.subplots_adjust(top=0.9) 
    fig.subplots_adjust(left=0.2) 
    fig.subplots_adjust(right=0.9)
#    plt.show()

    

def perfectControl():
    #For perfect cotnrol
    w = np.logspace(-3,0,1000)
    Gd1 = np.zeros((len(w)))
    Gd2 = np.zeros((len(w)))
    Gd3 = np.zeros((len(w)))
    for i in range(len(w)):
        Gt = G(w[i]*1j)                 #Gt just a temp assignmnet for G
        Gdt = Gd(w[i]*1j)               #Gdt just a temp assignmnet for Gd
#        Gd1[i] = la.norm(la.inv(Gt)*Gdt, ord=np.inf)
        Gd1[i] = utils.sigmas(la.inv(Gt)*Gdt)[0]
        Gd2[i] = utils.sigmas(la.inv(Gt)*Gdt)[1]
        Gd3[i] = utils.sigmas(la.inv(Gt)*Gdt)[2]
    plt.figure(6)
    plt.clf()
    plt.subplot(211)
    plt.semilogx(w, Gd1, 'r-', label = '|G$^{-1}$$_1$g$_d$|')
    plt.semilogx(w, Gd2, 'b-', label = '|G$^{-1}$$_2$g$_d$|')
    plt.semilogx(w, Gd3, 'k-', label = '|G$^{-1}$$_3$g$_d$|')
    plt.axvline(wB, color='green')
    plt.text(wB*1.1, np.max(Gd1), 'wB = %s rad/s'%(np.round(wB,3)), color='green')
    plt.ylabel('Magnitude')
    plt.xlabel('Frequency [rad/s)]')
    plt.axis([None, None, None, None])
    plt.grid(True)
    plt.legend(loc='upper left', fontsize = 12, ncol=1)
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
        U, Sv ,V = utils.SVD(G(w[i]*1j))
        Gdt = Gd(w[i]*1j)               #Gdt just a temp assignmnet for Gd
        S1[i] = Sv[0]
        S2[i] = Sv[1]
        S3[i] = Sv[2]
        Gd1[i] = np.max(np.abs(np.transpose(np.conj(U[0]))*Gdt) - 1)
        Gd2[i] = np.max(np.abs(np.transpose(np.conj(U[1]))*Gdt) - 1)
        Gd3[i] = np.max(np.abs(np.transpose(np.conj(U[2]))*Gdt) - 1)
    plt.subplot(212)
    plt.semilogx(w, S1, 'r-', label = '$\sigma$$_1$(G)')
    plt.semilogx(w, Gd1, 'r-', label = '|u$_1$$^H$g$_d$|', alpha = 0.4)
    plt.semilogx(w, S2, 'k-', label = '$\sigma$$_2$(G)')
    plt.semilogx(w, Gd2, 'k-', label = '|u$_2$$^H$g$_d$|', alpha = 0.4)
    plt.semilogx(w, S3, 'b-', label = '$\sigma$$_3$(G)')
    plt.semilogx(w, Gd3, 'b-', label = '|u$_3$$^H$g$_d$|', alpha = 0.4)
    plt.ylabel('Magnitude')
    plt.xlabel('Frequency [rad/s)]')
    plt.axis([None, None, None, None])
    plt.grid(True)
    plt.legend(loc='upper right', fontsize = 12)
    fig = plt.gcf()
    BG = fig.patch
    BG.set_facecolor('white')
    fig.subplots_adjust(bottom=0.2) 
    fig.subplots_adjust(top=0.9) 
    fig.subplots_adjust(left=0.2) 
    fig.subplots_adjust(right=0.9)   
#    plt.show()     

   
    
def MIMOnyqPlot():
    w = np.logspace(-3, 1, 1000)    
    Lin = np.zeros((len(w)), dtype=complex)
    x = np.zeros((len(w)))
    y = np.zeros((len(w)))
    for i in range(len(w)):
        Lin[i] = la.det(np.eye(3) + L(w[i]*1j))
        x[i] = np.real(Lin[i])
        y[i] = np.imag(Lin[i])        
    plt.figure(7)
    plt.clf()
    plt.plot(x, y, 'k-', lw=1)
    plt.xlabel('Re G(wj)')
    plt.ylabel('Im G(wj)')
    # plotting a unit circle
    x = np.linspace(-1, 1, 200)
    y_up = np.sqrt(1-(x)**2)
    y_down = -1*np.sqrt(1-(x)**2)
    plt.plot(x, y_up, 'b:', x, y_down, 'b:', lw=2)
    plt.plot(0, 0, 'r*', ms = 10)
    plt.grid(True)
    n = 3
    plt.axis([-n,n,-n,n])
    fig = plt.gcf()
    BG = fig.patch
    BG.set_facecolor('white')
    fig.subplots_adjust(bottom=0.2) 
    fig.subplots_adjust(top=0.9) 
    fig.subplots_adjust(left=0.2) 
    fig.subplots_adjust(right=0.9)  
#    plt.show()   



def uncertAnalisys():
    #Aditive uncertainty Function ==============================        
    def WA(s):
        K = 9
        ta1 = 1./0.35
        Wa = K/((ta1*s + 1))
        return(np.abs(Wa))
    #Multiplicative uncertainty Function =======================        
    def WM(s):
        tm1 = 1./0.09
        tm2 = 1./0.25
        Wm = (tm1*s + 0.2)/((tm2*s + 1))
        return(np.abs(Wm))
    def M(s):
        Ma = -WA(s) * Kc(s) * S(s)
        Mm = -WM(s) * T(s)
        return(Ma, Mm)
    w = np.logspace(-3,0,1000)
    Mat = np.zeros((len(w)))
    Mmt = np.zeros((len(w)))
    Su = np.zeros((len(w)))
    Tu = np.zeros((len(w)))
    WI = np.zeros((len(w)))
    for i in range(len(w)):
        Ma, Mm = M(w[i]*1j)
        Mat[i] = utils.sigmas(Ma)[0]
        Mmt[i] = utils.sigmas(Mm)[0]
        WI[i] = np.abs(WM(w[i]*1j))
        Su[i] = utils.sigmas(S(w[i]*1j))[0]
        Tu[i] = utils.sigmas(T(w[i]*1j))[0]
    plt.figure(8)
    plt.clf()
    plt.subplot(211)
    plt.semilogx(w, Mmt, 'r-', label = 'Multiplicative W$_O$')
    plt.semilogx(w, Mat, 'b-', label = 'Additive W$_A$')
    plt.axvline(wB_, color='blue', ls=':', lw='2')
    plt.text(wB_*1.1, np.max(Mmt)*0.9, 'req wB', color='blue')
    plt.axhline(1.0, color='blue', ls = ':', lw='2')
    plt.ylabel('Magnitude')
    plt.xlabel('Frequency [rad/s)]')
    plt.axis([None, None, None, None])
    plt.grid(True)
    plt.legend(loc='upper left', fontsize = 12, ncol=5)
    plt.subplot(212)
    plt.loglog(w, Tu, 'r-')
    plt.loglog(w, 1/WI, 'k:', lw='2')
    plt.ylabel('Magnitude')
    plt.xlabel('Frequency [rad/s)]')
    plt.axis([None, None, None, None])
    plt.text(0.2, np.max(Tu), '|T|$_{(not RS)}$', color='red')
#    plt.text(0.11, np.max(Tu)*0.4, '|T|$_{(RS)}$', color='green')
    plt.text(0.01, 5, '1/|W$_I$|')
    plt.grid(True)
    fig = plt.gcf()
    BG = fig.patch
    BG.set_facecolor('white')
    fig.subplots_adjust(bottom=0.2) 
    fig.subplots_adjust(top=0.9) 
    fig.subplots_adjust(left=0.2) 
    fig.subplots_adjust(right=0.9)   
#    plt.show()  
    

#=============================================================================
#=========================== OUTPUTS AND FIGURES =============================



poleValues, poleDirsIn, poleDirsOut = poles()

wC = bodeSVD()
wB = perf_Wp()
RGAw()
distRej()
perfectControl()
MIMOnyqPlot()
uncertAnalisys()
U, SV, V = utils.SVD(G(0))


print('The steady state SVD of the system is:')
print('')
print('Input directions:')
print(np.round(V,3))
print('')
print('Singular values:')
print(np.round(SV,3))
print('')
print('Output directions:')
print(np.round(U,3))
print('')
print('Condition number:')
print(np.round(SV[0]/SV[2],3))

plt.figure(9)
uncertainty.W()


#for i in range(len(poleValues)):
#    for j in range(len(poleDirsIn)):
#        print('%s'%(np.round(poleDirsIn[j,i],4)))

print('')
print('=============================== END ==================================') 
plt.show()
       