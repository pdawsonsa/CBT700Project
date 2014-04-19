# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 15:12:17 2014

@author: peterDawson

CBT700 Project: Controlability analysis of flotation bank
"""
print('')
print('========================== Running================================')
print('')

import control as cn
import numpy as np
import scipy.linalg as la
import scipy


Kp = np.array([[-300., -150., -50.], 
               [250., -250., -120.], 
               [125., 100., -75.]])  
taup = np.array([[120., 150., 100.],
                [80., 110., 200.],
                [60., 20., 70.]])
Dp = np.array([[20., 40., 30.],
               [25., 10., 18.],
               [35., 15., 20.]])


#def _Gp(s, In, Out): 
#    G = Kp[In,Out]*np.exp(-Dp[In,Out]*s)/(taup[In,Out]*s + 1)
#    return(G)  
#    
#    
#def _Gd(s, In, Out):  
#    G = Kd[In,Out]*np.exp(-Dd[In,Out]*s)/(taud[In,Out]*s + 1)
#    return(G) 
    

def _Gp(s):
    dim = np.shape(Kp)
    G = np.zeros((dim))
    G = Kp*np.exp(-Dp*s)/(taup*s + 1)
    return(G) 
    
    
print(_Gp(0.1))