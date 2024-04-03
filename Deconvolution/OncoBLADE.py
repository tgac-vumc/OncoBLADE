from numba import jit, njit 
import numpy as np
from numpy import transpose as t
import scipy.optimize
from scipy.special import loggamma
from scipy.special import gamma
from scipy.special import digamma
from scipy.special import polygamma

from sklearn.svm import NuSVR
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import KFold

from joblib import Parallel, delayed
import itertools
import time
import math

import warnings

__all__ = ['OncoBLADE', 'Framework', 'Framework_Iterative', 'Reestimate_Nu', 'Purify_AllGenes']

### Variational parameters Q(X|Nu, Omega) ###
# Nu: Nsample by Ngene by Ncell
# Omega : Ngene by Ncell


# Beta : Nsample by Ncell
# ExpQ : Nsample by Ngene by Ncell


@njit(fastmath=True)
def ExpF(Beta, Ncell):
    #NSample by Ncell (Expectation of F)
    output = np.empty(Beta.shape)
    for c in range(Ncell):
        output[:,c] = Beta[:,c]/np.sum(Beta, axis=1)
    return output


@njit(fastmath=True)
def ExpQ(Nu, Beta, Omega, Ngene, Ncell, Nsample):
    # Ngene by Nsample (Expected value of Y)
    ExpB = ExpF(Beta, Ncell) # Nsample by Ncell
    out = np.zeros((Ngene, Nsample))
    for i in range(Nsample):
        for c in range(Ncell):
            out[:,i] = out[:,i] + ExpB[i,c] * np.exp(Nu[i,:,c] + 0.5*np.square(Omega[:,c]))

    return out 


@njit(fastmath=True)
def VarQ(Nu, Beta, Omega, Ngene, Ncell, Nsample):
    # Ngene by Nsample (Variance value of Y)
    B0 = np.sum(Beta, axis=1) # Nsample
    Btilda = ExpF(Beta, Ncell) # Nsample by Ncell
    VarB = Btilda * (1-Btilda)
    for c in range(Ncell):
        VarB[:,c] = VarB[:,c] / (B0+1)

    # Nsample Ncell Ncell
    CovB = np.empty((Nsample, Ncell, Ncell))
    for l in range(Ncell):
        for k in range(Ncell):
            CovB[:,l,k] = - Btilda[:,l] * Btilda[:,k] / (1+B0)

    # Ngene by Nsample by Ncell by Ncell
    CovX = np.empty((Ngene, Nsample, Ncell, Ncell))
    for l in range(Ncell):
        for k in range(Ncell):
            for i in range(Nsample):
                CovX[:,i,l,k] = np.exp(Nu[i,:,k] + Nu[i,:,l] + \
                        0.5*(np.square(Omega[:,k]) + np.square(Omega[:,l])))

    VarTerm = np.zeros((Ngene, Nsample))
    for i in range(Nsample):
        for c in range(Ncell):
            VarTerm[:,i] = VarTerm[:,i] + \
                np.exp(2*Nu[i,:,c] + 2*np.square(Omega)[:,c])*(VarB[i,c] + np.square(Btilda[i,c])) \
                    - np.exp(2*Nu[i,:,c] + np.square(Omega[:,c]))*(np.square(Btilda[i,c]))

    # Ngene by Ncell
    CovTerm = np.zeros((Ngene, Nsample))
    for l in range(Ncell):
        for k in range(Ncell):
            if l != k:
                for i in range(Nsample):
                    CovTerm[:,i] = CovTerm[:,i] + CovX[:,i,l,k] * CovB[i,l,k]
     
    return VarTerm + CovTerm


@njit(fastmath=True)
def Estep_PY(Y, SigmaY, Nu, Omega, Beta, Ngene, Ncell, Nsample):
    Var = VarQ(Nu, Beta, Omega, Ngene, Ncell, Nsample)
    Exp = ExpQ(Nu, Beta, Omega, Ngene, Ncell, Nsample)

    a = Var / Exp / Exp
                            
    return np.sum(
            -0.5 / np.square(SigmaY) * (a + np.square((Y-np.log(Exp)) - 0.5 * a))
            )

       
@njit(fastmath=True)
def grad_Nu(Y, SigmaY, Nu, Omega, Beta, Mu0, Alpha0, Beta0, Kappa0, Ngene, Ncell, Nsample, weight):
    # return Nsample by Ngene by Ncell

    # gradient of PX (first term)
    AlphaN = Alpha0 + Nsample * 0.5
    NuExp = np.sum(Nu, 0)/Nsample

    Diff = np.zeros((Ngene, Ncell))
    ExpBetaN = Beta0 + (Nsample-1)/2*np.square(Omega) + \
            Kappa0*Nsample/(2*(Kappa0 + Nsample)) * (np.square(Omega)/Nsample + np.square(NuExp - Mu0))

    for i in range(Nsample):
        ExpBetaN = ExpBetaN + 0.5*np.square(Nu[i,:,:] - NuExp)
        Diff = Diff + (Nu[i,:,:] - NuExp) / Nsample

    Nominator = np.empty((Nsample, Ngene, Ncell)) 
    for i in range(Nsample):
        Nominator[i,:,:] = Nu[i,:,:] - NuExp - Diff + Kappa0 / (Kappa0+Nsample) * (NuExp - Mu0)
   
    grad_PX = - AlphaN * Nominator / ExpBetaN

    # gradient of PY (second term)
    # Ngene by Nsample (Variance value of Y)
    B0 = np.sum(Beta, axis=1) # Nsample
    Btilda = ExpF(Beta, Ncell) # Nsample by Ncell

    # Ngene by Ncell by Nsample
    Exp = ExpQ(Nu, Beta, Omega, Ngene, Ncell, Nsample) # Ngene by Nsample
    Var = VarQ(Nu, Beta, Omega, Ngene, Ncell, Nsample) # Ngene by Nsample

    # Ngene by Nsample by Ncell by Ncell
    CovB = np.empty((Nsample, Ncell, Ncell))
    for l in range(Ncell):
        for k in range(Ncell):
            CovB[:,l,k] = - Btilda[:,l] * Btilda[:,k] / (1+B0)

    ExpX = np.empty(Nu.shape) # Nsample by Ngene by Ncell
    for i in range(Nsample):
        ExpX[i,:,:] = np.exp(Nu[i,:,:] + 0.5*np.square(Omega))

    CovX = np.empty((Ngene, Nsample, Ncell, Ncell))
    for l in range(Ncell):
        for k in range(Ncell):
            for i in range(Nsample):
                CovX[:,i,l,k] = ExpX[i,:,l] * ExpX[i,:,k]

    # Ngene by Ncell by Nsample
    CovTerm = np.zeros((Ngene, Ncell, Nsample))
    for l in range(Ncell):
        for k in range(Ncell):
            if l != k:
                for i in range(Nsample):
                    CovTerm[:,l,i] = CovTerm[:,l,i] + 2*CovX[:,i,l,k]*CovB[i,l,k]

    # Ngene by Ncell by Nsample
    g_Exp = np.empty((Ngene, Ncell, Nsample))
    for c in range(Ncell):
        for i in range(Nsample):
            g_Exp[:,c,i] = ExpX[i,:,c]*Btilda[i,c]

    # Ngene by Ncell by Nsample
    g_Var = np.empty((Ngene, Ncell, Nsample))
    VarX = np.empty(Nu.shape) 
    for i in range(Nsample):
        VarX[i,:,:] = np.exp(2*Nu[i,:,:] + 2*np.square(Omega))

    VarB = Btilda * (1-Btilda)
    for c in range(Ncell):
        VarB[:,c] = VarB[:,c] / (B0+1)

    for c in range(Ncell):
        for i in range(Nsample):
            g_Var[:,c,i] = 2*VarX[i,:,c]*(VarB[i,c] + np.square(Btilda[i,c])) - 2*CovX[:,i,c,c]*np.square(Btilda[i,c])
    g_Var = g_Var + CovTerm

    # Ngene by Ncell by Nsample
    a = np.empty((Ngene, Ncell, Nsample))
    for c in range(Ncell):
        a[:,c,:] = (g_Var[:,c,:] - 2*g_Exp[:,c,:]/Exp*Var) / np.power(Exp,2)

    b = np.empty((Ngene, Ncell, Nsample))
    Diff = Y - np.log(Exp) - Var / (2*np.square(Exp))
    for c in range(Ncell):
        b[:,c,:] = - Diff * (2*g_Exp[:,c,:] / Exp + a[:,c,:])

    grad_PY = np.zeros((Nsample, Ngene, Ncell))
    for c in range(Ncell):
        grad_PY[:,:,c] = -np.transpose( 0.5/np.square(SigmaY) * (a[:,c,:] + b[:,c,:]))

    return grad_PX *(1/weight) + grad_PY


@njit(fastmath = True)
def grad_Omega(Y, SigmaY, Nu, Omega, Beta, Mu0, Alpha0, Beta0, Kappa0, Ngene, Ncell, Nsample, weight):
    # Ngene by Ncell

    # gradient of PX (first term)
    AlphaN = Alpha0 + Nsample * 0.5
    NuExp = np.sum(Nu, 0)/Nsample
    ExpBetaN = Beta0 + (Nsample-1)/2*np.square(Omega) + \
            Kappa0*Nsample/(2*(Kappa0 + Nsample)) * (np.square(Omega)/Nsample + np.square(NuExp - Mu0))

    for i in range(Nsample):
        ExpBetaN = ExpBetaN + 0.5*np.square(Nu[i,:,:] - NuExp)

    Nominator = - AlphaN * (Nsample-1)*Omega + Kappa0 /(Kappa0 + Nsample) * Omega
    grad_PX = Nominator / ExpBetaN

    # gradient of PY (second term)
    # Ngene by Nsample (Variance value of Y)
    B0 = np.sum(Beta, axis=1) # Nsample
    Btilda = ExpF(Beta, Ncell) # Nsample by Ncell

    # Ngene by Ncell by Nsample
    Exp = ExpQ(Nu, Beta, Omega, Ngene, Ncell, Nsample) # Ngene by Nsample
    Var = VarQ(Nu, Beta, Omega, Ngene, Ncell, Nsample) # Ngene by Nsample

    # Ngene by Nsample by Ncell by Ncell
    CovB = np.empty((Nsample, Ncell, Ncell))
    for l in range(Ncell):
        for k in range(Ncell):
            CovB[:,l,k] = - Btilda[:,l] * Btilda[:,k] / (1+B0)

    ExpX = np.exp(Nu) # Nsample by Ngene by Ncell
    for i in range(Nsample):
        ExpX[i,:,:] = ExpX[i,:,:]*np.exp(0.5*np.square(Omega))

    CovX = np.empty((Ngene, Nsample, Ncell, Ncell))
    for l in range(Ncell):
        for k in range(Ncell):
            for i in range(Nsample):
                CovX[:,i,l,k] = ExpX[i,:,l] * ExpX[i,:,k]

    # Ngene by Ncell by Nsample
    CovTerm = np.zeros((Ngene, Ncell, Nsample))
    for l in range(Ncell):
        for k in range(Ncell):
            if l != k:
                for i in range(Nsample):
                    CovTerm[:,l,i] = CovTerm[:,l,i] + 2*CovX[:,i,l,k]*CovB[i,l,k]*Omega[:,l]

    # Ngene by Ncell by Nsample
    g_Exp = np.empty((Ngene, Ncell, Nsample))
    for c in range(Ncell):
        for i in range(Nsample):
            g_Exp[:,c,i] = ExpX[i,:,c]*Btilda[i,c]*Omega[:,c]

    # Ngene by Ncell by Nsample
    g_Var = np.empty((Ngene, Ncell, Nsample))
    VarX = np.exp(2*Nu)
    for i in range(Nsample):
        VarX[i,:,:] = VarX[i,:,:] * np.exp(2*np.square(Omega))

    VarB = Btilda * (1-Btilda)
    for c in range(Ncell):
        VarB[:,c] = VarB[:,c] / (B0+1)

    for c in range(Ncell):
        for i in range(Nsample):
            g_Var[:,c,i] = 4*Omega[:,c]*VarX[i,:,c]*(VarB[i,c] + np.square(Btilda[i,c])) - 2*Omega[:,c]*CovX[:,i,c,c]*np.square(Btilda[i,c])
    g_Var = g_Var + CovTerm

    # Ngene by Ncell by Nsample
    a = np.empty((Ngene, Ncell, Nsample))
    for c in range(Ncell):
        a[:,c,:] = (g_Var[:,c,:] - 2*g_Exp[:,c,:]*Var/Exp) / np.power(Exp,2)

    b = np.empty((Ngene, Ncell, Nsample))
    Diff = Y - np.log(Exp) - Var / (2*np.square(Exp))
    for c in range(Ncell):
        b[:,c,:] = - Diff * (2*g_Exp[:,c,:] / Exp + a[:,c,:])

    grad_PY = np.zeros((Ngene, Ncell))
    for c in range(Ncell):
        grad_PY[:,c] = np.sum(-0.5/np.square(SigmaY) * (a[:,c,:] + b[:,c,:]), axis=1)

    # Q(X) (fourth term)
    grad_QX =  - Nsample / Omega

    return grad_PX * (1/weight) + grad_PY - grad_QX * (1/weight)


@njit(fastmath = True)
def g_Exp_Beta(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample):
    
    ExpX = np.exp(Nu)
    for i in range(Nsample):
        ExpX[i,:,:] = ExpX[i,:,:]*np.exp(0.5*np.square(Omega)) #Nsample by Ngene by Ncell
    B0mat = np.empty(Beta.shape)
    for c in range(Ncell):
        B0mat[:,c] =Beta[:,c]/np.square(B0)

    tmp = np.empty((Nsample, Ngene))
    tExpX = np.ascontiguousarray(ExpX.transpose(0,2,1)) ## Make tExpX contiguous again
    for i in range(Nsample):
        tmp[i,:] = np.dot(B0mat[i,:], tExpX[i,...])
    B0mat = tmp

    g_Exp = np.empty((Nsample, Ncell, Ngene))

    for s in range(Nsample):
        for c in range(Ncell):
            g_Exp[s,c,:] = t(ExpX[s,:,c] / B0[s]) - B0mat[s,:]

    return g_Exp


@njit(fastmath=True)
def g_Var_Beta(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample):
    
    B0Rep = np.empty(Beta.shape) # Nsample by Ncell
    for c in range(Ncell):
        B0Rep[:,c] = B0

    aa = (B0Rep - Beta)*B0Rep*(B0Rep + 1) - (3*B0Rep + 2) * Beta * (B0Rep - Beta)
    aa = aa/(np.power(B0Rep,3) * np.square(B0Rep + 1))
    aa = aa + 2*Beta*(B0Rep - Beta)/np.power(B0Rep,3)

    aaNotT = Beta * B0Rep * (B0Rep + 1) - (3*B0Rep + 2) * Beta * (B0Rep - Beta)
    aaNotT = aaNotT / (np.power(B0Rep,3) * np.square(B0Rep + 1))
    aaNotT = aaNotT + 2*Beta*(0 - Beta)/np.power(B0Rep,3)
    
    ExpX2 = 2*Nu #Nsample by Ngene by Ncell
    for i in range(Nsample):
        ExpX2[i,:,:,] = np.exp(ExpX2[i,:,:] + 2*np.square(Omega))

    g_Var = np.zeros((Nsample, Ncell, Ngene))

    for s in range(Nsample):
        for c in range(Ncell):
            g_Var[s,c,:] = t(ExpX2[s,:,c]) * aa[s,c]
  
    for i in range(Ncell):
        for j in range(Ncell):
            if i != j:
                for s in range(Nsample):
                    g_Var[s,i,:] = g_Var[s,i,:] + t(ExpX2[s,:,j])* aaNotT[s,j]

    B_B02 = Beta / np.square(B0Rep) # Beta / (Beta0^2) / Nsample by Ncell
    B0B0_1 = B0Rep * (B0Rep + 1) # Beta0 (Beta0+1) / Nsample by Nell
    B2_B03 = np.square(Beta) / np.power(B0Rep, 3) # Beta^2 / (Beta0^3) / Nsample by Ncell
        
    ExpX = np.empty(Nu.shape)
    for i in range(Nsample):
        ExpX[i,:,:] = np.exp(2*Nu[i,:,:]+np.square(Omega))

    for s in range(Nsample):
        for c in range(Ncell):
            g_Var[s,c,:] = g_Var[s,c,:] - 2 * t(ExpX[s,:,c]) * B_B02[s,c]
    
    Dot = np.zeros((Nsample, Ngene))
    for i in range(Nsample):
        for c in range(Ncell):
            Dot[i,:] = Dot[i,:] + B2_B03[i,c] * ExpX[i,:,c]

    for c in range(Ncell):
        g_Var[:,c,:] = g_Var[:,c,:] + 2*Dot
    
    # Ngene by Nsample by Ncell by N cell
    ExpX = np.empty((Nsample, Ngene, Ncell))
    for i in range(Nsample):
        ExpX[i,:,:] = np.exp(Nu[i,:,:] + 0.5*np.square(Omega))
    CovX = np.empty((Nsample, Ngene, Ncell, Ncell))
    for l in range(Ncell):
        for k in range(Ncell):
            for i in range(Nsample):
                CovX[i,:,l,k] = ExpX[i,:,l] * ExpX[i,:,k]
                
    gradCovB = np.empty((Nsample, Ncell, Ncell))
    B03_2_B03_B0_1 = (3*B0 + 2) / np.power(B0,3) / np.square(B0+1)
    for l in range(Ncell):
        for k in range(Ncell):
            gradCovB[:,l,k] = Beta[:,l] * Beta[:,k] * B03_2_B03_B0_1

    # Nsample by Ncell by Ncell by Ngene
    CovTerm1 = np.zeros((Nsample, Ncell, Ncell, Ngene))
    CovTerm2 = np.zeros((Nsample, Ncell, Ncell, Ngene))
    B_B0_1_B0B0_1 = Beta * (B0Rep + 1) / np.square(B0B0_1) # Nsample by Ncell
    for l in range(Ncell):
        for k in range(Ncell):
            for i in range(Nsample):
                if l != k:
                    CovTerm1[i,l,k,:] = gradCovB[i,l,k]*CovX[i,:,l,k]
                    CovTerm2[i,l,k,:] = B_B0_1_B0B0_1[i,l]*CovX[i,:,l,k]

    for c in range(Ncell):
        g_Var[:,c,:] = g_Var[:,c,:] + np.sum(np.sum(CovTerm1, axis=1), axis=1)
    g_Var = g_Var - 2*np.sum(CovTerm2, axis=1)

    return g_Var


@njit(fastmath=True)
def g_PY_Beta(Nu, Beta, Omega, Y, SigmaY, B0, Ngene, Ncell, Nsample):

    # Ngene by Nsample
    Exp = ExpQ(Nu, Beta, Omega, Ngene, Ncell, Nsample)
    Var = VarQ(Nu, Beta, Omega, Ngene, Ncell, Nsample)
        
    # Nsample by Ncell be Ngene
    g_Exp = g_Exp_Beta(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample)
    g_Var = g_Var_Beta(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample)
        
    # Nsample by Ncell by Ngene
    a = np.empty((Nsample, Ncell, Ngene))
    for c in range(Ncell):
        a[:,c,:] = np.divide((g_Var[:,c,:] * t(Exp) - 2 * g_Exp[:,c,:]*t(Var)),np.power(t(Exp),3))
    
    b = np.empty((Nsample, Ncell, Ngene))
    Var_Exp2 = np.divide(Var, 2*np.square(Exp))
    for s in range(Nsample):
        for c in range(Ncell):
            for g in range(Ngene):
                b[s,c,g] = - (Y[g,s] - np.log(Exp[g,s]) - Var_Exp2[g,s]) *(2*np.divide(g_Exp[s,c,g],Exp[g,s]) + a[s,c,g])

    grad_PY = np.zeros((Nsample, Ncell))
    for s in range(Nsample):
        for c in range(Ncell):
            grad_PY[s,c] = grad_PY[s,c] - np.sum(0.5 / np.square(SigmaY[:,s]) * (a[s,c,:] + b[s,c,:]))
    
    return grad_PY


@njit(fastmath=True)
def g_PY_Beta(Nu, Beta, Omega, Y, SigmaY, B0, Ngene, Ncell, Nsample):

    # Ngene by Nsample
    Exp = ExpQ(Nu, Beta, Omega, Ngene, Ncell, Nsample)
    Var = VarQ(Nu, Beta, Omega, Ngene, Ncell, Nsample)
        
    # Nsample by Ncell be Ngene
    g_Exp = g_Exp_Beta(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample)
    g_Var = g_Var_Beta(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample)
        
    # Nsample by Ncell by Ngene
    a = np.empty((Nsample, Ncell, Ngene))
    for c in range(Ncell):
        a[:,c,:] = np.divide((g_Var[:,c,:] * t(Exp) - 2 * g_Exp[:,c,:]*t(Var)),np.power(t(Exp),3))
    
    b = np.empty((Nsample, Ncell, Ngene))
    Var_Exp2 = np.divide(Var, 2*np.square(Exp))
    for s in range(Nsample):
        for c in range(Ncell):
            for g in range(Ngene):
                b[s,c,g] = - (Y[g,s] - np.log(Exp[g,s]) - Var_Exp2[g,s]) *(2*np.divide(g_Exp[s,c,g],Exp[g,s]) + a[s,c,g])

    grad_PY = np.zeros((Nsample, Ncell))
    for s in range(Nsample):
        for c in range(Ncell):
            grad_PY[s,c] = grad_PY[s,c] - np.sum(0.5 / np.square(SigmaY[:,s]) * (a[s,c,:] + b[s,c,:]))
    
    return grad_PY


@njit(fastmath = True)
def Estep_PX(Mu0, Nu, Omega, Alpha0, Beta0, Kappa0, Ncell, Nsample):
    NuExp = np.sum(Nu, 0)/Nsample # expected Nu, Ngene by Ncell
    AlphaN = Alpha0 + 0.5*Nsample # Posterior Alpha

    ExpBetaN = Beta0 + (Nsample-1)/2*np.square(Omega) + \
            Kappa0*Nsample/(2*(Kappa0 + Nsample)) * (np.square(Omega)/Nsample + np.square(NuExp - Mu0))

    for i in range(Nsample):
        ExpBetaN = ExpBetaN + 0.5*np.square(Nu[i,:,:] - NuExp)

    return np.sum(- AlphaN * np.log(ExpBetaN))


class OncoBLADE:
    def __init__(self, Y, SigmaY=0.05, Mu0=2, Alpha=1,\
            Alpha0=1, Beta0=1, Kappa0=1,\
            Nu_Init=None, Omega_Init=1, Beta_Init=None, \
            fix_Beta = False, fix_Nu=False, fix_Omega=False):
        self.weight = 1
        self.Y = Y
        self.Ngene, self.Nsample = Y.shape
        self.Fix_par = {
            'Beta': fix_Beta,
            'Nu': fix_Nu,
            'Omega': fix_Omega
        }

        # process input variable
        if not isinstance(Mu0, np.ndarray):
            self.Ncell = Mu0
            self.Mu0 = np.zeros((self.Ngene, self.Ncell))
        else:
            self.Ncell = Mu0.shape[1]
            self.Mu0 = Mu0

        if isinstance(SigmaY, np.ndarray):
            self.SigmaY = SigmaY
        else:
            self.SigmaY = np.ones((self.Ngene, self.Nsample))*SigmaY

        if isinstance(Alpha, np.ndarray):
            self.Alpha = Alpha
        else:
            self.Alpha = np.ones((self.Nsample, self.Ncell))*Alpha

        if isinstance(Omega_Init, np.ndarray):
            self.Omega = Omega_Init
        else:
            self.Omega = np.zeros((self.Ngene, self.Ncell)) + Omega_Init
 
        if Nu_Init is None:
            self.Nu = np.zeros((self.Nsample, self.Ngene, self.Ncell))
        else:
            self.Nu = Nu_Init
       
        if isinstance(Beta_Init, np.ndarray):
            self.Beta = Beta_Init
        else:
            self.Beta = np.ones((self.Nsample, self.Ncell))

        if isinstance(Alpha0, np.ndarray):
            self.Alpha0 = Alpha0 
        else:
            self.Alpha0 = np.ones((self.Ngene, self.Ncell))*Alpha0

        if isinstance(Beta0, np.ndarray):
            self.Beta0 = Beta0 
        else:
            self.Beta0 = np.ones((self.Ngene, self.Ncell))*Beta0

        if isinstance(Kappa0, np.ndarray):
            self.Kappa0 = Kappa0 
        else:
            self.Kappa0 = np.ones((self.Ngene, self.Ncell))*Kappa0
   
    def Ydiff(self, Nu, Beta):
        F = self.ExpF(Beta)
        Ypred = np.dot(np.exp(Nu), t(F))
        return np.sum(np.square(self.Y-Ypred))

    def ExpF(self, Beta):
        #NSample by Ncell (Expectation of F)
        return ExpF(Beta, self.Ncell)

    def ExpQ(self, Nu, Beta, Omega):
        # Ngene by Nsample (Expected value of Y)
        return ExpQ(Nu, Beta, Omega, self.Ngene, self.Ncell, self.Nsample)

    def VarQ(self, Nu, Beta, Omega):
        # Ngene by Nsample (Variance value of Y)
        return VarQ(Nu, Beta, Omega, self.Ngene, self.Ncell, self.Nsample)

    # Expectation of log P(X | mu0, Kappa0, Alpha0, Beta0)
    def Estep_PX(self, Nu, Omega):
        return Estep_PX(self.Mu0, Nu, Omega, self.Alpha0, self.Beta0, self.Kappa0, self.Ncell, self.Nsample)

    # Expectation of log P(Y|X,F)
    def Estep_PY(self, Nu, Omega, Beta):
        return Estep_PY(self.Y, self.SigmaY, Nu, Omega, Beta, self.Ngene, self.Ncell, self.Nsample)

    # Expectation of log P(F)
    def Estep_PF(self, Beta):
        return -(np.sum(loggamma(self.Alpha)) - np.sum(loggamma(self.Alpha.sum(axis=1)))) + \
            np.sum((self.Alpha-1) * (digamma(Beta) - \
                np.tile(digamma(np.sum(Beta, axis=1))[:,np.newaxis], [1,self.Ncell])))

    # Expectation of log Q(X)
    def Estep_QX(self, Omega):
        return -self.Nsample*np.sum(np.log(Omega))

    # Expectation of log Q(F)
    def Estep_QF(self, Beta):
        return -(np.sum(loggamma(Beta)) - np.sum(loggamma(Beta.sum(axis=1))))+ \
            np.sum((Beta-1) * (digamma(Beta) - \
                np.tile(digamma(np.sum(Beta, axis=1))[:,np.newaxis], [1,self.Ncell]))
                )

    def grad_Nu(self, Nu, Omega, Beta): 
        # return Ngene by Ncell
        return grad_Nu(self.Y, self.SigmaY, Nu, Omega, Beta, self.Mu0, self.Alpha0, self.Beta0, self.Kappa0, self.Ngene, self.Ncell, self.Nsample, self.weight)

    def grad_Omega(self, Nu, Omega, Beta):
        # return Ngene by Ncell
        return grad_Omega(self.Y, self.SigmaY, Nu, Omega, Beta,
                          self.Mu0, self.Alpha0, self.Beta0, self.Kappa0, self.Ngene, self.Ncell, self.Nsample, self.weight)

    def g_Exp_Beta(self, Nu, Beta, B0):
        return g_Exp_Beta(Nu, Omega, Beta, B0, self.Ngene, self.Ncell, self.Nsample)

    def grad_Beta(self, Nu, Omega, Beta):
        # return Nsample by Ncell
        B0 = np.sum(self.Beta, axis=1)
        
        grad_PY = g_PY_Beta(Nu, Beta, Omega, self.Y, self.SigmaY, B0, self.Ngene, self.Ncell, self.Nsample)

        grad_PF = (self.Alpha-1)*polygamma(1,Beta) - \
            np.tile(np.sum((self.Alpha-1)*np.tile(polygamma(1,B0)[:,np.newaxis], [1,self.Ncell]), axis=1)[:,np.newaxis], [1,self.Ncell])

        grad_QF = (Beta-1)*polygamma(1, Beta) - \
            np.tile(np.sum((Beta - 1) * np.tile(polygamma(1, B0)[:,np.newaxis], [1,self.Ncell]), axis=1)[:,np.newaxis], [1,self.Ncell])

        return grad_PY + grad_PF * np.sqrt(self.Ngene / self.Ncell) - grad_QF * np.sqrt(self.Ngene / self.Ncell)

    # E step
    def E_step(self, Nu, Beta, Omega):
        PX = self.Estep_PX(Nu, Omega) * (1/self.weight)
        PY = self.Estep_PY(Nu, Omega, Beta)
        PF = self.Estep_PF(Beta) * np.sqrt(self.Ngene / self.Ncell)
        QX = self.Estep_QX(Omega) * (1/self.weight)
        QF = self.Estep_QF(Beta) * np.sqrt(self.Ngene / self.Ncell)

        return PX+PY+PF-QX-QF     

    def Optimize(self):
            
            # loss function
        def loss(params):
            Nu = params[0:self.Ncell*self.Ngene*self.Nsample].reshape(self.Nsample, self.Ngene, self.Ncell)
            Omega = params[self.Ncell*self.Ngene*self.Nsample:(self.Ncell*self.Ngene*self.Nsample + self.Ngene*self.Ncell)].reshape(self.Ngene, self.Ncell)
            Beta = params[(self.Ncell*self.Ngene*self.Nsample + self.Ngene*self.Ncell):(self.Ncell*self.Ngene*self.Nsample + \
                    self.Ngene*self.Ncell + self.Nsample*self.Ncell)].reshape(self.Nsample, self.Ncell)

            if self.Fix_par['Nu']:
                Nu = self.Nu
            if self.Fix_par['Beta']:
                Beta = self.Beta
            if self.Fix_par['Omega']:
                Omega = self.Omega

            return -self.E_step(Nu, Beta, Omega)

        # gradient function
        def grad(params):
            Nu = params[0:self.Ncell*self.Ngene*self.Nsample].reshape(self.Nsample, self.Ngene, self.Ncell)
            Omega = params[self.Ncell*self.Ngene*self.Nsample:(self.Ncell*self.Ngene*self.Nsample + self.Ngene*self.Ncell)].reshape(self.Ngene, self.Ncell)
            Beta = params[(self.Ncell*self.Ngene*self.Nsample + self.Ngene*self.Ncell):(self.Ncell*self.Ngene*self.Nsample + \
                    self.Ngene*self.Ncell + self.Nsample*self.Ncell)].reshape(self.Nsample, self.Ncell)
            
            if self.Fix_par['Nu']:
                g_Nu = np.zeros(Nu.shape)
            else:
                g_Nu = -self.grad_Nu(Nu, Omega, Beta)
            
            if self.Fix_par['Omega']:
                g_Omega = np.zeros(Omega.shape)
            else:
                g_Omega = -self.grad_Omega(Nu, Omega, Beta)
            
            if self.Fix_par['Beta']:
                g_Beta = np.zeros(Beta.shape)
            else:
                g_Beta = -self.grad_Beta(Nu, Omega, Beta)

            g = np.concatenate((g_Nu.flatten(), g_Omega.flatten(), g_Beta.flatten()))

            return g

        # Perform Optimization
        Init = np.concatenate((self.Nu.flatten(), self.Omega.flatten(), self.Beta.flatten()))
        bounds = [(-np.inf, np.inf) if i < (self.Ncell*self.Ngene*self.Nsample) else (0.0000001, 100) for i in range(len(Init))]

        out = scipy.optimize.minimize(
                fun = loss, x0 = Init, bounds = bounds, jac = grad,
                options = {'disp': False},
                method='L-BFGS-B')

        params = out.x
    
        self.Nu = params[0:self.Ncell*self.Ngene*self.Nsample].reshape(self.Nsample, self.Ngene, self.Ncell)
        self.Omega = params[self.Ncell*self.Ngene*self.Nsample:(self.Ncell*self.Ngene*self.Nsample + self.Ngene*self.Ncell)].reshape(self.Ngene, self.Ncell)
        self.Beta = params[(self.Ncell*self.Ngene*self.Nsample + self.Ngene*self.Ncell):(self.Ncell*self.Ngene*self.Nsample + \
                        self.Ngene*self.Ncell + self.Nsample*self.Ncell)].reshape(self.Nsample, self.Ncell)

        self.log = out.success      
     
    # Reestimation of Nu at specific weight
    def Reestimate_Nu(self,weight=100):
        self.weight=weight
        self.Optimize()
        return self
   
    def Check_health(self):
        # check if optimization is done
        if not hasattr(self, 'log'):
            warnings.warn("No optimization is not done yet", Warning, stacklevel=2)

        # check values in hyperparameters
        if not np.all(np.isfinite(self.Y)):
            warnings.warn('non-finite values detected in bulk gene expression data (Y).', Warning, stacklevel=2)
            
        if np.any(self.Y < 0):
            warnings.warn('Negative expression levels were detected in bulk gene expression data (Y).', Warning, stacklevel=2)

        if np.any(self.Alpha <= 0):
            warnings.warn('Zero or negative values in Alpha', Warning, stacklevel=2)

        if np.any(self.Beta <= 0):
            warnings.warn('Zero or negative values in Beta', Warning, stacklevel=2)
 
        if np.any(self.Alpha0 <= 0):
            warnings.warn('Zero or negative values in Alpha0', Warning, stacklevel=2)
        
        if np.any(self.Beta0 <= 0):
            warnings.warn('Zero or negative values in Beta0', Warning, stacklevel=2)
        
        if np.any(self.Kappa0 <= 0):
            warnings.warn('Zero or negative values in Kappa0', Warning, stacklevel=2)

    def Update_Alpha(self, Expected=None, Temperature=None):# if Expected fraction is given, that part will be fixed
        # Updating Alpha
        Fraction = self.ExpF(self.Beta)
        if Expected is not None:  # Reflect the expected values
            # expectaion can be a diction (with two keys; Group and Expectation) or just a matrix
            if type(Expected) is dict:
                if "Group" in Expected:  # Group (Ngroup by Nctype matrix) indicates a group of cell types with known collective fraction
                    Group = Expected['Group']
                else:
                    Group = np.identity(Expected['Expectation'].shape[1])
                Expected = Expected['Expectation']
            else:
                Group = np.identity(Expected.shape[1])

            if self.Beta.shape[0] != Expected.shape[0] or self.Beta.shape[1] != Group.shape[1]:
                raise ValueError('Pre-determined fraction is in wrong shape (should be Nsample by Ncelltype)')

            # rescale the fraction to meet the expected fraction
            for sample in range(self.Nsample):
                IndG = np.where(~np.isnan(Expected[sample,:]))[0]
                IndCells = []

                for group in IndG:
                    IndCell = np.where(Group[group,:] == 1)[0]
                    Fraction[sample, IndCell] = Fraction[sample, IndCell] / np.sum(Fraction[sample,IndCell])  # make fraction sum to one for the group
                    Fraction[sample, IndCell] = Fraction[sample, IndCell] * Expected[sample, group]  # assign determined fraction for the group
                    IndCells = IndCells + list(IndCell)

                IndNan = np.setdiff1d(np.array(range(Group.shape[1])), np.array(IndCells))
                Fraction[sample, IndNan] = Fraction[sample, IndNan] / np.sum(Fraction[sample, IndNan])  # normalize the rest of cell types (sum to one)
                Fraction[sample, IndNan] = Fraction[sample, IndNan] * (1-np.sum(Expected[sample, IndG]))  # assign determined fraction for the rest of cell types

        if Temperature is not None:
            self.Alpha = Temperature * Fraction
        else:
            for sample in range(self.Nsample):
                self.Alpha[sample,:] = Fraction[sample,:] * np.sum(self.Beta[sample,:])

    def Update_Alpha_Group(self, Expected=None, Temperature=None):# if Expected fraction is given, that part will be fixed
        # Updating Alpha
        AvgBeta = np.mean(self.Beta, 0)
        Fraction_Avg = AvgBeta / np.sum(AvgBeta)

        if Expected is not None:  # Reflect the expected values
            # expectaion can be a diction (with two keys; Group and Expectation) or just a matrix
            if type(Expected) is dict:
                if "Group" in Expected:  # Group (Ngroup by Nctype matrix) indicates a group of cell types with known collective fraction
                    Group = Expected['Group']
                else:
                    Group = np.identity(Expected['Expectation'].shape[1])
                Expected = Expected['Expectation']
            else:
                Group = np.identity(Expected.shape[1])

            if self.Beta.shape[0] != Expected.shape[0] or self.Beta.shape[1] != Group.shape[1]:
                raise ValueError('Pre-determined fraction is in wrong shape (should be Nsample by Ncelltype)')

            # rescale the fraction to meet the expected fraction
            for sample in range(self.Nsample):
                Fraction = np.copy(Fraction_Avg)
                IndG = np.where(~np.isnan(Expected[sample,:]))[0]
                IndCells = []
                
                for group in IndG:
                    IndCell = np.where(Group[group,:] == 1)[0]
                    Fraction[IndCell] = Fraction[IndCell] / np.sum(Fraction[IndCell])  # make fraction sum to one for the group
                    Fraction[IndCell] = Fraction[IndCell] * Expected[sample, group]  # assign determined fraction for the group
                    IndCells = IndCells + list(IndCell)
                    
                IndNan = np.setdiff1d(np.array(range(Group.shape[1])), np.array(IndCells))
                Fraction[IndNan] = Fraction[IndNan] / np.sum(Fraction[IndNan])  # normalize the rest of cell types (sum to one)
                Fraction[IndNan] = Fraction[IndNan] * (1-np.sum(Expected[sample, IndG]))  # assign determined fraction for the rest of cell types
            
                AlphaSum = np.sum(AvgBeta[IndNan])/ np.sum(Fraction[IndNan])
                self.Alpha[sample, :] = Fraction * AlphaSum
        else:
            for sample in range(self.Nsample):
                self.Alpha[sample,:] = AvgBeta

    def Update_SigmaY(self, SampleSpecific=False):
        Var = VarQ(self.Nu, self.Beta, self.Omega, self.Ngene, self.Ncell, self.Nsample)
        Exp = ExpQ(self.Nu, self.Beta, self.Omega, self.Ngene, self.Ncell, self.Nsample)
        
        a = Var / Exp / Exp
        b = np.square((self.Y-np.log(Exp)) - 0.5 * a)

        if SampleSpecific:
            self.SigmaY = np.sqrt(a+b)
        else:  # shared in all samples
            self.SigmaY = np.tile(np.mean(np.sqrt(a+b), axis=1)[:,np.newaxis], [1,self.Nsample])

            
def Optimize(logY, SigmaY, Mu0, Alpha, Alpha0, Beta0, Kappa0, Nu_Init, Omega_Init, Nsample, Ncell, Init_Fraction):
    Beta_Init = np.random.gamma(shape=1, size=(Nsample, Ncell)) * 0.1 + t(Init_Fraction) * 10
    obs = OncoBLADE(logY, SigmaY, Mu0, Alpha, Alpha0, Beta0, Kappa0,
            Nu_Init, Omega_Init, Beta_Init, fix_Nu=True, fix_Omega=True)
    obs.Optimize()
    
    obs.Fix_par['Nu'] = False
    obs.Fix_par['Omega'] = False
    obs.Optimize()
    return obs


def NuSVR_job(X, Y, Nus, sample):
    X = np.exp(X) - 1
    sols = [NuSVR(kernel='linear', nu=nu).fit(X,Y[:, sample]) for nu in Nus]
    RMSE = [mse(sol.predict(X), Y[:, sample]) for sol in sols]
    return sols[np.argmin(RMSE)]


def SVR_Initialization(X, Y, Nus, Njob=1, fsel=0):
    Ngene, Nsample = Y.shape
    Ngene, Ncell = X.shape
    SVRcoef = np.zeros((Ncell, Nsample))
    Selcoef = np.zeros((Ngene, Nsample))

    sols = Parallel(n_jobs=Njob, verbose=10)(
            delayed(NuSVR_job)(X, Y, Nus, i)
            for i in range(Nsample)
            )

    for i in range(Nsample):
        Selcoef[sols[i].support_,i] = 1
        SVRcoef[:,i] = np.maximum(sols[i].coef_,0)

    Init_Fraction = SVRcoef
    for i in range(Nsample):
        Init_Fraction[:,i] = Init_Fraction[:,i]/np.sum(SVRcoef[:,i])

    if fsel > 0:
        Ind_use = Selcoef.sum(1) > Nsample * fsel
        print( "SVM selected " + str(Ind_use.sum()) + ' genes out of ' + str(len(Ind_use)) + ' genes')
    else:
        print("No feature filtering is done (fsel = 0)")
        Ind_use = np.ones((Ngene)) > 0

    return Init_Fraction, Ind_use   


def Iterative_Optimization(X, stdX, Y, Alpha, Alpha0, Kappa0, SY, Rep, Init_Fraction, Init_Trust=10,
                           Expected=None, iter=100, minDiff=10e-4, TempRange=None, Update_SigmaY=False):
    Ngene, Nsample = Y.shape
    Ncell = X.shape[1]

    Mu0 = X
    logY = np.log(Y+1)
    SigmaY = np.tile(np.std(logY,1)[:,np.newaxis], [1,Nsample]) * SY + 0.1
    Omega_Init = stdX
    Beta0 = Alpha0 * np.square(stdX)
    
    Nu_Init = np.zeros((Nsample, Ngene, Ncell))
    for i in range(Nsample):
        Nu_Init[i,:,:] = X

    #  Optimization without given Temperature
    Beta_Init = np.random.gamma(shape=1, size=(Nsample, Ncell)) + t(Init_Fraction) * Init_Trust
    obj = OncoBLADE(logY, SigmaY, Mu0, Alpha, Alpha0, Beta0, Kappa0,
                    Nu_Init, Omega_Init, Beta_Init)

    if TempRange is None:
        obj.Check_health()
        obj_func = [None] * iter
        obj_func[0] = obj.E_step(obj.Nu, obj.Beta, obj.Omega)

        for i in range(1,iter):
            obj.Optimize()
            obj.Update_Alpha_Group(Expected=Expected)
            if Update_SigmaY:
                obj.Update_SigmaY()
            obj_func[i] = obj.E_step(obj.Nu, obj.Beta, obj.Omega)

            # Check convergence
            if np.abs(obj_func[i] - obj_func[i-1]) < minDiff:
                break
       
    else:  #  Optimization with Temperature
        obj.Check_health()
        obj_func = [None] * len(TempRange)
        obj_func[0] = obj.E_step(obj.Nu, obj.Beta, obj.Omega)

        for i, temp in zip(range(1,len(TempRange)), TempRange):
            obj.Optimize()
            obj.Update_Alpha_Group(Expected=Expected, Temperature=temp)
            if Update_SigmaY:
                obj.Update_SigmaY()

            obj_func[i] = obj.E_step(obj.Nu, obj.Beta, obj.Omega)

            # Check convergence
            if np.abs(obj_func[i] - obj_func[i-1]) < minDiff:
                break

    return obj, obj_func, Rep


def Framework_Iterative(X, stdX, Y, Ind_Marker=None,  # all samples will be used
        Alpha=1, Alpha0=0.1, Kappa0=1, sY=1,
        Nrep=3, Njob=10, Nrepfinal=10, fsel=0, Update_SigmaY=False, Init_Trust= 10,
        Expectation=None, Temperature=None, IterMax=100):

    args = locals()
    Ngene, Nsample = Y.shape
    Ncell = X.shape[1]
    
    if Ind_Marker is None:
        Ind_Marker = [True] * Ngene
    
    X_small = X[Ind_Marker,:]
    Y_small = Y[Ind_Marker,:]
    stdX_small = stdX[Ind_Marker,:]

    Nmarker = Y_small.shape[0]
    Nsample_small = Y_small.shape[1]

    if Nmarker < Ngene:
        print("start optimization using marker genes: " + str(Nmarker) +\
            " genes out of " + str(Ngene) + " genes.")
    else:
        print("all of " + str(Ngene) + " genes are used for optimization.")
      
    print('Initialization with Support vector regression')
    Init_Fraction, Ind_use = SVR_Initialization(X_small, Y_small, Njob=Njob, Nus=[0.25, 0.5, 0.75])

    start = time.time()
    if Temperature is None or Temperature is False: #  Optimization without the temperature
        outs = Parallel(n_jobs=Njob, verbose=10)(
            delayed(Iterative_Optimization)(X_small[Ind_use,:], stdX_small[Ind_use,:], Y_small[Ind_use,:], 
                Alpha, Alpha0, Kappa0, sY, rep, Init_Fraction, Expected=Expectation, Init_Trust=Init_Trust, iter=IterMax,  
                Update_SigmaY = Update_SigmaY)
                for rep in range(Nrep)
            )
        
        ## Final OncoBLADE results
        outs, convs, Reps = zip(*outs)
        cri = [obj.E_step(obj.Nu, obj.Beta, obj.Omega) for obj in outs]
        out = outs[np.nanargmax(cri)]
        conv = convs[np.nanargmax(cri)]
    else:
        if Temperature is True:
            Temperature = [1, 100]
        else:
            if len(Temperature) != 2:
                raise ValueError('Temperature has to be either None, True or list of 2 temperature values (minimum and maximum temperatures)')
            if Temperature[1] < Temperature[0]:
                raise ValueError('A lower maximum temperature than minimum temperature is given')
        outs = Parallel(n_jobs=Njob, verbose=10)(
            delayed(Iterative_Optimization)(X_small[Ind_use,:], stdX_small[Ind_use,:], Y_small[Ind_use,:], 
                Alpha, Alpha0, Kappa0, sY, rep, Init_Fraction, Expected=Expectation, Init_Trust=Init_Trust, 
                    TempRange=np.linspace(Temperature[0], Temperature[1], IterMax),
                    Update_SigmaY = Update_SigmaY)
                for rep in range(Nrep)
            )

        ## Final OncoBLADE results
        outs, convs, Reps = zip(*outs)
        cri = [obj.E_step(obj.Nu, obj.Beta, obj.Omega) for obj in outs]
        out = outs[np.nanargmax(cri)]
        conv = convs[np.nanargmax(cri)]

    return out, conv, zip(outs, cri), args


def Parallel_Purification(obj, iter=1000, minDiff=10e-4, Update_SigmaY=False):
    obj.Check_health()
    obj_func = [float('nan')] * iter
    obj_func[0] = obj.E_step(obj.Nu, obj.Beta, obj.Omega)
    for i in range(1,iter):
        obj.Reestimate_Nu()
        if Update_SigmaY:
                obj.Update_SigmaY()
        obj_func[i] = obj.E_step(obj.Nu, obj.Beta, obj.Omega)

        # Check convergence
        if np.abs(obj_func[i] - obj_func[i-1]) < minDiff:
            break

    return obj, obj_func


# Purify all genes in parralel using fixed Beta
def Purify_AllGenes(OncoBLADE_object, Mu, Omega, Y, Ncores):
    obj = OncoBLADE_object['final_obj']
    Ngene, Nsample = Y.shape
    Ncell = Mu.shape[1]
    logY = np.log(Y+1)
    SigmaY = np.tile(np.std(logY,1)[:,np.newaxis], [1,Nsample]) * OncoBLADE_object['outs']['sY'] + 0.1
    Beta0 = OncoBLADE_object['outs']['Alpha0'][0] * np.square(Omega)
    Nu_Init = np.zeros((Nsample, Ngene, Ncell))
    for i in range(Nsample):
        Nu_Init[i,:,:] = Mu

    # Fetch objs per gene
    Ngene_total = Mu.shape[0]
    objs = []
    for ix in range(Ngene_total):
        objs.append(OncoBLADE(
            Y = np.atleast_2d(logY[ix,:]),
            SigmaY = np.atleast_2d(SigmaY[ix,:]),
            Mu0 = np.atleast_2d(Mu[ix,:]),
            Alpha = obj.Alpha,
            Alpha0 = OncoBLADE_object['outs']['Alpha0'][0],
            Beta0 = np.atleast_2d(Beta0[ix,:]),
            Kappa0 = OncoBLADE_object['outs']['Kappa0'],
            Nu_Init = np.reshape(np.atleast_3d(Nu_Init[:,ix,:]), (Nsample,1,Ncell)), ## Reshape else will be (Nsample,Ncell,Ngene/1)
            Omega_Init = np.atleast_2d(Omega[ix,:]),
            Beta_Init = obj.Beta,
            fix_Beta=True))

    outs = Parallel(n_jobs=Ncores, verbose=10)(
                delayed(Parallel_Purification)(obj)
                    for obj in objs
                )
       
    objs, obj_func = zip(*outs)
    ## sum ofv over all genes
    obj_func = np.sum(obj_func, axis = 0)
    logs = []
    ## Combine results from all genes
    for i,obj in enumerate(objs):
        logs.append(obj.log)
        if i==0:
            Y = objs[0].Y
            SigmaY = objs[0].SigmaY
            Mu0= objs[0].Mu0
            Alpha = objs[0].Alpha
            Alpha0 = objs[0].Alpha0
            Beta0 = objs[0].Beta0
            Kappa0 = objs[0].Kappa0
            Nu_Init = objs[0].Nu
            Omega_Init = objs[0].Omega
            Beta_Init = objs[0].Beta
        else:    
            Y = np.concatenate((Y,obj.Y))
            SigmaY = np.concatenate((SigmaY,obj.SigmaY))
            Mu0= np.concatenate((Mu0,obj.Mu0))
            Alpha0 = np.concatenate((Alpha0,obj.Alpha0))
            Beta0 = np.concatenate((Beta0,obj.Beta0))
            Kappa0 = np.concatenate((Kappa0,obj.Kappa0))
            Nu_Init = np.concatenate((Nu_Init,obj.Nu), axis = 1)
            Omega_Init = np.concatenate((Omega_Init,obj.Omega))
            
    ## Create final merged OncoBLADE obj to return
    obj = OncoBLADE(Y, SigmaY, Mu0, Alpha, Alpha0, Beta0, Kappa0, Nu_Init, Omega_Init, Beta_Init, fix_Beta =True)
    obj.log = logs
    
    return obj, obj_func
