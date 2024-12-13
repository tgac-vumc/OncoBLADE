# New imports
import torch
import torch.special


# Old imports
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

from joblib import Parallel, delayed, parallel_backend
import itertools
import time
import math

import warnings

from timeit import default_timer as timer


# The below function is a decorator to cast numpy arrays to torch tensors
def cast_args_to_torch(func):
    def wrapper(*args, **kwargs):
        # Get the first argument (assumed to be `self` if this is a class method)
        self = args[0]

        device = self.device
        # Check if the `self` object has a `device` attribute
        #if hasattr(self, 'device'):
        #    device = self.device
        #else:
        #    # Default to CPU if no device attribute is found
        #    device = torch.device("cpu")

        # Convert NumPy arrays in args to torch tensors and move to the correct device
        new_args = [torch.tensor(arg, dtype=torch.float64, device=device) if isinstance(arg, np.ndarray) else arg for arg in args]

        # Convert NumPy arrays in kwargs to torch tensors and move to the correct device
        new_kwargs = {k: torch.tensor(v, dtype=torch.float64, device=device) if isinstance(v, np.ndarray) else v for k, v in kwargs.items()}

        # Call the original function with converted arguments
        return func(*new_args, **new_kwargs)

    return wrapper


def ExpF_C(Beta):
    # Compute normalized Beta across the second axis (cells).
    return Beta / torch.sum(Beta, dim=1, keepdim=True)


def ExpQ_C(Nu, Beta, Omega):
    # Expected value of F (Nsample by Ncell)
    ExpB = ExpF_C(Beta)

    # Element-wise exponential and broadcasting of Nu and Omega
    out = torch.sum(ExpB.unsqueeze(1) * torch.exp(Nu + 0.5 * Omega**2), dim=-1)
    return out.T


def VarQ_C(Nu, Beta, Omega, Ngene, Ncell, Nsample):
    # Nu has shape (Nsample, Ngene, Ncell)
    # Beta has shape (Nsample, Ncell)
    # Omega has shape (Ngene, Ncell)

    # Sum Beta over the second axis (Nsample)
    B0 = torch.sum(Beta, dim=1, keepdim=True)  # Shape: (Nsample, 1)

    # Nsample by Ncell (ExpF_C is assumed to calculate the exp of Beta elements)
    Btilda = ExpF_C(Beta)  # Shape: (Nsample, Ncell)

    # Variance of B (vectorized)
    VarB = Btilda * (1 - Btilda) / (B0 + 1)  # Shape: (Nsample, Ncell)

    # Exponential terms calculation with broadcasting
    exp_2Nu = torch.exp(2 * Nu + 2 * Omega.unsqueeze(0).square()) # shape: (Nsample, Ngene, Ncell)
    exp_2Nu_half = torch.exp(2 * Nu + Omega.unsqueeze(0).square()) # shape: (Nsample, Ngene, Ncell)

    # Extend VarB and Btilda for broadcasting in the computation
    VarB_expanded = VarB.unsqueeze(1)  # shape: (Nsample, 1, Ncell)
    Btilda_expanded = Btilda.unsqueeze(1)  # shape: (Nsample, 1, Ncell)

    # Calculation of the variance term with appropriate broadcasting and summation
    VarTerm = torch.sum(
        exp_2Nu * (VarB_expanded + Btilda_expanded.square()) - exp_2Nu_half * Btilda_expanded.square(),
        dim=-1  # sum over the Ncell dimension
    ).T

    # Ngene by Nsample by Ncell by Ncell
    CovX = torch.exp(
        Nu.unsqueeze(2) + Nu.unsqueeze(3) +
        0.5 * (Omega**2).unsqueeze(0).unsqueeze(3) + 0.5 * (Omega**2).unsqueeze(0).unsqueeze(2)
    )

    # Permute to match (Ngene, Nsample, Ncell, Ncell)
    CovX = CovX.permute(1, 0, 2, 3)

    # Covariance of B (vectorized)
    CovB = -torch.einsum('ij,ik->ijk', Btilda, Btilda) / (1 + B0.unsqueeze(2)).to(Beta.device)  # Shape: (Nsample, Ncell, Ncell)

    # Create a mask to exclude diagonal elements where l == k
    # mask = torch.ones((Ncell, Ncell), dtype=torch.bool)
    # mask.fill_diagonal_(0)

    # Mask needs to be applied properly to match dimensions for broadcasting
    CovTerm = torch.zeros((Ngene, Nsample)).to(Beta.device)
    for l in range(Ncell):
        for k in range(Ncell):
            if l != k:
                # Extract slices for CovX and CovB where l != k and use broadcasting to perform multiplication
                # CovX[:, :, l, k] has shape (Ngene, Nsample)
                # CovB[:, l, k] has shape (Nsample,) but needs to be unsqueezed to broadcast correctly
                CovTerm += CovX[:, :, l, k] * CovB[:, l, k].unsqueeze(0)
    return CovTerm + VarTerm


def Estep_PY_C(Y, SigmaY, Nu, Omega, Beta, Ngene, Ncell, Nsample):
    Var = VarQ_C(Nu, Beta, Omega, Ngene, Ncell, Nsample)
    Exp = ExpQ_C(Nu, Beta, Omega)

    a = Var / Exp / Exp

    return torch.sum(
            -0.5 / torch.square(SigmaY) * (a + torch.square((Y-torch.log(Exp)) - 0.5 * a))
            )


def Estep_PX_C(Mu0, Nu, Omega, Alpha0, Beta0, Kappa0, Ncell, Nsample):
    NuExp = torch.sum(Nu, 0) / Nsample  # expected Nu, Ngene by Ncell
    AlphaN = Alpha0 + 0.5 * Nsample  # Posterior Alpha

    ExpBetaN = Beta0 + (Nsample - 1) / 2 * torch.square(Omega) + \
               Kappa0 * Nsample / (2 * (Kappa0 + Nsample)) * (torch.square(Omega) / Nsample + torch.square(NuExp - Mu0))

    # Vectorize the for loop
    ExpBetaN = ExpBetaN + 0.5 * torch.sum(torch.square(Nu - NuExp.unsqueeze(0)), dim=0)

    return torch.sum(-AlphaN * torch.log(ExpBetaN))


def grad_Nu_C(Y, SigmaY, Nu, Omega, Beta, Mu0, Alpha0, Beta0, Kappa0, Ngene, Ncell, Nsample, weight):
    # gradient of PX (first term)
    AlphaN = Alpha0 + Nsample * 0.5
    NuExp = torch.sum(Nu, dim=0) / Nsample

    Diff = torch.zeros((Ngene, Ncell), dtype=Nu.dtype, device=Nu.device)
    ExpBetaN = Beta0 + (Nsample - 1) / 2 * torch.square(Omega) + \
               Kappa0 * Nsample / (2 * (Kappa0 + Nsample)) * (torch.square(Omega) / Nsample + torch.square(NuExp - Mu0))

    Diff = torch.mean(Nu - NuExp, dim=0)  # Vectorized form of computing the mean difference
    ExpBetaN += 0.5 * torch.sum(torch.square(Nu - NuExp), dim=0)  # Vectorized summation over the sample dimension

    Nominator = Nu - NuExp - Diff + Kappa0 / (Kappa0 + Nsample) * (NuExp - Mu0)

    grad_PX = -AlphaN * Nominator / ExpBetaN

    # gradient of PY (second term)
    B0 = torch.sum(Beta, dim=1)  # Nsample
    Btilda = ExpF_C(Beta)   # Nsample by Ncell

    Exp = ExpQ_C(Nu, Beta, Omega)  # Ngene by Nsample
    Var = VarQ_C(Nu, Beta, Omega, Ngene, Ncell, Nsample)  # Ngene by Nsample

    # Covariance calculations
    CovB = - torch.einsum('il,ik->ilk', Btilda, Btilda) / (1 + B0[:, None, None])

    ExpX = torch.exp(Nu + 0.5 * torch.square(Omega))

    # Vectorized CovX computation
    ExpX_reshaped_l = ExpX.unsqueeze(3)  # Shape becomes (Nsample, Ngene, Ncell, 1)
    ExpX_reshaped_k = ExpX.unsqueeze(2)  # Shape becomes (Nsample, Ngene, 1, Ncell)

    CovX = ExpX_reshaped_l * ExpX_reshaped_k  # Element-wise multiplication, resulting in (Nsample, Ngene, Ncell, Ncell)
    CovX = CovX.permute(1, 0, 2, 3)  # Rearrange to match the desired (Ngene, Nsample, Ncell, Ncell) shape

    CovTerm = torch.zeros((Ngene, Ncell, Nsample), dtype=Nu.dtype, device=Nu.device)
    for l in range(Ncell):
        for k in range(Ncell):
            if l != k:
                CovTerm[:, l, :] += 2 * CovX[:, :, l, k] * CovB[:, l, k]
                # for i in range(Nsample):
                #     CovTerm[:, l, i] += 2 * CovX[:, i, l, k] * CovB[i, l, k]

    g_Exp = ExpX * Btilda.unsqueeze(1)  # This ensures broadcasting across the gene dimension
    g_Exp = g_Exp.permute(1, 2, 0)      # Permute to match the shape (Ngene, Ncell, Nsample)

    VarX = torch.exp(2 * Nu + 2 * torch.square(Omega))

    VarB = Btilda * (1 - Btilda)  # Element-wise multiplication
    VarB /= (B0 + 1).unsqueeze(1)  # Broadcasting the division over the Ncell dimension

    # First term (broadcast VarB + Btilda correctly)
    first_term = 2 * VarX * (VarB + torch.square(Btilda)).unsqueeze(1)

    # Extract diagonal elements from CovX (Ngene, Nsample, Ncell)
    diag_CovX = torch.diagonal(CovX, dim1=2, dim2=3)

    # Multiply by torch.square(Btilda) with broadcasting across Ngene
    second_term = 2 * diag_CovX * torch.square(Btilda).unsqueeze(0)

    # Final g_Var computation
    g_Var = first_term.permute(1, 2, 0) - second_term.permute(0, 2, 1)
    g_Var += CovTerm

    a = (g_Var - 2 * g_Exp / Exp.unsqueeze(1) * Var.unsqueeze(1)) / torch.pow(Exp.unsqueeze(1), 2)

    Diff = Y - torch.log(Exp) - Var / (2 * torch.square(Exp))

    # Vectorized operation: Broadcasting Diff across Ncell
    b = -Diff.unsqueeze(1) * (2 * g_Exp / Exp.unsqueeze(1) + a)

    # Compute 0.5 / torch.square(SigmaY) and expand it across the Ncell dimension
    scaling_factor = 0.5 / torch.square(SigmaY).unsqueeze(1)  # Shape: (Ngene, 1, Nsample)

    # Sum a and b and perform the element-wise multiplication with the scaling factor
    sum_ab = a + b  # Shape: (Ngene, Ncell, Nsample)

    # Apply the scaling factor and transpose
    grad_PY = -(scaling_factor * sum_ab).permute(2, 0, 1)  # Transpose to (Nsample, Ngene, Ncell)
    return grad_PX * (1 / weight) + grad_PY


def grad_Omega_C(Y, SigmaY, Nu, Omega, Beta, Mu0, Alpha0, Beta0, Kappa0, Ngene, Ncell, Nsample, weight):
    # gradient of PX (first term)
    AlphaN = Alpha0 + Nsample * 0.5
    NuExp = torch.sum(Nu, dim=0) / Nsample
    ExpBetaN = Beta0 + (Nsample - 1) / 2 * torch.square(Omega) + \
               Kappa0 * Nsample / (2 * (Kappa0 + Nsample)) * (torch.square(Omega) / Nsample + torch.square(NuExp - Mu0))

    ExpBetaN += 0.5 * torch.sum(torch.square(Nu - NuExp), dim=0)  # Vectorized loop

    Nominator = -AlphaN * (Nsample - 1) * Omega + Kappa0 / (Kappa0 + Nsample) * Omega
    grad_PX = Nominator / ExpBetaN

    # gradient of PY (second term)
    B0 = torch.sum(Beta, dim=1)  # Nsample
    Btilda = ExpF_C(Beta)   # Nsample by Ncell

    Exp = ExpQ_C(Nu, Beta, Omega)  # Ngene by Nsample
    Var = VarQ_C(Nu, Beta, Omega, Ngene, Ncell, Nsample)  # Ngene by Nsample

    # Ngene by Nsample by Ncell by Ncell
    CovB = -torch.einsum('il,ik->ilk', Btilda, Btilda) / (1 + B0[:, None, None])

    ExpX = torch.exp(Nu)  # Nsample by Ngene by Ncell
    ExpX = ExpX * torch.exp(0.5 * torch.square(Omega))

    # Reshaped ExpX for CovX computation
    ExpX_reshaped_l = ExpX.unsqueeze(3)  # Shape becomes (Nsample, Ngene, Ncell, 1)
    ExpX_reshaped_k = ExpX.unsqueeze(2)  # Shape becomes (Nsample, Ngene, 1, Ncell)

    # CovX is computed by multiplying reshaped tensors (Nsample, Ngene, Ncell, Ncell)
    CovX = ExpX_reshaped_l * ExpX_reshaped_k
    CovX = CovX.permute(1, 0, 2, 3)  # Rearranged to (Ngene, Nsample, Ncell, Ncell)

    # CovTerm computation
    CovTerm = torch.zeros((Ngene, Ncell, Nsample), dtype=Nu.dtype, device=Nu.device)
    for l in range(Ncell):
        for k in range(Ncell):
            if l != k:
                CovTerm[:, l, :] += 2 * CovX[:, :, l, k] * CovB[:, l, k].unsqueeze(0) * Omega[:, l].unsqueeze(1)

    # g_Exp computation (Ngene, Ncell, Nsample)
    g_Exp = (ExpX * Btilda.unsqueeze(1) * Omega.unsqueeze(0)).permute(1, 2, 0)

    # VarX computation (Nsample by Ngene by Ncell)
    VarX = torch.exp(2 * Nu + 2 * torch.square(Omega))

    VarB = Btilda * (1 - Btilda)
    VarB /= (B0 + 1).unsqueeze(1)

    # First term: Apply VarX and broadcast the scalar part (before multiplying Omega)
    first_term = 2 * VarX * (VarB + torch.square(Btilda)).unsqueeze(1)

    # Extract diagonal elements from CovX (Ngene, Nsample, Ncell)
    diag_CovX = torch.diagonal(CovX, dim1=2, dim2=3)

    # Second term: Use diagonal of CovX and Btilda squared
    second_term = 2 * diag_CovX * torch.square(Btilda).unsqueeze(0)

    # Now permute the first and second terms to have shape (Ngene, Ncell, Nsample)
    first_term_permuted = first_term.permute(1, 2, 0)
    second_term_permuted = second_term.permute(0, 2, 1)

    # Now, apply Omega[:, c] correctly after permuting
    first_term_with_Omega = 2 * Omega.unsqueeze(2) * first_term_permuted
    second_term_with_Omega = Omega.unsqueeze(2) * second_term_permuted

    # Final g_Var computation
    g_Var = first_term_with_Omega - second_term_with_Omega

    # Add CovTerm (which already has the correct shape)
    g_Var += CovTerm

    # a computation (Ngene, Ncell, Nsample)
    a = (g_Var - 2 * g_Exp * Var.unsqueeze(1) / Exp.unsqueeze(1)) / torch.square(Exp).unsqueeze(1)

    # Diff computation (Ngene, Nsample)
    Diff = Y - torch.log(Exp) - Var / (2 * torch.square(Exp))

    # b computation (Ngene, Ncell, Nsample)
    b = -Diff.unsqueeze(1) * (2 * g_Exp / Exp.unsqueeze(1) + a)

    # grad_PY computation (Ngene, Ncell)
    grad_PY = torch.sum(-0.5 / torch.square(SigmaY).unsqueeze(1) * (a + b), dim=2)

    # Q(X) term (fourth term)
    grad_QX = - Nsample / Omega
    return grad_PX * (1 / weight) + grad_PY - grad_QX * (1 / weight)


def g_Exp_Beta_C(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample):
    # Compute exp(Nu) element-wise
    ExpX = torch.exp(Nu)  # Shape: (Nsample, Ngene, Ncell)
    # Apply the element-wise multiplication with Omega, which has shape (Ngene, Ncell)
    ExpX = ExpX * torch.exp(0.5 * torch.square(Omega)).unsqueeze(0)

    # B0mat computation (element-wise division)
    B0mat = Beta / torch.square(B0.unsqueeze(1))  # Shape: (Nsample, Ncell)

    # Transpose ExpX to match the original NumPy (Nsample, Ncell, Ngene)
    tExpX = ExpX.transpose(1, 2)  # Shape: (Nsample, Ncell, Ngene)

    # Perform dot product of B0mat and tExpX
    B0mat = torch.matmul(B0mat.unsqueeze(1), tExpX).squeeze(1)  # Shape: (Nsample, 1, Ngene)

    # Divide ExpX by the broadcasted B0 (shape should match (Nsample, Ngene, Ncell))
    g_Exp = (ExpX / B0.unsqueeze(1).unsqueeze(2))  # Shape: (Nsample, Ngene, Ncell)

    # Subtract B0mat which has shape (Nsample, Ngene), broadcast it over Ncell
    g_Exp = g_Exp - B0mat.unsqueeze(2)  # Shape: (Nsample, Ngene, Ncell)
    return g_Exp.permute(0,2,1)


def g_Var_Beta_C(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample):
    # Broadcasting B0
    B0Rep = B0.unsqueeze(1)

    # Computing aa and aaNotT
    aa = (B0Rep - Beta) * B0Rep * (B0Rep + 1) - (3 * B0Rep + 2) * Beta * (B0Rep - Beta)
    aa = aa / (torch.pow(B0Rep, 3) * torch.square(B0Rep + 1))
    aa += 2 * Beta * (B0Rep - Beta) / torch.pow(B0Rep, 3)

    aaNotT = Beta * B0Rep * (B0Rep + 1) - (3 * B0Rep + 2) * Beta * (B0Rep - Beta)
    aaNotT = aaNotT / (torch.pow(B0Rep, 3) * torch.square(B0Rep + 1))
    aaNotT += 2 * Beta * (-Beta) / torch.pow(B0Rep, 3)

    # ExpX2 computation
    ExpX2 = torch.exp(2 * Nu + 2 * torch.square(Omega))

    # g_Var computation (initial step)
    g_Var = torch.transpose(ExpX2, 1, 2) * aa.unsqueeze(2)

    # # Add aaNotT contributions excluding the diagonal, using the 'Omega' pattern
    for i in range(Ncell):
        for j in range(Ncell):
            if i != j:
                 g_Var[:, i, :] += ExpX2[:, :, j] * aaNotT[:, j].unsqueeze(-1)

    # Element-wise Beta computations
    B_B02 = Beta / torch.square(B0Rep)
    B0B0_1 = B0Rep * (B0Rep + 1)
    B2_B03 = torch.square(Beta) / torch.pow(B0Rep, 3)

    # ExpX computation
    ExpX = torch.exp(2 * Nu + torch.square(Omega))

    # Subtract B_B02 terms
    g_Var -= 2 * torch.transpose(ExpX, 1, 2) * B_B02.unsqueeze(2)

    Dot = torch.sum(B2_B03.unsqueeze(1) * ExpX, dim=2)

    # Add Dot contributions
    g_Var += 2 * Dot.unsqueeze(1)

    # CovX computation
    ExpX = torch.exp(Nu + 0.5 * torch.square(Omega))
    CovX = torch.einsum('sil,sik->silk', ExpX, ExpX)

    # gradCovB computation
    B03_2_B03_B0_1 = (3 * B0 + 2) / torch.pow(B0, 3) / torch.square(B0 + 1)
    gradCovB = torch.einsum('sl,sk->slk', Beta, Beta) * B03_2_B03_B0_1.unsqueeze(1).unsqueeze(1)

    # CovTerm1 and CovTerm2 computation using the 'Omega' pattern
    CovTerm1 = torch.zeros((Nsample, Ncell, Ncell, Ngene), dtype=Nu.dtype, device=Nu.device)
    CovTerm2 = torch.zeros((Nsample, Ncell, Ncell, Ngene), dtype=Nu.dtype, device=Nu.device)

    B_B0_1_B0B0_1 = Beta * (B0Rep + 1) / torch.square(B0B0_1)
    for l in range(Ncell):
        for k in range(Ncell):
            if l != k:
                CovTerm1[:, l, k, :] = gradCovB[:, l, k].unsqueeze(-1) * CovX[:, :, l, k]
                CovTerm2[:, l, k, :] = B_B0_1_B0B0_1[:, l].unsqueeze(-1) * CovX[:, :, l, k]

    # Final accumulation for g_Var
    g_Var += torch.sum(CovTerm1, dim=[1, 2]).unsqueeze(1)
    g_Var -= 2 * torch.sum(CovTerm2, dim=1)
    return g_Var


def g_PY_Beta_C(Nu, Beta, Omega, Y, SigmaY, B0, Ngene, Ncell, Nsample):
    # Assuming these are the pre-ported PyTorch functions
    Exp = ExpQ_C(Nu, Beta, Omega)  # Ngene by Nsample
    Var = VarQ_C(Nu, Beta, Omega, Ngene, Ncell, Nsample)  # Ngene by Nsample

    g_Exp = g_Exp_Beta_C(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample)  # Shape: (Nsample, Ncell, Ngene)
    g_Var = g_Var_Beta_C(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample)  # Shape: (Nsample, Ncell, Ngene)

    # Correcting shapes for Exp and Var
    Exp = Exp.unsqueeze(-1).permute(1, 2, 0)  # Shape: (Nsample, 1, Ngene)
    Var = Var.unsqueeze(-1).permute(1, 2, 0)  # Shape: (Nsample, 1, Ngene)

    # Compute 'a' using broadcasting
    a = (g_Var * Exp - 2 * g_Exp * Var) / torch.pow(Exp, 3)

    # Undo the unsqueeze and permute for Exp and Var
    Exp = Exp.permute(2, 0, 1).squeeze(-1)  # Shape: (Ngene, Nsample)
    Var = Var.permute(2, 0, 1).squeeze(-1)  # Shape: (Ngene, Nsample)

    # Compute 'Var / (2 * Exp^2)'
    Var_Exp2 = Var / (2 * torch.square(Exp))  # Shape: (Ngene, Nsample)

     # Compute 'b'
    Y_t = Y.T.unsqueeze(1)  # Shape: (Ngene, 1, Nsample)
    Var_Exp2_t = Var_Exp2.T.unsqueeze(1)  # Shape: (Ngene, 1, Nsample)
    log_Exp = torch.log(Exp.squeeze(-1)).T.unsqueeze(1)  # Shape: (Ngene, 1, Nsample)

    # Reshape Exp to match g_Exp for broadcasting
    Exp_broadcast = Exp.T.unsqueeze(1)  # Shape: (Nsample, 1, Ngene)

    # Compute 'b' using the reshaped Exp for broadcasting
    b = -(Y_t - log_Exp - Var_Exp2_t) * (2 * (g_Exp / Exp_broadcast) + a)

    # Compute 'grad_PY'
    # 1. Compute element-wise sum of 'a' and 'b'
    sum_ab = a + b  # Shape: (Nsample, Ncell, Ngene)

    # 2. Square SigmaY and prepare it for broadcasting
    SigmaY_squared = torch.square(SigmaY)  # Shape: (Ngene, Nsample)

    # 3. Divide 0.5 by SigmaY_squared, transpose to align with 'sum_ab'
    factor = 0.5 / SigmaY_squared.T.unsqueeze(1)  # Shape: (Nsample, 1, Ngene)

    # 4. Multiply element-wise by 'sum_ab'
    weighted_sum_ab = factor * sum_ab  # Shape: (Nsample, Ncell, Ngene)

    # 5. Sum over the Ngene dimension (dimension 2)
    grad_PY = -torch.sum(weighted_sum_ab, dim=2)  # Shape: (Nsample, Ncell)
    return grad_PY



class OncoBLADE:
    def __init__(self, Y, SigmaY=0.05, Mu0=2, Alpha=1,
                 Alpha0=1, Beta0=1, Kappa0=1,
                 Nu_Init=None, Omega_Init=1, Beta_Init=None,
                 fix_Beta=False, fix_Nu=False, fix_Omega=False,
                 device=None):
        # Set the device (GPU if available, otherwise CPU)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        
        if torch.cuda.is_available():
            print("CUDA is available. Using GPU for computation.")
           
        else:
            print("CUDA is not available. Using CPU for computation.")
            
        
        
        # Convert Y to a tensor and move to the specified device
        self.weight = 1
        self.Y = torch.tensor(Y, dtype=torch.float64, device=self.device)
        self.Ngene, self.Nsample = self.Y.shape

        # Handle the fix parameters
        self.Fix_par = {
            'Beta': fix_Beta,
            'Nu': fix_Nu,
            'Omega': fix_Omega
        }

        if not isinstance(Mu0, torch.Tensor) and not isinstance(Mu0, np.ndarray):  # Check if Mu0 is a scalar
            self.Ncell = Mu0
            self.Mu0 = torch.zeros((self.Ngene, self.Ncell), dtype=torch.float64, device=self.device)
        else:
            self.Ncell = Mu0.shape[1]
            self.Mu0 = torch.tensor(Mu0, dtype=torch.float64, device=self.device)

        if isinstance(SigmaY, torch.Tensor):
            self.SigmaY = SigmaY.to(self.device)
        elif isinstance(SigmaY, np.ndarray):
            self.SigmaY = torch.tensor(SigmaY, device=self.device)
        else:
            self.SigmaY = torch.ones((self.Ngene, self.Nsample), dtype=torch.float64, device=self.device) * SigmaY

        if isinstance(Alpha, torch.Tensor):
            self.Alpha = Alpha.to(self.device)
        elif isinstance(Alpha, np.ndarray):
            self.Alpha = torch.tensor(Alpha, device=self.device)
        else:
            self.Alpha = torch.ones((self.Nsample, self.Ncell), dtype=torch.float64, device=self.device) * Alpha

        if isinstance(Omega_Init, torch.Tensor):
            self.Omega = Omega_Init.to(self.device)
        elif isinstance(Omega_Init, np.ndarray):
            self.Omega = torch.tensor(Omega_Init, device=self.device)
        else:
            self.Omega = torch.zeros((self.Ngene, self.Ncell), dtype=torch.float64, device=self.device) + Omega_Init

        if Nu_Init is None:
            self.Nu = torch.zeros((self.Nsample, self.Ngene, self.Ncell), dtype=torch.float64, device=self.device)
        else:
            self.Nu = torch.tensor(Nu_Init, dtype=torch.float64, device=self.device)

        if isinstance(Beta_Init, torch.Tensor):
            self.Beta = Beta_Init.to(self.device)
        elif isinstance(Beta_Init, np.ndarray):
            self.Beta = torch.tensor(Beta_Init, device=self.device)
        else:
            self.Beta = torch.ones((self.Nsample, self.Ncell), dtype=torch.float64, device=self.device)

        if isinstance(Alpha0, torch.Tensor):
            self.Alpha0 = Alpha0.to(self.device)
        elif isinstance(Alpha0, np.ndarray):
            self.Alpha0 = torch.tensor(Alpha0, device=self.device)
        else:
            self.Alpha0 = torch.ones((self.Ngene, self.Ncell), dtype=torch.float64, device=self.device) * Alpha0

        if isinstance(Beta0, torch.Tensor):
            self.Beta0 = Beta0.to(self.device)
        elif isinstance(Beta0, np.ndarray):
            self.Beta0 = torch.tensor(Beta0, device=self.device)
        else:
            self.Beta0 = torch.ones((self.Ngene, self.Ncell), dtype=torch.float64, device=self.device) * Beta0

        if isinstance(Kappa0, torch.Tensor):
            self.Kappa0 = Kappa0.to(self.device)
        elif isinstance(Kappa0, np.ndarray):
            self.Kappa0 = torch.tensor(Kappa0, device=self.device)
        else:
            self.Kappa0 = torch.ones((self.Ngene, self.Ncell), dtype=torch.float64, device=self.device) * Kappa0

    def to_device(self, device):
        self.device = torch.device(device)

        # Move all tensor attributes to the new device
        self.Y = self.Y.to(self.device)
        self.Mu0 = self.Mu0.to(self.device)
        self.SigmaY = self.SigmaY.to(self.device)
        self.Alpha = self.Alpha.to(self.device)
        self.Omega = self.Omega.to(self.device)
        self.Nu = self.Nu.to(self.device)
        self.Beta = self.Beta.to(self.device)
        self.Alpha0 = self.Alpha0.to(self.device)
        self.Beta0 = self.Beta0.to(self.device)
        self.Kappa0 = self.Kappa0.to(self.device)
        
    def Ydiff(self, Nu, Beta):
        F = self.ExpF(Beta)
        Ypred = torch.matmul(torch.exp(Nu), F.T)
        return torch.sum(torch.square(self.Y - Ypred))

    def ExpF(self, Beta):
        #NSample by Ncell (Expectation of F)
        return ExpF_C(Beta)

    def ExpQ(self, Nu, Beta, Omega):
        # Ngene by Nsample (Expected value of Y)
        return ExpQ_C(Nu, Beta, Omega)

    def VarQ(self, Nu, Beta, Omega):
        # Ngene by Nsample (Variance value of Y)
        return VarQ_C(Nu, Beta, Omega, self.Ngene, self.Ncell, self.Nsample)

    # Expectation of log P(X | mu0, Kappa0, Alpha0, Beta0)
    def Estep_PX(self, Nu, Omega):
        return Estep_PX_C(self.Mu0, Nu, Omega, self.Alpha0, self.Beta0, self.Kappa0, self.Ncell, self.Nsample)

    # Expectation of log P(Y|X,F)
    def Estep_PY(self, Nu, Omega, Beta):
        return Estep_PY_C(self.Y, self.SigmaY, Nu, Omega, Beta, self.Ngene, self.Ncell, self.Nsample)

    # Expectation of log P(F)
    def Estep_PF(self, Beta):
        # Summing across the last dimension (across cells)
        term1 = torch.sum(torch.special.gammaln(self.Alpha), dim=-1)  # Shape: (Nsample,)
        term2 = torch.special.gammaln(torch.sum(self.Alpha, dim=-1))  # Shape: (Nsample,)

        digamma_Beta = torch.special.digamma(Beta)  # Shape: (Nsample, Ncell)
        digamma_Beta_sum = torch.special.digamma(torch.sum(Beta, dim=-1)).unsqueeze(1)  # Shape: (Nsample, 1)

        term3 = torch.sum((self.Alpha - 1) * (digamma_Beta - digamma_Beta_sum), dim=-1)  # Shape: (Nsample,)

        # Summing the final result to get a scalar (total loss or score)
        return -(torch.sum(term1 - term2) + torch.sum(term3))

    # Expectation of log Q(X)
    def Estep_QX(self, Omega):
        return -self.Nsample * torch.sum(torch.log(Omega))

    # Expectation of log Q(F)
    def Estep_QF(self, Beta):
        # Compute loggamma(Beta) and sum over all elements
        term1 = torch.sum(torch.special.gammaln(Beta))

        # Compute loggamma of the row sums and then sum
        term2 = torch.sum(torch.special.gammaln(torch.sum(Beta, dim=1)))

        # Compute digamma(Beta)
        digamma_Beta = torch.special.digamma(Beta)

        # Compute digamma of row sums of Beta, and expand for broadcasting
        digamma_Beta_sum = torch.special.digamma(torch.sum(Beta, dim=1)).unsqueeze(1).expand(-1, self.Ncell)

        # Calculate the sum of (Beta - 1) * (digamma(Beta) - digamma_Beta_sum)
        term3 = torch.sum((Beta - 1) * (digamma_Beta - digamma_Beta_sum))

        # Return the final result
        return -(term1 - term2) + term3

    def grad_Nu(self, Nu, Omega, Beta):
        # return Ngene by Ncell
        return grad_Nu_C(self.Y, self.SigmaY, Nu, Omega, Beta, self.Mu0, self.Alpha0, self.Beta0, self.Kappa0, self.Ngene, self.Ncell, self.Nsample, self.weight)

    def grad_Omega(self, Nu, Omega, Beta):
        # return Ngene by Ncell
        return grad_Omega_C(self.Y, self.SigmaY, Nu, Omega, Beta,
                          self.Mu0, self.Alpha0, self.Beta0, self.Kappa0, self.Ngene, self.Ncell, self.Nsample, self.weight)

    def g_Exp_Beta(self, Nu, Beta, B0):
        return g_Exp_Beta(Nu, Omega, Beta, B0, self.Ngene, self.Ncell, self.Nsample)

    def grad_Beta(self, Nu, Omega, Beta):
        # B0 is the sum of Beta over the Ncell dimension (axis 1)
        B0 = torch.sum(Beta, dim=1)  # Shape: (Nsample,)

        # Compute grad_PY using the ported version of g_PY_Beta
        grad_PY = g_PY_Beta_C(Nu, Beta, Omega, self.Y, self.SigmaY, B0, self.Ngene, self.Ncell, self.Nsample)

        # Compute grad_PF
        polygamma_Beta = torch.special.polygamma(1, Beta)  # Shape: (Nsample, Ncell)
        polygamma_B0 = torch.special.polygamma(1, B0).unsqueeze(1)  # Shape: (Nsample, 1)

        grad_PF = (self.Alpha - 1) * polygamma_Beta - torch.sum((self.Alpha - 1) * polygamma_B0, dim=1, keepdim=True)

        # Compute grad_QF
        grad_QF = (Beta - 1) * polygamma_Beta - torch.sum((Beta - 1) * polygamma_B0, dim=1, keepdim=True)

        # Return the final result with scaling factors applied
        scaling_factor = torch.sqrt(torch.tensor(self.Ngene / self.Ncell, dtype=Beta.dtype, device=Beta.device))
        return grad_PY + grad_PF * scaling_factor - grad_QF * scaling_factor


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
            with torch.no_grad():
                params = torch.tensor(params, device=self.device)
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

                loss = -self.E_step(Nu, Beta, Omega)
                return loss.cpu().numpy()

        # gradient function
        def grad(params):
            with torch.no_grad():
                #s1 = timer()
                params = torch.tensor(params, device=self.device)
                Nu = params[0:self.Ncell*self.Ngene*self.Nsample].reshape(self.Nsample, self.Ngene, self.Ncell)
                Omega = params[self.Ncell*self.Ngene*self.Nsample:(self.Ncell*self.Ngene*self.Nsample + self.Ngene*self.Ncell)].reshape(self.Ngene, self.Ncell)
                Beta = params[(self.Ncell*self.Ngene*self.Nsample + self.Ngene*self.Ncell):(self.Ncell*self.Ngene*self.Nsample + \
                        self.Ngene*self.Ncell + self.Nsample*self.Ncell)].reshape(self.Nsample, self.Ncell)
                #e1 = timer()
                #print("        Time reshaping (ms)", 1000*(e1-s1))

                #s1 = timer()
                if self.Fix_par['Nu']:
                    g_Nu = torch.zeros(Nu.shape, device = self.device)
                else:
                    g_Nu = -self.grad_Nu(Nu, Omega, Beta)
                #e1 = timer()
                #print("        Time Nu_grad (ms)", 1000*(e1-s1))


                #s1 = timer()
                if self.Fix_par['Omega']:
                    g_Omega = torch.zeros(Omega.shape, device = self.device)
                else:
                    g_Omega = -self.grad_Omega(Nu, Omega, Beta)
                #e1 = timer()
                #print("        Time omega_grad (ms)", 1000*(e1-s1))

                #s1 = timer()
                if self.Fix_par['Beta']:
                    g_Beta = torch.zeros(Beta.shape, device = self.device)
                else:
                    g_Beta = -self.grad_Beta(Nu, Omega, Beta)
                #e1 = timer()
                #print("        Time beta_grad (ms)", 1000*(e1-s1))

                #s1 = timer()
                g = torch.cat((g_Nu.flatten(), g_Omega.flatten(), g_Beta.flatten()))
                #e1 = timer()
                #print("        Time flatten (ms)", 1000*(e1-s1))
                return g.cpu().numpy()


        # Perform Optimization
        Init = torch.cat((self.Nu.flatten(), self.Omega.flatten(), self.Beta.flatten()))
        bounds = [(-np.inf, np.inf) if i < (self.Ncell*self.Ngene*self.Nsample) else (0.0000001, 100) for i in range(len(Init))]

        s1 = timer()
        out = scipy.optimize.minimize(
                fun = loss, x0 = Init.cpu().numpy(), bounds = bounds, jac = grad,
                options = {'disp': True, 'maxiter' : 1000},
                method='L-BFGS-B')
        e1 = timer()
        print("        Time (s) scipy optimize", e1 - s1)

        params = out.x

        self.Nu = torch.tensor(params[0:self.Ncell*self.Ngene*self.Nsample].reshape(self.Nsample, self.Ngene, self.Ncell), device=self.device)
        self.Omega = torch.tensor(params[self.Ncell*self.Ngene*self.Nsample:(self.Ncell*self.Ngene*self.Nsample + self.Ngene*self.Ncell)].reshape(self.Ngene, self.Ncell), device=self.device)
        self.Beta = torch.tensor(params[(self.Ncell*self.Ngene*self.Nsample + self.Ngene*self.Ncell):(self.Ncell*self.Ngene*self.Nsample + \
                        self.Ngene*self.Ncell + self.Nsample*self.Ncell)].reshape(self.Nsample, self.Ncell), device=self.device)

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
        if not np.all(np.isfinite(self.Y.cpu().numpy())):
            warnings.warn('non-finite values detected in bulk gene expression data (Y).', Warning, stacklevel=2)

        if np.any(self.Y.cpu().numpy() < 0):
            warnings.warn('Negative expression levels were detected in bulk gene expression data (Y).', Warning, stacklevel=2)

        if np.any(self.Alpha.cpu().numpy() <= 0):
            warnings.warn('Zero or negative values in Alpha', Warning, stacklevel=2)

        if np.any(self.Beta.cpu().numpy() <= 0):
            warnings.warn('Zero or negative values in Beta', Warning, stacklevel=2)

        if np.any(self.Alpha0.cpu().numpy() <= 0):
            warnings.warn('Zero or negative values in Alpha0', Warning, stacklevel=2)

        if np.any(self.Beta0.cpu().numpy() <= 0):
            warnings.warn('Zero or negative values in Beta0', Warning, stacklevel=2)

        if np.any(self.Kappa0.cpu().numpy() <= 0):
            warnings.warn('Zero or negative values in Kappa0', Warning, stacklevel=2)

    def Update_Alpha(self, Expected=None, Temperature=None):# if Expected fraction is given, that part will be fixed
        # Updating Alpha
        Fraction = self.ExpF(self.Beta).cpu.numpy()
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
        self.Alpha = torch.tensor(self.Alpha, device=self.device)


    def Update_Alpha_Group(self, Expected=None, Temperature=None):# if Expected fraction is given, that part will be fixed
        # Updating Alpha
        AvgBeta = torch.mean(self.Beta, 0)
        Fraction_Avg = AvgBeta / torch.sum(AvgBeta)

        if Expected is not None:  # Reflect the expected values
            # expectaion can be a diction (with two keys; Group and Expectation) or just a matrix
            if type(Expected) is dict:
                if "Group" in Expected:  # Group (Ngroup by Nctype matrix) indicates a group of cell types with known collective fraction
                    Group = Expected['Group']
                else:
                    Group = torch.eye(Expected['Expectation'].shape[1], device=self.device)
                Expected = Expected['Expectation']
            else:
                Group = torch.eye(Expected.shape[1], device=self.device)

            if self.Beta.shape[0] != Expected.shape[0] or self.Beta.shape[1] != Group.shape[1]:
                raise ValueError('Pre-determined fraction is in wrong shape (should be Nsample by Ncelltype)')
            Expected = torch.tensor(Expected, device=self.device)

            # rescale the fraction to meet the expected fraction
            for sample in range(self.Nsample):
                Fraction = Fraction_Avg.clone()
                IndG = torch.where(~torch.isnan(Expected[sample, :]))[0]

                IndCells = []
                for group in IndG:
                    IndCell = torch.where(Group[group, :] == 1)[0]
                    Fraction[IndCell] = Fraction[IndCell] / torch.sum(Fraction[IndCell])
                    Fraction[IndCell] = Fraction[IndCell] * Expected[sample, group]
                    IndCells.extend(IndCell.tolist())

                IndNan = torch.tensor(list(set(range(Group.shape[1])) - set(IndCells)), device=Fraction.device)
                Fraction[IndNan] = Fraction[IndNan] / torch.sum(Fraction[IndNan])
                Fraction[IndNan] = Fraction[IndNan] * (1 - torch.sum(Expected[sample, IndG]))

                AlphaSum = torch.sum(AvgBeta[IndNan]) / torch.sum(Fraction[IndNan])
                self.Alpha[sample, :] = Fraction * AlphaSum
        else:
            for sample in range(self.Nsample):
                self.Alpha[sample,:] = AvgBeta


    def Update_SigmaY(self, SampleSpecific=False):
        Var = VarQ_C(self.Nu, self.Beta, self.Omega, self.Ngene, self.Ncell, self.Nsample)
        Exp = ExpQ_C(self.Nu, self.Beta, self.Omega)

        a = Var / Exp / Exp
        b = torch.square((self.Y - torch.log(Exp)) - 0.5 * a)

        if SampleSpecific:
            self.SigmaY = torch.sqrt(a+b)
        else:  # shared in all samples
            self.SigmaY = torch.mean(torch.sqrt(a + b), dim=1, keepdim=True).expand(-1, self.Nsample)



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

    with parallel_backend('threading', n_jobs=Njob):
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
                           Expected=None, iter=100, minDiff=1e-4, TempRange=None, Update_SigmaY=False):
    s1 = timer()
    Ngene, Nsample = Y.shape
    Ncell = X.shape[1]

    Mu0 = X
    logY = np.log(Y + 1)
    SigmaY = np.tile(np.std(logY, 1)[:, np.newaxis], [1, Nsample]) * SY + 0.1
    Omega_Init = stdX
    Beta0 = Alpha0 * np.square(stdX)

    Nu_Init = np.zeros((Nsample, Ngene, Ncell))
    for i in range(Nsample):
        Nu_Init[i, :, :] = X
    e1 = timer()
    print("    Time (s) Init part Iterative_optimization", e1 - s1)

    # Optimization without given Temperature
    s2 = timer()
    Beta_Init = np.random.gamma(shape=1, size=(Nsample, Ncell)) + t(Init_Fraction) * Init_Trust
    obj = OncoBLADE(logY, SigmaY, Mu0, Alpha, Alpha0, Beta0, Kappa0,
                    Nu_Init, Omega_Init, Beta_Init)

    obj.Check_health()
    obj_func = [None] * iter
    obj_func[0] = obj.E_step(obj.Nu, obj.Beta, obj.Omega)
    e2 = timer()
    print("    Time (s) second part of Iterative_optimization", e2 - s2)

    for i in range(1, iter):
        s3 = timer()
        obj.Optimize()
        print(type(obj.Nu))
        e3 = timer()
        print("    Time (s) obj.Optimizer()", e3 - s3)
        s4 = timer()
        obj.Update_Alpha_Group(Expected=Expected)
        print(type(obj.Nu))
        e4 = timer()
        print("    Time (s) obj.Update_Alpha_Group", e4 - s4)
        s5 = timer()
        if Update_SigmaY:
            obj.Update_SigmaY()
        print(type(obj.Nu))
        e5 = timer()
        print("    Time (s) obj.Update_SigmaY()", e5 - s5)

        s6 = timer()
        obj_func[i] = obj.E_step(obj.Nu, obj.Beta, obj.Omega)
        #obj_func[i] = obj.E_step(torch.tensor(obj.Nu), torch.tensor(obj.Beta), torch.tensor(obj.Omega))
        e6 = timer()
        print("    Time (s) obj.E_step()", e6 - s6)

        # Check convergence
        if torch.abs(obj_func[i] - obj_func[i - 1]) < minDiff:
            break

    return obj, obj_func, Rep


def Framework_Iterative(X, stdX, Y, Ind_Marker=None,
                        Alpha=1, Alpha0=0.1, Kappa0=1, sY=1,
                        Nrep=3, Njob=10, fsel=0, Update_SigmaY=False, Init_Trust=10,
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
    s1 = timer()
    Init_Fraction, Ind_use = SVR_Initialization(X_small, Y_small, Njob=Njob, Nus=[0.25, 0.5, 0.75])
    e1 = timer()
    print("Time (s) SVR_Init", e1-s1)

    if Temperature is None or Temperature is False: #  Optimization without the temperature
        # Run the optimizations sequentially
        s2 = timer()
        with parallel_backend('threading', n_jobs=Njob):
            outs = Parallel(n_jobs=Njob, verbose=10)(
                delayed(Iterative_Optimization)(X_small[Ind_use,:], stdX_small[Ind_use,:], Y_small[Ind_use,:],
                    Alpha, Alpha0, Kappa0, sY, rep, Init_Fraction, Expected=Expectation, Init_Trust=Init_Trust, iter=IterMax,
                    Update_SigmaY = Update_SigmaY)
                    for rep in range(Nrep)
                )
        #outs = [Iterative_Optimization(X_small[Ind_use, :], stdX_small[Ind_use, :], Y_small[Ind_use, :],
        #                               Alpha, Alpha0, Kappa0, sY, rep, Init_Fraction,
        #                               Expected=Expectation, Init_Trust=Init_Trust, iter=IterMax,
        #                               Update_SigmaY=Update_SigmaY) for rep in range(Nrep)]
        e2 = timer()
        print("Time (s) Iterative Optim loop", e2-s2)

        ## Final OncoBLADE results
        s3 = timer()
        outs, convs, Reps = zip(*outs)
        cri = [obj.E_step(obj.Nu, obj.Beta, obj.Omega).cpu().numpy() for obj in outs]
        out = outs[np.nanargmax(cri)]
        conv = convs[np.nanargmax(cri)]
        e3 = timer()
        print("Time (s) Final part framework", e3-s3)
    else:
        if Temperature is True:
            Temperature = [1, 100]
        else:
            if len(Temperature) != 2:
                raise ValueError('Temperature has to be either None, True or list of 2 temperature values (minimum and maximum temperatures)')
            if Temperature[1] < Temperature[0]:
                raise ValueError('A lower maximum temperature than minimum temperature is given')

        outs = [Iterative_Optimization(X_small[Ind_use, :], stdX_small[Ind_use, :], Y_small[Ind_use, :],
                                       Alpha, Alpha0, Kappa0, sY, rep, Init_Fraction,
                                       Expected=Expectation, Init_Trust=Init_Trust, TempRange=np.linspace(Temperature[0], Temperature[1], iter=IterMax),
                                       Update_SigmaY=Update_SigmaY) for rep in range(Nrep)]

        ## Final OncoBLADE results
        outs, convs, Reps = zip(*outs)
        cri = [obj.E_step(obj.Nu, obj.Beta, obj.Omega).cpu().numpy() for obj in outs]
        out = outs[np.nanargmax(cri)]
        conv = convs[np.nanargmax(cri)]
        
    

    return out, conv, zip(outs, cri), args



#########NUMBA functions for purification########

@njit(fastmath=True)
def ExpF_numba(Beta, Ncell):
    #NSample by Ncell (Expectation of F)
    output = np.empty(Beta.shape)
    for c in range(Ncell):
        output[:,c] = Beta[:,c]/np.sum(Beta, axis=1)
    return output



@njit(fastmath=True)
def ExpQ_numba(Nu, Beta, Omega, Ngene, Ncell, Nsample):
    # Ngene by Nsample (Expected value of Y)
    ExpB = ExpF_numba(Beta, Ncell) # Nsample by Ncell
    out = np.zeros((Ngene, Nsample))
    for i in range(Nsample):
        for c in range(Ncell):
            out[:,i] = out[:,i] + ExpB[i,c] * np.exp(Nu[i,:,c] + 0.5*np.square(Omega[:,c]))

    return out 


@njit(fastmath=True)
def VarQ_numba(Nu, Beta, Omega, Ngene, Ncell, Nsample):
    # Ngene by Nsample (Variance value of Y)
    B0 = np.sum(Beta, axis=1) # Nsample
    Btilda = ExpF_numba(Beta, Ncell) # Nsample by Ncell
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
def Estep_PY_numba(Y, SigmaY, Nu, Omega, Beta, Ngene, Ncell, Nsample):
    Var = VarQ_numba(Nu, Beta, Omega, Ngene, Ncell, Nsample)
    Exp = ExpQ_numba(Nu, Beta, Omega, Ngene, Ncell, Nsample)

    a = Var / Exp / Exp
                            
    return np.sum(
            -0.5 / np.square(SigmaY) * (a + np.square((Y-np.log(Exp)) - 0.5 * a))
            )


@njit(fastmath = True)
def Estep_PX_numba(Mu0, Nu, Omega, Alpha0, Beta0, Kappa0, Ncell, Nsample):
    NuExp = np.sum(Nu, 0)/Nsample # expected Nu, Ngene by Ncell
    AlphaN = Alpha0 + 0.5*Nsample # Posterior Alpha

    ExpBetaN = Beta0 + (Nsample-1)/2*np.square(Omega) + \
            Kappa0*Nsample/(2*(Kappa0 + Nsample)) * (np.square(Omega)/Nsample + np.square(NuExp - Mu0))

    for i in range(Nsample):
        ExpBetaN = ExpBetaN + 0.5*np.square(Nu[i,:,:] - NuExp)

    return np.sum(- AlphaN * np.log(ExpBetaN))

@njit(fastmath=True)
def grad_Nu_numba(Y, SigmaY, Nu, Omega, Beta, Mu0, Alpha0, Beta0, Kappa0, Ngene, Ncell, Nsample, weight):
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
    Btilda = ExpF_numba(Beta, Ncell) # Nsample by Ncell

    # Ngene by Ncell by Nsample
    Exp = ExpQ_numba(Nu, Beta, Omega, Ngene, Ncell, Nsample) # Ngene by Nsample
    Var = VarQ_numba(Nu, Beta, Omega, Ngene, Ncell, Nsample) # Ngene by Nsample

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
def grad_Omega_numba(Y, SigmaY, Nu, Omega, Beta, Mu0, Alpha0, Beta0, Kappa0, Ngene, Ncell, Nsample, weight):
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
    Btilda = ExpF_numba(Beta, Ncell) # Nsample by Ncell

    # Ngene by Ncell by Nsample
    Exp = ExpQ_numba(Nu, Beta, Omega, Ngene, Ncell, Nsample) # Ngene by Nsample
    Var = VarQ_numba(Nu, Beta, Omega, Ngene, Ncell, Nsample) # Ngene by Nsample

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
def g_Exp_Beta_numba(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample):
    
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
def g_Var_Beta_numba(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample):
    
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
    Exp = ExpQ_numba(Nu, Beta, Omega, Ngene, Ncell, Nsample)
    Var = VarQ_numba(Nu, Beta, Omega, Ngene, Ncell, Nsample)
        
    # Nsample by Ncell be Ngene
    g_Exp = g_Exp_Beta_numba(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample)
    g_Var = g_Var_Beta_numba(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample)
        
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
def g_PY_Beta_numba(Nu, Beta, Omega, Y, SigmaY, B0, Ngene, Ncell, Nsample):

    # Ngene by Nsample
    Exp = ExpQ_numba(Nu, Beta, Omega, Ngene, Ncell, Nsample)
    Var = VarQ_numba(Nu, Beta, Omega, Ngene, Ncell, Nsample)
        
    # Nsample by Ncell be Ngene
    g_Exp = g_Exp_Beta_numba(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample)
    g_Var = g_Var_Beta_numba(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample)
        
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


######Casting function#######

def to_torch(array, device='cuda'):
    return torch.tensor(array).to(device)

def convert_to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()
    return tensor

def ensure_numpy(array):
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()  # Ensures tensor is on CPU and converts to numpy
    return array

class OncoBLADE_numba:
    def __init__(self, Y, SigmaY=0.05, Mu0=2, Alpha=1,
                 Alpha0=1, Beta0=1, Kappa0=1,
                 Nu_Init=None, Omega_Init=1, Beta_Init=None,
                 fix_Beta=False, fix_Nu=False, fix_Omega=False):
        self.weight = 1

        # Ensure all tensor inputs are converted to numpy arrays
        self.Y = ensure_numpy(Y)
        self.Ngene, self.Nsample = self.Y.shape

        self.Mu0 = ensure_numpy(Mu0) if isinstance(Mu0, np.ndarray) else np.zeros((self.Ngene, Mu0))
        self.Ncell = self.Mu0.shape[1]

        self.SigmaY = ensure_numpy(SigmaY) if isinstance(SigmaY, np.ndarray) else np.ones((self.Ngene, self.Nsample)) * SigmaY
        self.Alpha = ensure_numpy(Alpha) if isinstance(Alpha, np.ndarray) else np.ones((self.Nsample, self.Ncell)) * Alpha
        self.Omega = ensure_numpy(Omega_Init) if isinstance(Omega_Init, np.ndarray) else np.zeros((self.Ngene, self.Ncell)) + Omega_Init
        self.Nu = ensure_numpy(Nu_Init) if Nu_Init is not None else np.zeros((self.Nsample, self.Ngene, self.Ncell))
        self.Beta = ensure_numpy(Beta_Init) if isinstance(Beta_Init, np.ndarray) else np.ones((self.Nsample, self.Ncell))
        self.Alpha0 = ensure_numpy(Alpha0) if isinstance(Alpha0, np.ndarray) else np.ones((self.Ngene, self.Ncell)) * Alpha0
        self.Beta0 = ensure_numpy(Beta0) if isinstance(Beta0, np.ndarray) else np.ones((self.Ngene, self.Ncell)) * Beta0
        self.Kappa0 = ensure_numpy(Kappa0) if isinstance(Kappa0, np.ndarray) else np.ones((self.Ngene, self.Ncell)) * Kappa0

        self.Fix_par = {
            'Beta': fix_Beta,
            'Nu': fix_Nu,
            'Omega': fix_Omega
        }


   
    def Ydiff(self, Nu, Beta):
        F = self.ExpF(Beta)
        Ypred = np.dot(np.exp(Nu), t(F))
        return np.sum(np.square(self.Y-Ypred))

    def ExpF_numba(self, Beta):
        #NSample by Ncell (Expectation of F)
        return ExpF_numba(Beta, self.Ncell)

    def ExpQ_numba(self, Nu, Beta, Omega):
        # Ngene by Nsample (Expected value of Y)
        return ExpQ_numba(Nu, Beta, Omega, self.Ngene, self.Ncell, self.Nsample)

    def VarQ_numba(self, Nu, Beta, Omega):
        # Ngene by Nsample (Variance value of Y)
        return VarQ_numba(Nu, Beta, Omega, self.Ngene, self.Ncell, self.Nsample)

    # Expectation of log P(X | mu0, Kappa0, Alpha0, Beta0)
    def Estep_PX(self, Nu, Omega):
        return Estep_PX_numba(self.Mu0, Nu, Omega, self.Alpha0, self.Beta0, self.Kappa0, self.Ncell, self.Nsample)

    # Expectation of log P(Y|X,F)
    def Estep_PY(self, Nu, Omega, Beta):
        return Estep_PY_numba(self.Y, self.SigmaY, Nu, Omega, Beta, self.Ngene, self.Ncell, self.Nsample)

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
        return grad_Nu_numba(self.Y, self.SigmaY, Nu, Omega, Beta, self.Mu0, self.Alpha0, self.Beta0, self.Kappa0, self.Ngene, self.Ncell, self.Nsample, self.weight)

    def grad_Omega(self, Nu, Omega, Beta):
        # return Ngene by Ncell
        return grad_Omega_numba(self.Y, self.SigmaY, Nu, Omega, Beta,
                          self.Mu0, self.Alpha0, self.Beta0, self.Kappa0, self.Ngene, self.Ncell, self.Nsample, self.weight)

    def g_Exp_Beta(self, Nu, Beta, B0):
        return g_Exp_Beta_numba(Nu, Omega, Beta, B0, self.Ngene, self.Ncell, self.Nsample)

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
        Var = VarQ_numba(self.Nu, self.Beta, self.Omega, self.Ngene, self.Ncell, self.Nsample)
        Exp = ExpQ_numba(self.Nu, self.Beta, self.Omega, self.Ngene, self.Ncell, self.Nsample)
        
        a = Var / Exp / Exp
        b = np.square((self.Y-np.log(Exp)) - 0.5 * a)

        if SampleSpecific:
            self.SigmaY = np.sqrt(a+b)
        else:  # shared in all samples
            self.SigmaY = np.tile(np.mean(np.sqrt(a+b), axis=1)[:,np.newaxis], [1,self.Nsample])




def Parallel_Purification(obj, iter=1000, minDiff=10e-4, Update_SigmaY=False):
    obj.Check_health()
    obj_func = [float('nan')] * iter
    obj_func[0] = obj.E_step(obj.Nu, obj.Beta, obj.Omega)
    for i in range(1, iter):
        obj.Reestimate_Nu()
        if Update_SigmaY:
            obj.Update_SigmaY()
        obj_func[i] = obj.E_step(obj.Nu, obj.Beta, obj.Omega)

        # Check for convergence
        if np.abs(obj_func[i] - obj_func[i-1]) < minDiff:
            break
    return obj, obj_func

def Purify_AllGenes(OncoBLADE_object, Mu, Omega, Y, Ncores):
    Mu = ensure_numpy(Mu)
    Omega = ensure_numpy(Omega)
    Y = ensure_numpy(Y)
    obj = OncoBLADE_object['final_obj']
    obj.Alpha = convert_to_numpy(obj.Alpha)
    obj.SigmaY = convert_to_numpy(obj.SigmaY)
    obj.Mu0 = convert_to_numpy(obj.Mu0)
    obj.Beta0 = convert_to_numpy(obj.Beta0)
    obj.Kappa0 = convert_to_numpy(obj.Kappa0)
    obj.Beta = convert_to_numpy(obj.Beta)
    
    Ngene, Nsample = Y.shape
    Ncell = Mu.shape[1]
    logY = np.log(Y+1)
    SigmaY = np.tile(np.std(logY,1)[:,np.newaxis], [1,Nsample]) * OncoBLADE_object['outs']['sY'] + 0.1
    Beta0 = OncoBLADE_object['outs']['Alpha0'] * np.square(Omega)
    Nu_Init = np.zeros((Nsample, Ngene, Ncell))
    for i in range(Nsample):
        Nu_Init[i,:,:] = Mu

    # Fetch objs per gene
    Ngene_total = Mu.shape[0]
    objs = []
    for ix in range(Ngene_total):
        objs.append(OncoBLADE_numba(
            Y = np.atleast_2d(logY[ix,:]),
            SigmaY = np.atleast_2d(SigmaY[ix,:]),
            Mu0 = np.atleast_2d(Mu[ix,:]),
            Alpha = obj.Alpha,
            Alpha0 = OncoBLADE_object['outs']['Alpha0'],
            Beta0 = np.atleast_2d(Beta0[ix,:]),
            Kappa0 = OncoBLADE_object['outs']['Kappa0'],
            Nu_Init = np.reshape(np.atleast_3d(Nu_Init[:,ix,:]), (Nsample,1,Ncell)), 
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
    obj = OncoBLADE_numba(Y, SigmaY, Mu0, Alpha, Alpha0, Beta0, Kappa0, Nu_Init, Omega_Init, Beta_Init, fix_Beta =True)
    obj.log = logs
    
    return obj, obj_func









