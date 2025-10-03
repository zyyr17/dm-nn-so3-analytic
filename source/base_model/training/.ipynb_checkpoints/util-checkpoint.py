import string
import random
import torch
import numpy as np
import torch.nn.functional as F


def batch_trace(X):
    return torch.einsum("bii->b", X)

def _softplus_penalty(values, threshold=0.0, epsilon=1e-9):
    """Applies a smooth softplus penalty to values violating a lower bound."""
    mask = values < threshold - epsilon
    if not torch.any(mask):
        return torch.zeros((), dtype=values.dtype, device=values.device, requires_grad=values.requires_grad)
    return torch.mean(torch.sum(F.softplus(threshold - values[mask]), dim=-1) ** 2)

def occupation_spectrum_penalty(P, S, eps=1e-10):
    """
    Penalizes occupation eigenvalues outside the interval [0, 2] after rotating P to an orthonormal AO basis.

    Parameters:
    -----------
    P : torch.Tensor, shape (B, N, N)
        Density matrix (assumed real and symmetric)
    S : torch.Tensor, shape (B, N, N)
        Overlap matrix (assumed real and symmetric)
    eps : float
        Numerical stabilizer added to S before Cholesky decomposition

    Returns:
    --------
    penalty : torch.Tensor
        Scalar penalty (mean over batch)
    ceigvals : torch.Tensor or None
        Eigenvalues of Q = LᵀPL (diagnostic), or None if decomposition fails
    """
    P = 0.5 * (P + P.transpose(-1, -2))
    S = 0.5 * (S + S.transpose(-1, -2))

    device, dtype = P.device, P.dtype
    B, N, _ = P.shape
    I = torch.eye(N, dtype=dtype, device=device)

    try:
        L = torch.linalg.cholesky(S + eps * I)
    except RuntimeError:
        print("[Error] Cholesky decomposition failed.")
        return torch.tensor(1e3, device=device), None

    Q = L.transpose(-1, -2) @ P @ L
    Q = 0.5 * (Q + Q.transpose(-1, -2))

    try:
        lam = torch.linalg.eigvalsh(Q)
    except RuntimeError:
        print("[Error] Eigenvalue decomposition failed.")
        return torch.tensor(1e3, device=device), None

    # Penalize eigenvalues outside [0, 2]
    penalty_low  = _softplus_penalty(lam, threshold=0.0)
    penalty_high = _softplus_penalty(2.0 - lam, threshold=0.0)
    total_penalty = penalty_low + penalty_high

    return total_penalty, lam.detach()
################################ZURIEL´s MODIFICATION############################################################


_sqrt2 = np.sqrt(2)

def generate_id(size=8, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits):
    return ''.join(random.SystemRandom().choice(chars) for _ in range(size))

def compute_error_dict(predictions, data, loss_weights, max_errors, beta, num_electrons, restricted = True):
#BETA MAPPING NOTE: (DFTDIFF, PSD, TRE, IDE) (0, 1, 2, 3)
    error_dict = {}
    error_dict['loss'] = 0.0
    restricted_factor = 2.0 if restricted else 1.0
    # Compute initial general errors    
    for key in loss_weights.keys():
        if loss_weights[key] > 0:
            device, dtype = predictions[key].device, predictions[key].dtype
            diff =  data[key] - predictions[key]
            mse  = torch.mean(diff**2)
            mae  = torch.mean(torch.abs(diff))
            dfts1 = torch.mean(torch.sum(diff**2, axis=(-1,-2))) 
            
            if mae > max_errors[key] or dfts1 > max_errors[key]:
                error_dict[key+'_mae']  = torch.tensor(max_errors[key])
                error_dict[key+'_rmse'] = torch.tensor(_sqrt2*max_errors[key])
            else:
                error_dict[key+'_mae']  = mae
                error_dict[key+'_rmse'] = torch.sqrt(mse)
            error_dict['loss'] +=  loss_weights[key]*beta[0]*dfts1

            if key == "density_matrix":
                num_electrons = num_electrons.to(dtype = dtype, device = device)
                S_real = data["overlap_matrix"]

                ose_loss = torch.zeros((), device=device, dtype=dtype)
                ose_loss_add = float(beta[1]) > 1e-6
                with torch.set_grad_enabled(ose_loss_add):
                    calculated_ose_loss, ceigvals = occupation_spectrum_penalty(predictions[key], S_real)
                    if ose_loss_add:
                        ose_loss += calculated_ose_loss
                        
               
                    

                if ose_loss > max_errors[key]:
                    error_dict[key+'_ose']  = torch.tensor(max_errors[key])
                else:
                    error_dict[key+'_ose']  = calculated_ose_loss
                error_dict['loss'] +=  loss_weights[key]*beta[1]*ose_loss
                    
                tre_loss= torch.zeros((), device=device, dtype=dtype)
                tre_loss_add = float(beta[2]) > 1e-6
                with torch.set_grad_enabled(tre_loss_add):
                    calculated_tre_loss = torch.mean((batch_trace(predictions[key]@S_real)/num_electrons-1.0)**2)
                    if tre_loss_add:
                        tre_loss += calculated_tre_loss



                    
                if tre_loss > max_errors[key]:
                    error_dict[key+'_tre']  = torch.tensor(max_errors[key])
                else:
                    error_dict[key+'_tre']  = calculated_tre_loss
                error_dict['loss'] +=  loss_weights[key]*beta[2]*tre_loss    
                ide_loss= torch.zeros((), device=device, dtype=dtype)

                ide_loss_add = float(beta[3]) > 1e-6
                with torch.set_grad_enabled(ide_loss_add):
                    calculated_ide_loss = torch.mean(torch.sum((predictions[key] @ S_real @ predictions[key] - restricted_factor * predictions[key]) ** 2, axis=(-2, -1)) / (restricted_factor * num_electrons) ** 2)
                    if ide_loss_add:
                        ide_loss += calculated_ide_loss
                        
                if ide_loss > max_errors[key]:
                    error_dict[key+'_ide']  = torch.tensor(max_errors[key])
                else:
                    error_dict[key+'_ide']  = calculated_ide_loss
                error_dict['loss'] +=  loss_weights[key]*beta[3]*ide_loss   
                
             
        
        
        
        
        
                        
       
       
    
    return error_dict, ceigvals    
    
def empty_error_dict(loss_weights, fill_value=0.0):
    error_dict = {'loss': fill_value}
    for key in loss_weights.keys():
        if loss_weights[key] > 0:
            error_dict[key + '_mae'] = fill_value
            error_dict[key + '_rmse'] = fill_value
            if key == "density_matrix":
                 error_dict[key + '_ose'] = fill_value
                 error_dict[key + '_ide'] = fill_value
                 error_dict[key + '_tre'] = fill_value
            
    return error_dict
