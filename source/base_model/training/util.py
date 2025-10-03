import string
import random
import torch
import numpy as np
import torch.nn.functional as F

# -------------------- small utils --------------------

_sqrt2 = np.sqrt(2.0)
def generate_id(size=8, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits):
    return ''.join(random.SystemRandom().choice(chars) for _ in range(size))
    
def batch_trace(X):
    if X.ndim == 2:  # single matrix
        return torch.trace(X).unsqueeze(0)
    return torch.einsum("...ii->...", X)

def hinge(x: torch.Tensor, eps=1e-4):
    # smooth-ish hinge: zero inside margin, identity outside
    return torch.where(x <= eps, torch.zeros_like(x), x)



# --------- analytic projection utilities (used for Moreau) ---------

def purify_util_func(P: torch.Tensor, S: torch.Tensor, restricted_scheme=True, purification_iterations=3):
    # McWeeny in AO metric (no eigen-decomp)
    P = 0.5 * (P + P.transpose(-1, -2))
    if restricted_scheme:
        P = 0.5 * P  # operate on 1e- density
    for _ in range(int(purification_iterations)):
        PSP = P @ S @ P
        SP  = S @ P
        P   = 3.0 * PSP - 2.0 * (PSP @ SP)
    if restricted_scheme:
        P = 2.0 * P
    return 0.5 * (P + P.transpose(-1, -2))

def force_trace_util_func(P: torch.Tensor, S: torch.Tensor, num_electrons, eps: float = 1e-6):
    P = 0.5 * (P + P.transpose(-1, -2))
    tr = torch.clamp(batch_trace(P @ S), min=eps)  # shape (B,)
    scale = (num_electrons / tr).view(-1, 1, 1)
    return scale * P

# -------------------- AO partitioning for block loss --------------------

def _shell_sizes_from_irreps(irreps_deg):
    return [2 * int(l) + 1 for l in irreps_deg]

def _build_shell_slices(chemical_dict):
    symbols     = chemical_dict["symbols"]
    irreps_degM = chemical_dict["irreps_deg"]
    ao_slices_atom_shell = []
    ao_slices_atom = []
    offset = 0
    for sym in symbols:
        sizes = _shell_sizes_from_irreps(irreps_degM[sym])
        shell_slices = []
        start_atom = offset
        for sz in sizes:
            shell_slices.append((offset, offset + sz))
            offset += sz
        ao_slices_atom_shell.append(shell_slices)
        ao_slices_atom.append((start_atom, offset))
    return ao_slices_atom_shell, ao_slices_atom, offset  # nao

def block_aware_error(M_pred, M_ref, M_weights, chemical_dict,
                      by="shell", normalize_block=True,
                      weight_gamma=1.0, weight_floor_frac=1e-3):
    """
    Weighted block Frobenius loss in AO space.
    """
    assert M_pred.shape == M_ref.shape == M_weights.shape
    device, dtype = M_pred.device, M_pred.dtype

    shell_slices_atom, atom_slices, nao = _build_shell_slices(chemical_dict)
    assert M_pred.shape[-1] == nao, f"AO dim {M_pred.shape[-1]} != {nao}"

    if by == "atom":
        parts = atom_slices
    elif by == "shell":
        parts = [sl for shells in shell_slices_atom for sl in shells]
    else:
        raise ValueError("by must be 'atom' or 'shell'")

    dM = M_pred - M_ref
    weights = []
    for (rs, re) in parts:
        for (cs, ce) in parts:
            Wblk = M_weights[..., rs:re, cs:ce]
            Wnorm = torch.linalg.matrix_norm(
                Wblk.reshape(-1, re - rs, ce - cs), ord='fro', dim=(-2, -1)
            ).mean()
            if normalize_block:
                Wnorm = Wnorm / float(max(1, (re - rs) * (ce - cs)))
            weights.append(Wnorm)

    weights = torch.stack(weights)
    mean_w = weights.mean().clamp_min(torch.finfo(dtype).eps)
    weights = torch.clamp(weights, min=weight_floor_frac * mean_w) ** weight_gamma
    weights = weights / weights.sum().clamp_min(torch.finfo(dtype).eps)

    total = torch.zeros((), device=device, dtype=dtype)
    idx = 0
    for (rs, re) in parts:
        for (cs, ce) in parts:
            dBlk = dM[..., rs:re, cs:ce]
            if normalize_block:
                blk_err = (dBlk ** 2).mean(dim=(-2, -1)).mean()
            else:
                blk_err = (dBlk ** 2).sum(dim=(-2, -1)).mean()
            total = total + weights[idx] * blk_err
            idx += 1
    return total

# -------------------- main: compute_error_dict (cheap + L/Q reuse) --------------------

def compute_error_dict(predictions, data, loss_weights, max_errors, beta,
                       num_electrons, chemical_dict, purification_iterations,
                       restricted=True,
                       # split the "main" β0 budget between Q-space and AO-block terms
                       main_split_Q_vs_AO=0.5,
                       # Determine from which value MOE will be material
                       moe_threeshold=1e-7,
                       # occupation sorting for stability near degeneracy
                       sort_occs=True):
    """
    Cheaper variant: builds L once, forms Q_pred/Q_ref once, reuses them across terms.
    Returns (metrics_dict, occ_eigs_pred) for logging.
    """
    any_tensor = next(iter(predictions.values()))
    device, dtype = any_tensor.device, any_tensor.dtype

    out = {'loss': torch.zeros((), device=device, dtype=dtype)}
    ceigvals = None  # occ eigvals (pred) for diagnostics
    Ne = num_electrons.to(device=device, dtype=dtype)  # shape (B,) or scalar
    occ_scale = 2.0 if restricted else 1.0  # used in IDE

    for key, w in loss_weights.items():
        if w <= 0:
            continue

        pred = predictions[key]
        ref  = data[key]
        diff = ref - pred
        mse  = (diff ** 2).mean()
        mae  = diff.abs().mean()
        maxv = torch.as_tensor(max_errors[key], device=device, dtype=dtype)

        # generic metrics (also for DM; AO space)
        out[f'{key}_mae']  = torch.minimum(mae,  maxv)
        out[f'{key}_rmse'] = torch.minimum(torch.sqrt(mse), _sqrt2 * maxv)

        if key != "density_matrix":
            base_term = (diff ** 2).sum(dim=(-2, -1)).mean() if diff.ndim >= 2 else (diff ** 2).mean()
            out['loss'] = out['loss'] + w * beta[0] * base_term
            continue

        # ---------------- density-matrix path (reusing L and Q) ----------------
        S  = data["overlap_matrix"]
       

         # (A) AO block-aware loss (S-weighted)
        
        q_frob = (pred - ref).pow(2).sum(dim=(-2, -1)).mean()

        # (B) Q-space global Frobenius. Edit 6th September 20225: working in Q space did not show more effectivity than ao space.
        if main_split_Q_vs_AO  < 1.0:
            ao_block = block_aware_error(pred, ref, S, chemical_dict)
        else:
            ao_block = torch.zeros((), device=device, dtype=dtype)
        

        b0 = float(beta[0])
        out['loss'] += w * b0 * (main_split_Q_vs_AO * q_frob + (1.0 - main_split_Q_vs_AO) * ao_block)

        if float(beta[1]) > 0.0:
            # Build L once (Cholesky with eigen-repair fallback)
            I = torch.eye(S.shape[-1], dtype=dtype, device=device)
            try:
                L = torch.linalg.cholesky(S + 1e-10 * I)
            except RuntimeError:
                s, U = torch.linalg.eigh(S)
                s = torch.clamp(s, min=1e-8)
                L = U @ torch.diag_embed(torch.sqrt(s))
    
            def to_Q(P):
                Q = L.transpose(-1, -2) @ P @ L
                return 0.5 * (Q + Q.transpose(-1, -2))
    
            # Q once for pred/ref
            Q_pred = to_Q(pred)
            Q_ref  = to_Q(ref)
            # (C) Occupation spectrum vs reference (Huber)
            lam_pred = torch.linalg.eigvalsh(Q_pred)
            lam_ref  = torch.linalg.eigvalsh(Q_ref)
            ceigvals = lam_pred.detach()  # for logging/plots
            if sort_occs:
                lam_pred, _ = lam_pred.sort(dim=-1, descending=True)
                lam_ref,  _ = lam_ref.sort(dim=-1,  descending=True)
            diff_occ = lam_pred - lam_ref
            absd = diff_occ.abs()
            delta = 1e-3  # Huber width
            huber = torch.where(absd <= delta, 0.5 * diff_occ**2, delta * (absd - 0.5 * delta))
            calc_ose = huber.mean(dim=-1).mean()           # mean over occs then batch
           
        else:
            calc_ose = torch.zeros((), device=device, dtype=dtype)
                    
        out[f'{key}_ose'] = torch.minimum(calc_ose, maxv)
        out['loss'] += w * beta[1] * calc_ose
        

        # (D) Trace (charge) penalty
        if float(beta[2]) >0.0:
            tr = batch_trace(pred @ S)            # (B,)
            calc_tre = ((tr / Ne) - 1.0).pow(2).mean()
        else:
            calc_tre = torch.zeros((), device=device, dtype=dtype)
        out[f'{key}_tre'] = torch.minimum(calc_tre, maxv)
        out['loss'] += w * beta[2] * hinge(calc_tre)

        if float(beta[3]) >0.0:
        # (E) Idempotency (hinged)
            calc_ide = ((pred @ S @ pred - occ_scale * pred) ** 2).sum(dim=(-2, -1)).mean() / (occ_scale * Ne).pow(2)
        else:
            calc_ide =  torch.zeros((), device=device, dtype=dtype)
        out[f'{key}_ide'] = torch.minimum(calc_ide, maxv)
        out['loss'] += w * beta[3] * hinge(calc_ide)

        
        if float(beta[4]) >0.0:
        # (F) Moreau / projection stabilizer (compare repaired vs pred in Q-space; reuse L)
            with torch.set_grad_enabled(False):
                P_rep = force_trace_util_func(pred, S, Ne)
                P_rep = purify_util_func(P_rep, S, restricted_scheme=restricted,
                                         purification_iterations=purification_iterations)
                P_rep = force_trace_util_func(P_rep, S, Ne)
    
            #Q_rep = to_Q(P_rep)
            calc_moe = (P_rep - pred).pow(2).sum(dim=(-2, -1)).mean()
        else:
            calc_moe, P_rep =  torch.zeros((), device=device, dtype=dtype), None
        out[f'{key}_moe'] = torch.minimum(calc_moe, maxv)
        out['loss'] += w * beta[4] * F.softplus(calc_moe-moe_threeshold)

    return out, ceigvals, P_rep
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
                 error_dict[key + '_moe'] = fill_value
            
    return error_dict
