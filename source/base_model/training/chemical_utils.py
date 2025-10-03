#!/usr/bin/env python3
import torch
from pyscf import gto, scf, dft
import numpy as np
import time
from contextlib import contextmanager
from pyscf.scf import hf as _hf  # for generalized eigproblem F C = S C eps
from argparse import Namespace
from pyscf.grad import rhf
from ase.io import read
from ase.units import Bohr
from typing import Tuple
from typing import List
import math
import os



###############PySCF STUFF###################################

mol_sym = "NITRATE"
basis_set =  "def2-SVPD"
unit = "Bohr"
charge = -1


# -------- Element settings / AO patterns --------
chemical_info_dict = {
    "H2O": {
        "symbols": ["O","H","H"],
         "irreps_deg": {"O": (0,0,0,1,1,2), "H": (0,0,1)}
    },
    "NH3": {
        "symbols": ["N","H","H","H"],
         "irreps_deg": {"N": (0,0,0,1,1,2), "H": (0,0,1)}
    },
    "CH4": {
        "symbols": ["C","H","H","H","H"],
         "irreps_deg":{"C": (0,0,0,1,1,2), "H": (0,0,1)}
    },
    "FH": {
        "symbols": ["F","H"],
         "irreps_deg":{"F": (0,0,0,1,1,2), "H": (0,0,1)}
    },
    "CH3CH2OH": {
        "symbols": ["C","C","O","H","H","H","H","H","H"],
         "irreps_deg":{"C": (0,0,0,1,1,2),
                      "O": (0,0,0,1,1,2),
                      "H": (0,0,1)}
    },
    "NITRATE": {
        "symbols": ["N","O","O","O"],
        # for def2-SVPD: N = 4s+2p+2d ; O = 4s+3p+2d
         "irreps_deg":{"N": (0,0,0,0,1,1,2,2),
                      "O": (0,0,0,0,1,1,1,2,2)}
    }
}


chemical_dict = chemical_info_dict[mol_sym]
chemical_symbols = chemical_dict["symbols"]
atoms_args = "".join(chemical_symbols)
def create_pyscf_molecule(coordinates: np.ndarray, chemical_symbols: List = chemical_symbols, basis_set: str = basis_set):
    if len(chemical_symbols) != coordinates.shape[0]:
        raise ValueError("Number of chemical symbols must match the number of atom coordinates.")
    
    atom_data = [[chemical_symbols[i], coordinates[i].tolist()] for i in range(len(chemical_symbols))]
    
    mol = gto.M(
        atom=atom_data,
        basis=basis_set,
        unit = unit,
        charge = charge,
        verbose=0
    )
    mol.build()
    return mol

def scf_iterations(preds, positions, Norb):
    positions = (1 * positions).detach().cpu().view(-1, len(chemical_symbols), 3).numpy()
    preds = (1 * preds).detach().cpu().view(-1, Norb, Norb).numpy()
    mols = [create_pyscf_molecule(positions[i]) for i in range(len(positions))]
    preds = transform(preds, convention = "phipy", atoms = atoms)
    
    niters = []
    for mol, dm in zip(mols, preds):
        mf = dft.RKS(mol).density_fit(auxbasis="def2-universal-jkfit")  # RI-J like ORCA
        mf.xc = 'PBE'
        mf.diis_space = 12              # robust DIIS
        mf.max_cycle = 180
        mf.conv_tol = 1e-9              # total energy conv (Eh)
        mf.conv_tol_grad = 1e-6         # |dE/dR| (Eh/Bohr) ~ VeryTight
        mf.conv_tol_density = 1e-8
        mf.small_rho_cutoff = 1e-10
# Practical convergence helpers ORCA often uses under the hood
        mf.level_shift = 0.3            # soften near-degeneracies in early cycles
        mf.damp = 0.2

# Coarse grid for SCF (≈ ORCA Grid4): (75,302) is a good proxy
        mf.grids.atom_grid = (75, 302)
        mf.grids.prune = True        
        counter = {'n': 0}
        def count_iterations(envs): counter['n'] += 1
        mf.callback = count_iterations
        mf.kernel(dm0=dm)
        niters.append(counter['n'])
    return niters    

###############PySCF STUFF###################################

 
convention_dict = {
   'pyscf': Namespace(
       atom_to_orbitals_map={'H': 'ssp', 'O': 'sssppd', 'C': 'sssppd', 'N': 'sssppd', 'F': 'sssppd'},
       orbital_idx_map={'s': [0], 'p': [1, 2, 0], 'd': [4, 2, 0, 1, 3]},
       orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
       orbital_order_map={'H': [0, 1, 2], 'O': [0, 1, 2, 3, 4, 5], 'C': [0, 1, 2, 3, 4, 5], 'N': [0, 1, 2, 3, 4, 5], 'F': [0, 1, 2, 3, 4, 5]},
   ),

   'orca': Namespace(
       atom_to_orbitals_map={'H': 'ssp', 'O': 'sssppd', 'C': 'sssppd', 'N': 'sssppd', 'F': 'sssppd'},
       orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [4, 2, 0, 1, 3]},
       orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
       orbital_order_map={'H': [0, 1, 2], 'O': [0, 1, 2, 3, 4, 5], 'C': [0, 1, 2, 3, 4, 5], 'N': [0, 1, 2, 3, 4, 5], 'F': [0, 1, 2, 3, 4, 5]},
   ),

   'pysch': Namespace(
       atom_to_orbitals_map={'H': 'ssp', 'O': 'sssppd', 'C': 'sssppd', 'N': 'sssppd', 'F': 'sssppd'},
       orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [2, 3, 1, 4, 0]},
       orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
       orbital_order_map={'H': [0, 1, 2], 'O': [0, 1, 2, 3, 4, 5], 'C': [0, 1, 2, 3, 4, 5], 'N': [0, 1, 2, 3, 4, 5], 'F': [0, 1, 2, 3, 4, 5]},
   ),

   'pyphi': Namespace(
       atom_to_orbitals_map={'H': 'ssp', 'O': 'sssppd', 'C': 'sssppd', 'N': 'sssppd', 'F': 'sssppd'},
       orbital_idx_map={'s': [0], 'p': [1, 2, 0], 'd': [0, 1, 2, 3, 4]},
       orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
       orbital_order_map={'H': [0, 1, 2], 'O': [0, 1, 2, 3, 4, 5], 'C': [0, 1, 2, 3, 4, 5], 'N': [0, 1, 2, 3, 4, 5], 'F': [0, 1, 2, 3, 4, 5]},
   ),

   'phipy': Namespace(
       atom_to_orbitals_map={'H': 'ssp', 'O': 'sssppd', 'C': 'sssppd', 'N': 'sssppd', 'F': 'sssppd'},
       orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [0, 1, 2, 3, 4]},
       orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
       orbital_order_map={'H': [0, 1, 2], 'O': [0, 1, 2, 3, 4, 5], 'C': [0, 1, 2, 3, 4, 5], 'N': [0, 1, 2, 3, 4, 5], 'F': [0, 1, 2, 3, 4, 5]},
   ),
}


ion_convention_dict = {
   'pyscf': Namespace(
       atom_to_orbitals_map={'H': 'ssp', 'O': 'sssspppdd', 'C': 'sssppd', 'N': 'ssssppdd', 'F': 'sssppd'},
       orbital_idx_map={'s': [0], 'p': [1, 2, 0], 'd': [4, 2, 0, 1, 3]},
       orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
       orbital_order_map={'H': [0, 1, 2], 'O': [0, 1, 2, 3, 4, 5], 'C': [0, 1, 2, 3, 4, 5], 'N': [0, 1, 2, 3, 4, 5], 'F': [0, 1, 2, 3, 4, 5]},
   ),

   'orca': Namespace(
       atom_to_orbitals_map={'H': 'ssp', 'O': 'sssspppdd', 'C': 'sssppd', 'N': 'ssssppdd', 'F': 'sssppd'},
       orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [4, 2, 0, 1, 3]},
       orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
       orbital_order_map={'H': [0, 1, 2], 'O': [0, 1, 2, 3, 4, 5], 'C': [0, 1, 2, 3, 4, 5], 'N': [0, 1, 2, 3, 4, 5], 'F': [0, 1, 2, 3, 4, 5]},
   ),

   'pysch': Namespace(
      atom_to_orbitals_map={'H': 'ssp', 'O': 'sssspppdd', 'C': 'sssppd', 'N': 'ssssppdd', 'F': 'sssppd'},
       orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [2, 3, 1, 4, 0]},
       orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
       orbital_order_map={'H': [0, 1, 2], 'O': [0, 1, 2, 3, 4, 5], 'C': [0, 1, 2, 3, 4, 5], 'N': [0, 1, 2, 3, 4, 5], 'F': [0, 1, 2, 3, 4, 5]},
   ),

   'pyphi': Namespace(
       atom_to_orbitals_map={'H': 'ssp', 'O': 'sssspppdd', 'C': 'sssppd', 'N': 'ssssppdd', 'F': 'sssppd'},
       orbital_idx_map={'s': [0], 'p': [1, 2, 0], 'd': [0, 1, 2, 3, 4]},
       orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
       orbital_order_map={'H': [0, 1, 2], 'O': [0, 1, 2, 3, 4, 5], 'C': [0, 1, 2, 3, 4, 5], 'N': [0, 1, 2, 3, 4, 5], 'F': [0, 1, 2, 3, 4, 5]},
   ),

   'phipy': Namespace(
      atom_to_orbitals_map={'H': 'ssp', 'O': 'sssspppdd', 'C': 'sssppd', 'N': 'ssssppdd', 'F': 'sssppd'},
       orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [0, 1, 2, 3, 4]},
       orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
       orbital_order_map={'H': [0, 1, 2], 'O': [0, 1, 2, 3, 4, 5], 'C': [0, 1, 2, 3, 4, 5], 'N': [0, 1, 2, 3, 4, 5], 'F': [0, 1, 2, 3, 4, 5]},
   ),
}




def transform(hamiltonians,convention, atoms = atoms_args, charge = charge):
    conv = convention_dict[convention] if charge == 0 else ion_convention_dict[convention] 
    #print('atoms', atoms)
    orbitals = ''
    orbitals_order = []
    for a in atoms:
        offset = len(orbitals_order)
        #print('svr aroms to orbs', conv.atom_to_orbitals_map[a])
        orbitals += conv.atom_to_orbitals_map[a]
        orbitals_order += [idx + offset for idx in conv.orbital_order_map[a]]

    #print('orbitals', orbitals)
    #print('orbitals order', orbitals_order)

    transform_indices = []
    transform_signs = []
    for orb in orbitals:
        offset = sum(map(len, transform_indices))
        map_idx = conv.orbital_idx_map[orb]
        map_sign = conv.orbital_sign_map[orb]
        transform_indices.append(np.array(map_idx) + offset)
        transform_signs.append(np.array(map_sign))

    transform_indices = [transform_indices[idx] for idx in orbitals_order]
    transform_signs = [transform_signs[idx] for idx in orbitals_order]
    #print('transform_indices', transform_indices)
    transform_indices = np.concatenate(transform_indices).astype(int)
    transform_signs = np.concatenate(transform_signs)


    hamiltonians_new = hamiltonians[...,transform_indices, :]
    hamiltonians_new = hamiltonians_new[...,:, transform_indices]
    hamiltonians_new = hamiltonians_new * transform_signs[:, None]
    hamiltonians_new = hamiltonians_new * transform_signs[None, :]

    return hamiltonians_new
    
    
    
    
    

# ============================== Linear algebra & diagnostics ==============================
def symm(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)

def cholesky_S(S: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    S = symm(S); n = S.shape[0]
    return np.linalg.cholesky(S + eps * np.eye(n))

def occupations_from_P(P: np.ndarray, S: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    L = cholesky_S(S, eps=eps)
    Q = symm(L.T @ P @ L)
    occs = np.linalg.eigvalsh(Q)
    return np.sort(occs)

def idempotency_error(P: np.ndarray, S: np.ndarray) -> float:
    # RKS closed-shell convention: PSP - 2P
    X = P @ S @ P - 2.0 * P
    return float(np.linalg.norm(X, "fro"))

def sweighted_fro_norm(A: np.ndarray, S: np.ndarray, eps: float = 1e-12) -> float:
    L = cholesky_S(S, eps=eps)
    B = L.T @ A @ L
    return float(np.linalg.norm(B, "fro"))

def trace_electrons(P: np.ndarray, S: np.ndarray) -> float:
    return float(np.trace(P @ S))

def rescale_trace(P: np.ndarray, S: np.ndarray, N_electrons: int) -> Tuple[float, np.ndarray]:
    tr = trace_electrons(P, S)
    if abs(tr) < 1e-14:
        return 1.0, symm(P.copy())
    alpha = N_electrons / tr
    return float(alpha), symm(alpha * P)

@contextmanager
def time_block():
    t0 = time.perf_counter()
    yield lambda: time.perf_counter() - t0

# ============================== ORCA-like df-JK PBE driver ==============================
def _attach_counter(mf):
    itercount = {"n": 0}
    def cb(_): itercount["n"] += 1
    mf.callback = cb
    return itercount

def make_orca_like_rks_df(mol, verbose=0):
    """
    ORCA-like SCF controls with density fitting:
      - df-JK with def2-svp-jkfit
      - PBE, pruned grid, atom_grid ~(75,302) ≈ ORCA Grid4
      - VeryTight-ish thresholds, damping/level_shift early stabilizers
      - init_guess='hcore' (ORCA default)
    """
    mf = dft.RKS(mol).density_fit(auxbasis="def2-universal-jkfit")

    # XC and grids
    mf.xc = 'PBE'
    mf.grids.prune = True
    mf.grids.atom_grid = (75, 302)  # ~ORCA Grid4

    # SCF controls
    mf.init_guess = 'hcore'
    mf.max_cycle = 180
    mf.diis_space = 12
    mf.conv_tol = 1e-9
    if hasattr(mf, 'conv_tol_grad'):
        mf.conv_tol_grad = 1e-6
    if hasattr(mf, 'conv_tol_density'):
        mf.conv_tol_density = 1e-8

    # Integration hygiene
    if hasattr(mf, 'small_rho_cutoff'):
        mf.small_rho_cutoff = 1e-10

    # Early stabilizers
    if hasattr(mf, 'level_shift'):
        mf.level_shift = 0.3
    if hasattr(mf, 'damp'):
        mf.damp = 0.2
    else:
        try:
            from pyscf.scf import addons as _addons
            mf = _addons.damping_(mf, factor=0.2)
        except Exception:
            pass

    mf.verbose = int(verbose)
    return mf

def run_orca_like_pbe_df(mol, dm0=None, newton_rescue: bool = True, verbose: int = 0):
    """
    ORCA-like PBE/df-JK single-stage SCF.
      - If dm0 is provided → seed with dm0 (mf.init_guess is ignored)
      - Else mf.init_guess='hcore' (baseline)
    Returns: (mf, E_final, stats)
    """
    mf = make_orca_like_rks_df(mol, verbose=verbose)
    # Use dm0 if provided (don't override stabilizers)
    itercount = _attach_counter(mf)
    stats = {
        "stage": "PBE/df-JK (ORCA-like Grid4)",
        "iters": 0, "time_s": 0.0, "converged": False,
        "newton": {"used": False, "iters": 0, "time_s": 0.0, "converged": False},
        "total_iters": 0, "total_time_s": 0.0,
    }
    with time_block() as elapsed:
        e_final = mf.kernel(dm0=dm0)
    t = elapsed()
    stats["iters"] = itercount["n"]
    stats["time_s"] = t
    stats["converged"] = bool(getattr(mf, "converged", False))
    stats["total_iters"] += itercount["n"]
    stats["total_time_s"] += t

    # Optional Newton/SOSCF rescue (remove stabilizers)
    if newton_rescue and not mf.converged:
        stats["newton"]["used"] = True
        if hasattr(mf, 'level_shift'): mf.level_shift = 0.0
        if hasattr(mf, 'damp'):        mf.damp = 0.0
        try:
            mfN = mf.newton()
        except Exception:
            mfN = scf.newton(mf)
        itercount = _attach_counter(mfN)
        with time_block() as elapsed:
            e_final = mfN.kernel(dm0=mf.make_rdm1())
        tN = elapsed()
        stats["newton"]["iters"] = itercount["n"]
        stats["newton"]["time_s"] = tN
        stats["newton"]["converged"] = bool(getattr(mfN, "converged", False))
        stats["total_iters"] += itercount["n"]
        stats["total_time_s"] += tN
        mf = mfN

    return mf, e_final, stats

# ============================== Direct physics from a given DM ==============================
def _safe_energy_elec(mf, dm, h1e, veff):
    e = mf.energy_elec(dm=dm, h1e=h1e, vhf=veff)
    return float(e[0]) if isinstance(e, (tuple, list)) else float(e)

def single_shot_from_dm(mol, dm, verbose=0):
    """
    Evaluate PBE/df-JK quantities ONCE from the provided density matrix (no SCF):
      - Build S, h1e, veff = V_H + V_xc from dm using ORCA-like PBE/df-JK settings
      - Form KS F and solve F C = S C ε
      - Compute total energy E_tot(dm) = E_elec(dm) + E_nuc
    Returns dict: E_tot_Ha, eps, C, S
    """
    dm = symm(np.asarray(dm, dtype=np.float64))
    mf = make_orca_like_rks_df(mol, verbose=verbose)

    S   = mol.intor("int1e_ovlp_sph")
    h1e = mf.get_hcore(mol)
    veff = mf.get_veff(mol, dm)
    F = mf.get_fock(h1e=h1e, s1e=S, vhf=veff, dm=dm)

    eps, C = _hf.eig(F, S)
    e_elec = _safe_energy_elec(mf, dm, h1e, veff)
    e_tot  = e_elec + mf.energy_nuc()

    return {"E_tot_Ha": float(e_tot),
            "eps": np.asarray(eps, dtype=float),
            "C":   np.asarray(C,   dtype=float),
            "S":   np.asarray(S,   dtype=float)}

def occupied_subspace_svals(CA, CB, S, nocc):
    OA = CA[:, :nocc]; OB = CB[:, :nocc]
    O  = OA.T @ (S @ OB)
    return np.linalg.svd(O, compute_uv=False).astype(float)

# ============================== Molecule builder (generic) ==============================
def build_mol_from_bohr(coords_bohr: np.ndarray,
                        chemical_symbols: List[str],
                        basis: str = "def2-SVP",
                        unit: str = "Bohr",
                        charge: int = charge,
                        spin: int = 0):
    """
    coords_bohr: (nat,3) in Bohr, atom order matches 'chemical_symbols'.
    spin: 2S = (#alpha - #beta); for closed-shell even-electron systems use 0.
    """
    R = np.asarray(coords_bohr, dtype=float)
    if R.ndim != 2 or R.shape[1] != 3 or R.shape[0] != len(chemical_symbols):
        raise ValueError(f"coords shape {R.shape} incompatible with symbols {len(chemical_symbols)}")
    atom = [(sym, tuple(R[i])) for i, sym in enumerate(chemical_symbols)]
    mol = gto.Mole()
    mol.atom = atom
    mol.basis = basis
    mol.unit = unit
    mol.charge = charge
    mol.spin = spin
    mol.build()
    return mol

# ============================== Evaluator (SCF & direct) ==============================
BASELINE_LABEL = "hcore (ORCA-like PBE/df-JK Grid4)"
def evaluate_dm_guess(mol, dm_pred: np.ndarray, name: str = "system",
                      enforce_trace: bool = False, eps: float = 1e-12,
                      newton_rescue: bool = True, verbose: int = 0):
    """
    (A) Baseline SCF (hcore) vs SCF seeded by predicted DM → iterations/time + SCF-vs-SCF ε, ψ.
    (B) Direct single-shot physics from predicted DM (no SCF) vs SCF reference → energy, ε, ψ.
    """
    nao = mol.nao
    assert dm_pred.shape == (nao, nao), f"dm_pred shape {dm_pred.shape} != ({nao},{nao})"
    dm_pred = symm(dm_pred).astype(np.float64, copy=False)

    S   = mol.intor("int1e_ovlp_sph")
    N_e = mol.nelectron
    nocc = N_e // 2  # RKS closed-shell assumption

    # Raw diagnostics on the NN guess (no fixups used for direct branch)
    occs_raw = occupations_from_P(dm_pred, S, eps=eps)
    tr_raw   = trace_electrons(dm_pred, S)
    idem_raw = idempotency_error(dm_pred, S)

    # Optional trace-corrected seed (SCF seeding only; direct branch uses raw P)
    alpha, dm_corr = rescale_trace(dm_pred, S, N_e)
    dm0_to_use = dm_corr if enforce_trace else dm_pred
    tag = "trace-corrected" if enforce_trace else "raw"

    # ---- Reference SCF (baseline: hcore) ----
    mf_ref, e_ref, stats_ref = run_orca_like_pbe_df(mol, dm0=None, newton_rescue=newton_rescue, verbose=verbose)
    P_ref   = mf_ref.make_rdm1()
    eps_ref = np.asarray(mf_ref.mo_energy, dtype=float)
    C_ref   = np.asarray(mf_ref.mo_coeff, dtype=float)

    # ---- SCF seeded with your DM ----
    mf_pred, e_pred, stats_pred = run_orca_like_pbe_df(mol, dm0=dm0_to_use, newton_rescue=newton_rescue, verbose=verbose)
    P_pred_final = mf_pred.make_rdm1()

    # ΔP vs reference final P
    dP = P_pred_final - P_ref
    dP_frob   = float(np.linalg.norm(dP, "fro"))
    dP_sw_fro = sweighted_fro_norm(dP, S)

    # SCF energy and ε differences
    dE_scf = float(e_pred - e_ref)
    eps_pred_scf = np.asarray(mf_pred.mo_energy, dtype=float)
    nscf = min(eps_ref.size, eps_pred_scf.size)
    de_scf = eps_pred_scf[:nscf] - eps_ref[:nscf]
    eps_mae_uEh_scf = 1e6 * float(np.mean(np.abs(de_scf)))
    eps_max_uEh_scf = 1e6 * float(np.max(np.abs(de_scf)))
    C_pred_scf = np.asarray(mf_pred.mo_coeff, dtype=float)
    svals_scf = occupied_subspace_svals(C_ref, C_pred_scf, S, nocc)
    psi_mean_scf = float(np.mean(svals_scf)); psi_min_scf  = float(np.min(svals_scf))

    # ---- Direct single-shot physics from your predicted DM (no SCF, raw P) ----
    ss = single_shot_from_dm(mol, dm_pred, verbose=0)
    E_tot_predDM = ss["E_tot_Ha"]; eps_predDM = ss["eps"]; C_predDM = ss["C"]
    nd = min(eps_ref.size, eps_predDM.size)
    de_direct = eps_predDM[:nd] - eps_ref[:nd]
    eps_mae_uEh_direct = 1e6 * float(np.mean(np.abs(de_direct)))
    eps_max_uEh_direct = 1e6 * float(np.max(np.abs(de_direct)))
    svals_direct = occupied_subspace_svals(C_ref, C_predDM, S, nocc)
    psi_mean_direct = float(np.mean(svals_direct)); psi_min_direct = float(np.min(svals_direct))
    dE_direct = float(E_tot_predDM - e_ref)

    same_stationary = abs(dE_scf) < 1e-10

    return {
        "name": name,
        "nao": nao,
        "N_electrons": N_e,
        "AO_ordering": "PySCF spherical",

        "SCF_baseline": {
            "E_final_Ha": float(e_ref),
            "iters_total": stats_ref["total_iters"],
            "time_total_s": stats_ref["total_time_s"],
            "converged": bool(getattr(mf_ref, "converged", False)),
        },
        "SCF_from_pred": {
            "dm_used": tag,
            "E_final_Ha": float(e_pred),
            "iters_total": stats_pred["total_iters"],
            "time_total_s": stats_pred["total_time_s"],
            "converged": bool(getattr(mf_pred, "converged", False)),
            "same_stationary_point": bool(same_stationary),
            "iteration_savings": stats_ref["total_iters"] - stats_pred["total_iters"],
            "time_savings_s": stats_ref["total_time_s"] - stats_pred["total_time_s"],
        },

        # ΔP vs final reference
        "P_metrics": {
            "||ΔP||_F": dP_frob,
            "||S^(1/2)ΔP S^(1/2)||_F": dP_sw_fro,
            "scaled_1e-6(||ΔP||_F)": 1e6 * dP_frob,
            "scaled_1e-6(||S^(1/2)ΔP S^(1/2)||_F)": 1e6 * dP_sw_fro,
        },

        # ε & ψ (SCF-vs-SCF)
        "epsilon_metrics": {"MAE_μEh": eps_mae_uEh_scf, "MaxAbs_μEh": eps_max_uEh_scf},
        "psi_metrics": {"mean": psi_mean_scf, "min": psi_min_scf},

        # Input guess diagnostics (raw P_pred)
        "diagnostics_guess": {
            "Tr(PS)": float(tr_raw),
            "Tr(PS)-N": float(tr_raw - N_e),
            "occ_min": float(occs_raw.min()),
            "occ_max": float(occs_raw.max()),
            "idempotency_Frob": float(idem_raw),
            "alpha_if_trace_corrected": float(alpha),
        },

        # SCF energy difference
        "delta_E": {"Ha": dE_scf, "μEh": 1e6 * dE_scf},

        # Direct single-shot physics from predicted DM (raw)
        "direct_from_pred_dm": {
            "E_tot_Ha": E_tot_predDM,
            "delta_E_direct": {"Ha": dE_direct, "μEh": 1e6 * dE_direct},
            "epsilon_vs_ref": {"MAE_μEh": eps_mae_uEh_direct, "MaxAbs_μEh": eps_max_uEh_direct},
            "psi_vs_ref": {"mean": psi_mean_direct, "min": psi_min_direct},
        },

        # Expose ref_dm and S for dataset aggregation
        "ref_dm": P_ref,
        "S": S,
    }

# ============================== Pretty printer ==============================
def print_benchmarks(rep):
    b = rep["SCF_baseline"]; p = rep["SCF_from_pred"]
    Pm = rep["P_metrics"]; Em = rep["epsilon_metrics"]; Psi = rep["psi_metrics"]; dE = rep["delta_E"]
    D  = rep["direct_from_pred_dm"]

    print(f"\n=== {rep['name']} ===")
    print(f"nao = {rep['nao']}, N_e = {rep['N_electrons']}, AO ordering = {rep['AO_ordering']}")
    print("\nSCF (ORCA-like PBE/df-JK, Grid4):")
    print(f"  Baseline (hcore): iters = {b['iters_total']:>3d}, time = {b['time_total_s']:.3f} s, E = {b['E_final_Ha']:.12f} Ha")
    print(f"  From NN {p['dm_used']}: iters = {p['iters_total']:>3d}, time = {p['time_total_s']:.3f} s, E = {p['E_final_Ha']:.12f} Ha")
    print(f"  Savings: Δiters = {p['iteration_savings']:+d}, Δtime = {p['time_savings_s']:+.3f} s")
    print(f"  SCF Energy diff: ΔE = {dE['Ha']:+.12e} Ha ({dE['μEh']:+.3f} μEh)")
    print(f"  Same stationary point flag: {p['same_stationary_point']}")

    print("\nTable-style metrics (ΔP vs ref final):")
    print("  P [×10^-6, dimensionless]")
    print(f"    ||ΔP||_F (μ-like): {Pm['scaled_1e-6(||ΔP||_F)']:.3f}")
    print(f"    ||S^(1/2)ΔP S^(1/2)||_F (μ-like): {Pm['scaled_1e-6(||S^(1/2)ΔP S^(1/2)||_F)']:.3f}")
    print("  ε (SCF-vs-SCF) [μEh]")
    print(f"    MAE: {Em['MAE_μEh']:.2f}   MaxAbs: {Em['MaxAbs_μEh']:.2f}")
    print("  ψ (SCF-vs-SCF) [cosine of principal angles]")
    print(f"    mean: {Psi['mean']:.6f}   min: {Psi['min']:.6f}")

    print("\nDirect-from-pred DM (no SCF) vs ref SCF:")
    print(f"  E_tot(P_pred) - E_ref: {D['delta_E_direct']['Ha']:+.12e} Ha ({D['delta_E_direct']['μEh']:+.3f} μEh)")
    print(f"  ε(P_pred) vs ε_ref [μEh]: MAE = {D['epsilon_vs_ref']['MAE_μEh']:.2f}, MaxAbs = {D['epsilon_vs_ref']['MaxAbs_μEh']:.2f}")
    print(f"  ψ(P_pred eigvecs vs ref occ): mean = {D['psi_vs_ref']['mean']:.6f}, min = {D['psi_vs_ref']['min']:.6f}")

    dg = rep["diagnostics_guess"]
    print("\nInput-guess diagnostics (raw P_pred):")
    print(f"  Tr(PS) = {dg['Tr(PS)']:.8f}  (Tr(PS)-N = {dg['Tr(PS)-N']:+.2e})")
    print(f"  occ range: [{dg['occ_min']:+.3e}, {dg['occ_max']:+.3e}]")
    print(f"  idempotency ‖PSP−2P‖_F = {dg['idempotency_Frob']:.3e}")

# ============================== Dataset runner → one .txt ==============================
def _safe_mean(xs):
    xs = np.asarray(xs, dtype=float)
    return float(np.mean(xs)) if xs.size else float("nan")

def _flatten_metrics(rep):
    base = rep["SCF_baseline"]; pred = rep["SCF_from_pred"]
    Pm = rep["P_metrics"]; Em = rep["epsilon_metrics"]; Psi = rep["psi_metrics"]
    dE = rep["delta_E"]; dg = rep["diagnostics_guess"]; D = rep["direct_from_pred_dm"]
    return {
        # SCF summary
        "iters_baseline": base["iters_total"],
        "iters_pred": pred["iters_total"],
        "iters_savings": pred["iteration_savings"],
        "time_baseline_s": base["time_total_s"],
        "time_pred_s": pred["time_total_s"],
        "time_savings_s": pred["time_savings_s"],
        "E_baseline_Ha": base["E_final_Ha"],
        "E_pred_Ha": pred["E_final_Ha"],
        "dE_Ha": dE["Ha"],
        "dE_uEh": dE["μEh"],
        # ΔP table
        "P_norm_F": Pm["||ΔP||_F"],
        "P_Sw_norm_F": Pm["||S^(1/2)ΔP S^(1/2)||_F"],
        "P_norm_F_x1e6": Pm["scaled_1e-6(||ΔP||_F)"],
        "P_Sw_norm_F_x1e6": Pm["scaled_1e-6(||S^(1/2)ΔP S^(1/2)||_F)"],
        # ε & ψ (SCF-vs-SCF)
        "eps_MAE_uEh": Em["MAE_μEh"], "eps_MaxAbs_uEh": Em["MaxAbs_μEh"],
        "psi_mean": Psi["mean"], "psi_min": Psi["min"],
        # Raw prediction sanity
        "TrPS_minus_N": dg["Tr(PS)-N"],
        "occ_min": dg["occ_min"],
        "occ_max": dg["occ_max"],
        "idem_Frob": dg["idempotency_Frob"],
        # Direct-from-pred (no SCF)
        "dE_direct_Ha": D["delta_E_direct"]["Ha"],
        "dE_direct_uEh": D["delta_E_direct"]["μEh"],
        "eps_direct_MAE_uEh": D["epsilon_vs_ref"]["MAE_μEh"],
        "eps_direct_MaxAbs_uEh": D["epsilon_vs_ref"]["MaxAbs_μEh"],
        "psi_direct_mean": D["psi_vs_ref"]["mean"],
        "psi_direct_min": D["psi_vs_ref"]["min"],
    }

def evaluate_dataset_to_txt(dms_valid, rs_valid, chemical_symbols,
                            outfile="benchmarks_summary.txt",
                            enforce_trace=False, basis="def2-SVP",
                            newton_rescue=True, verbose=0, print_each=False,
                            unit="Bohr", charge=0, spin=0):
    """
    Writes per-sample (optional) and dataset summary to one .txt file.
    Assumes all samples share the same atom order as 'chemical_symbols'.
    """
    n = len(dms_valid); assert n == len(rs_valid)
    reports, rows = [], []

    # Extra raw-prediction accumulators vs each sample’s ref DM
    trace_err_abs_list = []
    mae_vs_ref_list    = []
    occ_min_list, occ_max_list, idem_pred_list = [], [], []

    with open(outfile, "w") as fh:
        fh.write(f"# Baseline used: {BASELINE_LABEL}\n")
        fh.write(f"# enforce_trace on SCF seed: {enforce_trace}\n")
        fh.write(f"# atom order: {chemical_symbols}\n\n")

        for i in range(n):
            mol = build_mol_from_bohr(rs_valid[i], chemical_symbols, basis=basis, unit=unit, charge=charge, spin=spin)
            dm_pred = np.asarray(dms_valid[i], dtype=np.float64)
            if dm_pred.shape != (mol.nao, mol.nao):
                raise ValueError(f"[{i}] dm shape {dm_pred.shape} != ({mol.nao},{mol.nao})")

            rep = evaluate_dm_guess(
                mol, dm_pred, name=f"sample {i}",
                enforce_trace=enforce_trace, newton_rescue=newton_rescue, verbose=verbose
            )
            reports.append(rep); rows.append(_flatten_metrics(rep))

            # Raw-prediction sanity (vs this geometry’s ref DM)
            P_ref = rep["ref_dm"]; S = rep["S"]
            tr_err_abs = abs(trace_electrons(dm_pred, S) - mol.nelectron)
            mae_vs_ref = float(np.mean(np.abs(dm_pred - P_ref)))
            occs = occupations_from_P(dm_pred, S)
            idem_pred = idempotency_error(dm_pred, S)
            trace_err_abs_list.append(tr_err_abs); mae_vs_ref_list.append(mae_vs_ref)
            occ_min_list.append(float(occs.min())); occ_max_list.append(float(occs.max()))
            idem_pred_list.append(idem_pred)

            if print_each:
                from io import StringIO
                import sys
                buf, sys_stdout = StringIO(), sys.stdout
                sys.stdout = buf
                print_benchmarks(rep)
                sys.stdout = sys_stdout
                fh.write(buf.getvalue() + "\n")

        # Means/stds
        keys = rows[0].keys()
        means = {k: _safe_mean([r[k] for r in rows]) for k in keys}
        stds  = {f"{k}_std": float(np.std([r[k] for r in rows], ddof=1)) for k in keys}

        # Raw-prediction means
        mean_abs_trace_err = _safe_mean(trace_err_abs_list)
        mean_mae_vs_ref    = _safe_mean(mae_vs_ref_list)
        mean_occ_min       = _safe_mean(occ_min_list)
        mean_occ_max       = _safe_mean(occ_max_list)
        mean_idem_pred     = _safe_mean(idem_pred_list)

        # Convenience locals
        b_iter_mean = means["iters_baseline"]; p_iter_mean = means["iters_pred"]
        b_time_mean = means["time_baseline_s"]; p_time_mean = means["time_pred_s"]

        fh.write("\n================ Dataset Summary ================\n")
        fh.write(f"samples: {n}   basis: {basis}   baseline: {BASELINE_LABEL}\n")
        fh.write(f"seed enforce_trace: {enforce_trace}\n\n")

        fh.write("SCF (ORCA-like PBE/df-JK Grid4):   (mean ± std)\n")
        fh.write(f"  iters baseline: {b_iter_mean:.2f} ± {stds['iters_baseline_std']:.2f}\n")
        fh.write(f"  iters pred    : {p_iter_mean:.2f} ± {stds['iters_pred_std']:.2f}\n")
        fh.write(f"  Δiters (save) : {means['iters_savings']:+.2f} ± {stds['iters_savings_std']:.2f}\n")
        fh.write(f"  time baseline : {b_time_mean:.3f} ± {stds['time_baseline_s_std']:.3f} s\n")
        fh.write(f"  time pred     : {p_time_mean:.3f} ± {stds['time_pred_s_std']:.3f} s\n")
        fh.write(f"  Δtime (save)  : {means['time_savings_s']:+.3f} ± {stds['time_savings_s_std']:.3f} s\n")
        fh.write(f"  ΔE (SCF)      : {means['dE_Ha']:+.3e} ± {stds['dE_Ha_std']:.1e} Ha   "
                 f"({means['dE_uEh']:+.2f} ± {stds['dE_uEh_std']:.2f} μEh)\n")

        fh.write("\nTable-style (ΔP vs. ref final P):   (mean)\n")
        fh.write(f"  P: ||ΔP||_F = {means['P_norm_F']:.3e}   "
                 f"||S^1/2 ΔP S^1/2||_F = {means['P_Sw_norm_F']:.3e}\n")
        fh.write(f"     scaled ×1e-6: {means['P_norm_F_x1e6']:.2f} , {means['P_Sw_norm_F_x1e6']:.2f}\n")
        fh.write(f"  ε (SCF-vs-SCF): MAE = {means['eps_MAE_uEh']:.2f} μEh , MaxAbs = {means['eps_MaxAbs_uEh']:.2f} μEh\n")
        fh.write(f"  ψ (SCF-vs-SCF): mean = {means['psi_mean']:.6f} , min = {means['psi_min']:.6f}\n")

        fh.write("\nDirect-from-pred DM (no SCF) vs ref SCF:   (mean ± std)\n")
        fh.write(f"  ΔE_direct     : {means['dE_direct_Ha']:+.3e} ± {stds['dE_direct_Ha_std']:.1e} Ha   "
                 f"({means['dE_direct_uEh']:+.2f} ± {stds['dE_direct_uEh_std']:.2f} μEh)\n")
        fh.write(f"  ε_direct      : MAE = {means['eps_direct_MAE_uEh']:.2f} ± {stds['eps_direct_MAE_uEh_std']:.2f} μEh , "
                 f"MaxAbs = {means['eps_direct_MaxAbs_uEh']:.2f} ± {stds['eps_direct_MaxAbs_uEh_std']:.2f} μEh\n")
        fh.write(f"  ψ_direct      : mean = {means['psi_direct_mean']:.6f} ± {stds['psi_direct_mean_std']:.6f} , "
                 f"min = {means['psi_direct_min']:.6f} ± {stds['psi_direct_min_std']:.6f}\n")

        fh.write("\nInput-guess sanity (raw predictions; dataset means):\n")
        fh.write(f"  mean |Tr(PS)-N|         = {mean_abs_trace_err:.3e}\n")
        fh.write(f"  mean MAE(P_pred, P_ref) = {mean_mae_vs_ref:.3e}\n")
        fh.write(f"  mean occ range          ≈ [{mean_occ_min:+.3e}, {mean_occ_max:+.3e}]\n")
        fh.write(f"  mean ‖PSP−2P‖_F         = {mean_idem_pred:.3e}\n")
        fh.write("=================================================\n")

    extra_means = {
        "mean_abs_trace_err": mean_abs_trace_err,
        "mean_mae_vs_ref": mean_mae_vs_ref,
        "mean_occ_min": mean_occ_min,
        "mean_occ_max": mean_occ_max,
        "mean_idem_pred": mean_idem_pred,
        "baseline_label": BASELINE_LABEL,
        "atom_order": list(chemical_symbols),
    }
    return reports, {"mean": means, "std": stds, "extra_pred_means": extra_means}
    
