import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import List
from contextlib import contextmanager
from pyscf.scf import hf as _hf  # for generalized eigproblem F C = S C eps
from argparse import Namespace
from pyscf import gto, scf, dft
from pyscf.grad import rhf
from ase.io import read
from ase.units import Bohr
import math
from nn import *
from typing import Dict, Any, Tuple  #
import sqlite3 as sql
import matplotlib.pyplot as plt
import os

# ---------------- basic types / helpers ----------------
dtype = torch.float64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
T = lambda x: torch.from_numpy(x).to(dtype=dtype).to(device = device)

# FIX: PyTorch has no torch.frombuffer that returns numpy; use NumPy
read_binaries = lambda x: np.frombuffer(x, dtype=np.float64 if dtype == torch.float64 else np.float32)
key = "density_matrix"


###############PySCF STUFF###################################

print(f"Now working on {device}")
batch_size = 50
mol_sym = "H2O"
charge = 0
num_mols = 4999
unit = "Bohr"


mol_label = "ion" if charge != 0 else "neutral"
basis =  {"neutral":"def2-SVP", "ion":"def2-SVPD"}[mol_label]
auxbasis= {"neutral":"def2-universal-jkfit", "ion":"def2-svp-jkfit"}[mol_label]


spin = 0
dir_path = "2025-09-17_A9ssPcZw"

path_key = dir_path.split("_")[-1]
datasplit_path = os.path.join(dir_path, "datasplits.npz")
best_path = os.path.join(dir_path, "best_" + path_key + ".pth")

db_path = f"{mol_sym.lower()}_pbe-def2svp_{num_mols}.db" if charge == 0 else f"{mol_sym.lower()}_pbe-def2svpd_{num_mols}.db"
preds_path = f"{mol_sym.lower()}_preds.npy"

# -------- Element settings / AO patterns --------
chemical_info_dict_ion = {
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


chemical_info_dict =  {
    "H2O": {
        "symbols": ["O", "H", "H"],
        "irreps_deg": {"O": (0, 0, 0, 1, 1, 2), "H": (0, 0, 1)}
    },
    "NH3": {
        "symbols": ["N", "H", "H", "H"],
        "irreps_deg": {"N": (0, 0, 0, 1, 1, 2), "H": (0, 0, 1)}
    },
    "CH4": {
        "symbols": ["C", "H", "H", "H", "H"],
        "irreps_deg": {"C": (0, 0, 0, 1, 1, 2), "H": (0, 0, 1)}
    },
    "FH": {
        "symbols": ["F", "H"],
        "irreps_deg": {"F": (0, 0, 0, 1, 1, 2), "H": (0, 0, 1)}
    },
    "CH3CH2OH": {
        "symbols": ["C", "C", "O", "H", "H", "H", "H", "H", "H"],  
        "irreps_deg": {
            "C": (0, 0, 0, 1, 1, 2),
            "O": (0, 0, 0, 1, 1, 2),
            "H": (0, 0, 1)
        }
    },

     "URACIL": {
        "symbols": [*("C",)*2, "N", "C", "N", "C", *("O",)*2, *("H",)*4],  # 2C + O + 6H
        "irreps_deg": {
            "C": (0, 0, 0, 1, 1, 2),
            "O": (0, 0, 0, 1, 1, 2),
            "N": (0, 0, 0, 1, 1, 2),
            "H": (0, 0, 1)
        }    
    }
}

chem_dict = chemical_info_dict if charge == 0 else chemical_info_dict_ion
chemical_dict = chem_dict[mol_sym]
chemical_symbols = chemical_dict["symbols"]
atoms_args = "".join(chemical_symbols)
num_atoms = len(chemical_symbols)
irreps_degs = chemical_dict[ "irreps_deg"]


# NOTE: num_orbitals computed from atom_info may not equal AO dim of DB; we won’t use it to reshape blobs
num_orbitals = sum([np.sum(2*np.array(irreps_degs[sym]))+len(irreps_degs[sym]) for sym in chemical_symbols])
def build_pyscf_mol(coords_bohr: np.ndarray,
                        chemical_symbols: List[str],
                        basis: str = basis,
                        unit: str = unit,
                        charge: int = charge,
                        spin: int = spin):
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


convention_dict_ion = {
   'pyscf': Namespace(
       atom_to_orbitals_map={'H': 'ssp', 'O': 'sssspppdd', 'C': 'sssppd', 'N': 'ssssppdd', 'F': 'sssppd'},
       orbital_idx_map={'s': [0], 'p': [1, 2, 0], 'd': [4, 2, 0, 1, 3]},
       orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
       orbital_order_map={'H': [0, 1, 2], 'O': [0, 1, 2, 3, 4, 5, 6, 7, 8], 'C': [0, 1, 2, 3, 4, 5], 'N': [0, 1, 2, 3, 4, 5, 6 ,7], 'F': [0, 1, 2, 3, 4, 5]},
   ),

   'orca': Namespace(
       atom_to_orbitals_map={'H': 'ssp', 'O': 'sssspppdd', 'C': 'sssppd', 'N': 'ssssppdd', 'F': 'sssppd'},
       orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [4, 2, 0, 1, 3]},
       orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
       orbital_order_map={'H': [0, 1, 2], 'O': [0, 1, 2, 3, 4, 5, 6, 7, 8], 'C': [0, 1, 2, 3, 4, 5], 'N': [0, 1, 2, 3, 4, 5, 6 ,7], 'F': [0, 1, 2, 3, 4, 5]},
   ),

   'pysch': Namespace(
      atom_to_orbitals_map={'H': 'ssp', 'O': 'sssspppdd', 'C': 'sssppd', 'N': 'ssssppdd', 'F': 'sssppd'},
       orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [2, 3, 1, 4, 0]},
       orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
       orbital_order_map={'H': [0, 1, 2], 'O': [0, 1, 2, 3, 4, 5, 6, 7, 8], 'C': [0, 1, 2, 3, 4, 5], 'N': [0, 1, 2, 3, 4, 5, 6 ,7], 'F': [0, 1, 2, 3, 4, 5]},
   ),

   'pyphi': Namespace(
       atom_to_orbitals_map={'H': 'ssp', 'O': 'sssspppdd', 'C': 'sssppd', 'N': 'ssssppdd', 'F': 'sssppd'},
       orbital_idx_map={'s': [0], 'p': [1, 2, 0], 'd': [0, 1, 2, 3, 4]},
       orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
       orbital_order_map={'H': [0, 1, 2], 'O': [0, 1, 2, 3, 4, 5, 6, 7, 8], 'C': [0, 1, 2, 3, 4, 5], 'N': [0, 1, 2, 3, 4, 5, 6 ,7], 'F': [0, 1, 2, 3, 4, 5]},
   ),

   'phipy': Namespace(
      atom_to_orbitals_map={'H': 'ssp', 'O': 'sssspppdd', 'C': 'sssppd', 'N': 'ssssppdd', 'F': 'sssppd'},
       orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [0, 1, 2, 3, 4]},
       orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
       orbital_order_map={'H': [0, 1, 2], 'O': [0, 1, 2, 3, 4, 5, 6, 7, 8], 'C': [0, 1, 2, 3, 4, 5], 'N': [0, 1, 2, 3, 4, 5, 6 ,7], 'F': [0, 1, 2, 3, 4, 5]},
   ),
}




def transform(hamiltonians,convention, atoms = atoms_args, charge = charge):
    conv = convention_dict[convention] if charge == 0 else convention_dict_ion[convention] 
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
import numpy as np
from contextlib import contextmanager
import time
from typing import Tuple
from pyscf import gto, dft, scf
from pyscf.scf import hf as _hf


def symm(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)

def cholesky_S(S: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    S = symm(S); n = S.shape[0]
    return np.linalg.cholesky(S + eps * np.eye(n))

def occupations_from_P(P: np.ndarray, S: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    L = cholesky_S(S, eps=eps)
    Q = symm(L.T @ P @ L)
    return np.sort(np.linalg.eigvalsh(Q))

def occupation_mae(P_pred: np.ndarray, P_ref: np.ndarray, S: np.ndarray) -> float:
    occ_pred = occupations_from_P(P_pred, S)
    occ_ref = occupations_from_P(P_ref, S)
    return float(np.mean(np.abs(occ_pred - occ_ref)))

def idempotency_error(P: np.ndarray, S: np.ndarray) -> float:
    return float(np.linalg.norm(P @ S @ P - 2.0 * P, "fro"))

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

# ============================== SCF helper with iteration counter ==============================
class SCFCallback:
    def __init__(self):
        self.counter = 0
    def __call__(self, envs):
        self.counter += 1


# ============================== Evaluate guess function ==============================
def evaluate_dm_guess(mol, dm_pred: np.ndarray, name="system",
                      enforce_trace=False, eps=1e-12, verbose=0,
                      baseline_guess="atom", auxbasis=auxbasis):
    
    baseline_guess_dict = {
    "minao": scf.hf.init_guess_by_minao,
    "atom": scf.hf.init_guess_by_atom,
    "huckel": scf.hf.init_guess_by_huckel,
    "sad": scf.hf.init_guess_by_sad,
    "1e": scf.hf.init_guess_by_1e,
    "zero": lambda mol: None  # special case: starts with DM = 0
}
    assert baseline_guess in baseline_guess_dict, f"INVALID BASELINE SELECTED! VALID OPTIONS: {list(baseline_guess_dict.keys())}"
    nao, N_e = mol.nao, mol.nelectron
    S = mol.intor("int1e_ovlp_sph")
    occs_raw = occupations_from_P(dm_pred, S)
    tr_raw = trace_electrons(dm_pred, S)
    idem_raw = idempotency_error(dm_pred, S)

    alpha, dm_corr = rescale_trace(dm_pred, S, N_e)
    dm0_to_use = dm_corr if enforce_trace else dm_pred
    tag = "trace-corrected" if enforce_trace else "raw"

    # --- SCF baseline ---
    mf_ref = dft.RKS(mol).density_fit(auxbasis=auxbasis)
    mf_ref.xc = "PBE"
    mf_ref.grids.prune = True
    mf_ref.grids.atom_grid = (75, 302)
    mf_ref.max_cycle = 180
    mf_ref.diis_space = 12
    mf_ref.conv_tol = 1e-9
    mf_ref.verbose = verbose
    cb_ref = SCFCallback()
    mf_ref.callback = cb_ref

    with time_block() as t_ref:
        e_ref = mf_ref.kernel(dm0 = baseline_guess_dict[baseline_guess](mol))
    P_ref = mf_ref.make_rdm1()
    eps_ref = np.asarray(mf_ref.mo_energy)
    C_ref = np.asarray(mf_ref.mo_coeff)

    # --- SCF with predicted DM ---
    mf_pred = dft.RKS(mol).density_fit(auxbasis=auxbasis)
    mf_pred.xc = "PBE"
    mf_pred.grids.prune = True
    mf_pred.grids.atom_grid = (75, 302)
    mf_pred.max_cycle = 180
    mf_pred.diis_space = 12
    mf_pred.conv_tol = 1e-9
    mf_pred.verbose = verbose
    cb_pred = SCFCallback()
    mf_pred.callback = cb_pred

    with time_block() as t_pred:
        e_pred = mf_pred.kernel(dm0=dm0_to_use)
    P_pred_final = mf_pred.make_rdm1()
    eps_pred = np.asarray(mf_pred.mo_energy)
    C_pred = np.asarray(mf_pred.mo_coeff)

    dP = np.abs(dm0_to_use - P_ref)
    dP_MAE = np.mean(dP)

    eps_mae = np.mean(np.abs(eps_pred - eps_ref)) * 1e6
    eps_max = np.max(np.abs(eps_pred - eps_ref)) * 1e6

    # Subspace overlap (final SCF DM vs ref)
    O = C_ref[:, :N_e//2].T @ (S @ C_pred[:, :N_e//2])
    svals = np.linalg.svd(O, compute_uv=False)
    psi_mean, psi_min = float(np.mean(svals)), float(np.min(svals))

    # Subspace overlap (raw pred DM vs ref) ← new
    mf_raw = dft.RKS(mol).density_fit(auxbasis=auxbasis)
    mf_raw.xc = "PBE"
    mf_raw.grids.prune = True
    mf_raw.grids.atom_grid = (75, 302)
    mf_raw.max_cycle = 0
    mf_raw.kernel(dm0=dm_pred)
    C_raw = mf_raw.mo_coeff
    eps_raw = mf_raw.mo_energy
    O_raw = C_ref[:, :N_e//2].T @ (S @ C_raw[:, :N_e//2])
    cosine_similarity_raw = float(np.mean(np.linalg.svd(O_raw, compute_uv=False)))

    epsilon_mae_raw = float(np.mean(np.abs(eps_raw - eps_ref)) * 1e6)

    # Direct energy
    hcore = mf_pred.get_hcore()
    veff = mf_pred.get_veff(dm=dm_pred)
    e_direct = float(mf_pred.energy_elec(dm=dm_pred, h1e=hcore, vhf=veff)[0] + mf_pred.energy_nuc())

    eps_direct = _hf.eig(mf_pred.get_fock(h1e=hcore, s1e=S, vhf=veff), S)[0]
    eps_mae_direct = np.mean(np.abs(eps_direct - eps_ref)) * 1e6
    eps_max_direct = np.max(np.abs(eps_direct - eps_ref)) * 1e6

    return {
        "name": name,
        "nao": nao,
        "N_electrons": N_e,
        "SCF_baseline": {
            "E_final_Ha": float(e_ref),
            "iters_total": cb_ref.counter,
            "time_total_s": t_ref(),
            "init_guess": baseline_guess,
            "P": P_ref
        },
        "SCF_from_pred": {
            "dm_used": tag,
            "E_final_Ha": float(e_pred),
            "iters_total": cb_pred.counter,
            "time_total_s": t_pred(),
            "iteration_savings": cb_ref.counter - cb_pred.counter,
            "time_savings_s": t_ref() - t_pred(),
            "P": P_pred_final
        },
        "P_metrics": {
            "||ΔP||_MAE": float(dP_MAE),
            "S": S
        },
        "epsilon_metrics": {
            "MAE_μEh": float(eps_mae),
            "MaxAbs_μEh": float(eps_max),
            "MAE_raw_μEh": epsilon_mae_raw   # ← new
        },
        "psi_metrics": {
            "mean": float(psi_mean),
            "min": float(psi_min),
            "cosine_similarity_raw": cosine_similarity_raw  # ← new
        },
        "diagnostics_guess": {
            "Tr(PS)-N": tr_raw - N_e,
            "occ_min": float(occs_raw.min()),
            "occ_max": float(occs_raw.max()),
            "idempotency_Frob": idem_raw
        },
        "delta_E": {
            "Ha": float(e_pred - e_ref),
            "μEh": 1e6 * (e_pred - e_ref)
        },
        "direct_from_pred_dm": {
            "E_tot_Ha": float(e_direct),
            "delta_E_direct": {
                "Ha": float(e_direct - e_ref),
                "μEh": 1e6 * (e_direct - e_ref)
            },
            "epsilon_vs_ref": {
                "MAE_μEh": float(eps_mae_direct),
                "MaxAbs_μEh": float(eps_max_direct)
            },
            "psi_vs_ref": {
                "mean": float(psi_mean),
                "min": float(psi_min)
            }
        }
    }

def print_benchmarks(rep: dict, file=None):
    name = rep.get("name", "<unnamed>")
    e_pred = rep["SCF_from_pred"]["E_final_Ha"]
    e_ref = rep["SCF_baseline"]["E_final_Ha"]
    dE = rep["delta_E"]["Ha"]
    dP = rep["P_metrics"]["||ΔP||_MAE"]
    err_trace = rep["diagnostics_guess"]["Tr(PS)-N"]
    idem = rep["diagnostics_guess"]["idempotency_Frob"]
    occ_mae = occupation_mae(rep["SCF_from_pred"]["P"],
                             rep["SCF_baseline"]["P"],
                             rep["P_metrics"]["S"])
    
    iter_ref = rep["SCF_baseline"]["iters_total"]
    iter_pred = rep["SCF_from_pred"]["iters_total"]
    iter_save = rep["SCF_from_pred"]["iteration_savings"]
    
    time_ref = rep["SCF_baseline"]["time_total_s"]
    time_pred = rep["SCF_from_pred"]["time_total_s"]
    time_save = rep["SCF_from_pred"]["time_savings_s"]

    eps_mae_raw = rep["epsilon_metrics"]["MAE_raw_μEh"] * 1e-6
    cos_sim_raw = rep["psi_metrics"]["cosine_similarity_raw"]

    lines = [
        f"== {name} ==",
        f"  E_ref (Ha): {e_ref:.12f}",
        f"  E_pred (Ha): {e_pred:.12f}",
        f"  ΔE (Ha): {dE:+.6e}",
        f"  ||ΔP||_MAE: {dP:.6e}",
        f"  Trace error: {err_trace:+.2e}",
        f"  Idempotency error: {idem:.2e}",
        f"  Occ MAE: {occ_mae:.2e}",
        f"  ε_MAE_raw (Ha): {eps_mae_raw:.2e}",
        f"  Cosine similarity (raw): {cos_sim_raw:.5f}",
        f"  Iterations (baseline): {iter_ref}",
        f"  Iterations (pred): {iter_pred}",
        f"  Iteration savings: {iter_save:+d}",
        f"  Time (baseline) [s]: {time_ref:.2f}",
        f"  Time (pred) [s]: {time_pred:.2f}",
        f"  Time savings (s): {time_save:+.2f}",
        ""
    ]
    for l in lines:
        print(l, file=file)
# ============================== Dataset runner ==============================
def evaluate_dataset_to_txt(dms_valid, rs_valid, chemical_symbols,
                            outfile=f"benchmarks_summary_{mol_sym.lower()}.txt",
                            basis=basis, unit=unit, charge=charge, spin=spin,
                            baseline_guess="sad", print_each=True):
    if basis is None:
        raise ValueError("Please provide basis.")
    n = len(dms_valid)
    reports = []

    with open(outfile, "w") as fh:
        for i in range(n):
            mol = gto.Mole()
            mol.unit = unit
            mol.atom = [(chemical_symbols[j], tuple(rs_valid[i][j])) for j in range(len(chemical_symbols))]
            mol.charge, mol.spin, mol.basis = charge, spin, basis
            mol.build()
            rep = evaluate_dm_guess(mol, np.asarray(dms_valid[i]), name=f"sample {i}", baseline_guess=baseline_guess)
            reports.append(rep)
            if print_each:
                print_benchmarks(rep, file=fh)

        def agg(getter):
            vals = [getter(r) for r in reports]
            return np.mean(vals), np.std(vals, ddof=1)

        it_b, std_it_b = agg(lambda r: r["SCF_baseline"]["iters_total"])
        it_p, std_it_p = agg(lambda r: r["SCF_from_pred"]["iters_total"])
        tm_b, std_tm_b = agg(lambda r: r["SCF_baseline"]["time_total_s"])
        tm_p, std_tm_p = agg(lambda r: r["SCF_from_pred"]["time_total_s"])
        perc_iter = 100 * (it_b - it_p) / it_b
        perc_time = 100 * (tm_b - tm_p) / tm_b

        fh.write(f"\n================ Dataset Summary ({mol_sym})================\n")
        fh.write(f"samples: {n}   basis: {basis}   baseline: {baseline_guess}\n\n")
        fh.write(f"iters baseline: {it_b:.2f} ± {std_it_b:.2f}, iters pred: {it_p:.2f} ± {std_it_p:.2f}, "
                 f"Δiters: {it_b - it_p:+.2f} ({perc_iter:+.1f}%)\n")
        fh.write(f"time baseline: {tm_b:.3f} ± {std_tm_b:.3f} s, time pred: {tm_p:.3f} ± {std_tm_p:.3f} s, "
                 f"Δtime: {tm_b - tm_p:+.3f} s ({perc_time:+.1f}%)\n")

        mean, std = agg(lambda r: r["delta_E"]["Ha"])
        fh.write(f"ΔE (SCF) [Ha]: {mean:+.6e} ± {std:.6e}\n")

        mean, std = agg(lambda r: r["direct_from_pred_dm"]["delta_E_direct"]["Ha"])
        fh.write(f"ΔE_direct [Ha]: {mean:+.6e} ± {std:.6e}\n")

        mean, std = agg(lambda r: r["P_metrics"]["||ΔP||_MAE"])
        fh.write(f"||ΔP||_MAE: {mean:.6e} ± {std:.6e}\n")

        mean, std = agg(lambda r: r["epsilon_metrics"]["MAE_μEh"])
        fh.write(f"ε_MAE (SCF-vs-SCF) [Ha]: {mean*1e-6:.6e} ± {std*1e-6:.6e}\n")

        mean, std = agg(lambda r: r["epsilon_metrics"]["MaxAbs_μEh"])
        fh.write(f"ε_MaxAbs (SCF-vs-SCF) [Ha]: {mean*1e-6:.6e} ± {std*1e-6:.6e}\n")

        mean, std = agg(lambda r: r["direct_from_pred_dm"]["epsilon_vs_ref"]["MAE_μEh"])
        fh.write(f"ε_MAE_direct [Ha]: {mean*1e-6:.6e} ± {std*1e-6:.6e}\n")

        mean, std = agg(lambda r: r["direct_from_pred_dm"]["epsilon_vs_ref"]["MaxAbs_μEh"])
        fh.write(f"ε_MaxAbs_direct [Ha]: {mean*1e-6:.6e} ± {std*1e-6:.6e}\n")

        mean, std = agg(lambda r: r["diagnostics_guess"]["Tr(PS)-N"])
        fh.write(f"Trace error (Tr(PS)-N): {mean:+.6e} ± {std:.6e}\n")

        mean, std = agg(lambda r: r["diagnostics_guess"]["idempotency_Frob"])
        fh.write(f"Idempotency error: {mean:.6e} ± {std:.6e}\n")

        mean, std = agg(lambda r: r["diagnostics_guess"]["occ_min"])
        fh.write(f"Occ min: {mean:.6e} ± {std:.6e}\n")

        mean, std = agg(lambda r: r["diagnostics_guess"]["occ_max"])
        fh.write(f"Occ max: {mean:.6e} ± {std:.6e}\n")

        mean, std = agg(lambda r: occupation_mae(
            r["SCF_from_pred"].get("P", None),
            r["SCF_baseline"].get("P", None),
            r["P_metrics"].get("S", None)))
        fh.write(f"Occ MAE (pred vs real): {mean:.6e} ± {std:.6e}\n")

        mean, std = agg(lambda r: r["direct_from_pred_dm"]["E_tot_Ha"])
        fh.write(f"E_tot_direct (Ha): {mean:.12f} ± {std:.2e}\n")

        mean, std = agg(lambda r: r["psi_metrics"]["cosine_similarity_raw"])
        fh.write(f"Cosine similarity (raw): {mean:.5f} ± {std:.5f}\n")
        fh.write("=================================================\n")

    return reports


conn = sql.connect(db_path)
cursor = conn.cursor()
dmsb = cursor.execute("SELECT P FROM data").fetchall()
ssb  = cursor.execute("SELECT S FROM data").fetchall()
rsb  = cursor.execute("SELECT R FROM data").fetchall()
conn.close()

# FIX: coordinates are natoms x 3
rs = np.array([read_binaries(row[0]).reshape(num_atoms, 3) for row in rsb])
s0 = read_binaries(ssb[0][0])
ss = np.array([read_binaries(row[0]).reshape(num_orbitals, num_orbitals) for row in ssb])

# ---------------- splits & model paths ----------------

# === LOADED OBJECTS ===
datasplit = np.load(datasplit_path, allow_pickle=True)
valid_indices = [int(i) for i in datasplit["valid"]]

ss_list = ss[valid_indices]
rs_list = rs[valid_indices]

try:
    preds = np.load(preds_path, allow_pickle=True)
    print(f"PREDICTIONS SUCCESSFULLY LOADED from {preds_path}!!")
except FileNotFoundError:
    print(f"PREDICTIONS NOT FOUND at {preds_path}. NOW CALCULATING")

    model = NeuralNetwork(load_from=best_path)
    model.eval()
    model.downstream_mode = True
    model.to(dtype=dtype, device=device)

    offset = 0
    preds = []

    # safer batch size for GPU
    safe_batch = min(batch_size, 64)

    with torch.no_grad():
        while offset < len(rs_list):
            ss_batch = ss_list[offset:offset+safe_batch]
            rs_batch = rs_list[offset:offset+safe_batch]

            ss_batch = T(ss_batch)
            rs_batch = T(rs_batch)

            out = model(rs_batch, ss_batch)[key].detach().cpu().numpy()
            out = transform(out, convention="phipy", atoms=atoms_args)
            preds.extend(out)

            offset += safe_batch

    preds = np.array(preds)
    np.save(preds_path, preds)
    print(f"PREDICTIONS FOR {mol_sym} SUCCESSFULLY SAVED to {preds_path}!!!")
        


evaluate_dataset_to_txt(preds, rs_list, chemical_symbols)

        
    
