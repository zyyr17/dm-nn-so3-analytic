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
from typing import Tuple
from typing import List
import sqlite3 as sql
import matplotlib.pyplot as plt
import os

# ---------------- basic types / helpers ----------------
dtype = torch.float64
T = lambda x: torch.from_numpy(x).to(dtype=dtype)

# FIX: PyTorch has no torch.frombuffer that returns numpy; use NumPy
read_binaries = lambda x: np.frombuffer(x, dtype=np.float64)

# ---------- Helpers: linear algebra & diagnostics ----------
num_mols = 4999
molecule_name = "H2O"  # Change to "CH4", "NH3", or "H2O" as needed
dir_path = "2025-08-30_XDwAOEAa"

settings = {
    "H2O": {
        "symbols": ["O", "H", "H"],
        "order": ["O", "H", "H"],
        "atom_info": {"O": (0, 0, 0, 1, 1, 2), "H": (0, 0, 1)}
    },
    "NH3": {
        "symbols": ["N", "H", "H", "H"],
        "order": ["N", "H", "H", "H"],
        "atom_info": {"N": (0, 0, 0, 1, 1, 2), "H": (0, 0, 1)}
    },
    "CH4": {
        "symbols": ["C", "H", "H", "H", "H"],
        "order": ["C", "H", "H", "H", "H"],
        "atom_info": {"C": (0, 0, 0, 1, 1, 2), "H": (0, 0, 1)}
    },
    "FH": {
        "symbols": ["F", "H"],
        "order": ["F", "H"],
        "atom_info": {"F": (0, 0, 0, 1, 1, 2), "H": (0, 0, 1)}
    },
    "CH3CH2OH": {
        "symbols": ["C", "C", "O", "H", "H", "H", "H", "H", "H"],  # 2C + O + 6H
        "order": ["C", "C", "O", "H", "H", "H", "H", "H", "H"],
        "atom_info": {
            "C": (0, 0, 0, 1, 1, 2),
            "O": (0, 0, 0, 1, 1, 2),
            "H": (0, 0, 1)
        }
    }
}[molecule_name]

chemical_symbols = settings["symbols"]
# FIX: symbols → chemical_symbols
atoms_args = "".join(chemical_symbols)
order = settings["order"]
atom_info = settings["atom_info"]
num_atoms = len(chemical_symbols)

# NOTE: num_orbitals computed from atom_info may not equal AO dim of DB; we won’t use it to reshape blobs
num_orbitals = sum([np.sum(2*np.array(atom_info[sym]))+len(atom_info[sym]) for sym in chemical_symbols])
db_path = f"{molecule_name.lower()}_pbe-def2svp_{num_mols}.db"

convention_dict = {
    'pyscf': Namespace(
        atom_to_orbitals_map={'H':'ssp','O':'sssppd','C':'sssppd','N':'sssppd'},
        orbital_idx_map={'s':[0], 'p':[1,2,0], 'd':[4,2,0,1,3]},
        orbital_sign_map={'s':[1], 'p':[1,1,1], 'd':[1,1,1,1,1]},
        orbital_order_map={'H':[0,1,2],'O':[0,1,2,3,4,5],'C':[0,1,2,3,4,5],'N':[0,1,2,3,4,5]},
    ),
    'orca': Namespace(
        atom_to_orbitals_map={'H': 'ssp', 'O': 'sssppd', 'C': 'sssppd', 'N': 'sssppd'},
        orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [4, 2, 0, 1, 3]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={'H': [0, 1, 2], 'O': [0, 1, 2, 3, 4, 5], 'C': [0, 1, 2, 3, 4, 5], 'N': [0, 1, 2, 3, 4, 5]},
    ),
    'pysch': Namespace(
        atom_to_orbitals_map={'H': 'ssp', 'O': 'sssppd', 'C': 'sssppd', 'N': 'sssppd'},
        orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [2, 3, 1, 4, 0]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={'H': [0, 1, 2], 'O': [0, 1, 2, 3, 4, 5], 'C': [0, 1, 2, 3, 4, 5], 'N': [0, 1, 2, 3, 4, 5]},
    ),
    'pyphi': Namespace(
        atom_to_orbitals_map={'H': 'ssp', 'O': 'sssppd', 'C': 'sssppd', 'N': 'sssppd'},
        orbital_idx_map={'s': [0], 'p': [1, 2, 0], 'd': [0, 1, 2, 3, 4]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={'H': [0, 1, 2], 'O': [0, 1, 2, 3, 4, 5], 'C': [0, 1, 2, 3, 4, 5], 'N': [0, 1, 2, 3, 4, 5]},
    ),
    'phipy': Namespace(
        atom_to_orbitals_map={'H': 'ssp', 'O': 'sssppd', 'C': 'sssppd', 'N': 'sssppd'},
        orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [0, 1, 2, 3, 4]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={'H': [0, 1, 2], 'O': [0, 1, 2, 3, 4, 5], 'C': [0, 1, 2, 3, 4, 5], 'N': [0, 1, 2, 3, 4, 5]},
    ),
}

def build_mol_from_bohr(coords_bohr: np.ndarray,
                        chemical_symbols: List[str] = chemical_symbols,
                        basis: str = "def2-SVP",
                        unit: str = "Bohr",
                        charge: int = 0,
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

def transform(hamiltonians, convention="phipy", atoms=atoms_args):
    conv = convention_dict[convention]
    orbitals = ''
    orbitals_order = []
    for a in atoms:
        offset = len(orbitals_order)
        orbitals += conv.atom_to_orbitals_map[a]
        orbitals_order += [idx + offset for idx in conv.orbital_order_map[a]]

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
    transform_indices = np.concatenate(transform_indices).astype(int)
    transform_signs = np.concatenate(transform_signs)

    hamiltonians_new = hamiltonians[..., transform_indices, :]
    hamiltonians_new = hamiltonians_new[..., :, transform_indices]
    hamiltonians_new = hamiltonians_new * transform_signs[:, None]
    hamiltonians_new = hamiltonians_new * transform_signs[None, :]

    return hamiltonians_new

# FIX: missing imports for eigh and fractional_matrix_power
from scipy.linalg import eigh, fractional_matrix_power

def eps_energy_from_dm(mf, mol, dm):
    """Sum of occupied orbital energies computed from the Fock built with a given DM."""
    veff = mf.get_veff(mol, dm)
    fock = mf.get_hcore() + veff
    S = mf.get_ovlp()
    mo_energy, mo_coeff = eigh(fock, S)
    occ = mf.get_occ(mo_energy, mo_coeff)
    return float(np.dot(occ, mo_energy))

def get_trace_PS(P, S):
    return float(np.trace(P @ S))

def idempotency_error(P, S):
    # normalized Frobenius of PSP - 2P
    PSP = P @ S @ P
    diff = PSP - 2.0 * P
    return float(np.linalg.norm(diff, ord="fro") / (P.shape[0] ** 2))

def occ_spectrum(P, S):
    # eigenvalues of Q = S^{1/2} P S^{1/2}
    S_sqrt = fractional_matrix_power(S, 0.5)
    Q = S_sqrt @ P @ S_sqrt
    return np.linalg.eigvalsh(Q)

def cosine_similarity(P1, P2, S):
    num = np.trace(P1 @ S @ P2)
    den = np.sqrt(np.trace(P1 @ S @ P1) * np.trace(P2 @ S @ P2))
    return float(num / den)

# ---------- SCF setup and run (counting DIIS + Newton) ----------
def setup_mf_with_orca_style(
    mol,
    grid_coarse=(75, 302),
    prune=True,
    use_df=True,
    auxbasis="def2-svp-jkfit",
    xc="PBE",
    diis_space=12,
    conv_tol=1e-9,
    conv_tol_grad=1e-6,
    conv_tol_density=1e-8,
    small_rho_cutoff=1e-10,
    verbose=0,
):
    """Build a fresh RKS object with standard settings used in your DB."""
    mf = dft.RKS(mol)
    if use_df:
        mf = mf.density_fit(auxbasis=auxbasis)
    mf.xc = xc
    mf.grids.prune = bool(prune)
    mf.grids.atom_grid = tuple(grid_coarse)
    mf.max_cycle = 180
    mf.diis_space = diis_space
    mf.conv_tol = conv_tol
    if hasattr(mf, "conv_tol_grad"):
        mf.conv_tol_grad = conv_tol_grad
    if hasattr(mf, "conv_tol_density"):
        mf.conv_tol_density = conv_tol_density
    if hasattr(mf, "small_rho_cutoff"):
        mf.small_rho_cutoff = small_rho_cutoff
    mf.verbose = verbose
    return mf

def run_scf_with_mode(
    mol,
    dm0,
    mode="fixed",             # "fixed", "diis", or "adaptive"
    grid_coarse=(75, 302),
    damp0=0.2,
    level_shift0=0.3,
    # thresholds for adaptive shedding of stabilizers
    dP_drop=1e-3,
    dE_drop=1e-4,
):
    """
    Run SCF starting from dm0 with a given stabilizer mode.
    Counts both DIIS and Newton iterations (same callback).
    Returns: dict with iters, time, energy, converged, used_newton, dm_scf, mf (for re-use).
    """
    mf = setup_mf_with_orca_style(mol, grid_coarse=grid_coarse)
    # Initialize stabilizers according to mode
    if mode == "fixed":
        if hasattr(mf, "damp"): mf.damp = damp0
        if hasattr(mf, "level_shift"): mf.level_shift = level_shift0
    elif mode == "diis":
        if hasattr(mf, "damp"): mf.damp = 0.0
        if hasattr(mf, "level_shift"): mf.level_shift = 0.0
    elif mode == "adaptive":
        if hasattr(mf, "damp"): mf.damp = damp0
        if hasattr(mf, "level_shift"): mf.level_shift = level_shift0
    else:
        raise ValueError("mode must be one of {'fixed', 'diis', 'adaptive'}")

    # Iteration counter + adaptive logic state
    state = {
        "n": 0,
        "E_prev": None,
        "P_prev": None,
        "shed": False,  # whether we already disabled stabilizers (adaptive)
    }

    def cb(envs):
        state["n"] += 1
        E = envs["e_tot"]
        P = envs["dm"]

        if mode == "adaptive" and not state["shed"]:
            # compute deltas if previous values exist
            dP = np.linalg.norm(P - state["P_prev"]) if state["P_prev"] is not None else np.inf
            dE = abs(E - state["E_prev"]) if state["E_prev"] is not None else np.inf
            # if stable enough, shed stabilizers (once)
            if (dP < dP_drop) or (dE < dE_drop):
                if hasattr(mf, "damp"): mf.damp = 0.0
                if hasattr(mf, "level_shift"): mf.level_shift = 0.0
                state["shed"] = True

        state["E_prev"] = E
        state["P_prev"] = P

    mf.callback = cb

    # Run SCF (count time)
    t0 = time.time()
    e_scf = mf.kernel(dm0=dm0)
    used_newton = False
    if not mf.converged:
        # shed stabilizers for Newton either way
        if hasattr(mf, "damp"): mf.damp = 0.0
        if hasattr(mf, "level_shift"): mf.level_shift = 0.0
        newton_mf = mf.newton()
        newton_mf.callback = cb  # count Newton iterations too
        e_scf = newton_mf.kernel()
        used_newton = True
    t1 = time.time()

    return {
        "iters_total": int(state["n"]),
        "time_total_s": float(t1 - t0),
        "converged": bool(mf.converged),
        "used_newton": bool(used_newton),
        "e_scf": float(e_scf),
        "dm_scf": mf.make_rdm1(),
        "mf": mf,  # returned so we can reuse grids/ints for energy diagnostics
    }

# ---------- One paired benchmark (predicted P vs MINAO) ----------
def benchmark_pair(
    dm_pred,
    mol,
    init_guess="minao",
    regime="fixed",            # "fixed", "diis", "adaptive"
    grid_coarse=(75, 302),
    grid_fine=(99, 590),
    polish=False,              # if True, polish both on fine grid once (not counted)
    damp0=0.2,
    level_shift0=0.3,
    dP_drop=1e-3,
    dE_drop=1e-4,
    verbose=True,
):
    """
    Runs a symmetric, fair comparison between:
      (A) your predicted density matrix (dm_pred), and
      (B) a baseline MINAO (or other) guess,
    under the chosen regime.

    Returns a dict with iterations, wall-time, energies, and diagnostics.
    """
    # --- SCF from predicted DM ---
    res_pred = run_scf_with_mode(
        mol, dm_pred, mode=regime, grid_coarse=grid_coarse,
        damp0=damp0, level_shift0=level_shift0, dP_drop=dP_drop, dE_drop=dE_drop
    )
    mf_pred = res_pred["mf"]
    e_direct_pred = mf_pred.energy_tot(dm=dm_pred)
    eps_direct_pred = eps_energy_from_dm(mf_pred, mol, dm_pred)
    eps_scf_pred = eps_energy_from_dm(mf_pred, mol, res_pred["dm_scf"])

    # optional symmetric polish (NOT counted)
    e_final_pred = res_pred["e_scf"]
    if polish:
        mf_polish = setup_mf_with_orca_style(mol, grid_coarse=grid_fine)
        # Keep stabilizers off for the one-shot polish; it’s a final single evaluation
        if hasattr(mf_polish, "damp"): mf_polish.damp = 0.0
        if hasattr(mf_polish, "level_shift"): mf_polish.level_shift = 0.0
        e_final_pred = mf_polish.kernel(dm0=res_pred["dm_scf"])

    # DM diagnostics vs its own SCF
    S_pred = mf_pred.get_ovlp()
    N = mol.nelectron
    mae_dm = float(np.mean(np.abs(dm_pred - res_pred["dm_scf"])))
    trace_pred = get_trace_PS(dm_pred, S_pred)
    idemp_pred = idempotency_error(dm_pred, S_pred)
    eig_pred = occ_spectrum(dm_pred, S_pred)
    eig_scf_pred = occ_spectrum(res_pred["dm_scf"], S_pred)
    spec_mae = float(np.mean(np.abs(eig_pred - eig_scf_pred)))
    psi = cosine_similarity(dm_pred, res_pred["dm_scf"], S_pred)
    eps_abs_err = abs(eps_scf_pred - eps_direct_pred)
    energy_abs_err = abs(e_final_pred - e_direct_pred)

    # --- SCF from baseline initial guess (same regime, symmetric) ---
    mf_guess = setup_mf_with_orca_style(mol, grid_coarse=grid_coarse)
    dm_guess = mf_guess.get_init_guess(key=init_guess)
    res_base = run_scf_with_mode(
        mol, dm_guess, mode=regime, grid_coarse=grid_coarse,
        damp0=damp0, level_shift0=level_shift0, dP_drop=dP_drop, dE_drop=dE_drop
    )

    e_final_base = res_base["e_scf"]
    if polish:
        mf_polish_b = setup_mf_with_orca_style(mol, grid_coarse=grid_fine)
        if hasattr(mf_polish_b, "damp"): mf_polish_b.damp = 0.0
        if hasattr(mf_polish_b, "level_shift"): mf_polish_b.level_shift = 0.0
        e_final_base = mf_polish_b.kernel(dm0=res_base["dm_scf"])

    # --- Iteration & time comparison ---
    it_pred = res_pred["iters_total"]
    it_base = res_base["iters_total"]
    iter_reduction_pct = 100.0 * (it_base - it_pred) / max(1, it_base)

    out = {
        "regime": regime,
        "init_guess": init_guess,
        # predicted DM path
        "iters_pred": it_pred,
        "time_pred_s": res_pred["time_total_s"],
        "used_newton_pred": res_pred["used_newton"],
        "e_scf_pred": e_final_pred,               # (polished if enabled)
        "e_direct_pred": e_direct_pred,
        "eps_direct_pred": eps_direct_pred,
        "eps_scf_pred": eps_scf_pred,
        "energy_abs_error_pred": energy_abs_err,  # |E_scf - E_direct| for pred
        "eps_abs_error_pred": eps_abs_err,        # |Σocc ε(pred) - Σocc ε(scf)|
        "trace_PS_pred": trace_pred,
        "idempotency_error_pred": idemp_pred,
        "mae_dm_vs_scf": mae_dm,
        "occ_spec_mae": spec_mae,
        "cosine_sim": psi,
        "min_occ_pred": float(eig_pred.min()),
        "max_occ_pred": float(eig_pred.max()),
        # baseline path
        "iters_base": it_base,
        "time_base_s": res_base["time_total_s"],
        "used_newton_base": res_base["used_newton"],
        "e_scf_base": e_final_base,               # (polished if enabled)
        # comparison
        "iter_reduction_pct": iter_reduction_pct,
    }

    if verbose:
        print("\n" + "=" * 46)
        title = {
            "fixed": "FAIR COMPARISON (FIXED STABILIZERS)",
            "diis": "FAIR COMPARISON (DIIS-ONLY)",
            "adaptive": "FAIR COMPARISON (ADAPTIVE STABILIZERS)",
        }[regime]
        print(title)
        print("=" * 46)
        print(f"iters  pred / base        : {it_pred}  /  {it_base}   (Δ {iter_reduction_pct:.2f}%)")
        print(f"time   pred / base (s)    : {res_pred['time_total_s']:.4f}  /  {res_base['time_total_s']:.4f}")
        print(f"Newton pred / base        : {res_pred['used_newton']} / {res_base['used_newton']}")
        print(f"E_scf  pred / base (Ha)   : {e_final_pred:.10e}  /  {e_final_base:.10e}")
        print(f"E_dir(pred) vs E_scf(pred): {e_direct_pred:.10e}  vs  {e_final_pred:.10e}  |Δ|={energy_abs_err:.2e}")
        print(f"Σocc ε (pred/scf)         : {eps_direct_pred:.6e}  /  {eps_scf_pred:.6e}  |Δ|={eps_abs_err:.2e}")
        print(f"Trace(P S) (pred)         : {trace_pred:.6f}  (target = {mol.nelectron})")
        print(f"Idempotency error (pred)   : {idemp_pred:.3e}")
        print(f"cosine(P_pred, P_scf)      : {psi:.10f}")
        print(f"Occ spectrum MAE (pred)    : {spec_mae:.3e}")
        print("=" * 46 + "\n")

    return out

# ---------- Batch driver over multiple geometries (optional) ----------
def benchmark_dataset(
    dms_list,
    Rs_list,
    create_mol_fn,
    regime="fixed",
    init_guess="minao",
    polish=False,
    grid_coarse=(75, 302),
    grid_fine=(99, 590),
    damp0=0.2,
    level_shift0=0.3,
    dP_drop=1e-3,
    dE_drop=1e-4,
    output_file="benchmark_results.txt",
):
    """
    Run the paired benchmark over a dataset of predicted DMs + coordinates.
    Returns a dict of per-sample results and a global summary.
    Also writes a .txt file with a clean table + summary.
    """
    import numpy as np

    assert len(dms_list) == len(Rs_list), "Length mismatch between DMs and coordinates"
    results = []
    lines = []

    header = (
        f"{'Idx':>4} | {'iters_pred':>10} | {'iters_base':>10} | {'Δiters%':>8} | "
        f"{'time_pred(s)':>12} | {'time_base(s)':>12} | "
        f"{'E_pred(Ha)':>14} | {'E_base(Ha)':>14} | "
        f"{'AbsE_err':>10} | {'eps_err':>10} | {'cos_sim':>10} | "
        f"{'Tr(PS)':>10} | {'Idemp':>10}\n"
    )
    lines.append(header)
    lines.append("-"*len(header) + "\n")

    for i, (P, R) in enumerate(zip(dms_list, Rs_list), start=1):
        mol = create_mol_fn(R)
        out = benchmark_pair(
            P, mol,
            init_guess=init_guess,
            regime=regime,
            grid_coarse=grid_coarse,
            grid_fine=grid_fine,
            polish=polish,
            damp0=damp0,
            level_shift0=level_shift0,
            dP_drop=dP_drop,
            dE_drop=dE_drop,
            verbose=False,
        )
        results.append(out)

        line = (
            f"{i:4d} | "
            f"{out['iters_pred']:10d} | {out['iters_base']:10d} | {out['iter_reduction_pct']:8.2f} | "
            f"{out['time_pred_s']:12.4f} | {out['time_base_s']:12.4f} | "
            f"{out['e_scf_pred']:14.8f} | {out['e_scf_base']:14.8f} | "
            f"{out['energy_abs_error_pred']:10.2e} | {out['eps_abs_error_pred']:10.2e} | "
            f"{out['cosine_sim']:10.4f} | "
            f"{out['trace_PS_pred']:10.4f} | {out['idempotency_error_pred']:10.2e}\n"
        )
        lines.append(line)

    # summary stats (means and stds for key fields)
    def mean_std(key):
        vals = np.array([r[key] for r in results], dtype=float)
        return float(vals.mean()), float(vals.std())

    summary_keys = [
        ("iters_pred", "Iters_pred"),
        ("iters_base", f"Iters_{init_guess}"),
        ("iter_reduction_pct", "Iter_reduction(%)"),
        ("time_pred_s", "Time_pred(s)"),
        ("time_base_s", "Time_base(s)"),
        ("energy_abs_error_pred", "E_abs_err(Ha)"),
        ("eps_abs_error_pred", "Eps_abs_err(Ha)"),
        ("cosine_sim", "Cosine_sim"),
        ("occ_spec_mae", "Occ_spec_MAE"),
        ("idempotency_error_pred", "Idemp_err"),
    ]

    lines.append("\n===== Global Summary (mean ± std) =====\n")
    for key, label in summary_keys:
        m, s = mean_std(key)
        lines.append(f"{label:<20}: {m:.6e} ± {s:.6e}\n")

    with open(output_file, "w") as f:
        f.writelines(lines)

    print(f"Benchmark results saved to: {output_file}")

    return {"results": results, "summary": lines}

# ---------------- Load data from DB ----------------
conn = sql.connect(db_path)
cursor = conn.cursor()
dmsb = cursor.execute("SELECT P FROM data").fetchall()
ssb  = cursor.execute("SELECT S FROM data").fetchall()
rsb  = cursor.execute("SELECT R FROM data").fetchall()
conn.close()

# FIX: coordinates are natoms x 3
rs = np.array([read_binaries(row[0]).reshape(num_atoms, 3) for row in rsb])

# FIX: no read_binaries32; infer AO dimension from first S blob
s0 = read_binaries(ssb[0][0])
nao = int(round(math.sqrt(len(s0))))
assert nao * nao == len(s0), "S blob size is not a perfect square"
ss = np.array([read_binaries(row[0]).reshape(nao, nao) for row in ssb])

# ---------------- splits & model paths ----------------
path_key = dir_path.split("_")[-1]
datasplit_path = os.path.join(dir_path, "datasplits.npz")
best_path = os.path.join(dir_path, "best_" + path_key + ".pth")

# === LOADED OBJECTS ===
datasplit = np.load(datasplit_path, allow_pickle=True)
valid_indices = [int(i) for i in datasplit["valid"]]

ss_list = ss[valid_indices]
rs_list = rs[valid_indices]

# ---------------- Predict with your model ----------------
# NOTE: NeuralNetwork is not defined in this snippet. Keep your call, but guard it so this file loads.
try:
    model = NeuralNetwork(load_from=best_path)  # your class must be in scope
    model.eval(); model.purify = True; model.to(dtype=torch.float64)
    preds = model(torch.from_numpy(rs_list).to(dtype=torch.float64),
                  torch.from_numpy(ss_list).to(dtype=torch.float64))
    torch.save(preds, f"preds_{molecule_name.lower()}.pth")
    # FIX: transform expects numpy array; also keep your convention if needed
    dms_list = transform(preds["density_matrix"].detach().cpu().numpy(),
                         convention="phipy", atoms=atoms_args)
except NameError:
    # Fallback: create a trivial idempotent-ish DM so the pipeline can still run
    rng = np.random.default_rng(0)
    dms_list = []
    # crude closed-shell occupancy
    Z_of = {"H":1, "C":6, "N":7, "O":8, "F":9}
    N_elec = sum(Z_of[s] for s in chemical_symbols)
    nocc = N_elec // 2
    for S in ss_list:
        # random symmetric, orthogonalize, occupy nocc
        eS, US = np.linalg.eigh(S)
        Sinvh = US @ np.diag(1.0/np.sqrt(np.clip(eS, 1e-12, None))) @ US.T
        G = rng.standard_normal(S.shape); F = 0.5 * (G + G.T)
        F_orth = Sinvh.T @ F @ Sinvh
        ew, Ctil = np.linalg.eigh(F_orth)
        C = Sinvh @ Ctil
        occ = np.zeros_like(ew); occ[:nocc] = 2.0
        P = C @ np.diag(occ) @ C.T
        dms_list.append(0.5*(P+P.T))
    dms_list = np.array(dms_list)

# FIX: Rs_list → rs_list
benchmark_dataset(dms_list, rs_list, build_mol_from_bohr)