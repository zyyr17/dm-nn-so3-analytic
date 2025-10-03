import numpy as np
import sqlite3 as sql
import torch
from argparse import Namespace
from pyscf import gto, scf, dft

# -------- User settings --------
molecule_name = "CH3CH2OH"             # "H2O" | "NH3" | "CH4" | "FH" | "CH3CH2OH"
coords_npz = "ethanol_rmd17_30k_bohr_float64.npz"  # file with npz["R"] in Bohr
basis_name = "def2-SVP"
aux_jkfit  = "def2-svp-jkfit"
xc         = "PBE"
mode       = "adaptive"                # 'diis' | 'fixed' | 'adaptive'
max_samples = None                     # None -> use all; or set an int

dtype = torch.float64

# -------- Element settings / AO patterns --------
settings_all = {
    "H2O": {
        "symbols": ["O", "H", "H"],
        "atom_info": {"O": (0,0,0,1,1,2), "H": (0,0,1)}
    },
    "NH3": {
        "symbols": ["N", "H", "H", "H"],
        "atom_info": {"N": (0,0,0,1,1,2), "H": (0,0,1)}
    },
    "CH4": {
        "symbols": ["C", "H", "H", "H", "H"],
        "atom_info": {"C": (0,0,0,1,1,2), "H": (0,0,1)}
    },
    "FH": {
        "symbols": ["F", "H"],
        "atom_info": {"F": (0,0,0,1,1,2), "H": (0,0,1)}
    },
    "CH3CH2OH": {
        "symbols": ["C","C","O","H","H","H","H","H","H"],
        "atom_info": {"C": (0,0,0,1,1,2), "O": (0,0,0,1,1,2), "H": (0,0,1)}
    }
}
settings = settings_all[molecule_name]
symbols  = settings["symbols"]
atoms_str = "".join(symbols)  # fine for elements here (all single-letter)
atom_info = settings["atom_info"]
Z_of = {"H":1, "C":6, "N":7, "O":8, "F":9}
Z_vector = torch.tensor([Z_of[s] for s in symbols], dtype=torch.int32)
unique_Z = sorted(set(Z_vector.tolist()))
# For the basisset table, we’ll store one row per unique Z with an l-pattern blob
Z_list = unique_Z
orbitals_blobs = []
for Z in Z_list:
    sym = [k for k,v in Z_of.items() if v == Z][0]
    orbitals_blobs.append(np.array(atom_info[sym], dtype=np.int32).tobytes())

# -------- AO convention maps --------
convention_dict = {
   'pyscf': Namespace(
      atom_to_orbitals_map={'H':'ssp','O':'sssppd','C':'sssppd','N':'sssppd','F':'sssppd'},
      orbital_idx_map={'s':[0], 'p':[1,2,0], 'd':[4,2,0,1,3]},
      orbital_sign_map={'s':[1], 'p':[1,1,1], 'd':[1,1,1,1,1]},
      orbital_order_map={'H':[0,1,2],'O':[0,1,2,3,4,5],'C':[0,1,2,3,4,5],'N':[0,1,2,3,4,5],'F':[0,1,2,3,4,5]},
   ),
   # target convention you’re using downstream
   'pyphi': Namespace(
      atom_to_orbitals_map={'H':'ssp','O':'sssppd','C':'sssppd','N':'sssppd','F':'sssppd'},
      orbital_idx_map={'s':[0], 'p':[1,2,0], 'd':[0,1,2,3,4]},
      orbital_sign_map={'s':[1], 'p':[1,1,1], 'd':[1,1,1,1,1]},
      orbital_order_map={'H':[0,1,2],'O':[0,1,2,3,4,5],'C':[0,1,2,3,4,5],'N':[0,1,2,3,4,5],'F':[0,1,2,3,4,5]},
   ),
}

def transform(mat, convention="pyphi", atoms=atoms_str):
    """Reorder AO blocks to your target convention; preserves symmetry."""
    conv = convention_dict[convention]
    orbitals = ''
    order = []
    for a in atoms:
        offset = len(order)
        orbitals += conv.atom_to_orbitals_map[a]
        order += [idx + offset for idx in conv.orbital_order_map[a]]
    idxs, sgns = [], []
    for orb in orbitals:
        offset = sum(map(len, idxs))
        idxs.append(np.array(conv.orbital_idx_map[orb]) + offset)
        sgns.append(np.array(conv.orbital_sign_map[orb]))
    idxs = [idxs[i] for i in order]
    sgns = [sgns[i] for i in order]
    idxs = np.concatenate(idxs).astype(int)
    sgns = np.concatenate(sgns)
    m2 = mat[..., idxs, :][..., :, idxs]
    m2 = m2 * sgns[:,None]
    m2 = m2 * sgns[None,:]
    return m2

# -------- ORCA-like RKS runner (adaptive stabilizers) --------
def run_orca_like_rks(mol, mode="adaptive"):
    mf = dft.RKS(mol).density_fit(auxbasis=aux_jkfit)
    mf.xc = xc
    mf.grids.prune = True
    mf.grids.atom_grid = (75, 302)      # ~ ORCA Grid4
    mf.max_cycle = 180
    mf.diis_space = 12
    mf.conv_tol = 1e-9
    if hasattr(mf,'conv_tol_grad'):    mf.conv_tol_grad = 1e-6
    if hasattr(mf,'conv_tol_density'): mf.conv_tol_density = 1e-8
    if hasattr(mf,'small_rho_cutoff'): mf.small_rho_cutoff = 1e-10
    mf.verbose = 0

    # stabilizers
    if mode == "diis":   (ls0, dp0) = (0.0, 0.0)
    elif mode == "fixed":(ls0, dp0) = (0.3, 0.2)
    elif mode == "adaptive": (ls0, dp0) = (0.3, 0.2)
    else: raise ValueError("mode ∈ {'diis','fixed','adaptive'}")
    if hasattr(mf,'level_shift'): mf.level_shift = ls0
    if hasattr(mf,'damp'):        mf.damp = dp0

    state = {"shed": mode!="adaptive", "Eprev": None, "Pprev": None}
    def cb(envs):
        if state["shed"]: return
        E = envs["e_tot"]; P = envs["dm"]
        dE = abs(E - state["Eprev"]) if state["Eprev"] is not None else 1e9
        dP = np.linalg.norm(P - state["Pprev"]) if state["Pprev"] is not None else 1e9
        if (dP < 1e-3) or (dE < 1e-4):
            if hasattr(mf,'level_shift'): mf.level_shift = 0.0
            if hasattr(mf,'damp'):        mf.damp = 0.0
            state["shed"] = True
        state["Eprev"], state["Pprev"] = E, P

    mf.callback = cb if mode == "adaptive" else None

    e = mf.kernel()
    if not mf.converged:
        if hasattr(mf,'level_shift'): mf.level_shift = 0.0
        if hasattr(mf,'damp'):        mf.damp = 0.0
        try:    e = mf.newton().kernel()
        except: e = scf.newton(mf).kernel()

    # final polish on finer grid
    dm = mf.make_rdm1()
    mf.grids.atom_grid = (99, 590)      # ~ ORCA Grid5
    mf.grids.prune = True
    if hasattr(mf,'level_shift'): mf.level_shift = 0.0
    if hasattr(mf,'damp'):        mf.damp = 0.0
    e_final = mf.kernel(dm0=dm)
    return mf, e_final

def dump_binaries(a: np.ndarray) -> bytes:
    return np.ascontiguousarray(a).astype(np.float64).ravel().tobytes()

def get_physics(key, mf):
    if key == "S": return mf.get_ovlp()
    if key == "P": return mf.make_rdm1()
    if key == "E": return mf.e_tot
    if key == "H": return mf.get_fock()   # KS / Fock
    if key == "C": return mf.get_hcore()  # core H
    raise ValueError(key)

# -------- Load coordinates and build molecules --------
coords = np.load(coords_npz)["R"]  # [N, natoms, 3], Bohr
if max_samples is not None:
    coords = coords[:max_samples]
N = len(coords)

molecules = []
for R in coords:
    atom_list = [(s, r) for s, r in zip(symbols, R)]
    mol = gto.Mole()
    mol.atom = atom_list
    mol.unit = "Bohr"
    mol.basis = basis_name
    mol.charge = 0
    mol.spin = 0
    mol.verbose = 0
    mol.build()
    molecules.append(mol)

# -------- SCF & collect data --------
num_bulk_data = {}   # torch bundle
data_rows = []       # for SQLite 'data'

for i, mol in enumerate(molecules):
    mf, _ = run_orca_like_rks(mol, mode=mode)

    P = get_physics("P", mf); S = get_physics("S", mf)
    H = get_physics("H", mf); C = get_physics("C", mf)
    E = float(get_physics("E", mf))
    R = mol.atom_coords()  # Bohr

    # AO reordering (skip energy/positions)
    P = transform(P, convention="pyphi", atoms=atoms_str)
    S = transform(S, convention="pyphi", atoms=atoms_str)
    H = transform(H, convention="pyphi", atoms=atoms_str)
    C = transform(C, convention="pyphi", atoms=atoms_str)

    num_bulk_data[i] = {
        "density_matrix":  torch.tensor(P, dtype=dtype),
        "overlap_matrix":  torch.tensor(S, dtype=dtype),
        "core_hamiltonian":torch.tensor(C, dtype=dtype),
        "fock_matrix":     torch.tensor(H, dtype=dtype),
        "energy":          torch.tensor(E, dtype=dtype),
        "positions":       torch.tensor(R, dtype=dtype),
    }

    data_rows.append({
        "id": i,
        "R": dump_binaries(R),
        "E": E,
        "H": dump_binaries(H),
        "P": dump_binaries(P),
        "S": dump_binaries(S),
        "C": dump_binaries(C),
    })

# -------- Write SQLite DB (length-safe) --------
db_path = f"{molecule_name.lower()}_pbe-def2svp_{N}.db"
conn = sql.connect(db_path)
cur = conn.cursor()

cur.execute("""CREATE TABLE IF NOT EXISTS data (
    id INTEGER, R BLOB, E FLOAT, H BLOB, P BLOB, S BLOB, C BLOB
);""")
cur.execute("""CREATE TABLE IF NOT EXISTS nuclear_charges (
    id INTEGER, N INTEGER, Z BLOB
);""")
cur.execute("""CREATE TABLE IF NOT EXISTS basisset (
    Z INTEGER, orbitals BLOB
);""")
cur.execute("""CREATE TABLE IF NOT EXISTS metadata (
    id INTEGER, N INTEGER
);""")

# data
for r in data_rows:
    cur.execute("INSERT INTO data (id,R,E,H,P,S,C) VALUES (?,?,?,?,?,?,?)",
                (r["id"], r["R"], float(r["E"]), r["H"], r["P"], r["S"], r["C"]))

# nuclear_charges: single row describing the molecule template
Z_blob = Z_vector.cpu().numpy().astype(np.int32).tobytes()
cur.execute("INSERT INTO nuclear_charges (id, N, Z) VALUES (?,?,?)",
            (0, len(Z_vector), Z_blob))

# basisset: one per unique Z (use the blobs we built above)
cur.execute("DELETE FROM basisset")
for Z, orb_blob in zip(Z_list, orbitals_blobs):
    cur.execute("INSERT INTO basisset (Z, orbitals) VALUES (?,?)", (int(Z), orb_blob))

# metadata
cur.execute("INSERT INTO metadata (id, N) VALUES (?,?)", (0, N))

conn.commit(); conn.close()

# -------- Save torch bundle --------
torch.save(num_bulk_data, f"{molecule_name.lower()}_num_bulk_data_{N}.pth")

print(f"✅ Wrote DB: {db_path}")
print(f"✅ Wrote Torch: {molecule_name.lower()}_num_bulk_data_{N}.pth")