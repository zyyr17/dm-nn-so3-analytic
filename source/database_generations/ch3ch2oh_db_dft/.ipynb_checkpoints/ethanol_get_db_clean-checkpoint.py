import torch
import numpy as np
import sqlite3 as sql
from argparse import Namespace
from pyscf import gto, scf, dft
from pyscf.grad import rhf
from ase.io import read
from ase.units import Bohr
from scipy.linalg import block_diag
import math
dtype = torch.float64
T = lambda x: torch.tensor(x, dtype=dtype)

read_binaries64 = lambda x: torch.frombuffer(x, dtype=torch.float64)
dump_binaries = lambda c: c.contiguous().view(-1).to(dtype=dtype).cpu().numpy().tobytes() if isinstance(c, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(c)).to(dtype).view(-1).cpu().numpy().tobytes()

###################################AUXILIAR UTILITIES############################################

molecule_name = "CH3CH2OH"  # Change to "CH4", "NH3", or "H2O" as needed
num_mols = 30000
# === db_info structure ===
# === db_info structure ===

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

nuclear_data = {
    "H2O": torch.tensor([8, 1, 1], dtype=torch.int32),
    "NH3": torch.tensor([7, 1, 1, 1], dtype=torch.int32),
    "CH4": torch.tensor([6, 1, 1, 1, 1], dtype=torch.int32),
    "FH":  torch.tensor([9, 1], dtype=torch.int32),
    "CH3CH2OH": torch.tensor([6, 6, 8, 1, 1, 1, 1, 1, 1], dtype=torch.int32)
}
Z_lists = {
    "H2O": [1, 8],
    "NH3": [1, 7],
    "CH4": [1, 6],
    "FH":  [1, 9],
    "CH3CH2OH": [1, 6, 8]
}
xyz_paths = {
    "NH3": "NH3..DFT.MD.300K.PBE.light.01.Movie.xyz",
    "FH":  "FH..DFT.MD.300K.PBE.light.01.Movie.xyz",
    "CH4": "CH4..DFT.MD.300K.PBE.light.01.Movie.xyz",
    "H2O": "H2O..DFT.MD.300K.PBE.light.01.Movie.xyz"
    
}


symbols = settings["symbols"]
atoms_args = "".join(symbols)
order = settings["order"]
atom_info = settings["atom_info"]
Z_vector = nuclear_data[molecule_name]
Z_list = Z_lists[molecule_name]

db_info = {
    "data": {
        "id": [list(range(num_mols)), "INTEGER"],
        "R": [[], "BLOB"],
        "E": [[], "FLOAT"],
        "H": [[], "BLOB"],
        "P": [[], "BLOB"],
        "S": [[], "BLOB"],
        "C": [[], "BLOB"]
    },
    "nuclear_charges": {
        "id": [[0], "INTEGER"],
        "N": [[len(Z_vector)], "INTEGER"],
        "Z": [[Z_vector.numpy().tobytes()], "BLOB"]
    },
    "basisset": {
        "Z": [Z_list, "INTEGER"],
        "orbitals": [[np.array([0, 0, 1], dtype=np.int32).tobytes(), np.array([0, 0, 0, 1, 1, 2], dtype=np.int32).tobytes(), np.array([0, 0, 0, 1, 1, 2], dtype=np.int32).tobytes()], "BLOB"]
    },
    "metadata": {
        "id": [[0], "INTEGER"],
        "N": [[num_mols ], "INTEGER"]
    }
}
orbitals_data=db_info["basisset"]["orbitals"][0]
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
def get_physics(key, mf):
    methods = {
        "S": lambda: mf.get_ovlp(),
        "P": lambda: mf.make_rdm1(),
        "E": lambda: mf.e_tot,
        "H": lambda: mf.get_fock(),
        "C": lambda: mf.get_hcore()
    }
    assert key in methods, f"Invalid key: {key}"
    return methods[key]()


def transform(hamiltonians,convention = "pyphi", atoms = atoms_args ):
    conv = convention_dict[convention]
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



def run_orca_like_rks(mol, mode="adaptive"):
    """
    mode: "diis" (no stabilizers), "fixed" (always on), "adaptive" (on, then auto-off)
    """
    mf = dft.RKS(mol).density_fit(auxbasis='def2-svp-jkfit')
    mf.xc = 'PBE'
    mf.grids.prune = True
    mf.grids.atom_grid = (75, 302)  # ~Grid4
    mf.max_cycle = 180
    mf.diis_space = 12
    mf.conv_tol = 1e-9
    if hasattr(mf, 'conv_tol_grad'):    mf.conv_tol_grad = 1e-6
    if hasattr(mf, 'conv_tol_density'): mf.conv_tol_density = 1e-8
    if hasattr(mf, 'small_rho_cutoff'): mf.small_rho_cutoff = 1e-10
    mf.verbose = 0

    # set stabilizers per mode
    if mode == "diis":
        ls0, dp0 = 0.0, 0.0
    elif mode == "fixed":
        ls0, dp0 = 0.3, 0.2
    elif mode == "adaptive":
        ls0, dp0 = 0.3, 0.2
    else:
        raise ValueError("mode ∈ {'diis','fixed','adaptive'}")

    if hasattr(mf, 'level_shift'): mf.level_shift = ls0
    if hasattr(mf, 'damp'):        mf.damp = dp0

    # adaptive shedding (only if mode='adaptive')
    state = {"shed": (mode!="adaptive"), "Eprev": None, "Pprev": None, "n": 0}

    def cb(envs):
        state["n"] += 1
        if not state["shed"]:
            E = envs["e_tot"]; P = envs["dm"]
            dE = abs(E - state["Eprev"]) if state["Eprev"] is not None else float("inf")
            dP = np.linalg.norm(P - state["Pprev"]) if state["Pprev"] is not None else float("inf")
            # thresholds are loose to switch off early once stable
            if dP < 1e-3 or dE < 1e-4:
                if hasattr(mf,'level_shift'): mf.level_shift = 0.0
                if hasattr(mf,'damp'):        mf.damp = 0.0
                state["shed"] = True
            state["Eprev"], state["Pprev"] = E, P

    mf.callback = cb if mode == "adaptive" else None

    # first pass
    e_scf = mf.kernel()

    # rescue if needed
    if not mf.converged:
        if hasattr(mf,'level_shift'): mf.level_shift = 0.0
        if hasattr(mf,'damp'):        mf.damp = 0.0
        try:
            e_scf = mf.newton().kernel()
        except Exception:
            e_scf = scf.newton(mf).kernel()

    # final polish on finer grid (symmetric)
    dm = mf.make_rdm1()
    mf.grids.atom_grid = (99, 590)  # ~Grid5
    mf.grids.prune = True
    if hasattr(mf,'level_shift'): mf.level_shift = 0.0
    if hasattr(mf,'damp'):        mf.damp = 0.0
    e_final = mf.kernel(dm0=dm)
    return mf, e_final


coords_arrays = np.load("ethanol_rmd17_30k_bohr_float64.npz")["R"][:]

molecules = []
for coords_array in coords_arrays:
    atom_list = [(sym, pos) for sym, pos in zip(symbols, coords_array)]
    mol = gto.Mole()
    mol.atom = atom_list
    mol.unit = "Bohr"
    mol.basis = "def2-SVP"
    mol.build()
    molecules.append(mol)

quant_keys = ["density_matrix", "overlap_matrix", "positions", "core_hamiltonian", "fock_matrix", "energy"]
# === SCF + matrix extraction ===
num_bulk_data = {i: {qk: torch.from_numpy(mol.atom_coords()).to(dtype=dtype) if qk == "positions" else None for qk in quant_keys} for i, mol in enumerate(molecules)}
key_mapping = {
    "P": "density_matrix",
    "S": "overlap_matrix",
    "C": "core_hamiltonian",
    "H": "fock_matrix",
    "E": "energy"
}

###################################MAIN LOOP############################################
for ind, mol in enumerate(molecules):
    mf, _ = run_orca_like_rks(mol)

    for key in db_info["data"]:
        if key in ("id", "R"):
            continue
        value = np.asarray(get_physics(key, mf))
        if key != "E":
            value = transform(value, convention="pyphi", atoms=atoms_args)
        if key in key_mapping:
            num_bulk_data[ind][key_mapping[key]] = torch.from_numpy(value)
        if key == "E":
            db_info["data"][key][0].append(value)
        else:
            db_info["data"][key][0].append(dump_binaries(value))
    db_info["data"]["R"][0].append(dump_binaries(mol.atom_coords()))

##############################################################################


# === Save PyTorch files ===

torch.save(num_bulk_data, "num_bulk_data.pth")


# === Create SQLite DB ===
db_path = f"{molecule_name.lower()}_pbe-def2svp_{num_mols}.db"
conn = sql.connect(db_path)
cursor = conn.cursor()

# Fix: prevent overwriting global dtype
for table_name, table_data in db_info.items():
    columns = [f"{col} {sqltype}" for col, (_, sqltype) in table_data.items()]
    col_def_str = ", ".join(columns)
    cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({col_def_str});")

# Insert values
for table_name, table_data in db_info.items():
    keys = list(table_data.keys())
    values_lists = [table_data[k][0] for k in keys]
    num_rows = len(values_lists[0])
    for i in range(num_rows):
        row = [values[i] for values in values_lists]
        placeholders = ", ".join(["?"] * len(row))
        cursor.execute(f"INSERT INTO {table_name} ({', '.join(keys)}) VALUES ({placeholders});", row)

# === Clear old entries (optional, for safety) ===
cursor.execute("DELETE FROM basisset")

# === Insert updated basisset data ===
for Z, orb_blob in zip(Z_list, orbitals_data):
    cursor.execute("INSERT INTO basisset (Z, orbitals) VALUES (?, ?)", (Z, orb_blob))


conn.commit()
conn.close()
print(f"✅ Database created at {db_path}")

# PATCH DB

conn = sql.connect(db_path)
cursor = conn.cursor()

# === Clear old entries (optional, for safety) ===
cursor.execute("DELETE FROM basisset")

# === Insert updated basisset data ===
for Z, orb_blob in zip(Z_list, orbitals_data):
    cursor.execute("INSERT INTO basisset (Z, orbitals) VALUES (?, ?)", (Z, orb_blob))

# === Commit and close ===
conn.commit()
conn.close()

print("Successfully updated the 'basisset' table in", db_path)



