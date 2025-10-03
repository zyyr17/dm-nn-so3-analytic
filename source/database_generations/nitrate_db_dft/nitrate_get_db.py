import numpy as np
import sqlite3 as sql
import torch
from argparse import Namespace
from pyscf import gto, scf, dft
from ase import Atoms, units
from ase.optimize import BFGS
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.calculators.xtb import XTB



# ---------- 1) Build planar NO3- and pre-opt with xTB ----------
pos = np.array([
    [ 0.000000,  0.000000,  0.000000],   # N
    [ 1.240000,  0.000000,  0.000000],   # O
    [-0.620000,  1.073869,  0.000000],   # O
    [-0.620000, -1.073869,  0.000000],   # O
], dtype=float)

atoms = Atoms('NO3', positions=pos, pbc=False, charge=-1, magmom=None)  # closed shell
atoms.calc = XTB(method='GFN2-xTB', charge=-1, uhf=0)
BFGS(atoms, logfile=None).run(fmax=1e-3)

# ---------- 2) MD schedule ----------
temps = [200, 300, 450, 600]  # K
dt_fs = 0.5                   # timestep (fs)
steps_per_temp = 150000       # 150k * 0.5 fs = 75 ps per T
sample_stride = 10            # record every 10 steps -> 15k samples / T
target_total = 30000

frames = []
energies = []

for T in temps:
    # reinitialize velocities and thermostat per temperature
    MaxwellBoltzmannDistribution(atoms, temperature_K=T)
    dyn = Langevin(atoms, timestep=dt_fs*units.fs, temperature_K=T, friction=0.02)

    def grab():
        frames.append(atoms.get_positions().copy())
        energies.append(atoms.get_potential_energy())

    dyn.attach(grab, interval=sample_stride)
    dyn.run(steps_per_temp)

# Concatenate and convert to Bohr
R = np.stack(frames, axis=0)                        # (N, 4, 3), Å
E = np.array(energies, dtype=np.float64)            # eV
bohr = 0.529177210903
R_bohr = (R / bohr).astype(np.float64)

# ---------- 3) Simple geometric filtering to remove outliers ----------
# N at index 0, O at indices 1..3
def ok(geom):
    # distances N-O
    d = np.linalg.norm(geom[1:4] - geom[0], axis=1)
    if not np.all((1.15 <= d) & (d <= 1.35)):  # Å
        return False
    # angles O-N-O
    v = geom[1:4] - geom[0]
    cs = lambda a, b: np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))
    angs = []
    for i in range(3):
        for j in range(i+1,3):
            c = np.clip(cs(v[i], v[j]), -1.0, 1.0)
            ang = np.degrees(np.arccos(c))
            angs.append(ang)
    return np.all((100.0 <= np.array(angs)) & (np.array(angs) <= 140.0))

mask = np.array([ok(r) for r in R], dtype=bool)
R_bohr = R_bohr[mask]; E = E[mask]

# ---------- 4) Energy stratification (uniform over percentiles) ----------
N = len(R_bohr)
bins = 50
q = np.linspace(0, 1, bins+1)
edges = np.quantile(E, q)
idx_sel = []
for b in range(bins):
    inside = np.where((E >= edges[b]) & (E <= edges[b+1]))[0]
    if inside.size == 0: 
        continue
    # sample up to k from this bin
    k = int(np.ceil(target_total / bins))
    pick = np.random.choice(inside, size=min(k, inside.size), replace=False)
    idx_sel.append(pick)
idx_sel = np.concatenate(idx_sel, axis=0)
if idx_sel.size > target_total:
    idx_sel = np.random.choice(idx_sel, size=target_total, replace=False)

R_bohr = R_bohr[idx_sel]
E = E[idx_sel]

# ---------- 5) Save ----------
np.savez_compressed("nitrate_rmd_like_30000_bohr_float64.npz", R=R_bohr, E=E)
print("Saved:", R_bohr.shape, "frames")


# -------- User settings --------
molecule_name = "NITRATE"
coords_npz = "nitrate_rmd_like_30000_bohr_float64.npz"   # npz["R"] in Bohr, shape [N,4,3]
basis_name = "def2-SVPD"
aux_jkfit  = "def2-universal-jkfit"   # use universal JK-fit for diffuse basis
xc         = "PBE"
mode       = "adaptive"
max_samples = None

dtype = torch.float64
charge, spin = -1, 0

# -------- Element settings / AO patterns --------
settings_all = {
    "H2O": {
        "symbols": ["O","H","H"],
        "atom_info": {"O": (0,0,0,1,1,2), "H": (0,0,1)}
    },
    "NH3": {
        "symbols": ["N","H","H","H"],
        "atom_info": {"N": (0,0,0,1,1,2), "H": (0,0,1)}
    },
    "CH4": {
        "symbols": ["C","H","H","H","H"],
        "atom_info": {"C": (0,0,0,1,1,2), "H": (0,0,1)}
    },
    "FH": {
        "symbols": ["F","H"],
        "atom_info": {"F": (0,0,0,1,1,2), "H": (0,0,1)}
    },
    "CH3CH2OH": {
        "symbols": ["C","C","O","H","H","H","H","H","H"],
        "atom_info": {"C": (0,0,0,1,1,2),
                      "O": (0,0,0,1,1,2),
                      "H": (0,0,1)}
    },
    "NITRATE": {
        "symbols": ["N","O","O","O"],
        # for def2-SVPD: N = 4s+2p+2d ; O = 4s+3p+2d
        "atom_info": {"N": (0,0,0,0,1,1,2,2),
                      "O": (0,0,0,0,1,1,1,2,2)}
    }
}

settings = settings_all[molecule_name]
symbols  = settings["symbols"]
atoms_str = "".join(symbols)
atom_info = settings["atom_info"]

Z_of = {"H":1,"C":6,"N":7,"O":8,"F":9}
Z_vector = torch.tensor([Z_of[s] for s in symbols], dtype=torch.int32)
unique_Z = sorted(set(Z_vector.tolist()))

# basisset blobs
orbitals_blobs = []
for Z in unique_Z:
    sym = [k for k,v in Z_of.items() if v==Z][0]
    orbitals_blobs.append(np.array(atom_info[sym], dtype=np.int32).tobytes())




convention_dict = {
   'pyscf': Namespace(
      atom_to_orbitals_map={'H':'ssp','O':'sssspppdd','C':'sssspppdd','N':'ssssppdd','F':'sssppd'},
      orbital_idx_map={'s':[0], 'p':[1,2,0], 'd':[4,2,0,1,3]},
      orbital_sign_map={'s':[1], 'p':[1,1,1], 'd':[1,1,1,1,1]},
      orbital_order_map={'H':[0,1,2],'O':[0,1,2,3,4,5],'C':[0,1,2,3,4,5],'N':[0,1,2,3,4,5],'F':[0,1,2,3,4,5]},
   ),
   # target convention you’re using downstream
   'pyphi': Namespace(
      atom_to_orbitals_map={'H':'ssp','O':'sssspppdd','C':'ssssppdd','N':'ssssppdd','F':'sssppd'},
      orbital_idx_map={'s':[0], 'p':[1,2,0], 'd':[0,1,2,3,4]},
      orbital_sign_map={'s':[1], 'p':[1,1,1], 'd':[1,1,1,1,1]},
      orbital_order_map={'H':[0,1,2],'O':[0,1,2,3,4,5, 6, 7, 8],'C':[0,1,2,3,4,5],'N':[0,1,2,3,4,5, 6, 7],'F':[0,1,2,3,4,5]},
   ),
}


# -------- AO convention maps (unchanged) --------

def transform(mat, convention="pyphi", atoms=atoms_str):
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

# -------- ORCA-like RKS runner --------
def run_orca_like_rks(mol, mode="adaptive"):
    mf = dft.RKS(mol).density_fit(auxbasis=aux_jkfit)
    mf.xc = xc
    mf.grids.prune = True
    mf.grids.atom_grid = (75, 302)
    mf.max_cycle = 180
    mf.diis_space = 12
    mf.conv_tol = 1e-9
    mf.conv_tol_grad = 1e-6
    mf.conv_tol_density = 1e-8
    mf.small_rho_cutoff = 1e-12
    mf.verbose = 0
    if mode=="fixed":   (mf.level_shift,mf.damp)=(0.3,0.2)
    elif mode=="adaptive": (mf.level_shift,mf.damp)=(0.3,0.2)
    else: (mf.level_shift,mf.damp)=(0.0,0.0)
    state = {"shed": mode!="adaptive", "Eprev": None, "Pprev": None}
    def cb(envs):
        if state["shed"]: return
        E, P = envs["e_tot"], envs["dm"]
        dE = abs(E - state["Eprev"]) if state["Eprev"] is not None else 1e9
        dP = np.linalg.norm(P - state["Pprev"]) if state["Pprev"] is not None else 1e9
        if (dP<1e-3) or (dE<1e-4):
            mf.level_shift=0.0; mf.damp=0.0; state["shed"]=True
        state["Eprev"],state["Pprev"]=E,P
    mf.callback = cb if mode=="adaptive" else None
    e = mf.kernel()
    if not mf.converged:
        mf.level_shift=0.0; mf.damp=0.0
        try: e=mf.newton().kernel()
        except: e=scf.newton(mf).kernel()
    dm=mf.make_rdm1()
    mf.grids.atom_grid=(99,590); mf.grids.prune=True
    mf.level_shift=0.0; mf.damp=0.0
    e_final=mf.kernel(dm0=dm)
    return mf,e_final

def dump_binaries(a: np.ndarray) -> bytes:
    return np.ascontiguousarray(a).astype(np.float64).ravel().tobytes()

def get_physics(key, mf):
    if key=="S": return mf.get_ovlp()
    if key=="P": return mf.make_rdm1()
    if key=="E": return mf.e_tot
    if key=="H": return mf.get_fock()
    if key=="C": return mf.get_hcore()
    raise ValueError(key)

# -------- Load coordinates --------
coords = np.load(coords_npz)["R"]
if max_samples is not None:
    coords = coords[:max_samples]
N = len(coords)

molecules=[]
for R in coords:
    atom_list=[(s,r) for s,r in zip(symbols,R)]
    mol=gto.Mole()
    mol.atom=atom_list
    mol.unit="Bohr"
    mol.basis=basis_name
    mol.charge=charge
    mol.spin=spin
    mol.verbose=0
    mol.build()
    molecules.append(mol)

# -------- SCF & collect --------
num_bulk_data={}
data_rows=[]
for i,mol in enumerate(molecules):
    mf,_=run_orca_like_rks(mol,mode=mode)
    P,S,H,C,E=(get_physics("P",mf),get_physics("S",mf),
               get_physics("H",mf),get_physics("C",mf),
               float(get_physics("E",mf)))
    R=mol.atom_coords()
    # reorder
    P=transform(P,atoms=atoms_str)
    S=transform(S,atoms=atoms_str)
    H=transform(H,atoms=atoms_str)
    C=transform(C,atoms=atoms_str)
    num_bulk_data[i]={
        "density_matrix":torch.tensor(P,dtype=dtype),
        "overlap_matrix":torch.tensor(S,dtype=dtype),
        "core_hamiltonian":torch.tensor(C,dtype=dtype),
        "fock_matrix":torch.tensor(H,dtype=dtype),
        "energy":torch.tensor(E,dtype=dtype),
        "positions":torch.tensor(R,dtype=dtype),
    }
    data_rows.append({
        "id":i,"R":dump_binaries(R),"E":E,
        "H":dump_binaries(H),"P":dump_binaries(P),
        "S":dump_binaries(S),"C":dump_binaries(C),
    })

# -------- Write SQLite --------
db_path=f"{molecule_name.lower()}_pbe-def2svpd_{N}.db"
conn=sql.connect(db_path); cur=conn.cursor()
cur.execute("""CREATE TABLE IF NOT EXISTS data (
    id INTEGER, R BLOB, E FLOAT, H BLOB, P BLOB, S BLOB, C BLOB);""")
cur.execute("""CREATE TABLE IF NOT EXISTS nuclear_charges (
    id INTEGER, N INTEGER, Z BLOB);""")
cur.execute("""CREATE TABLE IF NOT EXISTS basisset (
    Z INTEGER, orbitals BLOB);""")
cur.execute("""CREATE TABLE IF NOT EXISTS metadata (
    id INTEGER, N INTEGER, charge INTEGER, spin INTEGER);""")
for r in data_rows:
    cur.execute("INSERT INTO data (id,R,E,H,P,S,C) VALUES (?,?,?,?,?,?,?)",
                (r["id"],r["R"],float(r["E"]),r["H"],r["P"],r["S"],r["C"]))
Z_blob=Z_vector.cpu().numpy().astype(np.int32).tobytes()
cur.execute("INSERT INTO nuclear_charges (id,N,Z) VALUES (?,?,?)",
            (0,len(Z_vector),Z_blob))
cur.execute("DELETE FROM basisset")
for Z,orb_blob in zip(unique_Z,orbitals_blobs):
    cur.execute("INSERT INTO basisset (Z,orbitals) VALUES (?,?)",(int(Z),orb_blob))
cur.execute("INSERT INTO metadata (id,N,charge,spin) VALUES (?,?,?,?)",
            (0,N,charge,spin))
conn.commit(); conn.close()

# -------- Save Torch bundle --------
torch.save(num_bulk_data,f"{molecule_name.lower()}_num_bulk_data_{N}.pth")

print(f"✅ Wrote DB: {db_path}")
print(f"✅ Wrote Torch: {molecule_name.lower()}_num_bulk_data_{N}.pth")