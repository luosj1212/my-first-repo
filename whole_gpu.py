import os
import json
import re
import math
import argparse
import shutil
import subprocess
import tempfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import numpy as np
import torch
from torch_geometric.data import Data, Batch

import MDAnalysis as mda
from rdkit import Chem

import func as FF
from integrated_force import GF_min, GF_max
from model_function import DualGCNWithAttention

import cut
from cut import (
    parse_itp_file,
    extract_surface_water_subsystems,
    extract_submolecule,
    convert_to_rdkit_molecule,
    add_hydrogens_to_breaks,
    remove_isolated_hydrogens,
)

# --------------------------------------------------------------------------------------
# Feature & graph helpers (kept consistent with training/integrated_force)
# --------------------------------------------------------------------------------------
ONEHOT_RULE = {"H": [1, 0, 0], "C": [0, 1, 0], "O": [0, 0, 1]}
_ELEMS = ["H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "Si", "B"]
_E2IDX = {e: i for i, e in enumerate(_ELEMS)}

def _to_builtin(obj):
    """Recursively convert numpy / torch scalars and arrays into plain
    Python types (int, float, list, dict) so that json.dumps can handle them.
    """
    # numpy scalar
    if isinstance(obj, np.generic):
        return obj.item()

    # numpy array
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # torch tensor
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()

    # list / tuple
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(x) for x in obj]

    # dict
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}

    # other types (str, int, float, None, etc.)
    return obj

def _node_features_onehot(rd_mol: Chem.Mol) -> np.ndarray:
    k = len(ONEHOT_RULE)
    x = np.zeros((rd_mol.GetNumAtoms(), k), dtype=np.float32)
    for i, a in enumerate(rd_mol.GetAtoms()):
        sym = a.GetSymbol().upper()
        if sym not in ONEHOT_RULE:
            raise ValueError(f"Unsupported element '{sym}' for one-hot features")
        x[i] = np.asarray(ONEHOT_RULE[sym], dtype=np.float32)
    return x


def _onehot_elem(symbol: str, dim: int = len(_ELEMS)):
    v = np.zeros(dim, dtype=np.float32)
    v[_E2IDX.get(symbol, 0)] = 1.0
    return v


def _basic_node_features(rd_mol: Chem.Mol) -> np.ndarray:
    feats = []
    for a in rd_mol.GetAtoms():
        vec = _onehot_elem(a.GetSymbol()).tolist()
        vec += [a.GetAtomicNum() / 100.0, a.GetDegree() / 4.0, float(a.GetIsAromatic())]
        feats.append(vec)
    return np.array(feats, dtype=np.float32)


def _adj_and_dist(rd_mol: Chem.Mol, coordsA: np.ndarray, bond_factor: float = 1.2):
    pos_nm = np.asarray(coordsA, dtype=np.float64) * 0.1
    n = pos_nm.shape[0]

    diff = pos_nm[:, None, :] - pos_nm[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)

    radii = np.array([FF.COVALENT_RADII.get(a.GetSymbol(), 0.1) for a in rd_mol.GetAtoms()], dtype=np.float64)
    rsum = radii[:, None] + radii[None, :]
    bonded = (dist <= (bond_factor * rsum)) & (dist > 1e-8)

    ui, uj = np.where(np.triu(bonded, k=1))
    if ui.size > 0:
        adj_pairs = np.stack([np.concatenate([ui, uj]), np.concatenate([uj, ui])], axis=0)
        adj_ei = adj_pairs.astype(np.int64)
        adj_w = np.ones(adj_ei.shape[1], dtype=np.float32)
    else:
        adj_ei = np.zeros((2, 0), dtype=np.int64)
        adj_w = np.zeros((0,), dtype=np.float32)

    rows, cols = np.where(~np.eye(n, dtype=bool))
    dist_ei = np.stack([rows, cols], axis=0).astype(np.int64)
    invd = 1.0 / (dist[rows, cols] + 1e-12)
    dist_w = invd.astype(np.float32)

    return adj_ei, adj_w, dist_ei, dist_w, dist


def _global_features_raw(coordsA: np.ndarray, rd_mol: Chem.Mol, center_local: int):
    coords_nm = coordsA * 0.1
    masses = np.array([a.GetMass() for a in rd_mol.GetAtoms()], dtype=np.float64)
    cm = (coords_nm * masses[:, None]).sum(axis=0) / (masses.sum() + 1e-12)
    rel = coords_nm - cm
    Ixx = np.sum(masses * (rel[:, 1] ** 2 + rel[:, 2] ** 2))
    Iyy = np.sum(masses * (rel[:, 0] ** 2 + rel[:, 2] ** 2))
    Izz = np.sum(masses * (rel[:, 0] ** 2 + rel[:, 1] ** 2))
    Ixy = np.sum(masses * rel[:, 0] * rel[:, 1])
    Ixz = np.sum(masses * rel[:, 0] * rel[:, 2])
    Iyz = np.sum(masses * rel[:, 1] * rel[:, 2])
    inertia = np.array([[Ixx, -Ixy, -Ixz], [-Ixy, Iyy, -Iyz], [-Ixz, -Iyz, Izz]], dtype=np.float64)
    eig = np.linalg.eigvalsh(inertia)
    I1, I2, I3 = np.sort(eig)
    NPR1 = float(I1 / (I3 + 1e-12))
    NPR2 = float(I2 / (I3 + 1e-12))
    rg_frame = float(np.sqrt(((coords_nm - coords_nm.mean(axis=0)) ** 2).sum(axis=1).mean()))
    rg_mol = float(np.sqrt(((coords_nm - coords_nm.mean(axis=0)) ** 2).sum(axis=1).mean()))

    R_max = 1.0
    dr = 0.05
    edges = np.arange(0.0, R_max + dr, dr)
    c = coords_nm[center_local]
    d = np.linalg.norm(coords_nm - c, axis=1)
    dens = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (d >= lo) & (d < hi)
        vol = (4.0 / 3.0) * math.pi * (hi**3 - lo**3)
        m = masses[mask].sum()
        dens.append(float(m / vol) if vol > 0 else 0.0)

    raw = [
        float(masses.sum()),
        int(np.array([a.GetSymbol() != "H" for a in rd_mol.GetAtoms()]).sum()),
        int(rd_mol.GetNumBonds()),
        float(I1),
        float(I2),
        float(I3),
        NPR1,
        NPR2,
        rg_frame,
        rg_mol,
    ] + dens
    norm = [(raw[i] - GF_min[i]) / (GF_max[i] - GF_min[i]) if GF_max[i] > GF_min[i] else 0.0 for i in range(len(raw))]
    return raw, norm


def build_data(rd_mol: Chem.Mol, coordsA: np.ndarray, center_local: int, x_override=None):
    if x_override is None:
        try:
            x = _node_features_onehot(rd_mol)
        except Exception:
            x = _basic_node_features(rd_mol)
    else:
        x = np.asarray(x_override)
    adj_ei, adj_w, dist_ei, dist_w, dist_full = _adj_and_dist(rd_mol, coordsA, bond_factor=1.2)
    global_raw, global_norm = _global_features_raw(coordsA, rd_mol, center_local)
    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        pos=torch.tensor(coordsA * 0.1, dtype=torch.float32),
        center_mask=torch.nn.functional.one_hot(torch.tensor(center_local), num_classes=x.shape[0]).bool(),
        adj_edge_index=torch.tensor(adj_ei, dtype=torch.long),
        adj_edge_weight=torch.tensor(adj_w, dtype=torch.float32),
        dist_edge_index=torch.tensor(dist_ei, dtype=torch.long),
        dist_edge_weight=torch.tensor(dist_w, dtype=torch.float32),
        global_features=torch.tensor(global_norm, dtype=torch.float32),
    )
    data._global_raw = global_raw
    data._dist_full = dist_full
    return data


def load_packaged_model(pkg_path: str, device: torch.device) -> torch.nn.Module:
    obj = torch.load(pkg_path, map_location=device)
    if isinstance(obj, torch.nn.Module):
        return obj.to(device).eval()
    if isinstance(obj, dict) and "arch" in obj and "state_dict" in obj:
        arch = obj["arch"]
        m = DualGCNWithAttention(**arch).to(device)
        m.load_state_dict(obj["state_dict"], strict=True)
        m.eval()
        return m
    sd = obj if isinstance(obj, dict) else None
    if sd is None:
        raise RuntimeError(f"Unsupported checkpoint format: {type(obj)} at {pkg_path}")
    in_w = sd.get("input_proj.weight", None)
    if in_w is None:
        raise RuntimeError("state_dict missing input_proj.weight; cannot infer dims. Use packaged checkpoint.")
    hidden_dim, input_dim = in_w.shape
    arch = dict(input_dim=input_dim, hidden_dim=hidden_dim, num_blocks=3, gat_heads=4, att_depth=1, rbf_K=24, dropout=0.15, global_feat_dim=30, output_dim=1)
    m = DualGCNWithAttention(**arch).to(device)
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m


# --------------------------------------------------------------------------------------
# Utilities for slicing and batching
# --------------------------------------------------------------------------------------

def _as_element(name: str) -> str:
    if not name:
        return "C"
    s = re.sub(r"\d+$", "", "".join(ch for ch in str(name) if ch.isalpha()))
    if not s:
        return "C"
    if len(s) == 1:
        return s.upper()
    return s[0].upper() + s[1:].lower()


def _normalize_dist_matrix(dist_full: np.ndarray) -> np.ndarray:
    inv = np.zeros_like(dist_full, dtype=np.float32)
    mask = ~np.eye(dist_full.shape[0], dtype=bool)
    inv[mask] = 1.0 / (dist_full[mask] + 1e-12)
    non_zero = inv[mask]
    if non_zero.size > 0:
        mn, mx = float(non_zero.min()), float(non_zero.max())
        norm = (inv - mn) / (mx - mn) if mx > mn else np.zeros_like(inv)
    else:
        norm = np.zeros_like(inv)
    return inv, norm


def _build_matrices_from_data(data: Data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = data.x.shape[0]
    adj = np.zeros((n, n), dtype=np.int64)
    if hasattr(data, "adj_edge_index"):
        ei = data.adj_edge_index.cpu().numpy()
        adj[ei[0], ei[1]] = 1
    dist_full = getattr(data, "_dist_full", None)
    if dist_full is None:
        dist_full = np.zeros((n, n), dtype=np.float32)
    dist_inv, dist_inv_norm = _normalize_dist_matrix(dist_full)
    return adj, dist_inv, dist_inv_norm


def _select_centers(u: mda.Universe, centers: str) -> List[int]:
    if centers == "water":
        return u.select_atoms("resname SOL").indices.tolist()
    if centers == "non-water":
        return u.select_atoms("not resname SOL").indices.tolist()
    return u.atoms.indices.tolist()


def _within_range(idx: int, idx_range: Optional[Tuple[int, int]]) -> bool:
    if idx_range is None:
        return True
    start, end = idx_range
    return (idx + 1) >= start and (idx + 1) <= end

def _prepare_slice(
    u: mda.Universe,
    center_idx: int,
    radius: float,
    water_radius: Optional[float],
    atoms_info=None,
    bonds_info=None,
) -> Optional[Tuple[Chem.Mol, np.ndarray, int, List[str], List[float]]]:
    """
    先用 extract_submolecule 做几何 cut，然后按顺序：
      1) remove_isolated_hydrogens：删掉原切片里“漂浮”的孤立氢
      2) add_hydrogens_to_breaks：在被截断的键上按 itp 补氢
    最后返回“删孤立氢 + 补断键氢”之后的 RDKit Mol 与坐标、类型等。
    """
    center_atom = u.atoms[center_idx]

    # 1) 几何 cut
    idx_list, sub_ag = extract_submolecule(
        u,
        center_atom,
        radius=radius,
        water_radius=water_radius,
    )
    if not idx_list:
        return None

    # 2) 基础 RDKit mol + 局部坐标
    rdkit_mol, sub_resnames = convert_to_rdkit_molecule(sub_ag)
    sub_coords_local = np.asarray(sub_ag.positions, dtype=np.float32)

    # 是否有 itp 信息（才能做补氢）
    use_h_fix = atoms_info is not None and bonds_info is not None

    rd_mol_final = rdkit_mol

    if use_h_fix:
        try:
            # ---- 第一步：先删除原切片里的孤立氢 ----
            rd_noH, res_noH = remove_isolated_hydrogens(rdkit_mol, sub_resnames)

            # ---- 第二步：在被截断的键上按 itp 补氢 ----
            rd_H, res_H = add_hydrogens_to_breaks(
                rd_noH,
                atoms_info,
                bonds_info,
                sub_coords_local,  # 还是用原 Universe 的局部坐标来判断几何
                idx_list,
                u,
            )

            rd_mol_final = rd_H
            # 如果以后想把 resnames 写回 .gro，可以顺便把 res_H 保存下来
            # resnames_final = res_H
        except Exception as e:
            print(f"[WARN] H-fix (remove+add) failed at center {center_idx}: {e}. Fallback to original slice.")
            rd_mol_final = rdkit_mol
    else:
        rd_mol_final = rdkit_mol

    # 3) 从最终 RDKit Mol 中取坐标 / 原子类型 / 电荷等
    conf = rd_mol_final.GetConformer()
    n = rd_mol_final.GetNumAtoms()
    coordsA = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        p = conf.GetAtomPosition(i)
        coordsA[i] = [float(p.x), float(p.y), float(p.z)]

    atom_types = [a.GetSymbol() for a in rd_mol_final.GetAtoms()]

    # charges：如果 sub_ag 上有 charges 属性，就拷过去；否则置 0
    if hasattr(sub_ag, "charges"):
        charges = np.asarray(sub_ag.charges, dtype=np.float32)
        if charges.shape[0] != n:
            # 如果切完 / 删氢 / 补氢之后原子数发生变化，简单兜底成全 0
            charges = np.zeros(n, dtype=np.float32)
    else:
        charges = np.zeros(n, dtype=np.float32)

    # 中心原子在切片中的 local index：
    # 这里简单地用“离原始中心最近的那个原子”作为切片内部中心
    center_pos = center_atom.position
    d2 = np.square(coordsA - center_pos).sum(axis=1)
    center_local = int(d2.argmin())

    return rd_mol_final, coordsA, center_local, atom_types, charges.tolist()

def _build_payload(args):
    gidx, mol_block, center_local = args
    rd_mol = Chem.MolFromMolBlock(mol_block, sanitize=False, removeHs=False)
    if rd_mol is None:
        raise ValueError("MolFromMolBlock failed in worker.")
    conf = rd_mol.GetConformer()
    posA = np.asarray(conf.GetPositions(), dtype=np.float32)
    data = build_data(rd_mol, posA, int(center_local))

    def _to_np(x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    payload = {
        "pos": _to_np(getattr(data, "pos", None)),
        "x": _to_np(getattr(data, "x", None)),
        "adj_edge_index": _to_np(getattr(data, "adj_edge_index", None)),
        "adj_edge_weight": _to_np(getattr(data, "adj_edge_weight", None)),
        "dist_edge_index": _to_np(getattr(data, "dist_edge_index", None)),
        "dist_edge_weight": _to_np(getattr(data, "dist_edge_weight", None)),
        "global_features": _to_np(getattr(data, "global_features", None)),
        "global_raw": getattr(data, "_global_raw", None),
        "dist_full": getattr(data, "_dist_full", None),
        "center": int(center_local),
        "num_nodes": int(getattr(data, "num_nodes", posA.shape[0])),
    }
    return int(gidx), payload


def _payload_to_pyg(payload):
    def _to_torch(x, dtype=None):
        if x is None:
            return None
        t = torch.from_numpy(x) if isinstance(x, np.ndarray) else torch.as_tensor(x)
        return t.to(dtype) if dtype is not None else t

    d = Data()
    if payload.get("pos") is not None:
        d.pos = _to_torch(payload["pos"], torch.float32)
    if payload.get("x") is not None:
        d.x = _to_torch(payload["x"], torch.float32)
    if payload.get("adj_edge_index") is not None:
        d.adj_edge_index = _to_torch(payload["adj_edge_index"], torch.long)
    if payload.get("adj_edge_weight") is not None:
        d.adj_edge_weight = _to_torch(payload["adj_edge_weight"], torch.float32)
    if payload.get("dist_edge_index") is not None:
        d.dist_edge_index = _to_torch(payload["dist_edge_index"], torch.long)
    if payload.get("dist_edge_weight") is not None:
        d.dist_edge_weight = _to_torch(payload["dist_edge_weight"], torch.float32)
    if payload.get("global_features") is not None:
        d.global_features = _to_torch(payload["global_features"], torch.float32)
    d.num_nodes = int(payload.get("num_nodes", d.pos.shape[0]))
    center_local = int(payload["center"])
    d.center_atom_index = torch.tensor([center_local], dtype=torch.long)
    d.center_mask = torch.nn.functional.one_hot(torch.tensor(center_local), num_classes=d.num_nodes).bool()
    d.center_idx = torch.tensor([center_local], dtype=torch.long)
    d._global_raw = payload.get("global_raw")
    d._dist_full = payload.get("dist_full")
    return d


def run_xtb_for_mol(rd_mol: Chem.Mol, coordsA: np.ndarray, xtb_path: str, total_charge: int = 0, method: str = "2") -> np.ndarray:
    n = rd_mol.GetNumAtoms()
    charges = np.zeros(n, dtype=np.float32)
    if xtb_path is None:
        return charges

    tmpdir = tempfile.mkdtemp(prefix="xtb_")
    try:
        xyz_path = os.path.join(tmpdir, "mol.xyz")
        with open(xyz_path, "w", encoding="utf-8") as fh:
            fh.write(f"{n}\n")
            fh.write("generated by whole_gpu.py\n")
            for atom, coord in zip(rd_mol.GetAtoms(), coordsA):
                fh.write(f"{atom.GetSymbol()} {coord[0]:.8f} {coord[1]:.8f} {coord[2]:.8f}\n")

        cmd = [xtb_path, xyz_path, "--gfn", str(method), "--chrg", str(total_charge), "--uhf", "0"]
        proc = subprocess.run(cmd, cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            print(f"[WARN] xTB failed with code {proc.returncode}: {proc.stderr.decode(errors='ignore')}")
            return charges

        charge_file = os.path.join(tmpdir, "charges")
        if os.path.exists(charge_file):
            try:
                arr = np.loadtxt(charge_file, dtype=float)
                if arr.ndim == 1:
                    parsed = arr
                elif arr.ndim == 2 and arr.shape[1] >= 2:
                    parsed = arr[:, 1]
                else:
                    parsed = arr.reshape(-1)
                if parsed.shape[0] == n:
                    charges = parsed.astype(np.float32)
                else:
                    print(f"[WARN] xTB charges length mismatch (got {parsed.shape[0]}, expect {n}); falling back to zeros.")
            except Exception as exc:
                print(f"[WARN] Failed to parse xTB charges: {exc}")
        else:
            print("[WARN] xTB charge file not found; using zeros.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return charges


def compute_coulomb_force_kbt_per_nm(charges: np.ndarray, coordsA: np.ndarray, center_local: int, temperature: float) -> np.ndarray:
    charges = np.asarray(charges, dtype=np.float64)
    coords_nm = np.asarray(coordsA, dtype=np.float64) * 0.1
    n = charges.shape[0]
    if n != coords_nm.shape[0] or n == 0:
        return np.zeros(3, dtype=np.float32)

    q_center = charges[center_local]
    center_pos_nm = coords_nm[center_local]

    diff_nm = coords_nm - center_pos_nm
    mask = np.ones(n, dtype=bool)
    mask[center_local] = False
    diff_nm = diff_nm[mask]
    other_q = charges[mask]

    norm_nm = np.linalg.norm(diff_nm, axis=1)
    safe_norm = np.where(norm_nm < 1e-12, 1e-12, norm_nm)

    diff_m = diff_nm * 1e-9
    norm_m = safe_norm * 1e-9

    ke = 8.9875517923e9
    e_charge = 1.602176634e-19
    k_B = 1.380649e-23

    factors = ke * (q_center * other_q) * (e_charge ** 2)
    scaled = factors[:, None] * diff_m / (norm_m[:, None] ** 3)
    force_N = scaled.sum(axis=0)
    force_kbt_per_nm = force_N * 1e9 / (k_B * float(temperature))
    return force_kbt_per_nm.astype(np.float32)


def worker_build_payload_and_xtb(args):
    (
        gidx,
        mol_block,
        center_local,
        temperature,
        xtb_path,
        xtb_total_charge,
        xtb_method,
    ) = args

    rd_mol = Chem.MolFromMolBlock(mol_block, sanitize=False, removeHs=False)
    if rd_mol is None:
        raise ValueError("MolFromMolBlock failed in worker.")
    conf = rd_mol.GetConformer()
    posA = np.asarray(conf.GetPositions(), dtype=np.float32)
    data = build_data(rd_mol, posA, int(center_local))

    def _to_np(x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    payload = {
        "pos": _to_np(getattr(data, "pos", None)),
        "x": _to_np(getattr(data, "x", None)),
        "adj_edge_index": _to_np(getattr(data, "adj_edge_index", None)),
        "adj_edge_weight": _to_np(getattr(data, "adj_edge_weight", None)),
        "dist_edge_index": _to_np(getattr(data, "dist_edge_index", None)),
        "dist_edge_weight": _to_np(getattr(data, "dist_edge_weight", None)),
        "global_features": _to_np(getattr(data, "global_features", None)),
        "global_raw": getattr(data, "_global_raw", None),
        "dist_full": getattr(data, "_dist_full", None),
        "center": int(center_local),
        "num_nodes": int(getattr(data, "num_nodes", posA.shape[0])),
    }

    charges = np.zeros(rd_mol.GetNumAtoms(), dtype=np.float32)
    coulomb = np.zeros(3, dtype=np.float32)
    if xtb_path is not None:
        try:
            # 仍然跑 xTB 拿电荷
            charges = run_xtb_for_mol(
                rd_mol, posA, xtb_path,
                total_charge=xtb_total_charge,
                method=xtb_method,
            )
            # 不再计算库仑力，coulomb 继续保持为 0
        except Exception as exc:
            print(f"[WARN] xTB pipeline failed: {exc}. Using zero charges and Coulomb force.")

    return int(gidx), payload, charges, coulomb


# --------------------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------------------

def run_inference(
    gro_dir: str,
    ckpt_map: Dict[str, str],
    itp_path: Optional[str],
    cut_workers: int,
    temperature: float,
    centers: str,
    radius_A: float,
    water_radius_A: Optional[float],
    max_slices: Optional[int],
    center_idx_range: Optional[Tuple[int, int]],
    xtb_path: Optional[str] = None,
    xtb_total_charge: int = 0,
    xtb_method: str = "2",
) -> List[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    single_ckpt = next(iter(ckpt_map.values())) if ckpt_map else None
    if not single_ckpt:
        raise RuntimeError("Need a checkpoint path via --ckpt_map")
    model = load_packaged_model(single_ckpt, device)
    try:
        torch.set_float32_matmul_precision("high")
        model = torch.compile(model, dynamic=True, mode="max-autotune")
    except Exception:
        pass

    results = []
    gro_files = [f for f in os.listdir(gro_dir) if f.endswith(".gro")]
    gro_files.sort()
    # 解析 itp，拿到 atoms_info / bonds_info，供补氢用
    atoms_info = bonds_info = None
    if itp_path:
        try:
            atoms_info, bonds_info = parse_itp_file(itp_path)
        except Exception as e:
            print(f"[WARN] Failed to parse ITP '{itp_path}': {e}. Fallback to simple cutoff.")
            atoms_info = bonds_info = None

    bs = 16
    use_amp = device.type == "cuda"
    use_workers = bool(cut_workers and cut_workers > 1)
    use_xtb = use_workers and xtb_path is not None
    executor = ProcessPoolExecutor(max_workers=cut_workers) if use_workers else None

    meta_map: Dict[int, dict] = {}
    pending = set()
    ready_buffer: List[Tuple[int, Data, np.ndarray, np.ndarray]] = []
    gidx_counter = 0
    stop_submission = False

    def handle_future(fut):
        if fut.cancelled():
            return
        try:
            gi, payload, charges_arr, coulomb_force = fut.result()
        except Exception as exc:
            print(f"[WARN] Worker failed: {exc}")
            return
        data_obj = _payload_to_pyg(payload)
        ready_buffer.append((gi, data_obj, charges_arr, coulomb_force))

    def flush_ready(force: bool = False):
        nonlocal results
        while ready_buffer and (force or len(ready_buffer) >= bs):
            chunk = ready_buffer[:bs]
            del ready_buffer[:bs]
            batch = Batch.from_data_list([itm[1] for itm in chunk])
            if hasattr(batch, "pin_memory"):
                batch = batch.pin_memory()
            batch = batch.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                _, forces = model.energy_and_forces(batch, center_only=True, create_graph=False)
            forces_np = forces.detach().cpu().numpy()

            for idx_in_batch, (gi, data_item, charges_arr, coulomb_force) in enumerate(chunk):
                if max_slices is not None and len(results) >= max_slices:
                    break
                meta_item = meta_map.get(gi, {})
                raw_gf = data_item._global_raw
                adj, dist_inv, dist_inv_norm = _build_matrices_from_data(data_item)
                center_idx0 = int(meta_item.get("center_global", -1))
                center_idx1 = center_idx0 + 1 if center_idx0 >= 0 else -1
                center_local = int(data_item.center_atom_index.item())
                coordsA = meta_item.get("coordsA")
                atom_types = meta_item.get("atom_types", [])
                coords_nm = coordsA * 0.1 if coordsA is not None else np.zeros((len(atom_types), 3), dtype=float)
                center_coord = coords_nm[center_local].tolist() if coords_nm.size else [0.0, 0.0, 0.0]
                center_atom_type = atom_types[center_local] if atom_types else "C"
                base_name = os.path.splitext(meta_item.get("gro_file", ""))[0]
                gro_name = f"{base_name}_center{center_idx1}.gro" if base_name else meta_item.get("gro_file", "")

                model_force = forces_np[idx_in_batch]
                coulomb_vec = coulomb_force if use_xtb else np.zeros(3, dtype=np.float32)
                total_force = model_force + coulomb_vec

                charges_for_json = charges_arr if use_xtb else meta_item.get("charges", np.zeros_like(charges_arr))

                result = {
                    "gro_file": gro_name,
                    "atom_types": atom_types,
                    "coordinates": coords_nm.tolist(),
                    "formal_charges": charges_for_json.tolist() if hasattr(charges_for_json, "tolist") else charges_for_json,
                    "adj_matrix": adj.tolist(),
                    "dist_inv_matrix_normalized": dist_inv_norm.tolist(),
                    "dist_inv_matrix": dist_inv.tolist(),
                    "atom_features": data_item.x.cpu().numpy().tolist(),
                    "center_force_kBT_per_nm": total_force.tolist(),
                    "model_force_kBT_per_nm": model_force.tolist(),
                    "coulomb_force_kBT_per_nm": coulomb_vec.tolist(),
                    "force_unit": "kBT/nm",
                    "charge_unit": "e",
                    "center_atom": {
                        "atom_index": center_idx1,
                        "center_idx": center_idx0,
                        "local_index": center_local,
                        "atom_type": center_atom_type,
                        "x": center_coord[0],
                        "y": center_coord[1],
                        "z": center_coord[2],
                        "element": _as_element(center_atom_type),
                    },
                    "global_features": raw_gf,
                    "normalizes_global_features": data_item.global_features.cpu().numpy().tolist(),
                }
                results.append(result)
            if max_slices is not None and len(results) >= max_slices:
                break

    # 遍历所有 .gro 文件，外面套一层进度条
    for gf in tqdm(gro_files, desc="Processing GRO files"):
        if max_slices is not None and len(results) >= max_slices:
            break
        u = mda.Universe(os.path.join(gro_dir, gf), convert_units=True)
        center_candidates = _select_centers(u, centers)
        center_candidates = [i for i in center_candidates if _within_range(i, center_idx_range)]

        for cidx in tqdm(center_candidates, desc=f"Slicing centers in {gf}", leave=False):
            if max_slices is not None and len(results) >= max_slices:
                stop_submission = True
                break
            if max_slices is not None and gidx_counter >= max_slices:
                stop_submission = True
                break

            prep = _prepare_slice(
                u,
                cidx,
                radius_A,
                water_radius_A,
                atoms_info=atoms_info,
                bonds_info=bonds_info,
            )

            if prep is None:
                continue
            rd_mol, coordsA, center_local, atom_types, charges = prep
            mol_block = Chem.MolToMolBlock(rd_mol)

            meta_map[gidx_counter] = {
                "coordsA": coordsA,
                "atom_types": atom_types,
                "charges": charges,
                "center_global": cidx,
                "gro_file": gf,
            }

            if use_workers:
                fut = executor.submit(
                    worker_build_payload_and_xtb,
                    (
                        gidx_counter,
                        mol_block,
                        center_local,
                        temperature,
                        xtb_path if use_xtb else None,
                        xtb_total_charge,
                        xtb_method,
                    ),
                )
                pending.add(fut)
                done, pending = wait(pending, timeout=0, return_when=FIRST_COMPLETED)
                for fu in done:
                    handle_future(fu)
                    flush_ready()
                if max_slices is not None and len(results) >= max_slices:
                    stop_submission = True
                    break
            else:
                gi, payload, charges_arr, coulomb_force = worker_build_payload_and_xtb(
                    (
                        gidx_counter,
                        mol_block,
                        center_local,
                        temperature,
                        None,
                        xtb_total_charge,
                        xtb_method,
                    )
                )
                ready_buffer.append((gi, _payload_to_pyg(payload), charges_arr, coulomb_force))
                flush_ready()

            gidx_counter += 1

        if stop_submission:
            break

    if use_workers:
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fu in done:
                handle_future(fu)
            flush_ready()
        if executor:
            executor.shutdown(wait=True, cancel_futures=False)

    flush_ready(force=True)
    return results


def parse_ckpt_map(val: str) -> Dict[str, str]:
    if os.path.isfile(val):
        with open(val, "r") as f:
            return json.load(f)
    mp = {}
    for kv in val.split(";"):
        if not kv:
            continue
        k, v = kv.split("=", 1)
        mp[k.strip()] = v.strip()
    return mp


def parse_center_range(val: Optional[str]) -> Optional[Tuple[int, int]]:
    if val is None:
        return None
    if ":" not in val:
        raise ValueError("center-index-range must be start:end")
    s, e = val.split(":", 1)
    return int(), int(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gro-dir", required=True, help="Directory containing .gro files")
    ap.add_argument("--ckpt_map", default="C=cho.pt;H=cho.pt;O=cho.pt", help="Mapping like C=ckptC.pth;H=ckptH.pth;O=ckptO.pth or a JSON file path")
    ap.add_argument("--itp", type=str, default=None, help="Topology .itp for cut")
    ap.add_argument("--cut-workers", type=int, default=0, help="Processes for slicing/graph building")
    ap.add_argument("--T", type=float, default=298.15)
    ap.add_argument("--centers", type=str, default="non-water")
    ap.add_argument("--radius", type=float, default=8.0)
    ap.add_argument("--water_radius", type=float, default=None)
    ap.add_argument("--out-json", type=str, default="predicted_molecules_with_forces.json")
    ap.add_argument("--max-slices", type=int, default=None, help="Limit total slices across all gro files")
    ap.add_argument("--center-index-range", type=str, default=None, help="1-based inclusive range start:end")
    ap.add_argument("--xtb-path", type=str, default=None, help="Path to xTB binary (enables xTB when set and cut_workers>1)")
    ap.add_argument("--xtb-total-charge", type=int, default=0, help="Total charge passed to xTB")
    ap.add_argument("--xtb-method", type=str, default="2", help="xTB GFN level (default: 2)")
    args = ap.parse_args()

    ckpt_map = parse_ckpt_map(args.ckpt_map)
    center_range = parse_center_range(args.center_index_range)

    results = run_inference(
        gro_dir=args.gro_dir,
        ckpt_map=ckpt_map,
        itp_path=args.itp,
        cut_workers=args.cut_workers,
        temperature=args.T,
        centers=args.centers,
        radius_A=args.radius,
        water_radius_A=args.water_radius,
        max_slices=args.max_slices,
        center_idx_range=center_range,
        xtb_path=args.xtb_path,
        xtb_total_charge=args.xtb_total_charge,
        xtb_method=args.xtb_method,
    )

    results_builtin = _to_builtin(results)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(results_builtin, fh, indent=2)


if __name__ == "__main__":
    main()
