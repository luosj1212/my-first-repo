
import os, sys, math, json, re, tempfile, io, time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
# 全局特征归一化范围（与 integrated_force 保持一致）
from integrated_force import GF_min, GF_max

import torch
from torch_geometric.data import Data, Batch

import MDAnalysis as mda
from rdkit import Chem
from rdkit.Chem import rdmolfiles

from ase import units
from ase import Atoms as ASEAtoms
from ase.neighborlist import NeighborList
from ase.calculators.calculator import Calculator, all_changes
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import func as FF   # 你上传的 func.py
from ase.io.trajectory import Trajectory
from ase.io import write
from concurrent.futures import ProcessPoolExecutor, as_completed

# from ase_sidecar_writer import save_sidecars_for_slice

# --- your model class ---
from model_function import DualGCNWithAttention

# --- optional user feature module (if available) ---
# _HAS_FUNC = False
# try:
#     import func
#     _HAS_FUNC = True
# except Exception:
#     _HAS_FUNC = False

# --- import your cutter (use cut_v8.py, which supports ITP) ---
import importlib.util
_CUT_PATH = os.path.join(os.path.dirname(__file__), 'cut_v8.py')
ONEHOT_RULE = {"H":[1,0,0], "C":[0,1,0], "O":[0,0,1]}

if not os.path.exists(_CUT_PATH):
    # fallback: keep compat with old filename
    _CUT_PATH = os.path.join(os.path.dirname(__file__), 'cut_v7_noitp.py')
    if not os.path.exists(_CUT_PATH):
        raise FileNotFoundError(f"Neither cut_v8.py nor cut_v7_noitp.py found near {__file__}")

spec = importlib.util.spec_from_file_location("cutitp", _CUT_PATH)
cutitp = importlib.util.module_from_spec(spec)
sys.modules["cutitp"] = cutitp
spec.loader.exec_module(cutitp)

from ase import units as _units
# === PATCH 1A: build_one_subgraph_proc → 返回 global_features，而不是 u ===
def build_one_subgraph_proc(args):
    gidx, mol_block, center_local = args
    rd_mol = Chem.MolFromMolBlock(mol_block, sanitize=False, removeHs=False)
    if rd_mol is None:
        raise ValueError("MolFromMolBlock failed in worker.")
    conf = rd_mol.GetConformer()
    posA = np.asarray(conf.GetPositions(), dtype=np.float32)  # Å
    center_local = int(center_local)

    data = build_data(rd_mol, posA, center_local)  # 训练期一致的构图

    def _to_np(x):
        if x is None: return None
        if isinstance(x, np.ndarray): return x
        if torch.is_tensor(x): return x.detach().cpu().numpy()
        return np.asarray(x)

    payload = {
        "pos":        _to_np(getattr(data, "pos", None)),             # (N,3) float32 (这里是 nm，见 build_data)
        "z":          _to_np(getattr(data, "z", None)),               # 如需的话
        "x":          _to_np(getattr(data, "x", None)),
        "adj_edge_index":  _to_np(getattr(data, "adj_edge_index", None)),
        "adj_edge_weight": _to_np(getattr(data, "adj_edge_weight", None)),
        "dist_edge_index": _to_np(getattr(data, "dist_edge_index", None)),
        "dist_edge_weight": _to_np(getattr(data, "dist_edge_weight", None)),
        # ⭐ 与训练一致的全局特征键名：
        "global_features": _to_np(getattr(data, "global_features", None)),
        "center":     center_local,
        "center_mask": _to_np(getattr(data, "center_mask", None)),
        "num_nodes":  int(getattr(data, "num_nodes", posA.shape[0] if posA is not None else 0)),
    }
    return (int(gidx), payload)
# === END PATCH 1A ===


# === PATCH 1B: payload_to_pyg → 复原 global_features ===
def payload_to_pyg(payload):
    from torch_geometric.data import Data
    def _to_torch(x, dtype=None):
        if x is None: return None
        t = torch.from_numpy(x) if isinstance(x, np.ndarray) else torch.as_tensor(x)
        return t.to(dtype) if dtype is not None else t

    d = Data()
    if payload.get("pos") is not None:
        d.pos = _to_torch(payload["pos"], torch.float32)  # (N,3), 单位 nm（与 build_data 一致）
    if payload.get("z") is not None:
        d.z = _to_torch(payload["z"], torch.long)
    if payload.get("x") is not None:
        d.x = _to_torch(payload["x"], torch.float32)

    if payload.get("adj_edge_index") is not None:
        d.adj_edge_index  = _to_torch(payload["adj_edge_index"], torch.long)
    if payload.get("adj_edge_weight") is not None:
        d.adj_edge_weight = _to_torch(payload["adj_edge_weight"], torch.float32)
    if payload.get("dist_edge_index") is not None:
        d.dist_edge_index = _to_torch(payload["dist_edge_index"], torch.long)
    if payload.get("dist_edge_weight") is not None:
        d.dist_edge_weight = _to_torch(payload["dist_edge_weight"], torch.float32)

    # ⭐ 训练期同名：
    if payload.get("global_features") is not None:
        d.global_features = _to_torch(payload["global_features"], torch.float32)

    num_nodes = int(payload.get("num_nodes", d.pos.shape[0] if hasattr(d, "pos") else 0))
    d.num_nodes = num_nodes

    center_local = int(payload["center"])
    d.center_atom_index = torch.tensor([center_local], dtype=torch.long)

    cm_np = payload.get("center_mask", None)
    if cm_np is not None:
        d.center_mask = _to_torch(cm_np, torch.bool).view(-1)
    else:
        cm = torch.zeros((num_nodes,), dtype=torch.bool)
        if 0 <= center_local < max(num_nodes, 1):
            cm[center_local] = True
        d.center_mask = cm

    d.center_idx = torch.tensor([center_local], dtype=torch.long)
    return d
# === END PATCH 1B ===


def kBT_in_eV(T: float) -> float:
    return float(_units.kB) * float(T)
def save_sidecars_for_slice(
    out_stem: str,
    rd_mol: Chem.Mol,
    coordsA: np.ndarray,               # Å
    center_local: int,
    bond_factor: float = 1.2,
):
    """
    生成 3 个旁路文件（与 xyzforce 的 sidecar 读取严格对齐）：
      1) {out_stem}.xyz        : 原子名，中心原子名末尾加 '0'
      2) {out_stem}.bonds.txt  : i j（0-based），几何阈值成键（COVALENT_RADII × bond_factor）
      3) {out_stem}.feats.npz  : center_local, node_feat(3维), global_features
    """

    # --- 节点特征：严格使用 func.generate_node_features（与训练完全一致） ---
    x_nodes = _node_features_onehot(rd_mol)
    x_nodes = x_nodes.detach().cpu().numpy() if torch.is_tensor(x_nodes) else np.asarray(x_nodes, dtype=np.float32)
    assert x_nodes.ndim == 2 and x_nodes.shape[1] == 3, f"node_feat should be (N,3), got {x_nodes.shape}"

    # --- 几何阈值成键（单位 nm） ---
    pos_nm = np.asarray(coordsA, dtype=np.float64) * 0.1
    n = pos_nm.shape[0]
    radii = np.array([FF.COVALENT_RADII.get(a.GetSymbol(), 0.1) for a in rd_mol.GetAtoms()], dtype=np.float64)
    rsum = radii[:, None] + radii[None, :]
    diff = pos_nm[:, None, :] - pos_nm[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    bonded = (dist <= (bond_factor * rsum)) & (dist > 1e-8)
    ui, uj = np.where(np.triu(bonded, k=1))
    bonds = list(zip(ui.tolist(), uj.tolist()))  # 0-based (i<j)

    # --- 全局特征：与 integrated_force.GF_min/GF_max 归一一致 ---
    gf = _global_features(coordsA, rd_mol, int(center_local)).astype(np.float32)

    # --- 写 xyz（中心原子名末尾加 '0'） ---
    xyz_path = f"{out_stem}.xyz"
    with open(xyz_path, "w", encoding="utf-8") as fx:
        fx.write(f"{n}\n")
        fx.write(f"center_local={int(center_local)}\n")
        for idx, a in enumerate(rd_mol.GetAtoms()):
            sym = a.GetSymbol()
            if idx == int(center_local):
                sym = f"{sym}0"
            x, y, z = coordsA[idx]
            fx.write(f"{sym} {x:.6f} {y:.6f} {z:.6f}\n")

    # --- 写 bonds.txt（0-based） ---
    with open(f"{out_stem}.bonds.txt", "w", encoding="utf-8") as fb:
        fb.write("# i j (0-based)\n")
        for i, j in bonds:
            fb.write(f"{i} {j}\n")

    # --- 写逆距离矩阵边列表（单位 nm^-1） ---
    # 与 _adj_and_dist 中一致：对所有 i != j 的成对边导出  i j invd_nm^{-1}
    diff_nm = pos_nm[:, None, :] - pos_nm[None, :, :]
    dist_nm = np.linalg.norm(diff_nm, axis=-1)  # (n,n)
    rows, cols = np.where(~np.eye(n, dtype=bool))
    invd = 1.0 / (dist_nm + 1e-12)
    with open(f"{out_stem}.invdist.txt", "w", encoding="utf-8") as fd:
        fd.write("# i j invd_nm^-1\n")
        for i, j in zip(rows.tolist(), cols.tolist()):
            fd.write(f"{i} {j} {invd[i, j]:.8f}\n")

    # --- 写 feats.npz ---
    np.savez_compressed(
        f"{out_stem}.feats.npz",
        center_local=np.int64(center_local),
        node_feat=np.asarray(x_nodes, dtype=np.float32),
        global_features=np.asarray(gf, dtype=np.float32),
    )

class _SimpleTimer:
    def __init__(self):
        self._t = time.perf_counter()
    def lap(self):
        t1 = time.perf_counter()
        dt = t1 - self._t
        self._t = t1
        return dt
    
def _node_features_onehot(rd_mol: Chem.Mol) -> np.ndarray:
    """
    将 RDKit 原子序列转为 one-hot 节点特征，维度 = len(ONEHOT_RULE)。
    只支持 ONEHOT_RULE 中声明过的元素（与训练数据一致）。
    """
    K = len(ONEHOT_RULE)
    x = np.zeros((rd_mol.GetNumAtoms(), K), dtype=np.float32)
    for i, a in enumerate(rd_mol.GetAtoms()):
        sym = a.GetSymbol().upper()  # 训练端 add_onehot.py 里是全大写比较
        if sym not in ONEHOT_RULE:
            # 和你的 add_onehot.py 一致：只支持规则里声明的元素
            raise ValueError(f"[onehot] Unsupported element '{sym}'. Add it to ONEHOT_RULE first.")
        x[i] = np.asarray(ONEHOT_RULE[sym], dtype=np.float32)
    return x

# ---------- simple fallbacks (replace with your training-time feature pipeline) ----------
_ELEMS = ["H","C","N","O","S","P","F","Cl","Br","I","Si","B"]
_E2IDX = {e:i for i,e in enumerate(_ELEMS)}
def as_float_tensor(x):
    """安全把输入变成 float32 tensor；若已是 tensor 就 clone().detach().float()。"""
    if torch.is_tensor(x):
        return x.clone().detach().to(torch.float32)
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)

def _rdmol_to_xyz_bytes(mol, decimals: int = 3) -> bytes:
    """把 RDKit mol 转成 XYZ 字节串；坐标四舍五入到指定小数位，提高缓存命中率。"""
    conf = mol.GetConformer()
    pos = conf.GetPositions()  # ndarray (N,3) in Å
    posq = np.round(pos, decimals=decimals)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    buf = io.StringIO()
    n = len(symbols)
    buf.write(f"{n}\n\n")
    for s, (x, y, z) in zip(symbols, posq):
        buf.write(f"{s} {x:.{decimals}f} {y:.{decimals}f} {z:.{decimals}f}\n")
    return buf.getvalue().encode("utf-8")

def _onehot_elem(symbol: str, dim: int=len(_ELEMS)):
    v = np.zeros(dim, dtype=np.float32)
    v[_E2IDX.get(symbol, 0)] = 1.0
    return v
def _rdmol_to_xyz(mol, fname):
    conf = mol.GetConformer()
    pos = conf.GetPositions()  # ndarray (N,3) in Å
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    with open(fname, "w") as f:
        f.write(f"{len(atoms)}\n\n")
        for s, (x, y, z) in zip(atoms, pos):
            f.write(f"{s} {x:.8f} {y:.8f} {z:.8f}\n")

def _basic_node_features(rd_mol: Chem.Mol) -> np.ndarray:
    feats=[]
    for a in rd_mol.GetAtoms():
        vec = _onehot_elem(a.GetSymbol()).tolist()
        vec += [a.GetAtomicNum()/100.0, a.GetDegree()/4.0, float(a.GetIsAromatic())]
        feats.append(vec)
    return np.array(feats, dtype=np.float32)
def _adj_and_dist(rd_mol: Chem.Mol, coordsA: np.ndarray, bond_factor: float = 1.2):
    """
    成键推断：使用 func.COVALENT_RADII 的几何阈值法：
        bonded(i,j) ↔ ||ri-rj||_nm <= bond_factor * (Rcov[i] + Rcov[j])
    距离边：不做 cutoff，构造完整 i≠j 的有向边集；权重为“归一化后的逆距离”
        w_ij = minmax( 1 / d_ij[nm] ) ∈ [0,1]（对本分子内所有非零逆距离做 min-max）
    返回：
        adj_ei:(2,Ea)  adj_w:(Ea,)
        dist_ei:(2,Ed) dist_w:(Ed,)
    注意：coordsA 传入为 Å，本函数内部统一换算到 nm 与半径单位一致
    """
    # import numpy as np

    # 位置：Å→nm
    pos_nm = np.asarray(coordsA, dtype=np.float64) * 0.1
    n = pos_nm.shape[0]

    # pairwise distance (nm)
    diff = pos_nm[:, None, :] - pos_nm[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)  # (n,n)

    # 共价半径（nm），与 func.COVALENT_RADII 对齐
    radii = np.array([FF.COVALENT_RADII.get(a.GetSymbol(), 0.1) for a in rd_mol.GetAtoms()],
                      dtype=np.float64)
    rsum = radii[:, None] + radii[None, :]

    # 成键掩码（去掉自连）
    bonded = (dist <= (bond_factor * rsum)) & (dist > 1e-8)

    # 无向成键对 i<j → 双向边
    ui, uj = np.where(np.triu(bonded, k=1))
    if ui.size > 0:
        adj_pairs = np.stack([np.concatenate([ui, uj]), np.concatenate([uj, ui])], axis=0)
        adj_ei = adj_pairs.astype(np.int64)
        adj_w  = np.ones(adj_ei.shape[1], dtype=np.float32)
    else:
        adj_ei = np.zeros((2, 0), dtype=np.int64)
        adj_w  = np.zeros((0,), dtype=np.float32)

    # 距离边：完整 i≠j（无截断）
    rows, cols = np.where(~np.eye(n, dtype=bool))
    dist_ei = np.stack([rows, cols], axis=0).astype(np.int64)

    invd = 1.0 / (dist[rows, cols] + 1e-12)  # nm^{-1}
    # nz = invd[invd > 0]
    # if nz.size > 0:
    #     mn, mx = float(nz.min()), float(nz.max())
    #     dist_w = ((invd - mn) / (mx - mn)) if mx > mn else np.zeros_like(invd)
    # else:
    #     dist_w = np.zeros_like(invd)
    dist_w = invd.astype(np.float32)

    return adj_ei, adj_w, dist_ei, dist_w


# -------------- global features (min-max normalized, same bins as your JSON pipeline) --------------
def _global_features(coordsA: np.ndarray, rd_mol: Chem.Mol, center_local: int):
    coords_nm = coordsA*0.1
    masses = np.array([a.GetMass() for a in rd_mol.GetAtoms()], dtype=np.float64)
    cm = (coords_nm*masses[:,None]).sum(axis=0)/(masses.sum()+1e-12)
    rel = coords_nm-cm
    Ixx = np.sum(masses*(rel[:,1]**2+rel[:,2]**2))
    Iyy = np.sum(masses*(rel[:,0]**2+rel[:,2]**2))
    Izz = np.sum(masses*(rel[:,0]**2+rel[:,1]**2))
    Ixy = np.sum(masses*rel[:,0]*rel[:,1])
    Ixz = np.sum(masses*rel[:,0]*rel[:,2])
    Iyz = np.sum(masses*rel[:,1]*rel[:,2])
    inertia = np.array([[ Ixx, -Ixy, -Ixz],[-Ixy, Iyy, -Iyz],[-Ixz,-Iyz, Izz]], dtype=np.float64)
    eig = np.linalg.eigvalsh(inertia); I1,I2,I3 = np.sort(eig)
    NPR1=float(I1/(I3+1e-12)); NPR2=float(I2/(I3+1e-12))
    rg_frame=float(np.sqrt(((coords_nm-coords_nm.mean(axis=0))**2).sum(axis=1).mean()))
    rg_mol  =float(np.sqrt(((coords_nm-coords_nm.mean(axis=0))**2).sum(axis=1).mean()))
    # densities
    R_max=1.0; dr=0.05; edges=np.arange(0.0,R_max+dr,dr)
    c=coords_nm[center_local]; d=np.linalg.norm(coords_nm-c,axis=1)
    dens=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(d>=lo)&(d<hi)
        vol=(4.0/3.0)*math.pi*(hi**3-lo**3); m=masses[mask].sum()
        dens.append(float(m/vol) if vol>0 else 0.0)
    raw=[float(masses.sum()), int(np.array([a.GetSymbol()!='H' for a in rd_mol.GetAtoms()]).sum()), int(rd_mol.GetNumBonds()),
         float(I1),float(I2),float(I3),NPR1,NPR2,rg_frame,rg_mol]+dens
    norm=[ (raw[i]-GF_min[i])/(GF_max[i]-GF_min[i]) if GF_max[i]>GF_min[i] else 0.0 for i in range(len(raw)) ]
    # print("len(raw)=", len(raw), "len(GF_min)=", len(GF_min), "len(GF_max)=", len(GF_max))

    return np.array(norm,dtype=np.float32)

# -------------- dataset builders --------------
def build_data(rd_mol: Chem.Mol, coordsA: np.ndarray, center_local: int, x_override=None):
    """
    与数据集/训练端对齐：
      - 节点特征：优先用 _node_features_onehot → (N,3)
      - 成键/距离图：几何阈值成键 + 无截断全连接距离边（见 _adj_and_dist）
      - 坐标：pos 用 nm（Å×0.1）
      - 全局特征：保持你现有实现与 GF_min/GF_max 归一化一致
    """
    # 1) 节点特征
    if x_override is None:
        try:
            x = _node_features_onehot(rd_mol)
            x = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x, dtype=np.float32)
        except Exception:
            # 兜底：仍可退回你已有的 basic 特征（如果保留了该函数）
            x = _basic_node_features(rd_mol)
    else:
        x = x_override.detach().cpu().numpy() if torch.is_tensor(x_override) else np.asarray(x_override)
    assert x.ndim == 2 and x.shape[1] == 3, f"expected (N,3) node feats, got {tuple(x.shape)}"

    # 2) 边（与数据集/训练端一致）
    adj_ei, adj_w, dist_ei, dist_w = _adj_and_dist(rd_mol, coordsA, bond_factor=1.2)

    # 3) 组 Data
    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        pos=torch.tensor(coordsA * 0.1, dtype=torch.float32),  # Å→nm
        center_mask=torch.nn.functional.one_hot(
            torch.tensor(center_local), num_classes=x.shape[0]
        ).bool(),
        adj_edge_index=torch.tensor(adj_ei, dtype=torch.long),
        adj_edge_weight=torch.tensor(adj_w, dtype=torch.float32),
        dist_edge_index=torch.tensor(dist_ei, dtype=torch.long),
        dist_edge_weight=torch.tensor(dist_w, dtype=torch.float32),
        global_features=torch.tensor(_global_features(coordsA, rd_mol, center_local), dtype=torch.float32),
    )
    return data



# -------------- model loading --------------
def load_packaged_model(pkg_path: str, device: torch.device) -> torch.nn.Module:
    obj=torch.load(pkg_path, map_location=device)
    if isinstance(obj, torch.nn.Module):
        return obj.to(device).eval()
    if isinstance(obj, dict) and "arch" in obj and "state_dict" in obj:
        arch = obj["arch"]
        m = DualGCNWithAttention(**arch).to(device)
        m.load_state_dict(obj["state_dict"], strict=True)
        m.eval()
        return m
    # raw state_dict fallback with heuristic shapes (not recommended)
    sd = obj if isinstance(obj, dict) else None
    if sd is None:
        raise RuntimeError(f"Unsupported checkpoint format: {type(obj)} at {pkg_path}")
    # heuristic infer
    in_w = sd.get("input_proj.weight", None)
    if in_w is None:
        raise RuntimeError("state_dict missing input_proj.weight; cannot infer dims. Use packaged checkpoint.")
    hidden_dim, input_dim = in_w.shape
    arch = dict(input_dim=input_dim, hidden_dim=hidden_dim, num_blocks=3, gat_heads=4, att_depth=1, rbf_K=24, dropout=0.15, global_feat_dim=30, output_dim=1)
    m = DualGCNWithAttention(**arch).to(device)
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m

def _smoothstep01(x: np.ndarray) -> np.ndarray:
    # 0→1 的三次平滑：3x^2 - 2x^3
    return np.where(x<=0, 0.0, np.where(x>=1.0, 1.0, x*x*(3.0 - 2.0*x)))



_BONDLEN_CH = 1.09
_BONDLEN_NH = 1.01
_BONDLEN_OH = 0.97
# 你也可以统一成 1.0~1.1 的范围；这里按元素区分
def _bondlen_XH(sym: str) -> float:
    if sym == 'C': return _BONDLEN_CH
    if sym == 'N': return _BONDLEN_NH
    if sym == 'O': return _BONDLEN_OH
    return 1.05  # 兜底

# 协价半径（Å）+ 安全因子，用于判定重原子-重原子成键（只为构 RDKit 时连键）
_RCOV = {'H':0.31, 'C':0.76, 'N':0.71, 'O':0.66}
_RCOV_SCALE = 1.20

def _infer_heavy_bonds(symbols: List[str], posA: np.ndarray) -> List[Tuple[int,int]]:
    bonds = []
    n = len(symbols)
    for i in range(n):
        if symbols[i]=='H': continue
        ri = _RCOV.get(symbols[i], 0.75)
        for j in range(i+1, n):
            if symbols[j]=='H': continue
            rj = _RCOV.get(symbols[j], 0.75)
            if np.linalg.norm(posA[i]-posA[j]) <= _RCOV_SCALE*(ri+rj):
                bonds.append((i,j))
    return bonds

def _to_rdkit_mol(symbols: List[str], posA: np.ndarray, heavy_bonds: List[Tuple[int,int]], H_parent: List[Tuple[int,int]]) -> Chem.Mol:
    """仅用于保持接口兼容的轻量构建：重原子间键=几何阈值；H 仅连到其 parent。"""
    em = Chem.RWMol()
    idx_map = []
    for s in symbols:
        a = Chem.Atom(s)
        idx_map.append(em.AddAtom(a))
    # 重原子键
    for i,j in heavy_bonds:
        em.AddBond(int(i), int(j), Chem.BondType.SINGLE)
    # 氢键
    for h_idx, parent in H_parent:
        em.AddBond(int(parent), int(h_idx), Chem.BondType.SINGLE)
    mol = em.GetMol()
    conf = Chem.Conformer(len(symbols))
    for i,(x,y,z) in enumerate(posA):
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)
    return mol

# ---------- GPU 几何补氢（sp3-only）核心 ----------
def _gpu_add_h_sp3(positions, symbols, cell, center_local, device):
    """
    positions: (M,3) float32 tensor [Å], on device
    symbols:   list[str] 长度 M（小片段内的符号）
    cell:      (3,3) float32 tensor
    返回：
      new_positions: (M+K,3) tensor
      new_symbols:   list[str]，追加 K 个 'H'
      H_parent_pairs: List[(h_global_idx, parent_local_idx)]
    """
    M = positions.shape[0]
    # 仅在片段内部做“重原子-重原子”邻接（距离阈值由协价半径决定）
    # 1) 计算最小镜像位移矩阵
    # 构网格配对
    src = torch.arange(M, device=device).repeat_interleave(M)
    dst = torch.arange(M, device=device).repeat(M)
    mask = src != dst
    src, dst = src[mask], dst[mask]

    Ra = positions[src]
    Rb = positions[dst]
    cell_inv = torch.inverse(cell)
    dR = Ra - Rb
    frac = dR @ cell_inv.T
    frac = frac - torch.floor(frac + 0.5)  # 最小镜像
    dR_pbc = frac @ cell.T
    dist = torch.norm(dR_pbc, dim=-1)

    # 2) 构重原子-重原子邻接
    rcov = torch.tensor([_RCOV.get(s,0.75) for s in symbols], device=device)
    ri = rcov[src]; rj = rcov[dst]
    cov_ok = (symbols_list:=symbols)  # 仅为了可读性
    heavy_mask = torch.tensor([s!='H' for s in symbols], device=device)
    hh_mask = heavy_mask[src] & heavy_mask[dst]
    bonded = hh_mask & (dist <= _RCOV_SCALE*(ri+rj))
    # 每个节点的重原子邻居索引列表
    deg = torch.zeros(M, device=device, dtype=torch.int32)
    deg.index_add_(0, src[bonded], torch.ones_like(src[bonded], dtype=torch.int32))
    # 3) 计算每个重原子缺氢数（sp3 → 4 度）
    valence = torch.full((M,), 4, device=device, dtype=torch.int32)
    valence[~heavy_mask] = 1  # 氢本身
    deficit = (valence - deg).clamp(min=0)
    deficit[~heavy_mask] = 0  # H 不补 H

    # 4) 准备每个重原子的现有键向量（单位向量）
    neighbors = [[] for _ in range(M)]
    for i,j,ok in zip(src.tolist(), dst.tolist(), bonded.tolist()):
        if ok:
            neighbors[i].append(j)
    unit_dirs = [None]*M
    eps = 1e-8
    for i in range(M):
        if not heavy_mask[i] or len(neighbors[i])==0:
            unit_dirs[i] = []
            continue
        ui = []
        ri = positions[i]
        for j in neighbors[i]:
            # i <- j 的方向
            v = positions[j] - ri
            # 最小镜像一下更稳当
            frac = (v @ cell_inv.T)
            frac = frac - torch.floor(frac + 0.5)
            v = (frac @ cell.T)
            n = torch.linalg.norm(v) + eps
            ui.append(v/n)
        unit_dirs[i] = ui

    # 5) 生成 H 方向（sp3-only，解析几何）
    H_dirs = [None]*M
    for i in range(M):
        if deficit[i]==0: 
            H_dirs[i]=[]
            continue
        ui = unit_dirs[i]
        di = int(deficit[i].item())
        # 没有邻居：放理想四面体的 4 个方向
        if len(ui)==0:
            T = torch.tensor([[ 1, 1, 1],
                              [ 1,-1,-1],
                              [-1, 1,-1],
                              [-1,-1, 1]], dtype=torch.float32, device=device)
            T = T/torch.linalg.norm(T, dim=-1, keepdim=True)  # 4x3
            H_dirs[i] = [T[k] for k in range(4)][:di]
            continue
        # 有 1 个邻居：其反方向是主轴，另外 3 个等角分布
        if len(ui)==1:
            u = ui[0]  # 指向邻居
            axis = -u / (torch.linalg.norm(u)+eps)
            # 任取一正交基
            a = axis
            # 找与 a 不平行的向量
            tmp = torch.tensor([1.0,0.0,0.0], device=device)
            if torch.allclose(torch.abs(torch.dot(tmp,a)), torch.tensor(1.0, device=device), atol=1e-4):
                tmp = torch.tensor([0.0,1.0,0.0], device=device)
            b = tmp - torch.dot(tmp,a)*a
            b = b / (torch.linalg.norm(b)+eps)
            # c = torch.cross(a,b)
            c = torch.linalg.cross(a, b)
            # 三向量在与 a 垂直的圆上相隔 120°
            ang = 109.47*np.pi/180.0
            sin_t = torch.sin(torch.tensor(ang, device=device))
            cos_t = torch.cos(torch.tensor(ang, device=device))
            # 将三方向绕 a 旋转：a*cos + (b*cosφ + c*sinφ)*sin
            dirs=[]
            for phi in [0, 2*np.pi/3, 4*np.pi/3]:
                cp = torch.cos(torch.tensor(phi, device=device)); sp = torch.sin(torch.tensor(phi, device=device))
                d = a*cos_t + (b*cp + c*sp)*sin_t
                dirs.append(d/ (torch.linalg.norm(d)+eps))
            H_dirs[i] = dirs[:di]
            continue
        # 有 2 个邻居：取它们的平分面 + 法向构成两个四面体方向
        if len(ui)==2:
            u1,u2 = ui[0], ui[1]
            b = (u1 + u2); b = b/(torch.linalg.norm(b)+eps)  # 角平分线
            n = torch.linalg.cross(u1, u2); n = n/(torch.linalg.norm(n)+eps) # 法向
            # 109.47° 的分解系数（√(2)/2 与 1/2）
            d1 = -( (np.sqrt(2)/2)*b + 0.5*n )
            d2 = -( (np.sqrt(2)/2)*b - 0.5*n )
            d1 = d1/(torch.linalg.norm(d1)+eps); d2 = d2/(torch.linalg.norm(d2)+eps)
            H_dirs[i] = [d1, d2][:di]
            continue
        # 有 3 个邻居：把 H 放在 -sum(u_i) 的方向
        if len(ui)>=3:
            s = torch.stack(ui[:3], dim=0).sum(0)
            d = -s/(torch.linalg.norm(s)+eps)
            H_dirs[i] = [d][:di]
            continue

    # 6) 生成 H 位置（按元素键长）
    new_pos = [positions]
    new_sym = list(symbols)
    H_parent_pairs = []
    for i in range(M):
        if heavy_mask := (symbols[i] != 'H'):
            for d in H_dirs[i]:
                bl = _bondlen_XH(symbols[i])
                pH = positions[i] + bl * d
                new_pos.append(pH.unsqueeze(0))
                new_sym.append('H')
                H_parent_pairs.append((len(new_sym)-1, i))  # (新增 H 的局部 idx, 其 parent 局部 idx)

    new_positions = torch.cat(new_pos, dim=0)  # (M+K,3)
    return new_positions, new_sym, H_parent_pairs

# ----------------- 邻居版 Slicer（GPU 补氢 sp3） -----------------
class Slicer:
    def __init__(self, gro_path: str, radius_A: float = 8.0, water_radius_A: Optional[float] = None, itp_path: Optional[str] = None, device: str = 'cuda'):
        self.radius_A = float(radius_A)
        self.water_radius_A = float(water_radius_A) if water_radius_A is not None else None
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 用 ASE 读一次（轻量）
        self._ase = r = self._load_ase(gro_path)
        self._symbols = r.get_chemical_symbols()
        self._cell = torch.tensor(r.get_cell()[:], dtype=torch.float32, device=self.device)  # 3x3

    def _load_ase(self, gro_path) -> ASEAtoms:
        # 只读一次；update 时只改 positions
        at = rdmolfiles  # 只是防止静态分析报未用；无实际意义
        from ase.io import read as ase_read
        atoms = ase_read(gro_path)
        atoms.set_pbc(True)
        return atoms

    def update(self, posA: np.ndarray):
        # 同步 ASE positions
        self._ase.set_positions(posA)

    def element_of(self, idx: int) -> str:
        # 从 ASE 符号
        return self._symbols[idx]

    def centers(self, sel: str) -> List[int]:
        if sel == "all":
            return list(range(len(self._ase)))
        if sel == "non-water":
            # GRO 里水常见为 SOL
            return [i for i,a in enumerate(self._ase) if (a.tag is None and a.symbol!='H' and (a.info.get('resname','')!='SOL')) or (getattr(a, 'resname', '')!='SOL')]
        # 简易：不做复杂选择表达式，必要时可加 MDAnalysis
        return list(range(len(self._ase)))

    def _subidx_by_neighborlist(self, center_idx: int) -> List[int]:
        R = float(self.radius_A)
        cuts = [R/2.0]*len(self._ase)
        nl = NeighborList(cuts, self_interaction=False, bothways=True)
        nl.update(self._ase)
        nbr_idx, offsets = nl.get_neighbors(center_idx)

        center_pos = self._ase.positions[center_idx]
        cell = self._ase.get_cell()
        sub_idx = [center_idx]
        for k,j in enumerate(nbr_idx):
            j_pos = self._ase.positions[j] + np.dot(offsets[k], cell)
            dist = float(np.linalg.norm(j_pos - center_pos))
            if self.water_radius_A is not None and getattr(self._ase[j], 'resname', '') == 'SOL':
                if dist <= self.water_radius_A + 1e-8: sub_idx.append(int(j))
            else:
                if dist <= self.radius_A + 1e-8: sub_idx.append(int(j))
        return sub_idx

    def slice_one(self, center_idx: int) -> Tuple[Chem.Mol, np.ndarray, int]:
        # 1) 索引集合
        sub_idx = self._subidx_by_neighborlist(center_idx)
        # 2) 取局部坐标 → GPU
        pos_np = self._ase.positions[sub_idx].astype(np.float32)
        pos = torch.from_numpy(pos_np).to(self.device)
        cell = self._cell
        symbols = [self._symbols[i] for i in sub_idx]

        # 3) GPU：补氢（sp3-only）
        posH, symH, H_parent = _gpu_add_h_sp3(pos, symbols, cell, center_local=0, device=self.device)

        # 4) 组轻量 RDKit Mol（CPU；很小的常数开销）
        pos_final = posH.detach().cpu().numpy()
        heavy_bonds = _infer_heavy_bonds(symH, pos_final)
        # H_parent 是 (h_local_idx, parent_local_idx)；构 RDKit 需要用最终索引
        rd_mol_h = _to_rdkit_mol(symH, pos_final, heavy_bonds, H_parent)

        # 5) 中心局部索引保持不变（我们没有重排重原子顺序，只是在尾部追加 H）
        # 用最近点保护一下（避免极端情况）
        cp = self._ase.positions[center_idx].astype(np.float32)
        c_local = int(np.linalg.norm(pos_final - cp[None,:], axis=1).argmin())

        return rd_mol_h, pos_final.astype(np.float32, copy=False), c_local

def _mark_dynamic_graph_tensors(batch):
    # 节点维(N)
    if hasattr(batch, "x") and isinstance(batch.x, torch.Tensor):
        torch._dynamo.mark_dynamic(batch.x, 0)
    if hasattr(batch, "pos") and isinstance(batch.pos, torch.Tensor):
        torch._dynamo.mark_dynamic(batch.pos, 0)
    # 边维(E) —— 你的构图里有两套边
    if hasattr(batch, "edge_index") and isinstance(batch.edge_index, torch.Tensor):
        torch._dynamo.mark_dynamic(batch.edge_index, 1)  # shape (2,E)
    if hasattr(batch, "adj_edge_index") and isinstance(batch.adj_edge_index, torch.Tensor):
        torch._dynamo.mark_dynamic(batch.adj_edge_index, 1)
    if hasattr(batch, "dist_edge_index") and isinstance(batch.dist_edge_index, torch.Tensor):
        torch._dynamo.mark_dynamic(batch.dist_edge_index, 1)
    if hasattr(batch, "adj_edge_weight") and isinstance(batch.adj_edge_weight, torch.Tensor):
        torch._dynamo.mark_dynamic(batch.adj_edge_weight, 0)       # (Ea,)
    if hasattr(batch, "dist_edge_weight") and isinstance(batch.dist_edge_weight, torch.Tensor):
        torch._dynamo.mark_dynamic(batch.dist_edge_weight, 0)      # (Ed,)

# -------------- multi-model calculator --------------
class MultiModelCalculator(Calculator):
    implemented_properties = ["energy","forces"]
    def __init__(self, gro_path: str, ckpt_map: Dict[str,str], centers: str="non-water",
                 radius_A: float=8.0, water_radius_A: Optional[float]=None,
                 T: float=298.15, device: Optional[str]=None,
                 itp_path: Optional[str]=None, cut_workers: Optional[int]=None, infer_batch_size: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.slicer = Slicer(gro_path, radius_A=radius_A, water_radius_A=water_radius_A, itp_path=itp_path)
        self.centers_sel = centers
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.kbT_eV = kBT_in_eV(T)
        # Load models per element
        single_ckpt = None
        if isinstance(ckpt_map, dict) and len(ckpt_map) >= 1:
            # 允许传多项，但我们只取第一项（建议只传一项）
            single_ckpt = next(iter(ckpt_map.values()))
        if single_ckpt is None and "ckpt" in kwargs and kwargs["ckpt"]:
            single_ckpt = kwargs["ckpt"]
        if not single_ckpt:
            raise RuntimeError("Need a single checkpoint path: pass --ckpt PATH or ckpt_map with 1 entry.")
        self.model = load_packaged_model(single_ckpt, self.device)
        # === compile the model ===
        try:
            torch.set_float32_matmul_precision("high")  # 对注意力/Conv 有益
            self.model = torch.compile(self.model, dynamic=True, mode="max-autotune")
        except Exception as e:
            print("[compile] fallback (skipped):", e)
        # 标记：首次 batch 时再做动态维标注
        self._dyn_marked = False

        self.center_list = self.slicer.centers(centers)

        import os as _os
        self.cut_workers = int(cut_workers if (cut_workers and cut_workers>0) else (_os.cpu_count() or 1))

 
        # 预计算常数：库伦常数( eV·nm / e^2 ) / kBT_eV
        self.maxF_dir = "maxF_slices"
        self._step = 0

        os.makedirs(self.maxF_dir, exist_ok=True)
        from collections import Counter
        elems = [self.slicer.element_of(i) for i in self.center_list]
        cnt = Counter(elems)
        print("[centers by element]", dict(cnt))
        # names = [''.join(ch for ch in self.slicer.u.atoms[i].name if ch.isalpha()) for i in self.center_list]
        # elems = [self.slicer.element_of(i) for i in self.center_list]
        # cnt = Counter(elems)
        # print("[centers by element]", dict(cnt))  # 例如 {'C':70,'O':37,'H':146,'Cl':2}

        # # 把“以 C 开头但被判成 Cl/Br 的”列出来，方便你核对
        # bad = [(i, self.slicer.u.atoms[i].name) for i in self.center_list
        #        if self.slicer.u.atoms[i].name.upper().startswith('C') and self.slicer.element_of(i) not in ('C',)]
        # if bad:
        #     print("[C-like but not grouped as C] (index, name):", bad[:20], " ... total", len(bad))
        self.infer_batch_size = int(max(1, infer_batch_size))

    def calculate(self, atoms=None, properties=('energy', 'forces'), system_changes=all_changes):
        # 让 ASE 的 Calculator 框架记录更改
        Calculator.calculate(self, atoms, properties, system_changes)

        # 当前 MD 步数（仅用于导出最大受力切片命名）
        self._step += 1

        # ---- 读取坐标并更新 slicer ----
        posA = self.atoms.get_positions()
        self.slicer.update(posA)

        # 结果容器
        # F_full = np.zeros((len(self.atoms), 3), dtype=np.float64)
        # E_sum = 0.0
        F_full_gpu = torch.zeros((len(self.atoms), 3), device=self.device, dtype=torch.float32)
        E_sum_gpu = torch.zeros((), device=self.device, dtype=torch.float32)
        center_force_gpu = torch.full((len(self.atoms),), -1.0, device=self.device, dtype=torch.float32)

        # center_force_list = []  # (gidx, |F|)

        # 统一模型：全体中心原子一次处理
        idxs = list(self.center_list)

        # 分段计时器
        _prof = {"slice": 0.0, "h2d": 0.0, "model": 0.0, "d2h": 0.0}
        _t = _SimpleTimer()

        # ---------- 1) 主进程切片：得到 (MolBlock, coordsA, center_local) ----------
        # 说明：MDAnalysis/Universe 不适合跨进程，这里在主进程完成“物理切片”，
        #       只把轻量的 MolBlock + 标量索引交给子进程做图构建。
        tasks = []
        for gidx in idxs:
            rd_mol_h, coordsA, c_local = self.slicer.slice_one(int(gidx))
            mol_block = Chem.MolToMolBlock(rd_mol_h)  # 包含坐标
            tasks.append((int(gidx), mol_block, int(c_local)))

        _prof["slice"] += _t.lap()

        # ---------- 2) 进程池并行构图：MolBlock → NumPy payload → 主进程还原为 PyG Data ----------
        data_map = {}  # gidx -> Data（张量都在 CPU 上，稍后统一搬到 GPU）
        if self.cut_workers and self.cut_workers > 1 and len(tasks) > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=self.cut_workers) as ex:
                futs = [ex.submit(build_one_subgraph_proc, t) for t in tasks]
                for fu in as_completed(futs):
                    gi, payload = fu.result()
                    data_map[gi] = payload_to_pyg(payload)
        else:
            for tsk in tasks:
                gi, payload = build_one_subgraph_proc(tsk)
                data_map[gi] = payload_to_pyg(payload)

        # ---------- 3) 批量推理（Pinned + non_blocking），严格保序 ----------
        bs = max(1, int(getattr(self, "infer_batch_size", 128)))
        use_amp = (hasattr(self, "device") and getattr(self.device, "type", None) == "cuda")

        # 更精确的 GPU 段计时（可选）
        if torch.cuda.is_available() and getattr(self.device, "type", "") == "cuda":
            _ev_start = torch.cuda.Event(enable_timing=True)
            _ev_end   = torch.cuda.Event(enable_timing=True)
        else:
            _ev_start = _ev_end = None

        for s in range(0, len(idxs), bs):
            t = min(s + bs, len(idxs))
            sub_idxs = [int(x) for x in idxs[s:t]]                 # ① 明确本批的中心原子顺序
            sub_list = [data_map[gi] for gi in sub_idxs]           # ② 按顺序取回 Data，顺序绝不乱

            # 组 Batch
            batch = Batch.from_data_list(sub_list)
            # Pinned memory（若 PyG 版本支持）
            if hasattr(batch, "pin_memory"):
                batch = batch.pin_memory()
            # H2D：non_blocking，减少等待
            batch = batch.to(self.device, non_blocking=True)
            _prof["h2d"] += _t.lap()
            # 首个 batch：告诉编译器哪些维度是动态的（避免 30~80 节点反复重编）
            if not getattr(self, "_dyn_marked", False):
                _mark_dynamic_graph_tensors(batch)
                self._dyn_marked = True
            # 模型前向（与训练保持同一 API；能量按样本求和）
            if _ev_start is not None:
                _ev_start.record()
            with torch.cuda.amp.autocast(enabled=use_amp):
                E_kBT, F_center = self.model.energy_and_forces(
                    batch, center_only=True, create_graph=False
                )
            if _ev_end is not None:
                _ev_end.record(); torch.cuda.synchronize()
                _prof["model"] += _ev_start.elapsed_time(_ev_end) / 1000.0
            else:
                _prof["model"] += _t.lap()

            # 回 CPU，回填 —— 力的单位换算请按你的训练标签确定：
            # 若训练力单位是 kBT/nm（你之前的默认），则保留 * (kbT_eV/10)
            # 若训练力单位已是 eV/Å，则用 F_np = F_center
            # F_center = F_center.detach().float().cpu().numpy()
            # F_np = F_center * (self.kbT_eV / 10.0)   # ← 若你的标签= eV/Å，改成：F_np = F_center
            F_center_eVA = F_center.to(torch.float32) * (self.kbT_eV / 10.0)
            sub_idxs_gpu = torch.as_tensor(sub_idxs, device=self.device, dtype=torch.long)
            F_full_gpu.index_copy_(0, sub_idxs_gpu, F_center_eVA)
            center_force_gpu.index_copy_(
                0, sub_idxs_gpu, torch.linalg.norm(F_center_eVA, dim=1)
            )
            E_sum_gpu += E_kBT.sum() * self.kbT_eV
            del batch, E_kBT, F_center, sub_list, sub_idxs_gpu
        _t.lap()
        torch.cuda.synchronize()
        E_sum = float(E_sum_gpu.detach().cpu().item())
        F_full = F_full_gpu.detach().cpu().numpy().astype(np.float64, copy=False)
        _prof["d2h"] += _t.lap()

        print(f"[prof] ALL slice={_prof['slice']:.3f}s  h2d={_prof['h2d']:.3f}s  model={_prof['model']:.3f}s  d2h+post={_prof['d2h']:.3f}s")
        self.results["energy"] = E_sum
        self.results["forces"] = F_full

        # ---- 导出“最大受力中心原子”的切片（xyz + sidecar），便于排查/可视化 ----
        gidx_max = int(center_force_gpu.detach().cpu().argmax().item())
        if gidx_max >= 0:  # 防守式判断
            rd_mol_h, coordsA, c_local = self.slicer.slice_one(gidx_max)
            stem = os.path.join(self.maxF_dir, f"frame_{self._step:06d}_center_{gidx_max}")
            save_sidecars_for_slice(
                out_stem=stem,
                rd_mol=rd_mol_h,
                coordsA=coordsA,     # Å
                center_local=int(c_local),
            )

def run(gro: str, ckpt_map: Dict[str,str], T: float=298.15, dt_fs: float=0.5, steps: int=100,
        centers: str="non-water", radius_A: float=8.0, water_radius_A: Optional[float]=None,
        itp_path: Optional[str]=None, cut_workers: Optional[int]=None):    
    u = mda.Universe(gro, convert_units=True)
    symbols=[]
    for at in u.atoms:
        s = ''.join(ch for ch in at.name if ch.isalpha())
        if len(s)>=2 and s[0].upper()=='C' and s[1].lower()=='l': symbols.append('Cl')
        elif len(s)>=2 and s[0].upper()=='B' and s[1].lower()=='r': symbols.append('Br')
        else: symbols.append(s[0].upper())
    atoms = ASEAtoms(symbols=symbols, positions=u.atoms.positions.copy(), pbc=False)
    calc = MultiModelCalculator(
        gro, ckpt_map,
        centers=centers, radius_A=radius_A, water_radius_A=water_radius_A, T=T,
        itp_path=itp_path, cut_workers=cut_workers
    )
    atoms.calc = calc
    MaxwellBoltzmannDistribution(atoms, temperature_K=T)
    dyn = Langevin(atoms, timestep=dt_fs*units.fs, temperature_K=T, friction=0.01)
    traj = Trajectory("traj.traj", "w", atoms, properties=["energy", "forces"])
    dyn.attach(traj.write, interval=1)

# 同时追加写入人类可读的 XYZ（便于快速预览/外部工具）
    dyn.attach(lambda: write("traj.xyz", atoms, format="extxyz", append=True), interval=1)
    def log():
        F = atoms.get_forces()
        maxF = float((F**2).sum(axis=1).max()**0.5)
        print(f"step {dyn.nsteps:5d} | max|F| = {maxF:.4f} eV/Å")
    dyn.attach(log, interval=1)
    print(f"[ASE-multi] dt={dt_fs}fs, centers={centers}, steps={steps}")
    dyn.run(steps)
    return atoms

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gro", required=True)
    ap.add_argument("--ckpt_map", required=True, help="Mapping like C=ckptC.pth;H=ckptH.pth;O=ckptO.pth or a JSON file path")
    # ap.add_argument("--ckpt_map", required=False, default="C=ccc.pt;O=ooo.pt;H=hhh.pt",
                # help="(Fixed default) C=c.pt;O=o.pt;H=h.pt; can still pass a JSON or mapping to override if needed")

    ap.add_argument("--itp", type=str, default=None, help="Topology .itp for cut_v8 (atoms/bonds)")
    ap.add_argument("--cut-workers", type=int, default=0, help="Threads for parallel slicing (default: CPU count)")
    ap.add_argument("--T", type=float, default=298.15)
    ap.add_argument("--dt", type=float, default=1)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--centers", type=str, default="non-water")
    ap.add_argument("--radius", type=float, default=8.0)
    ap.add_argument("--water_radius", type=float, default=None)
    # ap.add_argument("--coulomb", action="store_true", help="Enable long-range Coulomb correction beyond cutoff")
    ap.add_argument("--no-coulomb", dest="coulomb", action="store_false",
                help="Disable long-range Coulomb correction (default: enabled)")
    ap.set_defaults(coulomb=True)
    ap.add_argument("--epsr", type=float, default=80.0, help="Relative dielectric constant (default: 80.0)")
    ap.add_argument("--coulomb-on", type=float, default=0.8, help="Coulomb switch-on distance r_on [nm]")
    ap.add_argument("--coulomb-off", type=float, default=2.5, help="Coulomb switch-off distance r_off [nm]")

    args = ap.parse_args()
    # parse ckpt_map
    ckpt_map = {}
    if os.path.isfile(args.ckpt_map):
        with open(args.ckpt_map,"r") as f:
            ckpt_map = json.load(f)
    else:
        for kv in args.ckpt_map.split(";"):
            if not kv: continue
            k,v = kv.split("=",1)
            ckpt_map[k.strip()] = v.strip()
    run(args.gro, ckpt_map, T=args.T, dt_fs=args.dt, steps=args.steps,
        centers=args.centers, radius_A=args.radius, water_radius_A=args.water_radius,
        itp_path=args.itp, cut_workers=args.cut_workers)