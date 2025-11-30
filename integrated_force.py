#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
integrated_processed_only.py
一步生成 processed_molecules.json（含 adjusted_energy_diff）
"""
import os, re, json, numpy as np, torch
from tqdm import tqdm
from scipy.sparse.csgraph import connected_components
import func

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
# ───────── 根据你的目录结构修改这里 ──────────
GRO_DIR     = "gro_has_force"
CHARGE_DIR  = "charge"
FORCE_FILE  = "forces.txt"     # ← 你的力文件路径
OUT_JSON    = "processed_molecules_with_forces.json"
TEMPERATURE = 298.15                  # kBT 温度

# ------------------------------------------
_PREFIX_RE = re.compile(r'(output\[\d+_\d+\])')

def _as_element(name: str) -> str:
    """
    把 .gro 的原子名转换成 RDKit 需要的元素符号：
    - 去掉末尾的数字（如 C0 -> C, O12 -> O）
    - 只保留字母，并做大小写规范化（Cl, Br, Si 等两字符元素）
    - 若异常则回退为首字母大写
    """
    if not name:
        return "C"
    # 只保留字母，并去掉末尾所有数字
    s = re.sub(r'\d+$', '', ''.join(ch for ch in str(name) if ch.isalpha()))
    if not s:
        return "C"
    if len(s) == 1:
        return s.upper()
    return s[0].upper() + s[1:].lower()
    
def _get_prefix(s: str) -> str:
    """从路径或文件名中提取 'output[i_j]' 作为键"""
    base = os.path.basename(str(s))
    m = _PREFIX_RE.search(base)
    return m.group(1) if m else os.path.splitext(base)[0]

# ===== Unit conversion: Hartree/bohr → kBT/nm =====
E_HARTREE_J   = 4.3597447222071e-18  # J
A0_M          = 5.29177210903e-11    # m
K_B_J_PER_K   = 1.380649e-23         # J/K

def hartree_bohr_to_kbt_per_nm(values, temperature: float = 298.15):
    """
    输入可为标量/1D/2D；返回 np.ndarray（单位：kBT/nm）
    """
    values = np.asarray(values, dtype=float)
    factor = (E_HARTREE_J / A0_M) * 1e-9 / (K_B_J_PER_K * float(temperature))
    return values * factor

def parse_force_file(force_filepath: str, temperature: float = 298.15, gradient_input: bool = True) -> dict:
    """
    解析形如：
      output[0_131].out:  fx fy fz
    的文件。文件里的数值若是“梯度 dE/dx”，则 gradient_input=True，会自动乘以 -1 得到力 F=-∇E。
    返回：{ 'output[i_j]': np.array([Fx,Fy,Fz]) }（单位：kBT/nm）
    """
    forces = {}
    with open(force_filepath, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ":" not in ln:
                continue
            left, right = ln.split(":", 1)
            key = _get_prefix(left.strip())

            parts = right.replace(",", " ").split()
            if len(parts) < 3:
                continue

            vec_hb = np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float)
            # 单位换算 Hartree/bohr → kBT/nm
            vec = hartree_bohr_to_kbt_per_nm(vec_hb, temperature=temperature)
            # 若输入是梯度，则力 = -梯度
            if gradient_input:
                vec = -vec
            forces[key] = vec
    return forces

ATOM_TYPE_ADJ = {
    "C": -37.80955134357223,
    "O": -75.0072078888503,
    "H": -0.5054521818896177,
    # … 如有需要继续补充 …
}

def parse_energy(path):
    pat = re.compile(r"(output_\[\d+\s+\d+\])\.out: *(-?\d+\.\d+)")
    mp = {}
    with open(path) as fh:
        for ln in fh:
            m = pat.search(ln)
            if m:
                mp[m.group(1)] = float(m.group(2))
    return mp

# Global feature min and max values (combined from two datasets)
# Length: 30 dimensions corresponding to the global_features vector

GF_min = [
    150.174,  # molecular weight
    10,       # heavy atom count
    0,        # bond count
    0.510212, # principal moment I1
    6.37445,  # principal moment I2
    9.04858,  # principal moment I3
    0.0147833,# NPR1 (I1/I3)
    0.531987, # NPR2 (I2/I3)
    0.27531,  # global Rg_frame
    0.27531,  # local Rg_mol
    1925.14,  # shell mass density 0.00–0.05 nm
    0,        # shell mass density 0.05–0.10 nm
    202.646,  # shell mass density 0.10–0.15 nm
    0,        # shell mass density 0.15–0.20 nm
    31.5596,  # shell mass density 0.20–0.25 nm
    0,        # shell mass density 0.25–0.30 nm
    0,        # shell mass density 0.30–0.35 nm
    0,        # shell mass density 0.35–0.40 nm
    0,        # shell mass density 0.40–0.45 nm
    0,        # shell mass density 0.45–0.50 nm
    0,        # shell mass density 0.50–0.55 nm
    0,        # shell mass density 0.55–0.60 nm
    0,        # shell mass density 0.60–0.65 nm
    0,        # shell mass density 0.65–0.70 nm
    0,        # shell mass density 0.70–0.75 nm
    0,        # shell mass density 0.75–0.80 nm
    0,        # shell mass density 0.80–0.85 nm
    0,        # shell mass density 0.85–0.90 nm
    0,        # shell mass density 0.90–0.95 nm
    0         # shell mass density 0.95–1.00 nm
]

GF_max = [
    502.598,  # molecular weight
    34,       # heavy atom count
    27,       # bond count
    61.7004,  # principal moment I1
    100.588,  # principal moment I2
    117.075,  # principal moment I3
    0.882843, # NPR1 (I1/I3)
    0.999943, # NPR2 (I2/I3)
    0.607002, # global Rg_frame
    0.607002, # local Rg_mol
    30555.8,  # shell mass density 0.00–0.05 nm
    0,        # shell mass density 0.05–0.10 nm
    3018.18,  # shell mass density 0.10–0.15 nm
    1497.85,  # shell mass density 0.15–0.20 nm
    2349.53,  # shell mass density 0.20–0.25 nm
    2183.98,  # shell mass density 0.25–0.30 nm
    1489.77,  # shell mass density 0.30–0.35 nm
    1255.26,  # shell mass density 0.35–0.40 nm
    1365.32,  # shell mass density 0.40–0.45 nm
    916.556,  # shell mass density 0.45–0.50 nm
    814.154,  # shell mass density 0.50–0.55 nm
    625.871,  # shell mass density 0.55–0.60 nm
    521.579,  # shell mass density 0.60–0.65 nm
    478.725,  # shell mass density 0.65–0.70 nm
    387.671,  # shell mass density 0.70–0.75 nm
    280.996,  # shell mass density 0.75–0.80 nm
    224.613,  # shell mass density 0.80–0.85 nm
    239.125,  # shell mass density 0.85–0.90 nm
    160.015,  # shell mass density 0.90–0.95 nm
    149.09    # shell mass density 0.95–1.00 nm
]

# 你可以将这两个列表传递给 load_mol_data 的参数 gf_min, gf_max
# 例如:
# results = load_mol_data(gro, charge, energy, frame_dir, gf_min, gf_max)

def _find_center_from_subgro(gro_filepath: str):
    """
    在子分子 .gro 中查找原子名以 '0' 结尾的原子作为中心原子（如 C0、O0）。
    返回: (center_local_idx, center_atom_info)
      - center_local_idx: 本 .gro 内 0-based 索引；找不到则为 None
      - center_atom_info: dict，含 residue/atom_type/atom_index/xyz 及可选 frame/center_global
    """
    center_local_idx, center_atom_info = None, None
    with open(gro_filepath, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
        n_line = f.readline()
        if not n_line:
            raise ValueError(f"Invalid GRO: missing atom count in {gro_filepath}")
        try:
            n_atoms = int(n_line.strip().split()[0])
        except Exception as e:
            raise ValueError(f"Invalid GRO atom count line in {gro_filepath}: {n_line}") from e

        for i in range(n_atoms):
            line = f.readline()
            if not line:
                raise ValueError(f"GRO truncated before reading {n_atoms} atoms: {gro_filepath}")

            if len(line) >= 44:
                resid    = int(line[0:5])
                resname  = line[5:10].strip()
                atomname = line[10:15].strip()
                atomnum  = int(line[15:20])
                x = float(line[20:28]); y = float(line[28:36]); z = float(line[36:44])
            else:
                parts = line.split()
                if len(parts) < 7:
                    raise ValueError(f"Unrecognized GRO atom line: {line}")
                resid, resname, atomname, atomnum = int(parts[0]), parts[1], parts[2], int(parts[3])
                x, y, z = float(parts[4]), float(parts[5]), float(parts[6])

            if atomname.endswith("0") and center_local_idx is None:
                center_local_idx = i
                center_atom_info = {
                    "residue": f"{resid}{resname}",
                    "atom_type": atomname,   # 例如 'C0'
                    "atom_index": atomnum,   # 该行的原子编号
                    "x": x, "y": y, "z": z,
                    "center_idx": atomnum-1   # 该行的原子编号

                }

        # 读掉可选 box 行
        _ = f.readline()

    # 从首行提取 frame / center（元信息，可选）
    m = re.search(r'frame\s*=\s*(\d+)\s*,\s*center\s*=\s*(\d+)', header)
    if m:
        if center_atom_info is None:
            center_atom_info = {}
        center_atom_info.update({
            "frame": int(m.group(1)),
            "center_global": int(m.group(2))
        })
    return center_local_idx, center_atom_info

def load_mol_data(gro, charge, force_vec, gf_min=GF_min, gf_max=GF_max):
    """
    读取单个 .gro（子分子），中心原子在 .gro 内以 '...0' 命名（如 C0）。
    - 不再依赖 frame_X.gro
    - 标签为 'force_vec'（单位：kBT/nm），如果文件里的原本是梯度，已提前在 parse_force_file 中取了负号
    """
    R_max = 1.0
    dr    = 0.05
    bin_edges = np.arange(0.0, R_max + dr, dr)

    # 1) 解析 + 距离/邻接（沿用你 func.py 的函数）
    items = func.load_molecule_data_with_adj_and_dist(gro, charge)

    # 2) 在子分子 .gro 内找中心原子
    center_local_idx, center_atom_info = _find_center_from_subgro(gro)

    results = []
    for it in items:
        try:
            rdkit_mol = func.create_rdkit_molecule(it["atom_types"], it["coordinates"])
            if rdkit_mol is None:
                continue

            atom_masses = np.array([atom.GetMass() for atom in rdkit_mol.GetAtoms()], dtype=float)
            it["dist_inv_matrix_normalized"] = func.min_max_normalize_dist_inv_matrix(it["dist_inv_matrix"])

            orig = func.generate_node_features(rdkit_mol, it["charges"])
            orig_np = (orig.detach().cpu().numpy()
                       if isinstance(orig, torch.Tensor) else np.asarray(orig))
            coords = np.asarray(it["coordinates"], dtype=float)

            # 3) 连通分量（优先取包含中心原子的分量，否则取最大分量）
            adj = np.array(it["adj_matrix"], dtype=int)
            n_comp, labels = connected_components(adj, directed=False, return_labels=True)
            if center_local_idx is not None and 0 <= center_local_idx < len(labels):
                center_comp = int(labels[center_local_idx])
            else:
                comp_sizes = np.bincount(labels)
                center_comp = int(comp_sizes.argmax())

            coords_mol = coords[labels == center_comp]

            # 4) 全局特征（与你当前版本一致）
            mol_wt = float(atom_masses.sum())
            heavy_atom_count = sum(1 for atom in rdkit_mol.GetAtoms() if atom.GetSymbol() != 'H')
            bond_count = rdkit_mol.GetNumBonds()

            com = (coords * atom_masses[:, None]).sum(axis=0) / atom_masses.sum()
            rel = coords - com
            Ixx = np.sum(atom_masses * (rel[:,1]**2 + rel[:,2]**2))
            Iyy = np.sum(atom_masses * (rel[:,0]**2 + rel[:,2]**2))
            Izz = np.sum(atom_masses * (rel[:,0]**2 + rel[:,1]**2))
            Ixy = np.sum(atom_masses * rel[:,0] * rel[:,1])
            Ixz = np.sum(atom_masses * rel[:,0] * rel[:,2])
            Iyz = np.sum(atom_masses * rel[:,1] * rel[:,2])

            inertia = np.array([[ Ixx, -Ixy, -Ixz],
                                [-Ixy,  Iyy, -Iyz],
                                [-Ixz, -Iyz,  Izz]])
            eigvals, _ = np.linalg.eigh(inertia)
            I1, I2, I3 = np.sort(eigvals)
            NPR1 = I1 / I3 if I3 != 0 else 0.0
            NPR2 = I2 / I3 if I3 != 0 else 0.0

            cm_frame = coords.mean(axis=0)
            rg_frame = float(np.sqrt(((coords - cm_frame) ** 2).sum(axis=1).mean()))
            cm_mol = coords_mol.mean(axis=0)
            rg_mol = float(np.sqrt(((coords_mol - cm_mol) ** 2).sum(axis=1).mean()))

            # 5) 以中心原子坐标做壳层密度；若没有就用分量质心
            if center_local_idx is not None and 0 <= center_local_idx < len(coords):
                center_xyz_local = coords[center_local_idx]
            else:
                center_xyz_local = cm_mol

            dists = np.linalg.norm(coords - center_xyz_local, axis=1)
            shell_mass_density = []
            for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
                mask = (dists >= lo) & (dists < hi)
                volume = (4.0 / 3.0) * np.pi * (hi**3 - lo**3)
                m = atom_masses[mask].sum()
                shell_mass_density.append(m / volume if volume > 0 else 0.0)

            new_globals = [mol_wt, heavy_atom_count, bond_count, I1, I2, I3, NPR1, NPR2]
            global_features = new_globals + [rg_frame, rg_mol] + shell_mass_density

            # 6) Min–Max 归一化
            norm_gf = []
            for idx, val in enumerate(global_features):
                mn, mx = gf_min[idx], gf_max[idx]
                norm_gf.append((val - mn) / (mx - mn) if mx > mn else 0.0)

            results.append({
                "gro_file": os.path.basename(gro),
                "atom_types": it["atom_types"],
                "coordinates": it["coordinates"],
                "formal_charges": it["charges"],
                "adj_matrix": it["adj_matrix"].tolist(),
                "dist_inv_matrix_normalized": it["dist_inv_matrix_normalized"].tolist(),
                "dist_inv_matrix": it["dist_inv_matrix"].tolist(),
                "atom_features": orig_np.tolist(),
                # ↓↓↓ 仅力标签（kBT/nm），已保证 F = -∇E
                "force_label": np.asarray(force_vec, dtype=float).tolist(),
                "force_unit": "kBT/nm",
                "center_atom": center_atom_info,
                "global_features": global_features,
                "normalizes_global_features": norm_gf,
            })
        except Exception as e:
            print(f"[skip] molecule in {gro}: {e}")
    return results

def _worker(args):
    gf, forces_map = args
    # 以 output[i_j] 为键
    base = _get_prefix(gf)
    cf   = os.path.join(CHARGE_DIR, f"{base}.txt")
    if base not in forces_map or not os.path.exists(cf):
        return []  # 跳过

    mols = load_mol_data(
        os.path.join(GRO_DIR, gf), cf, forces_map[base]
    )
    return mols

def main():
    # 读力（文件里是梯度 → parse_force_file 会自动乘 -1 变成力）
    forces_map = parse_force_file(FORCE_FILE, temperature=TEMPERATURE, gradient_input=True)

    gro_files = [f for f in os.listdir(GRO_DIR) if f.endswith(".gro")]

    out = []
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as exe:
        futures = [exe.submit(_worker, (gf, forces_map)) for gf in gro_files]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            out.extend(fut.result())

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=4)
    print(f"[✓] {len(out)} molecules → {OUT_JSON}")

if __name__ == "__main__":
    main()

