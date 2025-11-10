#!/usr/bin/env python3
"""Linear least-squares fitting of Lennard–Jones parameters from force data.

This script ingests a dataset of molecular configurations where the force on a
designated *centre atom* is known.  By rewriting the Lennard–Jones (LJ) 12-6
potential into a linear form with respect to pair-specific coefficients,
``alpha_T`` and ``beta_T``, the fitting problem becomes a standard linear
least-squares system.  Optional Coulomb interactions can be removed prior to
fitting, bonded (bond/angle/dihedral) forces are subtracted using
``compare_forces.py`` helpers, and an additional non-linear recovery step can
estimate per-element ``sigma``/``epsilon`` parameters using the
Lorentz–Berthelot combining rules.

The script is intentionally self-contained and can generate a miniature
synthetic dataset for quick self-tests when invoked without explicit input
files.  Only :mod:`numpy` and :mod:`scipy` are required at runtime (``tqdm`` is
optional for progress reporting but not required).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys, csv
from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy
from numba import njit

from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

try:  # Optional but helpful progress bar
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None

from scipy.optimize import least_squares, lsq_linear, nnls

from compare_forces import infer_topology_from_summary, compute_forces

DEFAULT_ALPHA = 1e-6
DEFAULT_FUDGE_LJ = 0.5
DEFAULT_FUDGE_QQ = 0.5
DEFAULT_KE = 138.935456


@dataclass
class ProcessedEntry:
    """Container for a validated dataset entry."""

    index: int
    gro_file: Optional[str]
    atom_types: List[str]
    elements: List[str]
    coordinates: NDArray[np.float64]
    charges: Optional[NDArray[np.float64]]
    adjacency: NDArray[np.int32]
    center_index: int
    target_force: NDArray[np.float64]
    box: Optional[NDArray[np.float64]]
    topological_distances: NDArray[np.int32]
    coulomb_force: NDArray[np.float64]
    bond_force: NDArray[np.float64]
    angle_force: NDArray[np.float64]
    dihedral_force: NDArray[np.float64]
    bonded_force: NDArray[np.float64]

def load_force_map_from_csv(
    csv_path: str | Path,
    label: str,
):
    """
    从 scan_ff.py 输出的 CSV 里读取中心原子力 (Fx,Fy,Fz)。

    key: (gro_file, center_index:int)
    """
    csv_path = Path(csv_path)
    fmap = {}

    if not csv_path.is_file():
        print(f"[load_{label}] WARNING: {csv_path} 不存在，返回空字典")
        return fmap

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("center_index", "").upper() == "ERROR":
                continue
            try:
                gro = row["gro_file"]
                idx = int(row["center_index"])
                Fx = float(row["Fx"])
                Fy = float(row["Fy"])
                Fz = float(row["Fz"])
            except (KeyError, ValueError):
                continue

            key = (gro, idx)
            fmap[key] = np.array([Fx, Fy, Fz], dtype=float)

    print(f"[load_{label}] 读取 {csv_path}, 有效条目: {len(fmap)}")
    return fmap

def patch_missing_angle_params(summary):
    """
    为 compare_forces 补上缺失的角度参数（例如 'o|hh'），
    采用 k=0 的dummy参数，保证不影响能量和力，只是防止报错。
    """
    if not isinstance(summary, dict):
        return

    # 可能的容器名字，根据你的summary结构适配
    angle_dicts = []
    for key in ("angles", "angle_types", "angle_params"):
        val = summary.get(key)
        if isinstance(val, dict):
            angle_dicts.append(val)

    if not angle_dicts:
        return

    # 需要补的角度类型列表，可按报错逐步加
    missing_keys = ["o|hh"]

    for mk in missing_keys:
        for d in angle_dicts:
            if mk not in d:
                # k=0 不产生角度力；theta 给个正常值以防内部用到
                d[mk] = {
                    "k": 0.0,
                    "theta": 109.5
                }
                print(f"[patch] Inject dummy angle param for {mk}")

def compute_dihedral_forces_only(
    entries: Sequence[ProcessedEntry],
    summary: Mapping[str, object],
    verbose: bool = True,
) -> Dict[Tuple[str, int], np.ndarray]:
    """
    用 compare_forces 只计算 dihedral 对中心原子的力。

    返回:
      dih_map[(gro_file, center_index)] = np.array([Fx, Fy, Fz])
    """
    try:
        import compare_forces as cf
    except Exception as exc:
        raise RuntimeError("compare_forces.py is required to compute dihedral forces") from exc

    dih_map: Dict[Tuple[str, int], np.ndarray] = {}
    if not entries or not isinstance(summary, Mapping) or not summary:
        return dih_map

    for entry in entries:
        gro = entry.gro_file
        c0 = int(entry.center_index)
        coords = entry.coordinates
        box = entry.box if entry.box is not None else np.array([1e6, 1e6, 1e6], dtype=float)

        try:
            top = cf.infer_topology_from_summary(summary, coords, entry.atom_types)

            # 只保留包含中心原子的 dihedral，清空 bond/angle
            def _keep_dih(d):
                z = (d.i - 1, d.j - 1, d.k - 1, d.l - 1)
                return c0 in z

            top.bonds = []  # 不要 bond
            top.angles = []  # 不要 angle
            top.rb_dihedrals = [d for d in getattr(top, "rb_dihedrals", []) if _keep_dih(d)]

            contrib = cf.compute_forces(
                top,
                coords,
                box,
                enable_nb=False,
                enable_bond=False,
                enable_angle=False,
                enable_dih=True,
                split_nb=True,
            )
        except Exception as ex:
            if verbose:
                print(f"[dih_only][WARN] entry={entry.index} gro={gro} failed: {ex}", flush=True)
            continue

        # 抽取 DIH 项
        F = None
        for nm in ("DIH", "DIHEDRAL", "dih", "dihedral", "Dihedral"):
            if nm in contrib:
                F = np.asarray(contrib[nm], dtype=float)
                break
        if F is None or F.size == 0:
            if verbose:
                print(f"[dih_only] entry={entry.index} gro={gro}: no dihedral contribution", flush=True)
            continue

        if not (0 <= c0 < F.shape[0]):
            if verbose:
                print(f"[dih_only] entry={entry.index} gro={gro}: centre index {c0} out of range", flush=True)
            continue

        Fx, Fy, Fz = F[c0]
        key = (gro, c0)
        dih_map[key] = np.array([Fx, Fy, Fz], dtype=float)

    if verbose:
        print(f"[dih_only] 成功记录中心原子 dihedral 力: {len(dih_map)}", flush=True)

    return dih_map

def load_data(data_path: Optional[Path], summary_path: Optional[Path]) -> Tuple[List[MutableMapping[str, object]], MutableMapping[str, object]]:
    """Load dataset and summary JSON files.

    Parameters
    ----------
    data_path:
        Path to the dataset JSON.  If ``None`` an empty list is returned.
    summary_path:
        Path to the summary JSON.  If ``None`` an empty dictionary is returned.
    """

    if data_path is None:
        raw_data: List[MutableMapping[str, object]] = []
    else:
        with data_path.open("r", encoding="utf-8") as fh:
            raw_data = json.load(fh)
            if not isinstance(raw_data, list):
                raise ValueError("Dataset JSON must contain a list of entries")

    if summary_path is None:
        summary: MutableMapping[str, object] = {}
    else:
        with summary_path.open("r", encoding="utf-8") as fh:
            summary = json.load(fh)
            if not isinstance(summary, MutableMapping):
                raise ValueError("Summary JSON must contain a JSON object")

    return raw_data, summary

def _normalise_system(A: np.ndarray, y: np.ndarray):
    """按列标准化设计矩阵，并把 y 也按其 RMS 做一次缩放；返回 (A', y', col_scale, y_scale)。"""
    col_scale = np.linalg.norm(A, axis=0)
    # 防止 0 列
    col_scale[col_scale == 0] = 1.0
    A_norm = A / col_scale

    y_scale = max(1.0, float(np.linalg.norm(y) / math.sqrt(len(y))))
    y_norm = y / y_scale
    return A_norm, y_norm, col_scale, y_scale

def _denormalise_theta(theta_norm: np.ndarray, col_scale: np.ndarray, y_scale: float):
    """把解缩放回原始尺度。"""
    # 因为 A' = A / s, y' = y / t, 解 θ' ≈ argmin||A'θ' - y'||，对应 θ = (t/s) * θ'
    return (y_scale / col_scale) * theta_norm


def extract_element(type_label: str) -> str:
    """Extract the chemical element from an atom type string."""

    match = re.match(r"^[A-Z][a-z]?", type_label)
    if not match:
        raise ValueError(f"Cannot extract element from atom type '{type_label}'")
    return match.group(0)


def determine_element_labels(
    atom_types: Sequence[str],
    adjacency: NDArray[np.int32],
    element_map: Mapping[str, str],
) -> List[str]:
    """Assign element labels with hydroxyl specialisation when applicable."""

    base_labels: List[str] = []
    for atom_type in atom_types:
        if atom_type in element_map:
            mapped = str(element_map[atom_type])
        else:
            mapped = extract_element(atom_type)
        mapped = mapped.strip()
        if not mapped:
            raise ValueError(f"Element label for atom type '{atom_type}' is empty")
        if "_" in mapped:
            base_part, suffix = mapped.split("_", 1)
            if base_part:
                base_part = base_part[0].upper() + base_part[1:].lower()
            base_labels.append(f"{base_part}_{suffix}" if suffix else base_part)
        else:
            base_labels.append(mapped[0].upper() + mapped[1:].lower())

    adjacency_array = np.asarray(adjacency, dtype=int)
    if adjacency_array.shape[0] != len(atom_types):
        raise ValueError("Adjacency matrix size must match atom types for element determination")

    base_elements = [label.split("_")[0].upper() for label in base_labels]
    specialised: List[str] = []
    for idx, label in enumerate(base_labels):
        if "_" in label:
            specialised.append(label)
            continue
        base = base_elements[idx]
        neighbours = np.nonzero(adjacency_array[idx])[0]
        if base == "O":
            if any(base_elements[nb] == "H" for nb in neighbours):
                specialised.append("O_h")
            else:
                specialised.append("O")
        elif base == "H":
            if any(base_elements[nb] == "O" for nb in neighbours):
                specialised.append("H_h")
            else:
                specialised.append("H")
        else:
            specialised.append(base)

    return specialised

import math
import numpy as np

def _as_box_vec(box):
    """box 支持 [Lx,Ly,Lz] 或 3x3，对角取 L。否则返回 None。"""
    if box is None:
        return None
    B = np.asarray(box, dtype=float)
    if B.shape == (3,):
        return B
    if B.shape == (3, 3):
        return np.array([B[0,0], B[1,1], B[2,2]], dtype=float)
    return None

def _apply_mic(delta, box_vec):
    """最小镜像折回到 [-L/2, L/2] 区间，仅处理正交盒。"""
    if box_vec is None:
        return delta
    d = delta.copy()
    for k in range(3):
        L = box_vec[k]
        if L > 0:
            d[k] -= L * np.round(d[k] / L)
    return d

def _nearest_index_by_coord(target_xyz, coords, box=None):
    """返回与 target_xyz 最近的原子索引及最近距离（考虑 MIC）。"""
    coords = np.asarray(coords, dtype=float)   # 形状 (N,3)
    target = np.asarray(target_xyz, dtype=float).reshape(3,)
    box_vec = _as_box_vec(box)
    best_idx, best_d = -1, float("inf")
    for i in range(coords.shape[0]):
        disp = coords[i] - target
        disp = _apply_mic(disp, box_vec)
        d = float(np.linalg.norm(disp))
        if d < best_d:
            best_d, best_idx = d, i
    return best_idx, best_d

def infer_center_index_if_needed(entry, coords, box, entry_idx=None, gro=None):
    """
    逻辑：

      1) 坐标推断：
         - entry.center_coord
         - entry.center_atom.xyz
         - 由 entry.center_atom.x/y/z 组装
      2) 推断失败则抛出详细错误。
    """
    n_atoms = int(len(coords))
    # --- 坐标推断 ---
    # 尝试多种字段来源
    target_xyz = None
    if "center_coord" in entry and entry["center_coord"] is not None:
        target_xyz = np.asarray(entry["center_coord"], dtype=float)
    elif "center_atom" in entry and isinstance(entry["center_atom"], dict):
        ca = entry["center_atom"]
        if "xyz" in ca and ca["xyz"] is not None:
            target_xyz = np.asarray(ca["xyz"], dtype=float)
        else:
            # 你的 JSON 里是分开的 x/y/z
            if all(k in ca for k in ("x", "y", "z")):
                target_xyz = np.array([float(ca["x"]), float(ca["y"]), float(ca["z"])], dtype=float)

    if target_xyz is None or target_xyz.shape != (3,):
        raise ValueError(
            f"[entry={entry_idx}] Unable to infer centre: no valid centre coordinate found "
            f"(need center_coord or center_atom.xyz or center_atom.x/y/z); gro={gro}"
        )

    best_idx, min_d = _nearest_index_by_coord(target_xyz, coords, box=box)
    if not (0 <= best_idx < n_atoms) or math.isinf(min_d):
        raise IndexError(
            f"[entry={entry_idx}] Failed to infer centre index via coordinates; "
            f"min_dist={min_d}, gro={gro}"
        )
    return int(best_idx)

def _extract_atoms_tuple(item) -> Tuple[int, ...]:
    """
    从一个成键项对象里提取原子索引元组，但不修改其它参数字段。
    兼容格式：
      - {"atoms": [i,j]} / [i,j,k,l]
      - {"i": i, "j": j, "k": k, "l": l}
      - 直接是 list/tuple
    """
    if isinstance(item, Mapping):
        if "atoms" in item:
            return tuple(int(x) for x in item["atoms"])
        keys = [k for k in ("i", "j", "k", "l") if k in item]
        if keys:
            return tuple(int(item[k]) for k in keys)
    if isinstance(item, (list, tuple)):
        return tuple(int(x) for x in item)
    raise ValueError(f"Unrecognized topology item format (no atoms): {item!r}")

def _atoms_to_zero_based(atoms: Tuple[int, ...], n_atoms: int) -> Tuple[int, ...]:
    """
    自动检测 1-based：若 max==n_atoms 或 (min==1 且 max<=n_atoms)，统一减 1；否则视为 0-based。
    """
    if not atoms:
        return atoms
    mx, mn = max(atoms), min(atoms)
    if mx == n_atoms or (mn == 1 and mx <= n_atoms):
        return tuple(a - 1 for a in atoms)
    return atoms

def _clone_with_atoms_zero_based(item, n_atoms: int):
    """
    深拷贝单条成键项，把原子索引字段转为 0-based，保留其它参数不变。
    兼容 {"atoms":[...] } 或 {"i":..,"j":..,"k":..,"l":..} 两种写法。
    """
    cloned = deepcopy(item)
    # 优先 "atoms" 容器
    if isinstance(cloned, Mapping) and "atoms" in cloned:
        cloned["atoms"] = list(_atoms_to_zero_based(tuple(int(x) for x in cloned["atoms"]), n_atoms))
        return cloned
    # 其次独立字段
    if isinstance(cloned, Mapping):
        used = False
        for k in ("i", "j", "k", "l"):
            if k in cloned:
                used = True
        if used:
            tup = tuple(int(cloned[k]) for k in ("i", "j", "k", "l") if k in cloned)
            tup0 = _atoms_to_zero_based(tup, n_atoms)
            for idx, k in enumerate(("i", "j", "k", "l")):
                if k in cloned:
                    cloned[k] = int(tup0[idx])
            return cloned
    # list/tuple 直接替换为 {"atoms":[...]}
    if isinstance(cloned, (list, tuple)):
        tup0 = _atoms_to_zero_based(tuple(int(x) for x in cloned), n_atoms)
        return {"atoms": list(tup0)}
    # 其它：不动
    return cloned

def _filter_centre_terms(summary: Mapping[str, object], n_atoms: int, centre: int) -> Dict[str, List[dict]]:
    """
    返回只包含中心原子的 bonds/angles/dihedrals 列表（每条是深拷贝，保留完整参数字段），
    且原子索引统一为 0-based。
    """
    out = {"bonds": [], "angles": [], "dihedrals": []}
    for key in ("bonds", "angles", "dihedrals"):
        items = summary.get(key) or []
        for it in items:
            try:
                atoms_raw = _extract_atoms_tuple(it)
            except Exception:
                continue
            atoms0 = _atoms_to_zero_based(atoms_raw, n_atoms)
            if centre in atoms0:
                out[key].append(_clone_with_atoms_zero_based(it, n_atoms))
    return out
def compute_bonded_forces(
    entries: Sequence[ProcessedEntry],
    summary: Mapping[str, object],
    verbose: bool = False,
) -> None:
    """
    用完整 summary 构拓扑 -> 仅保留“含中心原子”的 bond/angle/dihedral -> 只算成键力 -> 取中心分量。
    - 对每个 entry：
        1) 用 compare_forces.infer_topology_from_summary(summary, coords, atom_types)
        2) 过滤 top.bonds / top.angles / top.rb_dihedrals，只保留含中心原子的项
        3) compute_forces(enable_bond=True, enable_angle=True, enable_dih=True, enable_nb=False)
        4) 把中心原子的 bond/angle/dihedral/bonded 力写回 entry
    - 之后如果你想用 CSV 覆盖 bond/angle，只要改 entry.bond_force/angle_force，再重算 bonded_force 即可。
    """
    if not entries or not isinstance(summary, Mapping) or not summary:
        return

    try:
        import compare_forces as cf
    except Exception as exc:
        raise RuntimeError("compare_forces.py is required to compute bonded forces") from exc


    def _get_component(contrib: Mapping[str, np.ndarray], names, shape):
        """从 compute_forces 的结果里按多个候选 key 抽一项；没有就返回全 0。"""
        for nm in names:
            if nm in contrib:
                arr = np.asarray(contrib[nm], dtype=float)
                if arr.shape == shape:
                    return arr
        return np.zeros(shape, dtype=float)

    for entry in entries:
        c0 = int(entry.center_index)
        coords = np.asarray(entry.coordinates, dtype=float)
        n_atoms = coords.shape[0]
        if not (0 <= c0 < n_atoms):
            if verbose:
                print(f"[bonded][WARN] entry={entry.index} centre index {c0} out of range (n_atoms={n_atoms})")
            continue

        # box：没有就给个超大盒，等价于不考虑 PBC
        if entry.box is not None:
            box = np.asarray(entry.box, dtype=float)
        else:
            box = np.array([1e6, 1e6, 1e6], dtype=float)

        try:
            # 1) 从完整 summary 推拓扑（这里可能会看到很多远端的 bond/angle/dihedral）
            top = cf.infer_topology_from_summary(summary, coords, entry.atom_types)

            # 2) 只保留含中心原子的项，丢掉其它：避免边缘缺参数类型触发错误
            # _filter_bonded_topology(top, c0)

            # 3) 计算 bonded 力；不算非键
            contrib = cf.compute_forces(
                top,
                coords,
                box,
                enable_nb=False,
                enable_bond=False,
                enable_angle=False,
                enable_dih=True,
                split_nb=True,
            )
        except Exception as exc:
            if verbose:
                print(f"[bonded][WARN] entry={entry.index} gro={entry.gro_file} failed: {exc}", flush=True)
            # 出错就把该 entry 的 bonded 分量清零
            entry.bond_force = np.zeros(3, float)
            entry.angle_force = np.zeros(3, float)
            entry.dihedral_force = np.zeros(3, float)
            entry.bonded_force = np.zeros(3, float)
            continue

        shape = coords.shape

        Fbond = np.zeros(shape, dtype=float)  # bond 全部来自 CSV
        Fang  = np.zeros(shape, dtype=float)  # angle 全部来自 CSV
        Fdih  = _get_component(contrib, ("DIH", "DIHEDRAL", "Dihedral", "dih", "dihedral"), shape)

        entry.bond_force     = Fbond[c0] if Fbond.size else np.zeros(3, float)
        entry.angle_force    = Fang[c0]  if Fang.size  else np.zeros(3, float)
        entry.dihedral_force = Fdih[c0]  if Fdih.size else np.zeros(3, float)
        entry.bonded_force   = entry.bond_force + entry.angle_force + entry.dihedral_force

        if verbose:
            print(
                f"[bonded] entry={entry.index} centre={c0} | "
                f"Fb={entry.bond_force} Fa={entry.angle_force} Fd={entry.dihedral_force}",
                flush=True,
            )


def bfs_topological_distances(adjacency: NDArray[np.int32], start: int) -> NDArray[np.int32]:
    """Compute shortest topological distances from ``start`` using BFS."""

    n_atoms = adjacency.shape[0]
    distances = np.full(n_atoms, np.iinfo(np.int32).max, dtype=np.int32)
    distances[start] = 0
    queue: List[int] = [start]
    head = 0
    while head < len(queue):
        current = queue[head]
        head += 1
        neighbours = np.nonzero(adjacency[current])[0]
        for nxt in neighbours:
            if distances[nxt] > distances[current] + 1:
                distances[nxt] = distances[current] + 1
                queue.append(int(nxt))
    return distances


def validate_and_fix_entry(
    entry: MutableMapping[str, object],
    entry_idx: int,
    element_map: Mapping[str, str],
) -> ProcessedEntry:
    """Validate a raw entry and convert it into :class:`ProcessedEntry`."""

    gro = entry.get("gro_file")
    atom_types = entry.get("atom_types")
    coordinates = entry.get("coordinates")
    adjacency = entry.get("adj_matrix")
    force_label = entry.get("center_force_kBT_per_nm")

    if atom_types is None or coordinates is None or adjacency is None or force_label is None:
        raise ValueError(
            f"Entry {entry_idx} (gro={gro}) is missing required fields"
        )

    atom_types_list = list(atom_types)
    n_atoms = len(atom_types_list)
    coords = np.asarray(coordinates, dtype=float)
    if coords.shape != (n_atoms, 3):
        raise ValueError(
            f"Entry {entry_idx} (gro={gro}) has mismatched coordinate shape {coords.shape} for {n_atoms} atoms"
        )

    adjacency_array = np.asarray(adjacency, dtype=int)
    if adjacency_array.shape != (n_atoms, n_atoms):
        raise ValueError(
            f"Entry {entry_idx} (gro={gro}) has invalid adjacency matrix shape {adjacency_array.shape}"
        )

    if not np.all(adjacency_array == adjacency_array.T):
        raise ValueError(
            f"Entry {entry_idx} (gro={gro}) adjacency matrix must be symmetric"
        )

    charges_list = entry.get("formal_charges")
    charges_array: Optional[NDArray[np.float64]]
    if charges_list is not None:
        charges_array = np.asarray(charges_list, dtype=float)
        if charges_array.shape != (n_atoms,):
            raise ValueError(
                f"Entry {entry_idx} (gro={gro}) has formal_charges length {charges_array.shape} expected {n_atoms}"
            )
    else:
        charges_array = None

    if len(force_label) != 3:
        raise ValueError(
            f"Entry {entry_idx} (gro={gro}) has invalid force_label length {len(force_label)}"
        )
    target_force = np.asarray(force_label, dtype=float)

    box = entry.get("box")
    box_array: Optional[NDArray[np.float64]]
    if box is not None:
        box_array = np.asarray(box, dtype=float)
        if box_array.shape != (3,):
            raise ValueError(
                f"Entry {entry_idx} (gro={gro}) has invalid box specification {box_array}"
            )
    else:
        box_array = None

    center_index = infer_center_index_if_needed(entry, coords, box_array, entry_idx, gro)

    elements = determine_element_labels(atom_types_list, adjacency_array.astype(np.int32), element_map)

    distances = bfs_topological_distances(adjacency_array.astype(np.int32), center_index)

    zero_vec = np.zeros(3, dtype=float)

    return ProcessedEntry(
        index=entry_idx,
        gro_file=gro,
        atom_types=atom_types_list,
        elements=elements,
        coordinates=coords,
        charges=charges_array,
        adjacency=adjacency_array.astype(np.int32),
        center_index=center_index,
        target_force=target_force,
        box=box_array,
        topological_distances=distances,
        coulomb_force=zero_vec.copy(),
        bond_force=zero_vec.copy(),
        angle_force=zero_vec.copy(),
        dihedral_force=zero_vec.copy(),
        bonded_force=zero_vec.copy(),
    )

@njit
def _mic_disp(disp, box):
    """
    Minimum image convention for a single displacement vector.
    box: 长度为3的一维数组; 若<=0视为无PBC
    """
    out = disp.copy()
    for k in range(3):
        L = box[k]
        if L > 0.0:
            # 使用四舍五入的最近镜像
            n = round(out[k] / L)
            out[k] -= n * L
    return out


@njit
def _enumerate_pairs_numeric(
    centre,
    coords,
    box,
    topo_distances,
    charges,
    ke,
    fudge_lj,
    fudge_qq,
    max_pairs
):
    """
    只做数值计算的核心：
      - 跳过 1-2 / 1-3
      - MIC
      - 算 distance, coeff_alpha, coeff_beta, coulomb_vec
    返回:
      count,
      neighbour_idx[count],
      distances[count],
      coeff_alpha[count,3],
      coeff_beta[count,3],
      coulomb[count,3],
      topo_dist[count]
    不涉及字符串/dict/对象，方便 njit。
    """
    n_atoms = coords.shape[0]
    neighbour_idx = np.empty(n_atoms, dtype=np.int64)
    distances = np.empty(n_atoms, dtype=np.float64)
    coeff_alpha = np.empty((n_atoms, 3), dtype=np.float64)
    coeff_beta = np.empty((n_atoms, 3), dtype=np.float64)
    coulomb = np.zeros((n_atoms, 3), dtype=np.float64)
    topo_out = np.empty(n_atoms, dtype=np.int64)

    centre_coord = coords[centre]

    has_charges = charges is not None

    count = 0
    for idx in range(n_atoms):
        if idx == centre:
            continue

        topo_distance = int(topo_distances[idx])
        if topo_distance <= 0:
            continue
        if topo_distance <= 2:  # 排除 1-2 和 1-3
            continue

        scale_lj = fudge_lj if topo_distance == 3 else 1.0
        scale_qq = fudge_qq if topo_distance == 3 else 1.0

        disp = coords[idx] - centre_coord
        disp = _mic_disp(disp, box)

        r2 = disp[0]*disp[0] + disp[1]*disp[1] + disp[2]*disp[2]
        if r2 == 0.0:
            continue
        r = math.sqrt(r2)
        inv_r = 1.0 / r
        r_hat0 = disp[0] * inv_r
        r_hat1 = disp[1] * inv_r
        r_hat2 = disp[2] * inv_r

        inv_r2 = inv_r * inv_r
        inv_r6 = inv_r2 * inv_r2 * inv_r2
        inv_r12 = inv_r6 * inv_r6

        # 这两个系数的定义要和你原来 enumerate_valid_pairs 里一致
        # alpha 对应 12-term，beta 对应 6-term，符号根据你原来写法
        c_alpha = -12.0 * inv_r12 * scale_lj
        c_beta = 6.0 * inv_r6 * scale_lj

        coeff_alpha[count, 0] = c_alpha * r_hat0
        coeff_alpha[count, 1] = c_alpha * r_hat1
        coeff_alpha[count, 2] = c_alpha * r_hat2

        coeff_beta[count, 0] = c_beta * r_hat0
        coeff_beta[count, 1] = c_beta * r_hat1
        coeff_beta[count, 2] = c_beta * r_hat2

        if has_charges:
            q_c = charges[centre]
            q_n = charges[idx]
            pref = ke * q_c * q_n * inv_r2 * scale_qq
            coulomb[count, 0] = pref * r_hat0
            coulomb[count, 1] = pref * r_hat1
            coulomb[count, 2] = pref * r_hat2

        neighbour_idx[count] = idx
        distances[count] = r
        topo_out[count] = topo_distance

        count += 1
        if max_pairs is not None and max_pairs > 0 and count >= max_pairs:
            break

    return (
        count,
        neighbour_idx,
        distances,
        coeff_alpha,
        coeff_beta,
        coulomb,
        topo_out,
    )

def enumerate_valid_pairs(
    entry: ProcessedEntry,
    ke: float,
    fudge_lj: float,
    fudge_qq: float,
    max_pairs: Optional[int] = None,
) -> List[Mapping[str, object]]:
    """Enumerate valid centre-neighbour pairs for an entry (Numba加速数值部分)."""

    centre = int(entry.center_index)
    coords = np.asarray(entry.coordinates, dtype=np.float64)
    topo = np.asarray(entry.topological_distances, dtype=np.int64)

    if entry.box is not None:
        box = np.asarray(entry.box, dtype=np.float64)
    else:
        box = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # 0 => 不做 MIC

    charges = None
    if entry.charges is not None:
        charges = np.asarray(entry.charges, dtype=np.float64)

    (
        count,
        neighbour_idx,
        distances,
        coeff_alpha,
        coeff_beta,
        coulomb,
        topo_out,
    ) = _enumerate_pairs_numeric(
        centre,
        coords,
        box,
        topo,
        charges,
        ke,
        fudge_lj,
        fudge_qq,
        -1 if max_pairs is None else max_pairs,
    )

    pairs: List[Mapping[str, object]] = []
    for k in range(count):
        idx = int(neighbour_idx[k])
        # 保持你原先 pair_type 的定义方式
        t_c = entry.elements[centre]
        t_n = entry.elements[idx]
        pair_type = (t_c, t_n) if t_c <= t_n else (t_n, t_c)

        pairs.append(
            {
                "pair_type": pair_type,
                "distance": float(distances[k]),
                "coeff_alpha": coeff_alpha[k].copy(),
                "coeff_beta": coeff_beta[k].copy(),
                "coulomb": coulomb[k].copy(),
                "neighbour_index": idx,
                "topo_distance": int(topo_out[k]),
            }
        )

    # 再做一次最邻近截断，逻辑不变
    if max_pairs is not None and len(pairs) > max_pairs:
        pairs.sort(key=lambda item: item["distance"])
        pairs = pairs[:max_pairs]

    return pairs


def build_pair_types(
    entries: Sequence[ProcessedEntry],
    ke: float,
    fudge_lj: float,
    fudge_qq: float,
    max_pairs_per_entry: Optional[int],
) -> Tuple[List[Tuple[str, str]], List[List[Mapping[str, object]]]]:
    """Collect unique pair types and per-entry pair data."""

    all_pair_types: Dict[Tuple[str, str], None] = {}
    entry_pairs: List[List[Mapping[str, object]]] = []

    iterator: Iterable[ProcessedEntry]
    if tqdm is not None:
        iterator = tqdm(entries, desc="Enumerating pairs", leave=False)
    else:
        iterator = entries

    for entry in iterator:
        pairs = enumerate_valid_pairs(entry, ke=ke, fudge_lj=fudge_lj, fudge_qq=fudge_qq, max_pairs=max_pairs_per_entry)
        entry_pairs.append(pairs)
        for pair in pairs:
            all_pair_types.setdefault(pair["pair_type"], None)
        # accumulate Coulomb contributions
        if pairs:
            coulomb_total = np.sum([pair["coulomb"] for pair in pairs], axis=0)
        else:
            coulomb_total = np.zeros(3, dtype=float)
        entry.coulomb_force = coulomb_total

    pair_type_list = sorted(all_pair_types.keys())
    return pair_type_list, entry_pairs


def build_design_matrix(
    entries: Sequence[ProcessedEntry],
    entry_pairs: Sequence[Sequence[Mapping[str, object]]],
    pair_type_list: Sequence[Tuple[str, str]],
    fit_dihedral: bool = False,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], Dict[Tuple[str, str], int], Optional[int]]:
    """Construct the linear system ``A @ theta ≈ y``."""
  
    n_entries = len(entries)
    n_pair_types = len(pair_type_list)
    base_cols = 2 * n_pair_types
    extra_dih_cols = 1 if fit_dihedral else 0
    rows = 3 * n_entries
    cols = base_cols + extra_dih_cols

    design = np.zeros((rows, cols), dtype=float)
    targets = np.zeros(rows, dtype=float)

    pair_index = {pair: idx for idx, pair in enumerate(pair_type_list)}
    dih_col_idx = base_cols if fit_dihedral else None

    for entry_idx, entry in enumerate(entries):
        row_base = 3 * entry_idx

        if fit_dihedral:
            # 只减去 Coulomb + bond + angle，把 dihedral 留给线性未知量
            target_force = entry.target_force - entry.coulomb_force - entry.bond_force - entry.angle_force
        else:
            # 原逻辑：把 bonded 全部减掉（含二面角）
            target_force = entry.target_force - entry.coulomb_force - entry.bonded_force

        targets[row_base:row_base + 3] = target_force

        # LJ 部分
        for pair in entry_pairs[entry_idx]:
            col_idx = pair_index[pair["pair_type"]]
            alpha_col = 2 * col_idx
            beta_col = alpha_col + 1
            design[row_base:row_base + 3, alpha_col] += pair["coeff_alpha"]
            design[row_base:row_base + 3, beta_col] += pair["coeff_beta"]

        # Dihedral 缩放参数列：基函数 = 目前的 dihedral_force
        if fit_dihedral and dih_col_idx is not None:
            design[row_base:row_base + 3, dih_col_idx] = entry.dihedral_force

    return design, targets, pair_index, dih_col_idx

def build_prior_from_summary(
    pair_type_list: Sequence[Tuple[str, str]],
    summary: Optional[dict],
) -> Optional[NDArray[np.float64]]:
    """
    根据 summary 里的原始 LJ 参数构造 theta0:
        theta = [alpha_ij, beta_ij] 按 pair_type_list 顺序展开。

    支持两种常见结构:

    1) dict 形式:
        summary["nonbond_params"][atomtype]["sigma"/"epsilon"]

    2) list 形式:
        summary["nonbond_params"] = [
            {"name" 或 "atom_type" 或 "type": "...", "sigma": ..., "epsilon": ...},
            ...
        ]
    """
    if summary is None:
        return None

    nb = summary.get("nonbond_params") or summary.get("lj")
    if nb is None:
        return None

    # ---- 统一成 {atomtype: {sigma:.., epsilon:..}} 的 dict ----
    from collections.abc import Mapping

    if isinstance(nb, Mapping):
        nb_dict = nb
    elif isinstance(nb, list):
        nb_dict: Dict[str, Dict[str, float]] = {}
        for item in nb:
            if not isinstance(item, Mapping):
                continue
            # 常见几种 key 里挑一个作为类型名
            key = (
                item.get("name")
                or item.get("atom_type")
                or item.get("type")
                or item.get("id")
            )
            if key is None:
                continue
            if "sigma" in item and "epsilon" in item:
                try:
                    nb_dict[str(key)] = {
                        "sigma": float(item["sigma"]),
                        "epsilon": float(item["epsilon"]),
                    }
                except (TypeError, ValueError):
                    continue
        nb_dict = nb_dict
    else:
        # 不认识的格式，放弃使用先验
        return None

    if not nb_dict:
        return None

    prior: list[float] = []
    for at_i, at_j in pair_type_list:
        p_i = nb_dict.get(at_i)
        p_j = nb_dict.get(at_j)
        if not p_i or not p_j:
            # 没有完整信息就退回 0，不强加先验
            prior.extend([0.0, 0.0])
            continue

        sigma_i = float(p_i["sigma"])
        epsilon_i = float(p_i["epsilon"])
        sigma_j = float(p_j["sigma"])
        epsilon_j = float(p_j["epsilon"])

        # Lorentz–Berthelot
        sigma_ij = 0.5 * (sigma_i + sigma_j)
        # 防止奇怪的负/零 epsilon
        if epsilon_i <= 0.0 or epsilon_j <= 0.0 or sigma_ij <= 0.0:
            prior.extend([0.0, 0.0])
            continue

        epsilon_ij = math.sqrt(epsilon_i * epsilon_j)

        alpha_ij = 4.0 * epsilon_ij * (sigma_ij ** 12)
        beta_ij  = 4.0 * epsilon_ij * (sigma_ij ** 6)

        prior.extend([alpha_ij, beta_ij])

    return np.asarray(prior, dtype=float)


def solve_least_squares(
    design: NDArray[np.float64],
    targets: NDArray[np.float64],
    solver: str,
    alpha: float,
    nonneg: bool,
    prior: Optional[NDArray[np.float64]] = None,
) -> Tuple[NDArray[np.float64], Dict[str, object]]:
    """
    解最小二乘问题，并带可选的岭回归正则:
        min ||A θ - y||^2 + alpha ||θ - prior||^2

    如果 prior 为 None，则退化为对 θ 本身的 L2 正则（原来的行为）。
    """
    A = np.asarray(design, dtype=float)
    y = np.asarray(targets, dtype=float)
    m, n = A.shape

    if y.shape[0] != m:
        raise ValueError(f"design/targets size mismatch: {A.shape} vs {y.shape}")

    # --- 岭回归：在归一化前直接做“伪观测”拼接，保证是在原始坐标系里惩罚 ---
    if alpha > 0.0:
        sqrt_alpha = math.sqrt(alpha)
        if prior is None:
            prior_vec = np.zeros(n, dtype=float)
        else:
            prior_vec = np.asarray(prior, dtype=float)
            if prior_vec.shape[0] != n:
                raise ValueError(
                    f"prior length mismatch: got {prior_vec.shape[0]}, expected {n}"
                )
        # 拼接伪数据点：sqrt(alpha) * (θ - prior) -> A_reg θ ≈ y_reg
        # A_reg = sqrt(alpha) * I, y_reg = sqrt(alpha) * prior
        A_reg = sqrt_alpha * np.eye(n, dtype=float)
        y_reg = sqrt_alpha * prior_vec
        A = np.vstack([A, A_reg])
        y = np.concatenate([y, y_reg])

    # --- 正常归一化 + 解 ---
    A_use, y_use, col_scale, y_scale = _normalise_system(A, y)

    diagnostics: Dict[str, object] = {
        "solver": solver,
        "alpha": float(alpha),
        "nonneg": bool(nonneg),
        "condition": None,
    }

    if solver == "qr":
        # 普通最小二乘
        Q, R = np.linalg.qr(A_use, mode="reduced")
        theta_norm = np.linalg.solve(R, Q.T @ y_use)

    elif solver == "ridge":
        # 上面已经把 ridge 变成伪观测拼进 A, y 了，这里就跟 qr 一样解
        Q, R = np.linalg.qr(A_use, mode="reduced")
        theta_norm = np.linalg.solve(R, Q.T @ y_use)

    elif solver == "nnls":
        # 非负约束：用 nnls 解归一化后的系统
        if not nonneg:
            raise ValueError("nnls solver requires nonneg=True")
        theta_norm, _ = scipy.optimize.nnls(A_use, y_use)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # 反归一化
    theta = _denormalise_theta(theta_norm, col_scale, y_scale)

    # 非负裁剪（如果要求 nonneg，且不是 nnls 求出来的）
    if nonneg and solver != "nnls":
        theta = np.maximum(theta, 0.0)

    # 一点诊断信息
    try:
        ATA = A_use.T @ A_use
        diagnostics["condition"] = float(
            np.linalg.cond(ATA)
        )
    except Exception:
        diagnostics["condition"] = None

    return theta, diagnostics


def evaluate_fit(
    design: NDArray[np.float64],
    targets: NDArray[np.float64],
    theta: NDArray[np.float64],
    entries: Sequence[ProcessedEntry], fit_dihedral: bool = False, dih_scale: float = 1.0
) -> Dict[str, object]:
    """Compute residual statistics and per-entry diagnostics."""

    predictions = design @ theta if design.size else np.zeros_like(targets)
    residuals = predictions - targets

    mse = float(np.mean(residuals ** 2))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(residuals)))

    mean_target = float(np.mean(targets))
    sst = float(np.sum((targets - mean_target) ** 2))
    sse = float(np.sum((targets - predictions) ** 2))
    if sst > 0:
        r2 = 1.0 - sse / sst
    else:
        r2 = 1.0 if sse == 0 else float("nan")

    per_entry: List[Dict[str, object]] = []
    for idx, entry in enumerate(entries):
        row_base = 3 * idx
        entry_pred = predictions[row_base:row_base + 3]
        entry_target = targets[row_base:row_base + 3]
        if fit_dihedral:
            total_pred = (
                entry_pred              # = LJ_fit + s_dih * F_dih_base
                + entry.coulomb_force
                + entry.bond_force
                + entry.angle_force
            )
        else:
            total_pred = entry_pred + entry.coulomb_force + entry.bonded_force
        # total_pred = entry_pred + entry.coulomb_force + entry.bonded_force
        total_target = entry.target_force
        entry_residual = total_pred - total_target
        per_entry.append(
            {
                "entry_idx": entry.index,
                "gro_file": entry.gro_file,
                "rmse": float(np.linalg.norm(entry_residual) / math.sqrt(3.0)),
                "mae": float(np.mean(np.abs(entry_residual))),
                "predicted_total_force": total_pred.tolist(),
                "predicted_lj_force": entry_pred.tolist(),
                "coulomb_force": entry.coulomb_force.tolist(),
                "bond_force": entry.bond_force.tolist(),
                "angle_force": entry.angle_force.tolist(),
                "dihedral_force": entry.dihedral_force.tolist(),
                "target_force": entry.target_force.tolist(),
            }
        )

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "per_entry": per_entry,
    }
    return metrics

def fit_single_entry(
    entry: ProcessedEntry,
    pairs: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
    solver: str,
    alpha: float,
    nonneg: bool,
    fix_hh_zero: bool = False,
    sigma_bounds: Optional[Tuple[float, float]] = None,
    epsilon_bounds: Optional[Tuple[float, float]] = None,
    sigma_soft_weight: float = 0.0,
    epsilon_soft_weight: float = 0.0, 
) -> Dict[str, object]:
    """
    对单个 entry 拟合一组局部 alpha/beta，并应用与 joint 类似的约束逻辑：

      - 线性最小二乘解出 pair-level alpha/beta
      - 若 fix_hh_zero: 含 H_h 的 pair 的 alpha/beta 强制为 0
      - 若给出 sigma/epsilon 的软范围参数，则对该 entry 的解调用 recover_sigma_epsilon
        得到 per-entry 的元素 sigma/epsilon（与 joint 同一套软限制& H_h 处理）

    注意：这里和 joint 一样，sigma/epsilon 的软限制只作用在
          recover_sigma_epsilon 这一步（非线性 post-process），
          不会反向修改线性解本身 —— 和全局版本保持一致。
    """

    # 没有 pair 就直接返回空
    if not pairs:
        return {
            "entry_idx": entry.index,
            "gro_file": entry.gro_file,
            "pair_types": [],
            "parameters": {"alpha": {}, "beta": {}},
            "metrics": {"rmse": None, "mae": None, "r2": None, "detail": {}},
            "solver_diagnostics": {},
            "design_matrix_shape": [3, 0],
            "recover_sigma_epsilon": None,
        }

    # 收集本 entry 的 pair types
    pair_types = sorted({pair["pair_type"] for pair in pairs})
    n_types = len(pair_types)

    design = np.zeros((3, 2 * n_types), dtype=float)
    pair_index = {pair: idx for idx, pair in enumerate(pair_types)}

    # 组装设计矩阵：和 joint 一致，列是 [alpha_T, beta_T]
    for pair in pairs:
        idx = pair_index[pair["pair_type"]]
        alpha_col = 2 * idx
        beta_col = alpha_col + 1
        design[:, alpha_col] += pair["coeff_alpha"]
        design[:, beta_col] += pair["coeff_beta"]

    # 目标：和 joint（未 fit_dihedral 时）一致，先减掉 Coulomb + bonded
    targets = entry.target_force - entry.coulomb_force - entry.bonded_force

    # 正则化 + 线性求解
    A_use, y_use, s_col, s_y = _normalise_system(design, targets)
    prior = build_prior_from_summary(pair_types, summary)  # summary_data 是你 main() 里读的 summary json

    theta_norm, diagnostics = solve_least_squares(
        A_use, y_use, solver, alpha, nonneg, prior=prior
    )   
    # theta_norm, diagnostics = solve_least_squares(A_use, y_use, solver, alpha, nonneg)
    theta = _denormalise_theta(theta_norm, s_col, s_y)

    # 约束 1：含 H_h 的 pair -> alpha=beta=0（可选）
    if fix_hh_zero:
        for pair, idx in pair_index.items():
            if "H_h" in pair:
                theta[2 * idx] = 0.0
                theta[2 * idx + 1] = 0.0

    # 计算这个 entry 的拟合质量（用 total force 比较）
    metrics = evaluate_fit(design, targets, theta, [entry])

    # 导出本 entry 的 pair-level 参数
    parameters = {
        "alpha": {"-".join(pair): float(theta[2 * idx]) for idx, pair in enumerate(pair_types)},
        "beta": {"-".join(pair): float(theta[2 * idx + 1]) for idx, pair in enumerate(pair_types)},
    }

    # 约束 2：对本 entry 的解做一次 sigma/epsilon 恢复（soft bound + H_h 处理）
    recovered = None
    if sigma_bounds is not None or epsilon_bounds is not None or fix_hh_zero:
        recovered = recover_sigma_epsilon(
            pair_types,
            theta,
            pair_index,
            verbose=False,
            sigma_bounds=sigma_bounds,
            epsilon_bounds=epsilon_bounds,
            sigma_soft_weight=sigma_soft_weight,
            epsilon_soft_weight=epsilon_soft_weight,
            fix_hh_zero=fix_hh_zero,
        )

    # 把该 entry 的 metrics 摘出来塞进结果里
    detail = metrics.get("per_entry", [])
    per_entry_metrics = detail[0] if detail else {}

    return {
        "entry_idx": entry.index,
        "gro_file": entry.gro_file,
        "pair_types": [list(pair) for pair in pair_types],
        "parameters": parameters,
        "metrics": {
            "rmse": metrics.get("rmse"),
            "mae": metrics.get("mae"),
            "r2": metrics.get("r2"),
            "detail": per_entry_metrics,
        },
        "solver_diagnostics": diagnostics,
        "design_matrix_shape": list(design.shape),
        "recover_sigma_epsilon": recovered,
    }

def recover_sigma_epsilon(
    pair_type_list: Sequence[Tuple[str, str]],
    theta: np.ndarray,
    pair_index: Mapping[Tuple[str, str], int],
    *,
    fix_hh_zero: bool = False,
    sigma_bounds: Optional[Tuple[float, float]] = None,
    epsilon_bounds: Optional[Tuple[float, float]] = None,
    sigma_soft_weight: float = 0.0,
    epsilon_soft_weight: float = 0.0,
    verbose: bool = False,
) -> Optional[Dict[str, object]]:
    """
    从 pair-level alpha/beta 恢复 per-element sigma/epsilon (Lorentz–Berthelot 规则)。

    特性:
      - fix_hh_zero=True 时:
          * 含 H_h 的 pair 不用于拟合
          * 输出中 H_h 的 sigma=0, epsilon=0
      - sigma_bounds / epsilon_bounds:
          * 在 log 空间作为 hard bounds: 在物理范围内拟合
      - sigma_soft_weight / epsilon_soft_weight:
          * 可选软惩罚，偏离范围时增加额外 residual（长度固定，不引发 shape 问题）

    用在 joint 和 per-entry 都是同一逻辑。
    """
    if not pair_type_list:
        return None

    # 1. 收集 pair-level alpha/beta
    alpha_values: Dict[Tuple[str, str], float] = {}
    beta_values: Dict[Tuple[str, str], float] = {}
    for pair, idx in pair_index.items():
        alpha_values[pair] = float(theta[2 * idx])
        beta_values[pair] = float(theta[2 * idx + 1])

    # 2. 元素集合（如启用 fix_hh_zero，就把 H_h 从拟合变量里拿掉）
    all_elems = sorted({elem for pair in pair_type_list for elem in pair})
    if not all_elems:
        return None

    elements: List[str] = []
    for elem in all_elems:
        if fix_hh_zero and elem == "H_h":
            continue
        elements.append(elem)

    if not elements:
        if verbose:
            print("[recover_sigma_epsilon] No elements left to fit after H_h removal", flush=True)
        return None

    element_to_index = {elem: i for i, elem in enumerate(elements)}

    # 3. 有效 pair: 用于初始猜和残差
    valid_pairs: List[Tuple[str, str]] = []
    for pair, a in alpha_values.items():
        b = beta_values[pair]
        # 跳过含 H_h 的 pair（这些我们逻辑上认为 LJ=0）
        if fix_hh_zero and ("H_h" in pair):
            continue
        # 只接受 alpha>0, beta>0 的 pair
        if a <= 0.0 or b <= 0.0:
            continue
        valid_pairs.append(pair)

    if not valid_pairs:
        if verbose:
            print("[recover_sigma_epsilon] No valid pairs with positive alpha/beta; skip", flush=True)
        return None

    # 4. 初始猜测: 从 valid_pairs 反推 sigma_ij, epsilon_ij 的平均
    sigma_guess: Dict[str, float] = {elem: 3.0 for elem in elements}
    epsilon_guess: Dict[str, float] = {elem: 0.1 for elem in elements}
    sig_est: Dict[str, List[float]] = {elem: [] for elem in elements}
    eps_est: Dict[str, List[float]] = {elem: [] for elem in elements}

    for pair in valid_pairs:
        a = alpha_values[pair]
        b = beta_values[pair]
        # 这里已保证 a>0, b>0
        ratio = a / b
        if ratio <= 0.0:
            continue
        sigma_ij = ratio ** (1.0 / 6.0)
        if sigma_ij <= 0.0:
            continue
        epsilon_ij = b / (4.0 * (sigma_ij ** 6))
        if epsilon_ij <= 0.0 or not np.isfinite(epsilon_ij):
            continue

        for elem in pair:
            if elem in sig_est:
                sig_est[elem].append(sigma_ij)
            if elem in eps_est:
                eps_est[elem].append(epsilon_ij)

    for elem in elements:
        if sig_est[elem]:
            sigma_guess[elem] = float(np.mean(sig_est[elem]))
        if eps_est[elem]:
            epsilon_guess[elem] = float(max(np.mean(eps_est[elem]), 1e-8))

    # 5. 在 log 空间构造初值和 bounds
    n = len(elements)
    x0 = np.zeros(2 * n, dtype=float)
    for i, elem in enumerate(elements):
        x0[i] = math.log(max(sigma_guess[elem], 1e-6))
        x0[n + i] = math.log(max(epsilon_guess[elem], 1e-12))

    sigma_min, sigma_max = sigma_bounds if sigma_bounds is not None else (None, None)
    eps_min, eps_max = epsilon_bounds if epsilon_bounds is not None else (None, None)

    lb = np.full(2 * n, -np.inf, dtype=float)
    ub = np.full(2 * n, +np.inf, dtype=float)

    if sigma_min is not None:
        v = math.log(max(sigma_min, 1e-8))
        lb[:n] = np.maximum(lb[:n], v)
    if sigma_max is not None:
        v = math.log(sigma_max)
        ub[:n] = np.minimum(ub[:n], v)
    if eps_min is not None:
        v = math.log(max(eps_min, 1e-12))
        lb[n:] = np.maximum(lb[n:], v)
    if eps_max is not None:
        v = math.log(eps_max)
        ub[n:] = np.minimum(ub[n:], v)

    # clamp x0 into [lb, ub]
    for i in range(2 * n):
        if np.isfinite(lb[i]) and x0[i] < lb[i]:
            x0[i] = lb[i] + 1e-8
        if np.isfinite(ub[i]) and x0[i] > ub[i]:
            x0[i] = ub[i] - 1e-8

    # 6. 残差函数：长度固定
    def residuals(x: np.ndarray) -> np.ndarray:
        sigma = np.exp(x[:n])
        epsilon = np.exp(x[n:])

        res: List[float] = []

        # 6.1 pair-level alpha/beta 拟合残差
        for pair in valid_pairs:
            i_elem, j_elem = pair
            if i_elem not in element_to_index or j_elem not in element_to_index:
                continue

            i = element_to_index[i_elem]
            j = element_to_index[j_elem]

            sigma_ij = 0.5 * (sigma[i] + sigma[j])
            epsilon_ij = math.sqrt(epsilon[i] * epsilon[j])

            alpha_ij = 4.0 * epsilon_ij * (sigma_ij ** 12)
            beta_ij = 4.0 * epsilon_ij * (sigma_ij ** 6)

            res.append(alpha_ij - alpha_values[pair])
            res.append(beta_ij - beta_values[pair])

        # 6.2 软惩罚（可选；确保每个参数都有固定位置的 penalty 项）
        if sigma_soft_weight > 0.0 and (sigma_min is not None or sigma_max is not None):
            for s in sigma:
                under = 0.0 if sigma_min is None else max(0.0, sigma_min - s)
                over = 0.0 if sigma_max is None else max(0.0, s - sigma_max)
                res.append(sigma_soft_weight * (under + over))

        if epsilon_soft_weight > 0.0 and (eps_min is not None or eps_max is not None):
            for e in epsilon:
                under = 0.0 if eps_min is None else max(0.0, eps_min - e)
                over = 0.0 if eps_max is None else max(0.0, e - eps_max)
                res.append(epsilon_soft_weight * (under + over))

        return np.asarray(res, dtype=float)

    # 7. 有界最小二乘
    result = least_squares(residuals, x0, bounds=(lb, ub))
    if verbose and not result.success:
        print(f"[recover_sigma_epsilon] Warning: {result.message}", flush=True)

    sigma = np.exp(result.x[:n])
    epsilon = np.exp(result.x[n:])

    # 8. 组装输出
    elem_params: Dict[str, Dict[str, float]] = {}
    for elem, s_val, e_val in zip(elements, sigma, epsilon):
        elem_params[elem] = {
            "sigma": float(s_val),
            "epsilon": float(e_val),
        }

    if fix_hh_zero:
        # 明确给 H_h 置 0（如果它在整体元素列表里）
        if "H_h" in all_elems:
            elem_params["H_h"] = {"sigma": 0.0, "epsilon": 0.0}

    return {
        "elements": elem_params,
        "residual_norm": float(np.linalg.norm(residuals(result.x))),
        "status": int(result.status),
        "message": result.message,
    }



def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fit Lennard-Jones linear parameters from force data")
    parser.add_argument("--data", type=Path, help="Dataset JSON file", default=None)
    parser.add_argument("--summary", type=Path, help="Topology summary JSON", default=None)
    parser.add_argument("--output", type=Path, help="Output JSON file", default=None)
    parser.add_argument("--solver", choices=["ridge", "nnls", "qr"], default="ridge")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="L2 regularisation strength")
    parser.add_argument("--nonneg", action="store_true", help="Enforce non-negative parameters")
    parser.add_argument("--no-nonneg", action="store_false", dest="nonneg", help="Disable non-negativity constraint")
    parser.add_argument(
        "--fit-mode",
        choices=["joint", "per-entry", "both"],
        default="joint",
        help="Select whether to fit jointly across all entries, per entry, or both",
    )
    parser.add_argument("--recover-sigma-eps", action="store_true", help="Recover per-element sigma/epsilon")
    parser.add_argument("--element-map", type=Path, help="Optional atom-type to element mapping JSON")
    parser.add_argument("--fudgeLJ", type=float, default=DEFAULT_FUDGE_LJ, help="1-4 LJ scaling factor")
    parser.add_argument("--fudgeQQ", type=float, default=DEFAULT_FUDGE_QQ, help="1-4 Coulomb scaling factor")
    parser.add_argument("--ke", type=float, default=DEFAULT_KE, help="Coulomb constant")
    parser.add_argument("--max-pairs-per-entry", type=int, default=None, help="Limit pairs per entry for debugging")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (currently unused)")
    parser.add_argument(
        "--group-size",
        type=int,
        default=None,
        help="If set, also fit LJ parameters jointly for non-overlapping blocks of this many entries"
    )
    # 物理范围 & 特殊约束
    parser.add_argument("--fix-hh-zero", action="store_true",
                        help="Force hydroxyl H (H_h) to have sigma=0, epsilon=0 in LJ recovery")
    parser.add_argument("--sigma-min", type=float, default=2.0,
                        help="Soft lower bound for element sigma in LJ recovery (same units as sigma)")
    parser.add_argument("--sigma-max", type=float, default=5.0,
                        help="Soft upper bound for element sigma in LJ recovery")
    parser.add_argument("--epsilon-min", type=float, default=1e-4,
                        help="Soft lower bound for element epsilon in LJ recovery")
    parser.add_argument("--epsilon-max", type=float, default=10.0,
                        help="Soft upper bound for element epsilon in LJ recovery")
    parser.add_argument("--sigma-soft-weight", type=float, default=1.0,
                        help="Soft penalty weight for sigma bounds in LJ recovery")
    parser.add_argument("--epsilon-soft-weight", type=float, default=1.0,
                        help="Soft penalty weight for epsilon bounds in LJ recovery")

    # 二面角拟合相关
    parser.add_argument("--fit-dihedral", action="store_true",
                        help="Include a global scaling parameter for dihedral forces in the linear fit")
    parser.add_argument("--dih-soft-center", type=float, default=1.0,
                        help="Soft prior center for dihedral scaling parameter (only with --fit-dihedral)")
    parser.add_argument("--dih-soft-weight", type=float, default=0.1,
                        help="Soft prior weight for dihedral scaling (only with --fit-dihedral)")

    parser.set_defaults(nonneg=True)
    args = parser.parse_args(argv)

    if args.seed is not None:
        np.random.seed(args.seed)

    if args.data is None or args.summary is None:
        print("No data/summary provided; generating synthetic dataset for self-test", flush=True)
        synthetic = generate_synthetic_dataset()
        raw_data = synthetic["data"]
        summary = synthetic["summary"]
    else:
        raw_data, summary = load_data(args.data, args.summary)

    if args.element_map is not None:
        with args.element_map.open("r", encoding="utf-8") as fh:
            element_map = json.load(fh)
            if not isinstance(element_map, Mapping):
                raise ValueError("Element map JSON must be an object")
    else:
        element_map = {}

    entries: List[ProcessedEntry] = []
    for idx, entry in enumerate(raw_data):
        processed = validate_and_fix_entry(entry, idx, element_map)
        entries.append(processed)

    if args.verbose:
        print(f"Loaded {len(entries)} entries", flush=True)

    compute_bonded_forces(entries, summary, verbose=args.verbose)
    # 1) 从 CSV 读 bond / angle（scan_ff 的输出）
    bond_force_map  = load_force_map_from_csv("center_bond_forces.csv",  label="bond")
    angle_force_map = load_force_map_from_csv("center_angle_forces.csv", label="angle")

    # 2) 用 compare_forces 只算 dihedral
    # patch_missing_angle_params(summary)
    # dih_force_map   = compute_dihedral_forces_only(entries, summary, verbose=True)
    # 3) 把 bond/angle/dihedral 力写回每个 entry
    missing_bond = missing_angle = missing_dih = 0
    for entry in entries:
        key = (entry.gro_file, int(entry.center_index))

        # dihedral_force 保留 compute_bonded_forces 的结果
        Fd = entry.dihedral_force if entry.dihedral_force is not None else np.zeros(3, float)

        Fb = bond_force_map.get(key)
        if Fb is None:
            Fb = entry.bond_force if entry.bond_force is not None else np.zeros(3, float)
            missing_bond += 1

        Fa = angle_force_map.get(key)
        if Fa is None:
            Fa = entry.angle_force if entry.angle_force is not None else np.zeros(3, float)
            missing_angle += 1

        entry.bond_force = np.asarray(Fb, dtype=float)
        entry.angle_force = np.asarray(Fa, dtype=float)
        entry.dihedral_force = np.asarray(Fd, dtype=float)
        entry.bonded_force = entry.bond_force + entry.angle_force + entry.dihedral_force

    # if args.verbose:
    #     print(f"[assign] entries using fallback bond from cf:   {missing_bond}", flush=True)
    #     print(f"[assign] entries using fallback angle from cf:  {missing_angle}", flush=True)

    if args.verbose:
        print(f"[assign] entries without bond CSV:   {missing_bond}", flush=True)
        print(f"[assign] entries without angle CSV:  {missing_angle}", flush=True)
        print(f"[assign] entries without dih force:  {missing_dih}", flush=True)


    defaults_section = summary.get("defaults") if isinstance(summary, Mapping) else None
    fudge_lj = args.fudgeLJ
    fudge_qq = args.fudgeQQ
    if isinstance(defaults_section, Mapping):
        fudge_lj = float(defaults_section.get("fudgeLJ", fudge_lj))
        fudge_qq = float(defaults_section.get("fudgeQQ", fudge_qq))
    else:
        if isinstance(summary, Mapping) and "fudgeLJ" in summary:
            fudge_lj = float(summary["fudgeLJ"])
        if isinstance(summary, Mapping) and "fudgeQQ" in summary:
            fudge_qq = float(summary["fudgeQQ"])

    pair_types, entry_pairs = build_pair_types(entries, args.ke, fudge_lj, fudge_qq, args.max_pairs_per_entry)

    if args.verbose:
        print(f"Identified {len(pair_types)} pair types: {pair_types}", flush=True)

    fit_mode = args.fit_mode
    joint_enabled = fit_mode in {"joint", "both"}
    per_entry_enabled = fit_mode in {"per-entry", "both"}

    if args.recover_sigma_eps and not joint_enabled:
        raise ValueError("--recover-sigma-eps requires joint fitting (fit-mode joint or both)")

    bonded_source = "none"
    if isinstance(summary, Mapping) and any(summary.get(key) for key in ("bonds", "angles", "dihedrals")):
        bonded_source = "summary"

    output: Dict[str, object] = {
        "meta": {
            "solver": args.solver,
            "alpha": args.alpha,
            "nonneg": args.nonneg,
            "fudgeLJ": fudge_lj,
            "fudgeQQ": fudge_qq,
            "ke": args.ke,
            "entry_count": len(entries),
            "fit_mode": fit_mode,
            "bonded_force_source": bonded_source,
        },
        "pair_types": [list(pair) for pair in pair_types],
    }

    joint_theta: Optional[NDArray[np.float64]] = None
    pair_index: Dict[Tuple[str, str], int] = {}

    if joint_enabled:
        design, targets, pair_index, dih_col_idx = build_design_matrix(
            entries, entry_pairs, pair_types, fit_dihedral=args.fit_dihedral
        )

        # 若拟合 dihedral 缩放，加一个软先验：s_dih ~ N(center, 1/weight)
        if args.fit_dihedral and dih_col_idx is not None and args.dih_soft_weight > 0.0:
            w = math.sqrt(args.dih_soft_weight)
            prior_row = np.zeros((1, design.shape[1]), dtype=float)
            prior_row[0, dih_col_idx] = w
            design = np.vstack([design, prior_row])
            targets = np.concatenate([targets, np.array([w * args.dih_soft_center], dtype=float)])

        if args.verbose:
            print(f"Design matrix shape: {design.shape}", flush=True)
            try:
                condition_number = float(np.linalg.cond(design))
            except np.linalg.LinAlgError:
                condition_number = float("inf")
            print(f"Condition number: {condition_number:.3e}", flush=True)
            if not math.isfinite(condition_number) or condition_number > 1e12:
                print(
                    "Warning: Ill-conditioned design matrix; consider adding regularisation or more diverse data.",
                    flush=True,
                )
        A_use, y_use, s_col, s_y = _normalise_system(design, targets)
        prior = build_prior_from_summary(pair_types, summary)  # summary_data 是你 main() 里读的 summary json

        theta_norm, diagnostics = solve_least_squares(A_use, y_use, args.solver, args.alpha, args.nonneg, prior=prior)   
        # theta_norm, diagnostics = solve_least_squares(A_use, y_use, args.solver, args.alpha, args.nonneg)
        joint_theta = _denormalise_theta(theta_norm, s_col, s_y)
        if args.fit_dihedral and dih_col_idx is not None:
            s_dih = float(joint_theta[dih_col_idx])
            output["meta"]["dihedral_scale"] = s_dih

        # joint_theta, diagnostics = solve_least_squares(design, targets, args.solver, args.alpha, args.nonneg)
        metrics = evaluate_fit(design, targets, joint_theta, entries)

        if args.verbose:
            print(f"Fit RMSE: {metrics['rmse']:.6f}", flush=True)
            print(f"Fit MAE: {metrics['mae']:.6f}", flush=True)
            print(f"Fit R^2: {metrics['r2']:.6f}", flush=True)

        # 可选：强制羟基氢 H_h 的 LJ 为 0
        if args.fix_hh_zero:
            for pair, idx in pair_index.items():
                if "H_h" in pair:
                    joint_theta[2 * idx] = 0.0
                    joint_theta[2 * idx + 1] = 0.0

        output["parameters"] = {
            "alpha": {"-".join(pair): float(joint_theta[2 * idx]) for pair, idx in pair_index.items()},
            "beta": {"-".join(pair): float(joint_theta[2 * idx + 1]) for pair, idx in pair_index.items()},
        }
        output["metrics"] = metrics
        output["solver_diagnostics"] = diagnostics
        output["design_matrix_shape"] = list(design.shape)
    else:
        output["parameters"] = {"alpha": {}, "beta": {}}
        output["metrics"] = {}
        output["solver_diagnostics"] = {}

    per_entry_results: List[Dict[str, object]] = []
    if per_entry_enabled:
        if args.verbose and fit_mode == "per-entry":
            print("Performing per-entry fits", flush=True)
        for entry, pairs in zip(entries, entry_pairs):
            per_entry_results.append(
                fit_single_entry(
                    entry,
                    pairs,
                    args.solver,
                    args.alpha,
                    args.nonneg,
                    fix_hh_zero=args.fix_hh_zero,
                    sigma_bounds=(args.sigma_min, args.sigma_max),
                    epsilon_bounds=(args.epsilon_min, args.epsilon_max),
                    sigma_soft_weight=args.sigma_soft_weight,
                    epsilon_soft_weight=args.epsilon_soft_weight,
                    summary=summary
                )
            )

    if per_entry_results:
        output["per_entry_fits"] = per_entry_results
        output["meta"]["per_entry_fit_count"] = len(per_entry_results)
    else:
        output["meta"]["per_entry_fit_count"] = 0

    if args.recover_sigma_eps and joint_theta is not None:
        recovered = recover_sigma_epsilon(
            pair_types,
            joint_theta,
            pair_index,
            verbose=args.verbose,
            sigma_bounds=(args.sigma_min, args.sigma_max),
            epsilon_bounds=(args.epsilon_min, args.epsilon_max),
            sigma_soft_weight=args.sigma_soft_weight,
            epsilon_soft_weight=args.epsilon_soft_weight,
            fix_hh_zero=args.fix_hh_zero,
        )
        output["recover_sigma_epsilon"] = recovered

    # ====== 分组拟合：每 group-size 个 entry 拟合一组参数 ======
    group_fits: List[Dict[str, object]] = []
    if args.group_size is not None and args.group_size > 0:
        gsz = int(args.group_size)
        # 使用已有的 entry_pairs；按顺序每 gsz 个 entry 一组
        for start in range(0, len(entries), gsz):
            end = min(len(entries), start + gsz)
            sub_entries = entries[start:end]
            sub_pairs_list = entry_pairs[start:end]

            # 收集该组内出现过的 pair_type
            local_pair_types_dict: Dict[Tuple[str, str], None] = {}
            for pairs in sub_pairs_list:
                for p in pairs:
                    local_pair_types_dict.setdefault(p["pair_type"], None)
            local_pair_types = sorted(local_pair_types_dict.keys())
            if not local_pair_types:
                continue  # 这一组没有有效 pair，跳过

            # 用这一组的 entries + pair 构造线性系统
            design_g, targets_g, pair_index_g, dih_col_idx_g = build_design_matrix(
                sub_entries,
                sub_pairs_list,
                local_pair_types,
                fit_dihedral=args.fit_dihedral,
            )

            if design_g.size == 0:
                continue


            # 正则化 + 解
            A_g, y_g, s_col_g, s_y_g = _normalise_system(design_g, targets_g)
            prior = build_prior_from_summary(pair_types, summary)  # summary_data 是你 main() 里读的 summary json

            theta_norm_g, diagnostics = solve_least_squares(
                design,
                targets,
                solver=args.solver,
                alpha=args.alpha,
                nonneg=args.nonneg,
                prior=prior,
            )   
            # theta_norm_g, diag_g = solve_least_squares(
            #     A_g, y_g, args.solver, args.alpha, args.nonneg
            # )
            theta_g = _denormalise_theta(theta_norm_g, s_col_g, s_y_g)

            # 可选：强制含 H_h 的 pair 为 0
            if args.fix_hh_zero:
                for pair, idx in pair_index_g.items():
                    if "H_h" in pair:
                        theta_g[2 * idx] = 0.0
                        theta_g[2 * idx + 1] = 0.0

            # 拟合质量（用这一组内的 entries）
            metrics_g = evaluate_fit(
                design_g,
                targets_g,
                theta_g,
                sub_entries,
                fit_dihedral=(args.fit_dihedral and dih_col_idx_g is not None),
            )

            group_out: Dict[str, object] = {
                "group_index": len(group_fits),
                "entry_range": [int(start), int(end - 1)],
                "parameters": {
                    "alpha": {
                        "-".join(pair): float(theta_g[2 * idx])
                        for pair, idx in pair_index_g.items()
                    },
                    "beta": {
                        "-".join(pair): float(theta_g[2 * idx + 1])
                        for pair, idx in pair_index_g.items()
                    },
                },
                "metrics": metrics_g,
                "solver_diagnostics": diag_g,
                "design_matrix_shape": list(design_g.shape),
            }

            # 如开启 recover-sigma-eps，则对该组也恢复一组 σ/ε
            if args.recover_sigma_eps:
                recovered_g = recover_sigma_epsilon(
                    local_pair_types,
                    theta_g,
                    pair_index_g,
                    verbose=False,
                    sigma_bounds=(args.sigma_min, args.sigma_max),
                    epsilon_bounds=(args.epsilon_min, args.epsilon_max),
                    sigma_soft_weight=args.sigma_soft_weight,
                    epsilon_soft_weight=args.epsilon_soft_weight,
                    fix_hh_zero=args.fix_hh_zero,
                )
                group_out["recover_sigma_epsilon"] = recovered_g

            group_fits.append(group_out)

        output["group_fits"] = group_fits
        output["meta"]["group_fit_count"] = len(group_fits)
        output["meta"]["group_size"] = gsz
    # ====== 分组拟合结束 ======


    if args.output is not None:
        with args.output.open("w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2)
        if args.verbose:
            print(f"Wrote results to {args.output}", flush=True)
    else:
        json.dump(output, sys.stdout, indent=2)
        print("", flush=True)

    return 0


def generate_synthetic_dataset() -> Dict[str, object]:
    """Generate a small synthetic dataset for self-testing."""

    sigma = {"C": 3.5, "H": 2.5, "O": 3.0}
    epsilon = {"C": 0.1, "H": 0.05, "O": 0.2}

    def lj_force(center_elem: str, neighbour_elem: str, displacement: NDArray[np.float64]) -> NDArray[np.float64]:
        r = float(np.linalg.norm(displacement))
        if r == 0:
            return np.zeros(3)
        r_hat = displacement / r
        sigma_ij = 0.5 * (sigma[center_elem] + sigma[neighbour_elem])
        epsilon_ij = math.sqrt(epsilon[center_elem] * epsilon[neighbour_elem])
        alpha = 4.0 * epsilon_ij * (sigma_ij ** 12)
        beta = 4.0 * epsilon_ij * (sigma_ij ** 6)
        prefactor = (12.0 * alpha * (r ** -13) - 6.0 * beta * (r ** -7))
        return prefactor * r_hat

    entries: List[MutableMapping[str, object]] = []
    for i in range(10):
        atom_types = ["C", "H", "O"]
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1 + 0.02 * i, 0.05 * i, 0.0],
                [0.2 + 0.01 * i, -0.03 * i, 0.0],
            ]
        )
        centre = 0
        forces = np.zeros(3, dtype=float)
        for idx in range(1, 3):
            disp = coords[idx] - coords[centre]
            forces += lj_force(atom_types[centre], atom_types[idx], disp)
        entry = {
            "gro_file": f"synthetic_{i}.gro",
            "atom_types": atom_types,
            "coordinates": coords.tolist(),
            "formal_charges": [0.0, 0.0, -0.5],
            "adj_matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            "center_atom": {"atom_index": 1},
            "force_label": forces.tolist(),
            "box": [3.0, 3.0, 3.0],
        }
        entries.append(entry)

    summary = {"fudgeLJ": DEFAULT_FUDGE_LJ, "fudgeQQ": DEFAULT_FUDGE_QQ}
    return {"data": entries, "summary": summary}


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())

