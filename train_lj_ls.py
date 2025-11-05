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
import sys
from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy

from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

try:  # Optional but helpful progress bar
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None

from scipy.optimize import least_squares, lsq_linear, nnls


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
      1) 先读 index：center_force_index 或 center_atom.atom_index（按 0-based 处理）。
      2) 若缺失/越界，则回退到坐标推断：
         - entry.center_coord
         - entry.center_atom.xyz
         - 由 entry.center_atom.x/y/z 组装
      3) 推断失败则抛出详细错误。
    """
    n_atoms = int(len(coords))

    # --- 1) 优先使用提供的 index ---
    ci_raw = None
    if "center_force_index" in entry:
        ci_raw = int(entry["center_force_index"])
    elif "center_atom" in entry and isinstance(entry["center_atom"], dict):
        ca = entry["center_atom"]
        if "atom_index" in ca:
            ci_raw = int(ca["atom_index"])
            # 如果你的 atom_index 实际是 1-based，请取消下一行注释：
            # ci_raw -= 1

    if ci_raw is not None:
        if 0 <= ci_raw < n_atoms:
            return ci_raw
        raise IndexError(
            f"[entry={entry_idx}] Provided center index out of bounds: {ci_raw} / n_atoms={n_atoms}"
            + (f", gro={gro}" if gro else "")
        )

    # --- 2) 回退：坐标推断 ---
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
    这样既不会因为过滤参数模板而丢键角二面角，又能避免分子边缘缺参数的报错。
    """
    if not entries or not isinstance(summary, Mapping) or not summary:
        return
    try:
        import compare_forces as cf
    except Exception as exc:
        raise RuntimeError("compare_forces.py is required to compute bonded forces") from exc

    for entry in entries:
        c0 = int(entry.center_index)           # 0-based
        n_atoms = entry.coordinates.shape[0]
        try:
            # 1) 用完整 summary 建拓扑（不要先过滤 summary 的 bonds/angles/dihedrals！）
            top = cf.infer_topology_from_summary(summary, entry.coordinates, entry.atom_types)

            # 2) 过滤拓扑对象里真正的实例列表（top 里是 1-based 索引）
            def _keep_bond(b):      # b.i, b.j 是 1-based
                return (b.i - 1) == c0 or (b.j - 1) == c0

            def _keep_angle(a):     # a.i, a.j, a.k 是 1-based
                return (a.i - 1) == c0 or (a.j - 1) == c0 or (a.k - 1) == c0

            def _keep_dih(d):       # d.i, d.j, d.k, d.l 是 1-based
                z = (d.i - 1, d.j - 1, d.k - 1, d.l - 1)
                return c0 in z

            # 就地“瘦身”拓扑，只保留与中心原子相关的项
            top.bonds        = [b for b in getattr(top, "bonds", [])        if _keep_bond(b)]
            top.angles       = [a for a in getattr(top, "angles", [])       if _keep_angle(a)]
            top.rb_dihedrals = [d for d in getattr(top, "rb_dihedrals", []) if _keep_dih(d)]

            # 3) 仅算成键项；非键关掉（避免 pairs14 / 排除表之类副作用）
            box = entry.box if entry.box is not None else np.array([1e6, 1e6, 1e6], dtype=float)
            contrib = cf.compute_forces(
                top, entry.coordinates, box,
                enable_nb=False, enable_bond=True, enable_angle=True, enable_dih=True,
                split_nb=True,
            )

        except Exception as exc:
            # 出错就置零但不中断
            entry.bond_force[:] = 0.0
            entry.angle_force[:] = 0.0
            entry.dihedral_force[:] = 0.0
            entry.bonded_force[:] = 0.0
            if verbose:
                print(f"[bonded][WARN] entry={entry.index} centre={c0} failed: {exc}", flush=True)
            continue

        # 4) 取中心原子分量（兼容不同大小写键名）
        def _pick(*names: str) -> np.ndarray:
            for nm in names:
                if nm in contrib:
                    return np.asarray(contrib[nm], dtype=float)
            return np.zeros_like(entry.coordinates)

        Fbond = _pick("BOND", "bond", "Bond")
        Fang  = _pick("ANGLE", "angle", "Angle")
        Fdih  = _pick("DIH", "DIHEDRAL", "dih", "dihedral", "Dihedral")

        entry.bond_force     = Fbond[c0] if Fbond.size else np.zeros(3, float)
        entry.angle_force    = Fang[c0]  if Fang.size  else np.zeros(3, float)
        entry.dihedral_force = Fdih[c0]  if Fdih.size  else np.zeros(3, float)
        entry.bonded_force   = entry.bond_force + entry.angle_force + entry.dihedral_force

        if verbose:
            print(f"[bonded] entry={entry.index} centre={c0} | "
                  f"keep: bonds={len(getattr(top,'bonds',[]))}, "
                  f"angles={len(getattr(top,'angles',[]))}, "
                  f"dihs={len(getattr(top,'rb_dihedrals',[]))} | "
                  f"Fb={entry.bond_force} Fa={entry.angle_force} Fd={entry.dihedral_force}",
                  flush=True)


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


def enumerate_valid_pairs(
    entry: ProcessedEntry,
    ke: float,
    fudge_lj: float,
    fudge_qq: float,
    max_pairs: Optional[int] = None,
) -> List[Mapping[str, object]]:
    """Enumerate valid centre-neighbour pairs for an entry."""

    centre = entry.center_index
    coords = entry.coordinates
    centre_coord = coords[centre]
    n_atoms = coords.shape[0]

    pairs: List[Mapping[str, object]] = []
    for idx in range(n_atoms):
        if idx == centre:
            continue
        topo_distance = int(entry.topological_distances[idx])
        if topo_distance <= 0:
            continue
        if topo_distance <= 2:  # Exclude 1-2 and 1-3
            continue

        scale_lj = fudge_lj if topo_distance == 3 else 1.0
        scale_qq = fudge_qq if topo_distance == 3 else 1.0

        disp = coords[idx] - centre_coord
        disp = _apply_mic(disp, entry.box)
        distance = float(np.linalg.norm(disp))
        if distance == 0.0:
            raise ValueError(
                f"Zero separation encountered for entry_idx={entry.index} gro={entry.gro_file} centre={centre} neighbour={idx}"
            )
        if distance < 0.10:
            continue
        r_hat = disp / distance

        pair_type = tuple(sorted((entry.elements[centre], entry.elements[idx])))

        inv_r7  = distance ** -7
        inv_r13 = distance ** -13
        inv_r7  = min(inv_r7,  1e7)   # 视数据量级可再放宽/收紧
        inv_r13 = min(inv_r13, 1e13)

        coeff_alpha = scale_lj * 12.0 * (distance ** -13) * r_hat
        coeff_beta = scale_lj * (-6.0) * (distance ** -7) * r_hat

        coulomb_vec = np.zeros(3, dtype=float)
        if entry.charges is not None:
            q_c = float(entry.charges[centre])
            q_n = float(entry.charges[idx])
            coulomb_prefactor = ke * q_c * q_n * (distance ** -2) * scale_qq
            coulomb_vec = coulomb_prefactor * r_hat

        pairs.append(
            {
                "pair_type": pair_type,
                "distance": distance,
                "coeff_alpha": coeff_alpha,
                "coeff_beta": coeff_beta,
                "coulomb": coulomb_vec,
                "neighbour_index": idx,
                "topo_distance": topo_distance,
            }
        )

    if max_pairs is not None and len(pairs) > max_pairs:
        pairs.sort(key=lambda item: item["distance"])  # deterministic selection of closest pairs
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
) -> Tuple[NDArray[np.float64], NDArray[np.float64], Dict[Tuple[str, str], int]]:
    """Construct the linear system ``A @ theta ≈ y``."""

    n_entries = len(entries)
    n_pair_types = len(pair_type_list)
    rows = 3 * n_entries
    cols = 2 * n_pair_types
    design = np.zeros((rows, cols), dtype=float)
    targets = np.zeros(rows, dtype=float)

    pair_index = {pair: idx for idx, pair in enumerate(pair_type_list)}

    for entry_idx, entry in enumerate(entries):
        row_base = 3 * entry_idx
        target_force = entry.target_force - entry.coulomb_force - entry.bonded_force
        targets[row_base:row_base + 3] = target_force

        for pair in entry_pairs[entry_idx]:
            col_idx = pair_index[pair["pair_type"]]
            alpha_col = 2 * col_idx
            beta_col = alpha_col + 1
            design[row_base:row_base + 3, alpha_col] += pair["coeff_alpha"]
            design[row_base:row_base + 3, beta_col] += pair["coeff_beta"]

    return design, targets, pair_index


def solve_least_squares(
    design: NDArray[np.float64],
    targets: NDArray[np.float64],
    solver: str,
    alpha: float,
    nonneg: bool,
) -> Tuple[NDArray[np.float64], Dict[str, float]]:
    """Solve the linear system with the selected solver."""

    solver = solver.lower()
    diagnostics: Dict[str, float] = {}

    if design.shape[1] == 0:
        return np.zeros(0, dtype=float), diagnostics

    if solver == "ridge":
        if alpha < 0:
            raise ValueError("Ridge regularisation alpha must be non-negative")
        if alpha > 0:
            sqrt_alpha = math.sqrt(alpha)
            aug_design = np.vstack([design, sqrt_alpha * np.eye(design.shape[1])])
            aug_targets = np.concatenate([targets, np.zeros(design.shape[1], dtype=float)])
        else:
            aug_design = design
            aug_targets = targets

        if nonneg:
            result = lsq_linear(aug_design, aug_targets, bounds=(0.0, np.inf))
            theta = result.x
            diagnostics["optimality"] = float(result.optimality)
        else:
            theta, *_ = np.linalg.lstsq(aug_design, aug_targets, rcond=None)
    elif solver == "qr":
        if nonneg:
            result = lsq_linear(design, targets, bounds=(0.0, np.inf))
            theta = result.x
            diagnostics["optimality"] = float(result.optimality)
        else:
            theta, *_ = np.linalg.lstsq(design, targets, rcond=None)
    elif solver == "nnls":
        theta, resnorm = nnls(design, targets)
        diagnostics["residual_norm"] = float(resnorm)
    else:
        raise ValueError(f"Unsupported solver '{solver}'")

    return theta, diagnostics


def evaluate_fit(
    design: NDArray[np.float64],
    targets: NDArray[np.float64],
    theta: NDArray[np.float64],
    entries: Sequence[ProcessedEntry],
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
        total_pred = entry_pred + entry.coulomb_force + entry.bonded_force
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
    solver: str,
    alpha: float,
    nonneg: bool,
) -> Dict[str, object]:
    """Solve a standalone fit for a single entry."""

    pair_types = sorted({pair["pair_type"] for pair in pairs})
    design = np.zeros((3, 2 * len(pair_types)), dtype=float)

    pair_index = {pair: idx for idx, pair in enumerate(pair_types)}
    for pair in pairs:
        idx = pair_index[pair["pair_type"]]
        alpha_col = 2 * idx
        beta_col = alpha_col + 1
        design[:, alpha_col] += pair["coeff_alpha"]
        design[:, beta_col] += pair["coeff_beta"]

    targets = entry.target_force - entry.coulomb_force - entry.bonded_force
    A_use, y_use, s_col, s_y = _normalise_system(design, targets)
    theta_norm, diagnostics = solve_least_squares(A_use, y_use, solver, alpha, nonneg)
    theta = _denormalise_theta(theta_norm, s_col, s_y)    
    metrics = evaluate_fit(design, targets, theta, [entry])

    parameters = {
        "alpha": {"-".join(pair): float(theta[2 * idx]) for idx, pair in enumerate(pair_types)},
        "beta": {"-".join(pair): float(theta[2 * idx + 1]) for idx, pair in enumerate(pair_types)},
    }

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
    }

def recover_sigma_epsilon(
    pair_type_list: Sequence[Tuple[str, str]],
    theta: NDArray[np.float64],
    pair_index: Mapping[Tuple[str, str], int],
    verbose: bool = False,
) -> Optional[Dict[str, object]]:
    """
    稳健恢复元素级 (sigma, epsilon)：
      1) 先基于 (alpha,beta) 对每个 pair 做闭式估计（带护栏/物理边界）；
      2) 仅用通过护栏的 pair，在 log 域 + 边界下拟合元素级参数（Lorentz–Berthelot）。
    """
    if not pair_type_list:
        return None

    # 取出 pair 级 alpha/beta（注意：设计矩阵里 beta 列系数为 -6/r^7，所以 theta[beta] 就是物理正量 4 ε σ^6）
    alpha_vals = {pair: float(theta[2 * pair_index[pair]]) for pair in pair_type_list}
    beta_vals  = {pair: float(theta[2 * pair_index[pair] + 1]) for pair in pair_type_list}

    # ★ Step 1: pair 级闭式 + 护栏
    def recover_pair(alpha, beta,
                     min_a=1e-12, min_b=1e-12, max_a=1e12, max_b=1e12,
                     sig_lo=0.15, sig_hi=0.50, eps_lo=1e-4, eps_hi=10.0):
        # 夹住到数值允许区间
        if not (np.isfinite(alpha) and np.isfinite(beta)):
            return None
        a = float(np.clip(alpha, min_a, max_a))
        b = float(np.clip(beta,  min_b, max_b))  # 物理 β 应为正

        sigma_ij = (a / b) ** (1.0 / 6.0)
        eps_ij   = (b * b) / (4.0 * a)
        # 物理边界（nm / kJ·mol^-1），可按体系调整
        if not (sig_lo <= sigma_ij <= sig_hi):  # e.g. 0.15~0.50 nm
            return None
        if not (eps_lo <= eps_ij <= eps_hi):    # e.g. 1e-4~10 kJ/mol
            return None
        return sigma_ij, eps_ij

    usable_pairs: List[Tuple[str, str, float, float]] = []
    for pair in pair_type_list:
        a = alpha_vals[pair]; b = beta_vals[pair]
        est = recover_pair(a, b)
        if est is not None:
            sij, eij = est
            usable_pairs.append((pair[0], pair[1], sij, eij))

    if not usable_pairs:
        if verbose:
            print("[recover_sigma_epsilon] No usable pairs passed the guards; abort.", flush=True)
        return None

    # ★ Step 2: 元素级 log-域 + 边界拟合
    elements = sorted({e for p in pair_type_list for e in p})
    idx = {el: k for k, el in enumerate(elements)}
    m = len(elements)

    # 初值：用通过护栏的 pair 的均值
    s_init = {el: 0.30 for el in elements}
    e_init = {el: 0.20 for el in elements}
    acc_s: Dict[str, List[float]] = {el: [] for el in elements}
    acc_e: Dict[str, List[float]] = {el: [] for el in elements}
    for ei, ej, sij, eij in usable_pairs:
        acc_s[ei].append(sij); acc_s[ej].append(sij)
        acc_e[ei].append(eij); acc_e[ej].append(eij)
    for el in elements:
        if acc_s[el]: s_init[el] = float(np.mean(acc_s[el]))
        if acc_e[el]: e_init[el] = float(max(np.mean(acc_e[el]), 1e-4))

    z0 = np.r_[np.log([s_init[el] for el in elements]),
               np.log([e_init[el] for el in elements])]

    # 边界（可按体系微调）
    lb = np.r_[np.log(0.15) * np.ones(m), np.log(1e-4) * np.ones(m)]
    ub = np.r_[np.log(0.50) * np.ones(m), np.log(10.0)  * np.ones(m)]

    # 残差：σ 直接残差，ε 用 log 残差，并加尺度权重（避免 α 与 β 量级差）
    def resid(z):
        x = z[:m]; y = z[m:]
        r = []
        for ei, ej, sij, eij in usable_pairs:
            i = idx[ei]; j = idx[ej]
            sij_pred = 0.5 * (np.exp(x[i]) + np.exp(x[j]))
            # σ 的尺度权重：~0.02 nm
            r.append((sij_pred - sij) / 0.02)
            # ε 用 log 残差 + 权重：~0.2
            r.append((0.5*(y[i] + y[j]) - np.log(eij)) / 0.2)
        return np.asarray(r, float)

    res = least_squares(resid, z0, bounds=(lb, ub), loss="huber", f_scale=1.0, max_nfev=5000)
    sigma = np.exp(res.x[:m]); epsilon = np.exp(res.x[m:])
    rec = {
        "elements": {el: {"sigma": float(s), "epsilon": float(e)} for el, s, e in zip(elements, sigma, epsilon)},
        "residual_norm": float(np.linalg.norm(resid(res.x))),
        "status": int(res.status),
        "message": res.message,
        "pair_used": len(usable_pairs),
        "pair_total": len(pair_type_list),
    }
    if verbose:
        print(f"[recover_sigma_epsilon] usable_pairs={len(usable_pairs)}/{len(pair_type_list)} | "
              f"residual_norm={rec['residual_norm']:.3e}", flush=True)
    return rec


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
        design, targets, pair_index = build_design_matrix(entries, entry_pairs, pair_types)

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
        theta_norm, diagnostics = solve_least_squares(A_use, y_use, args.solver, args.alpha, args.nonneg)
        joint_theta = _denormalise_theta(theta_norm, s_col, s_y)

        # joint_theta, diagnostics = solve_least_squares(design, targets, args.solver, args.alpha, args.nonneg)
        metrics = evaluate_fit(design, targets, joint_theta, entries)

        if args.verbose:
            print(f"Fit RMSE: {metrics['rmse']:.6f}", flush=True)
            print(f"Fit MAE: {metrics['mae']:.6f}", flush=True)
            print(f"Fit R^2: {metrics['r2']:.6f}", flush=True)

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
            per_entry_results.append(fit_single_entry(entry, pairs, args.solver, args.alpha, args.nonneg))

    if per_entry_results:
        output["per_entry_fits"] = per_entry_results
        output["meta"]["per_entry_fit_count"] = len(per_entry_results)
    else:
        output["meta"]["per_entry_fit_count"] = 0

    if args.recover_sigma_eps and joint_theta is not None:
        recovered = recover_sigma_epsilon(pair_types, joint_theta, pair_index, verbose=args.verbose)
        output["recover_sigma_epsilon"] = recovered

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

