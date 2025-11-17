#!/usr/bin/env python3
"""Optimize Lennard-Jones parameters to match centre-atom forces."""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
from types import SimpleNamespace

try:  # pragma: no cover - optional dependency guard
    import numpy as np
    from numpy.typing import NDArray
except ModuleNotFoundError as exc:  # pragma: no cover - clearer error when missing numpy
    raise SystemExit(
        "optimize_center_forces.py requires numpy. Please install numpy before running this script."
    ) from exc

try:  # pragma: no cover - optional dependency guard
    from scipy.optimize import minimize
except ModuleNotFoundError as exc:  # pragma: no cover - clearer error when missing SciPy
    raise SystemExit(
        "optimize_center_forces.py requires SciPy. Please install scipy before running this script."
    ) from exc

try:  # pragma: no cover - optional dependency guard
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch is optional unless GPU is requested
    torch = None

from compare_forces import (
    KELEC,
    Topology,
    build_exclusions,
    infer_topology_from_summary,
)

R_KJ_PER_MOL_K = 0.00831446261815324  # Boltzmann constant in kJ mol^-1 K^-1
DEFAULT_BOX = np.array([1e6, 1e6, 1e6], dtype=float)


@dataclass
class CenterForceEntry:
    index: int
    gro_file: str
    atom_types: List[str]
    coords: NDArray[np.float32]
    charges: NDArray[np.float32]
    center_index: int
    temperature: float
    target_force: NDArray[np.float32]
    bond_force_kj: NDArray[np.float32]
    angle_force_kj: NDArray[np.float32]
    box: NDArray[np.float32]
    coulomb_force_kj: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )
    dihedral_force_kj: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )


@dataclass
class OptimisableLJEntry:
    summary_index: int
    label: str
    sigma: float
    epsilon: float


@dataclass
class OptimisableDihedralEntry:
    summary_index: int
    label: str
    coeffs: Tuple[float, float, float, float, float, float]


@dataclass
class PreparedEntry:
    entry: CenterForceEntry
    topology: Topology
    temperature_factor: float
    coords_gpu: Optional[Any] = None
    atom_param_index: Optional[Any] = None
    sigma_constant_gpu: Optional[Any] = None
    epsilon_constant_gpu: Optional[Any] = None
    bond_force_gpu: Optional[Any] = None
    angle_force_gpu: Optional[Any] = None
    coulomb_force_gpu: Optional[Any] = None
    dihedral_force_gpu: Optional[Any] = None
    temperature_factor_gpu: Optional[Any] = None


@dataclass
class PairMetadata:
    """Metadata describing how each pair type maps to LJ summary indices."""

    idx_i: np.ndarray
    idx_j: np.ndarray
    active_mask: np.ndarray


@dataclass
class LinearizedLJSystem:
    """Linearized representation of LJ contributions for all entries."""

    A_alpha: np.ndarray
    A_beta: np.ndarray
    F_fixed_kJ: np.ndarray
    F_target_kBT: np.ndarray
    temperature_factors: np.ndarray
    pair_metadata: PairMetadata


def load_json(path: Path) -> Mapping[str, object] | Sequence[object]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_force_map(csv_path: Path, label: str) -> Dict[Tuple[str, int], NDArray[np.float32]]:
    fmap: Dict[Tuple[str, int], NDArray[np.float32]] = {}
    if not csv_path.is_file():
        print(f"[load_{label}] WARNING: {csv_path} not found; defaulting to zeros")
        return fmap

    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key_raw = row.get("center_index")
            gro = row.get("gro_file")
            if key_raw is None or gro is None:
                continue
            if str(key_raw).upper() == "ERROR":
                continue
            try:
                idx = int(key_raw)
                fx = float(row.get("Fx", 0.0))
                fy = float(row.get("Fy", 0.0))
                fz = float(row.get("Fz", 0.0))
            except ValueError:
                continue
            fmap[(gro, idx)] = np.array([fx, fy, fz], dtype=float)

    print(f"[load_{label}] entries: {len(fmap)} from {csv_path}")
    return fmap


def cache_force_component(
    entries: Sequence[CenterForceEntry],
    summary: MutableMapping[str, object],
    csv_path: Path,
    label: str,
    compute_fn: Callable[[Topology, NDArray[np.float32]], NDArray[np.float32]],
) -> Dict[Tuple[str, int], NDArray[np.float32]]:
    if csv_path.is_file():
        return load_force_map(csv_path, label)

    rows: List[Dict[str, object]] = []
    cache: Dict[Tuple[str, int], NDArray[np.float32]] = {}
    for entry in entries:
        top = infer_topology_from_summary(summary, entry.coords, entry.atom_types)
        for atom, charge in zip(top.atoms, entry.charges):
            atom.charge = float(charge)
        forces = np.asarray(compute_fn(top, entry.coords), dtype=float)
        vec = forces[entry.center_index]
        cache[(entry.gro_file, entry.center_index)] = vec
        rows.append(
            {
                "gro_file": entry.gro_file,
                "center_index": entry.center_index,
                "Fx": vec[0],
                "Fy": vec[1],
                "Fz": vec[2],
            }
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["gro_file", "center_index", "Fx", "Fy", "Fz"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[cache_{label}] wrote {len(rows)} entries to {csv_path}")
    return cache


def apply_cached_component(
    entries: Sequence[CenterForceEntry],
    cache: Mapping[Tuple[str, int], NDArray[np.float32]],
    attr: str,
) -> None:
    for entry in entries:
        vec = np.asarray(cache.get((entry.gro_file, entry.center_index), np.zeros(3)), dtype=float)
        setattr(entry, attr, vec)


def _infer_center_index(entry: Mapping[str, object]) -> Optional[int]:
    idx = entry.get("center_force_index")
    if isinstance(idx, (int, float)):
        return int(idx)
    atom = entry.get("center_atom")
    if isinstance(atom, Mapping) and "atom_index" in atom:
        try:
            return int(atom["atom_index"]) - 1
        except Exception:
            return None
    return None

def _find_center_index_by_coords(
    coords: NDArray[np.float32],
    atom: Mapping[str, object],
    entry_id: str = ""
) -> Optional[int]:
    """模仿 center_force_plot.py 的逻辑，通过坐标匹配中心原子 index。"""
    if not isinstance(atom, Mapping):
        return None
    try:
        center_coord = [
            float(atom["x"]),
            float(atom["y"]),
            float(atom["z"]),
        ]
    except Exception:
        # 没有 x/y/z 信息就返回 None，交给别的逻辑处理
        return None

    center = np.asarray(center_coord, float)
    diff2 = np.sum((coords - center) ** 2, axis=1)
    idx = int(np.argmin(diff2))
    if diff2[idx] > 1e-6:
        raise ValueError(
            f"Unable to locate center atom via coordinates for {entry_id} "
            f"(min squared distance {diff2[idx]:.3e})"
        )
    return idx


def _as_box(entry: Mapping[str, object]) -> NDArray[np.float32]:
    box = entry.get("box")
    if isinstance(box, Sequence) and len(box) == 3:
        arr = np.asarray(box, dtype=float).reshape(3,)
        if np.all(np.isfinite(arr)):
            return arr
    return DEFAULT_BOX.copy()

def build_entries(
    data: Sequence[Mapping[str, object]],
    bond_forces: Mapping[Tuple[str, int], NDArray[np.float32]],
    angle_forces: Mapping[Tuple[str, int], NDArray[np.float32]],
    limit: Optional[int] = None,
) -> List[CenterForceEntry]:
    entries: List[CenterForceEntry] = []
    for idx, item in enumerate(data):
        if limit is not None and len(entries) >= limit:
            break
        status = str(item.get("center_force_status", "")).lower()
        if status and status not in {"ok", "good", "done", "already has force"}:
            continue

        # 先读坐标（后面要用来按坐标找中心原子）
        coords = np.asarray(item.get("coordinates"), dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"entry {idx}: coordinates must be (N,3)")

        # gro 名，方便打印信息
        gro_file = str(item.get("gro_file", f"entry_{idx}.gro"))
        entry_id = f"{gro_file} (entry {idx})"

        # 1) 按坐标计算中心 index（和 center_force_plot.py 一致）
        center_atom = item.get("center_atom", {})
        center_idx_coords: Optional[int] = None
        try:
            center_idx_coords = _find_center_index_by_coords(coords, center_atom, entry_id=entry_id)
        except ValueError as e:
            # 找不到中心原子的话，给个提示，但不要立刻死掉
            print(f"[WARNING] {e}")

        # 2) 从 json 里读 index（兼容旧字段）
        center_idx_json: Optional[int] = _infer_center_index(item)

        # 3) 对比两个 index，如果都存在且不一样，就打印出来
        if center_idx_coords is not None and center_idx_json is not None and center_idx_coords != center_idx_json:
            print(
                f"[WARNING] center index mismatch for {gro_file}: "
                f"json={center_idx_json}, coords={center_idx_coords}"
            )

        # 4) 决定最终使用哪个 index：
        #    优先用“坐标算出的”（保证与 center_force_plot.py 一致），
        #    如果没有坐标信息，就退回 json 的 index。
        if center_idx_coords is not None:
            center_index = center_idx_coords
        else:
            print("coords failed")
            center_index = center_idx_json

        # 两种方式都失败，就跳过这一条数据
        if center_index is None:
            continue

        # === 新增：和 CSV 里的 index 对比 ===
        csv_indices = set()

        # 从 bond CSV 里收集所有这个 gro 对应的 index
        for (gro, idx_csv) in bond_forces.keys():
            if gro == gro_file:
                csv_indices.add(idx_csv)

        # 从 angle CSV 里也收集一遍
        for (gro, idx_csv) in angle_forces.keys():
            if gro == gro_file:
                csv_indices.add(idx_csv)

        # 如果 CSV 里有记录，而且存在和 center_index 不同的 index，就打印出来
        if csv_indices:
            mismatched = [i for i in csv_indices if i != center_index]
            if mismatched:
                print(
                    f"[WARNING] center index mismatch vs CSV for {gro_file}: "
                    f"chosen={center_index}, csv_indices={sorted(csv_indices)}"
                )
        # === 新增结束 ===

        atom_types = [str(x) for x in item.get("atom_types", [])]
        if len(atom_types) != coords.shape[0]:
            raise ValueError(f"entry {idx}: atom count mismatch between coordinates and atom_types")

        charges_raw = np.asarray(item.get("formal_charges", np.zeros(coords.shape[0])), dtype=float)
        if charges_raw.shape[0] != coords.shape[0]:
            raise ValueError(f"entry {idx}: formal_charges length mismatch")

        # 单位：kJ/mol/nm
        bond_force = np.asarray(bond_forces.get((gro_file, center_index), np.zeros(3)), dtype=float)
        angle_force = np.asarray(angle_forces.get((gro_file, center_index), np.zeros(3)), dtype=float)

        temp = float(item.get("center_force_temperature_K", 298.15))
        target_force = np.asarray(item.get("center_force_kBT_per_nm"), dtype=float)
        if target_force.shape != (3,):
            raise ValueError(f"entry {idx}: center_force_kBT_per_nm must be length-3")

        box = _as_box(item)

        entries.append(
            CenterForceEntry(
                index=len(entries),
                gro_file=gro_file,
                atom_types=atom_types,
                coords=coords,
                charges=charges_raw,
                center_index=center_index,
                temperature=temp,
                target_force=target_force,
                bond_force_kj=bond_force,
                angle_force_kj=angle_force,
                box=box,
            )
        )
    return entries

from collections import Counter

def debug_filtering(
    data: Sequence[Mapping[str, object]],
    bond_forces: Mapping[Tuple[str, int], NDArray[np.float32]],
    angle_forces: Mapping[Tuple[str, int], NDArray[np.float32]],
    limit: Optional[int] = None,
) -> None:
    stats = Counter()

    for idx, item in enumerate(data):
        if limit is not None and stats["kept"] >= limit:
            break

        stats["total"] += 1

        # 1) status 过滤
        status = str(item.get("center_force_status", "")).lower()
        if status and status not in {"ok", "good", "done"}:
            stats["bad_status"] += 1
            continue

        # 2) 坐标检查
        coords = np.asarray(item.get("coordinates"), dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            stats["bad_coords"] += 1
            continue

        gro_file = str(item.get("gro_file", f"entry_{idx}.gro"))
        entry_id = f"{gro_file} (entry {idx})"

        # 3) 通过坐标找中心
        center_atom = item.get("center_atom", {})
        center_idx_coords: Optional[int] = None
        try:
            center_idx_coords = _find_center_index_by_coords(coords, center_atom, entry_id=entry_id)
        except Exception:
            pass

        # 4) 从 JSON 字段猜中心 index
        center_idx_json: Optional[int] = _infer_center_index(item)

        if center_idx_coords is not None:
            center_index = center_idx_coords
        else:
            center_index = center_idx_json

        if center_index is None:
            stats["no_center_index"] += 1
            continue

        # 5) 看 CSV 里有没有对应的条目
        if (gro_file, center_index) not in bond_forces:
            stats["no_bond_force_in_csv"] += 1
        if (gro_file, center_index) not in angle_forces:
            stats["no_angle_force_in_csv"] += 1

        stats["kept"] += 1

    print("=== Filter stats ===")
    for k, v in stats.items():
        print(f"{k:25s}: {v}")


def select_optimisable_lj(summary: MutableMapping[str, object], fix_hydroxyl_h: bool = True) -> List[OptimisableLJEntry]:
    entries: List[OptimisableLJEntry] = []
    lj_list = summary.get("lj", [])
    if not isinstance(lj_list, list):
        raise ValueError("summary JSON missing 'lj' list")
    for idx, entry in enumerate(lj_list):
        element = str(entry.get("element", "")).lower()
        variant = str(entry.get("variant", "default"))
        if fix_hydroxyl_h and element == "h" and variant == "hydroxyl":
            continue
        try:
            sigma = float(entry["sigma"])
            epsilon = float(entry["epsilon"])
        except Exception as exc:
            raise ValueError(f"Invalid sigma/epsilon in lj entry {idx}") from exc
        label = f"{element}_{variant}_{idx}"
        entries.append(OptimisableLJEntry(summary_index=idx, label=label, sigma=sigma, epsilon=epsilon))
    if not entries:
        raise ValueError("No Lennard-Jones entries available for optimisation")
    return entries


def select_optimisable_dihedrals(summary: MutableMapping[str, object]) -> List[OptimisableDihedralEntry]:
    entries: List[OptimisableDihedralEntry] = []
    dihedral_list = summary.get("dihedrals", [])
    if not isinstance(dihedral_list, list):
        return entries
    for idx, entry in enumerate(dihedral_list):
        coeffs = entry.get("c")
        if not isinstance(coeffs, Sequence) or len(coeffs) != 6:
            continue
        label = str(entry.get("pattern", f"dih_{idx}"))
        entries.append(
            OptimisableDihedralEntry(
                summary_index=idx,
                label=label,
                coeffs=tuple(float(v) for v in coeffs),
            )
        )
    return entries


def _sync_topology_atomtypes(top: Topology, lj_list: Sequence[Mapping[str, object]]) -> None:
    for atom_type in top.atomtypes.values():
        idx = getattr(atom_type, "source_entry_idx", None)
        if idx is None:
            continue
        if idx < 0 or idx >= len(lj_list):
            continue
        entry = lj_list[idx]
        sigma = float(entry["sigma"])
        epsilon = float(entry["epsilon"])
        atom_type.sigma = sigma
        atom_type.epsilon = epsilon
        atom_type.C6 = float(entry.get("C6", 4.0 * epsilon * (sigma ** 6)))
        atom_type.C12 = float(entry.get("C12", 4.0 * epsilon * (sigma ** 12)))
        atom_type.source_entry_idx = int(idx)


def prepare_entries_for_prediction(
    summary: MutableMapping[str, object],
    entries: Sequence[CenterForceEntry],
    device: str,
    lj_targets: Sequence[OptimisableLJEntry],
) -> List[PreparedEntry]:
    prepared: List[PreparedEntry] = []
    use_gpu = device.lower() == "gpu" and torch is not None and torch.cuda.is_available()
    gpu_device = torch.device("cuda") if use_gpu else None
    target_lookup = {slot.summary_index: idx for idx, slot in enumerate(lj_targets)}

    for entry in entries:
        top = infer_topology_from_summary(summary, entry.coords, entry.atom_types)
        for atom, charge in zip(top.atoms, entry.charges):
            atom.charge = float(charge)
        factor = 1.0 / (R_KJ_PER_MOL_K * max(float(entry.temperature), 1e-6))
        coords_gpu = None
        atom_param_tensor = None
        sigma_const_tensor = None
        epsilon_const_tensor = None
        bond_force_gpu = None
        angle_force_gpu = None
        coulomb_force_gpu = None
        dihedral_force_gpu = None
        temp_factor_gpu = None
        if use_gpu:
            coords_gpu = torch.as_tensor(entry.coords, dtype=torch.float32, device=gpu_device)
            atom_param_idx = np.full(len(top.atoms), -1, dtype=np.int64)
            sigma_const = np.zeros(len(top.atoms), dtype=np.float32)
            epsilon_const = np.zeros(len(top.atoms), dtype=np.float32)
            for atom_idx, atom in enumerate(top.atoms):
                atom_type = top.atomtypes[atom.type_name]
                source_idx = getattr(atom_type, "source_entry_idx", None)
                sigma_const[atom_idx] = float(atom_type.sigma)
                epsilon_const[atom_idx] = float(atom_type.epsilon)
                if source_idx is None:
                    continue
                param_idx = target_lookup.get(int(source_idx))
                if param_idx is not None:
                    atom_param_idx[atom_idx] = int(param_idx)
            atom_param_tensor = torch.as_tensor(atom_param_idx, dtype=torch.long, device=gpu_device)
            sigma_const_tensor = torch.as_tensor(sigma_const, dtype=torch.float32, device=gpu_device)
            epsilon_const_tensor = torch.as_tensor(epsilon_const, dtype=torch.float32, device=gpu_device)
            bond_force_gpu = torch.as_tensor(entry.bond_force_kj, dtype=torch.float32, device=gpu_device)
            angle_force_gpu = torch.as_tensor(entry.angle_force_kj, dtype=torch.float32, device=gpu_device)
            coulomb_force_gpu = torch.as_tensor(entry.coulomb_force_kj, dtype=torch.float32, device=gpu_device)
            dihedral_force_gpu = torch.as_tensor(entry.dihedral_force_kj, dtype=torch.float32, device=gpu_device)
            temp_factor_gpu = torch.tensor(factor, dtype=torch.float32, device=gpu_device)
        prepared.append(
            PreparedEntry(
                entry=entry,
                topology=top,
                temperature_factor=factor,
                coords_gpu=coords_gpu,
                atom_param_index=atom_param_tensor,
                sigma_constant_gpu=sigma_const_tensor,
                epsilon_constant_gpu=epsilon_const_tensor,
                bond_force_gpu=bond_force_gpu,
                angle_force_gpu=angle_force_gpu,
                coulomb_force_gpu=coulomb_force_gpu,
                dihedral_force_gpu=dihedral_force_gpu,
                temperature_factor_gpu=temp_factor_gpu,
            )
        )
    return prepared


def refresh_prepared_topologies(
    prepared_entries: Sequence[PreparedEntry], summary: MutableMapping[str, object]
) -> None:
    lj_list = summary.get("lj", [])
    if not isinstance(lj_list, Sequence):
        return
    dihedral_list = summary.get("dihedrals", []) if isinstance(summary.get("dihedrals"), Sequence) else []
    for prepared in prepared_entries:
        _sync_topology_atomtypes(prepared.topology, lj_list)
        _sync_topology_dihedrals(prepared.topology, dihedral_list)


def _atomtype_summary_index(top: Topology, atom_idx: int) -> Optional[int]:
    atom = top.atoms[atom_idx]
    atom_type = top.atomtypes.get(atom.type_name)
    if atom_type is None:
        return None
    idx = getattr(atom_type, "source_entry_idx", None)
    if idx is None:
        return None
    return int(idx)


def _sync_topology_dihedrals(top: Topology, dihedral_entries: Sequence[Mapping[str, object]]) -> None:
    if not dihedral_entries:
        return
    for dih in top.rb_dihedrals:
        src = getattr(dih, "source_entry_idx", None)
        if src is None:
            continue
        if src < 0 or src >= len(dihedral_entries):
            continue
        entry = dihedral_entries[src]
        coeffs = entry.get("c") if isinstance(entry, Mapping) else None
        if not isinstance(coeffs, Sequence) or len(coeffs) != 6:
            continue
        dih.c = tuple(float(v) for v in coeffs)


def _collect_center_lj_pairs(
    top: Topology, coords: NDArray[np.float32], center_idx: int, rvdw: float
) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    coords_arr = np.asarray(coords, dtype=float)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
        raise ValueError("Coordinates must be of shape (N,3) for LJ aggregation")
    if rvdw <= 0.0:
        return []
    exclusion_pairs = set()
    for i, j in build_exclusions(top):
        pair = (min(i - 1, j - 1), max(i - 1, j - 1))
        exclusion_pairs.add(pair)
    pair14_pairs = {
        (min(i - 1, j - 1), max(i - 1, j - 1)) for (i, j) in getattr(top, "pairs14", set())
    }
    center_coord = coords_arr[center_idx]
    contributions: List[Tuple[int, np.ndarray, np.ndarray]] = []
    n_atoms = coords_arr.shape[0]
    for nb_idx in range(n_atoms):
        if nb_idx == center_idx:
            continue
        pair_key = (min(center_idx, nb_idx), max(center_idx, nb_idx))
        if pair_key in exclusion_pairs:
            continue
        disp = coords_arr[nb_idx] - center_coord
        r2 = float(np.dot(disp, disp))
        if r2 < 1e-24:
            continue
        dist = math.sqrt(r2)
        if dist >= rvdw:
            continue
        inv_r = 1.0 / dist
        r_hat = disp * inv_r
        invr2 = inv_r * inv_r
        invr6 = invr2 ** 3
        invr12 = invr6 ** 2
        scale = float(top.fudgeLJ) if pair_key in pair14_pairs else 1.0
        coeff_alpha = (-12.0 * invr12 * inv_r * scale) * r_hat
        coeff_beta = (6.0 * invr6 * inv_r * scale) * r_hat
        contributions.append((nb_idx, coeff_alpha, coeff_beta))
    return contributions


def build_linearized_lj_system(
    prepared_entries: Sequence[PreparedEntry],
    summary: MutableMapping[str, object],
    rvdw: float,
) -> Tuple[LinearizedLJSystem, np.ndarray, np.ndarray]:
    lj_list = summary.get("lj", [])
    if not isinstance(lj_list, Sequence):
        raise ValueError("summary JSON missing 'lj' list for LJ aggregation")
    n_entries = len(prepared_entries)
    sigma_all = np.array([float(item.get("sigma", 0.0)) for item in lj_list], dtype=float)
    epsilon_all = np.array([float(item.get("epsilon", 0.0)) for item in lj_list], dtype=float)
    pair_index: Dict[Tuple[int, int], int] = {}
    pair_order: List[Tuple[int, int]] = []
    per_entry_pairs: List[List[Tuple[int, np.ndarray, np.ndarray]]] = []
    F_fixed_kJ = np.zeros((n_entries, 3), dtype=np.float64)
    F_target_kBT = np.zeros((n_entries, 3), dtype=np.float64)
    temperature_factors = np.zeros(n_entries, dtype=np.float64)

    for entry_idx, prepared in enumerate(prepared_entries):
        entry = prepared.entry
        top = prepared.topology
        center_idx = int(entry.center_index)
        center_type_idx = _atomtype_summary_index(top, center_idx)
        if center_type_idx is None:
            raise ValueError(
                f"Atom type for center index {center_idx} missing LJ source index in entry {entry.gro_file}"
            )
        contributions: List[Tuple[int, np.ndarray, np.ndarray]] = []
        for nb_idx, coeff_alpha, coeff_beta in _collect_center_lj_pairs(
            top, entry.coords, center_idx, rvdw
        ):
            nb_type_idx = _atomtype_summary_index(top, nb_idx)
            if nb_type_idx is None:
                continue
            if center_type_idx <= nb_type_idx:
                pair_key = (center_type_idx, nb_type_idx)
            else:
                pair_key = (nb_type_idx, center_type_idx)
            if pair_key not in pair_index:
                pair_index[pair_key] = len(pair_order)
                pair_order.append(pair_key)
            p_idx = pair_index[pair_key]
            contributions.append((p_idx, coeff_alpha.astype(float), coeff_beta.astype(float)))
        per_entry_pairs.append(contributions)
        F_fixed_kJ[entry_idx] = (
            np.asarray(entry.bond_force_kj, dtype=float)
            + np.asarray(entry.angle_force_kj, dtype=float)
            + np.asarray(entry.dihedral_force_kj, dtype=float)
            + np.asarray(entry.coulomb_force_kj, dtype=float)
        )
        F_target_kBT[entry_idx] = np.asarray(entry.target_force, dtype=float)
        temperature_factors[entry_idx] = float(prepared.temperature_factor)

    n_pairs = len(pair_order)
    A_alpha = np.zeros((n_entries, n_pairs, 3), dtype=np.float64)
    A_beta = np.zeros((n_entries, n_pairs, 3), dtype=np.float64)
    for entry_idx, contribs in enumerate(per_entry_pairs):
        for p_idx, coeff_alpha, coeff_beta in contribs:
            A_alpha[entry_idx, p_idx, :] += coeff_alpha
            A_beta[entry_idx, p_idx, :] += coeff_beta

    idx_i = np.array([pair[0] for pair in pair_order], dtype=np.int64) if pair_order else np.zeros(0, dtype=np.int64)
    idx_j = np.array([pair[1] for pair in pair_order], dtype=np.int64) if pair_order else np.zeros(0, dtype=np.int64)
    active_mask = np.ones(len(pair_order), dtype=bool)
    metadata = PairMetadata(idx_i=idx_i, idx_j=idx_j, active_mask=active_mask)
    system = LinearizedLJSystem(
        A_alpha=A_alpha,
        A_beta=A_beta,
        F_fixed_kJ=F_fixed_kJ,
        F_target_kBT=F_target_kBT,
        temperature_factors=temperature_factors,
        pair_metadata=metadata,
    )
    return system, sigma_all, epsilon_all


def _compute_pair_alpha_beta_torch(
    sigma_full: "torch.Tensor",
    epsilon_full: "torch.Tensor",
    idx_i_t: "torch.Tensor",
    idx_j_t: "torch.Tensor",
    active_mask_t: "torch.Tensor",
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    if torch is None:
        raise RuntimeError("Torch is required for linearised LJ evaluation")
    if idx_i_t.numel() == 0:
        zero = torch.zeros(0, dtype=sigma_full.dtype, device=sigma_full.device)
        return zero, zero
    sigma_i = sigma_full[idx_i_t]
    sigma_j = sigma_full[idx_j_t]
    epsilon_i = epsilon_full[idx_i_t]
    epsilon_j = epsilon_full[idx_j_t]
    valid = active_mask_t
    valid = valid & (sigma_i > 0.0) & (sigma_j > 0.0) & (epsilon_i > 0.0) & (epsilon_j > 0.0)
    sig_mix = torch.sqrt(torch.clamp(sigma_i * sigma_j, min=1e-24))
    eps_mix = torch.sqrt(torch.clamp(epsilon_i * epsilon_j, min=1e-24))
    sigma6 = sig_mix.pow(6)
    beta = torch.where(valid, 4.0 * eps_mix * sigma6, torch.zeros_like(sig_mix))
    alpha = beta * sigma6
    return alpha, beta


def apply_lj_parameters(
    summary: MutableMapping[str, object],
    targets: Sequence[OptimisableLJEntry],
    sigma_values: Sequence[float],
    epsilon_values: Sequence[float],
) -> None:
    lj_list = summary.get("lj")
    if not isinstance(lj_list, list):
        raise ValueError("summary JSON missing 'lj' list")
    for slot, sigma, epsilon in zip(targets, sigma_values, epsilon_values):
        idx = slot.summary_index
        entry = lj_list[idx]
        entry["sigma"] = float(sigma)
        entry["epsilon"] = float(epsilon)
        entry["C6"] = float(4.0 * epsilon * (sigma ** 6))
        entry["C12"] = float(4.0 * epsilon * (sigma ** 12))


def apply_dihedral_parameters(
    summary: MutableMapping[str, object],
    targets: Sequence[OptimisableDihedralEntry],
    coeff_matrix: Sequence[Sequence[float]],
) -> None:
    dihedral_list = summary.get("dihedrals")
    if not isinstance(dihedral_list, list):
        raise ValueError("summary JSON missing 'dihedrals' list")
    for slot, coeffs in zip(targets, coeff_matrix):
        idx = slot.summary_index
        if idx < 0 or idx >= len(dihedral_list):
            continue
        entry = dihedral_list[idx]
        entry["c"] = [float(v) for v in coeffs]


def compute_nonbonded_forces(
    top: Topology,
    coords_in,
    rcoul: float = 1.2,
    rvdw: float = 1.2,
    do_lj: bool = True,
    do_coul: bool = True,
    device: str = "gpu",
    sigma_override=None,
    epsilon_override=None,
    return_numpy: bool = True,
) -> np.ndarray:
    """
    只计算非键相互作用的力 (LJ / 库伦)，单位 kJ/mol/nm。
    使用 do_lj / do_coul 控制是否计算对应项。
    device="gpu" 时使用 PyTorch + CUDA 对 LJ/库伦求和进行矢量化加速。
    """

    if device.lower() == "gpu":
        return _compute_nonbonded_forces_gpu(
            top,
            coords_in,
            rcoul,
            rvdw,
            do_lj,
            do_coul,
            sigma_override=sigma_override,
            epsilon_override=epsilon_override,
            return_numpy=return_numpy,
        )
    if not return_numpy:
        raise ValueError("CPU nonbonded force evaluation only supports numpy outputs")
    return _compute_nonbonded_forces_cpu(top, coords_in, rcoul, rvdw, do_lj, do_coul)


def _compute_nonbonded_forces_cpu(
    top: Topology,
    coords_in,
    rcoul: float,
    rvdw: float,
    do_lj: bool,
    do_coul: bool,
) -> np.ndarray:
    coords = np.asarray(coords_in, dtype=float)
    n = coords.shape[0]
    forces = np.zeros_like(coords)

    charges = np.array([a.charge for a in top.atoms], float)
    sigma = np.array([top.atomtypes[a.type_name].sigma for a in top.atoms], float)
    epsilon = np.array([top.atomtypes[a.type_name].epsilon for a in top.atoms], float)

    exclusions = build_exclusions(top)
    pairs14 = top.pairs14

    for i in range(n - 1):
        ri = coords[i]
        diff = coords[i + 1:] - ri
        j_idx_all = np.arange(i + 1, n)
        r2_all = np.einsum("ij,ij->i", diff, diff)
        mask = r2_all >= 1e-24
        if not np.any(mask):
            continue

        diff = diff[mask]
        r2 = r2_all[mask]
        r = np.sqrt(r2)
        j_idx = j_idx_all[mask]

        for dvec, dist, j in zip(diff, r, j_idx):
            pair = (i + 1, j + 1) if i < j else (j + 1, i + 1)
            if dist < 1e-24:
                continue

            excluded = pair in exclusions
            is14 = pair in pairs14

            if do_lj and (not excluded) and dist < rvdw:
                sig = math.sqrt(sigma[i] * sigma[j])
                eps = math.sqrt(epsilon[i] * epsilon[j])

                c6 = 4.0 * eps * (sig ** 6)
                c12 = 4.0 * eps * (sig ** 12)
                if is14:
                    c6 *= top.fudgeLJ
                    c12 *= top.fudgeLJ
                invr2 = 1.0 / (dist * dist)
                invr6 = invr2 ** 3
                invr12 = invr6 ** 2
                coef = (12.0 * c12 * invr12 - 6.0 * c6 * invr6) * invr2
                f = -coef * dvec
                forces[i] += f
                forces[j] -= f

            if do_coul and (not excluded) and dist < rcoul:
                qq = charges[i] * charges[j]
                if is14:
                    qq *= top.fudgeQQ
                invr = 1.0 / dist
                coef = KELEC * qq * (invr ** 3)
                f = -coef * dvec
                forces[i] += f
                forces[j] -= f

    return forces


def _compute_nonbonded_forces_gpu(
    top: Topology,
    coords_in,
    rcoul: float,
    rvdw: float,
    do_lj: bool,
    do_coul: bool,
    *,
    sigma_override=None,
    epsilon_override=None,
    return_numpy: bool = True,
):
    if torch is None:
        raise RuntimeError("GPU acceleration requires torch to be installed")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU acceleration requested but CUDA device is not available")

    device = torch.device("cuda")
    dtype = torch.float32
    if isinstance(coords_in, torch.Tensor):
        coords = coords_in.to(device=device, dtype=dtype)
    else:
        coords = torch.as_tensor(np.asarray(coords_in, dtype=float), dtype=dtype, device=device)
    n = coords.shape[0]
    forces = torch.zeros_like(coords)

    charges = torch.tensor([a.charge for a in top.atoms], dtype=dtype, device=device)
    if sigma_override is not None:
        sigma = sigma_override.to(device=device, dtype=dtype)
    else:
        sigma = torch.tensor([top.atomtypes[a.type_name].sigma for a in top.atoms], dtype=dtype, device=device)
    if epsilon_override is not None:
        epsilon = epsilon_override.to(device=device, dtype=dtype)
    else:
        epsilon = torch.tensor([top.atomtypes[a.type_name].epsilon for a in top.atoms], dtype=dtype, device=device)

    exclusion_mask = torch.zeros((n, n), dtype=torch.bool, device=device)
    for i, j in build_exclusions(top):
        i0, j0 = i - 1, j - 1
        exclusion_mask[i0, j0] = True
        exclusion_mask[j0, i0] = True

    pair14_mask = torch.zeros((n, n), dtype=torch.bool, device=device)
    for i, j in top.pairs14:
        i0, j0 = i - 1, j - 1
        pair14_mask[i0, j0] = True
        pair14_mask[j0, i0] = True

    diff = coords[None, :, :] - coords[:, None, :]
# 现在 diff[i, j] = coords[j] - coords[i]，和 CPU / compare_forces 一致

    r2 = torch.sum(diff * diff, dim=-1)
    triu_mask = torch.triu(torch.ones((n, n), dtype=torch.bool, device=device), diagonal=1)
    valid_mask = triu_mask & (r2 >= 1e-24)

    if do_lj and rvdw > 0:
        dist = torch.sqrt(torch.clamp(r2, min=1e-24))
        mask = valid_mask & (~exclusion_mask) & (dist < rvdw)
        if torch.any(mask):
            sig = torch.sqrt(sigma[:, None] * sigma[None, :])
            eps = torch.sqrt(epsilon[:, None] * epsilon[None, :])
            c6 = 4.0 * eps * (sig ** 6)
            c12 = 4.0 * eps * (sig ** 12)
            c6 = torch.where(pair14_mask, c6 * top.fudgeLJ, c6)
            c12 = torch.where(pair14_mask, c12 * top.fudgeLJ, c12)

            c6_vals = c6[mask]
            c12_vals = c12[mask]
            r2_vals = r2[mask]
            dvec = diff[mask]

            invr2 = 1.0 / r2_vals
            invr6 = invr2 ** 3
            invr12 = invr6 ** 2
            coef = (12.0 * c12_vals * invr12 - 6.0 * c6_vals * invr6) * invr2
            pair_forces = -coef[:, None] * dvec

            pairs = torch.nonzero(mask, as_tuple=False)
            forces.index_add_(0, pairs[:, 0], pair_forces)
            forces.index_add_(0, pairs[:, 1], -pair_forces)

    if do_coul and rcoul > 0:
        dist = torch.sqrt(torch.clamp(r2, min=1e-24))
        mask = valid_mask & (~exclusion_mask) & (dist < rcoul)
        if torch.any(mask):
            qq = charges[:, None] * charges[None, :]
            qq = torch.where(pair14_mask, qq * top.fudgeQQ, qq)
            qq_vals = qq[mask]
            r_vals = dist[mask]
            invr = 1.0 / r_vals
            coef = KELEC * qq_vals * (invr ** 3)
            dvec = diff[mask]
            pair_forces = -coef[:, None] * dvec

            pairs = torch.nonzero(mask, as_tuple=False)
            forces.index_add_(0, pairs[:, 0], pair_forces)
            forces.index_add_(0, pairs[:, 1], -pair_forces)

    if return_numpy:
        return forces.detach().cpu().numpy()
    return forces


def compute_dihedral_forces(
    top: Topology,
    coords_in,
) -> np.ndarray:
    """
    只计算 RB dihedral 的力，单位 kJ/mol/nm。
    """
    coords = np.asarray(coords_in, dtype=float)
    n = coords.shape[0]
    forces = np.zeros((n, 3), dtype=float)

    for dih in top.rb_dihedrals:
        i, j, k, l = dih.i - 1, dih.j - 1, dih.k - 1, dih.l - 1

        b1 = coords[i] - coords[j]
        b2 = coords[k] - coords[j]
        b3 = coords[l] - coords[k]

        c1 = np.cross(b2, b3)
        c2 = np.cross(b1, b2)

        nb2 = max(float(np.linalg.norm(b2)), 1e-12)
        nc1 = max(float(np.linalg.norm(c1)), 1e-12)
        nc2 = max(float(np.linalg.norm(c2)), 1e-12)

        x = float(np.dot(c2, c1))
        y = nb2 * float(np.dot(b1, c1))
        phi = math.atan2(y, x)

        # RB 势 dV/dphi
        c = dih.c
        cosp = math.cos(phi)
        sinp = math.sin(phi)
        s = 0.0
        cp = 1.0
        for n_ in range(1, 6):
            s += n_ * c[n_] * cp
            cp *= cosp
        dVdphi = -sinp * s

        dphi_di = (nb2 / (nc2 * nc2)) * c2
        dphi_dl = (nb2 / (nc1 * nc1)) * c1

        db1b2 = float(np.dot(b1, b2))
        db3b2 = float(np.dot(b3, b2))
        term_j1 = (db1b2 / nb2) / (nc2 * nc2)
        term_j2 = (db3b2 / nb2) / (nc1 * nc1)
        dphi_dj = term_j1 * c2 + term_j2 * c1
        dphi_dk = -(dphi_di + dphi_dj + dphi_dl)

        Fi = -dVdphi * dphi_di
        Fj = -dVdphi * dphi_dj
        Fk = -dVdphi * dphi_dk
        Fl = -dVdphi * dphi_dl

        # 扭矩修正（保持和原脚本一致的写法）
        m = 0.5 * (coords[j] + coords[k])
        ri, rj, rk, rl = coords[i] - m, coords[j] - m, coords[k] - m, coords[l] - m
        tau = np.cross(ri, Fi) + np.cross(rj, Fj) + np.cross(rk, Fk) + np.cross(rl, Fl)
        cross_bt = np.cross(b2, tau)
        denom = float(np.dot(b2, b2)) + 1e-30
        Delta = -cross_bt / denom

        Fj = Fj + Delta
        Fk = Fk - Delta

        forces[i] += Fi
        forces[j] += Fj
        forces[k] += Fk
        forces[l] += Fl

    return forces


def predict_forces(
    summary: MutableMapping[str, object],
    prepared_entries: Sequence[PreparedEntry],
    rvdw: float,
    device: str = "gpu",
    recompute_dihedral: bool = False,
) -> NDArray[np.float32]:
    """
    用和 compare_plot_csv.py 一致的方式计算预测力。

    - LJ：compute_nonbonded_forces(..., do_lj=True, do_coul=False, rvdw=rvdw, rcoul=0)
    - COUL：来自缓存的 CSV（单位 kJ/mol/nm）
    - DIH：来自缓存的 CSV（单位 kJ/mol/nm）
    - 再加 bond / angle（来自 CSV，单位 kJ/mol/nm）
    - 最后结果转成 kBT/nm
    """
    predictions: List[NDArray[np.float32]] = []

    use_gpu = device.lower() == "gpu"

    for prepared in prepared_entries:
        entry = prepared.entry
        coords_in: Any
        if use_gpu and prepared.coords_gpu is not None:
            coords_in = prepared.coords_gpu
        else:
            coords_in = entry.coords

        forces_lj_kj = compute_nonbonded_forces(
            prepared.topology,
            coords_in,
            rcoul=0.0,
            rvdw=rvdw,
            do_lj=True,
            do_coul=False,
            device=device,
        )

        idx = entry.center_index
        if recompute_dihedral:
            dihedral_force = compute_dihedral_forces(prepared.topology, entry.coords)
        else:
            dihedral_force = entry.dihedral_force_kj
        total_nb_kj = (
            forces_lj_kj[idx]
            + entry.coulomb_force_kj
            + dihedral_force
        )
        total_kj = total_nb_kj + entry.bond_force_kj + entry.angle_force_kj

        predictions.append(total_kj * prepared.temperature_factor)

    return np.vstack(predictions)


def run_linearized_torch_optimizer(
    system: LinearizedLJSystem,
    sigma_all0: np.ndarray,
    epsilon_all0: np.ndarray,
    trainable_summary_indices: Sequence[int],
    sigma0: NDArray[np.float32],
    epsilon0: NDArray[np.float32],
    args: argparse.Namespace,
    logger: OptimisationLogger,
) -> Tuple[NDArray[np.float32], NDArray[np.float32], Dict[str, object]]:
    if torch is None:
        raise RuntimeError("Torch is required for linearised GPU optimisation")
    want_gpu = args.device.lower() == "gpu"
    if want_gpu and not torch.cuda.is_available():
        print("[GPU] CUDA not available; falling back to CPU tensors for optimisation.")
    device = torch.device("cuda" if want_gpu and torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    tensor_kwargs = {"dtype": dtype, "device": device}
    A_alpha_t = torch.as_tensor(system.A_alpha, **tensor_kwargs)
    A_beta_t = torch.as_tensor(system.A_beta, **tensor_kwargs)
    F_fixed_t = torch.as_tensor(system.F_fixed_kJ, **tensor_kwargs)
    F_target_t = torch.as_tensor(system.F_target_kBT, **tensor_kwargs)
    temp_factors_t = torch.as_tensor(system.temperature_factors, **tensor_kwargs)
    idx_i_t = torch.as_tensor(system.pair_metadata.idx_i, dtype=torch.long, device=device)
    idx_j_t = torch.as_tensor(system.pair_metadata.idx_j, dtype=torch.long, device=device)
    active_mask_t = torch.as_tensor(system.pair_metadata.active_mask, dtype=torch.bool, device=device)
    sigma_base_t = torch.as_tensor(sigma_all0, **tensor_kwargs)
    epsilon_base_t = torch.as_tensor(epsilon_all0, **tensor_kwargs)
    trainable_idx_t = torch.as_tensor(trainable_summary_indices, dtype=torch.long, device=device)

    sigma_param = torch.nn.Parameter(torch.as_tensor(sigma0, **tensor_kwargs))
    epsilon_param = torch.nn.Parameter(torch.as_tensor(epsilon0, **tensor_kwargs))
    optimizer = torch.optim.Adam([sigma_param, epsilon_param], lr=float(args.learning_rate))
    early_stopper = EarlyStopping(
        patience=int(args.patience), initial_loss=float("inf"), min_delta=1e-9
    )
    best_sigma = sigma0.copy()
    best_epsilon = epsilon0.copy()
    best_loss = float("inf")
    target_np = system.F_target_kBT.astype(np.float32)

    def assemble_full(base: "torch.Tensor", updates: "torch.Tensor") -> "torch.Tensor":
        if trainable_idx_t.numel() == 0:
            return base
        full = base.clone()
        return full.index_copy(0, trainable_idx_t, updates)

    for iteration in range(1, int(args.max_iter) + 1):
        optimizer.zero_grad()
        sigma_clamped = torch.clamp(
            sigma_param, min=float(args.sigma_min), max=float(args.sigma_max)
        )
        epsilon_clamped = torch.clamp(
            epsilon_param, min=float(args.epsilon_min), max=float(args.epsilon_max)
        )
        sigma_full = assemble_full(sigma_base_t, sigma_clamped)
        epsilon_full = assemble_full(epsilon_base_t, epsilon_clamped)
        alpha_t, beta_t = _compute_pair_alpha_beta_torch(
            sigma_full, epsilon_full, idx_i_t, idx_j_t, active_mask_t
        )
        if alpha_t.numel() == 0:
            F_lj_t = torch.zeros_like(F_fixed_t)
        else:
            F_alpha = torch.einsum("epc,p->ec", A_alpha_t, alpha_t)
            F_beta = torch.einsum("epc,p->ec", A_beta_t, beta_t)
            F_lj_t = F_alpha + F_beta
        F_pred_kJ = F_fixed_t + F_lj_t
        F_pred_kBT = F_pred_kJ * temp_factors_t.unsqueeze(1)
        loss_tensor = torch.mean((F_pred_kBT - F_target_t) ** 2)

        if not torch.isfinite(loss_tensor):
            print(f"[ERROR] Non-finite loss encountered at iteration {iteration}")
            break

        loss_tensor.backward()

        for name, param in (("sigma", sigma_param), ("epsilon", epsilon_param)):
            if param.grad is not None and not torch.isfinite(param.grad).all():
                print(f"[ERROR] Non-finite gradient for {name} at iteration {iteration}")
                break

        optimizer.step()


        # === 新增：模拟退火噪声 ===
        if args.anneal_init > 0.0:
            # 归一化进度 [0,1]
            t = iteration / float(args.max_iter)
            # 退火曲线：T(t) = T0 * (1 - t^p) + T1 * t^p
            p = float(args.anneal_power)
            tp = t**p
            noise_scale = (1.0 - tp) * float(args.anneal_init) + tp * float(args.anneal_final)

            if noise_scale > 0.0:
                # 噪声尺度按参数取值区间来，避免量纲不一致
                sigma_range = float(args.sigma_max) - float(args.sigma_min)
                epsilon_range = float(args.epsilon_max) - float(args.epsilon_min)

                with torch.no_grad():
                    sigma_param.add_(
                        torch.randn_like(sigma_param) * noise_scale * sigma_range
                    )
                    epsilon_param.add_(
                        torch.randn_like(epsilon_param) * noise_scale * epsilon_range
                    )
                    # 退火之后再 clamp 一次，防止跳出边界
                    sigma_param.clamp_(float(args.sigma_min), float(args.sigma_max))
                    epsilon_param.clamp_(float(args.epsilon_min), float(args.epsilon_max))

        loss_value = float(loss_tensor.item())
        logger.eval_counter += 1

        if loss_value + 1e-12 < best_loss:
            best_loss = loss_value
            best_sigma = sigma_clamped.detach().cpu().numpy()
            best_epsilon = epsilon_clamped.detach().cpu().numpy()

        if iteration % max(int(args.log_interval), 1) == 0:
            pred_np = F_pred_kBT.detach().cpu().numpy().astype(np.float32)
            metrics = compute_metrics(pred_np, target_np)
            logger.log(
                "iteration",
                metrics,
                logger.snapshot_params(
                    sigma_clamped.detach().cpu().numpy(),
                    epsilon_clamped.detach().cpu().numpy(),
                ),
                iteration=iteration,
            )

        if early_stopper.best_loss is math.inf:
            early_stopper.best_loss = loss_value
        if early_stopper.update(loss_value):
            print(
                f"[EARLY STOP GPU] No improvement for {args.patience} iterations; stopping optimisation."
            )
            break

    meta = {
        "method": "TorchAdamLinearized",
        "success": True,
        "message": "Torch optimisation completed",
        "nfev": int(logger.eval_counter),
        "nit": int(logger.eval_counter),
        "device": device.type,
    }
    return best_sigma, best_epsilon, meta


def compute_metrics(pred: NDArray[np.float32], target: NDArray[np.float32]) -> Dict[str, Dict[str, float]]:
    diff = pred - target
    mse_axes = np.mean(diff ** 2, axis=0)
    mse_total = float(np.mean(diff ** 2))
    loss = {
        "x": float(mse_axes[0]),
        "y": float(mse_axes[1]),
        "z": float(mse_axes[2]),
        "total": mse_total,
    }
    r2_axes = {}
    for axis, name in enumerate(["x", "y", "z"]):
        y = target[:, axis]
        ss_res = float(np.sum((pred[:, axis] - y) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        if ss_tot <= 0.0:
            r2_axes[name] = float("nan")
        else:
            r2_axes[name] = float(1.0 - ss_res / ss_tot)
    flat_target = target.reshape(-1)
    flat_pred = pred.reshape(-1)
    ss_res = float(np.sum((flat_pred - flat_target) ** 2))
    ss_tot = float(np.sum((flat_target - np.mean(flat_target)) ** 2))
    if ss_tot <= 0.0:
        r2_total = float("nan")
    else:
        r2_total = float(1.0 - ss_res / ss_tot)
    r2_axes["total"] = r2_total
    return {"loss": loss, "r2": r2_axes}


class EarlyStopException(RuntimeError):
    """Raised internally to abort optimisation once patience is exceeded."""


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 1e-9, initial_loss: float = math.inf):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = initial_loss
        self.steps_since_best = 0

    def update(self, loss_value: float) -> bool:
        if loss_value + self.min_delta < self.best_loss:
            self.best_loss = loss_value
            self.steps_since_best = 0
            return False
        self.steps_since_best += 1
        return self.steps_since_best >= self.patience


class OptimisationLogger:
    def __init__(self, lj_labels: Sequence[str], dihedral_labels: Optional[Sequence[str]] = None):
        self.lj_labels = list(lj_labels)
        self.dihedral_labels = list(dihedral_labels or [])
        self.records: List[Dict[str, object]] = []
        self.eval_counter = 0
        self.best_loss = math.inf
        self.best_params_snapshot: Dict[str, Dict[str, float]] = {}
        self.best_iteration_label: Optional[str] = None

    def snapshot_params(
        self,
        sigma: Sequence[float],
        epsilon: Sequence[float],
        dihedrals: Optional[Sequence[Sequence[float]]] = None,
    ) -> Dict[str, Dict[str, float]]:
        out = {}
        for label, s, e in zip(self.lj_labels, sigma, epsilon):
            out[label] = {"sigma": float(s), "epsilon": float(e)}
        if dihedrals is not None:
            for label, coeffs in zip(self.dihedral_labels, dihedrals):
                out[label] = {f"c{i+1}": float(v) for i, v in enumerate(coeffs)}
        return out

    def log(self, stage: str, metrics: Dict[str, Dict[str, float]], params: Dict[str, Dict[str, float]], iteration: Optional[int] = None) -> None:
        record = {
            "stage": stage,
            "iteration": iteration,
            "loss": metrics["loss"],
            "r2": metrics["r2"],
            "params": params,
        }
        self.records.append(record)
        param_parts: List[str] = []
        for label in self.lj_labels:
            slot = params.get(label)
            if not slot:
                continue
            param_parts.append(
                f"{label}(sigma={slot['sigma']:.4f}, epsilon={slot['epsilon']:.4f})"
            )
        for label in self.dihedral_labels:
            slot = params.get(label)
            if not slot:
                continue
            coeff_desc = ", ".join(f"c{i+1}={slot.get(f'c{i+1}', float('nan')):.3f}" for i in range(6))
            param_parts.append(f"{label}({coeff_desc})")
        msg = f"[{stage}] loss_total={metrics['loss']['total']:.6f} r2_total={metrics['r2']['total']}"
        if iteration is not None:
            msg = f"{msg} (iter={iteration})"
        if param_parts:
            msg = f"{msg} | " + ", ".join(param_parts)
        print(msg)
        self._update_best(metrics["loss"]["total"], params, iteration, stage)
        self._print_best()

    def _update_best(
        self,
        loss_value: float,
        params: Dict[str, Dict[str, float]],
        iteration: Optional[int],
        stage: str,
    ) -> None:
        if loss_value >= self.best_loss - 1e-12:
            return
        self.best_loss = loss_value
        self.best_params_snapshot = params
        if iteration is None:
            self.best_iteration_label = stage
        else:
            self.best_iteration_label = str(iteration)

    def _print_best(self) -> None:
        if not self.best_params_snapshot:
            return
        parts = []
        for label in self.labels:
            params = self.best_params_snapshot.get(label)
            if not params:
                continue
            parts.append(
                f"{label}(sigma={params['sigma']:.4f}, epsilon={params['epsilon']:.4f})"
            )
        iter_label = self.best_iteration_label or "n/a"
        print(
            f"[best_so_far] loss_total={self.best_loss:.6f} @iter={iter_label}: "
            + ", ".join(parts)
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimise LJ parameters to reproduce centre forces")
    parser.add_argument("--data", type=Path, default=Path("../data.json"), help="Dataset JSON containing target forces")
    parser.add_argument("--summary", type=Path, default=Path("topology_summary.json"), help="Parameter summary JSON")
    parser.add_argument("--bond-csv", type=Path, default=Path("center_bond_forces.csv"), help="CSV file with bond forces (kJ/mol/nm)")
    parser.add_argument("--angle-csv", type=Path, default=Path("center_angle_forces.csv"), help="CSV file with angle forces (kJ/mol/nm)")
    parser.add_argument("--coulomb-csv", type=Path, default=Path("center_coulomb_forces.csv"), help="CSV cache for Coulomb forces (kJ/mol/nm)")
    parser.add_argument("--dihedral-csv", type=Path, default=Path("center_dihedral_forces.csv"), help="CSV cache for dihedral forces (kJ/mol/nm)")
    parser.add_argument("--output-summary", type=Path, default=Path("topology_summary.optimised.json"), help="Where to write the updated summary JSON")
    parser.add_argument("--log", type=Path, default=Path("optimization_log.json"), help="Where to store optimisation metrics")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of dataset entries (for debugging)")
    parser.add_argument("--rcoul", type=float, default=1.2, help="Coulomb cutoff distance (nm)")
    parser.add_argument("--rvdw", type=float, default=1.2, help="LJ cutoff distance (nm)")
    parser.add_argument("--sigma-min", type=float, default=0.1)
    parser.add_argument("--sigma-max", type=float, default=0.5)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--epsilon-max", type=float, default=10.0)
    parser.add_argument("--optimize-dihedrals", action="store_true")
    parser.add_argument("--dihedral-min", type=float, default=-10.0)
    parser.add_argument("--dihedral-max", type=float, default=10.0)
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default="gpu",
        help="Device for LJ force evaluation (GPU requires torch with CUDA)",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-2, help="Learning rate for the GPU optimiser")
    parser.add_argument("--patience", type=int, default=1000, help="Early stopping patience for both optimisers")
    parser.add_argument("--log-interval", type=int, default=100, help="Iterations between optimisation log entries")
    # === 新增：退火相关参数 ===
    parser.add_argument(
        "--anneal-init",
        type=float,
        default=0.02,
        help="Initial noise scale for simulated annealing (0 to disable)",
    )
    parser.add_argument(
        "--anneal-final",
        type=float,
        default=0.0,
        help="Final noise scale for simulated annealing",
    )
    parser.add_argument(
        "--anneal-power",
        type=float,
        default=1.0,
        help="Annealing schedule power (1.0=线性, 2.0=快降, <1=慢降)",
    )

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    summary = load_json(args.summary)
    if not isinstance(summary, MutableMapping):
        raise ValueError("summary JSON must be an object")

    raw_data = load_json(args.data)
    if not isinstance(raw_data, Sequence):
        raise ValueError("dataset JSON must be an array")

    bond_forces = load_force_map(args.bond_csv, "bond")
    angle_forces = load_force_map(args.angle_csv, "angle")


    entries = build_entries(raw_data, bond_forces, angle_forces, limit=args.limit)
    if not entries:
        raise ValueError("No valid entries found in dataset")

    coulomb_cache = cache_force_component(
        entries,
        summary,
        args.coulomb_csv,
        "coulomb",
        lambda top, coords: compute_nonbonded_forces(
            top,
            coords,
            rcoul=args.rcoul,
            rvdw=0.0,
            do_lj=False,
            do_coul=True,
            device="gpu",
        ),
    )
    dihedral_cache = cache_force_component(
        entries,
        summary,
        args.dihedral_csv,
        "dihedral",
        lambda top, coords: compute_dihedral_forces(top, coords),
    )
    apply_cached_component(entries, coulomb_cache, "coulomb_force_kj")
    apply_cached_component(entries, dihedral_cache, "dihedral_force_kj")

    lj_targets = select_optimisable_lj(summary, fix_hydroxyl_h=True)
    dihedral_targets = select_optimisable_dihedrals(summary) if args.optimize_dihedrals else []
    if args.optimize_dihedrals and not dihedral_targets:
        raise ValueError("No dihedral parameters available for optimisation")
    sigma0 = np.array([item.sigma for item in lj_targets], dtype=float)
    epsilon0 = np.array([item.epsilon for item in lj_targets], dtype=float)
    dihedral0 = np.array([item.coeffs for item in dihedral_targets], dtype=float)

    apply_lj_parameters(summary, lj_targets, sigma0, epsilon0)
    if dihedral_targets:
        apply_dihedral_parameters(summary, dihedral_targets, dihedral0)
    targets = np.vstack([entry.target_force for entry in entries])

    prepared_entries = prepare_entries_for_prediction(
        summary, entries, device=args.device, lj_targets=lj_targets
    )
    refresh_prepared_topologies(prepared_entries, summary)
    linear_system, sigma_all_summary, epsilon_all_summary = build_linearized_lj_system(
        prepared_entries, summary, args.rvdw
    )
    trainable_summary_indices = [slot.summary_index for slot in lj_targets]

    logger = OptimisationLogger([item.label for item in lj_targets], [d.label for d in dihedral_targets])

    initial_pred = predict_forces(
        summary,
        prepared_entries,
        rvdw=args.rvdw,
        device=args.device,
        recompute_dihedral=bool(dihedral_targets),
    )
    initial_metrics = compute_metrics(initial_pred, targets)
    logger.log("initial", initial_metrics, logger.snapshot_params(sigma0, epsilon0, dihedral0))

    use_torch_optimizer = torch is not None and not dihedral_targets
    result_meta: Dict[str, object]
    if use_torch_optimizer:
        sigma_opt, epsilon_opt, result_meta = run_linearized_torch_optimizer(
            linear_system,
            sigma_all_summary,
            epsilon_all_summary,
            trainable_summary_indices,
            sigma0,
            epsilon0,
            args,
            logger,
        )
    else:
        bounds: List[Tuple[float, float]] = []
        for _ in lj_targets:
            bounds.append((args.sigma_min, args.sigma_max))
        for _ in lj_targets:
            bounds.append((args.epsilon_min, args.epsilon_max))
        for _ in dihedral_targets:
            bounds.extend([(args.dihedral_min, args.dihedral_max)] * 6)

        def _pack(
            s: NDArray[np.float32], e: NDArray[np.float32], d: Optional[NDArray[np.float32]] = None
        ) -> NDArray[np.float32]:
            parts: List[np.ndarray] = [s, e]
            if dihedral_targets:
                d_arr = dihedral0 if d is None else np.asarray(d, dtype=float)
                parts.append(d_arr.reshape(-1))
            return np.concatenate(parts)

        def _unpack(
            vec: Sequence[float],
        ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
            vec = np.asarray(vec, dtype=float)
            n = len(lj_targets)
            s = vec[:n]
            e = vec[n : 2 * n]
            d: NDArray[np.float32] = np.empty((0, 6), dtype=float)
            if dihedral_targets:
                d = vec[2 * n :].reshape((-1, 6))
            return s, e, d

        best_state = {
            "loss": float(initial_metrics["loss"]["total"]),
            "vec": _pack(sigma0, epsilon0, dihedral0),
        }
        early_stopper = EarlyStopping(
            patience=int(args.patience), initial_loss=best_state["loss"], min_delta=1e-9
        )

        def objective(vec: Sequence[float]) -> float:
            sigma, epsilon, dihedrals = _unpack(vec)
            apply_lj_parameters(summary, lj_targets, sigma, epsilon)
            if dihedral_targets:
                apply_dihedral_parameters(summary, dihedral_targets, dihedrals)
            refresh_prepared_topologies(prepared_entries, summary)
            pred = predict_forces(
                summary,
                prepared_entries,
                rvdw=args.rvdw,
                device=args.device,
                recompute_dihedral=bool(dihedral_targets),
            )
            metrics = compute_metrics(pred, targets)
            logger.eval_counter += 1
            if logger.eval_counter % max(int(args.log_interval), 1) == 0:
                logger.log(
                    "iteration",
                    metrics,
                    logger.snapshot_params(sigma, epsilon, dihedrals if dihedral_targets else None),
                    iteration=logger.eval_counter,
                )

            loss_value = metrics["loss"]["total"]
            if loss_value < best_state["loss"] - 1e-12:
                best_state["loss"] = loss_value
                best_state["vec"] = _pack(sigma, epsilon, dihedrals)
            if early_stopper.update(loss_value):
                raise EarlyStopException(
                    f"No improvement for {args.patience} evaluations; stopping early."
                )
            return loss_value

        try:
            result = minimize(
                objective,
                _pack(sigma0, epsilon0, dihedral0),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": args.max_iter, "disp": True},
            )
        except EarlyStopException as exc:
            print(f"[EARLY STOP] {exc}")
            result = SimpleNamespace(
                x=best_state["vec"],
                success=False,
                message=str(exc),
                nfev=int(logger.eval_counter),
                nit=int(logger.eval_counter),
            )

        sigma_opt, epsilon_opt, dihedral_opt = _unpack(best_state["vec"])
        result_meta = {
            "method": "L-BFGS-B",
            "success": bool(result.success),
            "message": result.message,
            "nfev": int(result.nfev),
            "nit": int(result.nit),
        }
    apply_lj_parameters(summary, lj_targets, sigma_opt, epsilon_opt)
    if dihedral_targets:
        apply_dihedral_parameters(summary, dihedral_targets, dihedral_opt)
    refresh_prepared_topologies(prepared_entries, summary)
    final_pred = predict_forces(
        summary,
        prepared_entries,
        rvdw=args.rvdw,
        device=args.device,
        recompute_dihedral=bool(dihedral_targets),
    )
    final_metrics = compute_metrics(final_pred, targets)
    logger.log(
        "final",
        final_metrics,
        logger.snapshot_params(sigma_opt, epsilon_opt, dihedral_opt if dihedral_targets else None),
    )

    with args.output_summary.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")

    log_payload = {
        "meta": result_meta,
        "history": logger.records,
    }
    with args.log.open("w", encoding="utf-8") as fh:
        json.dump(log_payload, fh, indent=2)
        fh.write("\n")

    print(f"[DONE] Optimised summary written to {args.output_summary}")
    print(f"[DONE] Optimisation log written to {args.log}")
    # Optional: plot loss and R^2 vs iteration
    try:
        import matplotlib
        matplotlib.use("Agg")  # 💡 关键：切到非交互式后端
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        print(f"[WARN] matplotlib is not available or backend failed: {e}")
    else:
        # 从 logger.records 里取出所有 'iteration' 阶段的数据
        steps = []
        losses = []
        r2_values = []
        for rec in logger.records:
            if rec.get("stage") != "iteration":
                continue
            it = rec.get("iteration")
            if it is None:
                continue
            steps.append(int(it))
            losses.append(float(rec["loss"]["total"]))
            r2_values.append(float(rec["r2"]["total"]))

        if steps:
            # 按 step 排序一下
            order = sorted(range(len(steps)), key=lambda i: steps[i])
            steps_sorted = [steps[i] for i in order]
            losses_sorted = [losses[i] for i in order]
            r2_sorted = [r2_values[i] for i in order]

            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))

            ax1.plot(steps_sorted, losses_sorted, marker="o")
            ax1.set_ylabel("Loss (total)")
            ax1.grid(True, alpha=0.3)

            ax2.plot(steps_sorted, r2_sorted, marker="o")
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("R² (total)")
            ax2.grid(True, alpha=0.3)

            fig.tight_layout()
            out_png = args.log.with_suffix(".png")
            fig.savefig(out_png, dpi=150)
            print(f"[DONE] Saved optimisation plot to {out_png}")
        else:
            print("[WARN] No iteration records found; skipping plots.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
