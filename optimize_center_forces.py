#!/usr/bin/env python3
"""Optimize Lennard-Jones parameters to match centre-atom forces."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

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
    coords: NDArray[np.float64]
    charges: NDArray[np.float64]
    center_index: int
    temperature: float
    target_force: NDArray[np.float64]
    bond_force_kj: NDArray[np.float64]
    angle_force_kj: NDArray[np.float64]
    box: NDArray[np.float64]


@dataclass
class OptimisableLJEntry:
    summary_index: int
    label: str
    sigma: float
    epsilon: float


def load_json(path: Path) -> Mapping[str, object] | Sequence[object]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_force_map(csv_path: Path, label: str) -> Dict[Tuple[str, int], NDArray[np.float64]]:
    import csv

    fmap: Dict[Tuple[str, int], NDArray[np.float64]] = {}
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
    coords: NDArray[np.float64],
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


def _as_box(entry: Mapping[str, object]) -> NDArray[np.float64]:
    box = entry.get("box")
    if isinstance(box, Sequence) and len(box) == 3:
        arr = np.asarray(box, dtype=float).reshape(3,)
        if np.all(np.isfinite(arr)):
            return arr
    return DEFAULT_BOX.copy()

def build_entries(
    data: Sequence[Mapping[str, object]],
    bond_forces: Mapping[Tuple[str, int], NDArray[np.float64]],
    angle_forces: Mapping[Tuple[str, int], NDArray[np.float64]],
    limit: Optional[int] = None,
) -> List[CenterForceEntry]:
    entries: List[CenterForceEntry] = []
    for idx, item in enumerate(data):
        if limit is not None and len(entries) >= limit:
            break
        status = str(item.get("center_force_status", "")).lower()
        if status and status not in {"ok", "good", "done"}:
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


def compute_nonbonded_forces(
    top: Topology,
    coords_in,
    rcoul: float = 1.2,
    rvdw: float = 1.2,
    do_lj: bool = True,
    do_coul: bool = True,
) -> np.ndarray:
    """
    只计算非键相互作用的力 (LJ / 库伦)，单位 kJ/mol/nm。
    使用 do_lj / do_coul 控制是否计算对应项。
    """
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
        diff = coords[i + 1:] - ri                      # (m,3)
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

            # LJ
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

            # Coulomb
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
    entries: Sequence[CenterForceEntry],
    rcoul: float,
    rvdw: float,
) -> NDArray[np.float64]:
    """
    用和 compare_plot_csv.py 一致的方式计算预测力。

    - LJ：compute_nonbonded_forces(..., do_lj=True,  do_coul=False, rvdw=rvdw, rcoul=0)
    - COUL：compute_nonbonded_forces(..., do_lj=False, do_coul=True,  rvdw=0,    rcoul=rcoul)
    - DIH：compute_dihedral_forces(...)
    - 再加 bond / angle（来自 CSV，单位 kJ/mol/nm）
    - 最后结果转成 kBT/nm
    """
    predictions: List[NDArray[np.float64]] = []

    for entry in entries:
        # 1. 用 summary + 当前坐标/原子名 推拓扑
        top = infer_topology_from_summary(summary, entry.coords, entry.atom_types)

        # 2. 覆盖电荷，使之与 JSON 中 formal_charges 一致
        for atom, charge in zip(top.atoms, entry.charges):
            atom.charge = float(charge)

        T = float(entry.temperature)
        factor = 1.0 / (R_KJ_PER_MOL_K * max(T, 1e-6))  # = 1/kBT

        # 3. LJ 力：只算 LJ，不算库伦
        forces_lj_kj = compute_nonbonded_forces(
            top,
            entry.coords,
            rcoul=0.0,      # 不算库仑
            rvdw=rvdw,      # LJ 截断由参数控制
            do_lj=True,
            do_coul=False,
        )

        # 4. Coulomb 力：只算库仑，不算 LJ
        forces_coul_kj = compute_nonbonded_forces(
            top,
            entry.coords,
            rcoul=rcoul,    # 库仑截断
            rvdw=0.0,       # 不算 LJ
            do_lj=False,
            do_coul=True,
        )

        # 5. Dihedral 力
        forces_dih_kj = compute_dihedral_forces(top, entry.coords)

        # 6. 在中心原子上把三项加起来，再加 bond/angle
        idx = entry.center_index
        total_nb_kj = (
            forces_lj_kj[idx]
            + forces_coul_kj[idx]
            + forces_dih_kj[idx]
        )
        total_kj = (
            total_nb_kj
            + entry.bond_force_kj
            + entry.angle_force_kj
        )

        # 7. 转单位为 kBT/nm
        predictions.append(total_kj * factor)

    return np.vstack(predictions)

def compute_metrics(pred: NDArray[np.float64], target: NDArray[np.float64]) -> Dict[str, Dict[str, float]]:
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


class OptimisationLogger:
    def __init__(self, labels: Sequence[str]):
        self.labels = list(labels)
        self.records: List[Dict[str, object]] = []
        self.eval_counter = 0

    def snapshot_params(
        self, sigma: Sequence[float], epsilon: Sequence[float]
    ) -> Dict[str, Dict[str, float]]:
        out = {}
        for label, s, e in zip(self.labels, sigma, epsilon):
            out[label] = {"sigma": float(s), "epsilon": float(e)}
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
        msg = f"[{stage}] loss_total={metrics['loss']['total']:.6f} r2_total={metrics['r2']['total']}"
        if iteration is not None:
            msg = f"{msg} (iter={iteration})"
        print(msg)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimise LJ parameters to reproduce centre forces")
    parser.add_argument("--data", type=Path, default=Path("../data.json"), help="Dataset JSON containing target forces")
    parser.add_argument("--summary", type=Path, default=Path("topology_summary.json"), help="Parameter summary JSON")
    parser.add_argument("--bond-csv", type=Path, default=Path("center_bond_forces.csv"), help="CSV file with bond forces (kJ/mol/nm)")
    parser.add_argument("--angle-csv", type=Path, default=Path("center_angle_forces.csv"), help="CSV file with angle forces (kJ/mol/nm)")
    parser.add_argument("--output-summary", type=Path, default=Path("topology_summary.optimised.json"), help="Where to write the updated summary JSON")
    parser.add_argument("--log", type=Path, default=Path("optimization_log.json"), help="Where to store optimisation metrics")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of dataset entries (for debugging)")
    parser.add_argument("--rcoul", type=float, default=1.2, help="Coulomb cutoff distance (nm)")
    parser.add_argument("--rvdw", type=float, default=1.2, help="LJ cutoff distance (nm)")
    parser.add_argument("--sigma-min", type=float, default=0.1)
    parser.add_argument("--sigma-max", type=float, default=0.5)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--epsilon-max", type=float, default=5.0)
    parser.add_argument("--max-iter", type=int, default=100)
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

    lj_targets = select_optimisable_lj(summary, fix_hydroxyl_h=True)
    sigma0 = np.array([item.sigma for item in lj_targets], dtype=float)
    epsilon0 = np.array([item.epsilon for item in lj_targets], dtype=float)

    apply_lj_parameters(summary, lj_targets, sigma0, epsilon0)
    targets = np.vstack([entry.target_force for entry in entries])

    logger = OptimisationLogger([item.label for item in lj_targets])

    initial_pred = predict_forces(summary, entries, rcoul=args.rcoul, rvdw=args.rvdw)
    initial_metrics = compute_metrics(initial_pred, targets)
    logger.log("initial", initial_metrics, logger.snapshot_params(sigma0, epsilon0))

    bounds: List[Tuple[float, float]] = []
    for _ in lj_targets:
        bounds.append((args.sigma_min, args.sigma_max))
    for _ in lj_targets:
        bounds.append((args.epsilon_min, args.epsilon_max))

    def _pack(s: NDArray[np.float64], e: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.concatenate([s, e])

    def _unpack(vec: Sequence[float]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        vec = np.asarray(vec, dtype=float)
        n = len(lj_targets)
        return vec[:n], vec[n:]

    def objective(vec: Sequence[float]) -> float:
        sigma, epsilon = _unpack(vec)
        apply_lj_parameters(summary, lj_targets, sigma, epsilon)
        pred = predict_forces(summary, entries, rcoul=args.rcoul, rvdw=args.rvdw)
        metrics = compute_metrics(pred, targets)
        logger.eval_counter += 1
        logger.log(
            "iteration",
            metrics,
            logger.snapshot_params(sigma, epsilon),
            iteration=logger.eval_counter,
        )
        return metrics["loss"]["total"]

    result = minimize(
        objective,
        _pack(sigma0, epsilon0),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": args.max_iter, "disp": True},
    )

    sigma_opt, epsilon_opt = _unpack(result.x)
    apply_lj_parameters(summary, lj_targets, sigma_opt, epsilon_opt)
    final_pred = predict_forces(summary, entries, rcoul=args.rcoul, rvdw=args.rvdw)
    final_metrics = compute_metrics(final_pred, targets)
    logger.log("final", final_metrics, logger.snapshot_params(sigma_opt, epsilon_opt))

    with args.output_summary.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")

    log_payload = {
        "meta": {
            "method": "L-BFGS-B",
            "success": bool(result.success),
            "message": result.message,
            "nfev": int(result.nfev),
            "nit": int(result.nit),
        },
        "history": logger.records,
    }
    with args.log.open("w", encoding="utf-8") as fh:
        json.dump(log_payload, fh, indent=2)
        fh.write("\n")

    print(f"[DONE] Optimised summary written to {args.output_summary}")
    print(f"[DONE] Optimisation log written to {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
