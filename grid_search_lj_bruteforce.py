#!/usr/bin/env python3
"""Brute-force grid search of Lennard-Jones parameters for selected elements."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import itertools
import math

import numpy as np

try:  # Optional dependency for GPU backend
    import torch
except Exception:  # pragma: no cover - torch is optional unless GPU backend is requested
    torch = None

import min as ljfit


BOLTZMANN_KJ_PER_MOL_K = 0.00831446261815324
SEARCH_ELEMENTS = ("C", "H", "O", "O_h")
FIXED_ZERO_ELEMENT = "H_h"

DEFAULT_EPSILON_GRID = {
    elem: [round(0.2 * i, 10) for i in range(11)] for elem in SEARCH_ELEMENTS
}
DEFAULT_SIGMA_GRID = {
    elem: [round(0.1 + 0.05 * i, 10) for i in range(9)] for elem in SEARCH_ELEMENTS
}


@dataclass(frozen=True)
class PairMetadata:
    idx_i: np.ndarray
    idx_j: np.ndarray
    idx_i_safe: np.ndarray
    idx_j_safe: np.ndarray
    active_mask: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search for Lennard-Jones parameters")
    parser.add_argument("--data", default="/data/luo/mlff/model/cho_allmodel/data.json", type=Path, help="Path to dataset JSON")
    parser.add_argument("--summary", default="topology_summary.json", type=Path, help="Path to topology summary JSON")
    parser.add_argument("--element-map", type=str, default="", help="Custom atom type -> element mapping")
    parser.add_argument("--bond-csv", type=Path, default="center_bond_forces.csv", help="CSV file with bond forces (kJ/mol/nm by default)")
    parser.add_argument("--angle-csv", type=Path, default="center_angle_forces.csv", help="CSV file with angle forces (kJ/mol/nm by default)")
    parser.add_argument("--bond-angle-in-kbt", action="store_true", help="Bond/angle CSV forces are provided in kBT/nm")
    parser.add_argument(
        "--epsilon-grid",
        type=str,
        default="",
        help="Per-element epsilon grid overrides, format Element:start:stop:step",
    )
    parser.add_argument(
        "--sigma-grid",
        type=str,
        default="",
        help="Per-element sigma grid overrides, format Element:start:stop:step",
    )
    parser.add_argument("--max-pairs-per-entry", type=int, default=None, help="Limit for LJ pairs per entry")
    parser.add_argument("--fudge-lj", type=float, default=ljfit.DEFAULT_FUDGE_LJ, help="Fudge factor for 1-4 LJ")
    parser.add_argument("--fudge-qq", type=float, default=ljfit.DEFAULT_FUDGE_QQ, help="Fudge factor for 1-4 Coulomb")
    parser.add_argument("--ke", type=float, default=ljfit.DEFAULT_KE, help="Coulomb constant (kJ/mol)")
    parser.add_argument("--temperature", type=float, default=298.15, help="Temperature in Kelvin")
    parser.add_argument("--backend", choices=("cpu-mp", "gpu"), default="cpu-mp", help="Execution backend")
    parser.add_argument("--n-procs", type=int, default=None, help="Number of worker processes for CPU backend")
    parser.add_argument("--chunk-size", type=int, default=32, help="Parameter combinations per CPU task chunk")
    parser.add_argument("--gpu-batch-size", type=int, default=512, help="Batch size for GPU backend evaluation")
    parser.add_argument("--num-shards", type=int, default=1, help="Number of shards to split the parameter space")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard index for this run")
    parser.add_argument(
        "--global-metrics",
        action="store_true",
        help="Compute aggregate metrics across all entries for each parameter combination",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Only compute global metrics; skip per-entry and grouped best-parameter search and outputs",
    )
    parser.add_argument(
        "--metrics-threshold",
        type=float,
        default=0.8,
        help="Minimum R^2 required to record aggregate metrics",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Optional output file for aggregate metrics (defaults to <out-prefix>.metrics.json)",
    )
    parser.add_argument("--out-prefix", required=True, type=Path, help="Prefix for output JSON files")
    return parser.parse_args()


def parse_element_map(element_map_str: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not element_map_str:
        return mapping
    for item in element_map_str.split(","):
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid element mapping entry '{item}', expected 'type=Element'")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Invalid element mapping entry '{item}'")
        mapping[key] = value
    return mapping


def _generate_range(start: float, stop: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("Grid step must be positive")
    if stop < start:
        raise ValueError("Grid stop must be greater or equal to start")
    values: List[float] = []
    current = start
    while current <= stop + 1e-12:
        values.append(round(current, 10))
        current += step
    if not values:
        raise ValueError("Grid definition produced no values")
    return values


def parse_grid_config(
    config_str: str,
    defaults: Mapping[str, Sequence[float]],
    label: str,
) -> Dict[str, List[float]]:
    grid = {elem: list(values) for elem, values in defaults.items()}
    if not config_str:
        return grid

    normalized = config_str.replace(";", ",")
    for item in normalized.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 4:
            raise ValueError(
                f"Invalid {label} grid specification '{item}', expected Element:start:stop:step"
            )
        elem = parts[0].strip()
        if elem not in defaults:
            raise ValueError(
                f"Unknown element '{elem}' in {label} grid specification; expected one of {list(defaults.keys())}"
            )
        try:
            start = float(parts[1])
            stop = float(parts[2])
            step = float(parts[3])
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value in {label} grid specification '{item}'") from exc
        grid[elem] = _generate_range(start, stop, step)

    return grid


def load_bond_angle_maps(
    bond_csv: Optional[Path],
    angle_csv: Optional[Path],
    kbt: float,
    bond_angle_in_kbt: bool,
) -> Tuple[Dict[Tuple[str, int], np.ndarray], Dict[Tuple[str, int], np.ndarray]]:
    if bond_csv is not None:
        bond_map = ljfit.load_force_map_from_csv(str(bond_csv), "bond")
    else:
        bond_map = {}

    if angle_csv is not None:
        angle_map = ljfit.load_force_map_from_csv(str(angle_csv), "angle")
    else:
        angle_map = {}

    if bond_angle_in_kbt:
        scale = float(kbt)
        for fmap in (bond_map, angle_map):
            for key, vec in list(fmap.items()):
                fmap[key] = np.asarray(vec, dtype=float) * scale
    else:
        for fmap in (bond_map, angle_map):
            for key, vec in list(fmap.items()):
                fmap[key] = np.asarray(vec, dtype=float)

    return bond_map, angle_map


def apply_external_forces(
    entries: Sequence[ljfit.ProcessedEntry],
    dihedral_map: Mapping[Tuple[Optional[str], int], np.ndarray],
    bond_map: Mapping[Tuple[Optional[str], int], np.ndarray],
    angle_map: Mapping[Tuple[Optional[str], int], np.ndarray],
) -> None:
    for entry in entries:
        key = (entry.gro_file, int(entry.center_index))
        if key in dihedral_map:
            entry.dihedral_force = np.asarray(dihedral_map[key], dtype=float)
        else:
            entry.dihedral_force = np.zeros(3, dtype=float)

        if key in bond_map:
            entry.bond_force = np.asarray(bond_map[key], dtype=float)
        else:
            entry.bond_force = np.zeros(3, dtype=float)

        if key in angle_map:
            entry.angle_force = np.asarray(angle_map[key], dtype=float)
        else:
            entry.angle_force = np.zeros(3, dtype=float)

        entry.bonded_force = entry.bond_force + entry.angle_force + entry.dihedral_force


def precompute_aggregates(
    entries: Sequence[ljfit.ProcessedEntry],
    entry_pairs: Sequence[Sequence[Mapping[str, object]]],
    pair_type_list: Sequence[Tuple[str, str]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_entries = len(entries)
    n_pairs = len(pair_type_list)
    pair_index = {pair: idx for idx, pair in enumerate(pair_type_list)}

    A_alpha = np.zeros((n_entries, n_pairs, 3), dtype=np.float64)
    A_beta = np.zeros((n_entries, n_pairs, 3), dtype=np.float64)
    F_fixed_kJ = np.zeros((n_entries, 3), dtype=np.float64)
    F_target_kBT = np.zeros((n_entries, 3), dtype=np.float64)

    for entry_idx, entry in enumerate(entries):
        F_fixed_kJ[entry_idx] = (
            np.asarray(entry.bond_force, dtype=float)
            + np.asarray(entry.angle_force, dtype=float)
            + np.asarray(entry.dihedral_force, dtype=float)
            + np.asarray(entry.coulomb_force, dtype=float)
        )
        F_target_kBT[entry_idx] = np.asarray(entry.target_force, dtype=float)

        for pair in entry_pairs[entry_idx]:
            p_idx = pair_index[pair["pair_type"]]
            A_alpha[entry_idx, p_idx, :] += np.asarray(pair["coeff_alpha"], dtype=float)
            A_beta[entry_idx, p_idx, :] += np.asarray(pair["coeff_beta"], dtype=float)

    return A_alpha, A_beta, F_fixed_kJ, F_target_kBT


def build_parameter_space(
    epsilon_spec: str,
    sigma_spec: str,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], List[List[float]], List[List[float]]]:
    epsilon_grid = parse_grid_config(epsilon_spec, DEFAULT_EPSILON_GRID, "epsilon")
    sigma_grid = parse_grid_config(sigma_spec, DEFAULT_SIGMA_GRID, "sigma")
    epsilon_lists = [epsilon_grid[elem] for elem in SEARCH_ELEMENTS]
    sigma_lists = [sigma_grid[elem] for elem in SEARCH_ELEMENTS]
    return epsilon_grid, sigma_grid, epsilon_lists, sigma_lists


def prepare_pair_metadata(pair_type_list: Sequence[Tuple[str, str]]) -> PairMetadata:
    n_pairs = len(pair_type_list)
    idx_map = {elem: idx for idx, elem in enumerate(SEARCH_ELEMENTS)}
    idx_i = np.full(n_pairs, -1, dtype=np.int64)
    idx_j = np.full(n_pairs, -1, dtype=np.int64)
    active_mask = np.ones(n_pairs, dtype=bool)

    for p_idx, (elem_i, elem_j) in enumerate(pair_type_list):
        if elem_i == FIXED_ZERO_ELEMENT or elem_j == FIXED_ZERO_ELEMENT:
            active_mask[p_idx] = False
        idx_i_val = idx_map.get(elem_i, -1)
        idx_j_val = idx_map.get(elem_j, -1)
        idx_i[p_idx] = idx_i_val
        idx_j[p_idx] = idx_j_val
        if idx_i_val < 0 or idx_j_val < 0:
            active_mask[p_idx] = False

    idx_i_safe = np.where(idx_i >= 0, idx_i, 0)
    idx_j_safe = np.where(idx_j >= 0, idx_j, 0)

    return PairMetadata(
        idx_i=idx_i,
        idx_j=idx_j,
        idx_i_safe=idx_i_safe,
        idx_j_safe=idx_j_safe,
        active_mask=active_mask,
    )


def decode_index_to_values(index: int, value_lists: Sequence[Sequence[float]]) -> Tuple[float, ...]:
    result: List[float] = [0.0] * len(value_lists)
    for pos in range(len(value_lists) - 1, -1, -1):
        values = value_lists[pos]
        base = len(values)
        if base <= 0:
            raise ValueError("Each element must have at least one grid value")
        digit = index % base
        index //= base
        result[pos] = float(values[digit])
    return tuple(result)


def parameter_generator_shard(
    epsilon_lists: Sequence[Sequence[float]],
    sigma_lists: Sequence[Sequence[float]],
    shard_start: int,
    shard_end: int,
) -> Iterator[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
    n_elem = len(SEARCH_ELEMENTS)
    sigma_combinations = math.prod(len(values) for values in sigma_lists)
    for linear_idx in range(shard_start, shard_end):
        eps_idx = linear_idx // sigma_combinations
        sig_idx = linear_idx % sigma_combinations
        eps_combo = decode_index_to_values(eps_idx, epsilon_lists)
        sig_combo = decode_index_to_values(sig_idx, sigma_lists)
        yield eps_combo, sig_combo


def gather_all_elements(pair_type_list: Sequence[Tuple[str, str]]) -> List[str]:
    element_set = set()
    for a, b in pair_type_list:
        element_set.add(a)
        element_set.add(b)
    return sorted(element_set)


def make_param_dict(
    eps_combo: Sequence[float],
    sig_combo: Sequence[float],
    zero_only_elements: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    params: Dict[str, Dict[str, float]] = {}
    for idx, elem in enumerate(SEARCH_ELEMENTS):
        params[elem] = {"epsilon": float(eps_combo[idx]), "sigma": float(sig_combo[idx])}
    params[FIXED_ZERO_ELEMENT] = {"epsilon": 0.0, "sigma": 0.0}
    for elem in zero_only_elements:
        if elem not in params:
            params[elem] = {"epsilon": 0.0, "sigma": 0.0}
    return params


def compute_alpha_beta_single(
    eps_combo: Sequence[float],
    sig_combo: Sequence[float],
    metadata: PairMetadata,
) -> Tuple[np.ndarray, np.ndarray]:
    eps_arr = np.asarray(eps_combo, dtype=np.float64)
    sig_arr = np.asarray(sig_combo, dtype=np.float64)

    eps_i = eps_arr[metadata.idx_i_safe]
    eps_j = eps_arr[metadata.idx_j_safe]
    sig_i = sig_arr[metadata.idx_i_safe]
    sig_j = sig_arr[metadata.idx_j_safe]

    valid = (
        metadata.active_mask
        & (eps_i > 0.0)
        & (eps_j > 0.0)
        & (sig_i > 0.0)
        & (sig_j > 0.0)
    )

    alpha = np.zeros(metadata.idx_i.shape[0], dtype=np.float64)
    beta = np.zeros_like(alpha)
    if not np.any(valid):
        return alpha, beta

    eps_mix = np.sqrt(eps_i[valid] * eps_j[valid])
    sig_mix = np.sqrt(sig_i[valid] * sig_j[valid])
    sigma6 = sig_mix ** 6
    beta_vals = 4.0 * eps_mix * sigma6
    alpha_vals = beta_vals * sigma6
    alpha[valid] = alpha_vals
    beta[valid] = beta_vals
    return alpha, beta


# --- CPU multiprocessing backend helpers ----------------------------------------------------

_CPU_SHARED: Dict[str, object] = {}


def _init_cpu_worker(shared: Dict[str, object]) -> None:
    _CPU_SHARED.update(shared)


def _format_params_for_output(eps_combo: Sequence[float], sig_combo: Sequence[float]) -> Dict[str, Dict[str, float]]:
    zero_only_elements: Sequence[str] = _CPU_SHARED.get("zero_only_elements", [])
    return make_param_dict(eps_combo, sig_combo, zero_only_elements)


def _compute_entry_errors(
    eps_combo: Sequence[float],
    sig_combo: Sequence[float],
) -> Tuple[np.ndarray, Dict[str, Dict[str, float]], Optional[Dict[str, float]]]:
    metadata: PairMetadata = _CPU_SHARED["pair_metadata"]
    A_alpha: np.ndarray = _CPU_SHARED["A_alpha"]
    A_beta: np.ndarray = _CPU_SHARED["A_beta"]
    F_fixed_kJ: np.ndarray = _CPU_SHARED["F_fixed_kJ"]
    F_target_kBT: np.ndarray = _CPU_SHARED["F_target_kBT"]
    kbt: float = _CPU_SHARED["kbt"]
    collect_metrics: bool = bool(_CPU_SHARED.get("collect_metrics", False))
    metrics_threshold: float = float(_CPU_SHARED.get("metrics_threshold", 0.0))
    n_components: int = int(_CPU_SHARED.get("n_components", 0))
    target_tot: float = float(_CPU_SHARED.get("target_tot", 0.0))

    alpha, beta = compute_alpha_beta_single(eps_combo, sig_combo, metadata)

    if A_alpha.shape[1] == 0:
        F_lj = np.zeros_like(F_fixed_kJ)
    else:
        F_lj = (
            np.einsum("epc,p->ec", A_alpha, alpha, optimize=True)
            + np.einsum("epc,p->ec", A_beta, beta, optimize=True)
        )

    F_pred_kJ = F_fixed_kJ + F_lj
    F_pred_kBT = F_pred_kJ / kbt
    delta = F_pred_kBT - F_target_kBT
    entry_errors = np.sum(delta * delta, axis=1)

    params_dict = _format_params_for_output(eps_combo, sig_combo)
    metrics: Optional[Dict[str, float]] = None
    if collect_metrics and n_components > 0:
        sse_total = float(np.sum(entry_errors))
        abs_sum = float(np.sum(np.abs(delta)))
        if target_tot > 1e-12:
            r2 = 1.0 - sse_total / target_tot
        else:
            r2 = 1.0 if sse_total <= 1e-12 else float("-inf")
        if r2 >= metrics_threshold:
            rmse = math.sqrt(sse_total / n_components)
            mae = abs_sum / n_components
            metrics = {"r2": r2, "rmse": rmse, "mae": mae}

    return entry_errors, params_dict, metrics


def _cpu_worker(chunk: Sequence[Tuple[Tuple[float, ...], Tuple[float, ...]]]):
    n_entries: int = int(_CPU_SHARED["n_entries"])
    group10_slices: Sequence[Tuple[int, int]] = _CPU_SHARED["group10_slices"]
    group100_slices: Sequence[Tuple[int, int]] = _CPU_SHARED["group100_slices"]

    local_entry_best = np.full(n_entries, np.inf, dtype=np.float64)
    local_entry_params: List[Optional[Dict[str, Dict[str, float]]]] = [None] * n_entries
    local_group10_best = np.full(len(group10_slices), np.inf, dtype=np.float64)
    local_group10_params: List[Optional[Dict[str, Dict[str, float]]]] = [None] * len(group10_slices)
    local_group100_best = np.full(len(group100_slices), np.inf, dtype=np.float64)
    local_group100_params: List[Optional[Dict[str, Dict[str, float]]]] = [None] * len(group100_slices)

    metrics_records: List[Dict[str, object]] = []

    for eps_combo, sig_combo in chunk:
        entry_errors, params_dict, metrics = _compute_entry_errors(eps_combo, sig_combo)

        for entry_idx, err in enumerate(entry_errors):
            if err < local_entry_best[entry_idx]:
                local_entry_best[entry_idx] = float(err)
                local_entry_params[entry_idx] = params_dict

        for g_idx, (start, end) in enumerate(group10_slices):
            group_err = float(np.sum(entry_errors[start:end]))
            if group_err < local_group10_best[g_idx]:
                local_group10_best[g_idx] = group_err
                local_group10_params[g_idx] = params_dict

        for g_idx, (start, end) in enumerate(group100_slices):
            group_err = float(np.sum(entry_errors[start:end]))
            if group_err < local_group100_best[g_idx]:
                local_group100_best[g_idx] = group_err
                local_group100_params[g_idx] = params_dict

        if metrics is not None:
            record: Dict[str, object] = {"params": params_dict}
            record.update(metrics)
            metrics_records.append(record)

    entry_updates = [
        (idx, float(err), local_entry_params[idx])
        for idx, err in enumerate(local_entry_best)
        if local_entry_params[idx] is not None
    ]
    group10_updates = [
        (idx, float(err), local_group10_params[idx])
        for idx, err in enumerate(local_group10_best)
        if local_group10_params[idx] is not None
    ]
    group100_updates = [
        (idx, float(err), local_group100_params[idx])
        for idx, err in enumerate(local_group100_best)
        if local_group100_params[idx] is not None
    ]

    return entry_updates, group10_updates, group100_updates, len(chunk), metrics_records

def _cpu_worker_metrics(
    chunk: Sequence[Tuple[Tuple[float, ...], Tuple[float, ...]]]
):
    """Worker for metrics-only mode: no per-entry / per-group best tracking."""
    metrics_records: List[Dict[str, object]] = []

    for eps_combo, sig_combo in chunk:
        # 这一行重用你已有的误差 + metrics 逻辑
        _, params_dict, metrics = _compute_entry_errors(eps_combo, sig_combo)
        if metrics is not None:
            record: Dict[str, object] = {"params": params_dict}
            record.update(metrics)
            metrics_records.append(record)

    return len(chunk), metrics_records


def grid_search_cpu_metrics_only(
    epsilon_lists: Sequence[Sequence[float]],
    sigma_lists: Sequence[Sequence[float]],
    pair_metadata: PairMetadata,
    A_alpha: np.ndarray,
    A_beta: np.ndarray,
    F_fixed_kJ: np.ndarray,
    F_target_kBT: np.ndarray,
    kbt: float,
    zero_only_elements: Sequence[str],
    n_entries: int,
    shard_start: int,
    shard_end: int,
    n_procs: Optional[int],
    chunk_size: int,
    metrics_threshold: float,
    n_components: int,
    target_tot: float,
):
    """只算全局 metrics 的 CPU 多进程搜索，不维护 per-entry/10/100 最优解。"""
    total_combinations = max(shard_end - shard_start, 0)

    # 注意：这里强制 collect_metrics=True
    shared_payload = {
        "pair_metadata": pair_metadata,
        "A_alpha": A_alpha,
        "A_beta": A_beta,
        "F_fixed_kJ": F_fixed_kJ,
        "F_target_kBT": F_target_kBT,
        "kbt": float(kbt),
        "n_entries": int(n_entries),
        "group10_slices": [],          # metrics-only 不用，但没关系
        "group100_slices": [],         # metrics-only 不用，但没关系
        "zero_only_elements": list(zero_only_elements),
        "collect_metrics": True,
        "metrics_threshold": float(metrics_threshold),
        "n_components": int(n_components),
        "target_tot": float(target_tot),
    }

    metrics_records: List[Dict[str, object]] = []

    if total_combinations <= 0:
        print("[CPU-MP metrics-only] No parameter combinations assigned to this shard.")
        return metrics_records

    param_iter = parameter_generator_shard(
        epsilon_lists,
        sigma_lists,
        shard_start,
        shard_end,
    )
    batched = _batched_iterator(param_iter, max(1, int(chunk_size)))

    processed = 0
    start_time = time.time()

    with mp.Pool(processes=n_procs, initializer=_init_cpu_worker, initargs=(shared_payload,)) as pool:
        for count, local_metrics in pool.imap_unordered(_cpu_worker_metrics, batched):
            if local_metrics:
                metrics_records.extend(local_metrics)

            processed += count
            elapsed = time.time() - start_time
            if processed <= 0:
                est_left = float("inf")
            else:
                rate = processed / max(elapsed, 1e-12)
                remaining = max(total_combinations - processed, 0)
                est_left = remaining / max(rate, 1e-12)
            percent = (processed / total_combinations) * 100 if total_combinations else 100.0
            print(
                f"\r[CPU-MP metrics-only] {processed}/{total_combinations} ({percent:7.3f}%) | "
                f"elapsed {elapsed:7.1f}s | est left {est_left:7.1f}s",
                end="",
                flush=True,
            )

    print()

    return metrics_records


def build_group_slices(n_entries: int, group_size: int) -> List[Tuple[int, int]]:
    slices: List[Tuple[int, int]] = []
    for start in range(0, n_entries, group_size):
        end = min(n_entries, start + group_size)
        slices.append((start, end))
    return slices


def _batched_iterator(
    iterable: Iterable[Tuple[Tuple[float, ...], Tuple[float, ...]]],
    batch_size: int,
) -> Iterable[List[Tuple[Tuple[float, ...], Tuple[float, ...]]]]:
    batch: List[Tuple[Tuple[float, ...], Tuple[float, ...]]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def grid_search_cpu_mp(
    epsilon_lists: Sequence[Sequence[float]],
    sigma_lists: Sequence[Sequence[float]],
    pair_metadata: PairMetadata,
    A_alpha: np.ndarray,
    A_beta: np.ndarray,
    F_fixed_kJ: np.ndarray,
    F_target_kBT: np.ndarray,
    kbt: float,
    zero_only_elements: Sequence[str],
    n_entries: int,
    group10_slices: Sequence[Tuple[int, int]],
    group100_slices: Sequence[Tuple[int, int]],
    shard_start: int,
    shard_end: int,
    n_procs: Optional[int],
    chunk_size: int,
    collect_metrics: bool,
    metrics_threshold: float,
    n_components: int,
    target_tot: float,
):
    total_combinations = max(shard_end - shard_start, 0)

    shared_payload = {
        "pair_metadata": pair_metadata,
        "A_alpha": A_alpha,
        "A_beta": A_beta,
        "F_fixed_kJ": F_fixed_kJ,
        "F_target_kBT": F_target_kBT,
        "kbt": float(kbt),
        "n_entries": int(n_entries),
        "group10_slices": group10_slices,
        "group100_slices": group100_slices,
        "zero_only_elements": list(zero_only_elements),
        "collect_metrics": bool(collect_metrics),
        "metrics_threshold": float(metrics_threshold),
        "n_components": int(n_components),
        "target_tot": float(target_tot),
    }

    best_entry_errors = np.full(n_entries, np.inf, dtype=np.float64)
    best_entry_params: List[Optional[Dict[str, Dict[str, float]]]] = [None] * n_entries
    best_group10_errors = np.full(len(group10_slices), np.inf, dtype=np.float64)
    best_group10_params: List[Optional[Dict[str, Dict[str, float]]]] = [None] * len(group10_slices)
    best_group100_errors = np.full(len(group100_slices), np.inf, dtype=np.float64)
    best_group100_params: List[Optional[Dict[str, Dict[str, float]]]] = [None] * len(group100_slices)

    if total_combinations <= 0:
        print("[CPU-MP] No parameter combinations assigned to this shard.")
        return (
            best_entry_errors,
            best_entry_params,
            best_group10_errors,
            best_group10_params,
            best_group100_errors,
            best_group100_params,
            [],
        )

    param_iter = parameter_generator_shard(
        epsilon_lists,
        sigma_lists,
        shard_start,
        shard_end,
    )
    batched = _batched_iterator(param_iter, max(1, int(chunk_size)))

    processed = 0
    start_time = time.time()

    metrics_records: List[Dict[str, object]] = []

    with mp.Pool(processes=n_procs, initializer=_init_cpu_worker, initargs=(shared_payload,)) as pool:
        for (
            entry_updates,
            group10_updates,
            group100_updates,
            count,
            local_metrics,
        ) in pool.imap_unordered(_cpu_worker, batched):
            for idx, err, params in entry_updates:
                if err < best_entry_errors[idx]:
                    best_entry_errors[idx] = err
                    best_entry_params[idx] = params
            for idx, err, params in group10_updates:
                if err < best_group10_errors[idx]:
                    best_group10_errors[idx] = err
                    best_group10_params[idx] = params
            for idx, err, params in group100_updates:
                if err < best_group100_errors[idx]:
                    best_group100_errors[idx] = err
                    best_group100_params[idx] = params

            if local_metrics:
                metrics_records.extend(local_metrics)

            processed += count
            elapsed = time.time() - start_time
            if processed <= 0:
                est_left = float("inf")
            else:
                rate = processed / max(elapsed, 1e-12)
                remaining = max(total_combinations - processed, 0)
                est_left = remaining / max(rate, 1e-12)
            percent = (processed / total_combinations) * 100 if total_combinations else 100.0
            print(
                f"\r[CPU-MP] {processed}/{total_combinations} ({percent:7.3f}%) | "
                f"elapsed {elapsed:7.1f}s | est left {est_left:7.1f}s",
                end="",
                flush=True,
            )

    print()

    return (
        best_entry_errors,
        best_entry_params,
        best_group10_errors,
        best_group10_params,
        best_group100_errors,
        best_group100_params,
        metrics_records,
    )


def grid_search_gpu(
    epsilon_lists: Sequence[Sequence[float]],
    sigma_lists: Sequence[Sequence[float]],
    pair_metadata: PairMetadata,
    A_alpha: np.ndarray,
    A_beta: np.ndarray,
    F_fixed_kJ: np.ndarray,
    F_target_kBT: np.ndarray,
    kbt: float,
    zero_only_elements: Sequence[str],
    n_entries: int,
    group10_slices: Sequence[Tuple[int, int]],
    group100_slices: Sequence[Tuple[int, int]],
    shard_start: int,
    shard_end: int,
    batch_size: int,
    collect_metrics: bool,
    metrics_threshold: float,
    n_components: int,
    target_tot: float,
):
    if torch is None:
        raise RuntimeError("PyTorch is required for the GPU backend but is not available")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[GPU] CUDA device not available, using CPU tensors for evaluation.")

    tensor_kwargs = {"dtype": torch.float64, "device": device}
    A_alpha_t = torch.tensor(A_alpha, **tensor_kwargs)
    A_beta_t = torch.tensor(A_beta, **tensor_kwargs)
    F_fixed_kJ_t = torch.tensor(F_fixed_kJ, **tensor_kwargs)
    F_target_kBT_t = torch.tensor(F_target_kBT, **tensor_kwargs)
    kbt_t = torch.tensor(float(kbt), **tensor_kwargs)
    target_tot_t = torch.tensor(float(target_tot), **tensor_kwargs)
    target_tot_value = float(target_tot)
    n_components_f = float(n_components)

    idx_i_t = torch.tensor(pair_metadata.idx_i_safe, dtype=torch.long, device=device)
    idx_j_t = torch.tensor(pair_metadata.idx_j_safe, dtype=torch.long, device=device)
    active_mask_t = torch.tensor(pair_metadata.active_mask, dtype=torch.bool, device=device)

    def build_group_matrix(slices: Sequence[Tuple[int, int]]) -> torch.Tensor:
        if not slices:
            return torch.zeros((0, n_entries), **tensor_kwargs)
        matrix = torch.zeros((len(slices), n_entries), **tensor_kwargs)
        for g_idx, (start, end) in enumerate(slices):
            if end > start:
                matrix[g_idx, start:end] = 1.0
        return matrix

    group10_matrix_t = build_group_matrix(group10_slices)
    group100_matrix_t = build_group_matrix(group100_slices)

    best_entry_errors = np.full(n_entries, np.inf, dtype=np.float64)
    best_entry_params: List[Optional[Dict[str, Dict[str, float]]]] = [None] * n_entries
    best_group10_errors = np.full(len(group10_slices), np.inf, dtype=np.float64)
    best_group10_params: List[Optional[Dict[str, Dict[str, float]]]] = [None] * len(group10_slices)
    best_group100_errors = np.full(len(group100_slices), np.inf, dtype=np.float64)
    best_group100_params: List[Optional[Dict[str, Dict[str, float]]]] = [None] * len(group100_slices)
    metrics_records: List[Dict[str, object]] = []

    total_combinations = max(shard_end - shard_start, 0)
    if total_combinations <= 0:
        print("[GPU] No parameter combinations assigned to this shard.")
        return (
            best_entry_errors,
            best_entry_params,
            best_group10_errors,
            best_group10_params,
            best_group100_errors,
            best_group100_params,
            metrics_records,
        )

    param_iter = parameter_generator_shard(
        epsilon_lists,
        sigma_lists,
        shard_start,
        shard_end,
    )

    processed = 0
    start_time = time.time()

    while True:
        batch = list(itertools.islice(param_iter, max(1, int(batch_size))))
        if not batch:
            break

        eps_batch_t = torch.tensor([b[0] for b in batch], **tensor_kwargs)
        sig_batch_t = torch.tensor([b[1] for b in batch], **tensor_kwargs)

        eps_i = eps_batch_t[:, idx_i_t]
        eps_j = eps_batch_t[:, idx_j_t]
        sig_i = sig_batch_t[:, idx_i_t]
        sig_j = sig_batch_t[:, idx_j_t]

        valid_mask = (
            active_mask_t.unsqueeze(0)
            & (eps_i > 0.0)
            & (eps_j > 0.0)
            & (sig_i > 0.0)
            & (sig_j > 0.0)
        )

        eps_mix = torch.where(valid_mask, torch.sqrt(eps_i * eps_j), torch.zeros_like(eps_i))
        sig_mix = torch.where(valid_mask, torch.sqrt(sig_i * sig_j), torch.zeros_like(sig_i))
        sigma6 = sig_mix.pow(6)
        beta_t = torch.where(valid_mask, 4.0 * eps_mix * sigma6, torch.zeros_like(sig_mix))
        alpha_t = beta_t * sigma6

        if A_alpha_t.shape[1] == 0:
            F_lj_t = torch.zeros((eps_batch_t.shape[0], n_entries, 3), **tensor_kwargs)
        else:
            F_alpha = torch.einsum("epc,bp->bec", A_alpha_t, alpha_t)
            F_beta = torch.einsum("epc,bp->bec", A_beta_t, beta_t)
            F_lj_t = F_alpha + F_beta

        F_pred_kJ_t = F_fixed_kJ_t.unsqueeze(0) + F_lj_t
        F_pred_kBT_t = F_pred_kJ_t / kbt_t
        delta_t = F_pred_kBT_t - F_target_kBT_t.unsqueeze(0)
        entry_errors_batch_t = torch.sum(delta_t * delta_t, dim=2)
        sse_batch_t = torch.sum(entry_errors_batch_t, dim=1)
        abs_sum_batch_t = torch.sum(torch.abs(delta_t), dim=(1, 2))

        params_cache: Dict[int, Dict[str, Dict[str, float]]] = {}

        batch_min_errs, batch_min_idx = torch.min(entry_errors_batch_t, dim=0)
        batch_min_errs_np = batch_min_errs.cpu().numpy()
        batch_min_idx_np = batch_min_idx.cpu().numpy()
        improved_entries = np.nonzero(batch_min_errs_np < best_entry_errors)[0]
        for entry_idx in improved_entries:
            best_entry_errors[entry_idx] = float(batch_min_errs_np[entry_idx])
            combo_idx = int(batch_min_idx_np[entry_idx])
            if combo_idx not in params_cache:
                params_cache[combo_idx] = make_param_dict(
                    batch[combo_idx][0],
                    batch[combo_idx][1],
                    zero_only_elements,
                )
            best_entry_params[entry_idx] = params_cache[combo_idx]

        if group10_matrix_t.shape[0] > 0:
            group10_errors_batch_t = entry_errors_batch_t @ group10_matrix_t.T
            g10_min_errs, g10_min_idx = torch.min(group10_errors_batch_t, dim=0)
            g10_min_errs_np = g10_min_errs.cpu().numpy()
            g10_min_idx_np = g10_min_idx.cpu().numpy()
            improved_g10 = np.nonzero(g10_min_errs_np < best_group10_errors)[0]
            for g_idx in improved_g10:
                best_group10_errors[g_idx] = float(g10_min_errs_np[g_idx])
                combo_idx = int(g10_min_idx_np[g_idx])
                if combo_idx not in params_cache:
                    params_cache[combo_idx] = make_param_dict(
                        batch[combo_idx][0],
                        batch[combo_idx][1],
                        zero_only_elements,
                    )
                best_group10_params[g_idx] = params_cache[combo_idx]

        if group100_matrix_t.shape[0] > 0:
            group100_errors_batch_t = entry_errors_batch_t @ group100_matrix_t.T
            g100_min_errs, g100_min_idx = torch.min(group100_errors_batch_t, dim=0)
            g100_min_errs_np = g100_min_errs.cpu().numpy()
            g100_min_idx_np = g100_min_idx.cpu().numpy()
            improved_g100 = np.nonzero(g100_min_errs_np < best_group100_errors)[0]
            for g_idx in improved_g100:
                best_group100_errors[g_idx] = float(g100_min_errs_np[g_idx])
                combo_idx = int(g100_min_idx_np[g_idx])
                if combo_idx not in params_cache:
                    params_cache[combo_idx] = make_param_dict(
                        batch[combo_idx][0],
                        batch[combo_idx][1],
                        zero_only_elements,
                    )
                best_group100_params[g_idx] = params_cache[combo_idx]

        if collect_metrics and n_components_f > 0:
            if target_tot_value > 1e-12:
                r2_batch_t = 1.0 - sse_batch_t / target_tot_t
            else:
                r2_batch_t = torch.where(
                    sse_batch_t <= 1e-12,
                    torch.ones_like(sse_batch_t),
                    torch.full_like(sse_batch_t, float("-inf")),
                )
            mae_batch_t = abs_sum_batch_t / n_components_f
            rmse_batch_t = torch.sqrt(torch.clamp(sse_batch_t / n_components_f, min=0.0))
            mask = r2_batch_t >= metrics_threshold
            if torch.any(mask):
                idxs = torch.nonzero(mask, as_tuple=False).squeeze(-1).tolist()
                r2_vals = r2_batch_t[mask].detach().cpu().numpy()
                mae_vals = mae_batch_t[mask].detach().cpu().numpy()
                rmse_vals = rmse_batch_t[mask].detach().cpu().numpy()
                for pos, combo_idx in enumerate(idxs):
                    if combo_idx not in params_cache:
                        params_cache[combo_idx] = make_param_dict(
                            batch[combo_idx][0],
                            batch[combo_idx][1],
                            zero_only_elements,
                        )
                    metrics_records.append(
                        {
                            "params": params_cache[combo_idx],
                            "r2": float(r2_vals[pos]),
                            "mae": float(mae_vals[pos]),
                            "rmse": float(rmse_vals[pos]),
                        }
                    )

        processed += len(batch)
        elapsed = time.time() - start_time
        if processed <= 0:
            est_left = float("inf")
        else:
            rate = processed / max(elapsed, 1e-12)
            remaining = max(total_combinations - processed, 0)
            est_left = remaining / max(rate, 1e-12)
        percent = (processed / total_combinations) * 100 if total_combinations else 100.0
        print(
            f"\r[GPU] {processed}/{total_combinations} ({percent:7.3f}%) | "
            f"elapsed {elapsed:7.1f}s | est left {est_left:7.1f}s",
            end="",
            flush=True,
        )

    print()

    return (
        best_entry_errors,
        best_entry_params,
        best_group10_errors,
        best_group10_params,
        best_group100_errors,
        best_group100_params,
        metrics_records,
    )

def grid_search_gpu_metrics_only(
    epsilon_lists: Sequence[Sequence[float]],
    sigma_lists: Sequence[Sequence[float]],
    pair_metadata: PairMetadata,
    A_alpha: np.ndarray,
    A_beta: np.ndarray,
    F_fixed_kJ: np.ndarray,
    F_target_kBT: np.ndarray,
    kbt: float,
    zero_only_elements: Sequence[str],
    n_entries: int,
    shard_start: int,
    shard_end: int,
    batch_size: int,
    metrics_threshold: float,
    n_components: int,
    target_tot: float,
):
    """只算全局 metrics 的 GPU 搜索，不维护 per-entry/10/100 最优解。"""
    if torch is None:
        raise RuntimeError("PyTorch is required for the GPU backend but is not available")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[GPU metrics-only] CUDA device not available, using CPU tensors for evaluation.")

    tensor_kwargs = {"dtype": torch.float64, "device": device}
    A_alpha_t = torch.tensor(A_alpha, **tensor_kwargs)
    A_beta_t = torch.tensor(A_beta, **tensor_kwargs)
    F_fixed_kJ_t = torch.tensor(F_fixed_kJ, **tensor_kwargs)
    F_target_kBT_t = torch.tensor(F_target_kBT, **tensor_kwargs)
    kbt_t = torch.tensor(float(kbt), **tensor_kwargs)
    target_tot_t = torch.tensor(float(target_tot), **tensor_kwargs)
    target_tot_value = float(target_tot)
    n_components_f = float(n_components)



    idx_i_t = torch.tensor(pair_metadata.idx_i_safe, dtype=torch.long, device=device)
    idx_j_t = torch.tensor(pair_metadata.idx_j_safe, dtype=torch.long, device=device)
    active_mask_t = torch.tensor(pair_metadata.active_mask, dtype=torch.bool, device=device)

    metrics_records: List[Dict[str, object]] = []

    # === 新增：记录当前最优 r2 与参数 ===
    best_r2 = float("-inf")
    best_params: Optional[Dict[str, Dict[str, float]]] = None
    # =================================

    total_combinations = max(shard_end - shard_start, 0)
    if total_combinations <= 0:
        print("[GPU metrics-only] No parameter combinations assigned to this shard.")
        return metrics_records

    param_iter = parameter_generator_shard(
        epsilon_lists,
        sigma_lists,
        shard_start,
        shard_end,
    )

    processed = 0
    start_time = time.time()

    while True:
        batch = list(itertools.islice(param_iter, max(1, int(batch_size))))
        if not batch:
            break

        eps_batch_t = torch.tensor([b[0] for b in batch], **tensor_kwargs)
        sig_batch_t = torch.tensor([b[1] for b in batch], **tensor_kwargs)

        # 计算混合 eps / sigma
        eps_i = eps_batch_t[:, idx_i_t]
        eps_j = eps_batch_t[:, idx_j_t]
        sig_i = sig_batch_t[:, idx_i_t]
        sig_j = sig_batch_t[:, idx_j_t]

        valid_mask = (
            active_mask_t.unsqueeze(0)
            & (eps_i > 0.0)
            & (eps_j > 0.0)
            & (sig_i > 0.0)
            & (sig_j > 0.0)
        )

        eps_mix = torch.where(valid_mask, torch.sqrt(eps_i * eps_j), torch.zeros_like(eps_i))
        sig_mix = torch.where(valid_mask, torch.sqrt(sig_i * sig_j), torch.zeros_like(sig_i))
        sigma6 = sig_mix.pow(6)
        beta_t = torch.where(valid_mask, 4.0 * eps_mix * sigma6, torch.zeros_like(sig_mix))
        alpha_t = beta_t * sigma6

        # 计算 LJ 力
        if A_alpha_t.shape[1] == 0:
            F_lj_t = torch.zeros((eps_batch_t.shape[0], n_entries, 3), **tensor_kwargs)
        else:
            F_alpha = torch.einsum("epc,bp->bec", A_alpha_t, alpha_t)
            F_beta = torch.einsum("epc,bp->bec", A_beta_t, beta_t)
            F_lj_t = F_alpha + F_beta

        F_pred_kJ_t = F_fixed_kJ_t.unsqueeze(0) + F_lj_t
        F_pred_kBT_t = F_pred_kJ_t / kbt_t
        delta_t = F_pred_kBT_t - F_target_kBT_t.unsqueeze(0)

        # 每个组合的 SSE，总误差
        entry_errors_batch_t = torch.sum(delta_t * delta_t, dim=2)  # (B, E)
        sse_batch_t = torch.sum(entry_errors_batch_t, dim=1)        # (B,)
        abs_sum_batch_t = torch.sum(torch.abs(delta_t), dim=(1, 2)) # (B,)

        # 只算全局 metrics
        if n_components_f > 0:
            if target_tot_value > 1e-12:
                r2_batch_t = 1.0 - sse_batch_t / target_tot_t
            else:
                r2_batch_t = torch.where(
                    sse_batch_t <= 1e-12,
                    torch.ones_like(sse_batch_t),
                    torch.full_like(sse_batch_t, float("-inf")),
                )
            mae_batch_t = abs_sum_batch_t / n_components_f
            rmse_batch_t = torch.sqrt(torch.clamp(sse_batch_t / n_components_f, min=0.0))

            # 1️⃣ 先更新全局 best_r2（不看阈值）
            max_r2_t, max_idx_t = torch.max(r2_batch_t, dim=0)   # r2_batch_t: (B,)
            max_r2 = float(max_r2_t.item())
            max_idx = int(max_idx_t.item())
            if max_r2 > best_r2:
                best_r2 = max_r2
                best_params = make_param_dict(
                    batch[max_idx][0],
                    batch[max_idx][1],
                    zero_only_elements,
                )
                print(
                    f"\n[GPU metrics-only] NEW BEST (overall) r2 = {best_r2:.6f}, params = {best_params}",
                    flush=True,
                )

            # 2️⃣ 再用阈值筛选哪些要写进 metrics_records
            mask = r2_batch_t >= metrics_threshold
            if torch.any(mask):
                idxs = torch.nonzero(mask, as_tuple=False).squeeze(-1).tolist()
                r2_vals = r2_batch_t[mask].detach().cpu().numpy()
                mae_vals = mae_batch_t[mask].detach().cpu().numpy()
                rmse_vals = rmse_batch_t[mask].detach().cpu().numpy()
                for pos, combo_idx in enumerate(idxs):
                    params_dict = make_param_dict(
                        batch[combo_idx][0],
                        batch[combo_idx][1],
                        zero_only_elements,
                    )
                    metrics_records.append(
                        {
                            "params": params_dict,
                            "r2": float(r2_vals[pos]),
                            "mae": float(mae_vals[pos]),
                            "rmse": float(rmse_vals[pos]),
                        }
                    )

        processed += len(batch)
        elapsed = time.time() - start_time
        if processed <= 0:
            est_left = float("inf")
        else:
            rate = processed / max(elapsed, 1e-12)
            remaining = max(total_combinations - processed, 0)
            est_left = remaining / max(rate, 1e-12)
        percent = (processed / total_combinations) * 100 if total_combinations else 100.0

        # === 修改：进度条后带上当前 best r2 ===
        if best_r2 > float("-inf"):
            best_str = f"{best_r2:7.4f}"
        else:
            best_str = "   N/A"
        print(
            f"\r[GPU metrics-only] {processed}/{total_combinations} ({percent:7.3f}%) | "
            f"elapsed {elapsed:7.1f}s | est left {est_left:7.1f}s | best r2 {best_str}",
            end="",
            flush=True,
        )
        # ======================================

    print()

    return metrics_records


def save_results(
    out_prefix: Path,
    entries: Sequence[ljfit.ProcessedEntry],
    group10_slices: Sequence[Tuple[int, int]],
    group100_slices: Sequence[Tuple[int, int]],
    best_entry_errors: Sequence[float],
    best_entry_params: Sequence[Optional[Dict[str, Dict[str, float]]]],
    best_group10_errors: Sequence[float],
    best_group10_params: Sequence[Optional[Dict[str, Dict[str, float]]]],
    best_group100_errors: Sequence[float],
    best_group100_params: Sequence[Optional[Dict[str, Dict[str, float]]]],
) -> None:
    out_prefix = Path(out_prefix)
    if out_prefix.parent:
        out_prefix.parent.mkdir(parents=True, exist_ok=True)

    per_entry_records: List[MutableMapping[str, object]] = []
    for idx, entry in enumerate(entries):
        record: MutableMapping[str, object] = {
            "entry_index": idx,
            "gro_file": entry.gro_file,
            "center_index": int(entry.center_index),
            "best_error": float(best_entry_errors[idx]),
            "params": best_entry_params[idx],
        }
        per_entry_records.append(record)

    per10_records: List[MutableMapping[str, object]] = []
    for g_idx, (start, end) in enumerate(group10_slices):
        record = {
            "group_index": g_idx,
            "entry_start": start,
            "entry_end": end - 1 if end > start else start - 1,
            "best_group_error": float(best_group10_errors[g_idx]),
            "params": best_group10_params[g_idx],
        }
        per10_records.append(record)

    per100_records: List[MutableMapping[str, object]] = []
    for g_idx, (start, end) in enumerate(group100_slices):
        record = {
            "group_index": g_idx,
            "entry_start": start,
            "entry_end": end - 1 if end > start else start - 1,
            "best_group_error": float(best_group100_errors[g_idx]),
            "params": best_group100_params[g_idx],
        }
        per100_records.append(record)

    with (out_prefix.parent / f"{out_prefix.name}.per_entry.json").open("w", encoding="utf-8") as fh:
        json.dump(per_entry_records, fh, indent=2)

    with (out_prefix.parent / f"{out_prefix.name}.per10.json").open("w", encoding="utf-8") as fh:
        json.dump(per10_records, fh, indent=2)

    with (out_prefix.parent / f"{out_prefix.name}.per100.json").open("w", encoding="utf-8") as fh:
        json.dump(per100_records, fh, indent=2)


def save_metrics(
    out_prefix: Path,
    metrics_out: Optional[Path],
    metrics_records: Sequence[Mapping[str, object]],
) -> None:
    if metrics_out is None:
        out_prefix = Path(out_prefix)
        base_dir = out_prefix.parent if out_prefix.parent else Path(".")
        metrics_path = base_dir / f"{out_prefix.name}.metrics.json"
    else:
        metrics_path = Path(metrics_out)

    if metrics_path.parent:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_records = sorted(
        (dict(record) for record in metrics_records),
        key=lambda rec: rec.get("r2", float("-inf")),
        reverse=True,
    )

    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(sorted_records, fh, indent=2)


def build_entries_and_pairs(
    raw_data: Sequence[MutableMapping[str, object]],
    summary: Mapping[str, object],
    element_map: Mapping[str, str],
    kbt: float,
    bond_csv: Optional[Path],
    angle_csv: Optional[Path],
    bond_angle_in_kbt: bool,
    ke: float,
    fudge_lj: float,
    fudge_qq: float,
    max_pairs_per_entry: Optional[int],
) -> Tuple[List[ljfit.ProcessedEntry], List[Tuple[str, str]], List[List[Mapping[str, object]]]]:
    entries: List[ljfit.ProcessedEntry] = []
    for idx, entry in enumerate(raw_data):
        processed = ljfit.validate_and_fix_entry(entry, idx, element_map)
        entries.append(processed)

    if isinstance(summary, MutableMapping):
        ljfit.patch_missing_angle_params(summary)

    dihedral_map = ljfit.compute_dihedral_forces_only(entries, summary, verbose=True)
    bond_map, angle_map = load_bond_angle_maps(bond_csv, angle_csv, kbt, bond_angle_in_kbt)
    apply_external_forces(entries, dihedral_map, bond_map, angle_map)

    pair_type_list, entry_pairs = ljfit.build_pair_types(
        entries,
        ke=ke,
        fudge_lj=fudge_lj,
        fudge_qq=fudge_qq,
        max_pairs_per_entry=max_pairs_per_entry,
    )

    return entries, pair_type_list, entry_pairs


def main() -> None:
    args = parse_args()

    kbt = BOLTZMANN_KJ_PER_MOL_K * float(args.temperature)

    raw_data, summary = ljfit.load_data(Path(args.data), Path(args.summary))
    element_map = parse_element_map(args.element_map)

    entries, pair_type_list, entry_pairs = build_entries_and_pairs(
        raw_data,
        summary,
        element_map,
        kbt,
        args.bond_csv,
        args.angle_csv,
        args.bond_angle_in_kbt,
        args.ke,
        args.fudge_lj,
        args.fudge_qq,
        args.max_pairs_per_entry,
    )

    A_alpha, A_beta, F_fixed_kJ, F_target_kBT = precompute_aggregates(entries, entry_pairs, pair_type_list)
    pair_metadata = prepare_pair_metadata(pair_type_list)

    _, _, epsilon_lists, sigma_lists = build_parameter_space(
        args.epsilon_grid,
        args.sigma_grid,
    )
    eps_combinations = math.prod(len(values) for values in epsilon_lists)
    sigma_combinations = math.prod(len(values) for values in sigma_lists)
    total_combinations = eps_combinations * sigma_combinations

    num_shards = int(args.num_shards)
    if num_shards <= 0:
        raise ValueError("num-shards must be a positive integer")
    shard_index = int(args.shard_index)
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    shard_start = (total_combinations * shard_index) // num_shards
    shard_end = (total_combinations * (shard_index + 1)) // num_shards
    combinations_in_shard = shard_end - shard_start
    print(
        f"[INFO] shard {shard_index + 1}/{num_shards}: "
        f"combinations {combinations_in_shard} (indices {shard_start}-{shard_end - 1}) "
        f"out of {total_combinations}",
        flush=True,
    )

    n_entries = len(entries)
    group10_slices = build_group_slices(n_entries, 10)
    group100_slices = build_group_slices(n_entries, 100)

    all_elements = gather_all_elements(pair_type_list)
    zero_only_elements = [elem for elem in all_elements if elem not in SEARCH_ELEMENTS and elem != FIXED_ZERO_ELEMENT]

    target_flat = F_target_kBT.reshape(-1)
    n_components = int(target_flat.size)
    target_mean = float(np.mean(target_flat)) if n_components > 0 else 0.0
    target_tot = float(np.sum((target_flat - target_mean) ** 2)) if n_components > 0 else 0.0
    collect_metrics = bool(args.global_metrics)
    metrics_threshold = float(args.metrics_threshold)
    
    print("[DEBUG] n_entries =", len(entries))
    print("[DEBUG] n_components =", n_components)
    print("[DEBUG] target_tot  =", target_tot)
    print("[DEBUG] F_target_kBT min/max/mean =",
        float(target_flat.min()) if n_components > 0 else None,
        float(target_flat.max()) if n_components > 0 else None,
        float(target_flat.mean()) if n_components > 0 else None)

    if args.metrics_only:
        collect_metrics = True

    if args.metrics_only:
        # ---- 只算全局 metrics 的路径 ----
        if args.backend == "cpu-mp":
            metrics_records = grid_search_cpu_metrics_only(
                epsilon_lists,
                sigma_lists,
                pair_metadata,
                A_alpha,
                A_beta,
                F_fixed_kJ,
                F_target_kBT,
                kbt,
                zero_only_elements,
                n_entries,
                shard_start,
                shard_end,
                args.n_procs,
                args.chunk_size,
                metrics_threshold,
                n_components,
                target_tot,
            )
        else:
            metrics_records = grid_search_gpu_metrics_only(
                epsilon_lists,
                sigma_lists,
                pair_metadata,
                A_alpha,
                A_beta,
                F_fixed_kJ,
                F_target_kBT,
                kbt,
                zero_only_elements,
                n_entries,
                shard_start,
                shard_end,
                args.gpu_batch_size,
                metrics_threshold,
                n_components,
                target_tot,
            )

        # 只写 metrics，不写 per-entry / per10 / per100
        if collect_metrics:
            metrics_out = Path(args.metrics_out) if args.metrics_out is not None else None
            save_metrics(args.out_prefix, metrics_out, metrics_records)

    else:
        # ---- 原来的完整路径（保留原逻辑） ----
        if args.backend == "cpu-mp":
            results = grid_search_cpu_mp(
                epsilon_lists,
                sigma_lists,
                pair_metadata,
                A_alpha,
                A_beta,
                F_fixed_kJ,
                F_target_kBT,
                kbt,
                zero_only_elements,
                n_entries,
                group10_slices,
                group100_slices,
                shard_start,
                shard_end,
                args.n_procs,
                args.chunk_size,
                collect_metrics,
                metrics_threshold,
                n_components,
                target_tot,
            )
        else:
            results = grid_search_gpu(
                epsilon_lists,
                sigma_lists,
                pair_metadata,
                A_alpha,
                A_beta,
                F_fixed_kJ,
                F_target_kBT,
                kbt,
                zero_only_elements,
                n_entries,
                group10_slices,
                group100_slices,
                shard_start,
                shard_end,
                args.gpu_batch_size,
                collect_metrics,
                metrics_threshold,
                n_components,
                target_tot,
            )

        (
            best_entry_errors,
            best_entry_params,
            best_group10_errors,
            best_group10_params,
            best_group100_errors,
            best_group100_params,
            metrics_records,
        ) = results

        save_results(
            args.out_prefix,
            entries,
            group10_slices,
            group100_slices,
            best_entry_errors,
            best_entry_params,
            best_group10_errors,
            best_group10_params,
            best_group100_errors,
            best_group100_params,
        )

        if collect_metrics:
            metrics_out = Path(args.metrics_out) if args.metrics_out is not None else None
            save_metrics(args.out_prefix, metrics_out, metrics_records)

if __name__ == "__main__":
    main()
