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

import numpy as np

try:  # Optional dependency for GPU backend
    import torch
except Exception:  # pragma: no cover - torch is optional unless GPU backend is requested
    torch = None

import min as ljfit


BOLTZMANN_KJ_PER_MOL_K = 0.00831446261815324
SEARCH_ELEMENTS = ("C", "H", "O", "O_h")
FIXED_ZERO_ELEMENT = "H_h"


@dataclass(frozen=True)
class PairMetadata:
    idx_i: np.ndarray
    idx_j: np.ndarray
    idx_i_safe: np.ndarray
    idx_j_safe: np.ndarray
    active_mask: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search for Lennard-Jones parameters")
    parser.add_argument("--data", required=True, type=Path, help="Path to dataset JSON")
    parser.add_argument("--summary", required=True, type=Path, help="Path to topology summary JSON")
    parser.add_argument("--element-map", type=str, default="", help="Custom atom type -> element mapping")
    parser.add_argument("--bond-csv", type=Path, default=None, help="CSV file with bond forces (kJ/mol/nm by default)")
    parser.add_argument("--angle-csv", type=Path, default=None, help="CSV file with angle forces (kJ/mol/nm by default)")
    parser.add_argument("--bond-angle-in-kbt", action="store_true", help="Bond/angle CSV forces are provided in kBT/nm")
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


def build_parameter_space() -> Tuple[List[float], List[float]]:
    epsilon_values = [round(0.2 * i, 10) for i in range(11)]  # 0.0 to 2.0 inclusive
    sigma_values = [round(0.1 + 0.05 * i, 10) for i in range(9)]  # 0.1 to 0.5 inclusive
    return epsilon_values, sigma_values


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


def combination_from_index(index: int, values: Sequence[float], length: int) -> Tuple[float, ...]:
    base = len(values)
    result = [0.0] * length
    for pos in range(length - 1, -1, -1):
        result[pos] = float(values[index % base])
        index //= base
    return tuple(result)


def parameter_generator_shard(
    epsilon_values: Sequence[float],
    sigma_values: Sequence[float],
    shard_start: int,
    shard_end: int,
) -> Iterator[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
    n_elem = len(SEARCH_ELEMENTS)
    sigma_combinations = len(sigma_values) ** n_elem
    for linear_idx in range(shard_start, shard_end):
        eps_idx = linear_idx // sigma_combinations
        sig_idx = linear_idx % sigma_combinations
        eps_combo = combination_from_index(eps_idx, epsilon_values, n_elem)
        sig_combo = combination_from_index(sig_idx, sigma_values, n_elem)
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
) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    metadata: PairMetadata = _CPU_SHARED["pair_metadata"]
    A_alpha: np.ndarray = _CPU_SHARED["A_alpha"]
    A_beta: np.ndarray = _CPU_SHARED["A_beta"]
    F_fixed_kJ: np.ndarray = _CPU_SHARED["F_fixed_kJ"]
    F_target_kBT: np.ndarray = _CPU_SHARED["F_target_kBT"]
    kbt: float = _CPU_SHARED["kbt"]

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
    return entry_errors, params_dict


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

    for eps_combo, sig_combo in chunk:
        entry_errors, params_dict = _compute_entry_errors(eps_combo, sig_combo)

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

    return entry_updates, group10_updates, group100_updates, len(chunk)


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
    epsilon_values: Sequence[float],
    sigma_values: Sequence[float],
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
        )

    param_iter = parameter_generator_shard(
        epsilon_values,
        sigma_values,
        shard_start,
        shard_end,
    )
    batched = _batched_iterator(param_iter, max(1, int(chunk_size)))

    processed = 0
    start_time = time.time()

    with mp.Pool(processes=n_procs, initializer=_init_cpu_worker, initargs=(shared_payload,)) as pool:
        for entry_updates, group10_updates, group100_updates, count in pool.imap_unordered(_cpu_worker, batched):
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
    )


def grid_search_gpu(
    epsilon_values: Sequence[float],
    sigma_values: Sequence[float],
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
        )

    param_iter = parameter_generator_shard(
        epsilon_values,
        sigma_values,
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
    )


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

    epsilon_values, sigma_values = build_parameter_space()
    n_elements = len(SEARCH_ELEMENTS)
    eps_combinations = len(epsilon_values) ** n_elements
    sigma_combinations = len(sigma_values) ** n_elements
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

    if args.backend == "cpu-mp":
        results = grid_search_cpu_mp(
            epsilon_values,
            sigma_values,
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
        )
    else:
        results = grid_search_gpu(
            epsilon_values,
            sigma_values,
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
        )

    (
        best_entry_errors,
        best_entry_params,
        best_group10_errors,
        best_group10_params,
        best_group100_errors,
        best_group100_params,
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


if __name__ == "__main__":
    main()








