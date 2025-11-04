#!/usr/bin/env python3
"""Fit Lennard-Jones parameters and charges against force-labelled JSON data.

This trainer optimises Lennard-Jones (LJ) ``sigma``/``epsilon`` values together
with partial charges so that the analytic forces of a *centre atom* match the
reference forces found in a processed JSON dataset.  The dataset is expected to
contain per-frame coordinates, atom types, formal charges, an adjacency matrix
and the reference force on exactly one atom.

Key features
============
* **Hydroxyl-aware categories** – oxygen and hydrogen atoms are split into
  hydroxyl and non-hydroxyl variants.  Together with carbon this yields five
  canonical categories (``c``, ``o_hydroxyl``, ``o_nonhydroxyl``,
  ``h_hydroxyl``, ``h_nonhydroxyl``) that each receive independent LJ and charge
  parameters.
* **Charge optimisation** – charges can be optimised alongside LJ parameters.
  Initial charge guesses are taken from the dataset itself (averaged per
  category) to honour the behaviour of :mod:`compare_forces`.
* **Grouped fitting modes** – entries can be optimised individually, in
  user-defined groups ("every *n* entries together"), or globally across the
  entire dataset.
* **Topology-aware non-bonded forces** – 1-2 and 1-3 pairs are excluded from
  non-bonded calculations while 1-4 interactions are scaled by the fudge
  factors found in the summary file.
* **Two-stage annealed optimisation** – Lennard-Jones parameters are refined
  with frozen charges before a combined optimisation with charge updates.  Each
  stage supports gradient normalisation, clipping, annealing, and configurable
  multi-start restarts to mitigate local minima.
* **Simple multi-process parallelism** – independent dataset groups can be
  optimised concurrently using multiple worker processes to reduce overall wall
  clock time.

The initial LJ parameters are obtained from a *parameter summary* JSON file (as
produced by :func:`compare_forces.build_parameter_summary`).  Charges are drawn
from the dataset unless no atoms for a category are present, in which case the
summary charge is used.

Results are written to a JSON report containing per-group diagnostics,
including parameter summaries and per-entry residuals.
"""

from __future__ import annotations

import argparse
import json
import math
import numpy as np
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from compare_forces import KELEC, load_parameter_summary

# Boltzmann constant in kJ mol^-1 K^-1
KB_KJ_MOL_K = 0.00831446261815324

Vector = List[float]


@dataclass(frozen=True)
class ElementEntry:
    """Descriptor for the LJ parameters of one base element."""

    sigma: float
    epsilon: float
    element: str
    charge_variants: Sequence[float]
    charge: float


@dataclass(frozen=True)
class EntryMetadata:
    index: int
    gro_file: Optional[str]
    center_atom: Mapping[str, object]


@dataclass
class EntryData:
    coords: List[Vector]
    categories: List[str]
    center_index: int
    target_force: Vector
    metadata: EntryMetadata
    neighbors: List[List[int]]
    base_elements: List[str]
    topological_distances: List[int]


@dataclass
class CategoryInfo:
    key: str
    base_element: str
    sigma: float
    epsilon: float
    charge: float
    train_sigma: bool
    train_epsilon: bool
    train_charge: bool


@dataclass
class ParameterSpace:
    categories: Dict[str, CategoryInfo]
    train_categories: Tuple[str, ...]
    index_map: Dict[str, int]
    sigma_bounds: Tuple[float, float]
    epsilon_bounds: Tuple[float, float]
    charge_bounds: Tuple[float, float]

    def initial_theta(self) -> List[float]:
        theta: List[float] = []
        for category in self.train_categories:
            info = self.categories[category]
            if info.sigma <= 0.0 or info.epsilon <= 0.0:
                raise ValueError(
                    f"Cannot optimise LJ parameters for category '{category}' with non-positive values"
                )
            sigma = min(self.sigma_bounds[1], max(self.sigma_bounds[0], info.sigma))
            epsilon = min(self.epsilon_bounds[1], max(self.epsilon_bounds[0], info.epsilon))
            charge = info.charge
            if info.train_charge:
                charge = min(self.charge_bounds[1], max(self.charge_bounds[0], charge))
            theta.extend([math.log(sigma), math.log(epsilon), charge])
        return theta


@dataclass
class GroupContext:
    entries: List[EntryData]
    parameter_space: ParameterSpace
    comb_rule: int
    kbt: float
    category_counts: Mapping[str, int]
    temperature: float
    bond_lookup: Mapping[str, Sequence[Mapping[str, object]]]
    angle_lookup: Mapping[str, Sequence[Mapping[str, object]]]
    dihedral_lookup: Mapping[Tuple[str, str], Sequence[Mapping[str, object]]]
    fudge_lj: float
    fudge_qq: float


@dataclass
class ElementState:
    sigma: float
    epsilon: float
    charge: float
    c6: float
    c12: float
    dc6_dsigma: float
    dc6_depsilon: float
    dc12_dsigma: float
    dc12_depsilon: float
    train_sigma: bool
    train_epsilon: bool
    train_charge: bool


_CATEGORY_ORDER = [
    "c",
    "o_hydroxyl",
    "o_nonhydroxyl",
    "h_hydroxyl",
    "h_nonhydroxyl",
]

_CATEGORY_ALIASES = {
    "oh": "o_hydroxyl",
    "o_h": "o_hydroxyl",
    "ho": "h_hydroxyl",
    "h_h": "h_hydroxyl",
    "o_non": "o_nonhydroxyl",
    "h_non": "h_nonhydroxyl",
}


def _element_from_atom_type(atom_type: str) -> str:
    if not atom_type:
        return "x"
    return atom_type[0].lower()


def _dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _category_sort_key(category: str) -> Tuple[int, str]:
    try:
        return (_CATEGORY_ORDER.index(category), category)
    except ValueError:
        return (len(_CATEGORY_ORDER), category)


def _base_element_from_category(category: str) -> str:
    if category.startswith("o_"):
        return "o"
    if category.startswith("h_"):
        return "h"
    return category[:1]


def _resolve_train_targets(values: Sequence[str]) -> List[str]:
    categories: List[str] = []
    for raw in values:
        key = str(raw).strip().lower()
        if not key:
            continue
        key = _CATEGORY_ALIASES.get(key, key)
        if "_" in key or key in _CATEGORY_ORDER:
            categories.append(key)
            continue
        if key in {"o", "oxygen"}:
            categories.extend(["o_hydroxyl", "o_nonhydroxyl"])
        elif key in {"h", "hydrogen"}:
            categories.extend(["h_hydroxyl", "h_nonhydroxyl"])
        else:
            categories.append(key)
    seen = set()
    ordered: List[str] = []
    for cat in categories:
        if cat not in seen:
            seen.add(cat)
            ordered.append(cat)
    return ordered


def _build_neighbors(adj_matrix: Sequence[Sequence[float]]) -> List[List[int]]:
    neighbors: List[List[int]] = []
    for row in adj_matrix:
        row_neighbors = [j for j, value in enumerate(row) if float(value) != 0.0]
        neighbors.append(row_neighbors)
    return neighbors


def _compute_topological_distances(center: int, neighbors: Sequence[Sequence[int]]) -> List[int]:
    """Return the graph distance from ``center`` to every atom."""

    n = len(neighbors)
    distances = [-1 for _ in range(n)]
    queue: deque[int] = deque([center])
    distances[center] = 0
    while queue:
        current = queue.popleft()
        next_distance = distances[current] + 1
        for nb in neighbors[current]:
            if nb < 0 or nb >= n:
                continue
            if distances[nb] != -1:
                continue
            distances[nb] = next_distance
            queue.append(nb)
    return distances


def _vec_add(a: Vector, b: Vector) -> Vector:
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def _vec_sub(a: Vector, b: Vector) -> Vector:
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _vec_scale(v: Vector, scalar: float) -> Vector:
    return [v[0] * scalar, v[1] * scalar, v[2] * scalar]


def _vec_cross(a: Vector, b: Vector) -> Vector:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _vec_norm(v: Vector) -> float:
    return math.sqrt(_dot(v, v))


def _classify_atom_categories(atom_types: Sequence[str], adj_matrix: Sequence[Sequence[float]]) -> List[str]:
    base_elements = [_element_from_atom_type(t) for t in atom_types]
    neighbors = _build_neighbors(adj_matrix)
    categories: List[str] = []
    for idx, elem in enumerate(base_elements):
        if elem == "o":
            cat = "o_hydroxyl" if any(base_elements[nb] == "h" for nb in neighbors[idx]) else "o_nonhydroxyl"
        elif elem == "h":
            cat = "h_hydroxyl" if any(base_elements[nb] == "o" for nb in neighbors[idx]) else "h_nonhydroxyl"
        else:
            cat = elem
        categories.append(cat)
    return categories


def _angle_symmetry_key(pattern: str) -> str:
    if len(pattern) != 3:
        return pattern
    center = pattern[1]
    ends = "".join(sorted((pattern[0], pattern[2])))
    return f"{center}|{ends}"


def _canonicalize_dihedral_pattern(pattern: str) -> Tuple[str, bool]:
    reverse = pattern[::-1]
    if reverse < pattern:
        return reverse, False
    return pattern, True


def _dihedral_variant_label(has_start: bool, has_end: bool) -> str:
    if has_start and has_end:
        return "both"
    if has_start:
        return "start"
    if has_end:
        return "end"
    return "none"


def _is_hydroxyl_oxygen_category(category: str) -> bool:
    return category.startswith("o_") and "hydroxyl" in category


def _select_summary_entry(
    element: str,
    element_charges: Sequence[float],
    candidates: Sequence[Mapping[str, object]],
) -> ElementEntry:
    if not candidates:
        raise KeyError(f"No LJ parameters available for element '{element}'")

    if element_charges:
        best_entry: Optional[Mapping[str, object]] = None
        best_score = float("inf")
        for entry in candidates:
            variants = [float(v) for v in entry.get("charge_variants", [])]
            ref_charge = float(entry.get("charge", 0.0))
            if not variants:
                variants = [ref_charge]
            total = 0.0
            for q in element_charges:
                diff = min(abs(q - v) for v in variants)
                total += diff
            score = total / len(element_charges)
            if score < best_score:
                best_score = score
                best_entry = entry
        chosen = best_entry or candidates[0]
    else:
        chosen = candidates[0]

    sigma = float(chosen.get("sigma", 0.0))
    epsilon = float(chosen.get("epsilon", 0.0))
    charge_variants = [float(v) for v in chosen.get("charge_variants", [])]
    charge = float(chosen.get("charge", 0.0))
    return ElementEntry(
        sigma=sigma,
        epsilon=epsilon,
        element=element,
        charge_variants=tuple(charge_variants),
        charge=charge,
    )


def _build_bond_lookup(entries: Sequence[Mapping[str, object]]) -> Dict[str, List[Mapping[str, object]]]:
    lookup: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for entry in entries:
        pattern = str(entry.get("pattern", "")).strip().lower()
        if not pattern:
            continue
        key = "".join(sorted(pattern))
        lookup[key].append(entry)
    return lookup


def _build_angle_lookup(entries: Sequence[Mapping[str, object]]) -> Dict[str, List[Mapping[str, object]]]:
    lookup: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for entry in entries:
        sym_key = str(entry.get("symmetry_key", "")).strip().lower()
        if not sym_key:
            continue
        lookup[sym_key].append(entry)
    return lookup


def _build_dihedral_lookup(
    entries: Sequence[Mapping[str, object]]
) -> Dict[Tuple[str, str], List[Mapping[str, object]]]:
    lookup: Dict[Tuple[str, str], List[Mapping[str, object]]] = defaultdict(list)
    for entry in entries:
        pattern = str(entry.get("pattern", "")).strip().lower()
        if not pattern:
            continue
        variant = str(entry.get("hydroxyl_variant", "none")).strip().lower() or "none"
        lookup[(pattern, variant)].append(entry)
    return lookup


def _select_bond_entry(
    lookup: Mapping[str, Sequence[Mapping[str, object]]],
    elem_i: str,
    elem_j: str,
) -> Mapping[str, object]:
    key = "".join(sorted((elem_i, elem_j)))
    options = lookup.get(key)
    if not options:
        raise KeyError(f"No bond parameters for pattern '{key}'")
    return options[0]


def _select_angle_entry(
    lookup: Mapping[str, Sequence[Mapping[str, object]]],
    pattern: str,
) -> Mapping[str, object]:
    sym_key = _angle_symmetry_key(pattern)
    options = lookup.get(sym_key)
    if not options:
        raise KeyError(f"Missing angle parameters for symmetry '{sym_key}' (pattern {pattern})")
    if len(options) == 1:
        return options[0]
    pattern_rev = pattern[::-1]
    for entry in options:
        equivalents = [str(p).strip().lower() for p in entry.get("equivalent_patterns", [])]
        if pattern in equivalents or pattern_rev in equivalents:
            return entry
    return options[0]


def _select_dihedral_entry(
    lookup: Mapping[Tuple[str, str], Sequence[Mapping[str, object]]],
    canon_pattern: str,
    variant: str,
    pattern: str,
) -> Mapping[str, object]:
    variant_norm = variant or "none"
    options = lookup.get((canon_pattern, variant_norm))
    if not options:
        options = lookup.get((canon_pattern, "mixed")) or lookup.get((canon_pattern, "none"))
    if not options:
        raise KeyError(f"Missing dihedral parameters for pattern '{canon_pattern}' (variant {variant_norm})")
    if len(options) == 1:
        return options[0]
    pattern_rev = pattern[::-1]
    for entry in options:
        equivalents = [str(p).strip().lower() for p in entry.get("equivalent_patterns", [])]
        if pattern in equivalents or pattern_rev in equivalents:
            return entry
    return options[0]


def _prepare_element_state(
    sigma: float,
    epsilon: float,
    charge: float,
    train_sigma: bool,
    train_epsilon: bool,
    train_charge: bool,
) -> ElementState:
    sigma6 = sigma ** 6
    sigma12 = sigma6 ** 2
    c6 = 4.0 * epsilon * sigma6
    c12 = 4.0 * epsilon * sigma12
    if sigma > 0.0:
        sigma5 = sigma ** 5
        sigma11 = sigma ** 11
        dc6_dsigma = 24.0 * epsilon * sigma5
        dc12_dsigma = 48.0 * epsilon * sigma11
    else:
        dc6_dsigma = 0.0
        dc12_dsigma = 0.0
    dc6_depsilon = 4.0 * sigma6
    dc12_depsilon = 4.0 * sigma12
    return ElementState(
        sigma=sigma,
        epsilon=epsilon,
        charge=charge,
        c6=c6,
        c12=c12,
        dc6_dsigma=dc6_dsigma,
        dc6_depsilon=dc6_depsilon,
        dc12_dsigma=dc12_dsigma,
        dc12_depsilon=dc12_depsilon,
        train_sigma=train_sigma,
        train_epsilon=train_epsilon,
        train_charge=train_charge,
    )


def _combine_lj(
    elem_i: str,
    state_i: ElementState,
    elem_j: str,
    state_j: ElementState,
    comb_rule: int,
) -> Tuple[float, float, Dict[str, Dict[str, float]]]:
    """Return pair (C6, C12) and partial derivatives for the two categories."""

    partials: Dict[str, Dict[str, float]] = {}

    def _payload(state: ElementState, coeff6: float, coeff12: float) -> Dict[str, float]:
        payload: Dict[str, float] = {}
        if state.train_sigma:
            payload["dC6_dsigma"] = coeff6 * state.dc6_dsigma
            payload["dC12_dsigma"] = coeff12 * state.dc12_dsigma
        if state.train_epsilon:
            payload["dC6_depsilon"] = coeff6 * state.dc6_depsilon
            payload["dC12_depsilon"] = coeff12 * state.dc12_depsilon
        return payload

    if comb_rule in (1, 3):
        c6_i = state_i.c6
        c6_j = state_j.c6
        c12_i = state_i.c12
        c12_j = state_j.c12

        c6_pair = math.sqrt(c6_i * c6_j) if c6_i > 0.0 and c6_j > 0.0 else 0.0
        c12_pair = math.sqrt(c12_i * c12_j) if c12_i > 0.0 and c12_j > 0.0 else 0.0

        if c6_pair > 0.0:
            if c6_i > 0.0 and (state_i.train_sigma or state_i.train_epsilon):
                factor6 = 0.5 * c6_pair / c6_i
                factor12 = 0.5 * c12_pair / c12_i if c12_i > 0.0 else 0.0
                payload_i = _payload(state_i, factor6, factor12)
                if payload_i:
                    partials[elem_i] = payload_i
            if c6_j > 0.0 and (state_j.train_sigma or state_j.train_epsilon):
                factor6 = 0.5 * c6_pair / c6_j
                factor12 = 0.5 * c12_pair / c12_j if c12_j > 0.0 else 0.0
                payload_j = _payload(state_j, factor6, factor12)
                if payload_j:
                    partials.setdefault(elem_j, {})
                    for key, value in payload_j.items():
                        partials[elem_j][key] = partials[elem_j].get(key, 0.0) + value
        return c6_pair, c12_pair, partials

    if comb_rule == 2:
        sigma_pair = 0.5 * (state_i.sigma + state_j.sigma)
        epsilon_pair = math.sqrt(state_i.epsilon * state_j.epsilon)
        sigma_pair6 = sigma_pair ** 6
        sigma_pair12 = sigma_pair6 ** 2
        c6_pair = 4.0 * epsilon_pair * sigma_pair6
        c12_pair = 4.0 * epsilon_pair * sigma_pair12

        if state_i.train_sigma or state_i.train_epsilon:
            payload_i: Dict[str, float] = {}
            if state_i.train_sigma:
                payload_i["dC6_dsigma"] = 12.0 * epsilon_pair * sigma_pair ** 5
                payload_i["dC12_dsigma"] = 24.0 * epsilon_pair * sigma_pair ** 11
            if state_i.train_epsilon and state_i.epsilon > 0.0:
                payload_i["dC6_depsilon"] = 2.0 * epsilon_pair * sigma_pair6 / state_i.epsilon
                payload_i["dC12_depsilon"] = 2.0 * epsilon_pair * sigma_pair12 / state_i.epsilon
            if payload_i:
                partials[elem_i] = payload_i

        if state_j.train_sigma or state_j.train_epsilon:
            payload_j: Dict[str, float] = {}
            if state_j.train_sigma:
                payload_j["dC6_dsigma"] = 12.0 * epsilon_pair * sigma_pair ** 5
                payload_j["dC12_dsigma"] = 24.0 * epsilon_pair * sigma_pair ** 11
            if state_j.train_epsilon and state_j.epsilon > 0.0:
                payload_j["dC6_depsilon"] = 2.0 * epsilon_pair * sigma_pair6 / state_j.epsilon
                payload_j["dC12_depsilon"] = 2.0 * epsilon_pair * sigma_pair12 / state_j.epsilon
            if payload_j:
                partials.setdefault(elem_j, {})
                for key, value in payload_j.items():
                    partials[elem_j][key] = partials[elem_j].get(key, 0.0) + value

        return c6_pair, c12_pair, partials

    raise ValueError(f"Unsupported combination rule: {comb_rule}")


def _initialise_derivative_container(space: ParameterSpace) -> Dict[str, Dict[str, Vector]]:
    derivs: Dict[str, Dict[str, Vector]] = {}
    for category in space.train_categories:
        info = space.categories[category]
        container: Dict[str, Vector] = {}
        if info.train_sigma:
            container["sigma"] = [0.0, 0.0, 0.0]
        if info.train_epsilon:
            container["epsilon"] = [0.0, 0.0, 0.0]
        if info.train_charge:
            container["charge"] = [0.0, 0.0, 0.0]
        if container:
            derivs[category] = container
    return derivs


def _add_bond_forces(ctx: GroupContext, entry: EntryData, force: Vector) -> None:
    center = entry.center_index
    coords = entry.coords
    base_elements = entry.base_elements
    center_elem = base_elements[center]
    for neighbor in entry.neighbors[center]:
        params = _select_bond_entry(ctx.bond_lookup, center_elem, base_elements[neighbor])
        r_vec = _vec_sub(coords[neighbor], coords[center])
        r = _vec_norm(r_vec)
        if r < 1e-12:
            continue
        r0 = float(params.get("r0", 0.0))
        k = float(params.get("k", 0.0))
        fmag = -k * (r - r0)
        scale = fmag / r
        contribution = [component * scale for component in r_vec]
        force[0] -= contribution[0]
        force[1] -= contribution[1]
        force[2] -= contribution[2]


def _add_angle_forces(ctx: GroupContext, entry: EntryData, force: Vector) -> None:
    coords = entry.coords
    base_elements = entry.base_elements
    center_idx = entry.center_index
    neighbors = entry.neighbors
    n_atoms = len(coords)
    for j in range(n_atoms):
        nbs = neighbors[j]
        if len(nbs) < 2:
            continue
        for idx_a in range(len(nbs)):
            i = nbs[idx_a]
            for idx_b in range(idx_a + 1, len(nbs)):
                k = nbs[idx_b]
                if center_idx not in (i, j, k):
                    continue
                pattern = (base_elements[i] + base_elements[j] + base_elements[k]).lower()
                params = _select_angle_entry(ctx.angle_lookup, pattern)
                theta0 = math.radians(float(params.get("theta0_deg", 0.0)))
                k_theta = float(params.get("k_theta", 0.0))
                u = _vec_sub(coords[i], coords[j])
                v = _vec_sub(coords[k], coords[j])
                ru = _vec_norm(u)
                rv = _vec_norm(v)
                if ru < 1e-12 or rv < 1e-12:
                    continue
                cos_th = _dot(u, v) / (ru * rv)
                cos_th = max(-1.0, min(1.0, cos_th))
                theta = math.acos(cos_th)
                dVdtheta = k_theta * (theta - theta0)
                sin_th = max(1e-12, math.sqrt(1.0 - cos_th * cos_th))
                dth_du = [((v[c] / (ru * rv)) - (cos_th * u[c] / (ru * ru))) / sin_th for c in range(3)]
                dth_dv = [((u[c] / (ru * rv)) - (cos_th * v[c] / (rv * rv))) / sin_th for c in range(3)]
                Fi = _vec_scale(dth_du, -dVdtheta)
                Fk = _vec_scale(dth_dv, -dVdtheta)
                Fj = [- (Fi[c] + Fk[c]) for c in range(3)]
                if center_idx == i:
                    force[0] += Fi[0]
                    force[1] += Fi[1]
                    force[2] += Fi[2]
                if center_idx == j:
                    force[0] += Fj[0]
                    force[1] += Fj[1]
                    force[2] += Fj[2]
                if center_idx == k:
                    force[0] += Fk[0]
                    force[1] += Fk[1]
                    force[2] += Fk[2]


def _add_dihedral_forces(ctx: GroupContext, entry: EntryData, force: Vector) -> None:
    coords = entry.coords
    base_elements = entry.base_elements
    categories = entry.categories
    center_idx = entry.center_index
    neighbors = entry.neighbors
    n_atoms = len(coords)
    for j in range(n_atoms):
        for k in neighbors[j]:
            if j >= k:
                continue
            for i in neighbors[j]:
                if i == k:
                    continue
                for l in neighbors[k]:
                    if l == j:
                        continue
                    if center_idx not in (i, j, k, l):
                        continue
                    pattern = (base_elements[i] + base_elements[j] + base_elements[k] + base_elements[l]).lower()
                    canon_pattern, forward = _canonicalize_dihedral_pattern(pattern)
                    start_h = _is_hydroxyl_oxygen_category(categories[i])
                    end_h = _is_hydroxyl_oxygen_category(categories[l])
                    if not forward:
                        start_h, end_h = end_h, start_h
                    variant = _dihedral_variant_label(start_h, end_h)
                    params = _select_dihedral_entry(ctx.dihedral_lookup, canon_pattern, variant, pattern)
                    coeffs = [float(v) for v in params.get("c", [])]
                    if len(coeffs) < 6:
                        coeffs = coeffs + [0.0] * (6 - len(coeffs))
                    b1 = _vec_sub(coords[i], coords[j])
                    b2 = _vec_sub(coords[k], coords[j])
                    b3 = _vec_sub(coords[l], coords[k])
                    c1 = _vec_cross(b2, b3)
                    c2 = _vec_cross(b1, b2)
                    nb2 = max(_vec_norm(b2), 1e-12)
                    nc1 = max(_vec_norm(c1), 1e-12)
                    nc2 = max(_vec_norm(c2), 1e-12)
                    x = _dot(c2, c1)
                    y = nb2 * _dot(b1, c1)
                    phi = math.atan2(y, x)
                    cosp = math.cos(phi)
                    sinp = math.sin(phi)
                    s = 0.0
                    cp = 1.0
                    for n_idx in range(1, 6):
                        s += n_idx * coeffs[n_idx] * cp
                        cp *= cosp
                    dVdphi = -sinp * s
                    inv_nc2_sq = 1.0 / (nc2 * nc2)
                    inv_nc1_sq = 1.0 / (nc1 * nc1)
                    factor_i = nb2 * inv_nc2_sq
                    factor_l = nb2 * inv_nc1_sq
                    dphi_di = _vec_scale(c2, factor_i)
                    dphi_dl = _vec_scale(c1, factor_l)
                    db1b2 = _dot(b1, b2)
                    db3b2 = _dot(b3, b2)
                    term_j = _vec_scale(c2, (db1b2 / nb2) * inv_nc2_sq)
                    term_k = _vec_scale(c1, (db3b2 / nb2) * inv_nc1_sq)
                    dphi_dj = _vec_add(term_j, term_k)
                    dphi_dk = [- (dphi_di[c] + dphi_dj[c] + dphi_dl[c]) for c in range(3)]
                    Fi = _vec_scale(dphi_di, -dVdphi)
                    Fj = _vec_scale(dphi_dj, -dVdphi)
                    Fk = _vec_scale(dphi_dk, -dVdphi)
                    Fl = _vec_scale(dphi_dl, -dVdphi)
                    midpoint = [0.5 * (coords[j][c] + coords[k][c]) for c in range(3)]
                    tau = [0.0, 0.0, 0.0]
                    for r_vec, f_vec in (
                        (_vec_sub(coords[i], midpoint), Fi),
                        (_vec_sub(coords[j], midpoint), Fj),
                        (_vec_sub(coords[k], midpoint), Fk),
                        (_vec_sub(coords[l], midpoint), Fl),
                    ):
                        tau = _vec_add(tau, _vec_cross(r_vec, f_vec))
                    b2_dot = _dot(b2, b2) + 1e-30
                    delta = _vec_scale(_vec_cross(b2, tau), -1.0 / b2_dot)
                    Fj = _vec_add(Fj, delta)
                    Fk = _vec_sub(Fk, delta)
                    if center_idx == i:
                        force[0] += Fi[0]
                        force[1] += Fi[1]
                        force[2] += Fi[2]
                    if center_idx == j:
                        force[0] += Fj[0]
                        force[1] += Fj[1]
                        force[2] += Fj[2]
                    if center_idx == k:
                        force[0] += Fk[0]
                        force[1] += Fk[1]
                        force[2] += Fk[2]
                    if center_idx == l:
                        force[0] += Fl[0]
                        force[1] += Fl[1]
                        force[2] += Fl[2]


def _evaluate_force_and_derivatives_for_entry(
    ctx: GroupContext,
    states: Mapping[str, ElementState],
    entry: EntryData,
) -> Tuple[Vector, Dict[str, Dict[str, Vector]]]:
    center_idx = entry.center_index
    coords = entry.coords
    categories = entry.categories
    center_category = categories[center_idx]
    q_center = states[center_category].charge
    distances = entry.topological_distances

    force = [0.0, 0.0, 0.0]
    derivs = _initialise_derivative_container(ctx.parameter_space)

    for j, coord_j in enumerate(coords):
        if j == center_idx:
            continue

        distance = distances[j] if 0 <= j < len(distances) else -1
        if distance in (1, 2):
            continue

        scale_lj = 1.0
        scale_qq = 1.0
        if distance == 3:
            scale_lj = ctx.fudge_lj
            scale_qq = ctx.fudge_qq

        dx = coord_j[0] - coords[center_idx][0]
        dy = coord_j[1] - coords[center_idx][1]
        dz = coord_j[2] - coords[center_idx][2]
        r2 = dx * dx + dy * dy + dz * dz
        if r2 == 0.0:
            continue
        inv_r = 1.0 / math.sqrt(r2)
        inv_r2 = inv_r * inv_r
        inv_r3 = inv_r2 * inv_r
        inv_r6 = inv_r2 * inv_r2 * inv_r2
        inv_r7 = inv_r6 * inv_r
        inv_r13 = inv_r7 * inv_r6

        neighbor_category = categories[j]
        q_neighbor = states[neighbor_category].charge

        # Electrostatics
        coul_coeff = scale_qq * KELEC * q_center * q_neighbor * inv_r3
        fx = coul_coeff * dx
        fy = coul_coeff * dy
        fz = coul_coeff * dz
        force[0] += fx
        force[1] += fy
        force[2] += fz

        if center_category in derivs and "charge" in derivs[center_category]:
            coeff_center = scale_qq * KELEC * q_neighbor * inv_r3
            vec_center = derivs[center_category]["charge"]
            vec_center[0] += coeff_center * dx
            vec_center[1] += coeff_center * dy
            vec_center[2] += coeff_center * dz
        if neighbor_category in derivs and "charge" in derivs[neighbor_category]:
            coeff_neighbor = scale_qq * KELEC * q_center * inv_r3
            vec_neighbor = derivs[neighbor_category]["charge"]
            vec_neighbor[0] += coeff_neighbor * dx
            vec_neighbor[1] += coeff_neighbor * dy
            vec_neighbor[2] += coeff_neighbor * dz

        state_i = states[center_category]
        state_j = states[neighbor_category]
        c6_pair, c12_pair, partials = _combine_lj(
            center_category, state_i, neighbor_category, state_j, ctx.comb_rule
        )

        lj_scalar = scale_lj * (12.0 * c12_pair * inv_r13 - 6.0 * c6_pair * inv_r7)
        force[0] += lj_scalar * dx
        force[1] += lj_scalar * dy
        force[2] += lj_scalar * dz

        if not partials:
            continue

        for category, part in partials.items():
            if category not in derivs:
                continue
            if "dC12_dsigma" in part and "sigma" in derivs[category]:
                coeff_sigma = scale_lj * (
                    12.0 * inv_r13 * part.get("dC12_dsigma", 0.0)
                    - 6.0 * inv_r7 * part.get("dC6_dsigma", 0.0)
                )
                vec_sigma = derivs[category]["sigma"]
                vec_sigma[0] += coeff_sigma * dx
                vec_sigma[1] += coeff_sigma * dy
                vec_sigma[2] += coeff_sigma * dz
            if "dC12_depsilon" in part and "epsilon" in derivs[category]:
                coeff_eps = scale_lj * (
                    12.0 * inv_r13 * part.get("dC12_depsilon", 0.0)
                    - 6.0 * inv_r7 * part.get("dC6_depsilon", 0.0)
                )
                vec_eps = derivs[category]["epsilon"]
                vec_eps[0] += coeff_eps * dx
                vec_eps[1] += coeff_eps * dy
                vec_eps[2] += coeff_eps * dz

    _add_bond_forces(ctx, entry, force)
    _add_angle_forces(ctx, entry, force)
    _add_dihedral_forces(ctx, entry, force)

    return force, derivs

def _build_element_states(space: ParameterSpace, theta: Sequence[float]) -> Dict[str, ElementState]:
    states: Dict[str, ElementState] = {}
    for category, info in space.categories.items():
        if category in space.index_map:
            base = space.index_map[category]
            sigma = math.exp(theta[base]) if info.train_sigma else info.sigma
            epsilon = math.exp(theta[base + 1]) if info.train_epsilon else info.epsilon
            charge = theta[base + 2] if info.train_charge else info.charge
        else:
            sigma = info.sigma
            epsilon = info.epsilon
            charge = info.charge
        states[category] = _prepare_element_state(
            sigma,
            epsilon,
            charge,
            info.train_sigma,
            info.train_epsilon,
            info.train_charge,
        )
    return states


def _make_lj_only_parameter_space(space: ParameterSpace) -> ParameterSpace:
    categories: Dict[str, CategoryInfo] = {}
    for key, info in space.categories.items():
        categories[key] = CategoryInfo(
            key=info.key,
            base_element=info.base_element,
            sigma=info.sigma,
            epsilon=info.epsilon,
            charge=info.charge,
            train_sigma=info.train_sigma,
            train_epsilon=info.train_epsilon,
            train_charge=False,
        )

    train_categories = [
        category
        for category, info in categories.items()
        if info.train_sigma or info.train_epsilon
    ]
    train_categories.sort(key=_category_sort_key)
    index_map = {category: 3 * idx for idx, category in enumerate(train_categories)}

    return ParameterSpace(
        categories=categories,
        train_categories=tuple(train_categories),
        index_map=index_map,
        sigma_bounds=space.sigma_bounds,
        epsilon_bounds=space.epsilon_bounds,
        charge_bounds=space.charge_bounds,
    )


def _synchronise_states(
    destination: ParameterSpace,
    states: Mapping[str, ElementState],
    include_charge: bool,
) -> None:
    for category, state in states.items():
        info = destination.categories.get(category)
        if not info:
            continue
        info.sigma = state.sigma
        info.epsilon = state.epsilon
        if include_charge:
            info.charge = state.charge


def _perturb_theta(
    theta: List[float],
    space: ParameterSpace,
    stddev: float,
    rng: random.Random,
) -> None:
    if stddev <= 0.0:
        return
    for category in space.train_categories:
        base = space.index_map[category]
        info = space.categories[category]
        if info.train_sigma:
            theta[base] += rng.gauss(0.0, stddev)
        if info.train_epsilon:
            theta[base + 1] += rng.gauss(0.0, stddev)
        if info.train_charge:
            theta[base + 2] += rng.gauss(0.0, stddev)


def _prepare_gradients(
    grads: Sequence[float],
    normalise: bool,
    grad_clip: Optional[float],
) -> List[float]:
    processed = list(grads)
    if not processed:
        return processed
    if normalise:
        norm = math.sqrt(sum(g * g for g in processed))
        if norm > 0.0:
            processed = [g / norm for g in processed]
    if grad_clip is not None and grad_clip > 0.0:
        limit = float(grad_clip)
        processed = [max(-limit, min(limit, g)) for g in processed]
    return processed


def _evaluate_group_metrics(
    ctx: GroupContext,
    theta: Sequence[float],
    need_grad: bool,
) -> Tuple[float, List[float], List[Vector], List[float]]:
    states = _build_element_states(ctx.parameter_space, theta)
    total_loss = 0.0
    predictions: List[Vector] = []
    entry_losses: List[float] = []
    grads: List[float] = []
    if need_grad and ctx.parameter_space.train_categories:
        grads = [0.0 for _ in range(3 * len(ctx.parameter_space.train_categories))]
    factor = 2.0 / 3.0 / ctx.kbt if grads else 0.0

    for entry in ctx.entries:
        force, derivs = _evaluate_force_and_derivatives_for_entry(ctx, states, entry)
        pred_force_kbt = [component / ctx.kbt for component in force]
        residual = [pred_force_kbt[i] - entry.target_force[i] for i in range(3)]
        loss = sum(r * r for r in residual) / 3.0
        total_loss += loss
        predictions.append(pred_force_kbt)
        entry_losses.append(loss)

        if not grads:
            continue

        for category, partials in derivs.items():
            base = ctx.parameter_space.index_map.get(category)
            if base is None:
                continue
            info = ctx.parameter_space.categories[category]
            if info.train_sigma and "sigma" in partials:
                grads[base] += factor * _dot(residual, partials["sigma"]) * states[category].sigma
            if info.train_epsilon and "epsilon" in partials:
                grads[base + 1] += factor * _dot(residual, partials["epsilon"]) * states[category].epsilon
            if info.train_charge and "charge" in partials:
                grads[base + 2] += factor * _dot(residual, partials["charge"])

    count = len(ctx.entries)
    if count == 0:
        raise ValueError("Optimisation context contains no entries")

    avg_loss = total_loss / count
    if grads:
        grads = [g / count for g in grads]
    return avg_loss, grads, predictions, entry_losses


def _compute_total_charge(ctx: GroupContext, theta: Sequence[float]) -> float:
    """Return the total charge implied by ``theta`` for the optimisation context."""

    total = 0.0
    space = ctx.parameter_space
    for category, info in space.categories.items():
        count = ctx.category_counts.get(category, 0)
        if count == 0:
            continue
        if info.train_charge and category in space.index_map:
            base = space.index_map[category]
            charge = theta[base + 2]
        else:
            charge = info.charge
        total += count * charge
    return total


def _enforce_charge_neutrality(theta: List[float], ctx: GroupContext) -> None:
    """Project the charge parameters back onto the charge-neutral hyperplane."""

    space = ctx.parameter_space
    adjustable = []
    for category, info in space.categories.items():
        if not info.train_charge:
            continue
        if category not in space.index_map:
            continue
        count = ctx.category_counts.get(category, 0)
        if count == 0:
            continue
        adjustable.append((category, count, space.index_map[category] + 2))

    residual = _compute_total_charge(ctx, theta)
    if not adjustable:
        if abs(residual) > 1.0e-8:
            raise ValueError(
                "Charge neutrality cannot be satisfied because no trainable charges are available"
            )
        return

    min_charge, max_charge = space.charge_bounds
    tolerance = 1.0e-8

    # Iteratively shift charges until neutrality is satisfied or bounds are hit.
    active = list(adjustable)
    while active and abs(residual) > tolerance:
        total_weight = sum(weight for _, weight, _ in active)
        if total_weight <= 0.0:
            break
        delta = residual / total_weight
        saturated: List[Tuple[str, int, int]] = []
        for category, weight, index in active:
            updated = theta[index] - delta
            if updated < min_charge:
                theta[index] = min_charge
                saturated.append((category, weight, index))
            elif updated > max_charge:
                theta[index] = max_charge
                saturated.append((category, weight, index))
            else:
                theta[index] = updated

        residual = _compute_total_charge(ctx, theta)
        if saturated:
            active = [item for item in active if item not in saturated]
        else:
            # No bounds were hit; neutrality should now be satisfied up to tolerance.
            break

    residual = _compute_total_charge(ctx, theta)
    if abs(residual) > tolerance:
        raise ValueError(
            "Charge neutrality could not be enforced within the configured charge bounds"
        )


def _clamp_theta(theta: List[float], ctx: GroupContext) -> None:
    space = ctx.parameter_space
    for category in space.train_categories:
        base = space.index_map[category]
        info = space.categories[category]
        if info.train_sigma:
            sigma = math.exp(theta[base])
            sigma = min(space.sigma_bounds[1], max(space.sigma_bounds[0], sigma))
            theta[base] = math.log(sigma)
        else:
            theta[base] = math.log(info.sigma)
        if info.train_epsilon:
            epsilon = math.exp(theta[base + 1])
            epsilon = min(space.epsilon_bounds[1], max(space.epsilon_bounds[0], epsilon))
            theta[base + 1] = math.log(epsilon)
        else:
            theta[base + 1] = math.log(info.epsilon)
        if info.train_charge:
            charge = theta[base + 2]
            charge = min(space.charge_bounds[1], max(space.charge_bounds[0], charge))
            theta[base + 2] = charge
        else:
            theta[base + 2] = info.charge

    _enforce_charge_neutrality(theta, ctx)


def _summarise_categories(
    space: ParameterSpace,
    category_counts: Mapping[str, int],
    theta_best: Sequence[float],
) -> Dict[str, Mapping[str, object]]:
    summary: Dict[str, Mapping[str, object]] = {}
    for category, info in sorted(space.categories.items(), key=lambda item: _category_sort_key(item[0])):
        if category in space.index_map:
            base = space.index_map[category]
            sigma_opt = math.exp(theta_best[base]) if info.train_sigma else info.sigma
            epsilon_opt = math.exp(theta_best[base + 1]) if info.train_epsilon else info.epsilon
            charge_opt = theta_best[base + 2] if info.train_charge else info.charge
        else:
            sigma_opt = info.sigma
            epsilon_opt = info.epsilon
            charge_opt = info.charge
        summary[category] = {
            "base_element": info.base_element,
            "count": int(category_counts.get(category, 0)),
            "sigma_initial": info.sigma,
            "epsilon_initial": info.epsilon,
            "charge_initial": info.charge,
            "sigma_optimized": sigma_opt,
            "epsilon_optimized": epsilon_opt,
            "charge_optimized": charge_opt,
            "train_sigma": info.train_sigma,
            "train_epsilon": info.train_epsilon,
            "train_charge": info.train_charge,
        }
    return summary

def _collect_summary_entries(summary: Mapping[str, object]) -> Dict[str, List[Mapping[str, object]]]:
    lj_entries = summary.get("lj", [])
    result: Dict[str, List[Mapping[str, object]]] = {}
    for entry in lj_entries:
        element = str(entry.get("element", "")).strip().lower()
        if not element:
            continue
        result.setdefault(element, []).append(entry)
    return result

def _extract_center_index(item, coords_len, entry_idx):
    # 1) 先按既有字段取
    center_atom = dict(item.get("center_atom", {}) or {})
    if "atom_index" in center_atom:
        ci = int(center_atom["atom_index"])
    elif "center_force_index" in item:
        ci = int(item["center_force_index"])
    else:
        raise KeyError(
            f"[entry={entry_idx}] Missing center index: need center_atom.atom_index or center_force_index"
        )

    if 0 <= ci < coords_len:
        return ci

    gro = item.get("gro_file")
    raise IndexError(
        f"[entry={entry_idx}] Center atom index out of bounds: {ci} / n_atoms={coords_len}"
        + (f", gro={gro}" if gro else "")
    )

def _wrap_mic_delta(delta, box):
    """ 将位移按最小镜像折回到 [-L/2, L/2] 区间。
        box: 3x3 矩阵或 [Lx, Ly, Lz]；只处理正交盒更稳妥。"""
    if box is None:
        return delta
    # 支持 3向量或3x3（取对角）
    if isinstance(box, (list, tuple, np.ndarray)):
        B = np.array(box, float)
        if B.shape == (3,):
            L = B
        elif B.shape == (3, 3):
            L = np.array([B[0,0], B[1,1], B[2,2]], float)
        else:
            return delta
        for k in range(3):
            if L[k] > 0:
                delta[k] -= L[k] * np.round(delta[k] / L[k])
    return delta

def _infer_center_index_from_coord(center_xyz, coords, box=None, tol=1e-4):
    """给定中心坐标 center_xyz（长度3），在 coords(Nx3) 中找最近点索引。
       支持最小镜像。tol 是“完全一致”的阈值；若最近两点相差 < tol，视为歧义。"""
    if center_xyz is None:
        return None, None  # (idx, min_dist)
    c = np.asarray(center_xyz, dtype=float).reshape(3)
    C = np.asarray(coords, dtype=float).reshape(-1, 3)
    dists = np.empty(len(C), dtype=float)
    for i, p in enumerate(C):
        delta = p - c
        delta = _wrap_mic_delta(delta, box)
        dists[i] = math.sqrt((delta * delta).sum())
    argmin = int(np.argmin(dists))
    d0 = float(dists[argmin])
    # 简单歧义检查
    if len(C) > 1:
        d1 = float(np.partition(dists, 1)[1])
        if abs(d1 - d0) < tol:
            return None, d0  # 歧义
    return argmin, d0

def _get_box_if_any(item):
    """尝试从条目中取盒向量（可按你数据结构修改）。
       支持 'box' 为 [Lx,Ly,Lz] 或 3x3 矩阵。没有就返回 None。"""
    box = item.get("box")
    if box is None:
        # 也有人把它放在 item["cell"] 或 ["unitcell"]
        box = item.get("cell") or item.get("unitcell")
    return box

def _robust_center_index(item, coords, entry_idx, use_infer=True, tol=1e-4):
    n = len(coords)
    # 先尝试按字段读取 index
    try:
        return _extract_center_index(item, n, entry_idx)
    except Exception as e_idx:
        if not use_infer:
            raise
        # 回退：尝试坐标推断
        # 支持 item["center_coord"] 或 item["center_atom"]["xyz"]
        center_xyz = item.get("center_coord")
        if center_xyz is None:
            center_xyz = (item.get("center_atom") or {}).get("xyz")

        idx2, d0 = _infer_center_index_from_coord(
            center_xyz=center_xyz,
            coords=coords,
            box=_get_box_if_any(item),
            tol=tol
        )
        if idx2 is None:
            gro = item.get("gro_file")
            raise IndexError(
                f"[entry={entry_idx}] Cannot infer center index "
                f"(cause: {e_idx}); center_xyz={center_xyz}, min_dist={d0}, gro={gro}"
            )
        return idx2

def build_group_context(
    group_items: Sequence[Tuple[int, Mapping[str, object]]],
    summary: Mapping[str, object],
    train_lj_categories: Sequence[str],
    train_charge_categories: Sequence[str],
    temperature: float,
    sigma_bounds: Tuple[float, float],
    epsilon_bounds: Tuple[float, float],
    charge_bounds: Tuple[float, float],
) -> GroupContext:
    summary_by_element = _collect_summary_entries(summary)
    comb_rule = int(summary.get("defaults", {}).get("comb_rule", 3))

    entries: List[EntryData] = []
    category_counts: Counter[str] = Counter()
    charges_by_base: Dict[str, List[float]] = defaultdict(list)
    charges_by_category: Dict[str, List[float]] = defaultdict(list)

    for idx, item in group_items:
        atom_types = [str(t) for t in item.get("atom_types", [])]
        coords = [list(map(float, xyz)) for xyz in item.get("coordinates", [])]
        charges = [float(q) for q in item.get("formal_charges", [])]
        adjacency = item.get("adj_matrix")
        if adjacency is None:
            raise KeyError("Each dataset entry must contain 'adj_matrix'")
        if not coords or len(coords) != len(atom_types) or len(coords) != len(charges):
            raise ValueError("Mismatch between coordinates, atom types and charges")

        categories = _classify_atom_categories(atom_types, adjacency)
        if len(categories) != len(coords):
            raise ValueError("Category classification failed due to inconsistent lengths")
        neighbors = _build_neighbors(adjacency)
        base_elements_entry = [_base_element_from_category(cat) for cat in categories]

        center_atom = dict(item.get("center_atom", {}))
        center_idx = _robust_center_index(item, coords, idx, use_infer=True, tol=1e-4)

        if not (0 <= center_idx < len(coords)):
            raise IndexError("Center atom index out of bounds")

        target_force = [float(v) for v in item.get("force_label", [])]
        if len(target_force) != 3:
            raise ValueError("force_label must contain exactly three components")

        metadata = EntryMetadata(index=idx, gro_file=item.get("gro_file"), center_atom=center_atom)
        distances = _compute_topological_distances(center_idx, neighbors)

        entries.append(
            EntryData(
                coords=coords,
                categories=categories,
                center_index=center_idx,
                target_force=target_force,
                metadata=metadata,
                neighbors=neighbors,
                base_elements=base_elements_entry,
                topological_distances=distances,
            )
        )

        for atom_idx, category in enumerate(categories):
            base = _base_element_from_category(category)
            category_counts[category] += 1
            charges_by_base[base].append(charges[atom_idx])
            charges_by_category[category].append(charges[atom_idx])

    if not entries:
        raise ValueError("Group contains no entries")

    unique_categories = sorted(category_counts.keys(), key=_category_sort_key)
    train_lj_set = set(train_lj_categories)
    train_charge_set = set(train_charge_categories)

    category_info: Dict[str, CategoryInfo] = {}
    for category in unique_categories:
        base = _base_element_from_category(category)
        candidates = summary_by_element.get(base)
        if not candidates:
            raise KeyError(f"Summary does not contain LJ parameters for base element '{base}'")
        entry = _select_summary_entry(base, charges_by_base.get(base, ()), candidates)
        sigma = min(sigma_bounds[1], max(sigma_bounds[0], entry.sigma))
        epsilon = min(epsilon_bounds[1], max(epsilon_bounds[0], entry.epsilon))
        if charges_by_category.get(category):
            charge = sum(charges_by_category[category]) / len(charges_by_category[category])
        else:
            charge = entry.charge
        if base in {"o", "h"} and category not in {"o_hydroxyl", "o_nonhydroxyl", "h_hydroxyl", "h_nonhydroxyl"}:
            # Unclassified variants fall back to summary charge.
            charge = entry.charge
        charge = min(charge_bounds[1], max(charge_bounds[0], charge))
        category_info[category] = CategoryInfo(
            key=category,
            base_element=base,
            sigma=sigma,
            epsilon=epsilon,
            charge=charge,
            train_sigma=category in train_lj_set,
            train_epsilon=category in train_lj_set,
            train_charge=category in train_charge_set,
        )

    train_categories = [cat for cat in unique_categories if category_info[cat].train_sigma or category_info[cat].train_epsilon or category_info[cat].train_charge]
    index_map = {cat: 3 * idx for idx, cat in enumerate(train_categories)}

    space = ParameterSpace(
        categories=category_info,
        train_categories=tuple(train_categories),
        index_map=index_map,
        sigma_bounds=sigma_bounds,
        epsilon_bounds=epsilon_bounds,
        charge_bounds=charge_bounds,
    )
    kbt = KB_KJ_MOL_K * temperature
    bond_lookup = _build_bond_lookup(summary.get("bonds", []))
    angle_lookup = _build_angle_lookup(summary.get("angles", []))
    dihedral_lookup = _build_dihedral_lookup(summary.get("dihedrals", []))
    defaults = summary.get("defaults", {})
    fudge_lj = float(defaults.get("fudgeLJ", 1.0))
    fudge_qq = float(defaults.get("fudgeQQ", 1.0))
    return GroupContext(
        entries=entries,
        parameter_space=space,
        comb_rule=comb_rule,
        kbt=kbt,
        category_counts=category_counts,
        temperature=temperature,
        bond_lookup=bond_lookup,
        angle_lookup=angle_lookup,
        dihedral_lookup=dihedral_lookup,
        fudge_lj=fudge_lj,
        fudge_qq=fudge_qq,
    )


def _adam_optimize(
    ctx: GroupContext,
    theta0: Sequence[float],
    steps: int,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1.0e-8,
    anneal_rate: float = 1.0,
    grad_clip: Optional[float] = None,
    normalise_grads: bool = True,
    verbose: bool = False,
) -> Tuple[List[float], float, int]:
    if not theta0:
        return list(theta0), 0.0, 0

    theta = list(theta0)
    m = [0.0 for _ in theta]
    v = [0.0 for _ in theta]
    best_theta = list(theta)
    best_loss = float("inf")

    beta1_pow = 1.0
    beta2_pow = 1.0

    for step in range(1, steps + 1):
        loss, grads, _, _ = _evaluate_group_metrics(ctx, theta, need_grad=True)
        if loss < best_loss:
            best_loss = loss
            best_theta = list(theta)
        if verbose and (step == 1 or step % 500 == 0 or step == steps):
            print(f"[step {step:5d}] loss={loss:.6g}")
        if not grads:
            return best_theta, loss, step

        grads = _prepare_gradients(grads, normalise_grads, grad_clip)

        for i, g in enumerate(grads):
            m[i] = beta1 * m[i] + (1.0 - beta1) * g
            v[i] = beta2 * v[i] + (1.0 - beta2) * g * g

        beta1_pow *= beta1
        beta2_pow *= beta2

        lr = learning_rate * (anneal_rate ** (step - 1))
        for i in range(len(theta)):
            m_hat = m[i] / (1.0 - beta1_pow)
            v_hat = v[i] / (1.0 - beta2_pow)
            theta[i] -= lr * m_hat / (math.sqrt(v_hat) + eps)

        _clamp_theta(theta, ctx)

    return best_theta, best_loss, steps


def _optimise_with_restarts(
    ctx: GroupContext,
    theta0: Sequence[float],
    steps: int,
    learning_rate: float,
    anneal_rate: float,
    grad_clip: Optional[float],
    normalise_grads: bool,
    restarts: int,
    restart_std: float,
    seed: Optional[int],
    verbose: bool,
) -> Tuple[List[float], float, int, int]:
    if not theta0:
        return list(theta0), 0.0, 0, 0

    best_theta = list(theta0)
    best_loss = float("inf")
    best_steps = 0
    best_restart = 0

    total_restarts = max(1, restarts)
    rng = random.Random(seed) if seed is not None else random.Random()

    for restart in range(total_restarts):
        theta_init = list(theta0)
        if restart > 0:
            _perturb_theta(theta_init, ctx.parameter_space, restart_std, rng)
        _clamp_theta(theta_init, ctx)
        theta_candidate, loss_candidate, steps_taken = _adam_optimize(
            ctx,
            theta_init,
            steps=steps,
            learning_rate=learning_rate,
            anneal_rate=anneal_rate,
            grad_clip=grad_clip,
            normalise_grads=normalise_grads,
            verbose=verbose,
        )
        if loss_candidate < best_loss:
            best_loss = loss_candidate
            best_theta = theta_candidate
            best_steps = steps_taken
            best_restart = restart

    return best_theta, best_loss, best_steps, best_restart


def _optimise_group_from_kwargs(kwargs: Mapping[str, object]) -> Mapping[str, object]:
    """Adapter for :class:`concurrent.futures` executors."""

    return optimise_group(**kwargs)


def optimise_group(
    group_id: int,
    group_items: Sequence[Tuple[int, Mapping[str, object]]],
    summary: Mapping[str, object],
    train_lj_categories: Sequence[str],
    train_charge_categories: Sequence[str],
    temperature: float,
    steps: int,
    learning_rate: float,
    anneal_rate: float,
    sigma_bounds: Tuple[float, float],
    epsilon_bounds: Tuple[float, float],
    charge_bounds: Tuple[float, float],
    grad_clip: Optional[float],
    normalise_grads: bool,
    restarts: int,
    restart_std: float,
    restart_seed: Optional[int],
    verbose: bool,
    mode: str,
) -> Mapping[str, object]:
    ctx = build_group_context(
        group_items,
        summary,
        train_lj_categories=train_lj_categories,
        train_charge_categories=train_charge_categories,
        temperature=temperature,
        sigma_bounds=sigma_bounds,
        epsilon_bounds=epsilon_bounds,
        charge_bounds=charge_bounds,
    )

    theta_initial = ctx.parameter_space.initial_theta()
    if theta_initial:
        _clamp_theta(theta_initial, ctx)
        initial_states = _build_element_states(ctx.parameter_space, theta_initial)
        _synchronise_states(ctx.parameter_space, initial_states, include_charge=True)
    initial_loss, _, initial_preds, initial_entry_losses = _evaluate_group_metrics(
        ctx, theta_initial, need_grad=False
    )

    stage_records: List[Mapping[str, object]] = []

    # Stage 1 – optimise LJ parameters with frozen charges
    lj_space = _make_lj_only_parameter_space(ctx.parameter_space)
    ctx_lj = GroupContext(
        entries=ctx.entries,
        parameter_space=lj_space,
        comb_rule=ctx.comb_rule,
        kbt=ctx.kbt,
        category_counts=ctx.category_counts,
        temperature=ctx.temperature,
        bond_lookup=ctx.bond_lookup,
        angle_lookup=ctx.angle_lookup,
        dihedral_lookup=ctx.dihedral_lookup,
        fudge_lj=ctx.fudge_lj,
        fudge_qq=ctx.fudge_qq,
    )

    theta_lj0 = ctx_lj.parameter_space.initial_theta()
    if theta_lj0:
        _clamp_theta(theta_lj0, ctx_lj)
    stage1_seed = None if restart_seed is None else restart_seed
    theta_lj_best, _, steps_lj, restart_idx_lj = _optimise_with_restarts(
        ctx_lj,
        theta_lj0,
        steps=steps,
        learning_rate=learning_rate,
        anneal_rate=anneal_rate,
        grad_clip=grad_clip,
        normalise_grads=normalise_grads,
        restarts=restarts,
        restart_std=restart_std,
        seed=stage1_seed,
        verbose=verbose,
    )
    loss_lj_eval, _, _, _ = _evaluate_group_metrics(ctx_lj, theta_lj_best, need_grad=False)
    states_lj = _build_element_states(ctx_lj.parameter_space, theta_lj_best)
    _synchronise_states(ctx.parameter_space, states_lj, include_charge=False)
    stage_records.append(
        {
            "name": "lj_only",
            "best_loss": loss_lj_eval,
            "best_restart": restart_idx_lj,
            "steps_taken": steps_lj,
            "restarts": max(1, restarts) if theta_lj0 else 0,
            "trainable_categories": list(ctx_lj.parameter_space.train_categories),
        }
    )

    # Stage 2 – optimise LJ and charges together
    theta_full0 = ctx.parameter_space.initial_theta()
    if theta_full0:
        _clamp_theta(theta_full0, ctx)
    stage2_seed = None if restart_seed is None else restart_seed + 1
    theta_full_best, _, steps_full, restart_idx_full = _optimise_with_restarts(
        ctx,
        theta_full0,
        steps=steps,
        learning_rate=learning_rate,
        anneal_rate=anneal_rate,
        grad_clip=grad_clip,
        normalise_grads=normalise_grads,
        restarts=restarts,
        restart_std=restart_std,
        seed=stage2_seed,
        verbose=verbose,
    )
    states_full = _build_element_states(ctx.parameter_space, theta_full_best)
    _synchronise_states(ctx.parameter_space, states_full, include_charge=True)
    final_loss, _, final_preds, final_entry_losses = _evaluate_group_metrics(
        ctx, theta_full_best, need_grad=False
    )
    stage_records.append(
        {
            "name": "lj_and_charge",
            "best_loss": final_loss,
            "best_restart": restart_idx_full,
            "steps_taken": steps_full,
            "restarts": max(1, restarts) if theta_full0 else 0,
            "trainable_categories": list(ctx.parameter_space.train_categories),
        }
    )

    category_summary = _summarise_categories(ctx.parameter_space, ctx.category_counts, theta_full_best)

    entries_info: List[Mapping[str, object]] = []
    for entry, initial_pred, final_pred, loss_initial, loss_final in zip(
        ctx.entries,
        initial_preds,
        final_preds,
        initial_entry_losses,
        final_entry_losses,
    ):
        entries_info.append(
            {
                "index": entry.metadata.index,
                "gro_file": entry.metadata.gro_file,
                "center_atom": dict(entry.metadata.center_atom),
                "target_force_kbt_per_nm": list(entry.target_force),
                "initial_force_kbt_per_nm": list(initial_pred),
                "optimized_force_kbt_per_nm": list(final_pred),
                "initial_loss": loss_initial,
                "optimized_loss": loss_final,
            }
        )

    return {
        "group_id": group_id,
        "mode": mode,
        "group_size": len(group_items),
        "indices": [idx for idx, _ in group_items],
        "train_category_order": list(ctx.parameter_space.train_categories),
        "category_parameters": category_summary,
        "category_counts": {cat: int(count) for cat, count in ctx.category_counts.items()},
        "initial_loss": initial_loss,
        "optimized_loss": final_loss,
        "steps_taken": steps_full,
        "configured_steps": steps,
        "learning_rate": learning_rate,
        "anneal_rate": anneal_rate,
        "temperature": ctx.temperature,
        "comb_rule": ctx.comb_rule,
        "grad_clip": grad_clip,
        "grad_normalisation": normalise_grads,
        "restarts": max(1, restarts),
        "restart_std": restart_std,
        "training_stages": stage_records,
        "entries": entries_info,
    }

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="Parameter summary JSON file")
    parser.add_argument("--data", required=True, help="JSON file with processed molecules and forces")
    parser.add_argument("--output", required=True, help="Output JSON file for optimisation results")
    parser.add_argument(
        "--train-elements",
        nargs="+",
        default=["c", "o", "h"],
        help="Base elements whose categories should be optimised (default: c o h)",
    )
    parser.add_argument(
        "--train-categories",
        nargs="+",
        help="Explicit category keys to optimise (overrides --train-elements)",
    )
    parser.add_argument(
        "--disable-charge-training",
        action="store_true",
        help="Do not optimise charges even if categories are selected",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=1,
        help="Number of dataset entries to optimise together (default: 1)",
    )
    parser.add_argument(
        "--fit-all",
        action="store_true",
        help="Optimise a single parameter set against the entire dataset",
    )
    parser.add_argument(
        "--charge-bounds",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=(-1.5, 1.5),
        help="Bounds for charge optimisation (default: -1.5 1.5)",
    )
    parser.add_argument("--temperature", type=float, default=298.0, help="Temperature in Kelvin")
    parser.add_argument("--steps", type=int, default=4000, help="Optimisation steps (default: 4000)")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate for the Adam optimiser (default: 0.05)",
    )
    parser.add_argument(
        "--anneal-rate",
        type=float,
        default=1.0,
        help="Multiplicative annealing factor applied to the learning rate at each step",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Maximum absolute gradient value after normalisation (default: 1.0)",
    )
    parser.add_argument(
        "--disable-grad-normalisation",
        action="store_true",
        help="Disable gradient normalisation before clipping",
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=4,
        help="Number of optimisation restarts per stage (default: 4)",
    )
    parser.add_argument(
        "--restart-std",
        type=float,
        default=0.05,
        help="Standard deviation for Gaussian noise applied during restarts",
    )
    parser.add_argument(
        "--restart-seed",
        type=int,
        help="Seed controlling the random restarts",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for parallel group optimisation (default: 1)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-step loss information")
    return parser.parse_args(argv)


def _chunk_dataset(dataset: Sequence[Mapping[str, object]], size: int) -> List[List[Tuple[int, Mapping[str, object]]]]:
    groups: List[List[Tuple[int, Mapping[str, object]]]] = []
    for start in range(0, len(dataset), size):
        group: List[Tuple[int, Mapping[str, object]]] = []
        for offset in range(size):
            idx = start + offset
            if idx >= len(dataset):
                break
            group.append((idx, dataset[idx]))
        if group:
            groups.append(group)
    return groups


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    summary = load_parameter_summary(args.summary)
    with open(args.data, "r", encoding="utf-8") as fh:
        dataset = json.load(fh)
    if not isinstance(dataset, list):
        raise ValueError("Input data must be a list of molecule entries")
    if not dataset:
        raise ValueError("Dataset is empty")

    if args.train_categories:
        train_lj_categories = _resolve_train_targets(args.train_categories)
    else:
        train_lj_categories = _resolve_train_targets(args.train_elements)
    if not train_lj_categories:
        raise ValueError("No trainable categories specified")
    if args.disable_charge_training:
        train_charge_categories: Sequence[str] = []
    else:
        train_charge_categories = train_lj_categories

    charge_bounds = (float(args.charge_bounds[0]), float(args.charge_bounds[1]))
    if charge_bounds[0] > charge_bounds[1]:
        raise ValueError("charge-bounds must be provided as MIN MAX")

    sigma_bounds = (1.0e-3, 1.0)
    epsilon_bounds = (1.0e-4, 10.0)

    if args.fit_all:
        groups = [[(idx, item) for idx, item in enumerate(dataset)]]
        mode = "all"
    else:
        group_size = max(1, int(args.group_size))
        groups = _chunk_dataset(dataset, group_size)
        mode = "group" if group_size > 1 else "per-entry"

    worker_count = max(1, int(args.workers))
    grad_clip_value = float(args.grad_clip) if args.grad_clip is not None else None
    learning_rate = float(args.learning_rate)
    anneal_rate = float(args.anneal_rate)
    temperature = float(args.temperature)
    restart_std = float(args.restart_std)
    steps = int(args.steps)
    restarts = int(args.restarts)

    jobs: List[Dict[str, object]] = []
    for group_id, group_items in enumerate(groups):
        if args.verbose and (worker_count == 1 or len(groups) == 1):
            print(f"Optimising group {group_id} (size={len(group_items)})")
        job: Dict[str, object] = {
            "group_id": group_id,
            "group_items": group_items,
            "summary": summary,
            "train_lj_categories": train_lj_categories,
            "train_charge_categories": train_charge_categories,
            "temperature": temperature,
            "steps": steps,
            "learning_rate": learning_rate,
            "anneal_rate": anneal_rate,
            "sigma_bounds": sigma_bounds,
            "epsilon_bounds": epsilon_bounds,
            "charge_bounds": charge_bounds,
            "grad_clip": grad_clip_value,
            "normalise_grads": not args.disable_grad_normalisation,
            "restarts": restarts,
            "restart_std": restart_std,
            "restart_seed": args.restart_seed,
            "verbose": args.verbose and (worker_count == 1 or len(groups) == 1),
            "mode": mode,
        }
        jobs.append(job)

    results: List[Mapping[str, object]] = []
    if worker_count == 1 or len(jobs) == 1:
        for job in jobs:
            result = optimise_group(**job)
            results.append(result)
    else:
        if args.verbose:
            print(
                f"Optimising {len(jobs)} groups using {worker_count} worker processes"
            )
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for result in executor.map(_optimise_group_from_kwargs, jobs):
                results.append(result)

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
