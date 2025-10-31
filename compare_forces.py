
# compare_forces.py
# (see chat for detailed description)
from __future__ import annotations

import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from dataclasses import dataclass
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, MutableMapping, Sequence, Set, Tuple

try:  # pragma: no cover - 环境可能缺失 numpy
    import numpy as np  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - 仅在缺失时触发
    np = None  # type: ignore

try:  # pragma: no cover - 环境可能缺失 pandas
    import pandas as pd  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    pd = None  # type: ignore

KELEC = 138.935456  # kJ mol^-1 nm e^-2
# >>> ADDED FOR LJ-v DUMP
# 开关与暂存：按需收集 v = -(coef[:,None] * rvec)
# === invr (1/r) dump support ===
DUMP_INVR: bool = False
_invr_records: list = []
def enable_invr_dump(flag: bool):
    global DUMP_INVR, _invr_records
    DUMP_INVR = bool(flag)
    _invr_records = []


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError('该功能需要 numpy，请先安装 numpy')


def _require_pandas() -> None:
    if pd is None:
        raise RuntimeError('该功能需要 pandas，请先安装 pandas')
 
@dataclass
class AtomType:
    name: str
    sigma: float
    epsilon: float
    C6: float
    C12: float

@dataclass
class Atom:
    idx: int
    type_name: str
    name: str
    charge: float

@dataclass
class Bond:
    i: int; j: int; funct: int; r0: float; k: float

@dataclass
class Angle:
    i: int; j: int; k: int; funct: int; theta0_deg: float; k_theta: float

@dataclass
class RB_Dihedral:
    i:int; j:int; k:int; l:int; funct:int; c:Tuple[float,float,float,float,float,float]

@dataclass
class Topology:
    atomtypes: Dict[str, AtomType]
    atoms: List[Atom]
    bonds: List[Bond]
    angles: List[Angle]
    rb_dihedrals: List[RB_Dihedral]
    pairs14: Set[Tuple[int,int]]
    nrexcl: int
    fudgeLJ: float
    fudgeQQ: float
    comb_rule: int


def _element_from_atom(atom: Atom) -> str:
    """推断原子的元素标签（仅取首字母，兼容 C/H/O 系统）。"""

    name = atom.name.strip()
    if name:
        symbol = name[0]
    else:
        symbol = atom.type_name[:1]
    return symbol.lower()


def _element_from_name(name: str) -> str:
    name = name.strip()
    if not name:
        return "x"
    return name[0].lower()


def _build_bond_graph(top: Topology) -> Dict[int, Set[int]]:
    graph: Dict[int, Set[int]] = {atom.idx: set() for atom in top.atoms}
    for bond in top.bonds:
        graph.setdefault(bond.i, set()).add(bond.j)
        graph.setdefault(bond.j, set()).add(bond.i)
    return graph


def _is_hydroxyl_oxygen(idx: int, elements: Dict[int, str], graph: Dict[int, Set[int]]) -> bool:
    if elements.get(idx) != 'o':
        return False
    for nb in graph.get(idx, ()):  # O-H 相连即视为羟基氧
        if elements.get(nb) == 'h':
            return True
    return False


def _is_hydroxyl_hydrogen(idx: int, elements: Dict[int, str], graph: Dict[int, Set[int]]) -> bool:
    if elements.get(idx) != 'h':
        return False
    for nb in graph.get(idx, ()):  # H 与 O 相连即视为羟基氢
        if elements.get(nb) == 'o':
            return True
    return False


def _neighbor_signature(idx: int, elements: Dict[int, str], graph: Dict[int, Set[int]]) -> str:
    neighbors = sorted(elements.get(nb, '?') for nb in graph.get(idx, ()))
    return ''.join(neighbors)


def _angle_symmetry_key(pattern: str) -> str:
    if len(pattern) != 3:
        return pattern
    center = pattern[1]
    ends = ''.join(sorted((pattern[0], pattern[2])))
    return f"{center}|{ends}"


def _dihedral_variant_label(has_start: bool, has_end: bool) -> str:
    if has_start and has_end:
        return 'both'
    if has_start:
        return 'start'
    if has_end:
        return 'end'
    return 'none'


def _canonicalize_dihedral_pattern(pattern: str) -> Tuple[str, bool]:
    """返回 (规范化后的 pattern, 是否原始方向)。"""

    reverse = pattern[::-1]
    if reverse < pattern:
        return reverse, False
    return pattern, True


def _annotate_dihedral(
    dih: RB_Dihedral,
    pattern: str,
    elements: Dict[int, str],
    graph: Dict[int, Set[int]],
) -> Tuple[bool, Tuple[str, ...]]:
    """
    区分 O-C-C-O 中的羟基氧：
      - 返回 (是否存在羟基氧, 羟基所在端的标记集合)。
    """

    if pattern != 'occo':
        return False, tuple()

    hydroxyl_sides: List[str] = []
    if _is_hydroxyl_oxygen(dih.i, elements, graph):
        hydroxyl_sides.append('i')
    if _is_hydroxyl_oxygen(dih.l, elements, graph):
        hydroxyl_sides.append('l')
    return bool(hydroxyl_sides), tuple(hydroxyl_sides)

def parse_itp(itp_path: str) -> Topology:
    with open(itp_path, 'r', errors='ignore') as fh:
        lines = fh.read().splitlines()
    fudgeLJ=fudgeQQ=0.5; comb_rule=3; nrexcl=3
    atomtypes: Dict[str, AtomType] = {}
    atoms: List[Atom] = []
    bonds: List[Bond] = []
    angles: List[Angle] = []
    rb_dihedrals: List[RB_Dihedral] = []
    pairs14: Set[Tuple[int,int]] = set()

    sec=None
    for i,l in enumerate(lines):
        s = l.strip()
        if not s or s.startswith((';','#','; ')):
            continue
        if s.startswith('['):
            sec = s.strip('[]').strip().lower()
            continue

        if sec=='defaults':
            toks = s.split()
            if len(toks)>=5:
                try:
                    comb_rule = int(toks[1])
                    fudgeLJ = float(toks[3]); fudgeQQ = float(toks[4])
                except:
                    pass
            continue

        if sec=='moleculetype':
            toks=s.split()
            if len(toks)>=2 and not toks[0].startswith(';'):
                try:
                    nrexcl = int(toks[1])
                except: pass
            continue

        if sec=='atomtypes':
            toks = s.split()
            if len(toks)>=7 and toks[-3].lower()=='a':
                name=toks[0]
                sigma=float(toks[-2]); epsilon=float(toks[-1])
                C6 = 4.0*epsilon*(sigma**6)
                C12= 4.0*epsilon*(sigma**12)
                atomtypes[name]=AtomType(name,sigma,epsilon,C6,C12)
            continue

        if sec=='atoms':
            toks=s.split()
            if len(toks)>=8:
                idx=int(toks[0]); tname=toks[1]; aname=toks[4]; charge=float(toks[6])
                atoms.append(Atom(idx=idx, type_name=tname, name=aname, charge=charge))
            continue

        if sec=='bonds':
            toks=s.split()
            if len(toks)>=5:
                i1=int(toks[0]); j1=int(toks[1]); funct=int(toks[2]); r0=float(toks[3]); k=float(toks[4])
                bonds.append(Bond(i1,j1,funct,r0,k))
            continue

        if sec=='angles':
            toks=s.split()
            if len(toks)>=6:
                i1=int(toks[0]); j1=int(toks[1]); k1=int(toks[2]); funct=int(toks[3]); th0=float(toks[4]); kk=float(toks[5])
                angles.append(Angle(i=i1, j=j1, k=k1, funct=funct, theta0_deg=th0, k_theta=kk))
            continue

        if sec=='dihedrals':
            toks=s.split()
            if len(toks)>=11:
                i1,j1,k1,l1 = map(int,toks[:4]); funct=int(toks[4])
                if funct==3:
                    c = tuple(map(float, toks[5:11]))
                    rb_dihedrals.append(RB_Dihedral(i1,j1,k1,l1,funct,c))
            continue

        if sec=='pairs':
            toks=s.split()
            if len(toks)>=3:
                i1=int(toks[0]); j1=int(toks[1])
                a,b = (i1,j1) if i1<j1 else (j1,i1)
                pairs14.add((a,b))
            continue
    auto_pairs14 = set()
    for dih in rb_dihedrals:
        i, j, k, l = dih.i, dih.j, dih.k, dih.l
        a, b = (i, l) if i < l else (l, i)
        auto_pairs14.add((a, b))
    pairs14 |= auto_pairs14   # 并入已有显式 [pairs]

    return Topology(atomtypes, atoms, bonds, angles, rb_dihedrals, pairs14, nrexcl, fudgeLJ, fudgeQQ, comb_rule)


def topology_to_dict(top: Topology) -> Dict[str, object]:
    """序列化拓扑和参数，便于保存到 JSON。"""
    atomtypes = {
        name: {
            "sigma": at.sigma,
            "epsilon": at.epsilon,
            "C6": at.C6,
            "C12": at.C12,
        }
        for name, at in top.atomtypes.items()
    }
    atoms = [
        {
            "idx": atom.idx,
            "type_name": atom.type_name,
            "name": atom.name,
            "charge": atom.charge,
        }
        for atom in top.atoms
    ]
    bonds = [asdict(b) for b in top.bonds]
    angles = [asdict(a) for a in top.angles]
    dihedrals = [
        {
            "i": d.i,
            "j": d.j,
            "k": d.k,
            "l": d.l,
            "funct": d.funct,
            "c": list(d.c),
        }
        for d in top.rb_dihedrals
    ]
    data = {
        "atomtypes": atomtypes,
        "atoms": atoms,
        "bonds": bonds,
        "angles": angles,
        "rb_dihedrals": dihedrals,
        "pairs14": [list(pair) for pair in sorted(top.pairs14)],
        "nrexcl": top.nrexcl,
        "fudgeLJ": top.fudgeLJ,
        "fudgeQQ": top.fudgeQQ,
        "comb_rule": top.comb_rule,
    }
    return data


def topology_from_dict(data: MutableMapping[str, object]) -> Topology:
    """从 JSON 数据反序列化拓扑。"""
    atomtypes = {
        name: AtomType(
            name=name,
            sigma=values["sigma"],
            epsilon=values["epsilon"],
            C6=values["C6"],
            C12=values["C12"],
        )
        for name, values in data["atomtypes"].items()
    }

    atoms = [
        Atom(
            idx=item["idx"],
            type_name=item["type_name"],
            name=item["name"],
            charge=item["charge"],
        )
        for item in data["atoms"]
    ]

    bonds = [Bond(**item) for item in data.get("bonds", [])]
    angles = [Angle(**item) for item in data.get("angles", [])]
    dihedrals = [
        RB_Dihedral(
            i=item["i"],
            j=item["j"],
            k=item["k"],
            l=item["l"],
            funct=item["funct"],
            c=tuple(item["c"]),
        )
        for item in data.get("rb_dihedrals", [])
    ]
    pairs14 = {tuple(pair) for pair in data.get("pairs14", [])}

    return Topology(
        atomtypes=atomtypes,
        atoms=atoms,
        bonds=bonds,
        angles=angles,
        rb_dihedrals=dihedrals,
        pairs14=pairs14,
        nrexcl=int(data.get("nrexcl", 3)),
        fudgeLJ=float(data.get("fudgeLJ", 1.0)),
        fudgeQQ=float(data.get("fudgeQQ", 1.0)),
        comb_rule=int(data.get("comb_rule", 3)),
    )


def save_topology_json(top: Topology, path: str, indent: int = 2) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(topology_to_dict(top), f, indent=indent, ensure_ascii=False)


def load_topology_json(path: str) -> Topology:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return topology_from_dict(data)


def _increment_group(groups: MutableMapping[Tuple, dict], key: Tuple, payload: dict) -> None:
    if key in groups:
        groups[key]["count"] += 1
    else:
        data = dict(payload)
        data["count"] = 1
        groups[key] = data


def build_parameter_summary(top: Topology) -> Dict[str, object]:
    """聚合 LJ、键、角、二面角参数，方便检查与持久化。"""

    elements = {atom.idx: _element_from_atom(atom) for atom in top.atoms}
    graph = _build_bond_graph(top)

    type_counts: Dict[str, int] = {}
    type_element: Dict[str, str] = {}
    type_neighbor_patterns: Dict[str, Set[str]] = defaultdict(set)
    type_hydroxyl_flags: Dict[str, Set[str]] = defaultdict(set)
    type_charges: Dict[str, Set[float]] = defaultdict(set)
    for atom in top.atoms:
        type_counts[atom.type_name] = type_counts.get(atom.type_name, 0) + 1
        type_element.setdefault(atom.type_name, elements[atom.idx])
        type_neighbor_patterns[atom.type_name].add(_neighbor_signature(atom.idx, elements, graph))
        type_charges[atom.type_name].add(atom.charge)
        if _is_hydroxyl_oxygen(atom.idx, elements, graph):
            type_hydroxyl_flags[atom.type_name].add('oxygen')
        if _is_hydroxyl_hydrogen(atom.idx, elements, graph):
            type_hydroxyl_flags[atom.type_name].add('hydrogen')

    lj_groups: Dict[Tuple[str, float, float], dict] = {}
    for name, at in top.atomtypes.items():
        count = type_counts.get(name, 0)
        if count == 0:
            continue
        element = type_element.get(name, name[:1].lower())
        key = (element, round(at.sigma, 6), round(at.epsilon, 6))
        entry = lj_groups.get(key)
        if entry is None:
            entry = {
                "element": element,
                "sigma": at.sigma,
                "epsilon": at.epsilon,
                "C6": at.C6,
                "C12": at.C12,
                "count": 0,
                "_neighbor_patterns": set(),
                "_type_names": set(),
                "_charges": set(),
                "_variants": set(),
            }
            lj_groups[key] = entry
        entry["count"] += count
        entry["_neighbor_patterns"].update(type_neighbor_patterns.get(name, set()))
        entry["_type_names"].add(name)
        entry["_charges"].update(type_charges.get(name, {0.0}))
        variants = type_hydroxyl_flags.get(name)
        if element == 'o' and variants:
            entry["_variants"].add('hydroxyl')
        elif element == 'h' and variants:
            entry["_variants"].add('hydroxyl')
        else:
            entry["_variants"].add('default')

    bond_groups: Dict[Tuple, dict] = {}
    for b in top.bonds:
        e1 = elements[b.i]
        e2 = elements[b.j]
        pattern = "".join(sorted((e1, e2)))
        key = (pattern, b.funct, round(b.r0, 6), round(b.k, 6))
        payload = {
            "pattern": pattern,
            "funct": b.funct,
            "r0": b.r0,
            "k": b.k,
        }
        _increment_group(bond_groups, key, payload)

    angle_groups: Dict[Tuple, dict] = {}
    for ang in top.angles:
        pattern = "".join((elements[ang.i], elements[ang.j], elements[ang.k]))
        sym_key = _angle_symmetry_key(pattern)
        key = (sym_key, ang.funct, round(ang.theta0_deg, 6), round(ang.k_theta, 6))
        entry = angle_groups.get(key)
        if entry is None:
            entry = {
                "pattern": pattern,
                "funct": ang.funct,
                "theta0_deg": ang.theta0_deg,
                "k_theta": ang.k_theta,
                "symmetry_key": sym_key,
                "equivalent_patterns": set(),
                "count": 0,
            }
            angle_groups[key] = entry
        entry["count"] += 1
        entry["equivalent_patterns"].add(pattern)

    dihedral_groups: Dict[Tuple, dict] = {}
    for dih in top.rb_dihedrals:
        pattern = "".join(
            (
                elements[dih.i],
                elements[dih.j],
                elements[dih.k],
                elements[dih.l],
            )
        )
        has_hydroxyl, hydroxyl_ends = _annotate_dihedral(dih, pattern, elements, graph)
        canon_pattern, forward = _canonicalize_dihedral_pattern(pattern)
        adjusted_ends = hydroxyl_ends
        if not forward and hydroxyl_ends:
            swapped: List[str] = []
            for end in hydroxyl_ends:
                swapped.append('i' if end == 'l' else 'l')
            adjusted_ends = tuple(swapped)
        start_h = 'i' in adjusted_ends
        end_h = 'l' in adjusted_ends
        variant = _dihedral_variant_label(start_h, end_h)
        key = (canon_pattern, dih.funct, tuple(round(v, 6) for v in dih.c), variant)
        entry = dihedral_groups.get(key)
        if entry is None:
            entry = {
                "pattern": canon_pattern,
                "funct": dih.funct,
                "c": list(dih.c),
                "terminal_hydroxyl": variant != 'none',
                "hydroxyl_variant": variant,
                "equivalent_patterns": set(),
                "symmetry_key": canon_pattern,
                "count": 0,
            }
            dihedral_groups[key] = entry
        entry["count"] += 1
        entry["equivalent_patterns"].add(pattern)

    lj: List[dict] = []
    for entry in lj_groups.values():
        neighbor_patterns = sorted(p for p in entry.pop("_neighbor_patterns"))
        type_names = sorted(entry.pop("_type_names"))
        charges = sorted(entry.pop("_charges"))
        variants = entry.pop("_variants")
        variant = "mixed" if len(variants) > 1 else next(iter(variants))
        charge = charges[0] if charges else 0.0
        payload = dict(entry)
        payload.update(
            {
                "neighbor_patterns": neighbor_patterns,
                "type_names": type_names,
                "charge": charge,
                "charge_variants": charges,
                "variant": variant,
            }
        )
        lj.append(payload)

    lj.sort(key=lambda item: (item["element"], item["sigma"], item["epsilon"]))
    bonds = sorted(
        bond_groups.values(),
        key=lambda item: (item["pattern"], item["funct"], round(item["r0"], 6), round(item["k"], 6)),
    )
    angles_list: List[dict] = []
    for entry in angle_groups.values():
        payload = dict(entry)
        payload["equivalent_patterns"] = sorted(entry["equivalent_patterns"])
        angles_list.append(payload)
    angles = sorted(
        angles_list,
        key=lambda item: (
            item["symmetry_key"],
            item["funct"],
            round(item["theta0_deg"], 6),
            round(item["k_theta"], 6),
        ),
    )
    dihedrals_list: List[dict] = []
    for entry in dihedral_groups.values():
        payload = dict(entry)
        payload["equivalent_patterns"] = sorted(entry["equivalent_patterns"])
        dihedrals_list.append(payload)
    dihedrals = sorted(
        dihedrals_list,
        key=lambda item: (
            item["symmetry_key"],
            item["funct"],
            tuple(round(v, 6) for v in item["c"]),
            item.get("hydroxyl_variant", "none"),
        ),
    )

    return {
        "defaults": {
            "nrexcl": top.nrexcl,
            "fudgeLJ": top.fudgeLJ,
            "fudgeQQ": top.fudgeQQ,
            "comb_rule": top.comb_rule,
        },
        "lj": lj,
        "bonds": bonds,
        "angles": angles,
        "dihedrals": dihedrals,
    }


def save_parameter_summary(summary: Dict[str, object], path: str, indent: int = 2) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=indent, ensure_ascii=False)


def load_parameter_summary(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_gro(gro_path: str) -> Tuple[Any, Any, List[str]]:
    with open(gro_path, 'r', errors='ignore') as fh:
        lines = fh.read().splitlines()
    n = int(lines[1].strip())
    coords: List[List[float]] = []
    names: List[str] = []
    for i in range(2, 2 + n):
        line = lines[i]
        atom_name = line[10:15].strip()
        x = float(line[20:28])
        y = float(line[28:36])
        z = float(line[36:44])
        coords.append([x, y, z])
        names.append(atom_name)
    box = list(map(float, lines[2 + n].split()[:3]))
    if np is not None:
        return np.array(coords, dtype=float), np.array(box, dtype=float), names
    return coords, box, names


def _distance(coords: Any, i: int, j: int) -> float:
    if np is not None and isinstance(coords, np.ndarray):
        return float(np.linalg.norm(coords[i - 1] - coords[j - 1]))
    ax, ay, az = coords[i - 1]
    bx, by, bz = coords[j - 1]
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)


def _build_angle_entries(neighbors: Set[int], exclude: int) -> List[Tuple[int, int]]:
    items = sorted(nb for nb in neighbors if nb != exclude)
    return [(items[i], items[j]) for i in range(len(items)) for j in range(i + 1, len(items))]


def _select_lj_entry(
    element: str,
    neighbor_pattern: str,
    hydroxyl_flag: bool,
    lj_entries: List[dict],
) -> Tuple[int, dict]:
    candidates: List[Tuple[int, dict]] = []
    fallback: List[Tuple[int, dict]] = []
    for idx, entry in enumerate(lj_entries):
        if entry.get("element") != element:
            continue
        patterns = entry.get("neighbor_patterns") or []
        variant = entry.get("variant", "default")
        if patterns and neighbor_pattern and neighbor_pattern not in patterns:
            fallback.append((idx, entry))
            continue
        if variant == 'hydroxyl' and not hydroxyl_flag:
            fallback.append((idx, entry))
            continue
        if variant == 'default' and hydroxyl_flag:
            fallback.append((idx, entry))
            continue
        candidates.append((idx, entry))
    if candidates:
        return candidates[0]
    if fallback:
        return fallback[0]
    raise KeyError(f"无法为元素 {element} 匹配 LJ 参数（邻域 {neighbor_pattern!r}）")


def _match_angle_entry(pattern: str, sym_key: str, angle_lookup: Dict[str, List[dict]]) -> dict:
    options = angle_lookup.get(sym_key)
    if not options:
        raise KeyError(f"缺少角度参数：{sym_key}")
    if len(options) == 1:
        return options[0]
    for entry in options:
        eq = entry.get("equivalent_patterns", [])
        if pattern in eq or pattern[::-1] in eq:
            return entry
    return options[0]


def _match_dihedral_entry(
    pattern: str,
    canon_pattern: str,
    variant: str,
    dihedral_lookup: Dict[Tuple[str, str], List[dict]],
) -> dict:
    options = dihedral_lookup.get((canon_pattern, variant))
    if not options:
        # 尝试回退到不区分羟基的参数
        options = dihedral_lookup.get((canon_pattern, 'mixed')) or dihedral_lookup.get((canon_pattern, 'none'))
    if not options:
        raise KeyError(f"缺少二面角参数：{canon_pattern} ({variant})")
    if len(options) == 1:
        return options[0]
    for entry in options:
        eq = entry.get("equivalent_patterns", [])
        if pattern in eq or pattern[::-1] in eq:
            return entry
    return options[0]


def infer_topology_from_summary(
    summary: MutableMapping[str, object],
    coords: Any,
    atom_names: Sequence[str],
    bond_threshold: float = 1.2,
) -> Topology:
    """根据参数 summary 与新 GRO 原子顺序推断拓扑。"""

    n_atoms = len(atom_names)
    elements = {idx + 1: _element_from_name(atom_names[idx]) for idx in range(n_atoms)}

    bonds_summary = summary.get("bonds", [])
    bond_lookup: Dict[str, List[dict]] = defaultdict(list)
    for entry in bonds_summary:
        pattern = "".join(sorted(entry["pattern"]))
        bond_lookup[pattern].append(entry)

    adjacency: Dict[int, Set[int]] = {idx + 1: set() for idx in range(n_atoms)}
    bonds: List[Bond] = []
    for i in range(1, n_atoms + 1):
        for j in range(i + 1, n_atoms + 1):
            pattern = "".join(sorted((elements[i], elements[j])))
            options = bond_lookup.get(pattern)
            if not options:
                continue
            dist = _distance(coords, i, j)
            best = min(options, key=lambda item: abs(dist - float(item["r0"])))
            threshold = bond_threshold * float(best["r0"])
            if dist <= threshold:
                adjacency[i].add(j)
                adjacency[j].add(i)
                bonds.append(
                    Bond(
                        i=i,
                        j=j,
                        funct=int(best["funct"]),
                        r0=float(best["r0"]),
                        k=float(best["k"]),
                    )
                )

    lj_entries: List[dict] = list(summary.get("lj", []))
    entry_to_type_name: Dict[int, str] = {}
    type_usage: Dict[str, int] = defaultdict(int)
    atomtypes: Dict[str, AtomType] = {}
    atoms: List[Atom] = []

    for idx, name in enumerate(atom_names, start=1):
        element = elements[idx]
        neighbor_pattern = _neighbor_signature(idx, elements, adjacency)
        hydroxyl_flag = False
        if element == 'o':
            hydroxyl_flag = _is_hydroxyl_oxygen(idx, elements, adjacency)
        elif element == 'h':
            hydroxyl_flag = _is_hydroxyl_hydrogen(idx, elements, adjacency)
        entry_index, entry = _select_lj_entry(element, neighbor_pattern, hydroxyl_flag, lj_entries)
        type_name = entry_to_type_name.get(entry_index)
        if type_name is None:
            type_usage[element] += 1
            suffix = type_usage[element]
            base = element.upper()
            variant = entry.get("variant", "default")
            if variant == 'hydroxyl' and element in {'o', 'h'}:
                base += 'H'
            type_name = f"{base}_{suffix}"
            atomtypes[type_name] = AtomType(
                name=type_name,
                sigma=float(entry["sigma"]),
                epsilon=float(entry["epsilon"]),
                C6=float(entry.get("C6", 0.0)),
                C12=float(entry.get("C12", 0.0)),
            )
            entry_to_type_name[entry_index] = type_name
        atoms.append(
            Atom(
                idx=idx,
                type_name=type_name,
                name=name,
                charge=float(entry.get("charge", 0.0)),
            )
        )

    angle_lookup: Dict[str, List[dict]] = defaultdict(list)
    for entry in summary.get("angles", []):
        angle_lookup[entry.get("symmetry_key", entry.get("pattern"))].append(entry)

    angles: List[Angle] = []
    seen_angles: Set[Tuple[int, int, int]] = set()
    for j in range(1, n_atoms + 1):
        neighbors = adjacency[j]
        if len(neighbors) < 2:
            continue
        for i_idx, k_idx in _build_angle_entries(neighbors, exclude=j):
            key = tuple(sorted((i_idx, k_idx)) + [j])
            canonical = (key[0], j, key[1])
            if canonical in seen_angles:
                continue
            seen_angles.add(canonical)
            pattern = "".join((elements[i_idx], elements[j], elements[k_idx]))
            sym_key = _angle_symmetry_key(pattern)
            entry = _match_angle_entry(pattern, sym_key, angle_lookup)
            angles.append(
                Angle(
                    i=i_idx,
                    j=j,
                    k=k_idx,
                    funct=int(entry["funct"]),
                    theta0_deg=float(entry["theta0_deg"]),
                    k_theta=float(entry["k_theta"]),
                )
            )

    dihedral_lookup: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for entry in summary.get("dihedrals", []):
        variant = entry.get("hydroxyl_variant", "none")
        dihedral_lookup[(entry.get("symmetry_key", entry.get("pattern")), variant)].append(entry)

    rb_dihedrals: List[RB_Dihedral] = []
    seen_dihedrals: Set[Tuple[int, int, int, int]] = set()
    for j in range(1, n_atoms + 1):
        for k in adjacency[j]:
            if j >= k:
                continue
            for i_idx in sorted(adjacency[j] - {k}):
                for l_idx in sorted(adjacency[k] - {j}):
                    candidate = (i_idx, j, k, l_idx)
                    reverse = (l_idx, k, j, i_idx)
                    if reverse in seen_dihedrals:
                        continue
                    seen_dihedrals.add(candidate)
                    pattern = "".join(
                        (
                            elements[i_idx],
                            elements[j],
                            elements[k],
                            elements[l_idx],
                        )
                    )
                    canon_pattern, forward = _canonicalize_dihedral_pattern(pattern)
                    left_h = _is_hydroxyl_oxygen(i_idx, elements, adjacency) if elements[i_idx] == 'o' else False
                    right_h = _is_hydroxyl_oxygen(l_idx, elements, adjacency) if elements[l_idx] == 'o' else False
                    if not forward:
                        left_h, right_h = right_h, left_h
                    variant = _dihedral_variant_label(left_h, right_h)
                    entry = _match_dihedral_entry(pattern, canon_pattern, variant, dihedral_lookup)
                    rb_dihedrals.append(
                        RB_Dihedral(
                            i=i_idx,
                            j=j,
                            k=k,
                            l=l_idx,
                            funct=int(entry["funct"]),
                            c=tuple(float(v) for v in entry["c"]),
                        )
                    )

    pairs14: Set[Tuple[int, int]] = set()
    for dih in rb_dihedrals:
        a, b = dih.i, dih.l
        if a > b:
            a, b = b, a
        pairs14.add((a, b))

    defaults = summary.get("defaults", {})
    nrexcl = int(defaults.get("nrexcl", 3))
    fudgeLJ = float(defaults.get("fudgeLJ", 1.0))
    fudgeQQ = float(defaults.get("fudgeQQ", 1.0))
    comb_rule = int(defaults.get("comb_rule", 3))

    return Topology(
        atomtypes=atomtypes,
        atoms=atoms,
        bonds=bonds,
        angles=angles,
        rb_dihedrals=rb_dihedrals,
        pairs14=pairs14,
        nrexcl=nrexcl,
        fudgeLJ=fudgeLJ,
        fudgeQQ=fudgeQQ,
        comb_rule=comb_rule,
    )


def align_coords_to_topology(
    coords: Any,
    atom_names: Iterable[str],
    top: Topology,
) -> Tuple[Any, List[str]]:
    """根据原子名称调整坐标顺序，使其与拓扑一致。"""

    atom_names = list(atom_names)
    target_names = [atom.name for atom in top.atoms]
    if len(atom_names) != len(target_names):
        raise ValueError(
            f"原子数量不一致：GRO 文件含 {len(atom_names)} 个，拓扑含 {len(target_names)} 个"
        )

    if atom_names == target_names:
        return coords, list(atom_names)

    mapping: Dict[str, List[int]] = {}
    for idx, name in enumerate(atom_names):
        mapping.setdefault(name, []).append(idx)

    used_index: Dict[str, int] = {}
    if np is not None:
        reordered = np.zeros_like(coords)
    else:
        reordered = [[0.0, 0.0, 0.0] for _ in range(len(coords))]
    reordered_names: List[str] = []
    for pos, name in enumerate(target_names):
        candidates = mapping.get(name)
        if not candidates:
            raise KeyError(f"无法在 GRO 中找到与拓扑原子 {name!r} 对应的坐标")
        offset = used_index.get(name, 0)
        if offset >= len(candidates):
            raise ValueError(f"拓扑原子 {name!r} 出现次数多于 GRO 文件")
        gro_idx = candidates[offset]
        used_index[name] = offset + 1
        reordered[pos] = coords[gro_idx]
        reordered_names.append(atom_names[gro_idx])

    return reordered, reordered_names

def build_exclusions(top: Topology) -> Set[Tuple[int,int]]:
    n = len(top.atoms)
    adj = {i+1:set() for i in range(n)}
    for b in top.bonds:
        adj[b.i].add(b.j); adj[b.j].add(b.i)
    excl=set()
    for i in range(1,n+1):
        for j in adj[i]:
            a,b = (i,j) if i<j else (j,i)
            excl.add((a,b))
        if top.nrexcl>=2:
            for j in adj[i]:
                for k in adj[j]:
                    if k==i: continue
                    a,b = (i,k) if i<k else (k,i)
                    excl.add((a,b))
    return excl

def minimum_image_vec(rij: Any, box: Any) -> Any:
    _require_numpy()
    # rij shape (...,3); box shape (3,)
    out = rij.copy()
    for d in range(3):
        L=box[d]
        out[...,d] -= np.rint(out[...,d]/L)*L
    return out

def pair_lj_force(rvec, C6, C12):
    _require_numpy()
    r2 = np.sum(rvec*rvec, axis=1)
    invr2 = 1.0/np.clip(r2, 1e-24, None)
    invr6 = invr2**3
    invr12= invr6**2
    invr = np.sqrt(invr2)
    coef = (12.0*C12*invr12 - 6.0*C6*invr6) * invr2
    v = - (coef[:,None] * rvec)
    # >>> ADDED FOR LJ-v DUMP
    if DUMP_INVR:
        # 复制一份，避免后续原地操作影响已存数据
        _invr_records.append(invr.copy())
    return v

# === invr dump writers ===
def write_invr_distribution(path_csv: str = "invr_values.csv", path_summary: str = "invr_summary.csv"):
    """写出 1/r 的逐条明细与分位数摘要。"""
    if not _invr_records:
        print("[invr] no records to dump.")
        return
    _require_numpy()
    _require_pandas()
    invr = np.concatenate(_invr_records, axis=0)  # (M,)
    pd.DataFrame({"invr": invr}).to_csv(path_csv, index=False)
    qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    idx = ["min","p01","p05","q25","median","q75","p95","p99","max"]
    summ = pd.DataFrame({"invr": np.percentile(invr, qs)}, index=idx)
    summ.to_csv(path_summary)
    print(f"[invr] wrote {path_csv} and {path_summary}")
 
def pair_coul_force(rvec, qq):
    _require_numpy()
    r2 = np.sum(rvec*rvec, axis=1)
    invr = 1.0/np.sqrt(np.clip(r2,1e-24,None))
    coef = KELEC * qq * (invr**2)
    return - (coef[:,None] * rvec * invr[:,None])

def compute_forces(top:Topology, coords:Any, box:Any,
                   rcoul=1.2, rvdw=1.2,
                   enable_nb=True, enable_bond=True, enable_angle=True, enable_dih=True,
                   split_nb=False):
    _require_numpy()
    """
    返回：
      - 若 split_nb=False：返回 total_forces (N,3)
      - 若 split_nb=True ：返回 dict:
            {"LJ": F_lj, "COUL": F_coul, "BOND": F_bond, "ANGLE": F_angle, "DIH": F_dih, "TOTAL": F_sum}
    单位：kJ/mol/nm
    """
    n = len(top.atoms)
    zeros = np.zeros_like(coords)

    # 预先准备
    q = np.array([a.charge for a in top.atoms], dtype=float)
    C6 = np.array([top.atomtypes[a.type_name].C6 for a in top.atoms], dtype=float)
    C12= np.array([top.atomtypes[a.type_name].C12 for a in top.atoms], dtype=float)

    excl = build_exclusions(top)
    is14 = top.pairs14

    F_lj   = np.zeros_like(coords)
    F_coul = np.zeros_like(coords)
    F_bond = np.zeros_like(coords)
    F_ang  = np.zeros_like(coords)
    F_dih  = np.zeros_like(coords)

    # -------- 非键：LJ & COUL 分别累加 --------
    if enable_nb:
        for i in range(n-1):
            j_idx = np.arange(i+1, n, dtype=int)
            rij = coords[j_idx] - coords[i][None,:]
            rij = minimum_image_vec(rij, box)
            r = np.linalg.norm(rij, axis=1)

            mask_excl = np.array([ ((i+1, j+1) if i+1<j+1 else (j+1, i+1)) in excl for j in j_idx ])
            mask_14   = np.array([ ((i+1, j+1) if i+1<j+1 else (j+1, i+1)) in is14 for j in j_idx ])

            # LJ
            lj_mask = (~mask_excl) & (r < rvdw)
            if np.any(lj_mask):
                jj = j_idx[lj_mask]
                rij_eff = rij[lj_mask]
                C6ij = np.sqrt(C6[i]*C6[jj])
                C12ij= np.sqrt(C12[i]*C12[jj])
                fudge = np.ones_like(C6ij)
                fudge[mask_14[lj_mask]] *= top.fudgeLJ
                f_lj = pair_lj_force(rij_eff, C6ij*fudge, C12ij*fudge)  # 作用在 i 上
                F_lj[i]  += np.sum(f_lj, axis=0)
                F_lj[jj] -= f_lj

            # COUL
            coul_mask = (~mask_excl) & (r < rcoul)
            if np.any(coul_mask):
                jj = j_idx[coul_mask]
                rij_eff = rij[coul_mask]
                qq = q[i]*q[jj]
                fudge = np.ones_like(qq)
                fudge[mask_14[coul_mask]] *= top.fudgeQQ
                f_c = pair_coul_force(rij_eff, qq*fudge)                # 作用在 i 上
                F_coul[i]  += np.sum(f_c, axis=0)
                F_coul[jj] -= f_c

    # -------- 键：V=0.5*k*(r-r0)^2 --------
    if enable_bond:
        for b in top.bonds:
            if b.funct != 1: 
                continue
            i=b.i-1; j=b.j-1
            rij = coords[j]-coords[i]
            rij = minimum_image_vec(rij[None,:], box)[0]
            r = np.linalg.norm(rij)
            if r < 1e-12: 
                continue
            fmag = -b.k * (r - b.r0)
            fvec = fmag * (rij / r)
            F_bond[i] -= fvec
            F_bond[j] += fvec

    # -------- 角：V = 0.5*k*(θ-θ0)^2 => dV/dθ = k*(θ-θ0) --------
    if enable_angle:
        for ang in top.angles:
            if ang.funct != 1:
                continue
            i = ang.i-1; j = ang.j-1; k = ang.k-1
            u = minimum_image_vec((coords[i]-coords[j])[None,:], box)[0]
            v = minimum_image_vec((coords[k]-coords[j])[None,:], box)[0]
            ru = np.linalg.norm(u); rv = np.linalg.norm(v)
            if ru < 1e-12 or rv < 1e-12: 
                continue
            cos_th = np.clip(np.dot(u,v)/(ru*rv), -1.0, 1.0)
            th = np.arccos(cos_th)
            th0 = np.deg2rad(ang.theta0_deg)
            dVdth = ang.k_theta * (th - th0)      # ← 注意不是 2k
            sin_th = max(1e-12, np.sqrt(1.0 - cos_th*cos_th))
            dth_du = ((v/(ru*rv)) - (cos_th*u/(ru*ru))) / sin_th
            dth_dv = ((u/(ru*rv)) - (cos_th*v/(rv*rv))) / sin_th
            Fi = - dVdth * dth_du
            Fk = - dVdth * dth_dv
            Fj = - (Fi + Fk)
            F_ang[i] -= Fi; F_ang[j] -= Fj; F_ang[k] -= Fk


    # -------- RB 二面角（教科书 v/w 形式，与 GROMACS 号位一致）--------
            # -------- RB dihedral (GMX-compatible φ convention + torque fix) --------
    if enable_dih:
        for dih in top.rb_dihedrals:
            i = dih.i - 1; j = dih.j - 1; k = dih.k - 1; l = dih.l - 1
            # 最小影像 + 固定向量方向
            # b1 = r_i - r_j,  b2 = r_k - r_j,  b3 = r_l - r_k
            b1 = minimum_image_vec((coords[i] - coords[j])[None, :], box)[0]
            b2 = minimum_image_vec((coords[k] - coords[j])[None, :], box)[0]
            b3 = minimum_image_vec((coords[l] - coords[k])[None, :], box)[0]

            # 变体A（与 GMX 号位同源）: c1 = b3×b2, c2 = b1×b2
            c1 = np.cross(b2, b3)
            c2 = np.cross(b1, b2)
            nb2 = max(np.linalg.norm(b2), 1e-12)
            nc1 = max(np.linalg.norm(c1), 1e-12)
            nc2 = max(np.linalg.norm(c2), 1e-12)
            # φ 的定义：phi = -atan2( |b2|*(b1·c1),  c2·c1 )
            x = np.dot(c2, c1)
            y = nb2 * np.dot(b1, c1)
            phi = np.arctan2(y, x)

            # RB 势：V = Σ c_n cos^n φ   →   dV/dφ = + sinφ · Σ n c_n cos^{n-1}φ
            c = dih.c
            cosp = np.cos(phi); sinp = np.sin(phi)
            s = 0.0; cp = 1.0
            for n_ in range(1, 6):
                s += n_ * c[n_] * cp
                cp *= cosp
            dVdphi = - sinp * s
                # φ 的梯度
            dphi_di =  (nb2 / (nc2 * nc2)) * c2
            dphi_dl =   (nb2 / (nc1 * nc1)) * c1
            db1b2 = np.dot(b1, b2)
            db3b2 = np.dot(b3, b2)
            dphi_dj = (db1b2 / nb2) * (c2 / (nc2 * nc2)) + (db3b2 / nb2) * (c1 / (nc1 * nc1))
            dphi_dk = - (dphi_di + dphi_dj + dphi_dl)

                # 原始解析力
            Fi = - dVdphi * dphi_di
            Fj = - dVdphi * dphi_dj
            Fk = - dVdphi * dphi_dk
            Fl = - dVdphi * dphi_dl

                # 使用相对 j 的向量，避免原点依赖/PBC问题
            # 旧：以 j 为参考
            # ri = MIV(coords[i]-coords[j]); rj = 0; rk = MIV(coords[k]-coords[j]); rl = MIV(coords[l]-coords[j])
            # tau = cross(ri,Fi) + cross(rj,Fj) + cross(rk,Fk) + cross(rl,Fl)

            # 新：以 j-k 中点为参考（GMX 指纹一致）
            m = 0.5 * (coords[j] + coords[k])
            ri = minimum_image_vec((coords[i] - m)[None,:], box)[0]
            rj = minimum_image_vec((coords[j] - m)[None,:], box)[0]
            rk = minimum_image_vec((coords[k] - m)[None,:], box)[0]
            rl = minimum_image_vec((coords[l] - m)[None,:], box)[0]
            tau = np.cross(ri, Fi) + np.cross(rj, Fj) + np.cross(rk, Fk) + np.cross(rl, Fl)

            Delta = - np.cross(b2, tau) / (np.dot(b2, b2) + 1e-30)
            Fj += Delta
            Fk -= Delta
                # --------------------------------

            F_dih[i] += Fi; F_dih[j] += Fj; F_dih[k] += Fk; F_dih[l] += Fl

    if split_nb:
        F_tot = F_lj + F_coul + F_bond + F_ang + F_dih
        return {"LJ":F_lj, "COUL":F_coul, "BOND":F_bond, "ANGLE":F_ang, "DIH":F_dih, "TOTAL":F_tot}
    else:
        return F_lj + F_coul + F_bond + F_ang + F_dih



def save_df(arr: Any, path: str):
    _require_numpy()
    _require_pandas()
    df = pd.DataFrame(arr, columns=['Fx','Fy','Fz'])
    df.index = np.arange(1, len(df)+1)
    df.to_csv(path, index_label='atom')


def main():
    import argparse

    ap = argparse.ArgumentParser(description='比较/计算分子力场作用力的辅助工具')
    sub = ap.add_subparsers(dest='command', required=True)

    ap_extract = sub.add_parser('extract', help='从 ITP 文件提取并保存参数')
    ap_extract.add_argument('--itp', default='peg9.itp', help='输入 ITP 文件路径')
    ap_extract.add_argument('--out', default='topology.json', help='保存拓扑及参数的 JSON 文件路径')
    ap_extract.add_argument('--summary', help='参数汇总 JSON 输出路径（默认为 <out> 同目录下 *_summary.json）')

    ap_compute = sub.add_parser('compute', help='根据保存的参数与 GRO 坐标计算受力')
    ap_compute.add_argument('--gro', default='peg9.gro', help='输入 GRO 文件路径')
    ap_compute.add_argument('--params', default='topology.json', help='extract 阶段生成的 JSON 参数文件')
    ap_compute.add_argument('--summary', help='extract 阶段生成的参数汇总 JSON，提供时将根据 GRO 自动推断拓扑')
    ap_compute.add_argument('--rcoul', type=float, default=1.2)
    ap_compute.add_argument('--rvdw', type=float, default=1.2)
    ap_compute.add_argument('--per_term', action='store_true', help='导出逐项CSV（LJ/COUL/BOND/ANGLE/DIH/总和）')
    ap_compute.add_argument('--no_nb_split', action='store_true', help='与per_term合用时：不拆分LJ/COUL，只导出NB一项')
    ap_compute.add_argument('--dump_invr', action='store_true', help='导出 1/r 的明细与分布 (invr_values.csv, invr_summary.csv)')

    args = ap.parse_args()

    if args.command == 'extract':
        top = parse_itp(args.itp)
        out_path = Path(args.out)
        save_topology_json(top, str(out_path))

        if args.summary is None:
            summary_path = out_path.with_name(out_path.stem + '_summary.json')
        else:
            summary_path = Path(args.summary)

        summary = build_parameter_summary(top)
        save_parameter_summary(summary, str(summary_path))
        print(f"[OK] topology saved to {out_path} and summary saved to {summary_path}")
        return

    if args.command == 'compute':
        enable_invr_dump(args.dump_invr)

        coords_raw, box, atom_names = read_gro(args.gro)
        if args.summary:
            summary = load_parameter_summary(args.summary)
            top = infer_topology_from_summary(summary, coords_raw, atom_names)
            coords = coords_raw
        else:
            top = load_topology_json(args.params)
            coords, _ = align_coords_to_topology(coords_raw, atom_names, top)

        if args.per_term:
            parts = compute_forces(top, coords, box, args.rcoul, args.rvdw, split_nb=not args.no_nb_split)

            def dump(name: str, arr: Any) -> None:
                _require_numpy()
                _require_pandas()
                df = pd.DataFrame(arr, columns=['Fx','Fy','Fz'])
                df.index = np.arange(1, len(df)+1)
                df.to_csv(f'my_{name}.csv', index_label='atom')

            if isinstance(parts, dict):
                for k, v in parts.items():
                    dump(k, v)
            else:
                dump('TOTAL', parts)
            print('[OK] wrote per-term CSVs (my_*.csv)')
        else:
            myF = compute_forces(top, coords, box, args.rcoul, args.rvdw)
            save_df(myF, 'my_forces.csv')
            print('[OK] my_forces.csv')

        if args.dump_invr:
            # 计算结束后统一写出
            write_invr_distribution()
        return

if __name__=='__main__':
    main()
