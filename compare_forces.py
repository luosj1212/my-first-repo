
# compare_forces.py
# (see chat for detailed description)
from __future__ import annotations
import os, math, shutil, subprocess, re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set
import numpy as np
import pandas as pd

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

def parse_itp(itp_path: str) -> Topology:
    lines = open(itp_path, 'r', errors='ignore').read().splitlines()
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
                idx=int(toks[0]); tname=toks[1]; charge=float(toks[6])
                atoms.append(Atom(idx,tname,charge))
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

def read_gro(gro_path: str):
    lines = open(gro_path,'r',errors='ignore').read().splitlines()
    n = int(lines[1].strip())
    coords = []
    for i in range(2,2+n):
        line=lines[i]
        x=float(line[20:28]); y=float(line[28:36]); z=float(line[36:44])
        coords.append([x,y,z])
    box = list(map(float, lines[2+n].split()[:3]))
    return np.array(coords, dtype=float), np.array(box, dtype=float)

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

def minimum_image_vec(rij: np.ndarray, box: np.ndarray) -> np.ndarray:
    # rij shape (...,3); box shape (3,)
    out = rij.copy()
    for d in range(3):
        L=box[d]
        out[...,d] -= np.rint(out[...,d]/L)*L
    return out

def pair_lj_force(rvec, C6, C12):
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
    import numpy as np, pandas as pd
    invr = np.concatenate(_invr_records, axis=0)  # (M,)
    pd.DataFrame({"invr": invr}).to_csv(path_csv, index=False)
    qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    idx = ["min","p01","p05","q25","median","q75","p95","p99","max"]
    summ = pd.DataFrame({"invr": np.percentile(invr, qs)}, index=idx)
    summ.to_csv(path_summary)
    print(f"[invr] wrote {path_csv} and {path_summary}")
 
def pair_coul_force(rvec, qq):
    r2 = np.sum(rvec*rvec, axis=1)
    invr = 1.0/np.sqrt(np.clip(r2,1e-24,None))
    coef = KELEC * qq * (invr**2)
    return - (coef[:,None] * rvec * invr[:,None])

def compute_forces(top:Topology, coords:np.ndarray, box:np.ndarray,
                   rcoul=1.2, rvdw=1.2,
                   enable_nb=True, enable_bond=True, enable_angle=True, enable_dih=True,
                   split_nb=False):
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



def save_df(arr: np.ndarray, path: str):
    df = pd.DataFrame(arr, columns=['Fx','Fy','Fz'])
    df.index = np.arange(1, len(df)+1)
    df.to_csv(path, index_label='atom')

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--gro', default='peg9.gro')
    ap.add_argument('--itp', default='peg9.itp')
    ap.add_argument('--rcoul', type=float, default=1.2)
    ap.add_argument('--rvdw', type=float, default=1.2)
    ap.add_argument('--per_term', action='store_true', help='导出逐项CSV（LJ/COUL/BOND/ANGLE/DIH/总和）')
    ap.add_argument('--no_nb_split', action='store_true', help='与per_term合用时：不拆分LJ/COUL，只导出NB一项')
    ap.add_argument('--dump_invr', action='store_true', help='导出 1/r 的明细与分布 (invr_values.csv, invr_summary.csv)')

    args = ap.parse_args()
    top = parse_itp(args.itp)
    enable_invr_dump(args.dump_invr)

    coords, box = read_gro(args.gro)

    if args.per_term:
        parts = compute_forces(top, coords, box, args.rcoul, args.rvdw, split_nb=not args.no_nb_split)
        import pandas as pd, numpy as np
        def dump(name, arr):
            df = pd.DataFrame(arr, columns=['Fx','Fy','Fz'])
            df.index = np.arange(1, len(df)+1)
            df.to_csv(f'my_{name}.csv', index_label='atom')
        if isinstance(parts, dict):
            for k,v in parts.items():
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
 
if __name__=='__main__':
    main()
