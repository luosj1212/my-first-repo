import numpy as np
import torch
from itertools import combinations
import math
from collections import defaultdict
from itertools import combinations
from re import sub
import re

def sanitize_symbol(name: str) -> str:
    m = re.match(r"([A-Z][a-z]?)(?:\d*)", name)
    if not m:
        raise ValueError(f"无法解析原子符号: {name}")
    return m.group(1)

def read_gro(file_path):
    """
    简易示例：解析 .gro 文件，返回 (atom_names, coords_in_angstrom, box_in_angstrom)
      - atom_names: list[str]，每个元素是该原子的名称
      - coords_in_angstrom: shape = (N, 3) 的 numpy 数组，单位 Å
      - box_in_angstrom: shape = (3,) 的 numpy 数组，对应 x, y, z 方向盒子长度，单位 Å
    """
    atom_names = []
    coords = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # 第2行是原子数
    num_atoms = int(lines[1].strip())
    
    # 读取坐标行（第3行到第 2+num_atoms 行）
    for line in lines[2:2 + num_atoms]:
        parts = line.split()
        atom_name = parts[1]
        x, y, z = map(float, parts[-3:])  # 原文件中是 nm
        atom_names.append(atom_name)
        coords.append([x, y, z])
    
    coords = np.array(coords)  # shape = (num_atoms, 3)，单位 nm
    coords_in_angstrom = coords * 10.0  # 转到 Å
    
    # 读取最后一行的盒子尺寸（第 3+num_atoms 行，或 lines[-1]）
    box_parts = lines[2 + num_atoms].split()
    box_nm = list(map(float, box_parts[:3]))  # 只取前三个值对应 x,y,z（nm）
    box_in_angstrom = np.array(box_nm) * 10.0  # 转到 Å
    
    return atom_names, coords_in_angstrom, box_in_angstrom


def find_bonds(atom_names, coords, reference_bond_lengths, threshold_factor=1.2):
    """
    根据原子类型和阈值判断是否成键，返回 bonds = [(i, j, (type_i, type_j)), ...]
      - atom_names: [str], 每个原子的类型/名称
      - coords: shape = (N,3) 的 numpy数组 (单位Å)
      - reference_bond_lengths: dict {("C","H"): 1.09, ("C","C"): 1.54, ...}
      - threshold_factor: 乘以参考键长的因子，决定判断是否成键的阈值
    """
    N = len(atom_names)
    bonds = []
    
    for i in range(N):
        for j in range(i+1, N):
            atom_i = atom_names[i]
            atom_j = atom_names[j]
            r_ij = np.linalg.norm(coords[j] - coords[i])  # 距离 Å
            # print(f"{atom_i}-{atom_j}, distance = {r_ij:.3f} Å")

            
            # 将 (atom_i, atom_j) 排序成元组，以便在字典中查参数
            sorted_pair = tuple(sorted([atom_i, atom_j]))
            
            if sorted_pair in reference_bond_lengths:
                ref_len = reference_bond_lengths[sorted_pair]
                if r_ij < ref_len * threshold_factor:
                    # 认为它们成键
                    bonds.append((i, j, sorted_pair))
    return bonds

def find_angles(bond_dict, atom_names):
    """
    根据键连接关系（邻接字典）查找所有键角 (i-j-k)。
    返回的每个角包含原子索引 (i, j, k) 以及对应的角类型 (type_i, type_j, type_k)。
    其中 j 为中心原子，顺序保持为 i, j, k。
    """
    angles = []
    
    # j 作为中心原子，neighbors 里放的是和 j 直接成键的原子
    for j, neighbors in bond_dict.items():
        # 至少得有2个邻居，才能形成角
        if len(neighbors) < 2:
            continue

        if atom_names[j] =='H':
            continue
        
        # 两两组合
        from itertools import combinations
        for i, k in combinations(neighbors, 2):
            type_i = atom_names[i]
            type_j = atom_names[j]
            type_k = atom_names[k]
            angles.append((i, j, k, (type_i, type_j, type_k)))
    return angles





# -----------------------
# 你的参考键长 (单位 Å)
reference_bond_lengths = {
    ("C","H"): 1.09,
    ("C","C"): 1.54,
    ("H","O"): 0.96,
    ("C","O"): 1.41
}

# 你的 Morse 势参数 (以 (atom_i, atom_j) 排序为 key)
morse_params = {
    ("C","H"): {"D_e": 0.1513072, "a": 2.024696, "r_e": 1.090882, "E_min": 0},
    ("C","C"): {"D_e": 0.185166,  "a": 1.716607, "r_e": 1.513616, "E_min": 0},
    ("H","O"): {"D_e": 0.1681193, "a": 2.403812, "r_e": 0.9614723,"E_min": 0},
    ("C","O"): {"D_e": 0.1860028, "a": 1.882232, "r_e": 1.411854, "E_min": 0}
}

double_morse_params = {
    ("C","H"): {"De1": 1.262936e-02, "a1": 3.252809e+00, "re1": 1.126278e+00, "De2": 2.219744e-01, "a2": 1.426049e+00, "re2": 1.074146e+00, "C6":  2.248497e-35, "E0": 0},
    ("C","C"): {"De1": 4.932644e-03, "a1": 3.128515e+00, "re1": 1.592951e+00, "De2": 2.379999e-01, "a2": 1.430262e+00, "re2": 1.509827e+00, "C6":  1.148556e-02, "E0": 0},
    ("H","O"): {"De1": 2.971030e-02, "a1": 3.484522e+00, "re1": 9.832611e-01, "De2": 2.775051e-01, "a2": 1.386088e+00, "re2": 9.393914e-01, "C6":  1.163964e-07, "E0": 0},
    ("C","O"): {"De1": 1.108080e-02, "a1": 3.292682e+00, "re1": 1.425023e+00, "De2": 2.433501e-01, "a2": 1.460752e+00, "re2": 1.424307e+00, "C6":  3.057675e-04, "E0": 0}
}
# 你的多项式角度势参数 (以 (atom_i, atom_j, atom_k) 排序为 key)
angle_params = {
    ("H","C","O"): {'c0': 0,   'c1': 4.055346e-05, 'c2': 3.944165e-05, 
                    'c3': -8.370803e-08, 'c4': 1.172551e-09, 'c5': 9.882018e-14, 
                    'c6': -9.866312e-14, 'theta_0': 1.119558e+02},  
    ("C","C","H"): {'c0': 0, 'c1': 4.457161e-05, 'c2': 3.615568e-05, 
                    'c3': -4.327502e-08, 'c4': 1.287478e-09, 'c5': 2.624107e-13, 
                    'c6': 1.025311e-14, 'theta_0': 1.104139e+02},  
    ("C","C","O"): {'c0': 0, 'c1': 9.037476e-05, 'c2': 4.956524e-05, 
                    'c3': -3.488209e-07, 'c4': 7.719482e-09, 'c5': -1.281100e-10, 
                    'c6': 1.706349e-12, 'theta_0': 1.078330e+02},
    ("C","O","C"): {'c0': 0, 'c1': 5.333516e-05, 'c2': 3.692404e-05, 
                    'c3': -5.958462e-07, 'c4': 9.555204e-09, 'c5': -2.241497e-10, 
                    'c6': 2.745180e-12, 'theta_0': 1.137309e+02},
    ("C","O","H"): {'c0': 0, 'c1': 6.412168e-05, 'c2': 2.576354e-05, 
                    'c3': -1.837950e-07, 'c4': -6.558377e-10, 'c5': -3.665218e-12, 
                    'c6': -3.633240e-14, 'theta_0': 1.101647e+02},
    ("H","C","H"): {'c0': 0, 'c1': 3.957297e-05, 'c2': 3.393811e-05,
                    'c3': 9.044704e-09,  'c4': 5.393470e-10, 'c5': 4.478010e-12,
                    'c6': -1.039844e-13,'theta_0': 1.088857e+02},
    
}

# ---------- 定义势能函数 ----------
def morse_potential(r, D_e, a, r_e, E_min):
    return D_e * (1 - torch.exp(-a * (r - r_e)))**2 + E_min

def double_morse_disp(r,
                      De1, a1, re1,      # Morse 项 1（短程斥力）
                      De2, a2, re2,      # Morse 项 2（中程吸引）
                      C6,                # dispersion (C6 / r^6)
                      E0):               # 全局能量平移
    V1 = De1 * (1 - torch.exp(-a1 * (r - re1)))**2
    V2 = De2 * (1 - torch.exp(-a2 * (r - re2)))**2
    Vdisp = - C6 / r**6
    return V1 + V2 + Vdisp + E0

def angle_potential(theta, c0, c1, c2, c3, c4, c5, c6, theta_0):
    # 注意：theta_0如果是 np.pi*xxx，需要先转成 float 或者用 torch.tensor
    return (c0 + 
            c1*(theta - float(theta_0)) + 
            c2*(theta - float(theta_0))**2 +
            c3*(theta - float(theta_0))**3 + 
            c4*(theta - float(theta_0))**4 +
            c5*(theta - float(theta_0))**5 + 
            c6*(theta - float(theta_0))**6)

def compute_angle(coords, i, j, k):
    ji_vec = np.array(coords[i]) - np.array(coords[j])
    jk_vec = np.array(coords[k]) - np.array(coords[j])
    # 归一化向量
    ji_unit = ji_vec / np.linalg.norm(ji_vec)
    jk_unit = jk_vec / np.linalg.norm(jk_vec)
    # 计算夹角的余弦值并得到角度
    cos_theta = np.dot(ji_unit, jk_unit)
    cos_theta = max(-1.0, min(1.0, cos_theta))  # 数值稳定性处理
    theta_rad = math.acos(cos_theta)
    return math.degrees(theta_rad)

def total_energy(atom_coords_torch, bonds, angles, double_morse_params, angle_params):
    """
    计算总能量：
      - bonds = [(i, j, (atom_type_i, atom_type_j)), ...]
      - angles= [(i, j, k, (atom_type_i, atom_type_j, atom_type_k)), ...]
    """
    E_total = atom_coords_torch.new_zeros((), requires_grad=True)
    E_bond  = atom_coords_torch.new_zeros((), requires_grad=True)
    E_angle = atom_coords_torch.new_zeros((), requires_grad=True)
    # -- 1. Bond 能量
    for (i, j, pair_type) in bonds:
        params = double_morse_params[pair_type]
        ri = atom_coords_torch[i]
        rj = atom_coords_torch[j]
        r = torch.norm(ri - rj)
        E_bond = E_bond + double_morse_disp(r, **params)
        pot = double_morse_disp(r, **params)
      #  print("pot:", pot, "requires_grad=", pot.requires_grad)

        E_total = E_total + double_morse_disp(r, **params)

    
    # -- 2. Angle 能量 (i-j-k)
    for (i, j, k, angle_type) in angles:
        # 获取该角对应的参数（考虑对称性匹配）
        if angle_type in angle_params:
            angle_dict = angle_params[angle_type]
        else:
            # 反转 i 和 k 来匹配参数
            rev_type = (angle_type[2], angle_type[1], angle_type[0])
            if rev_type in angle_params:
                angle_dict = angle_params[rev_type]
            else:
                raise KeyError(f"未找到角类型 {angle_type} 的参数")
        # 计算当前角 i-j-k 的夹角值（单位：度）
        v1 = atom_coords_torch[i] - atom_coords_torch[j]
        v2 = atom_coords_torch[k] - atom_coords_torch[j]
        cos_theta = torch.dot(v1, v2) / (torch.norm(v1)*torch.norm(v2))
        theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))*180/torch.pi
        # print(theta)
        E_angle = E_angle + angle_potential(theta, **angle_dict)
        E_total = E_total + angle_potential(theta, **angle_dict)
    
    return E_total, E_angle, E_bond


# ----------------- 主程序 -----------------
if __name__ == "__main__":
    import sys

    # 1. 读取 .gro 文件

    if len(sys.argv) > 1:
        gro_file = sys.argv[1]
    else:
        gro_file = "../3m.gro"   
    
    atom_names, coords_in_angstrom, box= read_gro(gro_file)
    print("原子数:", len(atom_names))
    
    # 2. 根据参考键长判断哪些原子成键
    clean_names = [sanitize_symbol(n) for n in atom_names]   # 去编号
    bonds = find_bonds(clean_names, coords_in_angstrom,
                   reference_bond_lengths, threshold_factor=1.5)
    print('bonds', len(bonds))
    # bonds = find_bonds(atom_names, coords_in_angstrom, reference_bond_lengths, threshold_factor=1.2)
    print("识别到的成键对:", bonds)
    bond_dict = defaultdict(list)
    for (i, j, pair_type) in bonds:
        bond_dict[i].append(j)
        bond_dict[j].append(i)
    
    # 3. 根据 bonds 推断出 angles
    angles = find_angles(bond_dict, clean_names)
    print("识别到的三体组合:", angles)
    
    # 4. 将坐标转成 PyTorch 的张量 (可微分)
    coords_torch = torch.tensor(coords_in_angstrom, dtype=torch.float32, requires_grad=True)
    
    # 5. 计算总能量
    E, E_angle, E_bond = total_energy(coords_torch, bonds, angles, double_morse_params, angle_params)
    print("总能量 =", E.item(), )
    
    # 6. 计算受力: 负梯度
    # E.backward()  
    # forces = -coords_torch.grad  # shape = (N, 3)
    
    # # 7. 打印每个原子的力
    # print("每个原子的力 :")
    # for i, (aname, f) in enumerate(zip(atom_names, forces.detach().numpy())):
    #     print(f"  原子{i}({aname}) 力 = {f}")
    
    
    coords_torch.grad = None              # 先清空梯度
    E_bond.backward()    # 保留计算图，供后面再对 E_angle 做 backward
    F_bond = -coords_torch.grad.clone()   # 拷贝一份并取负号得到力

    # 3. 对 angle 势能做 backward
    coords_torch.grad = None  # 再清空梯度
    E_angle.backward()         # 这时计算图最终释放
    F_angle = -coords_torch.grad.clone()

    # 4. 将它们相加得到总力
    F_total = F_bond + F_angle

    # print("F_bond =", F_bond)
    # print("F_angle =", F_angle)
    # print("F_total =", F_total)

    # 这样 F_total 就相当于对 (E_bond + E_angle) 的负梯度
    # 如果你要验证，也可以这样做：
    # coords_torch.grad = None
    # E_total = E_bond + E_angle
    # E_total.backward()
    # F_check = -coords_torch.grad
    # print("F_total 是否等于 F_check？", torch.allclose(F_total, F_check))
