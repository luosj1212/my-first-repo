import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array, capped_distance
from MDAnalysis.lib.nsgrid import FastNS
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import json
import csv
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import tempfile

########################
# 工具函数
########################

def convert_to_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, list):
        return [convert_to_serializable(v) for v in value]
    elif isinstance(value, dict):
        return {k: convert_to_serializable(v) for k, v in value.items()}
    else:
        return value

def save_to_json(output_file, **kwargs):
    data = {key: convert_to_serializable(value) for key, value in kwargs.items()}
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def save_to_csv(output_file, atoms, coordinates, center_atoms, center_coords, center_forces,
                indices, submoleindex, submole):
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame Index', 'Center Index', 'Center Atom', 'Coordinates', 
                         'Forces', 'Submolecule Atoms', 'Submolecule Indices'])
        
        for i in range(len(indices)):
            writer.writerow([
                indices[i],
                center_atoms[i],
                center_coords[i],
                coordinates[i],
                center_forces[i],
                submole[i],
                submoleindex[i]
            ])



########################
# 解析 .itp，获取 (atom_id, atom_symbol) 和 bond 对
########################

def parse_itp_file(itp_file):
    atoms = []
    bonds = []
    with open(itp_file, 'r') as f:
        lines = f.readlines()
        atoms_section = False
        bonds_section = False
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith(";"):
                continue

            if line.startswith("[ atoms ]"):
                atoms_section = True
                bonds_section = False
                continue
            elif line.startswith("[ bonds ]"):
                atoms_section = False
                bonds_section = True
                continue
            elif line.startswith("[") and line.endswith("]"):
                atoms_section = False
                bonds_section = False

            if atoms_section:
                parts = line.split()
                if len(parts) > 4:
                    atom_id = int(parts[0])
                    atom_name = parts[4]
                    # 仅取第一个字符当元素符号 (C, H, O, etc)
                    atoms.append((atom_id, atom_name[0]))
            elif bonds_section:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        b1 = int(parts[0])
                        b2 = int(parts[1])
                        bonds.append((b1, b2))
                    except ValueError:
                        continue
    return atoms, bonds


########################
# 截取子分子
########################

def _as_pos2d32(ag):
    """positions -> contiguous (N,3) float32"""
    arr = np.asarray(ag.positions, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        arr = arr.reshape(-1, 3)
    return np.ascontiguousarray(arr, dtype=np.float32)


def extract_surface_water_subsystems(universe,
                                     polymer_sel="resname LIG",
                                     water_sel="resname SOL",
                                     center_water_max_dist=3.8,
                                     sub_poly_radius=6.0,
                                     sub_water_radius=3.2,
                                     include_hydrogen=True):

    polymer_atoms = universe.select_atoms(polymer_sel)
    water_atoms   = universe.select_atoms(water_sel)
    water_ox      = water_atoms.select_atoms("name O or name OW or name O*")

    subsystems = []
    if polymer_atoms.n_atoms == 0 or water_ox.n_atoms == 0:
        return subsystems

    # 确保坐标二维
    poly_pos  = _as_pos2d32(polymer_atoms)   # float32
    water_pos = _as_pos2d32(water_atoms)     # float32

# 如果不足 3 个原子，直接返回空
    if poly_pos.shape[0] <= 2 or water_pos.shape[0] <= 2:
        return []


    # Step 1: 找表面水 O
    d_pw = distance_array(water_ox.positions, polymer_atoms.positions, box=universe.dimensions)
    min_d_to_poly = d_pw.min(axis=1)
    surf_mask = (min_d_to_poly <= center_water_max_dist)
    surf_residues = water_ox[surf_mask].residues

    # print("polymer_atoms.n_atoms =", polymer_atoms.n_atoms)
    # print("water_atoms.n_atoms   =", water_atoms.n_atoms)
    # print("polymer positions shape:", np.asarray(polymer_atoms.positions).shape)
    # print("water positions shape  :", np.asarray(water_atoms.positions).shape)
    # print("has velocities? poly:", getattr(polymer_atoms, "velocities", None) is not None)
    # print("has velocities? water:", getattr(water_atoms, "velocities", None) is not None)
    box = np.asarray(universe.dimensions, dtype=np.float32)  # shape must be (6,), e.g. [Lx,Ly,Lz,α,β,γ]
    if box.shape != (6,):
        box = None
    # print(box.shape)


    # Step 3: 遍历表面水
    for res in surf_residues:
        centers = res.atoms.select_atoms("name OW or name O or name OH2 or name O*")
        if include_hydrogen:
            centers += res.atoms.select_atoms("name HW* or name H* or name H1 or name H2")

        for c in centers:
            cpos = np.asarray(c.position, dtype=np.float32).reshape(1, 3)

            # ---- 聚合物邻域 ----
            res_poly = capped_distance(
                cpos, poly_pos, max_cutoff=sub_poly_radius, box=box, return_distances=False
            )
            # 兼容不同返回形式
            if isinstance(res_poly, tuple):
                if len(res_poly) == 2:
                    _, poly_local_idx = res_poly
                elif len(res_poly) == 3:
                    _, poly_local_idx, _ = res_poly
                else:
                    raise RuntimeError(f"Unexpected capped_distance return (poly): len={len(res_poly)}")
            else:
                # 极少数版本/实现可能直接返回第二数组
                poly_local_idx = res_poly
            poly_idx = polymer_atoms.indices[poly_local_idx]

            # ---- 水邻域 ----
            res_water = capped_distance(
                cpos, water_pos, max_cutoff=sub_water_radius, box=box, return_distances=False
            )
            if isinstance(res_water, tuple):
                if len(res_water) == 2:
                    _, water_local_idx = res_water
                elif len(res_water) == 3:
                    _, water_local_idx, _ = res_water
                else:
                    raise RuntimeError(f"Unexpected capped_distance return (water): len={len(res_water)}")
            else:
                water_local_idx = res_water
            water_idx = water_atoms.indices[water_local_idx]


            atom_indices = np.unique(np.concatenate([poly_idx, water_idx]))
            subsystems.append({
                "center_index": c.index,
                "center_symbol": "O" if c.name.startswith(("O","OW","OH2")) else "H",
                "atom_indices": atom_indices
            })


    return subsystems


# --- NEW VERSION ---
def extract_submolecule(
    universe,
    center_atom,
    radius,
    water_radius=None,
    polymer_sel="not resname SOL",
    # --- 新增：氢键/亲水触发参数 ---
    hb_radius=3.4,          # 氢键触发：O···X ~ 3.3–3.5 Å
    polar_radius=3.8,       # 亲水位点触发：对聚合物极性位点(PEG的O)放宽到 ~3.8–4.0 Å
    follow_layer=True,      # 是否加“一圈伴随水”
    follow_radius=3.3,      # 伴随水的 O···O 阈值
    follow_cap=12           # 伴随水最多额外保留的水分子数
):
    """
    保留规则（合并取并集）：
      A) 基础水半径：水氧与 center 的距离 < radius  且  水氧与聚合物最近距离 < water_radius
      B) 氢键触发：水氧与任一聚合物原子的最近距离 < hb_radius 且 与 center 距离 < radius
      C) 亲水触发：水氧与任一聚合物“极性位点”（这里取聚合物中的 O*）最近距离 < polar_radius 且 与 center 距离 < radius
      D) 伴随水（一圈）：对已保留的水，再纳入与其 O···O 距离 ≤ follow_radius 的水，数量上限 follow_cap
    """
    center_pos = center_atom.position
    dist_center_all = distance_array(center_pos.reshape(1, 3), universe.atoms.positions)[0]

    # -------- 1) 非水（聚合物等）保持原逻辑 --------
    water_atoms   = universe.select_atoms("resname SOL")
    polymer_atoms = universe.select_atoms(polymer_sel)

    non_water_mask = dist_center_all[polymer_atoms.indices] < radius
    within_non_water = polymer_atoms[non_water_mask]

    # -------- 2) 水：三类触发 + 伴随水 --------
    if water_radius is not None and water_radius > 0 and water_atoms.n_atoms > 0:
        # 只以水氧作为代表点
        water_ox = water_atoms.select_atoms("name O or name OW or name O*")
        if water_ox.n_atoms == 0:
            within_water = universe.atoms[:0]  # 空
        else:
            # 与聚合物所有原子的距离（用于基础/氢键触发）
            d_poly_water = distance_array(water_ox.positions, polymer_atoms.positions) if polymer_atoms.n_atoms > 0 else np.full((water_ox.n_atoms,1), np.inf)
            min_d_to_poly = d_poly_water.min(axis=1)

            # 与聚合物“极性位点”（这里以名字以O开头的聚合物原子近似 PEG 的 O）距离
            polymer_polar = polymer_atoms.select_atoms("name O or name O*")
            if polymer_polar.n_atoms > 0:
                d_polar = distance_array(water_ox.positions, polymer_polar.positions)
                min_d_to_polar = d_polar.min(axis=1)
            else:
                min_d_to_polar = np.full(water_ox.n_atoms, np.inf)

            # 与 center 的距离（限定只考虑中心球内的水）
            d_center_ox = dist_center_all[water_ox.indices]

            # --- 触发条件 ---
            cond_base  = (min_d_to_poly  < water_radius) & (d_center_ox < radius)     # A
            cond_hb    = (min_d_to_poly  < hb_radius)    & (d_center_ox < radius)     # B
            cond_polar = (min_d_to_polar < polar_radius) & (d_center_ox < radius)     # C

            keep_mask = cond_base | cond_hb | cond_polar

            keep_residues = water_ox[keep_mask].residues

            # --- 一圈伴随水（可选） ---
            if follow_layer:
                # 以被保留水的 O 为核心，寻找 O···O ≤ follow_radius 的水
                kept_ox = water_ox[keep_mask]
                if kept_ox.n_atoms > 0:
                    d_oo = distance_array(kept_ox.positions, water_ox.positions)  # shape: (n_kept, n_all)
                    companion_mask = (d_oo <= follow_radius)
                    # 把所有满足的水 O index 收集起来
                    comp_idx = set()
                    for i in range(companion_mask.shape[0]):
                        js = np.where(companion_mask[i])[0]
                        for j in js:
                            # 也限定在中心球内
                            if d_center_ox[j] < radius:
                                comp_idx.add(j)
                            if len(comp_idx) >= follow_cap:
                                break
                        if len(comp_idx) >= follow_cap:
                            break

                    if len(comp_idx) > 0:
                        comp_residues = water_ox[list(comp_idx)].residues
                        keep_residues = keep_residues | comp_residues

            within_water = keep_residues.atoms
    else:
        within_water = universe.atoms[:0]

    # -------- 3) 合并 --------
    final_sub = within_non_water | within_water
    return list(final_sub.indices), final_sub



def write_xyz(submolecule, filename):
    atoms = list(submolecule)  # 强制列表化，确保可重复迭代
    with open(filename, 'w') as f:
        f.write(f"{len(atoms)}\n\n")
        for atom in atoms:
            symbol = atom.name[0]
            x, y, z = atom.position
            f.write(f"{symbol} {x:.4f} {y:.4f} {z:.4f}\n")



def convert_to_rdkit_molecule(submolecule):
    atoms = list(submolecule)
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.xyz', delete=False) as tmp:
        tmp.write(f"{len(atoms)}\n\n")
        for atom in atoms:
            symbol = atom.name[0]
            x, y, z = atom.position
            tmp.write(f"{symbol} {x:.4f} {y:.4f} {z:.4f}\n")
        tmp.flush()
        rdkit_mol = Chem.MolFromXYZFile(tmp.name)

    resnames = [atom.resname for atom in submolecule]
    if rdkit_mol is None:
        raise ValueError("Failed to convert submolecule to RDKit molecule")
    return rdkit_mol, resnames


########################
# 读取 .gro 并保存坐标到一个列表
########################

def read_gro_file(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)
    coords = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # gro 文件前两行是标题和原子数
        for line in lines[2:]:
            parts = line.split()
            if len(parts) < 6:
                continue
            # gro 默认单位 nm，这里乘 10 变 Å
            x = float(parts[3]) * 10
            y = float(parts[4]) * 10
            z = float(parts[5]) * 10
            coords.append((x, y, z))
    return coords


########################
# 在 A-B 之间加氢，得新的坐标
########################

def find_point_near_a(A, B, cut_atom_symbol, neighbor_atom_symbol):
    """
    A为留下来的原子，B为被切掉的原子位置
    如果被切的是氢，氢直接在B的位置上；
    如果被切的是其它原子，则根据键长调整位置。
    """
    bond_lengths = {
        ('C', 'H'): 1.09,
        ('C', 'O'): 1.43,
        ('C', 'C'): 1.54,
        ('O', 'H'): 0.96,
        ('N', 'H'): 1.01,
        ('C', 'N'): 1.47,
    }

    if cut_atom_symbol == 'H':
        # 被切掉的是氢，直接用氢的位置
        return tuple(B)
    else:
        # 根据原子对，取键长 (默认为1.0 Å)
        bond_length = bond_lengths.get((neighbor_atom_symbol, 'H'), 1.0)
        A = np.array(A)
        B = np.array(B)
        AB_vector = B - A
        length = np.linalg.norm(AB_vector)

        if length < 1e-8:
            return tuple(A)

        unit_vector = AB_vector / length
        # 根据键长重新确定氢位置
        new_H_position = A + bond_length * unit_vector
        return tuple(new_H_position)



########################
# 删除孤立氢
########################

def remove_isolated_hydrogens(mol, resnames, threshold=1.1):
    """
    RDKit Mol -> EditableMol, 删掉那些与任何原子距离都> threshold 的氢
    同时保留坐标
    """
    old_conf = mol.GetConformer()
    em = Chem.EditableMol(mol)

    atoms_to_remove = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'H':
            idx = atom.GetIdx()
            pos = np.array(old_conf.GetAtomPosition(idx))
            # 计算它与其他原子的最小距离
            dlist = []
            for nbr in mol.GetAtoms():
                nbr_idx = nbr.GetIdx()
                if nbr_idx == idx: 
                    continue
                nbrpos = np.array(old_conf.GetAtomPosition(nbr_idx))
                d = np.linalg.norm(pos - nbrpos)
                dlist.append(d)
            if dlist and min(dlist) > threshold:
                atoms_to_remove.append(idx)

    # 逆序删除
    for i in sorted(atoms_to_remove, reverse=True):
        em.RemoveAtom(i)
    new_mol = em.GetMol()

    # 重新建 Conformer 并拷贝坐标
    # (因为上面 removeAtom 可能打乱原子序)
    new_conf = Chem.Conformer(new_mol.GetNumAtoms())
    new_mol.AddConformer(new_conf, assignId=True)
    old_idx_to_new_idx = {}
    new_i = 0
    for old_idx, atom in enumerate(mol.GetAtoms()):
        if old_idx not in atoms_to_remove:
            p = old_conf.GetAtomPosition(old_idx)
            new_conf.SetAtomPosition(new_i, p)
            old_idx_to_new_idx[old_idx] = new_i
            new_i += 1
    new_resnames = [name for i, name in enumerate(resnames) if i not in atoms_to_remove]
    return new_mol, new_resnames


########################
# 补氢时，过滤无效 bond
########################

def add_hydrogens_to_breaks(mol_in, atoms_info, bonds_info, sub_coords_local, subindex, universe):
    old_conf = mol_in.GetConformer()
    em = Chem.EditableMol(mol_in)
    subidx_set = set(subindex)

    original_positions = [old_conf.GetAtomPosition(i) for i in range(mol_in.GetNumAtoms())]

    new_h_positions = []
    new_h_resnames = []

    global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(subindex)}

    for (b1, b2) in bonds_info:
        idx1, idx2 = b1 - 1, b2 - 1
        in_sub1, in_sub2 = idx1 in subidx_set, idx2 in subidx_set

        if in_sub1 ^ in_sub2:
            if in_sub1:
                local_idx1 = global_to_local[idx1]
                coordsA = sub_coords_local[local_idx1]
                coordsB = universe.atoms.positions[idx2]
                neighbor_symbol = atoms_info[idx1][1]
                cut_atom_symbol = atoms_info[idx2][1]

            else:  # in_sub2
                local_idx2 = global_to_local[idx2]
                coordsA = sub_coords_local[local_idx2]
                coordsB = universe.atoms.positions[idx1]
                neighbor_symbol = atoms_info[idx2][1]
                cut_atom_symbol = atoms_info[idx1][1]

            if neighbor_symbol != 'H':
                pos = find_point_near_a(coordsA, coordsB, cut_atom_symbol, neighbor_symbol)
                new_h_positions.append((pos, universe.atoms[idx1 if in_sub1 else idx2].resname))

    # 添加氢原子前记录原子数
    original_atom_num = mol_in.GetNumAtoms()

    # 添加所有新的氢原子
    for _ in new_h_positions:
        em.AddAtom(Chem.Atom("H"))

    # 构建新分子
    new_mol = em.GetMol()

    # 最关键的步骤：复制原始构象，而不是新建一个全为0的构象
    new_conf = Chem.Conformer(new_mol.GetNumAtoms())

    # 首先拷贝原有原子的坐标
    for idx in range(original_atom_num):
        new_conf.SetAtomPosition(idx, original_positions[idx])

    # 明确设置所有新增氢原子的坐标 (确保转为float！)
    for i, (pos, _) in enumerate(new_h_positions):
        new_conf.SetAtomPosition(original_atom_num + i, Chem.rdGeometry.Point3D(*(float(x) for x in pos)))

    # 彻底取代默认构象（关键！）
    new_mol.RemoveAllConformers()  # 清除之前的所有构象
    new_mol.AddConformer(new_conf, assignId=True)  # 添加新构象

    # 更新 resnames
    original_resnames = [universe.atoms[i].resname for i in subindex]
    new_resnames = original_resnames + [resname for _, resname in new_h_positions]

    return new_mol, new_resnames






########################
# 连通分子检查：只保留和中心原子在同一 connected component 的原子
########################

def keep_largest_connected_component(mol, center_coords, resnames):
    """
    用坐标比对，找出Mol中最接近 center_coords 的原子 -> 作为 "中心"
    然后用 RDKit 的遍历图算法，保留和它同一个connected component的所有原子。
    """
    from rdkit.Chem.rdchem import Atom
    conf = mol.GetConformer()
    min_d = 1e20
    c_idx = -1
    for i, at in enumerate(mol.GetAtoms()):
        pos = np.array(conf.GetAtomPosition(i))
        d = np.linalg.norm(pos - center_coords)
        if d < min_d:
            min_d = d
            c_idx = i

    # 构建 adjacency
    # RDKit bond( i->j ) means i <-> j
    adjacency = {}
    for i, at in enumerate(mol.GetAtoms()):
        adjacency[i] = []
    for bond in mol.GetBonds():
        i1 = bond.GetBeginAtomIdx()
        i2 = bond.GetEndAtomIdx()
        adjacency[i1].append(i2)
        adjacency[i2].append(i1)

    # BFS/DFS from c_idx
    visited = set()
    stack = [c_idx]
    while stack:
        top = stack.pop()
        if top not in visited:
            visited.add(top)
            for nb in adjacency[top]:
                if nb not in visited:
                    stack.append(nb)

    # visited 里就是和 center connected 的原子 idx
    # 现在要删除那些不在 visited 的原子
    remove_list = []
    for i, at in enumerate(mol.GetAtoms()):
        if i not in visited:
            remove_list.append(i)

    if not remove_list:
        return mol  # 全部连通

    em = Chem.EditableMol(mol)
    for i in sorted(remove_list, reverse=True):
        em.RemoveAtom(i)
    new_mol = em.GetMol()

    # 重新拷贝坐标
    old_conf = mol.GetConformer()
    new_conf = Chem.Conformer(new_mol.GetNumAtoms())
    new_mol.AddConformer(new_conf, assignId=True)

    idx_map = {}
    new_i = 0
    for i, at in enumerate(mol.GetAtoms()):
        if i not in remove_list:
            p = old_conf.GetAtomPosition(i)
            new_conf.SetAtomPosition(new_i, p)
            idx_map[i] = new_i
            new_i += 1

    new_resnames = [name for i, name in enumerate(resnames) if i not in remove_list]
    return new_mol, new_resnames

def process_single_frame(args):
    (frame_idx, gro_name, input_dir, atoms_info, bonds_info,
     wc_kwargs) = args

    u = mda.Universe(os.path.join(input_dir, gro_name))
    results = []

    # ====== 同时处理 O0/H0 ======
    subs = extract_surface_water_subsystems(u, **wc_kwargs)

    for sub in subs:
        sub_indices = sub["atom_indices"]
        center_atom_idx = sub["center_index"]
        center_symbol = sub["center_symbol"]

        submolecule = u.atoms[sub_indices]
        try:
            rd_mol, sub_resnames = convert_to_rdkit_molecule(submolecule)
            sub_coords_local = submolecule.positions
            rd_mol_H, new_resnames = add_hydrogens_to_breaks(
                rd_mol, atoms_info, bonds_info, sub_coords_local, sub_indices, u
            )
            rd_mol_H, new_resnames = remove_isolated_hydrogens(rd_mol_H, new_resnames)
        except Exception as e:
            print(f"❌ Failed at frame {frame_idx}, atom {center_atom_idx}: {e}")
            continue

        c_coords = u.atoms[center_atom_idx].position
        c_force = getattr(u.atoms[center_atom_idx], "force", np.array([0., 0., 0.]))

        final_conf = rd_mol_H.GetConformer()
        Natom_final = rd_mol_H.GetNumAtoms()
        final_coords = []
        final_symbols = []
        for i_atom in range(Natom_final):
            xyz = final_conf.GetAtomPosition(i_atom)
            final_coords.append([xyz.x, xyz.y, xyz.z])
            symb = rd_mol_H.GetAtomWithIdx(i_atom).GetSymbol()
            final_symbols.append(symb)

        # ===== 标记 O0 / H0 =====
        min_dist = 1e20
        c_i = -1
        for i_atom in range(Natom_final):
            xyz = np.array(final_coords[i_atom])
            d = np.linalg.norm(xyz - c_coords)
            if d < min_dist:
                min_dist = d
                c_i = i_atom
        final_symbols[c_i] = "O0" if center_symbol == "O" else "H0"

        results.append((
            final_symbols,
            new_resnames,
            np.array(final_coords, dtype=float),
            center_symbol,  # "O" 或 "H"
            c_coords,
            c_force,
            [a.name for a in submolecule],
            sub_indices,
            (frame_idx, center_atom_idx)
        ))

    return results



########################
# 主流程
########################

def process_gro_files(input_dir, itp_file, output_file):
    atoms_info, bonds_info = parse_itp_file(itp_file)

    all_gro = sorted(
        [fn for fn in os.listdir(input_dir) if fn.startswith("frame_") and fn.endswith(".gro")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )

    # 定义 wc_kwargs
    wc_kwargs = dict(
        polymer_sel="not resname SOL",
        water_sel="resname SOL",
        center_water_max_dist=3.8,
        sub_poly_radius=6.0,
        sub_water_radius=3.2,
        include_hydrogen=True
    )

    args_list = [
        (idx, gro_name, input_dir, atoms_info, bonds_info, wc_kwargs)
        for idx, gro_name in enumerate(all_gro)
    ]


    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(process_single_frame, args_list), total=len(args_list)))

    # 合并所有子结果
    all_atoms = []
    all_resnames = []
    all_coords = []
    all_center_atoms = []
    all_center_coords = []
    all_center_forces = []
    all_submole = []
    all_submoleindex = []
    all_indices = []

    for frame_results in results:
        for res in frame_results:
            atoms, resnames, coords, c_atom, c_coords, c_force, submole, submoleidx, indices = res
            all_atoms.append(atoms)
            all_resnames.append(resnames)
            all_coords.append(coords)
            all_center_atoms.append(c_atom)
            all_center_coords.append(c_coords)
            all_center_forces.append(c_force)
            all_submole.append(submole)
            all_submoleindex.append(submoleidx)
            all_indices.append(indices)

    np.savez(
        output_file,
        atoms=np.array(all_atoms, dtype=object),
        resnames=np.array(all_resnames, dtype=object),
        coordinates=np.array(all_coords, dtype=object),
        center_atoms=np.array(all_center_atoms, dtype=object),
        center_coords=np.array(all_center_coords, dtype=object),
        center_forces=np.array(all_center_forces, dtype=object),
        submole=np.array(all_submole, dtype=object),
        submoleindex=np.array(all_submoleindex, dtype=object),
        indices=np.array(all_indices, dtype=object)
    )


########################
# 主函数
########################

if __name__ == "__main__":
    # 根据需求改这些路径
    input_dir = "./pw"                # 存放 frame_X.gro
    itp_file  = "..//peg36.itp"         # itp
    output_file = "./peg36_25.npz"

    # 截断半径
    radius = 8.0
    water_radius = 2.5  # 如果要对水做特殊截断，可设置 water_radius=8.0

    # 确保目标目录存在
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    process_gro_files(input_dir, itp_file, output_file)
