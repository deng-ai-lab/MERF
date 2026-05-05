
from Bio import PDB
import os
import json


def load_residue_mapping(map_file_json):
    """
    一键读入 JSON 格式的映射文件。

    Args:
        map_file_json: JSON 映射文件路径

    Returns:
        mapping_dict: { 'H': { "52_A": 1, "53_ ": 2, ... }, ... }
    """
    try:
        with open(map_file_json, 'r') as f:
            mapping_dict = json.load(f)
        return mapping_dict
    except Exception as e:
        print(f"Error loading mapping file: {e}")
        return None


def renumber_and_create_map(pdb_file_in, pdb_file_out, map_file_out):
    """
    Renumbers residues in a PDB file sequentially, removes insertion codes,
    and generates a JSON mapping file of the changes.

    Three-stage approach:
    1. Read all residues and build mapping
    2. Apply temporary IDs to avoid conflicts
    3. Apply final sequential numbering uniformly
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("my_struct", pdb_file_in)

    # 存储映射: { 'H': { (old_seq, old_icode): new_seq, ... }, ... }
    chain_mapping = {}

    # ========== 第一阶段：读取所有残基并建立映射 ==========
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            residues_in_chain = list(chain.get_residues())

            chain_mapping[chain_id] = {}

            # 按顺序遍历残基，生成新编号
            for new_residue_id, residue in enumerate(residues_in_chain, start=1):
                old_id = residue.get_id()
                # old_id: (hetatm_flag, sequence_id, insertion_code)
                het_flag, old_seq_id, old_icode = old_id

                # 记录映射关系
                chain_mapping[chain_id][(old_seq_id, old_icode)] = new_residue_id

    # ========== 第二阶段：使用临时ID避免冲突 ==========
    # BioPython 要求残基 ID 唯一，直接修改会导致冲突
    # 因此先改成临时 ID（很大的数字），再改成最终 ID
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            residues_in_chain = list(chain.get_residues())

            for idx, residue in enumerate(residues_in_chain):
                old_id = residue.get_id()
                het_flag, old_seq_id, old_icode = old_id

                # 使用临时 ID（10000 + 索引）避免与原有 ID 冲突
                temp_id = (het_flag, 10000 + idx, ' ')
                residue.id = temp_id

    # ========== 第三阶段：修改为最终编号 ==========
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            residues_in_chain = list(chain.get_residues())

            for idx, residue in enumerate(residues_in_chain, start=1):
                old_id = residue.get_id()
                het_flag, _, _ = old_id

                # 现在可以安全地改成最终编号
                new_id = (het_flag, idx, ' ')
                residue.id = new_id

    # ========== 保存修改后的 PDB 文件 ==========
    io = PDB.PDBIO()
    io.set_structure(structure)

    try:
        io.save(pdb_file_out)
        print(f"Renumbered PDB saved to: {pdb_file_out}")
    except Exception as e:
        print(f"Error saving PDB: {e}")

    # ========== 保存映射文件（JSON 格式）==========
    try:
        # 转换为 JSON 兼容格式（键必须是字符串）
        mapping_dict = {}
        for chain_id, res_map in chain_mapping.items():
            mapping_dict[chain_id] = {}
            for (old_seq, old_icode), new_seq in res_map.items():
                # 使用 "old_seq_old_icode" 作为键，例如 "52_A" 或 "53_ "
                key = f"{old_seq}_{old_icode}"
                mapping_dict[chain_id][key] = new_seq

        with open(map_file_out, 'w') as f:
            json.dump(mapping_dict, f, indent=2)
        print(f"Mapping file saved to: {map_file_out}")
    except Exception as e:
        print(f"Error saving map file: {e}")

if __name__ == '__main__':

    pdb_dir = "/home/lfj/projects_dir/MERF/data/ABbind/PDBs/"
    save_dir = pdb_dir.replace("PDBs", "PDBs_renumbered")
    pdb_id_list = ['2NY7', '3BE1', 'HM_3BN9', '3BN9', '3BDY', '1N8Z']

    for pdb_id in pdb_id_list:
        INPUT_PDB = os.path.join(pdb_dir, f"{pdb_id}.pdb")
        OUTPUT_PDB = os.path.join(save_dir, f"{pdb_id}.pdb")
        MAP_FILE = os.path.join(save_dir, f"{pdb_id}_residue_mapping.json")

        renumber_and_create_map(INPUT_PDB, OUTPUT_PDB, MAP_FILE)