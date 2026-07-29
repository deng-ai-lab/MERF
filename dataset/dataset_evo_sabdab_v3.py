"""SABDAB v3 演化数据集。

v3 的策略 WT、奖励评分 WT 和 FoldX 父结构统一为同一个 RepairPDB 输出，
避免旧版在原始 PDB 与 PDBs_fixed 之间混用坐标和残基索引。
"""

from __future__ import annotations

import os

import torch

from dataset.dataset_evo import EvoBaseDataset
from protein.mutate_scripts_foldx_sabdab_v3 import (
    DEFAULT_FOLDX_BIN,
    ensure_mutant_structures,
    ensure_repaired_structure,
    mutant_pdb_path,
)
from protein.read_pdbs import KnnAgnet, KnnResidue, parse_pdb


AA_KEY = "ACDEFGHIKLMNPQRSTVWY"


class SAbDabEvoDatasetV3(EvoBaseDataset):
    """使用同一 repaired WT 的 SABDAB 演化数据集。"""

    def __init__(
        self,
        pdb_id,
        antibody_chain,
        partner,
        sequence,
        cdr,
        wt_dir,
        fixed_dir,
        mut_dir,
        knn_num,
        knn_agents_num,
        use_plm_embedding=False,
        plm_path=None,
        plm_embedding_path=None,
        device="cpu",
        foldx_workers=8,
        foldx_bin=DEFAULT_FOLDX_BIN,
    ):
        super().__init__(
            pdb_id=pdb_id,
            antibody_chain=antibody_chain,
            partner=partner,
            sequence=sequence,
            cdr=cdr,
            wt_dir=wt_dir,
            fix_dir=fixed_dir,
            mut_dir=mut_dir,
            knn_num=knn_num,
            knn_agents_num=knn_agents_num,
            use_plm_embedding=use_plm_embedding,
            plm_path=plm_path,
            plm_embedding_path=plm_embedding_path,
            device=device,
        )
        self.foldx_workers = foldx_workers
        self.foldx_bin = foldx_bin
        self.raw_wt_pdb_path = os.path.join(self.wt_dir, f"{self.pdb_id}.pdb")
        self.fixed_wt_pdb_path = os.path.join(self.fix_dir, f"{self.pdb_id}.pdb")
        ensure_repaired_structure(
            self.raw_wt_pdb_path,
            self.fixed_wt_pdb_path,
            foldx_bin=self.foldx_bin,
        )

    def _parse_with_embedding(self, pdb_path):
        complex_info = parse_pdb(pdb_path)
        if complex_info is None:
            raise ValueError(f"无法从 PDB 解析复合物: {pdb_path}")
        if self.use_plm_embedding:
            if self.pdb_id not in self.plm_embeddings_dict:
                raise KeyError(f"PLM embedding 中缺少 {self.pdb_id}")
            embedding = self.plm_embeddings_dict[self.pdb_id]
            if embedding.shape[0] != complex_info["aa"].shape[0]:
                raise ValueError(
                    f"{self.pdb_id} 的 PLM embedding 长度 {embedding.shape[0]} 与 "
                    f"repaired PDB 残基数 {complex_info['aa'].shape[0]} 不一致"
                )
            complex_info["plm_embedding"] = embedding
        return complex_info

    @staticmethod
    def _metadata_to_parsed_index_mapping(metadata_sequence, parsed_sequence):
        """用全局序列比对将 CSV 序列位置映射到 PDB 解析链位置。

        SABDAB CSV 与 PDB 偶尔存在单残基替换或缺失；不能要求 CDR 子串逐字
        完全相同。对角线步骤（包括错配）仍保留位置映射，只有 PDB 缺失的
        CSV 残基才映射为 None。
        """
        match_score = 2
        mismatch_score = -1
        gap_score = -2
        metadata_length = len(metadata_sequence)
        parsed_length = len(parsed_sequence)
        scores = [[0] * (parsed_length + 1) for _ in range(metadata_length + 1)]
        moves = [[None] * (parsed_length + 1) for _ in range(metadata_length + 1)]
        for metadata_index in range(1, metadata_length + 1):
            scores[metadata_index][0] = metadata_index * gap_score
            moves[metadata_index][0] = "up"
        for parsed_index in range(1, parsed_length + 1):
            scores[0][parsed_index] = parsed_index * gap_score
            moves[0][parsed_index] = "left"

        for metadata_index in range(1, metadata_length + 1):
            for parsed_index in range(1, parsed_length + 1):
                diagonal = scores[metadata_index - 1][parsed_index - 1] + (
                    match_score
                    if metadata_sequence[metadata_index - 1] == parsed_sequence[parsed_index - 1]
                    else mismatch_score
                )
                up = scores[metadata_index - 1][parsed_index] + gap_score
                left = scores[metadata_index][parsed_index - 1] + gap_score
                if diagonal >= up and diagonal >= left:
                    scores[metadata_index][parsed_index] = diagonal
                    moves[metadata_index][parsed_index] = "diagonal"
                elif up >= left:
                    scores[metadata_index][parsed_index] = up
                    moves[metadata_index][parsed_index] = "up"
                else:
                    scores[metadata_index][parsed_index] = left
                    moves[metadata_index][parsed_index] = "left"

        mapping = [None] * metadata_length
        metadata_index, parsed_index = metadata_length, parsed_length
        while metadata_index > 0 or parsed_index > 0:
            move = moves[metadata_index][parsed_index]
            if move == "diagonal":
                mapping[metadata_index - 1] = parsed_index - 1
                metadata_index -= 1
                parsed_index -= 1
            elif move == "up":
                metadata_index -= 1
            elif move == "left":
                parsed_index -= 1
            else:
                raise RuntimeError("全局序列比对回溯失败")
        return mapping

    def _make_cdr_mask(self, complex_info):
        """按 CSV CDR 区间映射 repaired WT 链，允许已验证的序列错配。"""
        chain_indices = [
            index
            for index, chain_id in enumerate(complex_info["chain_id"])
            if chain_id == self.antibody_chain
        ]
        if not chain_indices:
            raise ValueError(f"{self.pdb_id} 的 repaired PDB 缺少抗体链 {self.antibody_chain}")
        chain_sequence = "".join(
            AA_KEY[int(complex_info["aa"][index].item())] for index in chain_indices
        )
        cdr_start = self.sequence.find(self.cdr)
        if cdr_start < 0:
            raise ValueError(
                f"{self.pdb_id} 的 CDR {self.cdr} 不存在于 CSV 抗体序列中"
            )
        if self.sequence.find(self.cdr, cdr_start + 1) >= 0:
            raise ValueError(
                f"{self.pdb_id} 的 CDR {self.cdr} 在 CSV 抗体序列中不唯一，"
                "不能安全建立 mutation mask"
            )
        mapping = self._metadata_to_parsed_index_mapping(self.sequence, chain_sequence)
        parsed_offsets = mapping[cdr_start : cdr_start + len(self.cdr)]
        if len(parsed_offsets) != len(self.cdr):
            raise ValueError(
                f"{self.pdb_id} 的 CDR {self.cdr} 含有无法映射到 repaired PDB 的残基"
            )
        # PDB 若确实缺失 CSV 中的某个 CDR 残基，只能跳过该不存在的位点；
        # 其余已对齐的实际 PDB 残基仍可安全作为演化位点。
        selected_indices = [chain_indices[offset] for offset in parsed_offsets if offset is not None]
        if not selected_indices:
            raise ValueError(f"{self.pdb_id} 的 CDR {self.cdr} 没有可映射的 repaired PDB 位点")
        mask = torch.zeros_like(complex_info["aa"], dtype=torch.bool)
        mask[torch.tensor(selected_indices, dtype=torch.long)] = True
        return mask

    @staticmethod
    def _validate_wt_mut_layout(complex_wt_info, complex_mut_info, mutate_info):
        """FoldX 不应改变复合物的残基布局；否则按位置比较 aa 会失真。"""
        for key in ("chain_id", "icode"):
            if complex_wt_info[key] != complex_mut_info[key]:
                raise ValueError(f"{mutate_info} 的 WT/mut {key} 布局不一致")
        if not torch.equal(complex_wt_info["resseq"], complex_mut_info["resseq"]):
            raise ValueError(f"{mutate_info} 的 WT/mut resseq 布局不一致")
        if complex_wt_info["aa"].shape != complex_mut_info["aa"].shape:
            raise ValueError(f"{mutate_info} 的 WT/mut 残基数量不一致")

    def __getitem__(self, index):
        if index != 0:
            raise IndexError("SAbDabEvoDatasetV3 只有一个 WT 样本")
        complex_wt_info = self._parse_with_embedding(self.fixed_wt_pdb_path)
        mutation_mask = self._make_cdr_mask(complex_wt_info)
        batch = KnnResidue(num_neighbors=self.knn_num)(
            {"wt": complex_wt_info, "mutation_mask": mutation_mask}
        )
        batch["expand_data_info"] = {
            "pdb_id": self.pdb_id,
            "chain": self.partner,
            "antibody_chain": self.antibody_chain,
            "sequence": self.sequence,
            "cdr": self.cdr,
            "fixed_wt_pdb": self.fixed_wt_pdb_path,
        }
        return batch

    def getitem_by_mutate_info(self, mutate_info):
        mut_pdb_path = mutant_pdb_path(self.mut_dir, self.pdb_id, mutate_info)
        if not os.path.exists(mut_pdb_path):
            raise FileNotFoundError(f"v3 FoldX 候选结构不存在: {mut_pdb_path}")

        complex_wt_info = self._parse_with_embedding(self.fixed_wt_pdb_path)
        complex_mut_info = self._parse_with_embedding(mut_pdb_path)
        self._validate_wt_mut_layout(complex_wt_info, complex_mut_info, mutate_info)
        actual_mutation_mask = complex_wt_info["aa"] != complex_mut_info["aa"]
        if not actual_mutation_mask.any():
            raise ValueError(f"{mutate_info} 没有产生任何实际氨基酸替换")

        agent_mask = KnnAgnet(num_neighbors=self.knn_agents_num)(
            {
                "wt": complex_wt_info,
                "mut": complex_mut_info,
                "mutation_mask": actual_mutation_mask,
            }
        )
        for complex_info in (complex_wt_info, complex_mut_info):
            complex_info["agent_mask"] = agent_mask
            complex_info["PDB_id"] = self.pdb_id
            complex_info["mutate_info"] = mutate_info

        batch = KnnResidue(num_neighbors=self.knn_num)(
            {
                "wt": complex_wt_info,
                "mut": complex_mut_info,
                "mutation_mask": actual_mutation_mask,
            }
        )
        batch["mutation_mask"] = batch["wt"]["aa"] != batch["mut"]["aa"]
        batch["ddG"] = 0
        return batch

    def mutate_pdb(self, mutate_info_list):
        """只为当前候选池构建缺失 PDB，且始终从 v3 repaired WT 出发。"""
        return ensure_mutant_structures(
            pdb_id=self.pdb_id,
            mutate_infos=mutate_info_list,
            wt_pdb_path=self.raw_wt_pdb_path,
            fixed_pdb_path=self.fixed_wt_pdb_path,
            mutated_dir=self.mut_dir,
            workers=self.foldx_workers,
            foldx_bin=self.foldx_bin,
        )
