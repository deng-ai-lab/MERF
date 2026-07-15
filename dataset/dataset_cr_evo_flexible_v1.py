"""CR evolution dataset whose wild type can be any reverse-mutated variant."""

import csv
import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from dataset.dataset_cr_evo_v2 import CREvoDataset, MUTATION_RE
from protein.read_pdbs import KnnAgnet, KnnResidue, parse_pdb, parse_pdb_chain2sequence


def canonical_reverse_mutation_key(mutation_key):
    """Canonicalize mature-to-start mutation strings without requiring a dataset instance."""
    if mutation_key is None:
        mutations = []
    else:
        mutations = [item.strip() for item in str(mutation_key).split(",") if item.strip()]

    def sort_key(mutation):
        match = MUTATION_RE.match(mutation)
        if match is None:
            raise ValueError(f"Invalid CR mutation string: {mutation}")
        mature_aa, chain, position, immature_aa = match.groups()
        return int(position), chain, mature_aa, immature_aa

    return ",".join(sorted(mutations, key=sort_key))


class FlexibleCREvoDataset(CREvoDataset):
    """Reuse the v2 evolution interfaces, with an arbitrary CR variant as WT."""

    def __init__(
        self,
        pdb_id,
        antibody_chain,
        partner,
        sequence,
        cdr,
        cr_dir,
        pdb_dir,
        fixed_dir,
        mutated_dir,
        start_reverse_mutations,
        knn_num,
        knn_agents_num,
        use_plm_embedding=False,
        plm_path=None,
        plm_embedding_path=None,
        device="cpu",
    ):
        # 这里不调用父类 __init__：父类会把“最不成熟个体”固定为 WT。
        # 其余数据接口、动作编码和结构构图方法均直接复用父类实现。
        torch.utils.data.Dataset.__init__(self)
        self.pdb_id = pdb_id
        self.antibody_chain = antibody_chain
        self.partner = partner
        self.sequence = sequence
        self.cdr = cdr
        self.cr_dir = cr_dir
        self.pdb_dir = pdb_dir
        self.fixed_dir = fixed_dir
        self.mutated_dir = mutated_dir
        self.knn_num = knn_num
        self.knn_agents_num = knn_agents_num
        self.use_plm_embedding = use_plm_embedding
        self.plm_path = plm_path
        self.plm_embedding_path = plm_embedding_path
        self.device = device

        self.score_csv_path = os.path.join(self.cr_dir, f"{self.pdb_id}.csv")
        self.ref_csv_path = os.path.join(self.cr_dir, f"{self.pdb_id}_ref.csv")
        if self.plm_embedding_path is None:
            raise ValueError("plm_embedding_path must be provided for flexible CR evolution")

        all_reverse_mutations = self._load_reverse_mutations()
        self.start_reverse_key = canonical_reverse_mutation_key(start_reverse_mutations)
        self.start_reverse_mutation_set = set(self._split_key(self.start_reverse_key))
        unknown_mutations = self.start_reverse_mutation_set - set(all_reverse_mutations)
        if unknown_mutations:
            raise ValueError(
                f"Start mutations are not present in {self.score_csv_path}: "
                f"{sorted(unknown_mutations)}"
            )

        # CSV 中出现过的全部位点定义固定动作空间；0=逆向/非成熟，1=成熟。
        # 起点只决定 WT 结构，不能缩小策略能够搜索的位点集合。
        self.reverse_mutations = all_reverse_mutations
        self.reverse_mutation_set = set(self.reverse_mutations)
        self.site_infos = [self._parse_reverse_mutation(m) for m in self.reverse_mutations]
        self.position_to_site = {info["position"]: info for info in self.site_infos}
        self.mature_pdb_path = os.path.join(self.fixed_dir, f"{self.pdb_id}.pdb")
        self.full_reverse_key = self._canonical_reverse_key(self.reverse_mutations)
        self.immature_pdb_path = self._reverse_key_to_pdb_path(self.start_reverse_key)
        if not os.path.exists(self.immature_pdb_path):
            raise FileNotFoundError(
                f"Start PDB does not exist for {self.pdb_id}: {self.immature_pdb_path}"
            )

        self.score_col = None
        self.score_by_reverse_key = {}
        self.rank_by_reverse_key = {}
        self.candidate_reverse_keys = []
        self.candidate_action_matrix = None
        self._load_scores_and_candidates()
        if not self.candidate_reverse_keys:
            raise ValueError(f"No PDB-backed landscape candidates found for {self.pdb_id}")

        self.plm_embedding = None
        self.plm_embedding_key = self._variant_id(self.start_reverse_key)
        if self.use_plm_embedding:
            self._load_plm_embedding()

    def _load_scores_and_candidates(self):
        """Use all CSV/PDB landscape variants as targets while retaining global ranks."""
        score_df = pd.read_csv(self.score_csv_path)
        self.score_col = [col for col in score_df.columns if col.endswith("_score")][0]

        all_candidate_keys = set()
        for _, row in score_df.iterrows():
            reverse_key = self._canonical_reverse_key(self._split_key(row["mutant"]))
            all_candidate_keys.add(reverse_key)
            if pd.notna(row[self.score_col]):
                self.score_by_reverse_key[reverse_key] = float(row[self.score_col])

        if os.path.exists(self.ref_csv_path):
            ref_df = pd.read_csv(self.ref_csv_path)
            if len(ref_df) > 0 and pd.notna(ref_df[self.score_col].iloc[0]):
                self.score_by_reverse_key[""] = float(ref_df[self.score_col].iloc[0])
                all_candidate_keys.add("")

        # 评估排名沿用整个实验 landscape，便于比较不同起点的进化结果。
        ranked = sorted(
            self.score_by_reverse_key.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        self.rank_by_reverse_key = {
            reverse_key: rank
            for rank, (reverse_key, _) in enumerate(ranked, start=1)
        }

        all_candidate_keys.add(self.start_reverse_key)
        existing_candidate_keys = [
            reverse_key
            for reverse_key in all_candidate_keys
            if os.path.exists(self._reverse_key_to_pdb_path(reverse_key))
        ]
        self.candidate_reverse_keys = sorted(set(existing_candidate_keys), key=self._combo_sort_key)

        action_rows = []
        for reverse_key in self.candidate_reverse_keys:
            reverse_set = set(self._split_key(reverse_key))
            # 动作编码描述目标的绝对状态，而非相对起点的变化：0=逆向，1=成熟。
            action_rows.append([
                0 if mutation in reverse_set else 1
                for mutation in self.reverse_mutations
            ])
        self.candidate_action_matrix = torch.tensor(action_rows, dtype=torch.long)

    def getitem_by_forward_mutations(self, forward_mutations):
        """Construct start-WT to arbitrary landscape-target transition from an action vector."""
        forward_set = set(forward_mutations)
        reverse_key = self.forward_mutations_to_reverse_key(forward_set)
        mut_pdb_path = self._reverse_key_to_pdb_path(reverse_key)
        if not os.path.exists(mut_pdb_path):
            raise FileNotFoundError(f"Target PDB does not exist: {mut_pdb_path}")

        # WT 始终是用户传入的起点；mut 则是完整动作空间指定的目标 landscape 结构。
        complex_wt_info = self._parse_with_plm(self.immature_pdb_path)
        complex_mut_info = self._parse_with_plm(mut_pdb_path)
        actual_mutation_mask = complex_wt_info["aa"] != complex_mut_info["aa"]
        site_mask = self._make_site_mask(complex_wt_info)
        agent_center_mask = actual_mutation_mask if actual_mutation_mask.any() else site_mask
        agent_mask = KnnAgnet(num_neighbors=self.knn_agents_num)({
            "wt": complex_wt_info,
            "mut": complex_mut_info,
            "mutation_mask": agent_center_mask,
        })

        complex_wt_info["agent_mask"] = agent_mask
        complex_wt_info["PDB_id"] = self.pdb_id
        complex_wt_info["mutate_info"] = self.start_reverse_key
        complex_wt_info["start_mutate_info"] = self.start_reverse_key
        complex_mut_info["agent_mask"] = agent_mask
        complex_mut_info["PDB_id"] = self.pdb_id
        complex_mut_info["mutate_info"] = reverse_key
        complex_mut_info["start_mutate_info"] = self.start_reverse_key

        transform_mask = actual_mutation_mask if actual_mutation_mask.any() else site_mask
        batch = KnnResidue(num_neighbors=self.knn_num)({
            "wt": complex_wt_info,
            "mut": complex_mut_info,
            "mutation_mask": transform_mask,
        })
        batch["mutation_mask"] = batch["wt"]["aa"] != batch["mut"]["aa"]
        batch["ddG"] = 0
        return batch

    def _variant_id(self, reverse_key):
        if reverse_key == "":
            return self.pdb_id
        return f"{self.pdb_id}_{reverse_key}"

    def _load_plm_embedding(self):
        if not os.path.exists(self.plm_embedding_path):
            self._process_plm_embedding()
        embeddings = torch.load(self.plm_embedding_path, map_location="cpu")
        if self.plm_embedding_key not in embeddings:
            raise KeyError(
                f"PLM embedding key {self.plm_embedding_key} not found in {self.plm_embedding_path}"
            )
        self.plm_embedding = embeddings[self.plm_embedding_key]

    def _process_plm_embedding(self):
        """Generate the embedding for the selected WT start structure, not for immature WT."""
        if self.plm_path is None:
            raise ValueError("plm_path must be provided to build missing PLM embeddings")

        tokenizer = AutoTokenizer.from_pretrained(self.plm_path)
        model = AutoModel.from_pretrained(self.plm_path)
        model.eval()
        model = model.to(self.device)

        chain2sequence = parse_pdb_chain2sequence(self.immature_pdb_path)
        complex_info = parse_pdb(self.immature_pdb_path)
        chain_id_list = complex_info["chain_id"]
        if (
            list(chain2sequence.keys())[0] != chain_id_list[0]
            or list(chain2sequence.keys())[-1] != chain_id_list[-1]
        ):
            raise ValueError("chain order mismatch while building flexible CR PLM embedding")

        embedding_list = []
        for _, chain_sequence in tqdm(
            chain2sequence.items(),
            desc=f"Processing PLM embedding for {self.pdb_id}:{self.start_reverse_key or 'mature'}",
        ):
            inputs = tokenizer(chain_sequence, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state.squeeze(0)[1:len(chain_sequence) + 1, :].cpu()
            embedding_list.append(embedding)

        embedding = torch.cat(embedding_list, dim=0)
        if embedding.shape[0] != len(chain_id_list):
            raise ValueError("PLM embedding length mismatch")
        os.makedirs(os.path.dirname(self.plm_embedding_path), exist_ok=True)
        torch.save({self.plm_embedding_key: embedding}, self.plm_embedding_path)
