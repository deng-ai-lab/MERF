"""On-demand-structure CR evolution dataset for repaired ESMFold2 start structures.

The action at every CR site remains an *absolute* landscape state:
0 means the immature residue and 1 means the mature residue.  The user-selected
start only changes the WT structure.  When an action is nominated, this module
converts the absolute target into the relative FoldX mutation string needed to
edit that selected start structure.
"""

import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from dataset.dataset_cr_evo_v2 import CREvoDataset, MUTATION_RE
from protein.mutate_scripts_foldx_flexible_v4 import (
    DEFAULT_FOLDX_BIN,
    ensure_repaired_structure,
    ensure_target_structures,
    target_pdb_path,
)
from protein.read_pdbs import KnnAgnet, KnnResidue, parse_pdb, parse_pdb_chain2sequence


def canonical_reverse_mutation_key(mutation_key):
    """Canonicalize a mature-to-immature mutation set without a dataset instance."""
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


class FlexibleCREvoDatasetV4(CREvoDataset):
    """Flexible CR dataset that creates FoldX target structures only after policy nomination."""

    def __init__(
        self,
        pdb_id,
        antibody_chain,
        partner,
        sequence,
        cdr,
        cr_dir,
        start_dir,
        start_reverse_mutations,
        knn_num,
        knn_agents_num,
        use_plm_embedding=False,
        plm_path=None,
        plm_embedding_path=None,
        device="cpu",
        foldx_workers=8,
        foldx_bin=DEFAULT_FOLDX_BIN,
        repair_start=True,
    ):
        # 不调用 CREvoDataset.__init__：父类会把最不成熟个体固定成 WT，
        # 而 v4 的 WT 必须是用户选择、ESMFold2 重新预测的起点。
        torch.utils.data.Dataset.__init__(self)
        self.pdb_id = pdb_id
        self.antibody_chain = antibody_chain
        self.partner = partner
        self.sequence = sequence
        self.cdr = cdr
        self.cr_dir = cr_dir
        self.start_dir = start_dir
        self.knn_num = knn_num
        self.knn_agents_num = knn_agents_num
        self.use_plm_embedding = use_plm_embedding
        self.plm_path = plm_path
        self.plm_embedding_path = plm_embedding_path
        self.device = device
        self.foldx_workers = foldx_workers
        self.foldx_bin = foldx_bin
        if not repair_start:
            raise ValueError(
                "FlexibleCREvoDatasetV4 requires FoldX RepairPDB before mutation; "
                "raw ESMFold2 starts are not supported."
            )
        self.repair_start = True

        self.score_csv_path = os.path.join(self.cr_dir, f"{self.pdb_id}.csv")
        self.ref_csv_path = os.path.join(self.cr_dir, f"{self.pdb_id}_ref.csv")
        self.start_folder_id = os.path.basename(os.path.abspath(self.start_dir))
        self.raw_start_pdb_path = os.path.join(
            self.start_dir, "PDBs", f"{self.start_folder_id}.pdb"
        )
        self.fixed_start_pdb_path = os.path.join(
            self.start_dir, "PDBs_fixed", f"{self.start_folder_id}.pdb"
        )
        self.mutated_dir = os.path.join(self.start_dir, "PDBs_mutated")
        if not os.path.exists(self.raw_start_pdb_path):
            raise FileNotFoundError(f"ESMFold2 start PDB does not exist: {self.raw_start_pdb_path}")

        self.reverse_mutations = self._load_reverse_mutations()
        self.reverse_mutation_set = set(self.reverse_mutations)
        self.site_infos = [self._parse_reverse_mutation(mutation) for mutation in self.reverse_mutations]
        self.position_to_site = {info["position"]: info for info in self.site_infos}
        self.start_reverse_key = canonical_reverse_mutation_key(start_reverse_mutations)
        self.start_reverse_mutation_set = set(self._split_key(self.start_reverse_key))
        unknown = self.start_reverse_mutation_set - self.reverse_mutation_set
        if unknown:
            raise ValueError(f"Start mutations are absent from {self.score_csv_path}: {sorted(unknown)}")
        self.full_reverse_key = self._canonical_reverse_key(self.reverse_mutations)

        # 先对 ESMFold2 起点执行一次 RepairPDB；修复后结构同时作为 WT 和
        # 所有 BuildModel 候选的共同父结构，避免混入未经修复的坐标。
        ensure_repaired_structure(
            self.raw_start_pdb_path,
            self.fixed_start_pdb_path,
            foldx_bin=self.foldx_bin,
        )
        self.wt_pdb_path = self.fixed_start_pdb_path
        self.immature_pdb_path = self.wt_pdb_path

        self.score_col = None
        self.score_by_reverse_key = {}
        self.rank_by_reverse_key = {}
        self._load_scores_and_ranks()
        self.all_action_matrix = self._build_all_action_matrix()

        self.plm_embedding = None
        self.plm_embedding_key = self._variant_id(self.start_reverse_key)
        if self.use_plm_embedding:
            if self.plm_embedding_path is None:
                raise ValueError("plm_embedding_path must be provided when use_plm_embedding=True")
            self._load_plm_embedding()

    def __len__(self):
        return 1

    def __getitem__(self, index):
        complex_wt_info = self._parse_with_plm(self.wt_pdb_path)
        mutation_mask = self._make_site_mask(complex_wt_info)
        batch = KnnResidue(num_neighbors=self.knn_num)({
            "wt": complex_wt_info,
            "mutation_mask": mutation_mask,
        })
        batch["expand_data_info"] = {
            "pdb_id": self.pdb_id,
            "chain": self.partner,
            "antibody_chain": self.antibody_chain,
            "sequence": self.sequence,
            "cdr": self.cdr,
            "start_reverse_key": self.start_reverse_key,
        }
        return batch

    def _build_all_action_matrix(self):
        """Enumerate all binary landscape states; no PDB is generated by this step."""
        num_sites = len(self.reverse_mutations)
        state_ids = torch.arange(1 << num_sites, dtype=torch.long)
        site_offsets = torch.arange(num_sites, dtype=torch.long)
        return ((state_ids.unsqueeze(1) >> site_offsets.unsqueeze(0)) & 1).long()

    def action_tensor_to_reverse_key(self, action_tensor):
        actions = action_tensor.detach().cpu().reshape(-1).tolist()
        if len(actions) != len(self.reverse_mutations):
            raise ValueError(f"Expected {len(self.reverse_mutations)} actions, got {len(actions)}")
        reverse_mutations = [
            mutation for mutation, action in zip(self.reverse_mutations, actions) if int(action) == 0
        ]
        return self._canonical_reverse_key(reverse_mutations)

    def reverse_key_to_action_tensor(self, reverse_key):
        reverse_set = set(self._split_key(canonical_reverse_mutation_key(reverse_key)))
        return torch.tensor(
            [0 if mutation in reverse_set else 1 for mutation in self.reverse_mutations],
            dtype=torch.long,
        )

    def target_pdb_path(self, target_reverse_key):
        target_reverse_key = canonical_reverse_mutation_key(target_reverse_key)
        if target_reverse_key == self.start_reverse_key:
            return self.wt_pdb_path
        return target_pdb_path(self.mutated_dir, target_reverse_key)

    def foldx_mutations_from_target(self, target_reverse_key):
        """Map absolute target residues back to the relative edits required from this start.

        例如起点已经带有 ``RA30S``，目标状态选择成熟残基时，FoldX 命令是
        ``SA30R``；目标仍选择非成熟残基时，该位点不需要编辑。
        """
        target_set = set(self._split_key(canonical_reverse_mutation_key(target_reverse_key)))
        relative = []
        for mutation in self.reverse_mutations:
            start_is_reverse = mutation in self.start_reverse_mutation_set
            target_is_reverse = mutation in target_set
            if start_is_reverse == target_is_reverse:
                continue
            info = self._parse_reverse_mutation(mutation)
            if start_is_reverse and not target_is_reverse:
                relative.append(
                    f"{info['immature_aa']}{info['chain']}{info['position']}{info['mature_aa']}"
                )
            else:
                relative.append(
                    f"{info['mature_aa']}{info['chain']}{info['position']}{info['immature_aa']}"
                )
        return ",".join(sorted(relative, key=self._mutation_sort_key))

    def ensure_action_structures(self, action_tensors):
        """Batch-generate only the targets nominated by the current policy rollout."""
        if isinstance(action_tensors, torch.Tensor):
            actions = action_tensors.detach().cpu()
            if actions.dim() == 1:
                actions = actions.unsqueeze(0)
        else:
            actions = torch.as_tensor(action_tensors, dtype=torch.long)
            if actions.dim() == 1:
                actions = actions.unsqueeze(0)
        target_to_mutation = {}
        for action in actions:
            target_key = self.action_tensor_to_reverse_key(action)  # actions是直接表示某个site选用的是成熟还是不成熟，因此这里得到的是突变体所应该具有的逆向突变
            if target_key != self.start_reverse_key:
                target_to_mutation[target_key] = self.foldx_mutations_from_target(target_key)  # 这个函数计算突变目标相对于突变起点的差值
        paths = {self.start_reverse_key: self.wt_pdb_path}
        if target_to_mutation:
            paths.update(ensure_target_structures(
                self.raw_start_pdb_path,
                self.fixed_start_pdb_path,
                self.mutated_dir,
                target_to_mutation,
                workers=self.foldx_workers,
                foldx_bin=self.foldx_bin,
            ))
        return paths

    def getitem_by_action_tensor(self, action_tensor):
        target_reverse_key = self.action_tensor_to_reverse_key(action_tensor)
        mut_pdb_path = self.target_pdb_path(target_reverse_key)
        if not os.path.exists(mut_pdb_path):
            self.ensure_action_structures(action_tensor)
        if not os.path.exists(mut_pdb_path):
            raise FileNotFoundError(f"FoldX target PDB was not created: {mut_pdb_path}")

        complex_wt_info = self._parse_with_plm(self.wt_pdb_path)
        complex_mut_info = self._parse_with_plm(mut_pdb_path)
        actual_mutation_mask = complex_wt_info["aa"] != complex_mut_info["aa"]
        site_mask = self._make_site_mask(complex_wt_info)
        agent_center_mask = actual_mutation_mask if actual_mutation_mask.any() else site_mask
        agent_mask = KnnAgnet(num_neighbors=self.knn_agents_num)({
            "wt": complex_wt_info,
            "mut": complex_mut_info,
            "mutation_mask": agent_center_mask,
        })

        relative_mutations = self.foldx_mutations_from_target(target_reverse_key)
        complex_wt_info["agent_mask"] = agent_mask
        complex_wt_info["PDB_id"] = self.pdb_id
        complex_wt_info["mutate_info"] = self.start_reverse_key
        complex_wt_info["start_mutate_info"] = self.start_reverse_key
        complex_mut_info["agent_mask"] = agent_mask
        complex_mut_info["PDB_id"] = self.pdb_id
        complex_mut_info["mutate_info"] = target_reverse_key
        complex_mut_info["start_mutate_info"] = self.start_reverse_key
        complex_mut_info["foldx_mutate_info"] = relative_mutations

        transform_mask = actual_mutation_mask if actual_mutation_mask.any() else site_mask
        batch = KnnResidue(num_neighbors=self.knn_num)({
            "wt": complex_wt_info,
            "mut": complex_mut_info,
            "mutation_mask": transform_mask,
        })
        batch["mutation_mask"] = batch["wt"]["aa"] != batch["mut"]["aa"]
        batch["ddG"] = 0
        return batch

    def _load_scores_and_ranks(self):
        score_df = pd.read_csv(self.score_csv_path)
        score_columns = [column for column in score_df.columns if column.endswith("_score")]
        if len(score_columns) != 1:
            raise ValueError(f"Expected one score column in {self.score_csv_path}, found {score_columns}")
        self.score_col = score_columns[0]
        for _, row in score_df.iterrows():
            reverse_key = self._canonical_reverse_key(self._split_key(row["mutant"]))
            if pd.notna(row[self.score_col]):
                self.score_by_reverse_key[reverse_key] = float(row[self.score_col])
        ref_df = pd.read_csv(self.ref_csv_path)
        if len(ref_df) > 0 and pd.notna(ref_df[self.score_col].iloc[0]):
            self.score_by_reverse_key[""] = float(ref_df[self.score_col].iloc[0])
        ranked = sorted(self.score_by_reverse_key.items(), key=lambda item: item[1], reverse=True)
        self.rank_by_reverse_key = {
            reverse_key: rank for rank, (reverse_key, _) in enumerate(ranked, start=1)
        }

    def evaluate_reverse_key(self, reverse_key):
        reverse_key = canonical_reverse_mutation_key(reverse_key)
        return {
            "true_score": self.score_by_reverse_key.get(reverse_key),
            "rank": self.rank_by_reverse_key.get(reverse_key),
            "num_ranked": len(self.rank_by_reverse_key),
        }

    def _variant_id(self, reverse_key):
        return self.pdb_id if reverse_key == "" else f"{self.pdb_id}_{reverse_key}"

    def _load_plm_embedding(self):
        if not os.path.exists(self.plm_embedding_path):
            self._process_plm_embedding()
        embeddings = torch.load(self.plm_embedding_path, map_location="cpu")
        if self.plm_embedding_key not in embeddings:
            raise KeyError(f"PLM embedding {self.plm_embedding_key} missing from {self.plm_embedding_path}")
        self.plm_embedding = embeddings[self.plm_embedding_key]

    def _process_plm_embedding(self):
        if self.plm_path is None:
            raise ValueError("plm_path must be provided to create missing PLM embeddings")
        tokenizer = AutoTokenizer.from_pretrained(self.plm_path)
        model = AutoModel.from_pretrained(self.plm_path).to(self.device).eval()
        chain2sequence = parse_pdb_chain2sequence(self.wt_pdb_path)
        complex_info = parse_pdb(self.wt_pdb_path)
        chain_ids = complex_info["chain_id"]
        if list(chain2sequence.keys())[0] != chain_ids[0] or list(chain2sequence.keys())[-1] != chain_ids[-1]:
            raise ValueError("Chain order mismatch while creating v4 PLM embedding")
        embeddings = []
        for _, chain_sequence in tqdm(chain2sequence.items(), desc=f"PLM embedding {self.start_folder_id}"):
            inputs = tokenizer(chain_sequence, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                output = model(**inputs)
                embeddings.append(output.last_hidden_state.squeeze(0)[1:len(chain_sequence) + 1, :].cpu())
        embedding = torch.cat(embeddings, dim=0)
        if embedding.shape[0] != len(chain_ids):
            raise ValueError("PLM embedding length mismatch")
        os.makedirs(os.path.dirname(self.plm_embedding_path), exist_ok=True)
        torch.save({self.plm_embedding_key: embedding}, self.plm_embedding_path)
