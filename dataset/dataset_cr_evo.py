import csv
import os
import re

import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from protein.read_pdbs import parse_pdb, KnnResidue, KnnAgnet, parse_pdb_chain2sequence


AA_KEY = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AA_KEY)}
MUTATION_RE = re.compile(r"^([A-Z])([A-Za-z])(-?\d+)([A-Z])$")


class CREvoDataset(Dataset):
    def __init__(
            self, pdb_id, antibody_chain, partner, sequence, cdr,
            cr_dir, pdb_dir, fixed_dir, mutated_dir,
            knn_num, knn_agents_num,
            use_plm_embedding=False, plm_path=None, plm_embedding_path=None, device='cpu'
        ):
        super(CREvoDataset, self).__init__()

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
        assert self.plm_embedding_path is not None, "plm_embedding_path must be provided for caching PLM embeddings"

        self.reverse_mutations = self._load_reverse_mutations()  # reverse mutation指从成熟抗体（mut）到非成熟抗体（wt）方向的突变
        self.reverse_mutation_set = set(self.reverse_mutations)
        self.site_infos = [self._parse_reverse_mutation(m) for m in self.reverse_mutations]
        self.position_to_site = {info['position']: info for info in self.site_infos}
        self.full_reverse_key = self._canonical_reverse_key(self.reverse_mutations)
        self.immature_pdb_path = self._reverse_key_to_pdb_path(self.full_reverse_key)  # 从成熟抗体到非成熟抗体的完整突变组合对应的PDB路径
        self.mature_pdb_path = os.path.join(self.fixed_dir, f"{self.pdb_id}.pdb")

        self.score_col = None
        self.score_by_reverse_key = {}
        self.rank_by_reverse_key = {}
        self.candidate_reverse_keys = []
        self.candidate_action_matrix = None
        self._load_scores_and_candidates()

        self.plm_embedding = None
        if self.use_plm_embedding:
            self._load_plm_embedding()

    def __len__(self):
        return 1

    def __getitem__(self, index):
        complex_wt_info = self._parse_with_plm(self.immature_pdb_path)
        mutation_mask = self._make_site_mask(complex_wt_info)  # mutation_mask是一个布尔张量，长度等于复合物中残基的数量，True表示该残基是一个可变位点（即在reverse_mutations中），False表示该残基不是一个可变位点
        batch = KnnResidue(num_neighbors=self.knn_num)({
            'wt': complex_wt_info,
            'mutation_mask': mutation_mask,
        })
        batch['expand_data_info'] = {
            'pdb_id': self.pdb_id,
            'chain': self.partner,
            'antibody_chain': self.antibody_chain,
            'sequence': self.sequence,
            'cdr': self.cdr,
        }
        return batch

    def getitem_by_action_tensor(self, action_tensor):
        action_values = [int(v) for v in action_tensor.detach().cpu().tolist()]
        forward_mutations = [
            mutation for mutation, action in zip(self.reverse_mutations, action_values)
            if action == 1
        ]  # action_values中存储1代表选这个位点的正向突变（forward mutation），0代表选这个位点的反向突变（reverse mutation），因此forward_mutations是一个列表，包含了根据action_tensor选择的正向突变
        return self.getitem_by_forward_mutations(forward_mutations)

    def getitem_by_forward_mutations(self, forward_mutations):
        forward_set = set(forward_mutations)
        reverse_key = self.forward_mutations_to_reverse_key(forward_set)
        mut_pdb_path = self._reverse_key_to_pdb_path(reverse_key)

        complex_wt_info = self._parse_with_plm(self.immature_pdb_path)
        complex_mut_info = self._parse_with_plm(mut_pdb_path)

        actual_mutation_mask = (complex_wt_info['aa'] != complex_mut_info['aa'])
        site_mask = self._make_site_mask(complex_wt_info)
        agent_center_mask = actual_mutation_mask if actual_mutation_mask.any() else site_mask
        agent_mask = KnnAgnet(num_neighbors=self.knn_agents_num)({
            'wt': complex_wt_info,
            'mut': complex_mut_info,
            'mutation_mask': agent_center_mask,
        })

        complex_wt_info['agent_mask'] = agent_mask
        complex_wt_info['PDB_id'] = self.pdb_id
        complex_wt_info['mutate_info'] = self._canonical_reverse_key(forward_set)
        complex_mut_info['agent_mask'] = agent_mask
        complex_mut_info['PDB_id'] = self.pdb_id
        complex_mut_info['mutate_info'] = reverse_key

        transform_mask = actual_mutation_mask if actual_mutation_mask.any() else site_mask
        batch = KnnResidue(num_neighbors=self.knn_num)({
            'wt': complex_wt_info,
            'mut': complex_mut_info,
            'mutation_mask': transform_mask,
        })
        batch['mutation_mask'] = (batch['wt']['aa'] != batch['mut']['aa'])
        batch['ddG'] = 0
        return batch

    def get_site_action_tensors(self, batch, device):
        site_indices = torch.nonzero(batch['mutation_mask'][0], as_tuple=True)[0]
        positions = batch['wt']['resseq'][0][site_indices].detach().cpu().tolist()

        stay_actions = []
        forward_actions = []
        reverse_mutations = []
        for position in positions:
            info = self.position_to_site[int(position)]
            stay_actions.append(AA_TO_IDX[info['immature_aa']])
            forward_actions.append(AA_TO_IDX[info['mature_aa']])
            reverse_mutations.append(info['reverse_mutation'])

        if reverse_mutations != self.reverse_mutations:
            raise ValueError(
                f"Site order mismatch for {self.pdb_id}: {reverse_mutations} != {self.reverse_mutations}"
            )

        return (
            site_indices,
            torch.tensor(stay_actions, dtype=torch.long, device=device),
            torch.tensor(forward_actions, dtype=torch.long, device=device),
        )

    def choose_candidate_from_log_probs(self, binary_log_probs):
        if self.candidate_action_matrix is None:
            raise ValueError(f"No CR candidates loaded for {self.pdb_id}")
        # 对于每一个候选突变组合（reverse_key），计算模型在该组合对应的动作（candidate_actions）上的log概率，然后对这些log概率取平均，得到每个候选突变组合的平均log概率，最后选择平均log概率最高的候选突变组合作为模型的选择。
        candidate_actions = self.candidate_action_matrix.to(binary_log_probs.device)
        expanded_log_probs = binary_log_probs.unsqueeze(0).expand(candidate_actions.size(0), -1, -1)
        action_log_probs = expanded_log_probs.gather(2, candidate_actions.unsqueeze(-1)).squeeze(-1)
        candidate_scores = action_log_probs.mean(dim=1)
        best_idx = int(torch.argmax(candidate_scores).item())
        selected_actions = candidate_actions[best_idx].detach().clone()
        reverse_key = self.candidate_reverse_keys[best_idx]
        return selected_actions, reverse_key, candidate_scores[best_idx]

    def evaluate_reverse_key(self, reverse_key):
        return {
            'true_score': self.score_by_reverse_key.get(reverse_key),
            'rank': self.rank_by_reverse_key.get(reverse_key),
            'num_ranked': len(self.rank_by_reverse_key),
        }

    def reverse_key_to_forward_info(self, reverse_key):
        reverse_set = set(self._split_key(reverse_key))
        forward_mutations = [
            mutation for mutation in self.reverse_mutations
            if mutation not in reverse_set
        ]
        forward_info = [
            self._reverse_to_forward_mutation(mutation)
            for mutation in forward_mutations
        ]
        return self._canonical_reverse_key(forward_mutations), self._canonical_reverse_key(forward_info)

    def forward_mutations_to_reverse_key(self, forward_mutations):
        reverse_mutations = [
            mutation for mutation in self.reverse_mutations
            if mutation not in forward_mutations
        ]
        return self._canonical_reverse_key(reverse_mutations)

    def _load_reverse_mutations(self):
        mutations = set()
        with open(self.score_csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for mutation in self._split_key(row['mutant']):
                    mutations.add(mutation)
        return sorted(mutations, key=self._mutation_sort_key)

    def _load_scores_and_candidates(self):
        score_df = pd.read_csv(self.score_csv_path)
        self.score_col = [col for col in score_df.columns if col.endswith('_score')][0]

        candidate_keys = set()  # 所有候选突变组合的reverse_key集合，reverse_key是按照一定顺序排列的突变字符串，例如"PA28T,RA30S,KA59N,PA62Q,VA79A"
        for _, row in score_df.iterrows():
            reverse_key = self._canonical_reverse_key(self._split_key(row['mutant']))
            candidate_keys.add(reverse_key)
            if pd.notna(row[self.score_col]):
                self.score_by_reverse_key[reverse_key] = float(row[self.score_col])  # score_by_reverse_key是一个字典，key是reverse_key，value是测得的kd值

        if os.path.exists(self.ref_csv_path):
            ref_df = pd.read_csv(self.ref_csv_path)
            if len(ref_df) > 0 and self.score_col in ref_df.columns and pd.notna(ref_df[self.score_col].iloc[0]):
                self.score_by_reverse_key[''] = float(ref_df[self.score_col].iloc[0])
                candidate_keys.add('')  # 没有逆向突变代表着完全按照成熟抗体的序列

        candidate_keys.add(self.full_reverse_key)
        candidate_keys.add('')

        existing_candidate_keys = []
        for reverse_key in candidate_keys:
            if os.path.exists(self._reverse_key_to_pdb_path(reverse_key)):
                existing_candidate_keys.append(reverse_key)

        self.candidate_reverse_keys = sorted(existing_candidate_keys, key=self._combo_sort_key)
        action_rows = []  # candidate_action_matrix是一个01二维矩阵，每一行对应一个候选突变组合（reverse_key），每一列对应一个reverse_mutation
        for reverse_key in self.candidate_reverse_keys:
            reverse_set = set(self._split_key(reverse_key))
            action_rows.append([
                0 if mutation in reverse_set else 1  # 对每一个reverse_mutation，应该记录它现在的状态
                for mutation in self.reverse_mutations
            ])  # 1代表在这个候选突变组合中，这个位点的氨基酸是成熟抗体的氨基酸（forward mutation），0代表是非成熟抗体的氨基酸（reverse mutation）。也就是说，未来的动作中，如果做出的动作符合1，那就是合理的
        self.candidate_action_matrix = torch.tensor(action_rows, dtype=torch.long)

        ranked = sorted(
            self.score_by_reverse_key.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        self.rank_by_reverse_key = {
            reverse_key: rank
            for rank, (reverse_key, _) in enumerate(ranked, start=1)
        }

    def _load_plm_embedding(self):
        if not os.path.exists(self.plm_embedding_path):
            self._process_plm_embedding()
        embeddings = torch.load(self.plm_embedding_path, map_location='cpu')
        if self.pdb_id not in embeddings:
            raise KeyError(f"PLM embedding for {self.pdb_id} not found in {self.plm_embedding_path}")
        self.plm_embedding = embeddings[self.pdb_id]

    def _process_plm_embedding(self):
        if self.plm_path is None:
            raise ValueError("plm_path must be provided to build missing PLM embeddings")

        tokenizer = AutoTokenizer.from_pretrained(self.plm_path)
        model = AutoModel.from_pretrained(self.plm_path)
        model.eval()
        model = model.to(self.device)

        chain2sequence = parse_pdb_chain2sequence(self.immature_pdb_path)
        complex_info = parse_pdb(self.immature_pdb_path)
        chain_id_list = complex_info['chain_id']

        assert list(chain2sequence.keys())[0] == chain_id_list[0] and list(chain2sequence.keys())[-1] == chain_id_list[-1], "chain order mismatch!"

        embedding_list = []
        for _, sequence in tqdm(chain2sequence.items(), desc=f"Processing PLM embeddings for {self.pdb_id} immature"):
            inputs = tokenizer(sequence, return_tensors="pt")
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state.squeeze(0)[1:len(sequence)+1, :].cpu()
            embedding_list.append(embedding)

        embedding = torch.cat(embedding_list, dim=0)
        assert embedding.shape[0] == len(chain_id_list), "PLM embedding length mismatch!"

        os.makedirs(os.path.dirname(self.plm_embedding_path), exist_ok=True)
        torch.save({self.pdb_id: embedding}, self.plm_embedding_path)

    def _parse_with_plm(self, pdb_path):
        complex_info = parse_pdb(pdb_path)
        if self.use_plm_embedding:
            complex_info['plm_embedding'] = self.plm_embedding
        return complex_info

    def _make_site_mask(self, complex_info):
        mask = torch.zeros_like(complex_info['aa'], dtype=torch.bool)
        for idx, (chain_id, resseq) in enumerate(zip(complex_info['chain_id'], complex_info['resseq'])):
            if chain_id == self.antibody_chain and int(resseq.item()) in self.position_to_site:
                mask[idx] = True
        if int(mask.sum().item()) != len(self.site_infos):
            raise ValueError(
                f"Expected {len(self.site_infos)} mutable sites for {self.pdb_id}, found {int(mask.sum().item())}"
            )
        return mask

    def _reverse_key_to_pdb_path(self, reverse_key):
        if reverse_key == '':
            return self.mature_pdb_path
        return os.path.join(self.mutated_dir, f"{self.pdb_id}_{reverse_key}.pdb")

    def _parse_reverse_mutation(self, mutation):
        match = MUTATION_RE.match(mutation)
        if match is None:
            raise ValueError(f"Invalid mutation string: {mutation}")
        mature_aa, chain, position, immature_aa = match.groups()
        return {
            'reverse_mutation': mutation,
            'mature_aa': mature_aa,
            'chain': chain,
            'position': int(position),
            'immature_aa': immature_aa,
        }

    def _reverse_to_forward_mutation(self, mutation):
        info = self._parse_reverse_mutation(mutation)
        return f"{info['immature_aa']}{info['chain']}{info['position']}{info['mature_aa']}"

    def _split_key(self, mutation_key):
        if mutation_key is None:
            return []
        if isinstance(mutation_key, float) and pd.isna(mutation_key):
            return []
        mutation_key = str(mutation_key)
        if mutation_key == '' or mutation_key == 'nan':
            return []
        return [m for m in mutation_key.split(',') if m]

    def _canonical_reverse_key(self, mutations):
        return ','.join(sorted(list(mutations), key=self._mutation_sort_key))

    def _mutation_sort_key(self, mutation):
        info = self._parse_reverse_mutation(mutation)
        return info['position'], info['chain'], info['mature_aa'], info['immature_aa']

    def _combo_sort_key(self, reverse_key):
        mutations = self._split_key(reverse_key)
        return len(mutations), tuple(self._mutation_sort_key(m) for m in mutations)
