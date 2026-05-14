import numpy as np
import pandas as pd
import os
import hashlib
from collections import defaultdict
from torch.utils.data.dataset import Dataset
from protein.read_pdbs import read_pdb_3D, parse_pdb, KnnResidue, PaddingCollate, KnnAgnet, parse_pdb_chain2sequence
from utils.util import recursive_to
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# Amino acid key for mapping Q-values to amino acid types (same order as in evo scripts)
AA_KEY = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AA_KEY)}


class DDGBaseDataset(Dataset):
    def __init__(self, data_path, fix_dir, mut_dir, knn_num, knn_agents_num,
                use_plm_embedding=False, plm_path=None, plm_embedding_path=None, device='cpu',
                return_kl_target=False
        ):
        super(DDGBaseDataset, self).__init__()
        self.data_df = pd.read_csv(data_path, dtype={"pdb_id": "string"})
        self.fix_dir = fix_dir
        self.mut_dir = mut_dir
        self.knn_num = knn_num
        self.knn_agents_num = knn_agents_num

        self.use_plm_embedding = use_plm_embedding
        self.plm_path = plm_path
        self.plm_embedding_path = plm_embedding_path
        self.device = device
        self.return_kl_target = return_kl_target

        if self.use_plm_embedding:
            assert (self.plm_embedding_path is not None) or (self.plm_path is not None), "sth. about plm must be provided."
            self.process_pml_embedding()

    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, index):
        pdb_id = self.data_df['pdb_id'].iloc[index]
        mutate_info = self.data_df['mutant'].iloc[index]
        if 'ddG' in self.data_df.columns:
            ddG = self.data_df['ddG'].iloc[index]
        else:
            ddG = 0.0  # placeholder if no ground truth

        pdb_id = pdb_id.replace('+', '')
        pdb_id = pdb_id.replace('.00', '')
        
        mutate_info_processed = self.process_mutate_info(mutate_info)

        PDB_wt_file_path = os.path.join(self.fix_dir, f"{pdb_id}.pdb")
        PDB_mut_file_path = os.path.join(self.mut_dir, f"{pdb_id}_{mutate_info_processed}.pdb")

        complex_wt_info = parse_pdb(PDB_wt_file_path)
        complex_mut_info = parse_pdb(PDB_mut_file_path)

        # add plm embedding fetch
        if self.use_plm_embedding:
            assert pdb_id in self.plm_embeddings_dict, f"PLM embedding for {pdb_id} not found!"
            plm_embedding = self.plm_embeddings_dict[pdb_id]
            complex_wt_info['plm_embedding'] = plm_embedding
            complex_mut_info['plm_embedding'] = plm_embedding

        transform = KnnResidue(num_neighbors=self.knn_num)
        agent_select = KnnAgnet(num_neighbors=self.knn_agents_num)

        mutation_mask = (complex_wt_info['aa'] != complex_mut_info['aa'])

        agent_mask = agent_select({'wt': complex_wt_info, 'mut': complex_mut_info, 'mutation_mask': mutation_mask})

        complex_wt_info['agent_mask'] = agent_mask
        complex_wt_info['PDB_id'] = pdb_id
        complex_wt_info['mutate_info'] = mutate_info

        complex_mut_info['agent_mask'] = agent_mask
        complex_mut_info['PDB_id'] = pdb_id
        complex_mut_info['mutate_info'] = mutate_info

        batch = transform({'wt': complex_wt_info, 'mut': complex_mut_info, 'mutation_mask': mutation_mask})

        batch['ddG'] = ddG

        # Add KL target placeholder for DDGBaseDataset (no multi-mutation data available)
        if self.return_kl_target:
            # Return zero-filled tensor and flag indicating no valid KL target
            batch['kl_target'] = torch.zeros(20)
            batch['has_kl_target'] = torch.tensor(False)

        return batch

    def process_mutate_info(self, mutate_info):
        return mutate_info

    def process_pml_embedding(self):
        if (self.plm_embedding_path is not None) and os.path.exists(self.plm_embedding_path):
            self.plm_embeddings_dict = torch.load(self.plm_embedding_path, map_location='cpu')
            return
        
        self.plm_tokenizer = AutoTokenizer.from_pretrained(self.plm_path)
        self.plm_model = AutoModel.from_pretrained(self.plm_path)
        self.plm_model.eval()
        self.plm_model = self.plm_model.to(self.device)

        # embedding
        self.plm_embeddings_dict = {}

        unique_pdb_ids = self.data_df['pdb_id'].unique()
        for pdb_id in tqdm(unique_pdb_ids, desc="Processing PLM embeddings"):
            pdb_id = pdb_id.replace('+', '')
            pdb_id = pdb_id.replace('.00', '')

            PDB_wt_file_path = os.path.join(self.fix_dir, f"{pdb_id}.pdb")
            complex_wt_info = parse_pdb(PDB_wt_file_path)
            chain_id_list = complex_wt_info['chain_id']

            chain2sequence = parse_pdb_chain2sequence(PDB_wt_file_path)

            # 简单检查是否和主数据处理都sort了一样的chain顺序
            assert list(chain2sequence.keys())[0] == chain_id_list[0] and list(chain2sequence.keys())[-1] == chain_id_list[-1], "chain order mismatch!"

            embeddding_list = []
            for chain_id, sequence in chain2sequence.items():

                inputs = self.plm_tokenizer(sequence, return_tensors="pt")
                inputs = {key: val.to(self.device) for key, val in inputs.items()}

                with torch.no_grad():
                    outputs = self.plm_model(**inputs)
                    embedding = outputs.last_hidden_state.squeeze(0)[1:len(sequence)+1, :].cpu()  # remove cls and sep

                embeddding_list.append(embedding)
            
            # concat along the seq dim
            embedding = torch.cat(embeddding_list, dim=0)

            # assert the sequence length match
            assert embedding.shape[0] == len(chain_id_list), "PLM embedding length mismatch!"

            self.plm_embeddings_dict[pdb_id] = embedding

        # save
        torch.save(self.plm_embeddings_dict, self.plm_embedding_path)


class VMDataset(DDGBaseDataset):

    def __init__(self, data_path, fix_dir, mut_dir, knn_num, knn_agents_num,
                use_plm_embedding=False, plm_path=None, plm_embedding_path=None, device='cpu',
                return_kl_target=False
        ):
        # Call parent init
        super(VMDataset, self).__init__(
            data_path, fix_dir, mut_dir, knn_num, knn_agents_num,
            use_plm_embedding, plm_path, plm_embedding_path, device, return_kl_target
        )

        # Build site-specific ddG distribution index for KL loss
        if self.return_kl_target:
            self._build_site_ddg_index()

    def _build_site_ddg_index(self):
        """Build a dictionary mapping (pdb_id, position) to {mutant_aa: ddG}"""
        self.site_ddg_index = defaultdict(dict)

        for idx in tqdm(range(len(self.data_df)), desc="Building site ddG index"):
            pdb_id = self.data_df['pdb_id'].iloc[idx]
            pdb_id = pdb_id.replace('+', '').replace('.00', '')
            mutate_info = self.data_df['mutant'].iloc[idx]
            ddG = self.data_df['ddG'].iloc[idx]

            # Parse mutation info: format is like "S1Q" (wt_aa + position + mut_aa)
            wt_aa = mutate_info[0]
            position = str(int(mutate_info[1:-1]))
            mut_aa = mutate_info[-1]

            # Key is (pdb_id, position), value is {mut_aa: ddG}
            site_key = (pdb_id, position)
            self.site_ddg_index[site_key][mut_aa] = ddG
            # Also store wild-type aa as reference (ddG=0 for self-mutation)
            if wt_aa not in self.site_ddg_index[site_key]:
                self.site_ddg_index[site_key][wt_aa] = 0.0  # WT->WT has ddG=0

    def _get_kl_target(self, pdb_id, mutate_info):
        """Get the 20-dim ddG distribution for the mutation site.

        Returns:
            kl_target: tensor of shape (20,), missing positions filled with mean ddG
            has_kl_target: True if we have valid data for this site (at least 2 real values)
        """
        # Parse mutation info
        position = str(int(mutate_info[1:-1]))
        site_key = (pdb_id, position)

        # Get all mutations at this site
        site_ddg_dict = self.site_ddg_index.get(site_key, {})

        # Collect known ddG values and their positions
        known_ddg_values = []
        known_indices = []
        for aa, ddG in site_ddg_dict.items():
            if aa in AA_TO_IDX:
                known_indices.append(AA_TO_IDX[aa])
                known_ddg_values.append(ddG)

        # Check if we have enough valid entries (at least 2 real values)
        valid_count = len(known_ddg_values)
        has_kl_target = valid_count >= 3  # We require at least 3 known values to have a 'meaningful' distribution (including WT)

        # Fill all positions with mean ddG first, then overwrite known positions
        if valid_count > 0:
            mean_ddg = sum(known_ddg_values) / valid_count
            kl_target = torch.full((20,), mean_ddg)
            for idx, ddg in zip(known_indices, known_ddg_values):
                kl_target[idx] = ddg
        else:
            # No data available, fill with 0 (neutral)
            kl_target = torch.zeros(20)

        return kl_target, has_kl_target

    def __getitem__(self, index):
        # Call parent's __getitem__ to get the base batch
        batch = super(VMDataset, self).__getitem__(index)

        # Override KL target with actual distribution if return_kl_target is enabled
        if self.return_kl_target:
            pdb_id = self.data_df['pdb_id'].iloc[index]
            pdb_id = pdb_id.replace('+', '').replace('.00', '')
            mutate_info = self.data_df['mutant'].iloc[index]

            kl_target, has_kl_target = self._get_kl_target(pdb_id, mutate_info)
            batch['kl_target'] = kl_target
            batch['has_kl_target'] = torch.tensor(has_kl_target)

        return batch

    def process_mutate_info(self, mutate_info):
        assert len(mutate_info.split(',')) == 1, "Too many mutations, check data!"

        chain = 'A'
        wtname = mutate_info[0]
        resid = str(int(mutate_info[1:-1]))
        mutname = mutate_info[-1]
        mutate_info_processed = f"{wtname}{chain}{resid}{mutname}"

        return mutate_info_processed
    

class VMSiteDataset(VMDataset):
    """VM training dataset where one item is one (pdb_id, mutation position) site.

    Each epoch visits every site once and deterministically rotates through the
    feasible single mutations observed at that site.
    """

    def __init__(self, data_path, fix_dir, mut_dir, knn_num, knn_agents_num,
                use_plm_embedding=False, plm_path=None, plm_embedding_path=None, device='cpu',
                return_kl_target=False
        ):
        super(VMSiteDataset, self).__init__(
            data_path, fix_dir, mut_dir, knn_num, knn_agents_num,
            use_plm_embedding, plm_path, plm_embedding_path, device, return_kl_target
        )
        self.current_epoch = 0
        self.site_keys = []
        self.site_to_indices = defaultdict(list)
        self._build_site_samples()

    def _build_site_samples(self):
        for idx in tqdm(range(len(self.data_df)), desc="Building site samples"):
            pdb_id = self.data_df['pdb_id'].iloc[idx]
            pdb_id = pdb_id.replace('+', '').replace('.00', '')
            mutate_info = self.data_df['mutant'].iloc[idx]
            position = str(int(mutate_info[1:-1]))
            site_key = (pdb_id, position)

            if site_key not in self.site_to_indices:
                self.site_keys.append(site_key)
            self.site_to_indices[site_key].append(idx)

    def __len__(self):
        return len(self.site_keys)

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    @staticmethod
    def _stable_site_offset(site_key):
        key = f"{site_key[0]}:{site_key[1]}".encode("utf-8")
        return int(hashlib.md5(key).hexdigest()[:8], 16)

    def _select_row_index(self, site_index):
        site_key = self.site_keys[site_index]
        row_indices = self.site_to_indices[site_key]
        row_offset = (self.current_epoch + self._stable_site_offset(site_key)) % len(row_indices)
        return row_indices[row_offset]

    def __getitem__(self, index):
        return super(VMSiteDataset, self).__getitem__(self._select_row_index(index))


class CRDataset(DDGBaseDataset):
    """Dataset for CR deep scanning mutation data"""

    def __init__(self, data_path, fix_dir, mut_dir, knn_num, knn_agents_num,
                use_plm_embedding=False, plm_path=None, plm_embedding_path=None, device='cpu',
                target_column='score'):
        """
        Args:
            target_column: The column name for the target score (e.g., 'h1_score', 'h9_score')
        """
        self.target_column = target_column
        super(CRDataset, self).__init__(
            data_path, fix_dir, mut_dir, knn_num, knn_agents_num,
            use_plm_embedding, plm_path, plm_embedding_path, device
        )

    def __getitem__(self, index):
        pdb_id = self.data_df['pdb_id'].iloc[index]
        mutate_info = self.data_df['mutant'].iloc[index]

        # Get target value if available
        if self.target_column in self.data_df.columns:
            ddG = self.data_df[self.target_column].iloc[index]
        else:
            ddG = 0.0  # Placeholder if no ground truth

        ddG = -1 * ddG  # Convert to ddG-like format (higher score = better, so negate if original is like affinity)

        pdb_id = pdb_id.replace('+', '')
        pdb_id = pdb_id.replace('.00', '')

        # CR dataset mutant format: "PA28T,TA58A,PA62Q"
        # File format: cr6261_h1_PA28T,TA58A,PA62Q.pdb
        mutate_info_processed = self.process_mutate_info(mutate_info)

        PDB_wt_file_path = os.path.join(self.fix_dir, f"{pdb_id}.pdb")
        PDB_mut_file_path = os.path.join(self.mut_dir, f"{pdb_id}_{mutate_info_processed}.pdb")

        complex_wt_info = parse_pdb(PDB_wt_file_path)
        complex_mut_info = parse_pdb(PDB_mut_file_path)

        # Add plm embedding if needed
        if self.use_plm_embedding:
            assert pdb_id in self.plm_embeddings_dict, f"PLM embedding for {pdb_id} not found!"
            plm_embedding = self.plm_embeddings_dict[pdb_id]
            complex_wt_info['plm_embedding'] = plm_embedding
            complex_mut_info['plm_embedding'] = plm_embedding

        transform = KnnResidue(num_neighbors=self.knn_num)
        agent_select = KnnAgnet(num_neighbors=self.knn_agents_num)

        mutation_mask = (complex_wt_info['aa'] != complex_mut_info['aa'])

        agent_mask = agent_select({'wt': complex_wt_info, 'mut': complex_mut_info, 'mutation_mask': mutation_mask})

        complex_wt_info['agent_mask'] = agent_mask
        complex_wt_info['PDB_id'] = pdb_id
        complex_wt_info['mutate_info'] = mutate_info

        complex_mut_info['agent_mask'] = agent_mask
        complex_mut_info['PDB_id'] = pdb_id
        complex_mut_info['mutate_info'] = mutate_info

        batch = transform({'wt': complex_wt_info, 'mut': complex_mut_info, 'mutation_mask': mutation_mask})

        batch['ddG'] = ddG

        return batch
