import numpy as np
import pandas as pd
import os
import shutil
from torch.utils.data.dataset import Dataset
from protein.read_pdbs import read_pdb_3D, parse_pdb, KnnResidue, PaddingCollate, KnnAgnet, parse_pdb_chain2sequence
from utils.util import recursive_to
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from protein.mutate_scripts_foldx import mut_list

class EvoBaseDataset(Dataset):
    def __init__(
            self, pdb_id, antibody_chain, partner, sequence, cdr,
            wt_dir, fix_dir, mut_dir, knn_num, knn_agents_num,
            use_plm_embedding=False, plm_path=None, plm_embedding_path=None, device='cpu'
        ):
        super(EvoBaseDataset, self).__init__()

        self.pdb_id = pdb_id
        self.antibody_chain = antibody_chain
        self.partner = partner
        self.sequence = sequence
        self.cdr = cdr

        self.wt_dir = wt_dir
        self.fix_dir = fix_dir
        self.mut_dir = mut_dir
        self.knn_num = knn_num
        self.knn_agents_num = knn_agents_num

        self.use_plm_embedding = use_plm_embedding
        self.plm_path = plm_path
        self.plm_embedding_path = plm_embedding_path
        self.device = device

        if self.use_plm_embedding:
            assert (self.plm_embedding_path is not None) or (self.plm_path is not None), "sth. about plm must be provided."
            self.process_pml_embedding()

    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        PDB_wt_file_path = os.path.join(self.wt_dir, f"{self.pdb_id}.pdb")
        # PDB_wt_file_path = os.path.join(self.fix_dir, f"{self.pdb_id}.pdb")
        complex_wt_info = parse_pdb(PDB_wt_file_path)

        transform = KnnResidue(num_neighbors=self.knn_num)
        
        mutation_mask = torch.zeros_like(complex_wt_info['aa'], dtype=torch.bool)
        CDR_idx_start = self.sequence.find(self.cdr)
        CDR_idx_end = self.sequence.find(self.cdr) + len(self.cdr)
        bias = complex_wt_info['chain_id'].find(self.antibody_chain)
        mutation_mask[bias + CDR_idx_start:bias + CDR_idx_end] = True

        batch = transform({'wt': complex_wt_info, 'mutation_mask': mutation_mask})

        expand_data_info = {"pdb_id": self.pdb_id, "chain": self.partner,
                            "antibody_chain": self.antibody_chain, "sequence": self.sequence,
                            "cdr": self.cdr}
        
        batch['expand_data_info'] = expand_data_info

        return batch

    def getitem_by_mutate_info(self, mutate_info):
        PDB_wt_file_path = os.path.join(self.fix_dir, f"{self.pdb_id}.pdb")
        PDB_mut_file_path = os.path.join(self.mut_dir, f"{self.pdb_id}_{mutate_info}.pdb")

        complex_wt_info = parse_pdb(PDB_wt_file_path)
        complex_mut_info = parse_pdb(PDB_mut_file_path)

        # add plm embedding fetch
        if self.use_plm_embedding:
            assert self.pdb_id in self.plm_embeddings_dict, f"PLM embedding for {self.pdb_id} not found!"
            plm_embedding = self.plm_embeddings_dict[self.pdb_id]
            complex_wt_info['plm_embedding'] = plm_embedding
            complex_mut_info['plm_embedding'] = plm_embedding

        transform = KnnResidue(num_neighbors=self.knn_num)
        agent_select = KnnAgnet(num_neighbors=self.knn_agents_num)

        mutation_mask = (complex_wt_info['aa'] != complex_mut_info['aa'])

        agent_mask = agent_select({'wt': complex_wt_info, 'mut': complex_mut_info, 'mutation_mask': mutation_mask})

        complex_wt_info['agent_mask'] = agent_mask
        complex_wt_info['PDB_id'] = self.pdb_id
        complex_wt_info['mutate_info'] = mutate_info

        complex_mut_info['agent_mask'] = agent_mask
        complex_mut_info['PDB_id'] = self.pdb_id
        complex_mut_info['mutate_info'] = mutate_info

        batch = transform({'wt': complex_wt_info, 'mut': complex_mut_info, 'mutation_mask': mutation_mask})
        batch['ddG'] = 0

        return batch

    def process_pml_embedding(self):

        self.plm_embeddings_dict = {}  # 如果存在embedding文件，则覆盖这个dict，这样之后只会添加对应pdbid

        if (self.plm_embedding_path is not None) and os.path.exists(self.plm_embedding_path):
            self.plm_embeddings_dict = torch.load(self.plm_embedding_path, map_location='cpu')
            if self.pdb_id in self.plm_embeddings_dict:
                return
        
        self.plm_tokenizer = AutoTokenizer.from_pretrained(self.plm_path)
        self.plm_model = AutoModel.from_pretrained(self.plm_path)
        self.plm_model.eval()
        self.plm_model = self.plm_model.to(self.device)

        PDB_wt_file_path = os.path.join(self.wt_dir, f"{self.pdb_id}.pdb")
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

        self.plm_embeddings_dict[self.pdb_id] = embedding

        # save
        torch.save(self.plm_embeddings_dict, self.plm_embedding_path)

    def mutate_pdb(self, mutate_info_list):
        mut_list(self.pdb_id, mutate_info_list, self.wt_dir, self.fix_dir, self.mut_dir)


class EvoSarsDataset(Dataset):
    variants = ('wt', 'delta', 'gamma', 'omicron')

    def __init__(
            self, pdb_id, antibody_chain, sequence, cdr,
            wt_dir, variant_dir, evolved_dir, knn_num, knn_agents_num,
            use_plm_embedding=False, plm_path=None, plm_embedding_path=None, device='cpu'
        ):
        super(EvoSarsDataset, self).__init__()

        self.pdb_id = pdb_id
        self.antibody_chain = antibody_chain
        self.sequence = sequence
        self.cdr = cdr

        self.wt_dir = wt_dir
        self.variant_dir = variant_dir
        self.evolved_dir = evolved_dir
        self.knn_num = knn_num
        self.knn_agents_num = knn_agents_num

        self.use_plm_embedding = use_plm_embedding
        self.plm_path = plm_path
        self.plm_embedding_path = plm_embedding_path
        self.device = device

        os.makedirs(self.evolved_dir, exist_ok=True)

        if self.use_plm_embedding:
            assert (self.plm_embedding_path is not None) or (self.plm_path is not None), "sth. about plm must be provided."
            self.process_pml_embedding()

    def __len__(self):
        return 1

    def variant_pdb_id(self, variant):
        if variant == 'wt':
            return self.pdb_id
        return f'{self.pdb_id}_{variant}'

    def variant_pdb_path(self, variant):
        pdb_id = self.variant_pdb_id(variant)
        if variant == 'wt':
            return os.path.join(self.wt_dir, f'{pdb_id}.pdb')
        return os.path.join(self.variant_dir, f'{pdb_id}.pdb')

    def evolved_pdb_path(self, variant, mutate_info):
        return os.path.join(self.evolved_dir, f'{self.variant_pdb_id(variant)}_{mutate_info}.pdb')

    def _full_cdr_mask(self, complex_info):
        mutation_mask = torch.zeros_like(complex_info['aa'], dtype=torch.bool)
        cdr_start = self.sequence.find(self.cdr)
        if cdr_start < 0:
            raise ValueError(f'CDR {self.cdr} not found in antibody sequence for {self.pdb_id}')
        cdr_end = cdr_start + len(self.cdr)
        chain_start = complex_info['chain_id'].find(self.antibody_chain)
        if chain_start < 0:
            raise ValueError(f'Antibody chain {self.antibody_chain} not found in {self.pdb_id}')
        mutation_mask[chain_start + cdr_start:chain_start + cdr_end] = True
        return mutation_mask

    def _position_mask(self, complex_info, positions):
        positions = set(int(pos) for pos in positions)
        mutation_mask = torch.zeros_like(complex_info['aa'], dtype=torch.bool)
        for idx, (chain_id, resseq) in enumerate(zip(complex_info['chain_id'], complex_info['resseq'])):
            if chain_id == self.antibody_chain and int(resseq.item()) in positions:
                mutation_mask[idx] = True
        if int(mutation_mask.sum().item()) != len(positions):
            raise ValueError(f'Could not map all mutation positions {sorted(positions)} in {self.variant_pdb_id("wt")}')
        return mutation_mask

    def get_cdr_entries(self):
        complex_info = parse_pdb(self.variant_pdb_path('wt'))
        cdr_mask = self._full_cdr_mask(complex_info)
        entries = []
        for idx in torch.nonzero(cdr_mask, as_tuple=True)[0]:
            entries.append({
                'position': int(complex_info['resseq'][idx].item()),
                'aa': int(complex_info['aa'][idx].item()),
            })
        return entries

    def get_policy_item(self, variant, positions):
        pdb_path = self.variant_pdb_path(variant)
        complex_wt_info = parse_pdb(pdb_path)
        mutation_mask = self._position_mask(complex_wt_info, positions)

        if self.use_plm_embedding:
            pdb_id = self.variant_pdb_id(variant)
            assert pdb_id in self.plm_embeddings_dict, f"PLM embedding for {pdb_id} not found!"
            complex_wt_info['plm_embedding'] = self.plm_embeddings_dict[pdb_id]

        transform = KnnResidue(num_neighbors=self.knn_num)
        batch = transform({'wt': complex_wt_info, 'mutation_mask': mutation_mask})
        batch['expand_data_info'] = {
            'pdb_id': self.variant_pdb_id(variant),
            'base_pdb_id': self.pdb_id,
            'variant': variant,
            'antibody_chain': self.antibody_chain,
            'sequence': self.sequence,
            'cdr': self.cdr,
        }
        return batch

    def __getitem__(self, index):
        cdr_positions = [entry['position'] for entry in self.get_cdr_entries()]
        return self.get_policy_item('wt', cdr_positions)

    def getitem_by_mutate_info(self, variant, mutate_info):
        pdb_id = self.variant_pdb_id(variant)
        wt_path = self.variant_pdb_path(variant)
        mut_path = self.evolved_pdb_path(variant, mutate_info)

        complex_wt_info = parse_pdb(wt_path)
        complex_mut_info = parse_pdb(mut_path)

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
        batch['ddG'] = 0
        batch['expand_data_info'] = {
            'pdb_id': pdb_id,
            'base_pdb_id': self.pdb_id,
            'variant': variant,
            'antibody_chain': self.antibody_chain,
            'sequence': self.sequence,
            'cdr': self.cdr,
        }

        return batch

    def process_pml_embedding(self):
        self.plm_embeddings_dict = {}

        if (self.plm_embedding_path is not None) and os.path.exists(self.plm_embedding_path):
            self.plm_embeddings_dict = torch.load(self.plm_embedding_path, map_location='cpu')
            if all(self.variant_pdb_id(variant) in self.plm_embeddings_dict for variant in self.variants):
                return

        self.plm_tokenizer = AutoTokenizer.from_pretrained(self.plm_path)
        self.plm_model = AutoModel.from_pretrained(self.plm_path)
        self.plm_model.eval()
        self.plm_model = self.plm_model.to(self.device)

        for variant in self.variants:
            pdb_id = self.variant_pdb_id(variant)
            if pdb_id in self.plm_embeddings_dict:
                continue

            pdb_path = self.variant_pdb_path(variant)
            complex_wt_info = parse_pdb(pdb_path)
            chain_id_list = complex_wt_info['chain_id']
            chain2sequence = parse_pdb_chain2sequence(pdb_path)

            assert list(chain2sequence.keys())[0] == chain_id_list[0] and list(chain2sequence.keys())[-1] == chain_id_list[-1], "chain order mismatch!"

            embeddding_list = []
            for _, sequence in chain2sequence.items():
                inputs = self.plm_tokenizer(sequence, return_tensors="pt")
                inputs = {key: val.to(self.device) for key, val in inputs.items()}

                with torch.no_grad():
                    outputs = self.plm_model(**inputs)
                    embedding = outputs.last_hidden_state.squeeze(0)[1:len(sequence)+1, :].cpu()

                embeddding_list.append(embedding)

            embedding = torch.cat(embeddding_list, dim=0)
            assert embedding.shape[0] == len(chain_id_list), "PLM embedding length mismatch!"
            self.plm_embeddings_dict[pdb_id] = embedding

        torch.save(self.plm_embeddings_dict, self.plm_embedding_path)

    def _mutate_one_pdb(self, variant, mutate_info):
        output_path = self.evolved_pdb_path(variant, mutate_info)
        if os.path.exists(output_path):
            return

        from protein.mutate_scripts import Gen_mut

        protein_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'protein')
        source_path = self.variant_pdb_path(variant)
        work_pdb_id = self.variant_pdb_id(variant)
        safe_mutate_info = mutate_info.replace(',', '_').replace('/', '_').replace('\\', '_')
        workdir = f'pdbfile_{work_pdb_id}_{safe_mutate_info}'
        workdir_path = os.path.join(protein_dir, workdir)

        if os.path.exists(workdir_path):
            shutil.rmtree(workdir_path)
        os.makedirs(workdir_path, exist_ok=True)
        shutil.copyfile(source_path, os.path.join(workdir_path, f'{work_pdb_id}.pdb'))

        cwd = os.getcwd()
        try:
            os.chdir(protein_dir)
            for idx, single_mutate in enumerate(mutate_info.split(',')):
                chain_id = single_mutate[1]
                resid = str(int(single_mutate[2:-1]))
                mutname = single_mutate[-1]
                Gen_mut(workdir, work_pdb_id, chain_id, resid, mutname, idx)
            shutil.move(os.path.join(workdir_path, f'{work_pdb_id}_mut.pdb'), output_path)
        finally:
            os.chdir(cwd)
            if os.path.exists(workdir_path):
                shutil.rmtree(workdir_path)

    def mutate_pdb(self, mutate_info_list):
        for mutate_info in mutate_info_list:
            for variant in self.variants:
                self._mutate_one_pdb(variant, mutate_info)
