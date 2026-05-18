import numpy as np
import pandas as pd
import os
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