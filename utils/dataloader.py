import numpy as np
import os
from torch.utils.data.dataset import Dataset
from protein.read_pdbs import read_pdb_3D, parse_pdb, KnnResidue, PaddingCollate, KnnAgnet
from utils.util import recursive_to
import torch


class SKEMPIV2Dataset(Dataset):
    def __init__(self, data_df, knn_num, knn_agents_num):
        super(SKEMPIV2Dataset, self).__init__()

        self.data_df = data_df
        self.data_batches = len(data_df['PDB_id'])
        self.knn_num = knn_num
        self.knn_agents_num = knn_agents_num

    def __len__(self):
        return self.data_batches

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def __getitem__(self, index):
        # process sample info
        index = index % self.data_batches
        sample_info = self.data_df.iloc[index].values
        _, PDB_id, _, _, mutate_info, _, ddG = sample_info
        PDB_id = PDB_id.replace('+', '')
        PDB_id = PDB_id.replace('.00', '')
        mutate_info = mutate_info.replace(',', '_')

        # read wt and mut files
        path0 = os.getcwd()
        PDB_wt_file_path = path0 + '/data/SKEMPIv2/PDBs_fixed/' + str(PDB_id) + '.pdb'
        PDB_mut_file_path = path0 + '/data/SKEMPIv2/PDBs_mutated/' + str(PDB_id) + '_' + mutate_info + '.pdb'

        complex_wt_info = parse_pdb(PDB_wt_file_path)
        complex_mut_info = parse_pdb(PDB_mut_file_path)

        # preprocess structure info
        transform = KnnResidue(num_neighbors=self.knn_num)
        agent_select = KnnAgnet(num_neighbors=self.knn_agents_num)

        mutation_mask = (complex_wt_info['aa'] != complex_mut_info['aa'])

        agent_mask = agent_select({'wt': complex_wt_info, 'mut': complex_mut_info, 'mutation_mask': mutation_mask})

        complex_wt_info['agent_mask'] = agent_mask
        complex_wt_info['PDB_id'] = PDB_id
        complex_wt_info['mutate_info'] = mutate_info

        complex_mut_info['agent_mask'] = agent_mask
        complex_mut_info['PDB_id'] = PDB_id
        complex_mut_info['mutate_info'] = mutate_info

        batch = transform({'wt': complex_wt_info, 'mut': complex_mut_info, 'mutation_mask': mutation_mask})

        # label
        batch['ddG'] = ddG

        return batch


class ABbindDataset(Dataset):
    def __init__(self, data_df, knn_num, knn_agents_num):
        super(ABbindDataset, self).__init__()

        self.data_df = data_df
        self.data_batches = len(data_df['PDB_id'])
        self.knn_num = knn_num
        self.knn_agents_num = knn_agents_num

    def __len__(self):
        return self.data_batches

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def __getitem__(self, index):
        # process sample info
        index = index % self.data_batches
        sample_info = self.data_df.iloc[index].values
        PDB_id, partners, mutate_info, ddG = sample_info
        PDB_id = PDB_id.replace('+', '')
        PDB_id = PDB_id.replace('.00', '')
        mutate_info = mutate_info.replace(',', '_')
        mutate_info = mutate_info.replace(':', '')

        # read wt and mut files
        path0 = os.getcwd()
        PDB_wt_file_path = path0 + '/data/ABbind/PDBs_fixed/' + str(PDB_id) + '.pdb'
        PDB_mut_file_path = path0 + '/data/ABbind/PDBs_mutated/' + str(PDB_id) + '_' + mutate_info + '.pdb'

        complex_wt_info = parse_pdb(PDB_wt_file_path)
        complex_mut_info = parse_pdb(PDB_mut_file_path)

        # preprocess structure info
        transform = KnnResidue(num_neighbors=self.knn_num)
        agent_select = KnnAgnet(num_neighbors=self.knn_agents_num)

        mutation_mask = (complex_wt_info['aa'] != complex_mut_info['aa'])

        complex_wt_info['PDB_id'] = PDB_id
        complex_wt_info['mutate_info'] = mutate_info
        complex_mut_info['PDB_id'] = PDB_id
        complex_mut_info['mutate_info'] = mutate_info

        agent_mask = agent_select({'wt': complex_wt_info, 'mut': complex_mut_info, 'mutation_mask': mutation_mask})

        complex_wt_info['agent_mask'] = agent_mask
        complex_mut_info['agent_mask'] = agent_mask

        batch = transform({'wt': complex_wt_info, 'mut': complex_mut_info, 'mutation_mask': mutation_mask})

        # label
        batch['ddG'] = ddG
        
        return batch


class ABbindEvoDataset(Dataset):
    def __init__(self, data_df, knn_num, knn_agents_num):
        super(ABbindEvoDataset, self).__init__()

        self.data_df = data_df
        self.knn_num = knn_num
        self.knn_agents_num = knn_agents_num

    def __len__(self):
        return len(self.data_df['PDB_id'])

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def __getitem__(self, index):

        index = index % len(self.data_df['PDB_id'])
        
        PDB_id = self.data_df['PDB_id'].iloc[index]
        partners = self.data_df['chain'].iloc[index]
        antibody_type = self.data_df['antibody_type'].iloc[index]
        antibody_chain = self.data_df['antibody_chain'].iloc[index]
        seq_len = self.data_df['seq_len'].iloc[index]
        sequence = self.data_df['sequence'].iloc[index]
        CDRH3 = self.data_df['CDRH3'].iloc[index]
        mutate_info = self.data_df['mutate_info'].iloc[index]
        mutate_num = self.data_df['mutate_num'].iloc[index]
        docking_energy = 'no_docking'
        file_index = PDB_id
        
        path0 = os.getcwd()
        PDB_wt_file_path = path0 + '/data/ABbind/PDBs_fixed/' + PDB_id + '.pdb'
        complex_wt_info = parse_pdb(PDB_wt_file_path)

        transform = KnnResidue(num_neighbors=self.knn_num)

        mutation_mask = torch.zeros_like(complex_wt_info['aa'], dtype=torch.bool)
        CDR_idx_start = sequence.find(CDRH3)
        CDR_idx_end = sequence.find(CDRH3) + len(CDRH3)
        bias = complex_wt_info['chain_id'].find(antibody_chain)
        mutation_mask[bias + CDR_idx_start:bias + CDR_idx_end] = True

        batch = transform({'wt': complex_wt_info, 'mutation_mask': mutation_mask})

        expand_data_info = {"PDB_id": PDB_id, "chain": partners, "antibody_type": antibody_type,
                            "antibody_chain": antibody_chain, "seq_len": seq_len, "sequence": sequence,
                            "CDRH3": CDRH3, "mutate_info": mutate_info, "mutate_num": mutate_num,
                            "docking_energy": docking_energy, 'file_index': file_index}

        index_info = {"wt_path": PDB_wt_file_path}

        batch['expand_data_info'] = expand_data_info
        batch['index_info'] = index_info
        
        return batch


class SARSEvoDataset(Dataset):
    def __init__(self, data_df, knn_num, knn_agents_num):
        super(SARSEvoDataset, self).__init__()
        
        self.data_df = data_df
        self.knn_num = knn_num
        self.knn_agents_num = knn_agents_num
    
    def __len__(self):
        return len(self.data_df['PDB_id'])
    
    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a
    
    def __getitem__(self, index):
        
        index = index % len(self.data_df['PDB_id'])
        PDB_id = self.data_df['PDB_mut_id'].iloc[index]
        partners = self.data_df['partner'].iloc[index]
        antibody_type = self.data_df['antibody_type'].iloc[index]
        antibody_chain = self.data_df['antibody_chain'].iloc[index]
        sequence = self.data_df['seq'].iloc[index]
        CDRH3 = self.data_df['CDRH3'].iloc[index]

        path0 = os.getcwd()
        
        PDB_wt_file_path = path0 + '/data/SARS_COV_2/PDBs_mutated/' + PDB_id + '.pdb'
        
        complex_wt_info = parse_pdb(PDB_wt_file_path)
        
        transform = KnnResidue(num_neighbors=self.knn_num)
        
        mutation_mask = torch.zeros_like(complex_wt_info['aa'], dtype=torch.bool)
        CDR_idx_start = sequence.find(CDRH3)
        CDR_idx_end = sequence.find(CDRH3) + len(CDRH3)
        bias = complex_wt_info['chain_id'].find(antibody_chain)
        mutation_mask[bias + CDR_idx_start:bias + CDR_idx_end] = True
        
        batch = transform({'wt': complex_wt_info, 'mutation_mask': mutation_mask})
        
        expand_data_info = {"PDB_id": PDB_id, "chain": partners, "antibody_type": antibody_type,
                            "antibody_chain": antibody_chain, "sequence": sequence,
                            "CDRH3": CDRH3}
        
        index_info = {"wt_path": PDB_wt_file_path}
        
        batch['expand_data_info'] = expand_data_info
        batch['index_info'] = index_info
        
        return batch
