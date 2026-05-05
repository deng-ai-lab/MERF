import numpy as np
import os
from torch.utils.data.dataset import Dataset
from protein.read_pdbs import read_pdb_3D, parse_pdb, KnnResidue, PaddingCollate, KnnAgnet
from utils.util import recursive_to
import torch


class DDGBaseDataset(Dataset):
    def __init__(self, data_df, fix_dir, mut_dir, knn_num, knn_agents_num):
        super(DDGBaseDataset, self).__init__()


class VMDataset(Dataset):
    def __init__(self, data_df, fix_dir, mut_dir, knn_num, knn_agents_num):
        super(VMDataset, self).__init__()

        self.data_df = data_df
        self.data_batches = len(data_df)
        self.fix_dir = fix_dir
        self.mut_dir = mut_dir
        self.knn_num = knn_num
        self.knn_agents_num = knn_agents_num

    def __len__(self):
        return self.data_batches

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def __getitem__(self, index):
        # process sample info
        pdb_id = self.data_df['pdb_id'].iloc[index]
        mutate_info = self.data_df['mutant'].iloc[index]
        ddG = self.data_df['ddG'].iloc[index]

        pdb_id = pdb_id.replace('+', '')
        pdb_id = pdb_id.replace('.00', '')
        # mutate_info = mutate_info.replace(',', '_')  # TODO: maybe no need

        assert len(mutate_info.split(',')) == 1, "Too many mutations, check data!"

        chain = 'A'
        wtname = mutate_info[0]
        resid = str(int(mutate_info[1:-1]))
        mutname = mutate_info[-1]
        mutate_info_reformat = f"{wtname}{chain}{resid}{mutname}"

        PDB_wt_file_path = os.path.join(self.fix_dir, f"{pdb_id}.pdb")
        PDB_mut_file_path = os.path.join(self.mut_dir, f"{pdb_id}_{mutate_info_reformat}.pdb")

        complex_wt_info = parse_pdb(PDB_wt_file_path)
        complex_mut_info = parse_pdb(PDB_mut_file_path)

        # preprocess structure info
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

        # label
        batch['ddG'] = ddG

        return batch


class SKEMPIV2Dataset(Dataset):
    def __init__(self, data_df, wt_dir, mut_dir, knn_num, knn_agents_num):
        super(SKEMPIV2Dataset, self).__init__()

        self.data_df = data_df
        self.data_batches = len(data_df)
        self.wt_dir = wt_dir 
        self.mut_dir = mut_dir
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

        # mutate_info = mutate_info.replace(',', '_')  # temp for old model

        # read wt and mut files
        PDB_wt_file_path = os.path.join(self.wt_dir, f"{PDB_id}.pdb")
        PDB_mut_file_path = os.path.join(self.mut_dir, f"{PDB_id}_{mutate_info}.pdb")

        complex_wt_info = parse_pdb(PDB_wt_file_path)
        complex_mut_info = parse_pdb(PDB_mut_file_path)

        # preprocess structure info
        transform = KnnResidue(num_neighbors=self.knn_num)
        agent_select = KnnAgnet(num_neighbors=self.knn_agents_num)

        try:
            mutation_mask = (complex_wt_info['aa'] != complex_mut_info['aa'])
        except:
            print(1)

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
    def __init__(self, data_df, wt_dir, mut_dir, knn_num, knn_agents_num):
        super(ABbindDataset, self).__init__()

        self.data_df = data_df
        self.wt_dir = wt_dir
        self.mut_dir = mut_dir
        self.data_batches = len(data_df)
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

        # temp for old model
        # mutate_info = mutate_info.replace(',', '_')
        # mutate_info = mutate_info.replace(':', '')

        # read wt and mut files
        path0 = os.getcwd()
        PDB_wt_file_path = os.path.join(self.wt_dir, f"{PDB_id}.pdb")
        PDB_mut_file_path = os.path.join(self.mut_dir, f"{PDB_id}_{mutate_info}.pdb")

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

