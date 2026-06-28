import os
import shutil

import torch
from torch.utils.data.dataset import Dataset
from transformers import AutoModel, AutoTokenizer

from protein.read_pdbs import parse_pdb, KnnResidue, KnnAgnet, parse_pdb_chain2sequence
from protein.mutate_scripts_foldx import mut_list


class EvoTCRDataset(Dataset):
    """
    Dataset for selective TCR evolution.

    Manages two kinds of complexes:
      - target  : the complex we want to improve affinity towards (PDBs/)
      - offtarget: complexes we want to reduce affinity towards (PDBs_offtarget/)

    Mutated structures are written to separate directories:
      - target mutations   → PDBs_mutated/
      - offtarget mutations → PDBs_offtarget_mutated/
    """

    def __init__(
            self, pdb_id, antibody_chain, partner, sequence, cdr,
            wt_dir, fix_dir, mut_dir,
            offtarget_names, offtarget_wt_dir, offtarget_fix_dir, offtarget_mut_dir,
            knn_num, knn_agents_num,
            use_plm_embedding=False, plm_path=None, plm_embedding_path=None, device='cpu',
        ):
        super().__init__()

        self.pdb_id = pdb_id
        self.antibody_chain = antibody_chain
        self.partner = partner
        self.sequence = sequence
        self.cdr = cdr

        # target directories
        self.wt_dir = wt_dir
        self.fix_dir = fix_dir
        self.mut_dir = mut_dir

        # offtarget
        self.offtarget_names = list(offtarget_names)     # e.g. ['FLDLGPPGI', 'VMAEAPPGV', ...]
        self.offtarget_wt_dir = offtarget_wt_dir
        self.offtarget_fix_dir = offtarget_fix_dir
        self.offtarget_mut_dir = offtarget_mut_dir

        self.knn_num = knn_num
        self.knn_agents_num = knn_agents_num
        self.use_plm_embedding = use_plm_embedding
        self.plm_path = plm_path
        self.plm_embedding_path = plm_embedding_path
        self.device = device

        os.makedirs(self.fix_dir, exist_ok=True)
        os.makedirs(self.mut_dir, exist_ok=True)
        os.makedirs(self.offtarget_fix_dir, exist_ok=True)
        os.makedirs(self.offtarget_mut_dir, exist_ok=True)

        if self.use_plm_embedding:
            assert (self.plm_embedding_path is not None) or (self.plm_path is not None), \
                "plm_embedding_path or plm_path must be provided."
            self._process_plm_embeddings()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_cdr_entries(self):
        """Return list of {position, aa} dicts for each CDR residue."""
        complex_info = parse_pdb(os.path.join(self.wt_dir, f'{self.pdb_id}.pdb'))
        cdr_mask = self._cdr_mask(complex_info)
        entries = []
        for idx in torch.nonzero(cdr_mask, as_tuple=True)[0]:
            entries.append({
                'position': int(complex_info['resseq'][idx].item()),
                'aa': int(complex_info['aa'][idx].item()),
            })
        return entries

    def get_policy_item(self, positions):
        """Return a batch item for the target complex at the given CDR positions."""
        pdb_path = os.path.join(self.wt_dir, f'{self.pdb_id}.pdb')
        complex_info = parse_pdb(pdb_path)
        mutation_mask = self._position_mask(complex_info, positions)
        if self.use_plm_embedding:
            complex_info['plm_embedding'] = self.plm_embeddings[self.pdb_id]
        transform = KnnResidue(num_neighbors=self.knn_num)
        batch = transform({'wt': complex_info, 'mutation_mask': mutation_mask})
        batch['expand_data_info'] = {
            'pdb_id': self.pdb_id,
            'antibody_chain': self.antibody_chain,
            'sequence': self.sequence,
            'cdr': self.cdr,
        }
        return batch

    def get_offtarget_policy_item(self, offtarget_name, positions):
        """Return a batch item for a given offtarget complex at the given CDR positions."""
        pdb_path = os.path.join(self.offtarget_wt_dir, f'{offtarget_name.lower()}.pdb')
        complex_info = parse_pdb(pdb_path)
        mutation_mask = self._position_mask(complex_info, positions)
        if self.use_plm_embedding:
            key = f'offtarget_{offtarget_name}'
            complex_info['plm_embedding'] = self.plm_embeddings[key]
        transform = KnnResidue(num_neighbors=self.knn_num)
        batch = transform({'wt': complex_info, 'mutation_mask': mutation_mask})
        batch['expand_data_info'] = {
            'pdb_id': offtarget_name,
            'antibody_chain': self.antibody_chain,
            'sequence': self.sequence,
            'cdr': self.cdr,
        }
        return batch

    def getitem_by_mutate_info(self, mutate_info):
        """Return wt/mut pair for the target complex."""
        wt_path = os.path.join(self.fix_dir, f'{self.pdb_id}.pdb')
        mut_path = os.path.join(self.mut_dir, f'{self.pdb_id}_{mutate_info}.pdb')
        return self._make_wt_mut_item(wt_path, mut_path, self.pdb_id, mutate_info, plm_key=self.pdb_id)

    def getitem_offtarget_by_mutate_info(self, offtarget_name, mutate_info):
        """Return wt/mut pair for an offtarget complex."""
        wt_path = os.path.join(self.offtarget_fix_dir, f'{offtarget_name.lower()}.pdb')
        mut_path = os.path.join(self.offtarget_mut_dir, f'{offtarget_name.lower()}_{mutate_info}.pdb')
        plm_key = f'offtarget_{offtarget_name}'
        return self._make_wt_mut_item(wt_path, mut_path, offtarget_name, mutate_info, plm_key=plm_key)

    def mutate_pdb(self, mutate_info_list):
        """Generate mutant PDBs for both target and offtarget complexes."""
        # target
        mut_list(self.pdb_id, mutate_info_list, self.wt_dir, self.fix_dir, self.mut_dir)
        # offtargets
        for offtarget_name in self.offtarget_names:
            ot_lower = offtarget_name.lower()
            mut_list(ot_lower, mutate_info_list, self.offtarget_wt_dir, self.offtarget_fix_dir, self.offtarget_mut_dir)

    # ------------------------------------------------------------------
    # Dataset protocol (returns target policy item for index 0)
    # ------------------------------------------------------------------

    def __len__(self):
        return 1

    def __getitem__(self, index):
        cdr_positions = [entry['position'] for entry in self.get_cdr_entries()]
        return self.get_policy_item(cdr_positions)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cdr_mask(self, complex_info):
        mask = torch.zeros_like(complex_info['aa'], dtype=torch.bool)
        cdr_start = self.sequence.find(self.cdr)
        if cdr_start < 0:
            raise ValueError(f'CDR {self.cdr} not found in sequence for {self.pdb_id}')
        cdr_end = cdr_start + len(self.cdr)
        chain_start = complex_info['chain_id'].find(self.antibody_chain)
        if chain_start < 0:
            raise ValueError(f'Antibody chain {self.antibody_chain} not found in {self.pdb_id}')
        mask[chain_start + cdr_start:chain_start + cdr_end] = True
        return mask

    def _position_mask(self, complex_info, positions):
        positions = set(int(p) for p in positions)
        mask = torch.zeros_like(complex_info['aa'], dtype=torch.bool)
        for idx, (chain_id, resseq) in enumerate(zip(complex_info['chain_id'], complex_info['resseq'])):
            if chain_id == self.antibody_chain and int(resseq.item()) in positions:
                mask[idx] = True
        if int(mask.sum().item()) != len(positions):
            raise ValueError(
                f'Could not map all positions {sorted(positions)} in {self.pdb_id}. '
                f'Found {int(mask.sum().item())} matches.'
            )
        return mask

    def _make_wt_mut_item(self, wt_path, mut_path, pdb_id, mutate_info, plm_key):
        complex_wt_info = parse_pdb(wt_path)
        complex_mut_info = parse_pdb(mut_path)

        if self.use_plm_embedding and plm_key in self.plm_embeddings:
            emb = self.plm_embeddings[plm_key]
            complex_wt_info['plm_embedding'] = emb
            complex_mut_info['plm_embedding'] = emb

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
        return batch

    def _process_plm_embeddings(self):
        self.plm_embeddings = {}

        if self.plm_embedding_path is not None and os.path.exists(self.plm_embedding_path):
            self.plm_embeddings = torch.load(self.plm_embedding_path, map_location='cpu')

        all_needed = [self.pdb_id] + [f'offtarget_{n}' for n in self.offtarget_names]
        if all(k in self.plm_embeddings for k in all_needed):
            return

        tokenizer = AutoTokenizer.from_pretrained(self.plm_path)
        model = AutoModel.from_pretrained(self.plm_path)
        model.eval()
        model = model.to(self.device)

        # target
        if self.pdb_id not in self.plm_embeddings:
            pdb_path = os.path.join(self.wt_dir, f'{self.pdb_id}.pdb')
            self.plm_embeddings[self.pdb_id] = self._embed_pdb(pdb_path, tokenizer, model)

        # offtargets
        for offtarget_name in self.offtarget_names:
            key = f'offtarget_{offtarget_name}'
            if key not in self.plm_embeddings:
                pdb_path = os.path.join(self.offtarget_wt_dir, f'{offtarget_name.lower()}.pdb')
                self.plm_embeddings[key] = self._embed_pdb(pdb_path, tokenizer, model)

        if self.plm_embedding_path is not None:
            torch.save(self.plm_embeddings, self.plm_embedding_path)

    def _embed_pdb(self, pdb_path, tokenizer, model):
        complex_info = parse_pdb(pdb_path)
        chain_id_list = complex_info['chain_id']
        chain2sequence = parse_pdb_chain2sequence(pdb_path)

        assert (
            list(chain2sequence.keys())[0] == chain_id_list[0]
            and list(chain2sequence.keys())[-1] == chain_id_list[-1]
        ), "chain order mismatch!"

        embedding_list = []
        for _, seq in chain2sequence.items():
            inputs = tokenizer(seq, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                emb = outputs.last_hidden_state.squeeze(0)[1:len(seq) + 1, :].cpu()
            embedding_list.append(emb)

        embedding = torch.cat(embedding_list, dim=0)
        assert embedding.shape[0] == len(chain_id_list), 'PLM embedding length mismatch!'
        return embedding
