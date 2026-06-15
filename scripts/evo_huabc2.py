import datetime
import os
import sys

import pandas as pd
import torch
import torch.optim as optim
from Bio.PDB import MMCIFParser, PDBIO
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dataset_evo import EvoBaseDataset
from model.MERF_v6 import MERF
from protein.read_pdbs import PaddingCollate
from scripts.evo_sabdab import (
    PYROSETTA_TOP_K,
    evaluate_top_candidates_with_pyrosetta,
    load_model_args_from_checkpoint,
    train_one_epoch,
    update_candidate_scores,
)
from utils.arguments import get_evolution_args
from utils.losshistory import LossHistory
from utils.util import seed_all


cpu_num = 32
torch.set_num_threads(cpu_num)
print(cpu_num)


def ensure_pdb_from_cif(pdb_id, pdb_dir):
    pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
    cif_path = os.path.join(pdb_dir, f"{pdb_id}.cif")

    if os.path.exists(pdb_path):
        return pdb_path
    if not os.path.exists(cif_path):
        raise FileNotFoundError(f'Neither {pdb_path} nor {cif_path} exists.')

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(pdb_id, cif_path)
    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_path)
    return pdb_path


if __name__ == '__main__':
    args = get_evolution_args()
    checkpoint_args_path, checkpoint_model_arg_keys = load_model_args_from_checkpoint(args)
    print(args)

    seed = args.seed
    seed_all(seed)
    print(f"setting random seed...{seed}")

    device = torch.device(f"cuda:{args.gpu_idx}" if args.is_cuda else "cpu")

    loss_dir = "logs/evo_huabc2/"
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S')
    loss_dir = os.path.join(loss_dir, time_str)

    huabc2_dir = '/home/dataset-local/projects_dir/MERF/data/HuABC2'
    train_path = os.path.join(huabc2_dir, 'huabc2_evo.csv')
    train_df = pd.read_csv(train_path, dtype={"pdb_id": "string"})

    wt_dir = os.path.join(huabc2_dir, 'PDBs')
    fix_dir = os.path.join(huabc2_dir, 'PDBs_fixed')
    mut_dir = os.path.join(huabc2_dir, 'PDBs_evo')
    plm_embedding_path = os.path.join(huabc2_dir, 'PLM_embeddings_huabc2.pkl')
    os.makedirs(fix_dir, exist_ok=True)
    os.makedirs(mut_dir, exist_ok=True)

    if args.dataset_idx is not None:
        if args.dataset_idx < 0 or args.dataset_idx >= len(train_df):
            raise IndexError(f'dataset_idx {args.dataset_idx} is out of range for {train_path} with {len(train_df)} rows')
        dataset_indices = [args.dataset_idx]
        print(f'Only evolving HuABC2 dataset row {args.dataset_idx}.')
    else:
        dataset_indices = range(len(train_df))

    for i in dataset_indices:
        pdb_id = train_df['pdb_id'].iloc[i].replace('+', '').replace('.00', '')
        antibody_chain = train_df['antibody_chain'].iloc[i]
        partner = train_df['partner'].iloc[i]
        sequence = train_df['sequence'].iloc[i]
        cdrs = [train_df['cdr1'].iloc[i], train_df['cdr2'].iloc[i], train_df['cdr3'].iloc[i]]
        cdr = cdrs[2]

        ensure_pdb_from_cif(pdb_id, wt_dir)

        print(f'Start evolving {pdb_id}...')
        print(f'Antibody chain: {antibody_chain}, Partner: {partner}, CDRH3: {cdr}')
        print(f'Sequence length: {len(sequence)}; sequence: {sequence}')
        print(f'CDRH3 length: {len(cdr)}, CDRH3 sequence: {cdr}')
        print('----------------------------------------')

        train_dataset = EvoBaseDataset(
            pdb_id, antibody_chain, partner, sequence, cdr,
            wt_dir, fix_dir, mut_dir,
            knn_num=args.knn_neighbors_num, knn_agents_num=args.knn_agents_num,
            use_plm_embedding=args.use_plm_embedding, plm_path=args.plm_path, plm_embedding_path=plm_embedding_path,
            device=device
        )

        collate_fn = PaddingCollate()

        this_loss_dir = os.path.join(loss_dir, pdb_id)
        loss_history = LossHistory(this_loss_dir, is_evolve=True)
        tb_writer = SummaryWriter(log_dir=this_loss_dir)
        loss_history.write(str(args) + '\n')
        if checkpoint_args_path is not None:
            loss_history.write(f'Loaded model args from {checkpoint_args_path}\n')
            loss_history.write(f'Overrode model args: {", ".join(checkpoint_model_arg_keys)}\n')

        try:
            tb_writer.add_text('run/args', str(args), 0)
            if checkpoint_args_path is not None:
                tb_writer.add_text('run/checkpoint_args_path', checkpoint_args_path, 0)
                tb_writer.add_text('run/overrode_model_args', ', '.join(checkpoint_model_arg_keys), 0)
            tb_writer.add_text('data/pdb_id', pdb_id, 0)
            tb_writer.add_text('data/cdrh3', cdr, 0)
            tb_writer.add_scalar('data/dataset_idx', i, 0)

            evo_model = MERF(args).to(device)
            evo_model.load_state_dict(torch.load(args.model_load_path, map_location=device))

            optimizer = optim.Adam(evo_model.parameters(), lr=args.lr, weight_decay=0.1)

            reference_model = MERF(args).to(device)
            reference_model.load_state_dict(evo_model.state_dict())
            reference_model.eval()

            reward_model = MERF(args).to(device)
            reward_model.load_state_dict(torch.load(args.model_load_path, map_location=device))
            reward_model.eval()

            loss_history.write(f'\nLoading model from {args.model_load_path}\n')

            total_epoch = args.n_epoch
            best_candidate_scores = {}
            for epoch in range(total_epoch):
                epoch_metrics = train_one_epoch(
                    args, evo_model, reward_model, reference_model, optimizer, epoch, total_epoch,
                    train_dataset, collate_fn, loss_history, device, tb_writer=tb_writer
                )
                update_candidate_scores(best_candidate_scores, epoch_metrics['candidate_scores'])

            evaluate_top_candidates_with_pyrosetta(
                pdb_id, partner, wt_dir, fix_dir, mut_dir, best_candidate_scores,
                this_loss_dir, loss_history, tb_writer=tb_writer, top_k=PYROSETTA_TOP_K
            )
        finally:
            tb_writer.flush()
            tb_writer.close()
