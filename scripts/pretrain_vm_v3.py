import os
import numpy as np
import pandas as pd
import pickle

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoModelForMaskedLM

from model.MERF_v3 import MERF

from dataset.dataset_ddg_v2 import VMDatasetV2, DDGBaseDataset
from utils.arguments import get_pretrain_args
from utils.util import *
from utils.losshistory import LossHistory

from protein.read_pdbs import PaddingCollate2 as PaddingCollate

from tqdm import tqdm

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from torch.utils.tensorboard import SummaryWriter


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_one_epoch(model, optimizer, epoch, n_epoch, epoch_size_train, epoch_size_val,
                    train_loader, val_loader, loss_history, device, use_rank_loss=False, writer=None):
    print("Start Train")
    model.train()

    train_loss = train_mse_loss = train_rank_loss = val_loss = 0

    with tqdm(total=epoch_size_train, desc=f'Epoch {epoch + 1}/{n_epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(train_loader):
            batch = recursive_to(batch, device)
            q_tot, loss, mse_loss, rank_loss = model(batch, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_mse_loss += mse_loss.item()
            train_rank_loss += rank_loss.item()

            if use_rank_loss:
                pbar.set_postfix(**{'total': loss.item(), 'mse': mse_loss.item(), 'rank': rank_loss.item(), 'lr': get_lr(optimizer)})
            else:
                pbar.set_postfix(**{'train_loss': loss.item(), 'lr': get_lr(optimizer)})
            pbar.update(1)

            if writer is not None and (iteration + 1) % 10 == 0:
                global_step = epoch * epoch_size_train + iteration
                writer.add_scalar('Step/train_total_loss', loss.item(), global_step)
                writer.add_scalar('Step/train_mse_loss', mse_loss.item(), global_step)
                writer.add_scalar('Step/train_rank_loss', rank_loss.item(), global_step)
                writer.add_scalar('Step/learning_rate', get_lr(optimizer), global_step)

    print('Finish Train')
    print('Start Validation')
    model.eval()

    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{n_epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(val_loader):
            batch = recursive_to(batch, device)
            ddG = batch['ddG'].to(device).float()
            with torch.no_grad():
                q_tot, _, _, _ = model(batch, device)
                loss = torch.nn.MSELoss()(q_tot, ddG)
                val_loss += loss.item()
                pbar.set_postfix(**{'val_loss': loss.item()})
                pbar.update(1)

    loss_history.append_loss(train_loss / epoch_size_train, val_loss / epoch_size_val)
    print('Finish Validation')
    print(f'Epoch: {epoch + 1}/{n_epoch}')
    if use_rank_loss:
        print('Train Loss: %.4f (MSE: %.4f, Rank: %.4f) || Val Loss: %.4f' % (
            train_loss / epoch_size_train, train_mse_loss / epoch_size_train,
            train_rank_loss / epoch_size_train, val_loss / epoch_size_val))
    else:
        print('Train Loss: %.4f || Val Loss: %.4f' % (train_loss / epoch_size_train, val_loss / epoch_size_val))

    save_path = os.path.join(loss_history.log_dir, f'Epoch%d_MERF_v3.pth' % (epoch + 1))
    torch.save(model.state_dict(), save_path)

    if writer is not None:
        writer.add_scalar('Loss/train_total', train_loss / epoch_size_train, epoch)
        writer.add_scalar('Loss/train_mse', train_mse_loss / epoch_size_train, epoch)
        writer.add_scalar('Loss/train_rank', train_rank_loss / epoch_size_train, epoch)
        writer.add_scalar('Loss/val', val_loss / epoch_size_val, epoch)
        writer.add_scalar('LR/learning_rate', get_lr(optimizer), epoch)

    return val_loss / epoch_size_val


def val(model, val_loader, device):
    pre, label, PDB_id_list, mutation_info_list = [], [], [], []
    for batch in tqdm(val_loader, desc="Validation", ncols=80):
        label.append(batch['ddG'].item())
        with torch.no_grad():
            batch = recursive_to(batch, device)
            pred, _, _, _ = model(batch, device)
            pre.append(pred.item())
        PDB_id_list.append(batch['wt']['PDB_id'][0])
        mutation_info_list.append(batch['wt']['mutate_info'][0])

    return pd.DataFrame({
        "PDB_id": np.array(PDB_id_list),
        "mutate_info": np.array(mutation_info_list),
        "Prediction": np.array(pre),
        "Ground Truth": np.array(label)
    })


if __name__ == '__main__':
    args = get_pretrain_args()
    print(args)

    seed_all(args.seed)
    print(f"setting random seed...{args.seed}")

    device = torch.device(args.gpu_idx if args.is_cuda else "cpu")

    loss_dir = "logs/pretrain_vm_v3/"
    loss_history = LossHistory(loss_dir)
    loss_history.write(str(args) + '\n')

    writer = SummaryWriter(log_dir=loss_history.log_dir)
    print(f"TensorBoard log directory: {loss_history.log_dir}")

    args_save_path = os.path.join(loss_history.log_dir, 'args.pkl')
    with open(args_save_path, 'wb') as f:
        pickle.dump(args, f)

    # ----------------------- dataset ----------------------- #
    train_vm_path = "/home/dataset-local/projects_dir/MERF/data/VenusMaxwell/train_data.csv"
    val_vm_path = "/home/dataset-local/projects_dir/MERF/data/VenusMaxwell/valid_data.csv"
    wt_vm_dir = "/home/dataset-local/projects_dir/MERF/data/VenusMaxwell/PDBs_fixed"
    mut_vm_dir = "/home/dataset-local/projects_dir/MERF/data/VenusMaxwell/PDBs_mutated"
    train_vm_embedding_path = "/home/dataset-local/projects_dir/MERF/data/VenusMaxwell/train_esm2_650_embeddings.pkl"
    val_vm_embedding_path = "/home/dataset-local/projects_dir/MERF/data/VenusMaxwell/valid_esm2_650_embeddings.pkl"

    train_vm_dataset = VMDatasetV2(
        train_vm_path, wt_vm_dir, mut_vm_dir,
        knn_num=args.knn_neighbors_num, knn_agents_num=args.knn_agents_num,
        use_plm_embedding=args.use_plm_embedding, plm_path=args.plm_path,
        plm_embedding_path=train_vm_embedding_path, device=device,
        return_rank_target=args.use_rank_loss
    )
    val_vm_dataset = VMDatasetV2(
        val_vm_path, wt_vm_dir, mut_vm_dir,
        knn_num=args.knn_neighbors_num, knn_agents_num=args.knn_agents_num,
        use_plm_embedding=args.use_plm_embedding, plm_path=args.plm_path,
        plm_embedding_path=val_vm_embedding_path, device=device,
        return_rank_target=args.use_rank_loss
    )

    train_sk_path = '/home/dataset-local/projects_dir/MERF/data/SKEMPIv2/SKEMPIv2.csv'
    train_sk_wt_dir = '/home/dataset-local/projects_dir/MERF/data/SKEMPIv2/PDBs_fixed'
    train_sk_mut_dir = '/home/dataset-local/projects_dir/MERF/data/SKEMPIv2/PDBs_mutated'
    sk_embedding_path = '/home/dataset-local/projects_dir/MERF/data/SKEMPIv2/skempiv2_esm2_650_embeddings.pkl'

    train_sk_dataset = DDGBaseDataset(
        train_sk_path, train_sk_wt_dir, train_sk_mut_dir,
        knn_num=args.knn_neighbors_num, knn_agents_num=args.knn_agents_num,
        use_plm_embedding=args.use_plm_embedding, plm_path=args.plm_path,
        plm_embedding_path=sk_embedding_path, device=device,
        return_kl_target=args.use_rank_loss
    )

    val_ab_path = '/home/dataset-local/projects_dir/MERF/data/ABbind/AB-Bind_645pMulti.csv'
    val_ab_wt_dir = '/home/dataset-local/projects_dir/MERF/data/ABbind/PDBs_fixed'
    val_ab_mut_dir = '/home/dataset-local/projects_dir/MERF/data/ABbind/PDBs_mutated'
    ab_embedding_path = '/home/dataset-local/projects_dir/MERF/data/ABbind/abbind_esm2_650_embeddings.pkl'

    val_ab_dataset = DDGBaseDataset(
        val_ab_path, val_ab_wt_dir, val_ab_mut_dir,
        knn_num=args.knn_neighbors_num, knn_agents_num=args.knn_agents_num,
        use_plm_embedding=args.use_plm_embedding, plm_path=args.plm_path,
        plm_embedding_path=ab_embedding_path, device=device,
        return_kl_target=args.use_rank_loss
    )

    train_dataset = torch.utils.data.ConcatDataset([train_vm_dataset, train_sk_dataset])
    val_dataset = torch.utils.data.ConcatDataset([val_vm_dataset, val_ab_dataset])

    print(f"Train dataset sizes - VM: {len(train_vm_dataset)}, SK: {len(train_sk_dataset)}, Total: {len(train_dataset)}")
    print(f"Val dataset sizes - VM: {len(val_vm_dataset)}, AB: {len(val_ab_dataset)}, Total: {len(val_dataset)}")

    epoch_size_train = len(train_dataset) // args.batch_size
    epoch_size_val = len(val_dataset) // args.batch_size

    collate_fn = PaddingCollate()

    if args.use_weighted_sampling:
        len_vm, len_sk = len(train_vm_dataset), len(train_sk_dataset)
        weight_per_vm = args.vm_sampling_weight / len_vm
        weight_per_sk = args.sk_sampling_weight / len_sk
        sample_weights = [weight_per_vm] * len_vm + [weight_per_sk] * len_sk

        if args.max_repetition_factor > 0:
            smaller = min(len_vm, len_sk)
            num_samples = min(len(train_dataset), int(smaller * args.max_repetition_factor * 2))
        else:
            num_samples = len(train_dataset)

        train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_samples, replacement=True)
        epoch_size_train = num_samples // args.batch_size
        print(f"Using weighted sampling — VM: {args.vm_sampling_weight}, SK: {args.sk_sampling_weight}, samples/epoch: {num_samples}")
        gen_train = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size,
                               pin_memory=False, drop_last=True, num_workers=args.num_works, collate_fn=collate_fn)
    else:
        gen_train = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                               pin_memory=False, drop_last=True, num_workers=args.num_works, collate_fn=collate_fn)

    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size,
                         pin_memory=False, drop_last=True, num_workers=args.num_works, collate_fn=collate_fn)

    # ----------------------- Load PLM model for lm_head ----------------------- #
    print(f"Loading PLM model from {args.plm_path} for lm_head ...")
    plm_model = AutoModelForMaskedLM.from_pretrained(args.plm_path)
    plm_model.eval()
    # Freeze PLM weights — only lm_head is used and we keep it frozen too
    for param in plm_model.parameters():
        param.requires_grad = False

    # ----------------------- Build MERF v3 ----------------------- #
    model = MERF(args, plm_model=plm_model).to(device)

    # Only optimize structure branch parameters (PLM lm_head is frozen)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )

    # ----------------------- LR Scheduler ----------------------- #
    if args.use_cosine_lr:
        if args.cosine_warmup_epochs > 0:
            warmup_scheduler = LinearLR(optimizer, start_factor=args.cosine_min_lr / args.lr,
                                        end_factor=1.0, total_iters=args.cosine_warmup_epochs)
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epoch - args.cosine_warmup_epochs,
                                                 eta_min=args.cosine_min_lr)
            lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                                        milestones=[args.cosine_warmup_epochs])
            print(f"Cosine LR with {args.cosine_warmup_epochs} warmup epochs, min_lr={args.cosine_min_lr}")
        else:
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epoch, eta_min=args.cosine_min_lr)
            print(f"Cosine LR (no warmup), min_lr={args.cosine_min_lr}")

    # ----------------------- Training loop ----------------------- #
    print(f"Rank Loss: {args.use_rank_loss}, weight: {args.rank_loss_weight}")
    for epoch in range(args.n_epoch):
        val_loss = train_one_epoch(
            model, optimizer, epoch, args.n_epoch, epoch_size_train, epoch_size_val,
            gen_train, gen_val, loss_history, device,
            use_rank_loss=args.use_rank_loss, writer=writer
        )
        if args.use_cosine_lr:
            lr_scheduler.step()
        else:
            if epoch == 10:
                for p in optimizer.param_groups:
                    p['lr'] = 1e-5

    # ----------------------- Final validation ----------------------- #
    gen_val_vm = DataLoader(val_vm_dataset, shuffle=False, batch_size=1, pin_memory=False,
                            drop_last=False, num_workers=args.num_works, collate_fn=collate_fn)
    print("\n" + "="*80 + "\nVM Validation Set Results:\n" + "="*80)
    results_vm = val(model, gen_val_vm, device)
    pear_vm, p_vm = pearsonr(results_vm["Prediction"], results_vm["Ground Truth"])
    rmse_vm = np.sqrt(mean_squared_error(results_vm["Prediction"], results_vm["Ground Truth"]))
    print(f'VM pearson-r={pear_vm:.4f} (p={p_vm:.4e}), rmse={rmse_vm:.4f}')

    gen_val_ab = DataLoader(val_ab_dataset, shuffle=False, batch_size=1, pin_memory=False,
                            drop_last=False, num_workers=args.num_works, collate_fn=collate_fn)
    print("\n" + "="*80 + "\nAB-Bind Validation Set Results:\n" + "="*80)
    results_ab = val(model, gen_val_ab, device)
    pear_ab, p_ab = pearsonr(results_ab["Prediction"], results_ab["Ground Truth"])
    rmse_ab = np.sqrt(mean_squared_error(results_ab["Prediction"], results_ab["Ground Truth"]))
    print(f'AB-Bind pearson-r={pear_ab:.4f} (p={p_ab:.4e}), rmse={rmse_ab:.4f}')

    writer.add_scalar('Metrics/VM_pearson_r', pear_vm, args.n_epoch)
    writer.add_scalar('Metrics/VM_rmse', rmse_vm, args.n_epoch)
    writer.add_scalar('Metrics/ABbind_pearson_r', pear_ab, args.n_epoch)
    writer.add_scalar('Metrics/ABbind_rmse', rmse_ab, args.n_epoch)
    writer.close()
    print(f"\nTensorBoard logs saved to: {loss_history.log_dir}")
