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

from model.MERF_v4 import MERF

from dataset.dataset_ddg import VMDataset, DDGBaseDataset
from utils.arguments import get_pretrain_args
from utils.util import *
from utils.losshistory import LossHistory

from protein.read_pdbs import PaddingCollate

from tqdm import tqdm

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from torch.utils.tensorboard import SummaryWriter


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_one_epoch(model, optimizer, epoch, n_epoch, epoch_size_train,
                    train_loader, device, use_kl_loss=False, writer=None):
    model.train()
    train_loss = train_mse_loss = train_kl_loss = 0

    with tqdm(total=epoch_size_train, desc=f'Epoch {epoch + 1}/{n_epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(train_loader):
            batch = recursive_to(batch, device)
            q_tot, loss, mse_loss, kl_loss = model(batch, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_mse_loss += mse_loss.item()
            train_kl_loss += kl_loss.item()

            if use_kl_loss:
                pbar.set_postfix(**{'total': loss.item(), 'mse': mse_loss.item(), 'kl': kl_loss.item(), 'lr': get_lr(optimizer)})
            else:
                pbar.set_postfix(**{'train_loss': loss.item(), 'lr': get_lr(optimizer)})
            pbar.update(1)

            if writer is not None and (iteration + 1) % 10 == 0:
                global_step = epoch * epoch_size_train + iteration
                writer.add_scalar('Step/train_total_loss', loss.item(), global_step)
                writer.add_scalar('Step/train_mse_loss', mse_loss.item(), global_step)
                writer.add_scalar('Step/train_kl_loss', kl_loss.item(), global_step)
                writer.add_scalar('Step/learning_rate', get_lr(optimizer), global_step)

    if use_kl_loss:
        print('Epoch %d/%d  Train Loss: %.4f (MSE: %.4f, KL: %.4f)' % (
            epoch + 1, n_epoch,
            train_loss / epoch_size_train,
            train_mse_loss / epoch_size_train,
            train_kl_loss / epoch_size_train))
    else:
        print('Epoch %d/%d  Train Loss: %.4f' % (epoch + 1, n_epoch, train_loss / epoch_size_train))

    save_path = os.path.join(loss_history.log_dir, f'Epoch%d_MERF.pth' % (epoch + 1))
    torch.save(model.state_dict(), save_path)

    if writer is not None:
        writer.add_scalar('Loss/train_total', train_loss / epoch_size_train, epoch)
        writer.add_scalar('Loss/train_mse', train_mse_loss / epoch_size_train, epoch)
        writer.add_scalar('Loss/train_kl', train_kl_loss / epoch_size_train, epoch)
        writer.add_scalar('LR/learning_rate', get_lr(optimizer), epoch)


def periodic_val(model, val_vm_loader, val_ab_loader, epoch, device, writer=None):
    """Run full per-sample validation on both val sets, print and log pearsonr + rmse."""
    model.eval()
    print(f'\n--- Periodic Validation (Epoch {epoch + 1}) ---')

    for tag, loader in [('VM', val_vm_loader), ('ABbind', val_ab_loader)]:
        results = val(model, loader, device)
        pear, p_val = pearsonr(results["Prediction"], results["Ground Truth"])
        rmse = np.sqrt(mean_squared_error(results["Prediction"], results["Ground Truth"]))
        print(f'  {tag}: pearson-r={pear:.4f} (p={p_val:.4e}), rmse={rmse:.4f}')
        if writer is not None:
            writer.add_scalar(f'PeriodicVal/{tag}_pearson_r', pear, epoch)
            writer.add_scalar(f'PeriodicVal/{tag}_rmse', rmse, epoch)

    print()


def val(model, val_loader, device):
    pre, label, PDB_id_list, mutation_info_list = [], [], [], []
    for batch in tqdm(val_loader, desc="Validation", ncols=80):
        labels = batch['ddG'].tolist()
        with torch.no_grad():
            batch = recursive_to(batch, device)
            preds, _, _, _ = model(batch, device)
            preds = preds.view(-1).tolist()
        pre.extend(preds)
        label.extend(labels)
        PDB_id_list.extend(batch['wt']['PDB_id'])
        mutation_info_list.extend(batch['wt']['mutate_info'])

    return pd.DataFrame({
        "PDB_id": PDB_id_list,
        "mutate_info": mutation_info_list,
        "Prediction": np.array(pre),
        "Ground Truth": np.array(label)
    })


if __name__ == '__main__':
    args = get_pretrain_args()
    print(args)

    seed_all(args.seed)
    print(f"setting random seed...{args.seed}")

    device = torch.device(args.gpu_idx if args.is_cuda else "cpu")

    loss_dir = "logs/pretrain_vm_v4/"
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

    train_vm_dataset = VMDataset(
        train_vm_path, wt_vm_dir, mut_vm_dir,
        knn_num=args.knn_neighbors_num, knn_agents_num=args.knn_agents_num,
        use_plm_embedding=args.use_plm_embedding, plm_path=args.plm_path,
        plm_embedding_path=train_vm_embedding_path, device=device,
        return_kl_target=args.use_kl_loss
    )
    val_vm_dataset = VMDataset(
        val_vm_path, wt_vm_dir, mut_vm_dir,
        knn_num=args.knn_neighbors_num, knn_agents_num=args.knn_agents_num,
        use_plm_embedding=args.use_plm_embedding, plm_path=args.plm_path,
        plm_embedding_path=val_vm_embedding_path, device=device,
        return_kl_target=args.use_kl_loss
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
        return_kl_target=args.use_kl_loss
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
        return_kl_target=args.use_kl_loss
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

    gen_val_vm = DataLoader(val_vm_dataset, shuffle=False, batch_size=args.batch_size, pin_memory=False,
                            drop_last=False, num_workers=args.num_works, collate_fn=collate_fn)
    gen_val_ab = DataLoader(val_ab_dataset, shuffle=False, batch_size=args.batch_size, pin_memory=False,
                            drop_last=False, num_workers=args.num_works, collate_fn=collate_fn)

    # ----------------------- Load PLM model for lm_head ----------------------- #
    print(f"Loading PLM model from {args.plm_path} for lm_head ...")
    plm_model = AutoModelForMaskedLM.from_pretrained(args.plm_path)
    plm_model.eval()
    # Freeze PLM weights — only lm_head is used and we keep it frozen too
    for param in plm_model.parameters():
        param.requires_grad = False

    # ----------------------- Build MERF v4 ----------------------- #
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
    print(f"KL Loss: {args.use_kl_loss}, weight: {args.kl_loss_weight}")
    for epoch in range(args.n_epoch):
        train_one_epoch(
            model, optimizer, epoch, args.n_epoch, epoch_size_train,
            gen_train, device, use_kl_loss=args.use_kl_loss, writer=writer
        )

        if (epoch + 1) % 10 == 0:
            periodic_val(model, gen_val_vm, gen_val_ab, epoch, device, writer=writer)

        if args.use_cosine_lr:
            lr_scheduler.step()
        else:
            if epoch == 10:
                for p in optimizer.param_groups:
                    p['lr'] = 1e-5

    writer.close()
    print(f"\nTensorBoard logs saved to: {loss_history.log_dir}")
