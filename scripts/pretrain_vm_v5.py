import os
import numpy as np
import pandas as pd
import pickle

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torch.utils.data import DataLoader

from model.MERF_v6 import MERF

from dataset.dataset_ddg_v5 import VMDataset, VMSiteDataset, DDGBaseDataset, CRDataset
from utils.arguments import get_pretrain_args
from utils.util import *
from utils.losshistory import LossHistory

from protein.read_pdbs import PaddingCollate

from tqdm import tqdm

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from torch.utils.tensorboard import SummaryWriter

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_one_epoch(model, optimizer, epoch, n_epoch, epoch_size_train, train_loader, device, use_kl_loss=False, writer=None):
    model.train()
    train_loss = train_mse_loss = train_kl_loss = 0

    with tqdm(total=epoch_size_train, desc=f'Epoch {epoch + 1}/{n_epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(train_loader):
            batch = recursive_to(batch, device)
            q_tot_wt, loss, mse_loss, kl_loss = model(batch, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_mse_loss += mse_loss.item()
            train_kl_loss += kl_loss.item()

            if use_kl_loss:
                pbar.set_postfix(**{'total_loss': loss.item(), 'mse': mse_loss.item(), 'kl': kl_loss.item(), 'lr': get_lr(optimizer)})
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


def periodic_val(model, val_vm_loader, val_ab_loader, epoch, device, writer=None,
                 cr_6261_h1_loader=None, cr_6261_h9_loader=None, fae7_loader=None):
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

    # CR datasets: spearman-r and pearson-r only
    for tag, loader in [('CR_6261_H1', cr_6261_h1_loader), ('CR_6261_H9', cr_6261_h9_loader)]:
        if loader is None:
            continue
        results = val(model, loader, device)
        pred = results["Prediction"]
        gt = results["Ground Truth"]
        valid_mask = ~(np.isnan(gt) | np.isnan(pred) | np.isinf(gt) | np.isinf(pred))
        pred_v, gt_v = pred[valid_mask], gt[valid_mask]
        pear, _ = pearsonr(pred_v, gt_v)
        spear, _ = spearmanr(pred_v, gt_v)
        print(f'  {tag}: pearson-r={pear:.4f}, spearman-r={spear:.4f}')
        if writer is not None:
            writer.add_scalar(f'PeriodicVal/{tag}_pearson_r', pear, epoch)
            writer.add_scalar(f'PeriodicVal/{tag}_spearman_r', spear, epoch)

    # 7FAE dataset: avg_rank of key mutations
    if fae7_loader is not None:
        key_mutations = ['TH31W', 'AH53F', 'NH57L', 'RH103M', 'LH104F']
        predictions, pdb_ids, mutation_infos = [], [], []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(fae7_loader, desc="Val 7FAE", ncols=80):
                batch = recursive_to(batch, device)
                pred, _, _, _ = model(batch, device)
                if pred.dim() == 0:
                    predictions.append(pred.item())
                else:
                    predictions.extend(pred.cpu().tolist())
                pdb_ids.extend(batch['wt']['PDB_id'])
                mutation_infos.extend(batch['wt']['mutate_info'])

        fae_df = pd.DataFrame({'mutate_info': mutation_infos, 'Prediction': predictions})
        fae_df['rank'] = fae_df['Prediction'].rank(ascending=True, method='min').astype(int)
        valid_ranks = []
        for key_mut in key_mutations:
            mask = fae_df['mutate_info'] == key_mut
            if mask.any():
                valid_ranks.append(fae_df.loc[mask, 'rank'].values[0])
        avg_rank = np.mean(valid_ranks) if valid_ranks else -1
        print(f'  7FAE: avg_rank={avg_rank:.2f} (found {len(valid_ranks)}/{len(key_mutations)} key mutations)')
        if writer is not None:
            writer.add_scalar('PeriodicVal/7FAE_avg_rank', avg_rank, epoch)

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

    return pd.DataFrame({"PDB_id": PDB_id_list, "mutate_info": mutation_info_list,
                         "Prediction": np.array(pre), "Ground Truth": np.array(label)})


if __name__ == '__main__':
    # ----------------------- environment setting ----------------------- #
    args = get_pretrain_args()
    print(args)

    seed = args.seed
    seed_all(seed)
    print(f"setting random seed...{seed}")

    GPU_indx = args.gpu_idx
    device = torch.device(GPU_indx if args.is_cuda else "cpu")

    loss_dir = "logs/pretrain_vm/"
    loss_history = LossHistory(loss_dir)
    loss_history.write(str(args) + '\n')
    start_time_str = loss_history.get_str()

    # Initialize TensorBoard writer (same directory as loss_history)
    writer = SummaryWriter(log_dir=loss_history.log_dir)
    print(f"TensorBoard log directory: {loss_history.log_dir}")

    # Save args for later testing
    args_save_path = os.path.join(loss_history.log_dir, 'args.pkl')
    with open(args_save_path, 'wb') as f:
        pickle.dump(args, f)
    print(f"Args saved to: {args_save_path}")

    # ----------------------- dataset ----------------------- #
    # read in VM train and valid sets
    train_vm_path = "/home/dataset-local/projects_dir/MERF/data/VenusMaxwell/train_data.csv"
    val_vm_path = "/home/dataset-local/projects_dir/MERF/data/VenusMaxwell/valid_data.csv"
    wt_vm_dir = "/home/dataset-local/projects_dir/MERF/data/VenusMaxwell/PDBs_fixed"
    mut_vm_dir = "/home/dataset-local/projects_dir/MERF/data/VenusMaxwell/PDBs_mutated"
    train_vm_embedding_path = "/home/dataset-local/projects_dir/MERF/data/VenusMaxwell/train_esm2_650_embeddings.pkl"
    val_vm_embedding_path = "/home/dataset-local/projects_dir/MERF/data/VenusMaxwell/valid_esm2_650_embeddings.pkl"

    train_vm_dataset = VMSiteDataset(
        train_vm_path, wt_vm_dir, mut_vm_dir,
        knn_num=args.knn_neighbors_num, knn_agents_num=args.knn_agents_num,
        use_plm_embedding=args.use_plm_embedding, plm_path=args.plm_path, plm_embedding_path=train_vm_embedding_path, device=device,
        return_kl_target=args.use_kl_loss
    )
    val_vm_dataset = VMDataset(
        val_vm_path, wt_vm_dir, mut_vm_dir,
        knn_num=args.knn_neighbors_num, knn_agents_num=args.knn_agents_num,
        use_plm_embedding=args.use_plm_embedding, plm_path=args.plm_path, plm_embedding_path=val_vm_embedding_path, device=device,
        return_kl_target=args.use_kl_loss
    )

    # read in skempiv2 as extra train_set
    train_sk_path = '/home/dataset-local/projects_dir/MERF/data/SKEMPIv2/SKEMPIv2.csv'
    train_sk_wt_dir = '/home/dataset-local/projects_dir/MERF/data/SKEMPIv2/PDBs_fixed'
    train_sk_mut_dir = '/home/dataset-local/projects_dir/MERF/data/SKEMPIv2/PDBs_mutated'
    sk_embedding_path = '/home/dataset-local/projects_dir/MERF/data/SKEMPIv2/skempiv2_esm2_650_embeddings.pkl'

    train_sk_dataset = DDGBaseDataset(
        train_sk_path, train_sk_wt_dir, train_sk_mut_dir,
        knn_num=args.knn_neighbors_num, knn_agents_num=args.knn_agents_num,
        use_plm_embedding=args.use_plm_embedding, plm_path=args.plm_path, plm_embedding_path=sk_embedding_path, device=device,
        return_kl_target=args.use_kl_loss
    )

    # read in abbind as extra val_set
    val_ab_path = '/home/dataset-local/projects_dir/MERF/data/ABbind/AB-Bind_645pMulti.csv'
    val_ab_wt_dir = '/home/dataset-local/projects_dir/MERF/data/ABbind/PDBs_fixed'
    val_ab_mut_dir = '/home/dataset-local/projects_dir/MERF/data/ABbind/PDBs_mutated'
    ab_embedding_path = '/home/dataset-local/projects_dir/MERF/data/ABbind/abbind_esm2_650_embeddings.pkl'

    val_ab_dataset = DDGBaseDataset(
        val_ab_path, val_ab_wt_dir, val_ab_mut_dir,
        knn_num=args.knn_neighbors_num, knn_agents_num=args.knn_agents_num,
        use_plm_embedding=args.use_plm_embedding, plm_path=args.plm_path, plm_embedding_path=ab_embedding_path, device=device,
        return_kl_target=args.use_kl_loss
    )
    
    # combine datasets
    train_dataset = torch.utils.data.ConcatDataset([train_vm_dataset, train_sk_dataset])
    val_dataset = torch.utils.data.ConcatDataset([val_vm_dataset, val_ab_dataset])

    print(f"Train dataset sizes - VM sites: {len(train_vm_dataset)}, SK: {len(train_sk_dataset)}, Total: {len(train_dataset)}")
    print(f"Val dataset sizes - VM: {len(val_vm_dataset)}, AB: {len(val_ab_dataset)}, Total: {len(val_dataset)}")

    # create dataloader
    collate_fn = PaddingCollate()

    data_generator = torch.Generator()
    data_generator.manual_seed(170)
    print("Using site-level VM training: each epoch visits every VM site once and SKEMPIv2 once")
    print("Training DataLoader uses deterministic shuffle with fixed data seed 170")
    gen_train = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=False,
                           drop_last=False, num_workers=args.num_works, collate_fn=collate_fn,
                           generator=data_generator)
    epoch_size_train = len(gen_train)

    gen_val_vm = DataLoader(val_vm_dataset, shuffle=False, batch_size=args.batch_size, pin_memory=False,
                            drop_last=False, num_workers=args.num_works, collate_fn=collate_fn)
    gen_val_ab = DataLoader(val_ab_dataset, shuffle=False, batch_size=args.batch_size, pin_memory=False,
                            drop_last=False, num_workers=args.num_works, collate_fn=collate_fn)

    # CR and 7FAE test sets for periodic_val
    cr_base_wt_dir = '/home/dataset-local/projects_dir/MERF/data/CR/PDBs_fixed'
    cr_base_mut_dir = '/home/dataset-local/projects_dir/MERF/data/CR/PDBs_mutated'

    val_cr_h1_dataset = CRDataset(
        '/home/dataset-local/projects_dir/MERF/data/CR/cr6261_h1.csv',
        cr_base_wt_dir, cr_base_mut_dir,
        knn_num=args.knn_neighbors_num, knn_agents_num=args.knn_agents_num,
        use_plm_embedding=args.use_plm_embedding, plm_path=args.plm_path,
        plm_embedding_path='/home/dataset-local/projects_dir/MERF/data/CR/cr6261_h1_esm2_650_embeddings.pkl',
        device=device, target_column='h1_score'
    )
    val_cr_h9_dataset = CRDataset(
        '/home/dataset-local/projects_dir/MERF/data/CR/cr6261_h9.csv',
        cr_base_wt_dir, cr_base_mut_dir,
        knn_num=args.knn_neighbors_num, knn_agents_num=args.knn_agents_num,
        use_plm_embedding=args.use_plm_embedding, plm_path=args.plm_path,
        plm_embedding_path='/home/dataset-local/projects_dir/MERF/data/CR/cr6261_h9_esm2_650_embeddings.pkl',
        device=device, target_column='h9_score'
    )
    val_7fae_dataset = DDGBaseDataset(
        '/home/dataset-local/projects_dir/MERF/data/7FAE/7FAE.csv',
        '/home/dataset-local/projects_dir/MERF/data/7FAE/PDBs_fixed',
        '/home/dataset-local/projects_dir/MERF/data/7FAE/PDBs_mutated',
        knn_num=args.knn_neighbors_num, knn_agents_num=args.knn_agents_num,
        use_plm_embedding=args.use_plm_embedding, plm_path=args.plm_path,
        plm_embedding_path='/home/dataset-local/projects_dir/MERF/data/7FAE/7fae_esm2_650_embeddings.pkl',
        device=device
    )

    gen_val_cr_h1 = DataLoader(val_cr_h1_dataset, shuffle=False, batch_size=args.batch_size, pin_memory=False,
                               drop_last=False, num_workers=args.num_works, collate_fn=collate_fn)
    gen_val_cr_h9 = DataLoader(val_cr_h9_dataset, shuffle=False, batch_size=args.batch_size, pin_memory=False,
                               drop_last=False, num_workers=args.num_works, collate_fn=collate_fn)
    gen_val_7fae = DataLoader(val_7fae_dataset, shuffle=False, batch_size=args.batch_size, pin_memory=False,
                              drop_last=False, num_workers=args.num_works, collate_fn=collate_fn)

    # ----------------------- Initial Networks and optimizer ----------------------- #
    model = MERF(args).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # ----------------------- Learning Rate Scheduler ----------------------- #
    if args.use_cosine_lr:
        # Cosine annealing with optional warmup
        if args.cosine_warmup_epochs > 0:
            # Warmup scheduler: linearly increase from min_lr to lr
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=args.cosine_min_lr / args.lr,  # start from min_lr
                end_factor=1.0,  # end at lr
                total_iters=args.cosine_warmup_epochs
            )
            # Cosine scheduler after warmup
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=args.n_epoch - args.cosine_warmup_epochs,
                eta_min=args.cosine_min_lr
            )
            # Combine warmup and cosine
            lr_scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[args.cosine_warmup_epochs]
            )
            print(f"Using Cosine LR scheduler with {args.cosine_warmup_epochs} warmup epochs")
            print(f"  - Initial LR: {args.lr}, Min LR: {args.cosine_min_lr}")
        else:
            # Pure cosine annealing without warmup
            lr_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=args.n_epoch,
                eta_min=args.cosine_min_lr
            )
            print(f"Using Cosine LR scheduler (no warmup)")
            print(f"  - Initial LR: {args.lr}, Min LR: {args.cosine_min_lr}")
    else:
        print("Using custom LR adjustment (manual)")

    # ----------------------- fit epoches ----------------------- #
    print(f"KL Loss enabled: {args.use_kl_loss}, weight: {args.kl_loss_weight}")
    for epoch in range(args.n_epoch):
        train_vm_dataset.set_epoch(epoch)
        train_one_epoch(model, optimizer, epoch, args.n_epoch, epoch_size_train, gen_train, device, use_kl_loss=args.use_kl_loss, writer=writer)

        if (epoch + 1) % 10 == 0:
            periodic_val(model, gen_val_vm, gen_val_ab, epoch, device, writer=writer,
                         cr_6261_h1_loader=gen_val_cr_h1, cr_6261_h9_loader=gen_val_cr_h9,
                         fae7_loader=gen_val_7fae)

        # Update learning rate scheduler
        if args.use_cosine_lr:
            lr_scheduler.step()
        else:
            if epoch == 10:
                for p in optimizer.param_groups:
                    p['lr'] = 1e-5

    writer.close()
    print(f"\nTensorBoard logs saved to: {loss_history.log_dir}")
    
