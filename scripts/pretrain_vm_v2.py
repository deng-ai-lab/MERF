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

from model.MERF_v2 import MERF

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


def train_one_epoch(model, optimizer, epoch, n_epoch, epoch_size_train, epoch_size_val, train_loader, val_loader, loss_history, device, use_rank_loss=False, writer=None):
    # ----------------------- Train ----------------------- #
    print("Start Train")
    model.train()

    train_loss = 0
    train_mse_loss = 0
    train_rank_loss = 0
    val_loss = 0

    with tqdm(total=epoch_size_train, desc=f'Epoch {epoch + 1}/{n_epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(train_loader):

            batch = recursive_to(batch, device)
            q_tot_wt, loss, mse_loss, rank_loss = model(batch, device)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
            train_mse_loss += mse_loss.item()
            train_rank_loss += rank_loss.item()

            if use_rank_loss:
                pbar.set_postfix(**{'total_loss': loss.item(), 'mse': mse_loss.item(), 'rank': rank_loss.item(), 'lr': get_lr(optimizer)})
            else:
                pbar.set_postfix(**{'train_loss': loss.item(), 'lr': get_lr(optimizer)})
            pbar.update(1)

            # Log to TensorBoard every 10 steps
            if writer is not None and (iteration + 1) % 10 == 0:
                global_step = epoch * epoch_size_train + iteration
                writer.add_scalar('Step/train_total_loss', loss.item(), global_step)
                writer.add_scalar('Step/train_mse_loss', mse_loss.item(), global_step)
                writer.add_scalar('Step/train_rank_loss', rank_loss.item(), global_step)
                writer.add_scalar('Step/learning_rate', get_lr(optimizer), global_step)

    print('Finish Train')

    # ----------------------- Validation ----------------------- #
    print('Start Validation')

    model.eval()

    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{n_epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(val_loader):
            batch = recursive_to(batch, device)

            ddG = batch['ddG'].to(device)
            ddG = ddG.to(torch.float32)

            with torch.no_grad():
                q_tot_wt, _, _, _ = model(batch, device)

                loss_fn = torch.nn.MSELoss()

                loss = loss_fn(q_tot_wt, ddG)

                val_loss += loss.item()

                pbar.set_postfix(**{'val_loss': loss.item(), 'lr': get_lr(optimizer)})
                pbar.update(1)

    loss_history.append_loss(train_loss / (epoch_size_train), val_loss / (epoch_size_val))

    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(n_epoch))
    if use_rank_loss:
        print('Train Loss: %.4f (MSE: %.4f, Rank: %.4f) || Val Loss: %.4f ' % (
            train_loss / (epoch_size_train),
            train_mse_loss / (epoch_size_train),
            train_rank_loss / (epoch_size_train),
            val_loss / (epoch_size_val)))
    else:
        print('Train Loss: %.4f || Val Loss: %.4f ' % (train_loss / (epoch_size_train), val_loss / (epoch_size_val)))

    print('Saving state, iter:', str(epoch + 1))
    save_path = os.path.join(loss_history.log_dir, f'Epoch%d_MERF_v2.pth' % ((epoch + 1)))
    torch.save(model.state_dict(), save_path)

    # ----------------------- TensorBoard Logging ----------------------- #
    if writer is not None:
        # Log training losses
        writer.add_scalar('Loss/train_total', train_loss / epoch_size_train, epoch)
        writer.add_scalar('Loss/train_mse', train_mse_loss / epoch_size_train, epoch)
        writer.add_scalar('Loss/train_rank', train_rank_loss / epoch_size_train, epoch)

        # Log validation loss
        writer.add_scalar('Loss/val', val_loss / epoch_size_val, epoch)

        # Log learning rate
        writer.add_scalar('LR/learning_rate', get_lr(optimizer), epoch)

    return val_loss / (epoch_size_val)


def val(model, val_loader, device):
    pre = []
    label = []
    PDB_id_list = []
    mutation_info_list = []

    # add tqdm for val progress bar
    for iteration, batch in enumerate(tqdm(val_loader, desc="Validation", ncols=80)):
        label.append(batch['ddG'].item())
        with torch.no_grad():
            batch = recursive_to(batch, device)
            pred, _, _, _ = model(batch, device)
            pre.append(pred.item())

        PDB_id_list.append(batch['wt']['PDB_id'][0])
        mutation_info_list.append(batch['wt']['mutate_info'][0])

    results = pd.DataFrame({"PDB_id": np.array(PDB_id_list), "mutate_info": np.array(mutation_info_list),
                         "Prediction": np.array(pre), "Ground Truth": np.array(label)})
    return results


if __name__ == '__main__':
    # ----------------------- environment setting ----------------------- #
    args = get_pretrain_args()
    print(args)

    seed = args.seed
    seed_all(seed)
    print(f"setting random seed...{seed}")

    GPU_indx = args.gpu_idx
    device = torch.device(GPU_indx if args.is_cuda else "cpu")

    loss_dir = "logs/pretrain_vm_v2/"
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

    train_vm_dataset = VMDatasetV2(
        train_vm_path, wt_vm_dir, mut_vm_dir,
        knn_num=args.knn_neighbors_num, knn_agents_num=args.knn_agents_num,
        use_plm_embedding=args.use_plm_embedding, plm_path=args.plm_path, plm_embedding_path=train_vm_embedding_path, device=device,
        return_rank_target=args.use_rank_loss
    )
    val_vm_dataset = VMDatasetV2(
        val_vm_path, wt_vm_dir, mut_vm_dir,
        knn_num=args.knn_neighbors_num, knn_agents_num=args.knn_agents_num,
        use_plm_embedding=args.use_plm_embedding, plm_path=args.plm_path, plm_embedding_path=val_vm_embedding_path, device=device,
        return_rank_target=args.use_rank_loss
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
        return_kl_target=args.use_rank_loss
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
        return_kl_target=args.use_rank_loss
    )
    
    # combine datasets
    train_dataset = torch.utils.data.ConcatDataset([train_vm_dataset, train_sk_dataset])
    val_dataset = torch.utils.data.ConcatDataset([val_vm_dataset, val_ab_dataset])

    print(f"Train dataset sizes - VM: {len(train_vm_dataset)}, SK: {len(train_sk_dataset)}, Total: {len(train_dataset)}")
    print(f"Val dataset sizes - VM: {len(val_vm_dataset)}, AB: {len(val_ab_dataset)}, Total: {len(val_dataset)}")

    # These will be updated if weighted sampling is used
    epoch_size_train = len(train_dataset) // args.batch_size
    epoch_size_val = len(val_dataset) // args.batch_size

    # create dataloader
    collate_fn = PaddingCollate()

    # ----------------------- Weighted Sampling for Balanced Training ----------------------- #
    if args.use_weighted_sampling:
        len_vm = len(train_vm_dataset)
        len_sk = len(train_sk_dataset)

        weight_per_vm_sample = args.vm_sampling_weight / len_vm
        weight_per_sk_sample = args.sk_sampling_weight / len_sk
        sample_weights = [weight_per_vm_sample] * len_vm + [weight_per_sk_sample] * len_sk

        # Strategy: limit repetition of smaller dataset based on max_repetition_factor
        if args.max_repetition_factor > 0:
            smaller_dataset_size = min(len_vm, len_sk)
            # For balanced sampling (weight=1:1), each dataset gets 50% of samples
            # So smaller dataset needs to be sampled (num_samples/2) times
            # To limit repetition to max_repetition_factor, we set:
            # num_samples/2 <= smaller_dataset_size * max_repetition_factor
            # => num_samples <= smaller_dataset_size * max_repetition_factor * 2
            num_samples_per_epoch = min(
                len(train_dataset),  # Original: use all samples
                int(smaller_dataset_size * args.max_repetition_factor * 2)  # Limit repetition
            )
        else:
            num_samples_per_epoch = len(train_dataset)

        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=num_samples_per_epoch,
            replacement=True
        )

        print(f"Using weighted sampling - VM weight: {args.vm_sampling_weight}, SK weight: {args.sk_sampling_weight}")
        print(f"Samples per epoch: {num_samples_per_epoch} (max_repetition_factor: {args.max_repetition_factor if args.max_repetition_factor > 0 else 'unlimited'})")

        # Calculate expected sampling counts
        expected_vm = int(num_samples_per_epoch * args.vm_sampling_weight / (args.vm_sampling_weight + args.sk_sampling_weight))
        expected_sk = int(num_samples_per_epoch * args.sk_sampling_weight / (args.vm_sampling_weight + args.sk_sampling_weight))

        print(f"Expected samples per epoch - VM: ~{expected_vm} (avg repetition: {expected_vm/len_vm:.2f}x), "
              f"SK: ~{expected_sk} (avg repetition: {expected_sk/len_sk:.2f}x)")

        # Update epoch size based on actual sampling
        epoch_size_train = num_samples_per_epoch // args.batch_size

        gen_train = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, pin_memory=False,
                               drop_last=True, num_workers=args.num_works, collate_fn=collate_fn)
    else:
        print("Using standard random sampling (shuffle=True)")
        gen_train = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=False,
                               drop_last=True, num_workers=args.num_works, collate_fn=collate_fn)

    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=False,
                         drop_last=True, num_workers=args.num_works, collate_fn=collate_fn)

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
    print(f"Rank Loss enabled: {args.use_rank_loss}, weight: {args.rank_loss_weight}, tie_threshold: {args.rank_tie_threshold}")
    for epoch in range(args.n_epoch):
        val_loss = train_one_epoch(model, optimizer, epoch, args.n_epoch, epoch_size_train, epoch_size_val, gen_train, gen_val, loss_history, device, use_rank_loss=args.use_rank_loss, writer=writer)

        # Update learning rate scheduler
        if args.use_cosine_lr:
            lr_scheduler.step()
        else:
            # Use ReduceLROnPlateau with val_loss or manual adjustment at epoch 10
            if epoch == 10:
                for p in optimizer.param_groups:
                    p['lr'] = 1e-5

    # ----------------------- record predtion results of val_set after training ----------------------- #
    # Test VM validation set
    gen_val_vm_for_test = DataLoader(val_vm_dataset, shuffle=False, batch_size=1, pin_memory=False, drop_last=False, num_workers=args.num_works, collate_fn=collate_fn)
    print("\n" + "="*80)
    print("VM Validation Set Results:")
    print("="*80)
    results_vm = val(model, gen_val_vm_for_test, device)
    pear_vm, p_value_vm = pearsonr(results_vm["Prediction"], results_vm["Ground Truth"])
    print(f'VM pearson-r={pear_vm:.4f} (p-value={p_value_vm:.4e})')
    mse_vm = mean_squared_error(results_vm["Prediction"], results_vm["Ground Truth"])
    rmse_vm = np.sqrt(mse_vm)
    print(f'VM rmse={rmse_vm:.4f}')

    # Test AB-Bind validation set
    gen_val_ab_for_test = DataLoader(val_ab_dataset, shuffle=False, batch_size=1, pin_memory=False, drop_last=False, num_workers=args.num_works, collate_fn=collate_fn)
    print("\n" + "="*80)
    print("AB-Bind Validation Set Results:")
    print("="*80)
    results_ab = val(model, gen_val_ab_for_test, device)
    pear_ab, p_value_ab = pearsonr(results_ab["Prediction"], results_ab["Ground Truth"])
    print(f'AB-Bind pearson-r={pear_ab:.4f} (p-value={p_value_ab:.4e})')
    mse_ab = mean_squared_error(results_ab["Prediction"], results_ab["Ground Truth"])
    rmse_ab = np.sqrt(mse_ab)
    print(f'AB-Bind rmse={rmse_ab:.4f}')

    # ----------------------- Log final metrics to TensorBoard ----------------------- #
    writer.add_scalar('Metrics/VM_pearson_r', pear_vm, args.n_epoch)
    writer.add_scalar('Metrics/VM_rmse', rmse_vm, args.n_epoch)
    writer.add_scalar('Metrics/ABbind_pearson_r', pear_ab, args.n_epoch)
    writer.add_scalar('Metrics/ABbind_rmse', rmse_ab, args.n_epoch)

    # Close TensorBoard writer
    writer.close()
    print(f"\nTensorBoard logs saved to: {loss_history.log_dir}")
    