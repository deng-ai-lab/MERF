import os
import numpy as np
import pandas as pd
import pickle

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model.MERF import MERF

from dataset.dataset_ddg import DDGBaseDataset, VMDataset
from utils.arguments import get_pretrain_args
from utils.util import *
from utils.losshistory import LossHistory

from protein.read_pdbs import PaddingCollate

from tqdm import tqdm

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_one_epoch(model, optimizer, epoch, n_epoch, epoch_size_train, epoch_size_val, train_loader, val_loader, loss_history, device):
    # ----------------------- Train ----------------------- #
    print("Start Train")
    model.train()

    train_loss = 0
    val_loss = 0

    with tqdm(total=epoch_size_train, desc=f'Epoch {epoch + 1}/{n_epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(train_loader):

            batch = recursive_to(batch, device)
            q_tot_wt, loss, _, _ = model(batch, device)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            train_loss += loss.item()

            pbar.set_postfix(**{'train_loss': loss.item(), 'lr': get_lr(optimizer)})
            pbar.update(1)

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
    print('Train Loss: %.4f || Val Loss: %.4f ' % (train_loss / (epoch_size_train), val_loss / (epoch_size_val)))

    print('Saving state, iter:', str(epoch + 1))
    save_path = os.path.join(loss_history.log_dir, 'Epoch%d_MERF.pth' % ((epoch + 1)))
    torch.save(model.state_dict(), save_path)

    return val_loss / (epoch_size_val)


def val(model, val_loader, device):
    pre = []
    label = []
    PDB_id_list = []
    mutation_info_list = []

    with tqdm(total=epoch_size_val, desc='Evaluating', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(val_loader):
            label.append(batch['ddG'].item())
            with torch.no_grad():
                batch = recursive_to(batch, device)
                pred, _, _, _ = model(batch, device)
                pre.append(pred.item())
                pbar.update(1)

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

    loss_dir = "logs/pretrain_sk/"
    loss_history = LossHistory(loss_dir)
    loss_history.write(str(args) + '\n')
    start_time_str = loss_history.get_str()

    # Save args for later testing
    args_save_path = os.path.join(loss_history.log_dir, 'args.pkl')
    with open(args_save_path, 'wb') as f:
        pickle.dump(args, f)
    print(f"Args saved to: {args_save_path}")

    # ----------------------- dataset ----------------------- #
    # read in skempiv2 as train_set
    train_path = '/home/lfj/projects_dir/MERF/data/SKEMPIv2/SKEMPIv2.csv'
    train_wt_dir = '/home/lfj/projects_dir/MERF/data/SKEMPIv2/PDBs_fixed'
    train_mut_dir = '/home/lfj/projects_dir/MERF/data/SKEMPIv2/PDBs_mutated'
    sk_embedding_path = '/home/lfj/projects_dir/MERF/data/SKEMPIv2/skempiv2_esm2_650_embeddings.pkl'
    # train_wt_dir = '/home/lfj/projects_dir/MERF/data/SKEMPIv2/PDBs_fixed_jackal'
    # train_mut_dir = '/home/lfj/projects_dir/MERF/data/SKEMPIv2/PDBs_mutated_jackal'

    train_dataset = DDGBaseDataset(
        train_path, train_wt_dir, train_mut_dir,
        knn_num=args.knn_neighbors_num, knn_agents_num=args.knn_agents_num,
        use_plm_embedding=args.use_plm_embedding, plm_path=args.plm_path, plm_embedding_path=sk_embedding_path, device=device
    )

    # read in abbind as val_set
    val_path = '/home/lfj/projects_dir/MERF/data/ABbind/AB-Bind_645pMulti.csv'
    val_wt_dir = '/home/lfj/projects_dir/MERF/data/ABbind/PDBs_fixed'
    val_mut_dir = '/home/lfj/projects_dir/MERF/data/ABbind/PDBs_mutated'
    ab_embedding_path = '/home/lfj/projects_dir/MERF/data/ABbind/abbind_esm2_650_embeddings.pkl'
    # val_path = '/home/lfj/projects_dir/MERF/data/ABbind/AB-Bind_645pMulti_old.csv'
    # val_wt_dir = '/home/lfj/projects_dir/MERF/data/ABbind/PDBs_fixed_jackal'
    # val_mut_dir = '/home/lfj/projects_dir/MERF/data/ABbind/PDBs_mutated_jackal'

    val_dataset = DDGBaseDataset(
        val_path, val_wt_dir, val_mut_dir,
        knn_num=args.knn_neighbors_num, knn_agents_num=args.knn_agents_num,
        use_plm_embedding=args.use_plm_embedding, plm_path=args.plm_path, plm_embedding_path=ab_embedding_path, device=device
    )
    
    print(len(train_dataset), len(val_dataset))

    epoch_size_train = len(train_dataset) // args.batch_size
    epoch_size_val = len(val_dataset) // args.batch_size

    # create dataloader
    collate_fn = PaddingCollate()
    gen_train = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=False,
                           drop_last=True, num_workers=args.num_works, collate_fn=collate_fn)
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=False,
                         drop_last=True, num_workers=args.num_works, collate_fn=collate_fn)

    # ----------------------- Initial Networks and optimizer ----------------------- #
    model = MERF(args).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.1)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

    # ----------------------- fit epoches ----------------------- #
    for epoch in range(args.n_epoch):
        val_loss = train_one_epoch(model, optimizer, epoch, args.n_epoch, epoch_size_train, epoch_size_val, gen_train, gen_val, loss_history, device)
        if epoch == 10:
            for p in optimizer.param_groups:
                p['lr'] = 1e-5

    # ----------------------- record predtion results of val_set after training ----------------------- #
    gen_val_for_test = DataLoader(val_dataset, shuffle=False, batch_size=1, pin_memory=False, drop_last=False, num_workers=args.num_works, collate_fn=collate_fn)
    
    results = val(model, gen_val_for_test, device)

    results_save_path = os.path.join(loss_history.log_dir, "abbind_test_results.csv")
    results.to_csv(results_save_path, index=False)

    pear, p_value = pearsonr(results["Prediction"], results["Ground Truth"])
    print(f'pearson-r={pear}')
    mse = mean_squared_error(results["Prediction"], results["Ground Truth"])
    rmse = np.sqrt(mse)
    print(f'rmse={rmse}')
