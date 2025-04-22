import os
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model.MERF import MERF

from utils.dataloader import SKEMPIV2Dataset, ABbindDataset
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


def train_one_epoch(MERF, optimizer, epoch, n_epoch, epoch_size_train, epoch_size_val, train_loader, val_loader, loss_history, device):
    # ----------------------- Train ----------------------- #
    print("Start Train")
    MERF.train()

    train_loss = 0
    val_loss = 0

    with tqdm(total=epoch_size_train, desc=f'Epoch {epoch + 1}/{n_epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(train_loader):
            batch = recursive_to(batch, device)
            q_tot_wt, loss = MERF(batch, device)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            train_loss += loss.item()

            pbar.set_postfix(**{'train_loss': loss.item(), 'lr': get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    # ----------------------- Validation ----------------------- #
    print('Start Validation')

    MERF.eval()

    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{n_epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(val_loader):
            batch = recursive_to(batch, device)

            ddG = batch['ddG'].to(device)
            ddG = ddG.to(torch.float32)

            with torch.no_grad():
                q_tot_wt, _ = MERF(batch, device)

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
    torch.save(MERF.state_dict(), 'logs/pretrain/' + loss_history.get_str() + '/Epoch%d_MERF.pth' % ((epoch + 1)))

    return val_loss / (epoch_size_val)


def val(MERF, val_loader, device):
    pre = []
    label = []
    index = 0
    PDB_id_list = []
    mutation_info_list = []

    for iteration, batch in enumerate(val_loader):
        label.append(batch['ddG'].item())
        with torch.no_grad():
            batch = recursive_to(batch, device)
            pred, _ = MERF(batch, device)
            pre.append(pred.item())
            index += 1
            if index % 50 == 0:
                print(index)

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

    loss_dir = "logs/pretrain/"
    loss_history = LossHistory(loss_dir)
    loss_history.write(str(args) + '\n')
    start_time_str = loss_history.get_str()

    # ----------------------- dataset ----------------------- #
    # read in skempiv2 as train_set
    train_path = 'data/SKEMPIv2/SKEMPIv2.csv'
    train_df = pd.read_csv(train_path, dtype={"PDB_id": "string"})

    # read in abbind as val_set
    val_path = 'data/ABbind/AB-Bind_645pMulti.csv'
    val_df = pd.read_csv(val_path, dtype={"PDB_id": "string"})

    print(len(train_df["PDB_id"]), len(val_df["PDB_id"]))

    # create dataset
    epoch_size_train = len(train_df["PDB_id"]) // args.batch_size
    epoch_size_val = len(val_df["PDB_id"]) // args.batch_size

    train_dataset = SKEMPIV2Dataset(train_df, knn_num=args.knn_neighbors_num, knn_agents_num=args.knn_agents_num)
    val_dataset = ABbindDataset(val_df, knn_num=args.knn_neighbors_num, knn_agents_num=args.knn_agents_num)
    
    # create dataloader
    collate_fn = PaddingCollate()
    gen_train = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=False,
                           drop_last=True, num_workers=args.num_works, collate_fn=collate_fn)
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=False,
                         drop_last=True, num_workers=args.num_works, collate_fn=collate_fn)

    # ----------------------- Initial Networks and optimizer ----------------------- #
    MERF = MERF(args).to(device)

    optimizer = optim.Adam(MERF.parameters(), lr=args.lr, weight_decay=0.1)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

    # ----------------------- fit epoches ----------------------- #
    for epoch in range(args.n_epoch):
        val_loss = train_one_epoch(MERF, optimizer, epoch, args.n_epoch, epoch_size_train, epoch_size_val, gen_train, gen_val, loss_history, device)
        if epoch == 10:
            for p in optimizer.param_groups:
                p['lr'] = 1e-5

    # ----------------------- record predtion results of val_set after training ----------------------- #
    gen_val_for_test = DataLoader(val_dataset, shuffle=False, batch_size=1, pin_memory=False, drop_last=False, num_workers=args.num_works, collate_fn=collate_fn)
    
    results = val(MERF, gen_val_for_test, device)

    results_save_path = 'logs/pretrain/' + loss_history.get_str() + "abbind_test_results.csv"
    results.to_csv(results_save_path, index=False)

    pear, p_value = pearsonr(results["Prediction"], results["Ground Truth"])
    print(f'pearson-r={pear}')
    mse = mean_squared_error(results["Prediction"], results["Ground Truth"])
    rmse = np.sqrt(mse)
    print(f'rmse={rmse}')
