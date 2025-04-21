import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import datetime
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim

from utils.dataloader import SKEMPIV2Dataset, ABbindDataset
from utils.arguments import get_common_args
from utils.util import *

from protein.read_pdbs import PaddingCollate
from model.MERF import MERF
from utils.losshistory import LossHistory
from torch.utils.tensorboard import SummaryWriter


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_one_epoch(MERF, optimizer,
                    epoch, n_epoch, epoch_size_train, epoch_size_val,
                    train_loader, val_loader,
                    loss_history, device, global_train_step):
    # ----------------------- Train ----------------------- #
    print("Start Train")
    MERF.train()

    train_loss = 0
    val_loss = 0

    with tqdm(total=epoch_size_train, desc=f'Epoch {epoch + 1}/{n_epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(train_loader):
            batch = recursive_to(batch, device)
            q_tot_wt, loss, ten_l1_loss = MERF(batch, device)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            train_loss += loss.item()

            pbar.set_postfix(**{'train_loss': loss.item(),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

            writer.add_scalar('Train/Total_loss', loss, global_train_step)
            writer.add_scalar('Train/10_times_l1_loss', ten_l1_loss, global_train_step)

            global_train_step += 1

    print('Finish Train')

    # ----------------------- Val ----------------------- #
    print('Start Validation')

    MERF.eval()

    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{n_epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(val_loader):
            batch = recursive_to(batch, device)

            ddG = batch['ddG'].to(device)
            ddG = ddG.to(torch.float32)

            with torch.no_grad():
                q_tot_wt, _, _ = MERF(batch, device)

                loss_fn = torch.nn.MSELoss()

                loss_wt = loss_fn(q_tot_wt, ddG)

                loss = loss_wt

                val_loss += loss.item()

                pbar.set_postfix(**{'val_loss': loss.item(),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)

    loss_history.append_loss(train_loss / (epoch_size_train), val_loss / (epoch_size_val))

    writer.add_scalar('Val/Total_loss', val_loss / (epoch_size_val), epoch)
    
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(n_epoch))
    print('Train Loss: %.4f || Val Loss: %.4f ' % (train_loss / (epoch_size_train), val_loss / (epoch_size_val)))

    print('Saving state, iter:', str(epoch + 1))
    torch.save(MERF.state_dict(),
               'logs/abbind/' + loss_history.get_str() + '/Epoch%d_qmix_merf.pth' % (
                   (epoch + 1)))

    return val_loss / (epoch_size_val), global_train_step


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
            q_tot_wt, _, _ = MERF(batch, device)
            pred = q_tot_wt
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
    args = get_common_args()
    print(args)

    seed = args.seed
    seed_all(seed)
    print(f"setting random seed...{seed}")

    GPU_indx = args.gpu_idx
    device = torch.device(GPU_indx if args.is_cuda else "cpu")

    loss_dir = "logs/abbind/"
    loss_history = LossHistory(loss_dir)
    loss_history.write(str(args) + '\n')
    start_time_str = loss_history.get_str()
    
    tensor_log_dir = os.path.join('tensor_logs/pretrain', start_time_str)
    writer = SummaryWriter(tensor_log_dir)

    # ----------------------- data read in and create dataset ----------------------- #
    # read in skempiv2 as train_set
    train0_path = f'data/SKEMPIv2/Multi_10fold/train_1.csv'
    val0_path = f'data/SKEMPIv2/Multi_10fold/test_1.csv'

    train0_df = pd.read_csv(train0_path, dtype={"PDB_id": "string"})
    val0_df = pd.read_csv(val0_path, dtype={"PDB_id": "string"})

    train_df = pd.concat([train0_df, val0_df])

    # read in abbind as val_set
    test_path = 'data/ABbind/AB-Bind_645pMulti_filtered.csv'
    val_df = pd.read_csv(test_path, dtype={"PDB_id": "string"})

    print(len(train_df["PDB_id"]), len(val_df["PDB_id"]))

    epoch_size_train = len(train_df["PDB_id"]) // args.batch_size
    epoch_size_val = len(val_df["PDB_id"]) // args.batch_size
    if epoch_size_train == 0 or epoch_size_val == 0:
        raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

    train_dataset = SKEMPIV2Dataset(train_df, is_train=True, knn_num=args.knn_neighbors_num,
                                    knn_agents_num=args.knn_agents_num)
    val_dataset = ABbindDataset(val_df, is_train=True, knn_num=args.knn_neighbors_num,
                                knn_agents_num=args.knn_agents_num)
    collate_fn = PaddingCollate()
    gen_train = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=False,
                           drop_last=True, num_workers=args.num_works, collate_fn=collate_fn)
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=False,
                         drop_last=True, num_workers=args.num_works, collate_fn=collate_fn)

    # ----------------------- Initial Networks ----------------------- #
    MERF = MERF(args).to(device)

    # init optimizer
    optimizer = optim.Adam(MERF.parameters(), lr=args.lr, weight_decay=0.1)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

    # ----------------------- fit one epoch ----------------------- #
    global_train_step = 0
    for epoch in range(args.n_epoch):
        val_loss, global_train_step = train_one_epoch(MERF, optimizer,
                                   epoch, args.n_epoch, epoch_size_train, epoch_size_val,
                                   gen_train, gen_val,
                                   loss_history, device, global_train_step)
        if epoch == 10:
            for p in optimizer.param_groups:
                p['lr'] = 1e-5

    # ----------------------- record pred in val_set after training ----------------------- #
    gen_val_for_test = DataLoader(val_dataset, shuffle=False, batch_size=1, pin_memory=False,
                                  drop_last=True, num_workers=args.num_works, collate_fn=collate_fn)
    results = val(MERF, gen_val_for_test, device)

    results_save_path = "results/ABbind/abbind_test_results_" + start_time_str + ".csv"
    results_pdf_save_path = "results/ABbind/abbind_test_results_" + start_time_str + ".pdf"
    results_txt_save_path = "results/ABbind/abbind_test_results_" + start_time_str + ".txt"
    results.to_csv(results_save_path, index=False)
    pear, p_value = pearsonr(results["Prediction"], results["Ground Truth"])
    print(f'pearson-r={pear}')
    mse = mean_squared_error(results["Prediction"], results["Ground Truth"])
    rmse = np.sqrt(mse)
    print(f'rmse={rmse}')

    with open(results_txt_save_path, 'a') as f:
        f.write(f'pearson-r={pear}\n')
        f.write(f'rmse={rmse}\n')

    writer.close()