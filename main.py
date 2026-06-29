import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.optim
from lifelines.utils import concordance_index
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ExactSizeSurvivalDataset import MILDataset
from model import calculate_geometric_reg, cfg, RIMA
from models.RIMA.KaplanMeier import plot_kaplan_meier2
from models.RIMA.utils_cox import set_seed, return_splits, EarlyStopping


def c_index(y_pred, y_time, y_event):
    time = y_time
    event = y_event
    return concordance_index(time, -np.exp(y_pred), event)


def coxph_loss(y_pred, y_time, y_event):
    time = y_time
    event = y_event

    sort_time = torch.argsort(time, 0, descending=True)
    risk = torch.gather(y_pred, 0, sort_time)



    max_risk = torch.max(risk)

    exp_risk_shifted = torch.exp(risk - max_risk)

    cumsum_exp_shifted = torch.cumsum(exp_risk_shifted, 0)


    epsilon = 1e-8
    log_risk = torch.log(cumsum_exp_shifted + epsilon) + max_risk


    censored_likelihood = (risk - log_risk) * event
    censored_likelihood = torch.sum(censored_likelihood)
    censored_likelihood = censored_likelihood / y_time.shape[0]
    return -censored_likelihood


def get_args():
    parser = argparse.ArgumentParser(description='MIL main parameters')

    # General params.
    parser.add_argument('--experiment_name', type=str, default='RIMA', help='experiment name')
    parser.add_argument('--MIL_model', type=str, default='RIMA',  # [修改] 默认跑新模型
                        choices=['ABMIL', 'CLAM_SB', 'CLAM_MB', 'MeanMIL', 'MaxMIL', 'DSMIL', 'TransMIL'],
                        help='MIL model to use')
    parser.add_argument('--metric2save', type=str, default='c-index', choices=['c-index'])
    parser.add_argument('--device_ids', type=str, default=0, help='gpu devices for training')
    parser.add_argument('--seed', type=int, default=3721, help='random seed')
    parser.add_argument('--fold', type=int, default=1, help='fold number')
    parser.add_argument('--dataset', type=str, default='COAD')


    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr_patience', type=int, default=8)
    parser.add_argument('--max_lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--fold_list', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--test', action='store_false', help='use test dataset')


    parser.add_argument('--csv_dir', type=str,
                        default="./csv/COAD")
    parser.add_argument('--label_xlsx', type=str,
                        default="./csv/COAD.csv")
    parser.add_argument('--feat_dir', type=str, default='./DATA/COAD/feat')
    parser.add_argument('--coords_dir', type=str, default='./DATA/COAD/h5_files')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt/COAD')
    parser.add_argument('--logger_dir', type=str, default='./logger/COAD')
    parser.add_argument('--results_dir', type=str, default='./results/COAD')


    parser.add_argument('--genomics_txt', type=str, default="./gene/COAD.csv",
                        help='Path to raw genomics txt file')

    args = parser.parse_args()
    return args


def MIL_train_epoch(fold, epoch, model, optimizer, loader, device, model_name, args=None):
    model.train()
    total_epoch_loss = 0.
    update_steps = 0


    batch_logits, batch_times, batch_events, batch_geo_losses = [], [], [], []


    all_all_logits, all_all_times, all_all_events = [], [], []

    current_lr = optimizer.param_groups[0]['lr']
    optimizer.zero_grad()


    effective_batch_size = 30

    with tqdm(total=len(loader), desc=f'[Fold:{fold}, Epoch:{epoch}]') as pbar:
        for i, sample in enumerate(loader):



            feat = sample['feat'].to(device)
            time = sample['time'].to(device)
            event = sample['event'].to(device)
            gene = sample.get('gene', None)
            if gene is not None:
                gene = gene.to(device)

            if feat.shape[1] == 0:
                pbar.update(1)
                continue


            if model_name == 'RIMA':

                risk_score, T, G_V, G_G, v_geo, g_geo = model(feat, gene)


                geo_loss = calculate_geometric_reg(T, G_V, G_G, v_geo, g_geo)
                batch_geo_losses.append(geo_loss)



            batch_logits.append(risk_score.view(-1))
            batch_times.append(time.view(-1))
            batch_events.append(event.view(-1))


            all_all_logits.append(risk_score.detach().cpu().numpy().reshape(-1))
            all_all_times.append(time.detach().cpu().numpy().reshape(-1))
            all_all_events.append(event.detach().cpu().numpy().reshape(-1))


            if len(batch_logits) >= effective_batch_size or i == len(loader) - 1:
                logits_cat = torch.cat(batch_logits)
                times_cat = torch.cat(batch_times)
                events_cat = torch.cat(batch_events)


                loss_cox = coxph_loss(logits_cat, times_cat, events_cat)


                if model_name == 'RIMA' and len(batch_geo_losses) > 0:
                    loss_geo = torch.stack(batch_geo_losses).mean()
                    total_loss = loss_cox + loss_geo



                total_loss.backward()


                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                optimizer.zero_grad()


                last_loss_val = total_loss.item()
                total_epoch_loss += last_loss_val
                update_steps += 1


                batch_logits, batch_times, batch_events, batch_geo_losses = [], [], [], []


                pbar.set_postfix(lr=f"{current_lr:.6f}",
                                 loss=f"{last_loss_val:.4f}",
                                 geo=f"{loss_geo.item():.4f}" if model_name == 'RIMA' else "N/A")

            pbar.update(1)

    # --- 3. 计算 Epoch 指标 ---
    final_logits = np.concatenate(all_all_logits)
    final_times = np.concatenate(all_all_times)
    final_events = np.concatenate(all_all_events)

    # 计算生存分析核心指标：C-index
    epoch_cindex = c_index(final_logits, final_times, final_events)
    avg_loss = total_epoch_loss / update_steps if update_steps > 0 else 0

    print(f"\n[Epoch {epoch} Done] Avg Loss: {avg_loss:.5f}, C-index: {epoch_cindex:.4f}")
    return avg_loss, epoch_cindex


def MIL_pred(model, loader, device, fold, args, status='Val', model_name='RIMA'):
    model.eval()
    WSI_logits, slide_time, slide_event, slide_ids = [], [], [], []
    s = []
    with torch.no_grad():
        with tqdm(total=len(loader), desc=f'[{status} Fold:{fold}]') as pbar:
            for i, sample in enumerate(loader):
                # 1. 数据提取
                slide_id = sample['slide_id']




                feat, time, event = sample['feat'], sample['time'], sample['event']
                gene = sample.get('gene', None)
                # text = sample.get('text', None)

                # 跳过空样本
                if feat.shape[1] == 0:
                    pbar.update(1)
                    continue

                slide_ids.append(slide_id)
                feat, time, event = feat.to(device), time.to(device), event.to(device)
                if gene is not None:
                    gene = gene.to(device)


                if model_name == 'RIMA':

                    risk_score, T, G_V, G_G, v_geo, g_geo = model(feat, gene)



                WSI_logits.append(risk_score.view(-1).cpu())
                slide_time.append(time.view(-1).cpu())
                slide_event.append(event.view(-1).cpu())

                pbar.update(1)


            WSI_logits = torch.cat(WSI_logits)
            slide_time = torch.cat(slide_time)
            slide_event = torch.cat(slide_event)

            risk_median = torch.median(WSI_logits).item()
            print(f"Risk median: {risk_median:.6f}")


            coxloss = coxph_loss(WSI_logits, slide_time, slide_event)
            cindex = c_index(WSI_logits.cpu().numpy(),
                             slide_time.cpu().numpy(),
                             slide_event.cpu().numpy())


            if status != 'Val':
                predictions_np = WSI_logits.cpu().numpy().flatten()
                times_np = slide_time.cpu().numpy().flatten()
                events_np = slide_event.cpu().numpy().flatten()

                results_dict = {
                    'slide_id': [sid[0] if isinstance(sid, list) else sid for sid in slide_ids],
                    'survival_days': times_np,
                    'event_status': events_np.astype(int),
                    'predicted_risk': predictions_np
                }
                results_df = pd.DataFrame(results_dict)

                try:


                    outpath = os.path.join(args.results_dir,
                                           str(args.experiment_name))
                    os.makedirs(outpath, exist_ok=True)

                    csv_name = f'survival_{fold}.csv'
                    results_df.to_csv(os.path.join(outpath, csv_name), index=False, encoding='utf-8-sig')
                    print(f"\n✅ Fold {fold} 结果已保存至: {outpath}")


                    plot_kaplan_meier2(predictions_np, times_np, events_np,
                                       output_path=os.path.join(outpath, f'km_{fold}.jpg'), xlim=(0, 2500),
                                       xticks=[0, 500, 1000, 1500, 2000, 2500],
                                       xlabel="Disease-Specific Survival (Day)",
                                       ylabel="Survival Probability")

                except Exception as e:
                    print(f"\n❌ 保存文件时出错: {e}")

            print(f'Final Results - Loss: {coxloss:.4f}, C-index: {cindex:.4f}')

    return coxloss, cindex


if __name__ == '__main__':
    args = get_args()

    # set device
    device = torch.device('cuda:{}'.format(args.device_ids))
    print('Using GPU ID: {}'.format(args.device_ids))

    # set random seed
    set_seed(args.seed)
    print('Using Random Seed: {}'.format(str(args.seed)))

    # set tensorboard
    args.logger_dir = os.path.join(args.logger_dir, args.experiment_name)
    os.makedirs(args.logger_dir, exist_ok=True)
    writer = SummaryWriter(args.logger_dir)

    fold_list = args.fold_list
    cindex_list = []

    for fold in fold_list:
        csv_path = os.path.join(args.csv_dir, 'fold_{}.csv'.format(fold))
        feat_dir = args.feat_dir

        if args.test:
            train_dataset, val_dataset, test_dataset = return_splits(csv_path=csv_path, test=True)
        else:
            train_dataset, val_dataset = return_splits(csv_path=csv_path, test=False)


        train_dset = MILDataset(args, train_dataset, feat_dir, args.label_xlsx, args.genomics_txt, args.coords_dir)
        train_loader = DataLoader(train_dset, batch_size=1, shuffle=True, num_workers=0)

        val_dset = MILDataset(args, test_dataset, feat_dir, args.label_xlsx, args.genomics_txt, args.coords_dir)
        val_loader = DataLoader(val_dset, batch_size=1, shuffle=False, num_workers=0)

        if args.test:
            test_dset = MILDataset(args, test_dataset, feat_dir, args.label_xlsx, args.genomics_txt, args.coords_dir)
            test_loader = DataLoader(test_dset, batch_size=1, shuffle=False, num_workers=0)

        model_dir = os.path.join(args.ckpt_dir, args.experiment_name)
        os.makedirs(model_dir, exist_ok=True)


        if 'RIMA' == args.MIL_model:
            model = RIMA(cfg)



        else:
            raise NotImplementedError

        model = model.to(device)
        lr = args.max_lr
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        model_path = os.path.join(model_dir, '{}_model_{}.pth'.format(args.MIL_model, fold))

        if not os.path.exists(model_path):
            early_stopping = EarlyStopping(model_path=model_path, patience=args.patience, verbose=True,
                                           count_loss=False)
            for epoch in range(args.epochs):
                train_loss, train_c_index = MIL_train_epoch(fold, epoch, model, optimizer, train_loader, device,
                                                            model_name=args.MIL_model)
                val_loss, val_c_index = MIL_pred(model, val_loader, device, fold, args, model_name=args.MIL_model, )

                if args.metric2save == 'c-index':
                    counter = early_stopping(epoch, val_loss, model, val_cidx=val_c_index)

                if early_stopping.early_stop:
                    print('Early Stopping')
                    break
                if counter > 0 and counter % args.lr_patience == 0:
                    if lr > args.min_lr:
                        early_stopping.reset()
                        lr = lr / 10 if lr / 10 >= args.min_lr else args.min_lr
                        for params in optimizer.param_groups:
                            params['lr'] = lr



        pretrained_dict = torch.load(model_path, map_location='cpu')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model.load_state_dict(pretrained_dict, strict=False)

        if args.test:


            test_loss, test_c_index = MIL_pred(model, test_loader, device, fold, args, 'Test',
                                               model_name=args.MIL_model)


            cindex_list.append(test_c_index)

    print(cindex_list)
    print(f'{np.mean(cindex_list):.4f}±{np.std(cindex_list):.4f}')
