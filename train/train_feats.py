# Training script for feature extraction module
# HRegNet (Original HRegNet model)
# SVD head
# Loss = Tf loss only

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR



from models import HierFeatureExtraction
from models.utils import set_seed
from losses import matching_loss, prob_chamfer_loss

from tqdm import tqdm
import argparse
import wandb

#MAN Dataset
from config import Config
import dataset
import sys
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='HRegNet')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--root', type=str, default='')
    parser.add_argument('--npoints', type=int, default=16384)
    parser.add_argument('--temp', type=float, default=0.1)
    parser.add_argument('--runname', type=str, default='')
    parser.add_argument('--wandb_dir', type=str, default='')
    parser.add_argument('--sigma_max', type=float, default=3.0)
    parser.add_argument('--dataset', type=str, default='man')
    parser.add_argument('--data_list', type=str, default='')
    parser.add_argument('--voxel_size', type=float, default=0.3)
    parser.add_argument('--augment', type=float, default=0.0)
    parser.add_argument('--ckpt_dir', type=str, default='')
    parser.add_argument('--pretrain_detector', type=str, default='')
    parser.add_argument('--train_desc', action='store_true')
    parser.add_argument('--freeze_detector', action='store_true')
    parser.add_argument('--use_fps', action='store_true')
    parser.add_argument('--use_weights', action='store_true')
    parser.add_argument('--use_wandb', action='store_true')

    return parser.parse_args()

def calc_losses(ret_dict_src, ret_dict_dst, gt_R, gt_t, args):
    l_chamfer_1 = prob_chamfer_loss(ret_dict_src['xyz_1'], ret_dict_dst['xyz_1'], \
        ret_dict_src['sigmas_1'], ret_dict_dst['sigmas_1'], gt_R, gt_t)
    l_chamfer_2 = prob_chamfer_loss(ret_dict_src['xyz_2'], ret_dict_dst['xyz_2'], \
        ret_dict_src['sigmas_2'], ret_dict_dst['sigmas_2'], gt_R, gt_t)
    l_chamfer_3 = prob_chamfer_loss(ret_dict_src['xyz_3'], ret_dict_dst['xyz_3'], \
        ret_dict_src['sigmas_3'], ret_dict_dst['sigmas_3'], gt_R, gt_t)
    l_chamfer = l_chamfer_1 + l_chamfer_2 + l_chamfer_3
    
    if not args.train_desc:
        return l_chamfer
    else:
        l_matching_1 = matching_loss(ret_dict_src['xyz_1'], ret_dict_src['sigmas_1'], ret_dict_src['desc_1'], \
            ret_dict_dst['xyz_1'], ret_dict_dst['sigmas_1'], ret_dict_dst['desc_1'], gt_R, gt_t, args.temp, args.sigma_max)
        l_matching_2 = matching_loss(ret_dict_src['xyz_2'], ret_dict_src['sigmas_2'], ret_dict_src['desc_2'], \
            ret_dict_dst['xyz_2'], ret_dict_dst['sigmas_2'], ret_dict_dst['desc_2'], gt_R, gt_t, args.temp, args.sigma_max)
        l_matching_3 = matching_loss(ret_dict_src['xyz_3'], ret_dict_src['sigmas_3'], ret_dict_src['desc_3'], \
            ret_dict_dst['xyz_3'], ret_dict_dst['sigmas_3'], ret_dict_dst['desc_3'], gt_R, gt_t, args.temp, args.sigma_max)
        l_matching = l_matching_1 + l_matching_2 + l_matching_3
        return l_chamfer, l_matching


def val_feats(args, net):
    if args.dataset == 'kitti':
        val_seqs = ['06','07']
        #val_dataset = KittiDataset(args.root, val_seqs, args.npoints, args.voxel_size, args.data_list, 0.0)
    elif args.dataset == 'nuscenes':
        val_seqs = ['val']
        #val_dataset = NuscenesDataset(args.root, val_seqs, args.npoints, args.voxel_size, args.data_list, 0.0)
    elif args.dataset == 'man':
        config = Config()
        config.dataset_config.split = 'mini_val'
        loader = dataset.TruckScenesLoader()
        man_data = dataset.TruckScenesDataset(loader(config, verbose=False), config)
        val_dataset = dataset.TruckScenesPerturbation(dataset=man_data, config=config)
        gt_Transformation = np.eye(4)
    else:  
        raise('Not implemented')

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            num_workers=4,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True)
    
    net.eval()
    total_l_chamfer = 0
    total_l_matching = 0
    count = 0
    pbar = tqdm(enumerate(val_loader))

    with torch.no_grad():
        for i, data in pbar:
            # if args.dataset == 'kitti'  or 'nuscenes':
            #     src_points, dst_points, gt_R, gt_t = data
                
            if args.dataset == 'man':
                src_points = data['uncalibed_pcd']
                dst_points = data['pcd_left']

                gt_Transformation = data['igt']

                gt_Transformation = torch.inverse(gt_Transformation)

                gt_R = gt_Transformation[:,:3,:3].contiguous()
                gt_t = gt_Transformation[:,:3,3].contiguous()
            

            src_points = src_points.cuda()
            dst_points = dst_points.cuda()
            gt_R = gt_R.cuda()
            gt_t = gt_t.cuda()

            ret_dict_src = net(src_points)
            ret_dict_dst = net(dst_points)

            if args.train_desc:
                l_chamfer, l_matching = calc_losses(ret_dict_src, ret_dict_dst, gt_R, gt_t, args)
                total_l_chamfer += l_chamfer.item()
                total_l_matching += l_matching.item()
            else:
                l_chamfer = calc_losses(ret_dict_src, ret_dict_dst, gt_R, gt_t, args)
                total_l_chamfer += l_chamfer.item()
            count += 1

    if args.train_desc:
        total_l_chamfer = total_l_chamfer/count
        total_l_matching = total_l_matching/count
        return total_l_chamfer, total_l_matching
    else:
        total_l_chamfer = total_l_chamfer/count
        return total_l_chamfer

def train_feats(args):
    if args.dataset == 'kitti':
        train_seqs = ['00','01','02','03','04','05']
        #train_dataset = KittiDataset(args.root, train_seqs, args.npoints, args.voxel_size, args.data_list, args.augment)
    elif args.dataset == 'nuscenes':
        train_seqs = ['train']
        #train_dataset = NuscenesDataset(args.root, train_seqs, args.npoints, args.voxel_size, args.data_list, args.augment)
    elif args.dataset == 'man':
        config = Config()
        loader = dataset.TruckScenesLoader()
        man_data = dataset.TruckScenesDataset(loader(config, verbose=False), config)
        train_dataset = dataset.TruckScenesPerturbation(dataset=man_data, config=config)
        gt_Transformation = np.eye(4)
    else:  
        raise('Not implemented')
    
    
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=4,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)
    
    net = HierFeatureExtraction(args)
    if args.train_desc:
        net.load_state_dict(torch.load(args.pretrain_detector))
    if args.use_wandb:
        wandb.watch(net)

    net.cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    best_train_loss = float('inf')
    best_val_loss = float('inf')
    best_train_epoch = 0
    best_val_epoch = 0

    for epoch in tqdm(range(args.epochs)):

        net.train()
        count = 0
        total_loss = 0
        total_l_chamfer = 0
        total_l_matching = 0

        pbar = tqdm(enumerate(train_loader))

        for i, data in pbar:
            # if args.dataset == 'kitti'  or 'nuscenes':
            #     src_points, dst_points, gt_R, gt_t = data

            if args.dataset == 'man':
                src_points = data['uncalibed_pcd']
                dst_points = data['pcd_left']

                gt_Transformation = data['igt']
                gt_Transformation = torch.inverse(gt_Transformation)
                
                gt_R = gt_Transformation[:,:3,:3].contiguous()
                gt_t = gt_Transformation[:,:3,3].contiguous()
            
            assert gt_R.is_contiguous()
            assert gt_t.is_contiguous()
             
            src_points = src_points.cuda()
            dst_points = dst_points.cuda()
            gt_R = gt_R.cuda()
            gt_t = gt_t.cuda()

            optimizer.zero_grad()
            ret_dict_src = net(src_points)
            ret_dict_dst = net(dst_points)
            if args.train_desc:
                l_chamfer, l_matching, mi_loss = calc_losses(ret_dict_src, ret_dict_dst, gt_R, gt_t, args)
                loss = l_chamfer + l_matching + mi_loss
            else:
                l_chamfer = calc_losses(ret_dict_src, ret_dict_dst, gt_R, gt_t, args)
                loss = l_chamfer
            loss.backward()
            optimizer.step()

            count += 1
            total_loss += loss.item()
            total_l_chamfer += l_chamfer.item()
            if args.train_desc:
                total_l_matching += l_matching.item()

            if i % 10 == 0:
                pbar.set_description('Train Epoch:{}[{}/{}({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, i, len(train_loader), 100. * i/len(train_loader), loss.item()
                ))

        total_loss /= count
        total_l_chamfer /= count
        if args.train_desc:
            total_l_matching /= count
        
        if args.train_desc:
            val_chamfer, val_matching = val_feats(args, net)
            total_val_loss = val_chamfer + val_matching
        else:
            val_chamfer = val_feats(args, net)
            total_val_loss = val_chamfer
        
        if args.use_wandb:
            if args.train_desc:
                wandb.log({"train/chamfer":total_l_chamfer,
                           "train/matching":total_l_matching,
                           "val/chamfer":val_chamfer,
                           "val/matching":val_matching})
            else:
                wandb.log({"train/chamfer":total_l_chamfer,
                            "val/chamfer":val_chamfer})
        
        print('\n Epoch {} finished. Training loss: {:.4f} Valiadation loss: {:.4f}'.\
            format(epoch+1, total_loss, total_val_loss))
        
        ckpt_dir = os.path.join(args.ckpt_dir, args.dataset + '_ckpt_'+args.runname)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        
        if total_loss < best_train_loss:
            torch.save(net.state_dict(), os.path.join(ckpt_dir, 'best_train.pth'))
            best_train_loss = total_loss
            best_train_epoch = epoch + 1
        
        if total_val_loss < best_val_loss:
            torch.save(net.state_dict(), os.path.join(ckpt_dir, 'best_val.pth'))
            best_val_loss = total_val_loss
            best_val_epoch = epoch + 1
        
        print('Best train epoch: {} Best train loss: {:.4f} Best val epoch: {} Best val loss: {:.4f}'.format(
            best_train_epoch, best_train_loss, best_val_epoch, best_val_loss
        ))
        
        scheduler.step()

if __name__ == '__main__':
    args = parse_args()
    #print(args.train_desc)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seed(args.seed)

    if args.use_wandb:
        import wandb
        wandb.init(config=args, project='PCDReg_HRegNet', name=args.dataset+'_'+args.runname, dir=args.wandb_dir)
    train_feats(args)

