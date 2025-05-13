# Training script
# Model V2 (aka Adaption 1)
# SVD head
# Loss = Tf + Ch + MI


import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

#from data.kitti_data import KittiDataset
#from data.nuscenes_data import NuscenesDataset

from models import Model_V2
from losses import transformation_loss
from models.utils import set_seed

from tqdm import tqdm
import argparse
import wandb

#MAN Dataset
from config import Config
import dataset
import sys
import numpy as np

# MI and Chamfer distance loss
from losses import ChamferDistanceLoss
from losses import GlobalinfolossNet, LocalinfolossNet, DeepMILoss

def parse_args():
    parser = argparse.ArgumentParser(description='HRegNet')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--root', type=str, default='')
    parser.add_argument('--npoints', type=int, default=16384)
    parser.add_argument('--voxel_size', type=float, default=0.3)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--runname', type=str, default='')
    parser.add_argument('--dataset', type=str, default='man')
    parser.add_argument('--augment', type=float, default=0.0)
    parser.add_argument('--ckpt_dir', type=str, default='')
    parser.add_argument('--wandb_dir', type=str, default='')
    parser.add_argument('--freeze_detector', action='store_true')
    parser.add_argument('--freeze_feats', action='store_true')
    parser.add_argument('--use_fps', action='store_true')
    parser.add_argument('--data_list', type=str, default='')
    parser.add_argument('--use_weights', action='store_true')
    parser.add_argument('--pretrain_backbone_feats', type=str, default=None)
    parser.add_argument('--pretrain_model_feats', type=str, default=None)
    parser.add_argument('--alpha', type=float, default=1.0)
    
    return parser.parse_args()

def val_reg(args, net, chamfer_loss, mi_loss, max_val_c_loss, epoch):
    if args.dataset == 'kitti':
        val_seqs = ['06','07']
        #val_dataset = KittiDataset(args.root, val_seqs, args.npoints, args.voxel_size, args.data_list, 0.0)
    elif args.dataset == 'nuscenes':
        val_seqs = ['val']
        #val_dataset = NuscenesDataset(args.root, val_seqs, args.npoints, args.voxel_size, args.data_list, 0.0)
    elif args.dataset == 'man':
        config = Config(dataset='man')
        config.dataset_config.split = 'val'
        loader = dataset.TruckScenesLoader()
        man_data = dataset.TruckScenesDataset(loader(config, verbose=False), config)
        val_dataset = dataset.TruckScenesPerturbation(dataset=man_data, config=config)
        gt_Transformation = np.eye(4)
    elif args.dataset == 'audi':
        config = Config(dataset='audi')
        config.dataset_config.split = 'val'
        audi_data = dataset.A2D2Dataset(config=config)
        val_dataset = dataset.A2D2Perturbation(dataset=audi_data, config=config)
        gt_Transformation = np.eye(4)

    else:  
        raise('Not implemented')

    val_loader = DataLoader(val_dataset,
                            batch_size=config.dataset_config.batch_size,
                            num_workers=6,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=True)
    


    alpha = config.dataset_config.loss_weights[0]
    beta = config.dataset_config.loss_weights[1]

    # Switch both Model and MI Loss function to eval
    net.eval()
    mi_loss.eval()

    #print(f"mi_loss is on: {next(mi_loss.parameters()).device}")
    total_loss = 0
    total_c_loss = 0
    total_l_trans = 0
    total_js_loss = 0
    total_normalized_c_loss = 0
    total_R_err = 0.0
    total_t_err = 0.0
    total_geod_dist = 0.0
    total_eucl_dist = 0.0



    count = 0
    pbar = tqdm(enumerate(val_loader))
    with torch.no_grad():
    
        
        for i, data in pbar: 
            if args.dataset == 'man' or 'audi':
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

            ret_dict, corres_dict = net(src_points, dst_points)
            

            c_loss = chamfer_loss(ret_dict['src_xyz_2_trans'], ret_dict['dst_feats']['xyz_2'])
            js_loss = mi_loss(x_global        =corres_dict['src_dst_weights_2'],\
                                    x_global_prime =ret_dict['src_dst_weights_2_prime'],\
                                    x_local        =ret_dict['src_dst_feats_2'], \
                                    x_local_prime  =ret_dict['src_dst_feats_2_prime'],\
                                    c_local        =ret_dict['src_feats']['desc_2'],\
                                    c_global       =ret_dict['src_feats']['sigmas_2'])


            if epoch==0 and i==0:
                normalized_c_loss = c_loss.item() / (c_loss.item() + 1e-6)
                max_val_c_loss = c_loss.item()
            else:
                max_val_c_loss = max(max_val_c_loss, c_loss.item())
                normalized_c_loss = c_loss.item() / (max_val_c_loss + 1e-6)

            l_trans, _, _, R_err_, geodesic_dist, T_err_, eucl_dist\
                                     = transformation_loss(ret_dict['rotation'][-1], \
                                        ret_dict['translation'][-1],\
                                        gt_R, gt_t, args.alpha)
            


            total_loss += l_trans.item() + normalized_c_loss + js_loss.item()
            total_c_loss += c_loss.item()
            total_normalized_c_loss += normalized_c_loss
            total_js_loss += js_loss.item()
            total_l_trans += l_trans.item()

            total_R_err += R_err_
            total_t_err += T_err_

            total_geod_dist += geodesic_dist.mean().item()
            total_eucl_dist += eucl_dist.mean().item()


            count += 1
      
    total_loss = total_loss/count
    total_c_loss = total_c_loss/count
    total_normalized_c_loss = total_normalized_c_loss/count
    total_js_loss = total_js_loss/count
    total_l_trans = total_l_trans/count
    
    total_R_err += total_R_err/count
    total_t_err += total_t_err/count

    total_geod_dist += total_geod_dist/count 
    total_eucl_dist += total_eucl_dist/count

    if args.use_wandb:
        wandb.log({
                    f"val/rotation-x-error": total_R_err[0].item(),  # Rotation error along x-axis
                    f"val/rotation-y-error": total_R_err[1].item(),  # Rotation error along y-axis
                    f"val/rotation-z-error": total_R_err[2].item(),  # Rotation error along z-axis
                    f"val/geodesic-distance": total_geod_dist,
                    f"val/translation-error-x": total_t_err[0].item(),  # Translation error along x-axis
                    f"val/translation-error-y": total_t_err[1].item(),  # Translation error along y-axis
                    f"val/translation-error-z": total_t_err[2].item(),  # Translation error along z-axis
                    f"val/euclidean-distance": total_eucl_dist}, step=epoch+1)


    return total_loss, total_normalized_c_loss, total_c_loss, total_js_loss, total_l_trans, max_val_c_loss  


def train_reg(args):

    if args.dataset == 'kitti':
        train_seqs = ['00','01','02','03','04','05']
        #train_dataset = KittiDataset(args.root, train_seqs, args.npoints, args.voxel_size, args.data_list, args.augment)
    elif args.dataset == 'nuscenes':
        train_seqs = ['train']
        #train_dataset = NuscenesDataset(args.root, train_seqs, args.npoints, args.voxel_size, args.data_list, args.augment)
    
    elif args.dataset == 'man':
        config = Config(dataset='man')
        loader = dataset.TruckScenesLoader()
        config.dataset_config.split = 'train'
        #config.dataset_config.version = 'v1.0-test'
        man_data = dataset.TruckScenesDataset(loader(config, verbose=True), config)
        train_dataset = dataset.TruckScenesPerturbation(dataset=man_data, config=config)
        gt_Transformation = np.eye(4)
    
    elif args.dataset == 'audi':
        config = Config(dataset='audi')
        audi_data = dataset.A2D2Dataset(config=config)
        train_dataset = dataset.A2D2Perturbation(dataset=audi_data, config=config)
        gt_Transformation = np.eye(4)
  
    else:  
        raise('Not implemented')
    
    train_loader = DataLoader(train_dataset,
                              batch_size=config.dataset_config.batch_size,
                              num_workers=6,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(config=config.to_dict(), project='HRegNet', name=args.dataset+'_'+args.runname, dir=args.wandb_dir)


    # Initialize Model
    net = Model_V2(args)
    #net.feature_extraction.load_state_dict(torch.load(args.pretrain_feats))
    net.load_state_dict(torch.load(args.pretrain_model_feats)['net_state_dict'])
    
    # Initialize Losses
    chamfer_loss = ChamferDistanceLoss(scale=50.0, reduction='mean')
    mi_loss = DeepMILoss(global_in_channels=512,local_in_channels=128) # local_channels = Feature dim // 2

    mi_loss.load_state_dict(torch.load(args.pretrain_model_feats)['mi_loss_state_dict'])

    # Watch the network using wandb
    if args.use_wandb:
        wandb.watch(net)
        wandb.watch(mi_loss)
    
    # Move model and losses to GPU
    net.cuda()
    chamfer_loss.cuda()
    mi_loss.cuda()

    model_params = list(net.parameters()) + list(mi_loss.parameters())
    optimizer_net = optim.Adam(model_params, lr=args.lr)
    #optimizer_mi_loss = optim.Adam(, lr=args.lr)

    scheduler_net = StepLR(optimizer_net, step_size=10, gamma=0.5)
    #scheduler_mi_loss = StepLR(, step_size=10, gamma=0.5)

    best_train_loss = float('inf')
    best_val_loss = float('inf')
    alpha = config.dataset_config.loss_weights[0]
    beta = config.dataset_config.loss_weights[1]
    gamma = config.dataset_config.loss_weights[1]

    max_c_loss = 0.0
    max_val_c_loss = 0.0

    # Train loop
    for epoch in tqdm(range(args.epochs)):
        net.train()
        mi_loss.train()

        count = 0
        total_loss = 0
        total_c_loss = 0
        total_l_trans = 0
        total_js_loss = 0
        total_normalized_c_loss = 0

        total_R_err = 0.0
        total_t_err = 0.0

        total_geod_dist = 0.0
        total_eucl_dist = 0.0
        

        pbar = tqdm(enumerate(train_loader))

        for i, data in pbar:
            
            if args.dataset == 'man' or 'audi':
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

            optimizer_net.zero_grad()
            #optimizer_mi_loss.zero_grad()
            
            ret_dict, corres_dict = net(src_points, dst_points)

            # Chamfer Loss
            c_loss = chamfer_loss(ret_dict['src_xyz_2_trans'], ret_dict['dst_feats']['xyz_2'])
            
            # MI Loss
            js_loss = mi_loss(x_global        =corres_dict['src_dst_weights_2'],\
                                    x_global_prime =ret_dict['src_dst_weights_2_prime'],\
                                    x_local        =ret_dict['src_dst_feats_2'], \
                                    x_local_prime  =ret_dict['src_dst_feats_2_prime'],\
                                    c_local        =ret_dict['src_feats']['desc_2'],\
                                    c_global       =ret_dict['src_feats']['sigmas_2'])


            if epoch==0 and i==0:
                normalized_c_loss = c_loss.item() / (c_loss.item() + 1e-6)
                max_c_loss = c_loss.item()
            else:
                max_c_loss = max(max_c_loss, c_loss.item())
                normalized_c_loss = c_loss.item() / (max_c_loss + 1e-6)

            l_trans = 0.0
            l_R = 0.0
            l_t = 0.0

            # Transformation Loss
            # for idx in range(3):
            #     l_trans_, l_R_, l_t_, R_err_, geodesic_dist, T_err_, eucl_dist \
            #                                  = transformation_loss(ret_dict['rotation'][idx],\
            #                                                     ret_dict['translation'][idx],\
            #                                                     gt_R, gt_t, args.alpha)
                
            #     l_trans += l_trans_
            #     l_R += l_R_
            #     l_t += l_t_
            
            _, _, _, R_err_, geodesic_dist, T_err_, eucl_dist \
                                = transformation_loss(ret_dict['rotation'][-1],\
                                                ret_dict['translation'][-1],\
                                                gt_R, gt_t, args.alpha)
                

                
            
            #l_trans = l_trans / 3.0
            #loss = l_trans*alpha + normalized_c_loss*beta + js_loss*gamma
            loss = normalized_c_loss*beta + js_loss*gamma 

            loss.backward()
            
            optimizer_net.step()
            #optimizer_mi_loss.step()
            
            total_loss += loss.item()
            total_c_loss += c_loss.item()
            total_normalized_c_loss += normalized_c_loss
            #total_l_trans += l_trans.item()
            total_js_loss += js_loss.item()

            total_R_err += R_err_
            total_t_err += T_err_
            total_geod_dist += geodesic_dist.mean().item()
            total_eucl_dist += eucl_dist.mean().item()


            count += 1


            # if i % 10 == 0:
            #     pbar.set_description('Train Epoch:{}[{}/{}({:.0f}%)]\tLoss: {:.6f} \tchamfer_loss:{:.6f} \tMI_loss:{:.6f} \ttransformation loss:{:.6f}'.format(
            #         epoch+1, i, len(train_loader), 100. * i/len(train_loader), loss.item(), normalized_c_loss, js_loss.item(), l_trans.item()    
            #     ))
            
            if i % 10 == 0:
                pbar.set_description('Train Epoch:{}[{}/{}({:.0f}%)]\tLoss: {:.6f} \tchamfer_loss:{:.6f} \tMI_loss:{:.6f}'.format(
                    epoch+1, i, len(train_loader), 100. * i/len(train_loader), loss.item(), normalized_c_loss, js_loss.item()))

        total_loss = total_loss/count
        total_c_loss = total_c_loss/count
        total_normalized_c_loss = total_normalized_c_loss/count
        total_js_loss = total_js_loss/count
        #total_l_trans = total_l_trans/count
        total_R_err += total_R_err/count
        total_t_err += total_t_err/count
        total_geod_dist += total_geod_dist/count 
        total_eucl_dist += total_eucl_dist/count 
        
        # Validation loop
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        total_val_loss, val_normalized_c_loss, val_c_loss, val_js_loss, max_val_c_loss \
                                              = val_reg(args, net, chamfer_loss, mi_loss, max_val_c_loss, epoch)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        if args.use_wandb:
            wandb.log({"train/total loss":total_loss,
                       "train/normalized chamfer loss":total_normalized_c_loss,
                       "train/chamfer loss":total_c_loss,
                       "train/MI loss":total_js_loss,
                       #"train/transformation loss":total_l_trans,   
                       "train/learning rate":scheduler_net.get_last_lr()[0],
                        "train/rotation-x-error": total_R_err[0].item(),  # Rotation error along x-axis
                        "train/rotation-y-error": total_R_err[1].item(),  # Rotation error along y-axis
                        "train/rotation-z-error": total_R_err[2].item(),  # Rotation error along z-axis
                        "train/geodesic-distance": total_geod_dist,
                        "train/translation-error-x": total_t_err[0].item(),  # Translation error along x-axis
                        "train/translation-error-y": total_t_err[1].item(),  # Translation error along y-axis
                        "train/translation-error-z": total_t_err[2].item(),  # Translation error along z-axis
                        "train/euclidean-distance": total_eucl_dist,
                       "val/total loss": total_val_loss,
                       "val/normalized chamfer loss":val_normalized_c_loss,
                       "val/chamfer loss":val_c_loss,
                       "val/MI loss":val_js_loss}, step=epoch+1)
                       #"val/transformation loss":val_trans_loss}
                    

        print('\n Epoch {} finished. Training loss: {:.4f} Valiadation loss: {:.4f}'.\
            format(epoch+1, total_loss, total_val_loss))
        
        ckpt_dir = os.path.join(args.ckpt_dir, args.dataset + '_ckpt_'+args.runname)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        
        
        if total_val_loss < best_val_loss:
            torch.save({'net_state_dict': net.state_dict(),\
                        'mi_loss_state_dict': mi_loss.state_dict()},\
                         os.path.join(ckpt_dir, 'best_val.pth'))
            best_val_loss = total_val_loss
            best_val_epoch = epoch + 1
        
        print('Best val epoch: {} Best train loss: {:.4f} Best val epoch: {} Best val loss: {:.4f}'.format(
            best_val_epoch, best_train_loss, best_val_epoch, best_val_loss
        ))

        scheduler_net.step()
        #scheduler_mi_loss.step()


if __name__ == '__main__':
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seed(args.seed)

    train_reg(args)