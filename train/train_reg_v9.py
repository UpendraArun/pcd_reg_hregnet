# Training script
# Model V4 
# SVD head
# Loss = Overlap + MI

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.amp import autocast, GradScaler


#from data.kitti_data import KittiDataset
#from data.nuscenes_data import NuscenesDataset

from models import Model_V4
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
from losses import OverlapAwareCircleLoss
from losses import GlobalinfolossNet, LocalinfolossNet, DeepMILoss

# Dataloader
from dataset.data_loader import load_dataset

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

def val_reg(val_loader, args, net, overlap_loss, mi_loss, max_val_c_loss, epoch, val_result_dict):
    config = Config(args)
    # val_dataset = load_dataset(config=config, split='val')

    # val_loader = DataLoader(val_dataset,
    #                         batch_size=config.dataset_config.batch_size,
    #                         num_workers=6,
    #                         shuffle=False,
    #                         pin_memory=True,
    #                         drop_last=True)


    alpha = config.dataset_config.loss_weights[0]
    beta = config.dataset_config.loss_weights[1]

    # Switch both Model and MI Loss function to eval
    net.eval()
    mi_loss.eval()

    #print(f"mi_loss is on: {next(mi_loss.parameters()).device}")
    total_loss = 0
    total_c_loss = 0
    #total_l_trans = 0
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
            gt_R = gt_R.cuda().detach()
            gt_t = gt_t.cuda().detach()

            ret_dict = net(src_points, dst_points)
            

            c_loss = overlap_loss(ret_dict['coord_dist'], ret_dict['feats_dist'])
            js_loss = mi_loss(x_global             =ret_dict['src_dst_weights_2'],\
                                    x_global_prime =ret_dict['src_dst_weights_2_prime'],\
                                    x_local        =ret_dict['src_dst_feats_2'], \
                                    x_local_prime  =ret_dict['src_dst_feats_2_prime'],\
                                    c_local        =ret_dict['src_feats_desc_2'],\
                                    c_global       =ret_dict['src_feats_sigmas_2'])


            if epoch==0 and i==0:
                normalized_c_loss = c_loss.item() / (c_loss.item() + 1e-6)
                max_val_c_loss = c_loss.item()
            else:
                max_val_c_loss = max(max_val_c_loss, c_loss.item())
                normalized_c_loss = c_loss.item() / (max_val_c_loss + 1e-6)

            _, _, _, R_err_, geodesic_dist, T_err_, eucl_dist\
                                     = transformation_loss(ret_dict['rotation'][-1].detach(), \
                                        ret_dict['translation'][-1].detach(),\
                                        gt_R.detach(), gt_t.detach(), args.alpha)
            


            #total_loss += l_trans.item() + normalized_c_loss + js_loss.item()
            total_loss += c_loss.item() + js_loss.item()
            total_c_loss += c_loss.item()
            total_normalized_c_loss += normalized_c_loss
            total_js_loss += js_loss.item()
            #total_l_trans += l_trans.item()

            total_R_err += R_err_
            total_t_err += T_err_

            total_geod_dist += geodesic_dist.mean().item()
            total_eucl_dist += eucl_dist.mean().item()


            count += 1
      
    total_loss = total_loss/count
    total_c_loss = total_c_loss/count
    total_normalized_c_loss = total_normalized_c_loss/count
    total_js_loss = total_js_loss/count
    
    total_R_err = total_R_err/count
    total_t_err = total_t_err/count

    total_geod_dist = total_geod_dist/count 
    total_eucl_dist = total_eucl_dist/count

    if args.use_wandb:
        wandb.log({
                    f"val/rotation-x-error": total_R_err[0].item(),  # Rotation error along x-axis
                    f"val/rotation-y-error": total_R_err[1].item(),  # Rotation error along y-axis
                    f"val/rotation-z-error": total_R_err[2].item(),  # Rotation error along z-axis
                    f"val/rotation-error-xyz": total_R_err.mean().item(),
                    f"val/geodesic-distance": total_geod_dist,
                    f"val/translation-error-x": total_t_err[0].item(),  # Translation error along x-axis
                    f"val/translation-error-y": total_t_err[1].item(),  # Translation error along y-axis
                    f"val/translation-error-z": total_t_err[2].item(),
                    f"val/translation-error-xyz": total_t_err.mean().item(),  # Translation error along z-axis
                    f"val/euclidean-distance": total_eucl_dist}, step=epoch+1)

    val_result_dict['val/loss'] = total_loss
    val_result_dict['val/RRE'] = total_geod_dist
    val_result_dict['val/RTE'] = total_eucl_dist
    val_result_dict['val/Rot_Err'] = total_R_err.mean().item()
    val_result_dict['val/Trans_Err'] = total_t_err.mean().item()

    return total_loss, total_normalized_c_loss, total_c_loss, total_js_loss, max_val_c_loss, val_result_dict


def train_reg(args):

    #torch.cuda.empty_cache()
    # Load dataset
    config = Config(args)
    train_dataset = load_dataset(config=config, split='train')
    
    train_loader = DataLoader(train_dataset,
                              batch_size=config.dataset_config.batch_size,
                              num_workers=6,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)
    
    val_dataset = load_dataset(config=config, split='val')

    val_loader = DataLoader(val_dataset,
                            batch_size=config.dataset_config.batch_size,
                            num_workers=6,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=True)

    print(f"Batch size is: {train_loader.batch_size}")

    # Create checkpoint directory
    ckpt_dir = os.path.join(args.ckpt_dir, args.dataset + '_ckpt_'+args.runname)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    

    # Initialize Model
    net = Model_V4(args)

    # Initialize Losses
    #chamfer_loss = ChamferDistanceLoss(scale=50.0, reduction='mean')
    overlap_loss = OverlapAwareCircleLoss()
    mi_loss = DeepMILoss(global_in_channels=512,local_in_channels=128) # local_channels = Feature dim // 2
    
     # Load pretrained data
    #net.feature_extraction.load_state_dict(torch.load(args.pretrain_feats))
    net.load_state_dict(torch.load(args.pretrain_model_feats)['net_state_dict'], strict=False)
    mi_loss.load_state_dict(torch.load(args.pretrain_model_feats)['mi_loss_state_dict'])

    # Initialize wandb and watch network
    if args.use_wandb:
        wandb.init(config=config.to_dict(), project='HRegNet', name=args.dataset+'_'+args.runname, dir=args.wandb_dir)
        wandb.watch([net, mi_loss])
    
    # Move model and losses to GPU
    net.cuda()
    overlap_loss.cuda()
    mi_loss.cuda()

    # Initialize Optimizer and Scheduler
    model_params = list(net.parameters()) + list(mi_loss.parameters())
    optimizer_net = optim.Adam(model_params, lr=args.lr)
    scheduler_net = StepLR(optimizer_net, step_size=10, gamma=0.5)

    scaler = GradScaler()

    # Initialize the val helper dicts
    val_result_dict = {
        "val/loss": float('inf'),
        "val/RRE": float('inf'),
        "val/RTE": float('inf'),
        "val/Rot_Err": float('inf'),
        "val/Trans_Err": float('inf')}

    best_val_result_dict = val_result_dict.copy()

    train_result_dict = {}
    best_train_result_dict = {}

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
                src_points = data['uncalibed_pcd'].cuda()
                dst_points = data['pcd_left'].cuda()

                gt_Transformation = torch.inverse(data['igt'])

                gt_R = gt_Transformation[:,:3,:3].contiguous().cuda().detach()
                gt_t = gt_Transformation[:,:3,3].contiguous().cuda().detach()

            #src_points = src_points.cuda()
            #dst_points = dst_points.cuda()
            #gt_R = gt_R.cuda().detach()
            #gt_t = gt_t.cuda().detach()

            optimizer_net.zero_grad()
            #with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            ret_dict = net(src_points, dst_points)

            # Chamfer Loss
            c_loss = overlap_loss(ret_dict['coord_dist'], ret_dict['feats_dist'])
            
            # MI Loss
            js_loss = mi_loss(x_global             =ret_dict['src_dst_weights_2'],\
                                    x_global_prime =ret_dict['src_dst_weights_2_prime'],\
                                    x_local        =ret_dict['src_dst_feats_2'], \
                                    x_local_prime  =ret_dict['src_dst_feats_2_prime'],\
                                    c_local        =ret_dict['src_feats_desc_2'],\
                                    c_global       =ret_dict['src_feats_sigmas_2'])


            if epoch==0 and i==0:
                normalized_c_loss = c_loss.item() / (c_loss.item() + 1e-6)
                max_c_loss = c_loss.item()
            else:
                max_c_loss = max(max_c_loss, c_loss.item())
                normalized_c_loss = c_loss.item() / (max_c_loss + 1e-6)

            # Calculate Rotation and Translation errors
            _, _, _, R_err_, geodesic_dist, T_err_, eucl_dist \
                                = transformation_loss(ret_dict['rotation'][-1].detach(),\
                                                ret_dict['translation'][-1].detach(),\
                                                gt_R.detach(), gt_t.detach(), args.alpha)
                

            #l_trans = l_trans / 3.0
            #loss = l_trans*alpha + normalized_c_loss*beta + js_loss*gamma
            # loss = normalized_c_loss*config.dataset_config.loss_weights[1]\
            #         + js_loss*config.dataset_config.loss_weights[0]

            loss = c_loss*config.dataset_config.loss_weights[1]\
                     + js_loss*config.dataset_config.loss_weights[0]

            loss.backward()
            

            
            optimizer_net.step()

            # scaler.scale(loss).backward()
            # scaler.step(optimizer_net)
            # scaler.update()
            
            total_loss += loss.item()
            total_c_loss += c_loss.item()
            total_normalized_c_loss += normalized_c_loss
            total_js_loss += js_loss.item()

            total_R_err += R_err_
            total_t_err += T_err_
            total_geod_dist += geodesic_dist.mean().item()
            total_eucl_dist += eucl_dist.mean().item()

            count += 1
            
            if i % 10 == 0:
                pbar.set_description('Train Epoch:{}[{}/{}({:.0f}%)]\tLoss: {:.6f} \t overlap_loss:{:.6f} \tMI_loss:{:.6f}'.format(
                    epoch+1, i, len(train_loader), 100. * i/len(train_loader), loss.item(), c_loss.item(), js_loss.item()))

        total_loss = total_loss/count
        total_c_loss = total_c_loss/count
        total_normalized_c_loss = total_normalized_c_loss/count
        total_js_loss = total_js_loss/count
        total_R_err = total_R_err/count
        total_t_err = total_t_err/count
        total_geod_dist = total_geod_dist/count 
        total_eucl_dist = total_eucl_dist/count 
        
        # Validation loop
        #torch.cuda.synchronize()
        #torch.cuda.empty_cache()
        total_val_loss, val_normalized_c_loss, val_c_loss, val_js_loss, max_val_c_loss, val_result_dict \
                                              = val_reg(val_loader, args, net, overlap_loss, mi_loss, max_val_c_loss, epoch, val_result_dict)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        if args.use_wandb:
            wandb.log({"train/total loss":total_loss,
                       "train/normalized overlap loss":total_normalized_c_loss,
                       "train/overlap loss":total_c_loss,
                       "train/MI loss":total_js_loss,
                       "train/learning rate":scheduler_net.get_last_lr()[0],
                        "train/rotation-x-error": total_R_err[0].item(),  # Rotation error along x-axis
                        "train/rotation-y-error": total_R_err[1].item(),  # Rotation error along y-axis
                        "train/rotation-z-error": total_R_err[2].item(),  # Rotation error along z-axis
                        "train/rotation-error-xyz": total_R_err.mean().item(),
                        "train/geodesic-distance": total_geod_dist,
                        "train/translation-error-x": total_t_err[0].item(),  # Translation error along x-axis
                        "train/translation-error-y": total_t_err[1].item(),  # Translation error along y-axis
                        "train/translation-error-z": total_t_err[2].item(),  # Translation error along z-axis
                        "train/translation-error-xyz": total_t_err.mean().item(),
                        "train/euclidean-distance": total_eucl_dist,
                       "val/total loss": total_val_loss,
                       "val/normalized overlap loss":val_normalized_c_loss,
                       "val/overlap loss":val_c_loss,
                       "val/MI loss":val_js_loss}, step=epoch+1)
                    

        print('\n Epoch {} finished. Training loss: {:.4f} Valiadation loss: {:.4f}'.\
            format(epoch+1, total_loss, total_val_loss))
        
        ckpt_dir = os.path.join(args.ckpt_dir, args.dataset + '_ckpt_'+args.runname)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        
        
        for key in val_result_dict:
            if abs(val_result_dict[key]) < abs(best_val_result_dict[key]):  
                print(f"Got a better {key} with value {val_result_dict[key]}")  
                best_val_result_dict[key] = val_result_dict[key]
                
                torch.save({'net_state_dict': net.state_dict(),
                        'mi_loss_state_dict': mi_loss.state_dict(),
                         'optimizer_state_dict': optimizer_net.state_dict(),
                         'scheduler_state_dict': scheduler_net.state_dict(),
                         'epoch': epoch},os.path.join(ckpt_dir, f"best_{key.replace('/', '_')}.pth"))
        
        print("Best Validation Results:", ", ".join(f"{key}: {value:.4f}" for key, value in best_val_result_dict.items()))

        scheduler_net.step()


if __name__ == '__main__':
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seed(args.seed)

    train_reg(args)