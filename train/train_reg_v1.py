# Training script
# HRegNet 
# Regression head
# Loss = Tf

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

#from data.kitti_data import KittiDataset
#from data.nuscenes_data import NuscenesDataset

from models import HRegNet
from losses import transformation_loss
from models.utils import set_seed

from tqdm import tqdm
import argparse
import wandb

#MAN Dataset
from config import Config
import sys

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

def val_reg(val_loader,args, net, epoch, val_result_dict):


    net.eval()

    total_loss = 0
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

            ret_dict = net(src_points, dst_points)
            

            l_trans, _, _, R_err_, geodesic_dist, T_err_, eucl_dist\
                                     = transformation_loss(ret_dict['rotation'][-1], \
                                        ret_dict['translation'][-1],\
                                        gt_R, gt_t, args.alpha)
            


            total_loss += l_trans.item()

            total_R_err += R_err_
            total_t_err += T_err_

            total_geod_dist += geodesic_dist.mean().item()
            total_eucl_dist += eucl_dist.mean().item()


            count += 1

    if count > 0:  
        total_loss = total_loss/count
        
        total_R_err = total_R_err/count
        total_t_err = total_t_err/count

        total_geod_dist = total_geod_dist/count 
        total_eucl_dist = total_eucl_dist/count

    if args.use_wandb:
        wandb.log({
                    f"val/rotation-error-x": total_R_err[0].item(),  # Rotation error along x-axis
                    f"val/rotation-error-y": total_R_err[1].item(),  # Rotation error along y-axis
                    f"val/rotation-error-z": total_R_err[2].item(),  # Rotation error along z-axis
                    f"val/rotation-error-xyz": total_R_err.mean().item(), # Rotation error mean
                    f"val/geodesic-distance": total_geod_dist,
                    f"val/translation-error-x": total_t_err[0].item(),  # Translation error along x-axis
                    f"val/translation-error-y": total_t_err[1].item(),  # Translation error along y-axis
                    f"val/translation-error-z": total_t_err[2].item(),  # Translation error along z-axis
                    f"val/translation-error-xyz": total_t_err.mean().item(), # Translation error mean
                    f"val/euclidean-distance": total_eucl_dist}, step=epoch+1)

    val_result_dict['val/loss'] = total_loss
    val_result_dict['val/RRE'] = total_geod_dist
    val_result_dict['val/RTE'] = total_eucl_dist
    val_result_dict['val/Rot_Err'] = total_R_err.mean().item()
    val_result_dict['val/Trans_Err'] = total_t_err.mean().item()

    return total_loss, val_result_dict


def train_reg(args):

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

    # Create checkpoint directory
    ckpt_dir = os.path.join(args.ckpt_dir, args.dataset + '_ckpt_'+args.runname)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
    
    # Initialize Model
    net = HRegNet(args)

    # Load pretrained data

    net.load_state_dict(torch.load(args.pretrain_model_feats)['net_state_dict'], strict=False)
  

    # Initialize wandb and watch network
    if args.use_wandb:
        wandb.init(config=config.to_dict(), project='HRegNet', name=args.dataset+'_'+args.runname, dir=args.wandb_dir)
        wandb.watch(net)

    
    # Move model and losses to GPU
    net.cuda()


    # Initialize Optimizer and Scheduler
    optimizer_net = optim.Adam(net.parameters(), lr=args.lr)
    scheduler_net = StepLR(optimizer_net, step_size=10, gamma=0.5)

    # Initialize the val helper dicts
    val_result_dict = {
        "val/loss": float('inf'),
        "val/RRE": float('inf'),
        "val/RTE": float('inf'),
        "val/Rot_Err": float('inf'),
        "val/Trans_Err": float('inf')}

    best_val_result_dict = val_result_dict.copy()

    # Train loop
    for epoch in tqdm(range(args.epochs)):
        
        # Set the model to train()
        net.train()

        count = 0
        total_loss = 0
        total_l_trans = 0

        total_R_err = 0.0
        total_t_err = 0.0

        total_geod_dist = 0.0
        total_eucl_dist = 0.0
        

        pbar = tqdm(enumerate(train_loader))

        for i, data in pbar:
            
            if args.dataset == 'man' or 'audi':
                src_points = data['uncalibed_pcd']
                dst_points = data['pcd_left']

                gt_Transformation = torch.inverse(data['igt'])

                gt_R = gt_Transformation[:,:3,:3].contiguous()
                gt_t = gt_Transformation[:,:3,3].contiguous()

            src_points = src_points.cuda()
            dst_points = dst_points.cuda()
            gt_R = gt_R.cuda()
            gt_t = gt_t.cuda()

            optimizer_net.zero_grad()
            
            ret_dict = net(src_points, dst_points)

            l_trans = 0.0
            l_R = 0.0
            l_t = 0.0

            # Transformation Loss
            for idx in range(3):
                l_trans_, l_R_, l_t_, R_err_, geodesic_dist, T_err_, eucl_dist \
                                             = transformation_loss(ret_dict['rotation'][idx],\
                                                                ret_dict['translation'][idx],\
                                                                gt_R, gt_t, args.alpha)
                
                l_trans += l_trans_
                l_R += l_R_
                l_t += l_t_
                

                
            
            l_trans = l_trans / 3.0

            loss = l_trans*config.dataset_config.loss_weights[0]

            loss.backward()
            
            optimizer_net.step()
            
            total_loss += loss.item()

            total_R_err += R_err_
            total_t_err += T_err_
            total_geod_dist += geodesic_dist.mean().item()
            total_eucl_dist += eucl_dist.mean().item()


            count += 1


            if i % 10 == 0:
                pbar.set_description('Train Epoch:{}[{}/{}({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, i, len(train_loader), 100. * i/len(train_loader), loss.item()))

        if count > 0: 
            total_loss = total_loss/count

            total_R_err = total_R_err/count
            total_t_err = total_t_err/count

            total_geod_dist = total_geod_dist/count 
            total_eucl_dist = total_eucl_dist/count 
        
        # Validation loop
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        total_val_loss, val_result_dict = val_reg(val_loader, args, net, epoch, val_result_dict)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        if args.use_wandb:
            wandb.log({"train/total loss":total_loss,
                       "train/transformation loss":total_l_trans,   
                       "train/learning rate":scheduler_net.get_last_lr()[0],
                        "train/rotation-error-x": total_R_err[0].item(),  # Rotation error along x-axis
                        "train/rotation-error-y": total_R_err[1].item(),  # Rotation error along y-axis
                        "train/rotation-error-z": total_R_err[2].item(),  # Rotation error along z-axis
                        "train/rotation-error-xyz": total_R_err.mean().item(),
                        "train/geodesic-distance": total_geod_dist,
                        "train/translation-error-x": total_t_err[0].item(),  # Translation error along x-axis
                        "train/translation-error-y": total_t_err[1].item(),  # Translation error along y-axis
                        "train/translation-error-z": total_t_err[2].item(),  # Translation error along z-axis
                        "train/translation-error-xyz": total_t_err.mean().item(),
                        "train/euclidean-distance": total_eucl_dist,
                       "val/total loss": total_val_loss}, step=epoch+1)
        

        print('\n Epoch {} finished. Training loss: {:.4f} Validation loss: {:.4f}'.\
            format(epoch+1, total_loss, total_val_loss))


        for key in val_result_dict:
            if abs(val_result_dict[key]) < abs(best_val_result_dict[key]):  
                print(f"Got a better {key} with value {val_result_dict[key]}")  
                best_val_result_dict[key] = val_result_dict[key]
                
                torch.save({'net_state_dict': net.state_dict(),
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