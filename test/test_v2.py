# Test script
# Any model
# without multi layer logging

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from torch.utils.data import DataLoader

#from data.kitti_data import KittiDataset
#from data.nuscenes_data import NuscenesDataset

from models.model_v3.models import Model_V3
from models.utils import calc_error_np, set_seed

import argparse
import datetime
from tqdm import tqdm

#MAN Dataset
from config import Config
import dataset
import sys
import numpy as np

from losses import transformation_loss
from metrics.calibeval import CalibEval
from dataset.data_loader import load_dataset
 
def parse_args():
    parser = argparse.ArgumentParser(description='HRegNet')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--root', type=str, default='')
    parser.add_argument('--npoints', type=int, default=16384)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dataset', type=str, default='man')
    parser.add_argument('--use_fps', action='store_true')
    parser.add_argument('--data_list', type=str, default='')
    parser.add_argument('--use_weights', action='store_true')
    parser.add_argument('--pretrain_weights', type=str, default='/workspace/ckpt/man_ckpt_T10_Registration/best_val.pth')
    parser.add_argument('--voxel_size', type=float, default=0.3)
    parser.add_argument('--save_dir',type=str, default='')
    parser.add_argument('--augment', type=float, default=0.0)
    parser.add_argument('--freeze_detector', action='store_true')
    parser.add_argument('--freeze_feats', action='store_true')
    
    return parser.parse_args()

def get_pred_tf(pred_R:torch.Tensor, pred_T:torch.Tensor) -> torch.Tensor:
        
        B = pred_R.shape[0]  # Batch size

        # Create an identity matrix of shape (B, 4, 4)
        transformation_matrix = torch.eye(4, device=pred_R.device).repeat(B, 1, 1)

        # Assign rotation part (top-left 3x3)
        transformation_matrix[:, :3, :3] = pred_R

        # Assign translation part (top-right column)
        transformation_matrix[:, :3, 3] = pred_T

        return transformation_matrix


def test(args):

     # Load dataset
    config = Config(args)
    test_dataset = load_dataset(config=config, split='test')

    # Initialize the Dataloader
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    #Initiailze the model
    net = Model_V3(args).cuda()
    
    # Load the weights
    checkpoint = torch.load(args.pretrain_weights)
    if "net_state_dict" in checkpoint:
        net.load_state_dict(checkpoint["net_state_dict"]) 
    else:
        raise KeyError("Key 'net_state_dict' not found in checkpoint file")

    # Set the model to eval() mode
    net.eval()

    results_dir = os.path.join(args.save_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    config.dataset_config.results_path = results_dir

    # Initialize CalibEval
    calib_metrics = CalibEval(config=config, translation_threshold=None, rotation_threshold=None)
    calib_metrics.reset()

    with torch.no_grad():
        
        for i, data  in enumerate(tqdm(test_loader, desc="Evaluating", unit="sample", total=len(test_loader.dataset))):
            
            #start_t = datetime.datetime.now()
            src_points = data['uncalibed_pcd'].contiguous().cuda()
            dst_points = data['pcd_left'].contiguous().cuda()
            igt = data['igt'].contiguous().cuda()

            ret_dict = net(src_points, dst_points)
            
            #end_t = datetime.datetime.now()

            pred_tf = get_pred_tf(ret_dict['rotation'][-1], ret_dict['translation'][-1])
            calib_metrics.add_batch(igt, pred_tf)

        calib_metrics.get_stats()
        rot_err, trans_err, geodesic = calib_metrics.get_stats()
        sd_rot, sd_trans, sd_dR, sd_dT = calib_metrics.getSD()

        metrics_dict = {
            'rot': rot_err.mean().item(),
            'trans': trans_err.mean().item(),
            'sd_rot': sd_rot.mean().item(),
            'sd_trans': sd_trans.mean().item(),
            'dR': geodesic[0],
            'dT': geodesic[1],
            'sd_dR': sd_dR.mean().item(),
            'sd_dT': sd_dT.mean().item()
        }


        calib_metrics.save_results()
        
        for k,v in metrics_dict.items():
            print(k,v)


if __name__ == '__main__':
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seed(args.seed)

    test(args)