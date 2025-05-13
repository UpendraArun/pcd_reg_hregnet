# Test script
# Any model
# with multi layer logging

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from torch.utils.data import DataLoader

#from data.kitti_data import KittiDataset
#from data.nuscenes_data import NuscenesDataset

from models import HRegNet, Model_V2, Model_V6
from models.utils import calc_error_np, set_seed

import argparse
import datetime
from tqdm import tqdm
import time

#MAN Dataset
from config import Config
import dataset
import sys
import numpy as np

from losses import transformation_loss
from metrics.calibeval import CalibEval, MultiLayerCalibEval
from dataset.data_loader import load_dataset
 
def parse_args():
    parser = argparse.ArgumentParser(description='HRegNet')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--root', type=str, default='')
    parser.add_argument('--npoints', type=int, default=16384)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--dataset', type=str, default='man')
    parser.add_argument('--use_fps', action='store_true')
    parser.add_argument('--data_list', type=str, default='')
    parser.add_argument('--use_weights', action='store_true')
    parser.add_argument('--pretrain_weights', type=str, default='')
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
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    #Initiailze the model
    net = Model_V6(args).cuda()

    config.dataset_config.model = net.__class__.__name__


    # Load the weights
    checkpoint = torch.load(args.pretrain_weights)
    if "net_state_dict" in checkpoint:
        net.load_state_dict(checkpoint["net_state_dict"],strict=False) 
    else:
        raise KeyError("Key 'net_state_dict' not found in checkpoint file")

    # Set the model to eval() mode
    net.eval()

    results_dir = os.path.join(args.save_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    config.dataset_config.results_path = results_dir

    # Initialize MultiLayer CalibEval
    calib_metrics = MultiLayerCalibEval(config=config, num_layers=3, translation_threshold=None,rotation_threshold=None)


    with torch.no_grad():
        total_time = 0.0
        num_batches = 0
        for i, data in enumerate(tqdm(test_loader, desc="Evaluating", unit="batch", total=len(test_loader))):
            # Retrieve inputs
            src_points = data['uncalibed_pcd'].contiguous().cuda()
            dst_points = data['pcd_left'].contiguous().cuda()
            igt = data['igt'].contiguous().cuda()

            # Synchronize GPU to ensure accurate timing
            #torch.cuda.synchronize()
            #start_time = time.time()

            ret_dict = net(src_points, dst_points)

            #torch.cuda.synchronize()
            #end_time = time.time()

            #batch_time = end_time - start_time
            #total_time += batch_time
            num_batches += 1

            # Process predictions (for calibration evaluation)
            pred_tf_1 = get_pred_tf(ret_dict['rotation'][0], ret_dict['translation'][0])
            pred_tf_2 = get_pred_tf(ret_dict['rotation'][1], ret_dict['translation'][1])
            pred_tf_3 = get_pred_tf(ret_dict['rotation'][2], ret_dict['translation'][2])
            
            calib_metrics.add_batch(layer=0, gt_tf=igt, pred_tf=pred_tf_1)
            calib_metrics.add_batch(layer=1, gt_tf=igt, pred_tf=pred_tf_2)
            calib_metrics.add_batch(layer=2, gt_tf=igt, pred_tf=pred_tf_3)

        #avg_time = total_time / num_batches
        #print(f"Average runtime per batch: {avg_time:.6f} seconds")
        calib_metrics.save_all_results(os.path.join(args.save_dir,"results.json"))

        

        


if __name__ == '__main__':
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seed(args.seed)

    test(args)