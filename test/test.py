
# Test script
# Any model 
# without detailed logging, only final results logged

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch

#from data.kitti_data import KittiDataset
#from data.nuscenes_data import NuscenesDataset

from models import HRegNet
from models.utils import calc_error_np, set_seed

import argparse
import datetime

#MAN Dataset
from config import Config
import dataset
import sys
import numpy as np
 
def parse_args():
    parser = argparse.ArgumentParser(description='HRegNet')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--root', type=str, default='')
    parser.add_argument('--npoints', type=int, default=16384)
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

def test(args):
    if args.dataset == 'kitti':
        test_seqs = ['08','09','10']
        #test_dataset = KittiDataset(args.root, test_seqs, args.npoints, args.voxel_size, args.data_list, args.augment)
    elif args.dataset == 'nuscenes':
        test_seqs = ['test']
        #test_dataset = NuscenesDataset(args.root, test_seqs, args.npoints, args.voxel_size, args.data_list, args.augment)
    elif args.dataset == 'man':
        config = Config(dataset='man')
        if config.dataset_config.version == 'v1.0-mini':
            config.dataset_config.split = 'val'
        elif config.dataset_config.version == 'v1.0-test':
            config.dataset_config.split = 'test'

        loader = dataset.TruckScenesLoader()
        man_data = dataset.TruckScenesDataset(loader(config, verbose=True), config)
        test_dataset = dataset.TruckScenesPerturbation(dataset=man_data, config=config)
        gt_Transformation = np.eye(4)
    elif args.dataset == 'audi':
        config = Config(dataset='audi')
        config.dataset_config.split = 'test'
        audi_data = dataset.A2D2Dataset(config=config)
        test_dataset = dataset.A2D2Perturbation(dataset=audi_data, config=config)
        gt_Transformation = np.eye(4)
    else:  
        raise('Not implemented')

    
    net = HRegNet(args).cuda()
    net.load_state_dict(torch.load(args.pretrain_weights))

    net.eval()

    trans_error_list = []
    rot_error_list = []
    pred_T_list = []
    delta_t_list = []
    trans_thresh = 2.0
    rot_thresh = 5.0
    success_idx = []

    with torch.no_grad():
        for idx in range(test_dataset.__len__()):
            start_t = datetime.datetime.now()
            
            # src_points, dst_points, gt_R, gt_t = test_dataset[idx]

            src_points = test_dataset[idx]['uncalibed_pcd'].contiguous()
            #print(src_points.shape)
            dst_points = test_dataset[idx]['pcd_left'].contiguous()
            #print(dst_points.shape)
            gt_Transformation = test_dataset[idx]['igt']
            gt_Transformation = torch.inverse(gt_Transformation)
            gt_R = gt_Transformation[:3,:3].contiguous()
            gt_t = gt_Transformation[:3,3].contiguous()
            #assert gt_R.is_contiguous()
            #assert gt_t.is_contiguous()
            
            src_points = src_points.unsqueeze(0).cuda()
            #assert src_points.is_contiguous()
            #print(src_points.shape)
            dst_points = dst_points.unsqueeze(0).cuda()
            #print(dst_points.shape)
            assert dst_points.is_contiguous()
            gt_R = gt_R.numpy()
            gt_t = gt_t.numpy()

            ret_dict, _ = net(src_points, dst_points)
            
            end_t = datetime.datetime.now()
            
            pred_R = ret_dict['rotation'][-1]
            pred_t = ret_dict['translation'][-1]
            pred_R = pred_R.squeeze().cpu().numpy()
            pred_t = pred_t.squeeze().cpu().numpy()
            rot_error, trans_error = calc_error_np(pred_R, pred_t, gt_R, gt_t)
            pred_T = np.zeros((3,4))
            gt_T = np.zeros((3,4))
            pred_T[:3,:3] = pred_R
            pred_T[:3,3] = pred_t
            gt_T[:3,:3] = gt_R
            gt_T[:3,3] = gt_t
            pred_T = pred_T.flatten()
            gt_T = gt_T.flatten()
            pred_T_list.append(pred_T)
            print('{:d}: trans: {:.4f} rot: {:.4f}'.format(idx, trans_error, rot_error))
            trans_error_list.append(trans_error)
            rot_error_list.append(rot_error)
            
            if trans_error < trans_thresh and rot_error < rot_thresh:
                success_idx.append(idx)
            
            delta_t = (end_t - start_t).microseconds
            delta_t_list.append(delta_t)
    
    success_rate = len(success_idx)/test_dataset.__len__()
    trans_error_array = np.array(trans_error_list)
    rot_error_array = np.array(rot_error_list)
    trans_mean = np.mean(trans_error_array[success_idx])
    trans_std = np.std(trans_error_array[success_idx])
    rot_mean = np.mean(rot_error_array[success_idx])
    rot_std = np.std(rot_error_array[success_idx])
    delta_t_array = np.array(delta_t_list)
    delta_t_mean = np.mean(delta_t_array)

    print('Translation mean: {:.4f}'.format(trans_mean))
    print('Translation std: {:.4f}'.format(trans_std))
    print('Rotation mean: {:.4f}'.format(rot_mean))
    print('Rotation std: {:.4f}'.format(rot_std))
    print('Runtime: {:.4f}'.format(delta_t_mean))
    print('Success rate: {:.4f}'.format(success_rate))
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    pred_T_array = np.array(pred_T_list)
    np.savetxt(os.path.join(args.save_dir, args.dataset+'_pred.txt'), pred_T_array)
    np.savetxt(os.path.join(args.save_dir, args.dataset+'_trans_error.txt'), trans_error_list)
    np.savetxt(os.path.join(args.save_dir, args.dataset+'_rot_error.txt'), rot_error_list)

    f_summary = open(os.path.join(args.save_dir, args.dataset+'_summary.txt'), 'w')
    f_summary.write('Dataset: '+args.dataset+'\n')
    f_summary.write('Translation threshold: {:.2f}\n'.format(trans_thresh))
    f_summary.write('Rotation threshold: {:.2f}\n'.format(rot_thresh))
    f_summary.write('Translation mean: {:.4f}\n'.format(trans_mean))
    f_summary.write('Translation std: {:.4f}\n'.format(trans_std))
    f_summary.write('Rotation mean: {:.4f}\n'.format(rot_mean))
    f_summary.write('Rotation std: {:.4f}\n'.format(rot_std))
    f_summary.write('Runtime: {:.4f}\n'.format(delta_t_mean))
    f_summary.write('Success rate: {:.4f}\n'.format(success_rate))
    f_summary.close()

    print('Saved results to ' + args.save_dir)

if __name__ == '__main__':
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seed(args.seed)

    test(args)