import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import KeypointDetectorSelfAttention, MultiHeadCrossAttention, CorrespondenceEstimator 
from .layers import WeightedSVDHead, RegressionHead, Regression_6dR_3dt_Head
from transform import SO3

class HierFeatureExtraction(nn.Module):
    def __init__(self, args):
        super(HierFeatureExtraction, self).__init__()

        self.use_fps = args.use_fps
        self.use_weights = args.use_weights

        self.detector_1 = KeypointDetectorSelfAttention(nsample=1024, k=64, in_channels=0, out_channels=[32,32,64], fps=self.use_fps)
        self.detector_2 = KeypointDetectorSelfAttention(nsample=512, k=32, in_channels=64, out_channels=[64,64,128], fps=self.use_fps)
        self.detector_3 = KeypointDetectorSelfAttention(nsample=256, k=16, in_channels=128, out_channels=[128,128,256], fps=self.use_fps)

        if args.freeze_detector:
            for p in self.parameters():
                p.requires_grad = False
        
        
    
    def forward(self, points):
        
        xyz_1, sigmas_1, attentive_feature_1 = self.detector_1(points, None)
        
        #print(xyz_1.shape, sigmas_1.shape, attentive_feature_1.shape, grouped_features_1.shape, attentive_feature_map_1.shape)
        
        
        if self.use_weights:
            weights_1 = 1.0/(sigmas_1 + 1e-5)
            weights_1_mean = torch.mean(weights_1, dim=1, keepdim=True)
            weights_1 = weights_1 / weights_1_mean
            
            xyz_2, sigmas_2, attentive_feature_2 = self.detector_2(xyz_1, attentive_feature_1, weights_1)
            

            weights_2 = 1.0/(sigmas_2 + 1e-5)
            weights_2_mean = torch.mean(weights_2, dim=1, keepdim=True)
            weights_2 = weights_2 / weights_2_mean
            
            xyz_3, sigmas_3, attentive_feature_3 = self.detector_3(xyz_2, attentive_feature_2, weights_2)
            
        else:

            xyz_2, sigmas_2, attentive_feature_2 = self.detector_2(xyz_1, attentive_feature_1)
            xyz_3, sigmas_3, attentive_feature_3 = self.detector_3(xyz_2, attentive_feature_2)

        
        ret_dict = {}
        
        
        ret_dict['xyz_1'] = xyz_1
        ret_dict['xyz_2'] = xyz_2
        ret_dict['xyz_3'] = xyz_3
        ret_dict['sigmas_1'] = sigmas_1
        ret_dict['sigmas_2'] = sigmas_2
        ret_dict['sigmas_3'] = sigmas_3
        ret_dict['attentive_feat_map_1'] = attentive_feature_1
        ret_dict['attentive_feat_map_2'] = attentive_feature_2
        ret_dict['attentive_feat_map_3'] = attentive_feature_3

        return ret_dict

class Model_V5(nn.Module):
    
    def __init__(self, args):
        
        super(Model_V5, self).__init__()
        self.feature_extraction = HierFeatureExtraction(args)

        self.cross_attn_3 = MultiHeadCrossAttention(feature_dim=256, num_heads=4)
        self.cross_attn_2 = MultiHeadCrossAttention(feature_dim=128, num_heads=4)
        self.cross_attn_1 = MultiHeadCrossAttention(feature_dim=64, num_heads=4)

        self.corres_3 = CorrespondenceEstimator()
        self.corres_2 = CorrespondenceEstimator()
        self.corres_1 = CorrespondenceEstimator()

        self.svd_head = WeightedSVDHead()


    def forward(self, src_points, dst_points):
        
        # Extract multi-level keypoints and features
        src = self.feature_extraction(src_points)
        dst = self.feature_extraction(dst_points)

        # LEVEL 3: Coarse Registration
        attn_feats_3, attn_weights_3 = self.cross_attn_3(src['attentive_feat_map_3'], dst['attentive_feat_map_3'])
        corres_xyz_3, corres_weights_3 = self.corres_3(dst['xyz_3'], attn_feats_3, attn_weights_3, src['sigmas_3'])

        R3, t3 = self.svd_head(src['xyz_3'], corres_xyz_3, corres_weights_3)

        # Transform Level 2 keypoints using R3, t3
        src_xyz_2_trans = (R3 @ src['xyz_2'].permute(0,2,1)) + t3.unsqueeze(2)
        src_xyz_2_trans = src_xyz_2_trans.permute(0,2,1)

        # LEVEL 2: Fine Registration
        #print(f"feats_left shape: {src['attentive_feat_map_2'].shape}")  # Expected [B, N, C]
        #print(f"feats_left shape: {dst['attentive_feat_map_2'].shape}")  # Expected [B, N, C]

        attn_feats_2, attn_weights_2 = self.cross_attn_2(src['attentive_feat_map_2'], dst['attentive_feat_map_2'])
        corres_xyz_2, corres_weights_2 = self.corres_2(dst['xyz_2'], attn_feats_2, attn_weights_2, src['sigmas_2'])

        R2_, t2_ = self.svd_head(src_xyz_2_trans, corres_xyz_2, corres_weights_2)

        # Compute accumulated transformation T2
        T3 = torch.eye(4, device=R3.device).repeat(R3.shape[0], 1, 1)
        T3[:, :3, :3] = R3
        T3[:, :3, 3] = t3.squeeze()

        T2_ = torch.eye(4, device=R2_.device).repeat(R2_.shape[0], 1, 1)
        T2_[:, :3, :3] = R2_
        T2_[:, :3, 3] = t2_.squeeze()

        T2 = T2_ @ T3  # Accumulate transformation
        R2, t2 = T2[:, :3, :3], T2[:, :3, 3]

        # Transform Level 1 keypoints using R2, t2
        src_xyz_1_trans = (R2 @ src['xyz_1'].permute(0,2,1)) + t2.unsqueeze(2)
        src_xyz_1_trans = src_xyz_1_trans.permute(0,2,1)

        # **LEVEL 1: FINEST ALIGNMENT**
        attn_feats_1, attn_weights_1 = self.cross_attn_1(src['attentive_feat_map_1'], dst['attentive_feat_map_1'])
        corres_xyz_1, corres_weights_1 = self.corres_1(dst['xyz_1'], attn_feats_1, attn_weights_1, src['sigmas_1'])

        R1_, t1_ = self.svd_head(src_xyz_1_trans, corres_xyz_1, corres_weights_1)

        # Compute final transformation T1
        T1_ = torch.eye(4, device=R1_.device).repeat(R1_.shape[0], 1, 1)
        T1_[:, :3, :3] = R1_
        T1_[:, :3, 3] = t1_.squeeze()

        T1 = T1_ @ T2 
        R1, t1 = T1[:, :3, :3], T1[:, :3, 3]

        
        ret_dict = {}
        ret_dict['rotation'] = [R3, R2, R1] # Tf loss - Rotations
        ret_dict['translation'] = [t3, t2, t1] # Tf loss - Translations
        
        # ret_dict['src_feats_desc_2'] = src_feats['desc_2'] # MI Loss - c_local
        # ret_dict['src_feats_sigmas_2'] = src_feats['sigmas_2'] # MI Loss - c_global
        
        # ret_dict['src_xyz_2_trans'] = src_xyz_2_trans # chamfer loss - src points
        # ret_dict['dst_xyz_2'] = dst_feats['xyz_2'] # chamfer loss - dst points

        # ret_dict['src_dst_feats_2'] = src_dst_feats_2 #x_local
        # ret_dict['src_dst_feats_2_prime'] = src_dst_feats_2_prime #x_local_prime
        
        
        # ret_dict['src_dst_weights_2'] = src_dst_weights_2 #x_global
        # ret_dict['src_dst_weights_2_prime'] = src_dst_weights_2_prime #x_global_prime

        
        return ret_dict

if __name__ == '__main__':
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser('HRegNet')

        parser.add_argument('--npoints', type=int, default=16384, help='number of input points')
        parser.add_argument('--freeze_detector', action='store_true')
        parser.add_argument('--use_fps', action='store_false')
        parser.add_argument('--freeze_features', action='store_true')
        parser.add_argument('--use_weights', action='store_true')

        return parser.parse_args()
    
    args = parse_args()
    args.use_fps = True
    args.use_weights = True
    model = Model_V5(args).cuda()
    xyz1 = torch.rand(2,16384,3).cuda()
    xyz2 = torch.rand(2,16384,3).cuda()
    ret_dict = model(xyz1, xyz2)
    print(ret_dict['rotation'][-1].shape)
    print(ret_dict['translation'][-1].shape)