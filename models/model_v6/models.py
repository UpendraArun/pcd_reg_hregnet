import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import KeypointDetector, CoarseReg, FineReg1, FineReg2, WeightedSVDHead
from .ptv3_mod import PointTransformerEncoder, compute_batch_indices  # import your module

class HierFeatureExtraction(nn.Module):
    def __init__(self, args):
        super(HierFeatureExtraction, self).__init__()

        self.use_fps = args.use_fps
        self.use_weights = args.use_weights

        # Keypoint detectors remain the same.
        self.detector_1 = KeypointDetector(nsample=1024, k=64, in_channels=0, out_channels=[32,32,64], fps=self.use_fps)
        self.detector_2 = KeypointDetector(nsample=512, k=32, in_channels=64, out_channels=[64,64,128], fps=self.use_fps)
        self.detector_3 = KeypointDetector(nsample=256, k=16, in_channels=128, out_channels=[128,128,256], fps=self.use_fps)

        # if args.freeze_detector:
        #     for p in self.parameters():
        #         p.requires_grad = False

        if args.freeze_detector:
            for detector in [self.detector_1, self.detector_2, self.detector_3]:
                for p in detector.parameters():
                    p.requires_grad = False

        
        self.ptv3_mod_1 = PointTransformerEncoder(
            in_channels=64,
            enc_depths=(2, 2, 2),
            enc_channels=(64, 64, 64),
            enc_num_head=(2, 4, 8),
            enc_patch_size=(256, 256, 256)
        )
        self.ptv3_mod_2 = PointTransformerEncoder(
            in_channels=128,
            enc_depths=(2, 2, 2),
            enc_channels=(128, 128, 128),
            enc_num_head=(2, 4, 8),
            enc_patch_size=(128, 128, 128)
        )
        self.ptv3_mod_3 = PointTransformerEncoder(
            in_channels=256,
            enc_depths=(2, 2, 2),
            enc_channels=(256, 256, 256),
            enc_num_head=(2, 4, 8),
            enc_patch_size=(64, 64, 64)
        )
    
    def forward(self, points):
        # Detector level 1
        xyz_1, sigmas_1, att_feat_1, grp_feat_1, att_feat_map_1 = self.detector_1(points, None)
        
        # Use your ptv3_mod_1 to extract descriptors from xyz_1:
        B, N, _ = xyz_1.shape
        data_dict = {
            "coord": xyz_1.view(-1, 3),
            "grid_size": 0.01,  # adjust as needed
            "batch": compute_batch_indices(xyz_1),
            "feat": att_feat_1.permute(0, 2, 1).reshape(-1, att_feat_1.shape[1])
        }
        desc_1 = self.ptv3_mod_1(data_dict)
        desc_1 = desc_1.view(B, N, -1).permute(0,2,1)

        
        # Detector level 2
        if self.use_weights:
            weights_1 = 1.0/(sigmas_1 + 1e-5)
            weights_1_mean = torch.mean(weights_1, dim=1, keepdim=True)
            weights_1 = weights_1 / weights_1_mean
            xyz_2, sigmas_2, att_feat_2, grp_feat_2, att_feat_map_2 = self.detector_2(xyz_1, att_feat_1, weights_1)
        else:
            xyz_2, sigmas_2, att_feat_2, grp_feat_2, att_feat_map_2 = self.detector_2(xyz_1, att_feat_1)
        
        # Use ptv3_mod_2 for level 2 descriptors:
        B, N, _ = xyz_2.shape
        data_dict = {
            "coord": xyz_2.view(-1, 3),
            "grid_size": 0.01,
            "batch": compute_batch_indices(xyz_2),
            "feat": att_feat_2.permute(0, 2, 1).reshape(-1, att_feat_2.shape[1])
        }
        desc_2 = self.ptv3_mod_2(data_dict)
        desc_2 = desc_2.view(B, N, -1).permute(0,2,1)
        

        # Detector level 3
        if self.use_weights:
            weights_2 = 1.0/(sigmas_2 + 1e-5)
            weights_2_mean = torch.mean(weights_2, dim=1, keepdim=True)
            weights_2 = weights_2 / weights_2_mean
            xyz_3, sigmas_3, att_feat_3, grp_feat_3, att_feat_map_3 = self.detector_3(xyz_2, att_feat_2, weights_2)
        else:
            xyz_3, sigmas_3, att_feat_3, grp_feat_3, att_feat_map_3 = self.detector_3(xyz_2, att_feat_2)
        
        # Use ptv3_mod_3 for level 3 descriptors:
        B, N, _ = xyz_3.shape
        data_dict = {
            "coord": xyz_3.view(-1, 3),
            "grid_size": 0.01,
            "batch": compute_batch_indices(xyz_3),
            "feat": att_feat_3.permute(0, 2, 1).reshape(-1, att_feat_3.shape[1])
        }
        desc_3 = self.ptv3_mod_3(data_dict)
        desc_3 = desc_3.view(B, N, -1).permute(0,2,1)
        
        ret_dict = {
            'xyz_1': xyz_1, 
            'xyz_2': xyz_2,
            'xyz_3': xyz_3,
            'sigmas_1': sigmas_1, 
            'sigmas_2': sigmas_2, 
            'sigmas_3': sigmas_3,
            'desc_1': desc_1, 
            'desc_2': desc_2, 
            'desc_3': desc_3
        }
        return ret_dict


class Model_V6(nn.Module):

    def __init__(self, args):
        super(Model_V6, self).__init__()
        self.feature_extraction = HierFeatureExtraction(args)

        # Freeze pretrained features when train
        if args.freeze_feats:
            for p in self.parameters():
                p.requires_grad = False
        
        self.coarse_corres = CoarseReg(k=8, in_channels=256, use_sim=True, use_neighbor=True)
        self.fine_corres_2 = FineReg2(k=8, in_channels=128)
        self.fine_corres_1 = FineReg1(k=8, in_channels=64)

        self.svd_head = WeightedSVDHead()
    
    def forward(self, src_points, dst_points):
        # Feature extraction
        src_feats = self.feature_extraction(src_points)
        dst_feats = self.feature_extraction(dst_points)

        # for key, value in src_feats.items():
        #     if torch.isnan(value).any():
        #         print(f"NaN detected in src_feats[{key}]!")

        # for key, value in dst_feats.items():
        #     if torch.isnan(value).any():
        #         print(f"NaN detected in dst_feats[{key}]!")



        # for key, val in src_feats.items():
        #     print(f"src_feats[{key}] requires_grad: {val.requires_grad}")
        # for key, val in dst_feats.items():
        #     print(f"dst_feats[{key}] requires_grad: {val.requires_grad}")


        # Coarse registration: Layer 3
        src_xyz_corres_3, src_dst_weights_3, coord_dist, feats_dist = self.coarse_corres(src_feats['xyz_3'], src_feats['desc_3'], dst_feats['xyz_3'], \
            dst_feats['desc_3'], src_feats['sigmas_3'], dst_feats['sigmas_3'])

         # Check requires_grad for coarse registration
        # print(f"src_xyz_corres_3 requires_grad: {src_xyz_corres_3.requires_grad}")
        # print(f"src_dst_weights_3 requires_grad: {src_dst_weights_3.requires_grad}")


        R3, t3 = self.svd_head(src_feats['xyz_3'], src_xyz_corres_3, src_dst_weights_3)

        # Check requires_grad after svd_headf
        # print(f"R3 requires_grad: {R3.requires_grad}")
        # print(f"t3 requires_grad: {t3.requires_grad}")

        # Fine registration: Layer 2
        src_xyz_2_trans = torch.matmul(R3, src_feats['xyz_2'].permute(0,2,1).contiguous()) + t3.unsqueeze(2)
        src_xyz_2_trans = src_xyz_2_trans.permute(0,2,1).contiguous()

        
        src_xyz_corres_2, src_dst_weights_2, src_dst_weights_2_prime, src_dst_feats_2, src_dst_feats_2_prime = self.fine_corres_2(src_xyz_2_trans, src_feats['desc_2'], dst_feats['xyz_2'], \
            dst_feats['desc_2'], src_feats['sigmas_2'], dst_feats['sigmas_2'])
        
        R2_, t2_ = self.svd_head(src_xyz_2_trans, src_xyz_corres_2, src_dst_weights_2)
        
        
        T3 = torch.zeros(R3.shape[0],4,4).cuda()
        T3[:,:3,:3] = R3
        T3[:,:3,3] = t3
        T3[:,3,3] = 1.0
        T2_ = torch.zeros(R2_.shape[0],4,4).cuda()
        T2_[:,:3,:3] = R2_
        T2_[:,:3,3] = t2_
        T2_[:,3,3] = 1.0
        T2 = torch.matmul(T2_, T3)
        R2 = T2[:,:3,:3]
        t2 = T2[:,:3,3]

        # Fine registration: Layer 1
        src_xyz_1_trans = torch.matmul(R2, src_feats['xyz_1'].permute(0,2,1).contiguous()) + t2.unsqueeze(2)
        src_xyz_1_trans = src_xyz_1_trans.permute(0,2,1).contiguous()
        src_xyz_corres_1, src_dst_weights_1 = self.fine_corres_1(src_xyz_1_trans, src_feats['desc_1'], dst_feats['xyz_1'], \
            dst_feats['desc_1'], src_feats['sigmas_1'], dst_feats['sigmas_1'])
        R1_, t1_ = self.svd_head(src_xyz_1_trans, src_xyz_corres_1, src_dst_weights_1)
        
        
        T1_ = torch.zeros(R1_.shape[0],4,4).cuda()
        T1_[:,:3,:3] = R1_
        T1_[:,:3,3] = t1_
        T1_[:,3,3] = 1.0

        T1 = torch.matmul(T1_, T2)
        R1 = T1[:,:3,:3]
        t1 = T1[:,:3,3]

        # corres_dict = {}
        ret_dict = {}
        ret_dict['src_xyz_corres_3'] = src_xyz_corres_3
        ret_dict['src_xyz_corres_2'] = src_xyz_corres_2
        ret_dict['src_xyz_corres_1'] = src_xyz_corres_1
        ret_dict['src_dst_weights_3'] = src_dst_weights_3
        ret_dict['src_dst_weights_2'] = src_dst_weights_2
        ret_dict['src_dst_weights_1'] = src_dst_weights_1

        # ret_dict = {}
        # ret_dict['rotation'] = [R3, R2, R1]
        # ret_dict['translation'] = [t3, t2, t1]
        # ret_dict['src_feats'] = src_feats
        # ret_dict['dst_feats'] = dst_feats

        # ret_dict['src_xyz_2_trans'] = src_xyz_2_trans
        # ret_dict['src_dst_feats_2'] = src_dst_feats_2
        # ret_dict['src_dst_feats_2_prime'] = src_dst_feats_2_prime
        # ret_dict['src_dst_weights_2_prime'] = src_dst_weights_2_prime

        #ret_dict = {}
        ret_dict['rotation'] = [R3, R2, R1] # Tf loss - Rotations
        ret_dict['translation'] = [t3, t2, t1] # Tf loss - Translations
        ret_dict['src_feats'] = src_feats
        ret_dict['dst_feats'] = dst_feats

        ret_dict['src_feats_desc_2'] = src_feats['desc_2'] # MI Loss - c_local
        ret_dict['src_feats_sigmas_2'] = src_feats['sigmas_2'] # MI Loss - c_global
        
        ret_dict['src_xyz_2_trans'] = src_xyz_2_trans # chamfer loss - src points
        ret_dict['dst_xyz_2'] = dst_feats['xyz_2'] # chamfer loss - dst points

        ret_dict['src_dst_feats_2'] = src_dst_feats_2 #x_local
        ret_dict['src_dst_feats_2_prime'] = src_dst_feats_2_prime #x_local_prime
        
        
        ret_dict['src_dst_weights_2'] = src_dst_weights_2 #x_global
        ret_dict['src_dst_weights_2_prime'] = src_dst_weights_2_prime #x_global_prime

        ret_dict['coord_dist'] = coord_dist
        ret_dict['feats_dist'] = feats_dist

        
        return ret_dict#, corres_dict 

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
    model = Model_V6(args).cuda()
    xyz1 = torch.rand(2,16384,3).cuda()
    xyz2 = torch.rand(2,16384,3).cuda()
    ret_dict = model(xyz1, xyz2)
    print(ret_dict['rotation'][-1].shape)
    print(ret_dict['translation'][-1].shape)