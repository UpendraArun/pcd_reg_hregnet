import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models import furthest_point_sample, weighted_furthest_point_sample, gather_operation
from pytorch3d.ops import knn_points, knn_gather
from pytorch3d.transforms import axis_angle_to_matrix, rotation_6d_to_matrix


def knn_group(xyz1, xyz2, features2, k):
    '''
    Input:
        xyz1: query points, [B,M,3] 
        xyz2: database points, [B,N,3]
        features2: [B,C,N]
        k: int
    Output:
        grouped_features: [B,4+C,M,k]
        knn_xyz: [B,M,k,3]
    '''
    _, knn_idx, knn_xyz = knn_points(xyz1, xyz2, K=k, return_nn=True)
    rela_xyz = knn_xyz - xyz1.unsqueeze(2).repeat(1,1,k,1) # [B,M,k,3]
    rela_dist = torch.norm(rela_xyz, dim=-1, keepdim=True) # [B,M,k,1]
    grouped_features =  torch.cat([rela_xyz,rela_dist], dim=-1)
    if features2 is not None:
        knn_features = knn_gather(features2.permute(0,2,1).contiguous(), knn_idx)
        grouped_features = torch.cat([rela_xyz,rela_dist,knn_features],dim=-1) # [B,M,k,4+C]
    return grouped_features.permute(0,3,1,2).contiguous(), knn_xyz

def calc_cosine_similarity(desc1, desc2):
    '''
    Input:
        desc1: [B,N,*,C]
        desc2: [B,N,*,C]
    Ret:
        similarity: [B,N,*]
    '''
    inner_product = torch.sum(torch.mul(desc1, desc2), dim=-1, keepdim=False)
    norm_1 = torch.norm(desc1, dim=-1, keepdim=False)
    norm_2 = torch.norm(desc2, dim=-1, keepdim=False)
    similarity = inner_product/(torch.mul(norm_1, norm_2)+1e-6)
    return similarity

# def initialize_weights(m):
#     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#         nn.init.xavier_uniform_(m.weight)
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)
#     elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
#         nn.init.constant_(m.weight, 1)
#         nn.init.constant_(m.bias, 0)

def initialize_weights(module):

    if isinstance(module, nn.Linear):  # Fully connected layers
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):  # Convolution layers
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):  # BatchNorm
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)




class KeypointDetectorSelfAttention(nn.Module):
    def __init__(self, nsample, k, in_channels, out_channels, fps=True):
        super(KeypointDetectorSelfAttention, self).__init__()

        self.nsample = nsample
        self.k = k
        self.fps = fps

        out_channels = [in_channels + 4, *out_channels]
        layers = []
        for i in range(1, len(out_channels)):
            layers += [
                nn.Conv2d(out_channels[i-1], out_channels[i], kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels[i]),
                nn.ReLU()
            ]
        self.convs = nn.Sequential(*layers)
        self.C_o1 = out_channels[-1]

        # Self-Attention Layers
        self.q_proj = nn.Conv2d(self.C_o1, self.C_o1 // 4, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(self.C_o1, self.C_o1 // 4, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(self.C_o1, self.C_o1, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.mlp1 = nn.Sequential(nn.Conv1d(self.C_o1, self.C_o1, kernel_size=1),
                                  nn.BatchNorm1d(self.C_o1),
                                  nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Conv1d(self.C_o1, self.C_o1, kernel_size=1),
                                  nn.BatchNorm1d(self.C_o1),
                                  nn.ReLU())
        self.mlp3 = nn.Sequential(nn.Conv1d(self.C_o1, 1, kernel_size=1))
        self.softplus = nn.Softplus()
        self.apply(initialize_weights)
     
    
    def forward(self, xyz, features, weights=None):
        if self.fps:
            if weights is None:
                fps_idx = furthest_point_sample(xyz, self.nsample)
                sampled_xyz = gather_operation(xyz.permute(0,2,1).contiguous(), fps_idx).permute(0,2,1).contiguous()
            else:
                fps_idx = weighted_furthest_point_sample(xyz, weights, self.nsample)
                sampled_xyz = gather_operation(xyz.permute(0,2,1).contiguous(), fps_idx).permute(0,2,1).contiguous()
        else:
            N = xyz.shape[1]
            rand_idx = torch.randperm(N)[:self.nsample]
            sampled_xyz = xyz[:,rand_idx,:]
        
        grouped_features, knn_xyz = knn_group(sampled_xyz, xyz, features, self.k)  # [B, 4+C1, M, k]
        embedding = self.convs(grouped_features)  # [B, C_o, M, k]

        # Compute Queries, Keys, and Values
        Q = self.q_proj(embedding).permute(0, 2, 3, 1)  # [B, M, k, C_o//4]
        K = self.k_proj(embedding).permute(0, 2, 1, 3)  # [B, M, C_o//4, k]
        V = self.v_proj(embedding).permute(0, 2, 3, 1)  # [B, C_o, M, k]

        # Compute scaled dot-product attention
        attention_scores = torch.matmul(Q, K) / (K.shape[-1] ** 0.5)  # [B, M, k, k]
        attention_weights = self.softmax(attention_scores)  # [B, M, k, k]

        # Apply attention to Values
        attentive_features = torch.matmul(attention_weights, V)  # [B, M, k, C_o]
        attentive_features = attentive_features.permute(0, 3, 1, 2)  # [B, C_o, M, k]

        # Aggregate over k neighbors
        attentive_feature = torch.sum(attentive_features, dim=-1)  # [B, C_o, M]

        # Compute keypoint locations (weighted sum over neighbors)
        weights_xyz = attention_weights.sum(dim=-2, keepdim=True)  # [B, M, 1, k]
        keypoints = torch.sum(weights_xyz * knn_xyz.permute(0, 1, 3, 2), dim=-1)  # [B, M, 3]

        # Compute sigmas (uncertainty)
        sigmas = self.mlp3(self.mlp2(self.mlp1(attentive_feature)))
        sigmas = self.softplus(sigmas) + 0.001
        sigmas = sigmas.squeeze(1)

        return keypoints, sigmas, attentive_feature


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=4):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads  # Split feature space
        self.scale = self.head_dim ** 0.5

        self.q_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.k_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.v_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.out_proj = nn.Linear(feature_dim, feature_dim)  # Merge heads

        self.softmax = nn.Softmax(dim=-1)
        self.apply(initialize_weights)

    def forward(self, feats_left, feats_right):
        """
        feats_left:  [B, N, C] - features from left point cloud
        feats_right: [B, N, C] - features from right point cloud
        """
        feats_left = feats_left.permute(0, 2, 1).contiguous()  # Now [B, N, C]
        feats_right = feats_right.permute(0, 2, 1).contiguous()  # Now [B, M, C]

        B, N, C = feats_left.shape
        _, M, _ = feats_right.shape

        # Project features to multi-head queries, keys, and values
        Q = self.q_proj(feats_left).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, N, head_dim]
        K = self.k_proj(feats_right).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, M, head_dim]
        V = self.v_proj(feats_right).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, M, head_dim]

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, heads, N, M]
        attention_weights = self.softmax(attention_scores)  # [B, heads, N, M]

        # Apply attention weights to Values
        attended_features = torch.matmul(attention_weights, V)  # [B, heads, N, head_dim]

        # Merge heads back
        attended_features = attended_features.transpose(1, 2).contiguous().view(B, N, C)  # [B, N, C]
        attended_features = self.out_proj(attended_features)  # Final projection

        return attended_features, attention_weights



class CorrespondenceEstimator(nn.Module):
    '''
    Computes correspondences using cross-attended features directly (no k-NN).
    
    Input:
        src_xyz:   [B, N, 3] - Source keypoints
        src_feat:  [B, N, C] - Source cross-attended features
        dst_xyz:   [B, N, 3] - Destination keypoints
        dst_feat:  [B, N, C] - Destination cross-attended features
        attn_scores: [B, N, N] - Attention scores from cross-attention
        
    Output:
        corres_xyz: [B, N, 3] - Estimated correspondences
        corres_weights: [B, N] - Correspondence confidence weights
    '''
    def __init__(self):
        super(CorrespondenceEstimator, self).__init__()
        self.apply(initialize_weights)

    def forward(self, dst_xyz, src_feat, attn_scores, sigmas):
        B, N, C = src_feat.shape
        _, M, _ = dst_xyz.shape
        
        # Normalize attention scores (Softmax over destination points)
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, N, M]

        # Compute weighted sum of destination keypoints
        attn_weights = attn_weights.mean(dim=1)  # Reduce across heads -> [B, N, M]
        corres_xyz = torch.bmm(attn_weights, dst_xyz)  # Now works: [B, N, M] × [B, M, 3] -> [B, N, 3]


        # Compute correspondence confidence as max attention weight per source point
        corres_weights, _ = torch.max(attn_weights, dim=-1)  # [B, N]

        # Incorporate sigmas into correspondence weights
        corres_weights = corres_weights * sigmas  # [B, N]

        return corres_xyz, corres_weights



# class CoarseReg(nn.Module):
#     '''
#     Params:
#         k: number of candidate keypoints
#         in_channels: input channel number
#         use_sim: use original similarity features
#         use_neighbor: use neighbor aware similarity features
#     Input:
#         src_xyz: [B,N,3]
#         src_desc: [B,C,N]
#         dst_xyz: [B,N,3]
#         dst_desc: [B,C,N]
#         src_weights: [B,N]
#         dst_weights: [B,N]
#     Output:
#         corres_xyz: [B,N,3]
#         weights: [B,N]
#     '''
#     def __init__(self, k, in_channels, use_sim=True, use_neighbor=True):
#         super(CoarseReg, self).__init__()

#         self.k = k

#         self.use_sim = use_sim
#         self.use_neighbor = use_neighbor

#         if self.use_sim and self.use_neighbor:
#             out_channels = [in_channels*2+16, in_channels*2, in_channels*2, in_channels*2]
#         elif self.use_sim:
#             out_channels = [in_channels*2+14, in_channels*2, in_channels*2, in_channels*2]
#         elif self.use_neighbor:
#             out_channels = [in_channels*2+14, in_channels*2, in_channels*2, in_channels*2]
#         else:
#             out_channels = [in_channels*2+12, in_channels*2, in_channels*2, in_channels*2]
        
#         layers = []

#         for i in range(1, len(out_channels)):
#             layers += [nn.Conv2d(out_channels[i-1], out_channels[i], kernel_size=1, bias=False),
#                        nn.BatchNorm2d(out_channels[i]),
#                        nn.ReLU()]
#         self.convs_1 = nn.Sequential(*layers)

#         out_channels_nbr = [in_channels+4, in_channels, in_channels, in_channels]
#         self_layers = []
#         for i in range(1, len(out_channels_nbr)):
#             self_layers += [nn.Conv2d(out_channels_nbr[i-1], out_channels_nbr[i], kernel_size=1, bias=False),
#                        nn.BatchNorm2d(out_channels_nbr[i]),
#                        nn.ReLU()]
#         self.convs_2 = nn.Sequential(*self_layers)

#         self.mlp1 = nn.Sequential(nn.Conv1d(in_channels*2, in_channels*2, kernel_size=1),
#                                   nn.BatchNorm1d(in_channels*2),
#                                   nn.ReLU())
#         self.mlp2 = nn.Sequential(nn.Conv1d(in_channels*2, in_channels*2, kernel_size=1),
#                                   nn.BatchNorm1d(in_channels*2),
#                                   nn.ReLU())
#         self.mlp3 = nn.Sequential(nn.Conv1d(in_channels*2, 1, kernel_size=1))
#         #self.mlpx = nn.Sequential(nn.Conv1d(in_channels*2, in_channels, kernel_size=1),
#                                   #nn.BatchNorm1d(in_channels),
#                                   #nn.ReLU())

#     def forward(self, src_xyz, src_desc, dst_xyz, dst_desc, src_weights, dst_weights):
#         src_desc = src_desc.permute(0,2,1).contiguous().to(torch.float32)
#         dst_desc = dst_desc.permute(0,2,1).contiguous().to(torch.float32)
        
#         # src_knn_idx is B,N1,k having ids of kNN dst desc for every src desc
#         _, src_knn_idx, src_knn_desc = knn_points(src_desc, dst_desc, K=self.k, return_nn=True)
#         src_knn_xyz = knn_gather(dst_xyz, src_knn_idx) # [B,N,k,3]
        
#         src_xyz_expand = src_xyz.unsqueeze(2).repeat(1,1,self.k,1)
#         src_desc_expand = src_desc.unsqueeze(2).repeat(1,1,self.k,1) # [B,N,k,C]
        
#         src_rela_xyz = src_knn_xyz - src_xyz_expand # [B,N,k,3]
#         src_rela_dist = torch.norm(src_rela_xyz, dim=-1, keepdim=True) # [B,N,k,1]
        
#         src_weights_expand = src_weights.unsqueeze(-1).unsqueeze(-1).repeat(1,1,self.k,1) # [B,N,k,1]
#         src_knn_weights = knn_gather(dst_weights.unsqueeze(-1), src_knn_idx) # [B,N,k,1]

#         if self.use_sim:
#             # construct original similarity features
#             dst_desc_expand_N = dst_desc.unsqueeze(2).repeat(1,1,src_xyz.shape[1],1) # [B,N2,N1,C]
#             src_desc_expand_N = src_desc.unsqueeze(1).repeat(1,dst_xyz.shape[1],1,1) # [B,N2,N1,C]

#             dst_src_cos = calc_cosine_similarity(dst_desc_expand_N, src_desc_expand_N) # [B,N2,N1]
#             dst_src_cos_max = torch.max(dst_src_cos, dim=2, keepdim=True)[0] # [B,N2,1]
#             dst_src_cos_norm = dst_src_cos/(dst_src_cos_max+1e-6) # [B,N2,N1]

#             src_dst_cos = dst_src_cos.permute(0,2,1) # [B,N1,N2]
#             src_dst_cos_max = torch.max(src_dst_cos, dim=2, keepdim=True)[0] # [B,N1,1]
#             src_dst_cos_norm = src_dst_cos/(src_dst_cos_max+1e-6) # [B,N1,N2]
            
#             dst_src_cos_knn = knn_gather(dst_src_cos_norm, src_knn_idx) # [B,N1,k,N1]
#             dst_src_cos = torch.zeros(src_knn_xyz.shape[0], src_knn_xyz.shape[1], \
#                 src_knn_xyz.shape[2]).cuda() # [B,N1,k]
#             for i in range(src_xyz.shape[1]):
#                 dst_src_cos[:,i,:] = dst_src_cos_knn[:,i,:,i]
            
#             src_dst_cos_knn = knn_gather(src_dst_cos_norm.permute(0,2,1), src_knn_idx) # [B, N1, k, N2]
#             src_dst_cos = torch.zeros(src_knn_xyz.shape[0], src_knn_xyz.shape[1], \
#                 src_knn_xyz.shape[2]).cuda() # [B,N1,k]
#             for i in range(src_xyz.shape[1]):
#                 src_dst_cos[:,i,:] = src_dst_cos_knn[:,i,:,i]

#         if self.use_neighbor:
#             _, src_nbr_knn_idx, src_nbr_knn_xyz = knn_points(src_xyz, src_xyz, K=self.k, return_nn=True)
#             src_nbr_knn_feats = knn_gather(src_desc, src_nbr_knn_idx) # [B,N,k,C]
#             src_nbr_knn_rela_xyz = src_nbr_knn_xyz - src_xyz_expand # [B,N,k,3]
#             src_nbr_knn_rela_dist = torch.norm(src_nbr_knn_rela_xyz, dim=-1, keepdim=True) # [B,N,k]
#             src_nbr_feats = torch.cat([src_nbr_knn_feats, src_nbr_knn_rela_xyz, src_nbr_knn_rela_dist], dim=-1) # [B,N,k,C+3+1]

#             _, dst_nbr_knn_idx, dst_nbr_knn_xyz = knn_points(dst_xyz, dst_xyz, K=self.k, return_nn=True)
#             dst_nbr_knn_feats = knn_gather(dst_desc, dst_nbr_knn_idx) # [B,N,k,C]
#             dst_xyz_expand = dst_xyz.unsqueeze(2).repeat(1,1,self.k,1)
#             dst_nbr_knn_rela_xyz = dst_nbr_knn_xyz - dst_xyz_expand # [B,N,k,3]
#             dst_nbr_knn_rela_dist = torch.norm(dst_nbr_knn_rela_xyz, dim=-1, keepdim=True) # [B,N,k]
#             dst_nbr_feats = torch.cat([dst_nbr_knn_feats, dst_nbr_knn_rela_xyz, dst_nbr_knn_rela_dist], dim=-1) # [B,N,k,C+3+1]

#             src_nbr_weights = self.convs_2(src_nbr_feats.permute(0,3,1,2).contiguous()) # [B, 1 , N, k]
#             src_nbr_weights = torch.max(src_nbr_weights, dim=1, keepdim=False)[0] # [B,N,k]
#             src_nbr_weights = F.softmax(src_nbr_weights, dim=-1) # [B,N,k]
#             src_nbr_desc = torch.sum(torch.mul(src_nbr_knn_feats, src_nbr_weights.unsqueeze(-1)),dim=2, keepdim=False) # [B, 256, 256]

#             dst_nbr_weights = self.convs_2(dst_nbr_feats.permute(0,3,1,2).contiguous()) # [B,1,N,k]
#             dst_nbr_weights = torch.max(dst_nbr_weights, dim=1, keepdim=False)[0] # [B,N,k]
#             dst_nbr_weights = F.softmax(dst_nbr_weights, dim=-1) # [B,N,k]
#             dst_nbr_desc = torch.sum(torch.mul(dst_nbr_knn_feats, dst_nbr_weights.unsqueeze(-1)),dim=2, keepdim=False) #[B, 256, 256]


#             dst_nbr_desc_expand_N = dst_nbr_desc.unsqueeze(2).repeat(1,1,src_xyz.shape[1],1) # [B,N2,N1,C]
#             src_nbr_desc_expand_N = src_nbr_desc.unsqueeze(1).repeat(1,dst_xyz.shape[1],1,1) # [B,N2,N1,C]

#             dst_src_nbr_cos = calc_cosine_similarity(dst_nbr_desc_expand_N, src_nbr_desc_expand_N) # [B,N2,N1]
#             dst_src_nbr_cos_max = torch.max(dst_src_nbr_cos, dim=2, keepdim=True)[0] # [B,N2,1]
#             dst_src_nbr_cos_norm = dst_src_nbr_cos/(dst_src_nbr_cos_max+1e-6) # [B,N2,N1]

#             src_dst_nbr_cos = dst_src_nbr_cos.permute(0,2,1) # [B,N1,N2]
#             src_dst_nbr_cos_max = torch.max(src_dst_nbr_cos, dim=2, keepdim=True)[0] # [B,N1,1]
#             src_dst_nbr_cos_norm = src_dst_nbr_cos/(src_dst_nbr_cos_max+1e-6) # [B,N1,N2]
            
#             dst_src_nbr_cos_knn = knn_gather(dst_src_nbr_cos_norm, src_knn_idx) # [B,N1,k,N1]
#             dst_src_nbr_cos = torch.zeros(src_knn_xyz.shape[0], src_knn_xyz.shape[1], \
#                 src_knn_xyz.shape[2]).to(src_knn_xyz.cuda()) # [B,N1,k]
#             for i in range(src_xyz.shape[1]):
#                 dst_src_nbr_cos[:,i,:] = dst_src_nbr_cos_knn[:,i,:,i]
            
#             src_dst_nbr_cos_knn = knn_gather(src_dst_nbr_cos_norm.permute(0,2,1), src_knn_idx)
#             src_dst_nbr_cos = torch.zeros(src_knn_xyz.shape[0], src_knn_xyz.shape[1], \
#                 src_knn_xyz.shape[2]).to(src_knn_xyz.cuda()) # [B,N1,k]
#             for i in range(src_xyz.shape[1]):
#                 src_dst_nbr_cos[:,i,:] = src_dst_nbr_cos_knn[:,i,:,i]
            
#         geom_feats = torch.cat([src_rela_xyz, src_rela_dist, src_xyz_expand, src_knn_xyz],dim=-1) # [B,N,k,10]
#         desc_feats = torch.cat([src_desc_expand, src_knn_desc, src_weights_expand, src_knn_weights],dim=-1) # [B,N,k,2C+2]
        
        
#         if self.use_sim and self.use_neighbor:
#             similarity_feats = torch.cat([src_dst_cos.unsqueeze(-1), dst_src_cos.unsqueeze(-1), \
#                 src_dst_nbr_cos.unsqueeze(-1), dst_src_nbr_cos.unsqueeze(-1)], dim=-1)
#         elif self.use_sim:
#             similarity_feats = torch.cat([src_dst_cos.unsqueeze(-1), dst_src_cos.unsqueeze(-1)],dim=-1)
#         elif self.use_neighbor:
#             similarity_feats = torch.cat([src_dst_nbr_cos.unsqueeze(-1), dst_src_nbr_cos.unsqueeze(-1)], dim=-1)
#         else:
#             similarity_feats = None
        

#         if self.use_sim or self.use_neighbor:
#             feats = torch.cat([geom_feats, desc_feats, similarity_feats],dim=-1)
#         else:
#             feats = torch.cat([geom_feats, desc_feats],dim=-1)

#         feats = self.convs_1(feats.permute(0,3,1,2)) # [B,C,N,k]
#         attentive_weights = torch.max(feats, dim=1)[0]
#         attentive_weights = F.softmax(attentive_weights, dim=-1) # [B,N,k]
        
#         corres_xyz = torch.sum(torch.mul(attentive_weights.unsqueeze(-1), src_knn_xyz), dim=2, keepdim=False) # [B,N,3]
        
#         attentive_feats = torch.sum(torch.mul(attentive_weights.unsqueeze(1), feats), dim=-1, keepdim=False) # [B,N,C]
#         #attentive_features = (self.mlpx(attentive_feats)).permute(0,2,1).contiguous() # [B,C,N]
#         #attentive_features_prime = attentive_features[torch.randperm(attentive_features.size(0))] # [B,C,N]

#         #print(attentive_features.shape,attentive_features_prime.shape)

#         weights = self.mlp3(self.mlp2(self.mlp1(attentive_feats))) # [B,1,N]
#         weights = torch.sigmoid(weights.squeeze(1)) # [B,N]
#         #weights_prime = weights[torch.randperm(weights.size(0))] # [B,N]
#         #print(weights.shape,weights_prime.shape)

#         return corres_xyz, weights #, weights_prime, attentive_features, attentive_features_prime

# class FineReg1(nn.Module):
#     '''
#     Params:
#         k: number of candidate keypoints
#         in_channels: input channel number
#     Input:
#         src_xyz: [B,N,3]
#         src_desc: [B,C,N]
#         dst_xyz: [B,N,3]
#         dst_desc: [B,C,N]
#         src_weights: [B,N]
#         dst_weights: [B,N]
#     Output:
#         corres_xyz: [B,N,3]
#         weights: [B,N]
#     '''
#     def __init__(self, k, in_channels):
#         super(FineReg1, self).__init__()
#         self.k = k
#         out_channels = [in_channels*2+12, in_channels*2, in_channels*2, in_channels*2]
#         layers = []
#         for i in range(1, len(out_channels)):
#             layers += [nn.Conv2d(out_channels[i-1], out_channels[i], kernel_size=1, bias=False),
#                        nn.BatchNorm2d(out_channels[i]),
#                        nn.ReLU()]
#         self.convs_1 = nn.Sequential(*layers)

#         self.mlp1 = nn.Sequential(nn.Conv1d(in_channels*2, in_channels*2, kernel_size=1),
#                                   nn.BatchNorm1d(in_channels*2),
#                                   nn.ReLU())
#         self.mlp2 = nn.Sequential(nn.Conv1d(in_channels*2, in_channels*2, kernel_size=1),
#                                   nn.BatchNorm1d(in_channels*2),
#                                   nn.ReLU())
#         self.mlp3 = nn.Sequential(nn.Conv1d(in_channels*2, 1, kernel_size=1))
    
#     def forward(self, src_xyz, src_feat, dst_xyz, dst_feat, src_weights, dst_weights):
#         _, src_knn_idx, src_knn_xyz = knn_points(src_xyz.to(torch.float32), dst_xyz.to(torch.float32), K=self.k, return_nn=True)
#         src_feat = src_feat.permute(0,2,1).contiguous()
#         dst_feat = dst_feat.permute(0,2,1).contiguous()
#         src_knn_feat = knn_gather(dst_feat, src_knn_idx) # [B,N,k,C]
#         src_xyz_expand = src_xyz.unsqueeze(2).repeat(1,1,self.k,1)
#         src_feat_expand = src_feat.unsqueeze(2).repeat(1,1,self.k,1)
#         src_rela_xyz = src_knn_xyz - src_xyz_expand # [B,N,k,3]
#         src_rela_dist = torch.norm(src_rela_xyz, dim=-1, keepdim=True) # [B,N,k,1]
#         src_weights_expand = src_weights.unsqueeze(-1).unsqueeze(-1).repeat(1,1,self.k,1) # [B,N,k,1]
#         src_knn_weights = knn_gather(dst_weights.unsqueeze(-1), src_knn_idx) # [B,N,k,1]
#         feats = torch.cat([src_rela_xyz, src_rela_dist, src_xyz_expand, src_knn_xyz, \
#             src_feat_expand, src_knn_feat, src_weights_expand, src_knn_weights], dim=-1)
#         feats = self.convs_1(feats.permute(0,3,1,2).contiguous()) # [B,C,N,k]
#         attentive_weights = torch.max(feats, dim=1)[0]
#         attentive_weights = F.softmax(attentive_weights, dim=-1) # [B,N,k]
#         corres_xyz = torch.sum(torch.mul(attentive_weights.unsqueeze(-1), src_knn_xyz), dim=2, keepdim=False) # [B,N,3]
#         attentive_feats = torch.sum(torch.mul(attentive_weights.unsqueeze(1), feats), dim=-1, keepdim=False) # [B,N,C]
#         weights = self.mlp3(self.mlp2(self.mlp1(attentive_feats))) # [B,1,N]
#         weights = torch.sigmoid(weights.squeeze(1))

#         return corres_xyz, weights

# class FineReg2(nn.Module):
#     '''
#     Params:
#         k: number of candidate keypoints
#         in_channels: input channel number
#     Input:
#         src_xyz: [B,N,3]
#         src_desc: [B,C,N]
#         dst_xyz: [B,N,3]
#         dst_desc: [B,C,N]
#         src_weights: [B,N]
#         dst_weights: [B,N]
#     Output:
#         corres_xyz: [B,N,3]
#         weights: [B,N]
#     '''
#     def __init__(self, k, in_channels):
#         super(FineReg2, self).__init__()
#         self.k = k
#         out_channels = [in_channels*2+12, in_channels*2, in_channels*2, in_channels*2]
#         layers = []
#         for i in range(1, len(out_channels)):
#             layers += [nn.Conv2d(out_channels[i-1], out_channels[i], kernel_size=1, bias=False),
#                        nn.BatchNorm2d(out_channels[i]),
#                        nn.ReLU()]
#         self.convs_1 = nn.Sequential(*layers)

#         self.mlp1 = nn.Sequential(nn.Conv1d(in_channels*2, in_channels*2, kernel_size=1),
#                                   nn.BatchNorm1d(in_channels*2),
#                                   nn.ReLU())
#         self.mlp2 = nn.Sequential(nn.Conv1d(in_channels*2, in_channels*2, kernel_size=1),
#                                   nn.BatchNorm1d(in_channels*2),
#                                   nn.ReLU())
#         self.mlp3 = nn.Sequential(nn.Conv1d(in_channels*2, 1, kernel_size=1))
#         self.mlpx = nn.Sequential(nn.Conv1d(in_channels*2, in_channels, kernel_size=1),
#                                   nn.BatchNorm1d(in_channels),
#                                   nn.ReLU())
    
#     def forward(self, src_xyz, src_feat, dst_xyz, dst_feat, src_weights, dst_weights):
        
#         #print(src_xyz.shape, src_feat.shape, src_weights.shape)
#         # kNN feature extraction
#         _, src_knn_idx, src_knn_xyz = knn_points(src_xyz.to(torch.float32), dst_xyz.to(torch.float32), K=self.k, return_nn=True)
#         src_feat = src_feat.permute(0,2,1).contiguous()
#         dst_feat = dst_feat.permute(0,2,1).contiguous()
#         src_knn_feat = knn_gather(dst_feat, src_knn_idx) # [B,N,k,C]
        
#         # kNN spatial relative
#         src_xyz_expand = src_xyz.unsqueeze(2).repeat(1,1,self.k,1)
#         src_feat_expand = src_feat.unsqueeze(2).repeat(1,1,self.k,1)
#         src_rela_xyz = src_knn_xyz - src_xyz_expand # [B,N,k,3]
#         src_rela_dist = torch.norm(src_rela_xyz, dim=-1, keepdim=True) # [B,N,k,1]
#         src_weights_expand = src_weights.unsqueeze(-1).unsqueeze(-1).repeat(1,1,self.k,1) # [B,N,k,1]
#         src_knn_weights = knn_gather(dst_weights.unsqueeze(-1), src_knn_idx) # [B,N,k,1]
        
#         feats = torch.cat([src_rela_xyz, src_rela_dist, src_xyz_expand, src_knn_xyz, \
#             src_feat_expand, src_knn_feat, src_weights_expand, src_knn_weights], dim=-1)
#         feats = self.convs_1(feats.permute(0,3,1,2).contiguous()) # [B,C,N,k]
        
#         attentive_weights = torch.max(feats, dim=1)[0]
#         attentive_weights = F.softmax(attentive_weights, dim=-1) # [B,N,k]
#         corres_xyz = torch.sum(torch.mul(attentive_weights.unsqueeze(-1), src_knn_xyz), dim=2, keepdim=False) # [B,N,3]
        
        
#         attentive_feats = torch.sum(torch.mul(attentive_weights.unsqueeze(1), feats), dim=-1, keepdim=False) # [B,N,C]
#         attentive_features = (self.mlpx(attentive_feats))#.permute(0,2,1).contiguous() # [B,C,N]
#         attentive_features_prime = attentive_features[torch.randperm(attentive_features.size(0))] # [B,C,N]
#         #print(attentive_features.shape,attentive_features_prime.shape)
        
#         weights = self.mlp3(self.mlp2(self.mlp1(attentive_feats))) # [B,1,N]
#         weights = torch.sigmoid(weights.squeeze(1)) # [B,N]
#         weights_prime = weights[torch.randperm(weights.size(0))] # [B,N]
#         #print(weights.shape,weights_prime.shape)
        
        
#         return corres_xyz, weights, weights_prime, attentive_features, attentive_features_prime

class WeightedSVDHead(nn.Module):
    '''
    Input:
        src: [B,N,3]
        src_corres: [B,N,3]
        weights: [B,N]
    Output:
        r: [B,3,3]
        t: [B,3]
    '''
    def __init__(self):
        super(WeightedSVDHead, self).__init__()
    
    def forward(self, src, src_corres, weights):
        eps = 1e-4
        sum_weights = torch.sum(weights,dim=1,keepdim=True) + eps
        weights = weights/sum_weights
        weights = weights.unsqueeze(2)

        src_mean = torch.matmul(weights.transpose(1,2),src)/(torch.sum(weights,dim=1).unsqueeze(1)+eps)
        src_corres_mean = torch.matmul(weights.transpose(1,2),src_corres)/(torch.sum(weights,dim=1).unsqueeze(1)+eps)

        src_centered = src - src_mean # [B,N,3]
        src_corres_centered = src_corres - src_corres_mean # [B,N,3]

        weight_matrix = torch.diag_embed(weights.squeeze(2))
        
        cov_mat = torch.matmul(src_centered.transpose(1,2),torch.matmul(weight_matrix,src_corres_centered))

        try:
            u, s, v = torch.svd(cov_mat)
        except Exception as e:
            r = torch.eye(3).cuda()
            r = r.repeat(src_mean.shape[0],1,1)
            t = torch.zeros((src_mean.shape[0],3,1)).cuda()
            t = t.view(t.shape[0], 3)

            return r, t
        
        tm_determinant = torch.det(torch.matmul(v.transpose(1,2), u.transpose(1,2)))
        
        determinant_matrix = torch.diag_embed(torch.cat((torch.ones((tm_determinant.shape[0], 2)).cuda(),tm_determinant.unsqueeze(1)), 1))

        r = torch.matmul(v, torch.matmul(determinant_matrix, u.transpose(1,2)))

        t = src_corres_mean.transpose(1,2) - torch.matmul(r, src_mean.transpose(1,2))
        t = t.view(t.shape[0], 3)
        
        return r, t
    


class Regression_6dR_3dt_Head(nn.Module):
    def __init__(self):
        super(Regression_6dR_3dt_Head, self).__init__()
        
        
        self.fc1_rot = nn.Linear(6, 64)
        self.fc2_rot = nn.Linear(64, 32)
        self.fc3_rot = nn.Linear(32, 6)  # Rotation head

        self.fc1_trans = nn.Linear(6, 64)
        self.fc2_trans = nn.Linear(64, 32)
        self.fc3_trans = nn.Linear(64, 3)  # Translation head

    
    def forward(self, src, src_corres, weights):
        """
        src: (B, N, 3) - Source correspondences
        src_corres: (B, N, 3) - Target correspondences
        weights: (B, N) - Correspondence weights
        """

        # Normalize weights
        sum_weights = torch.sum(weights, dim=1, keepdim=True) + 1e-4
        weights = weights / sum_weights  # (B, N)
        weights = weights.unsqueeze(2)  # (B, N, 1)
        
        # Compute weighted means
        src_mean = (weights * src).sum(dim=1)  # (B, 3)
        src_corres_mean = (weights * src_corres).sum(dim=1)  # (B, 3)
        
        # Compute input features (difference between weighted means)
        x = torch.cat([src_mean, src_corres_mean], dim=1)  # (B, 6)
        
        # Pass through fully connected layers
        x_rot = F.relu(self.fc1_rot(x))
        x_rot = F.relu(self.fc2_rot(x_rot))
        rot_6d = self.fc3_rot(x_rot)

        x_trans = F.relu(self.fc1_trans(x))
        x_trans = F.relu(self.fc2_trans(x_trans))
        translation = self.fc3_trans(x_trans)

        
        # Convert 6D representation to a valid rotation matrix
        #rotation = self.compute_rotation_matrix_from_6d(rot_6d)
        rotation = rotation_6d_to_matrix(rot_6d)
        
        return rotation, translation
    
    def compute_rotation_matrix_from_6d(self, x):
        """
        Converts a 6D representation into a valid rotation matrix using manual L2 normalization.
        x: (B, 6) → rotation matrix (B, 3, 3)
        """
        B = x.shape[0]
        x = x.view(B, 3, 2)  # Reshape to (B, 3, 2)

        # Manual L2 normalization (instead of F.normalize)
        def l2_normalize(v, dim=1, eps=1e-6):
            return v / (torch.sqrt(torch.sum(v ** 2, dim=dim, keepdim=True)) + eps)

        b1 = l2_normalize(x[:, :, 0])  # First basis vector
        proj_b2 = (b1 * torch.sum(b1 * x[:, :, 1], dim=1, keepdim=True))  # Projection of b2 onto b1
        b2 = l2_normalize(x[:, :, 1] - proj_b2)  # Make b2 orthogonal to b1
        b3 = torch.cross(b1, b2, dim=1)  # Compute the third basis vector

        R = torch.stack([b1, b2, b3], dim=-1)  # (B, 3, 3)
        return R


class RegressionHead(nn.Module):
    def __init__(self):
        super(RegressionHead, self).__init__()
        
        
        self.fc1_rot = nn.Linear(6, 128)
        self.fc2_rot = nn.Linear(128, 64)
        self.fc3_rot = nn.Linear(64, 3)  # Rotation head

        self.fc1_trans = nn.Linear(6, 128)
        self.fc2_trans = nn.Linear(128, 64)
        self.fc3_trans = nn.Linear(64, 3)  # Translation head

    
    def forward(self, src, src_corres, weights):
        """
        src: (B, N, 3) - Source correspondences
        src_corres: (B, N, 3) - Target correspondences
        weights: (B, N) - Correspondence weights
        """

        # Normalize weights
        sum_weights = torch.sum(weights, dim=1, keepdim=True) + 1e-4
        weights = weights / sum_weights  # (B, N)
        weights = weights.unsqueeze(2)  # (B, N, 1)
        
        # Compute weighted means
        src_mean = (weights * src).sum(dim=1)  # (B, 3)
        src_corres_mean = (weights * src_corres).sum(dim=1)  # (B, 3)
        
        # Compute input features (difference between weighted means)
        x = torch.cat([src_mean, src_corres_mean], dim=1)  # (B, 6)
        
        # Pass through fully connected layers
        x_rot = F.relu(self.fc1_rot(x))
        x_rot = F.relu(self.fc2_rot(x_rot))
        rotation = self.fc3_rot(x_rot)

        x_trans = F.relu(self.fc1_trans(x))
        x_trans = F.relu(self.fc2_trans(x_trans))
        translation = self.fc3_trans(x_trans)

        
        return rotation, translation