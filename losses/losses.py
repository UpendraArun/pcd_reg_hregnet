import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points, knn_gather
from pytorch3d.transforms import matrix_to_euler_angles

from pytorch3d.loss import chamfer_distance

def prob_chamfer_loss(keypoints1, keypoints2, sigma1, sigma2, gt_R, gt_t):
    """
    Calculate probabilistic chamfer distance between keypoints1 and keypoints2
    Input:
        keypoints1: [B,M,3]
        keypoints2: [B,M,3]
        sigma1: [B,M]
        sigma2: [B,M]
        gt_R: [B,3,3]
        gt_t: [B,3]
    """
    keypoints1 = keypoints1.permute(0,2,1).contiguous()
    keypoints1 = torch.matmul(gt_R, keypoints1) + gt_t.unsqueeze(2)
    keypoints2 = keypoints2.permute(0,2,1).contiguous()
    B, M = keypoints1.size()[0], keypoints1.size()[2]
    N = keypoints2.size()[2]

    keypoints1_expanded = keypoints1.unsqueeze(3).expand(B,3,M,N)
    keypoints2_expanded = keypoints2.unsqueeze(2).expand(B,3,M,N)

    # diff: [B, M, M]
    diff = torch.norm(keypoints1_expanded-keypoints2_expanded, dim=1, keepdim=False)

    if sigma1 is None or sigma2 is None:
        min_dist_forward, _ = torch.min(diff, dim=2, keepdim=False)
        forward_loss = min_dist_forward.mean()

        min_dist_backward, _ = torch.min(diff, dim=1, keepdim=False)
        backward_loss = min_dist_backward.mean()

        loss = forward_loss + backward_loss
    
    else:
        min_dist_forward, min_dist_forward_I = torch.min(diff, dim=2, keepdim=False)
        selected_sigma_2 = torch.gather(sigma2, dim=1, index=min_dist_forward_I)
        sigma_forward = (sigma1 + selected_sigma_2)/2
        forward_loss = (torch.log(sigma_forward)+min_dist_forward/sigma_forward).mean()

        min_dist_backward, min_dist_backward_I = torch.min(diff, dim=1, keepdim=False)
        selected_sigma_1 = torch.gather(sigma1, dim=1, index=min_dist_backward_I)
        sigma_backward = (sigma2 + selected_sigma_1)/2
        backward_loss = (torch.log(sigma_backward)+min_dist_backward/sigma_backward).mean()

        loss = forward_loss + backward_loss
    return loss

def matching_loss(src_kp, src_sigma, src_desc, dst_kp, dst_sigma, dst_desc, gt_R, gt_t, temp=0.1, sigma_max=3.0):
    src_kp = src_kp.permute(0,2,1).contiguous()
    src_kp = torch.matmul(gt_R, src_kp) + gt_t.unsqueeze(2)
    dst_kp = dst_kp.permute(0,2,1).contiguous()

    src_desc = src_desc.unsqueeze(3) # [B,C,M,1]
    dst_desc = dst_desc.unsqueeze(2) # [B,C,1,M]

    desc_dists = torch.norm((src_desc - dst_desc), dim=1) # [B,M,M]
    desc_dists_inv = 1.0/(desc_dists + 1e-3)
    desc_dists_inv = desc_dists_inv/temp

    score_src = F.softmax(desc_dists_inv, dim=2)
    score_dst = F.softmax(desc_dists_inv, dim=1).permute(0,2,1)

    src_kp = src_kp.permute(0,2,1)
    dst_kp = dst_kp.permute(0,2,1)

    src_kp_corres = torch.matmul(score_src, dst_kp)
    dst_kp_corres = torch.matmul(score_dst, src_kp)

    diff_forward = torch.norm((src_kp - src_kp_corres), dim=-1)
    diff_backward = torch.norm((dst_kp - dst_kp_corres), dim=-1)

    src_weights = torch.clamp(sigma_max - src_sigma, min=0.01)
    src_weights_mean = torch.mean(src_weights, dim=1, keepdim=True)
    src_weights = (src_weights/src_weights_mean).detach()

    dst_weights = torch.clamp(sigma_max - dst_sigma, min=0.01)
    dst_weights_mean = torch.mean(dst_weights, dim=1, keepdim=True)
    dst_weights = (dst_weights/dst_weights_mean).detach()

    loss_forward = (src_weights * diff_forward).mean()
    loss_backward = (dst_weights * diff_backward).mean()

    loss = loss_forward + loss_backward

    return loss

def transformation_loss(pred_R, pred_t, gt_R, gt_t, alpha=1.0):
    '''
    Input:
        pred_R: [B,3,3]
        pred_t: [B,3]
        gt_R: [B,3,3]
        gt_t: [B,3]
        alpha: weight
    '''
    '''
    Output:
        loss  : 1
        loss_R:
        loss_t:
        R_err :
        geodesic_dist :
        T_err :
        eucl_dist :

    '''
    Identity = []
    for i in range(pred_R.shape[0]):
        Identity.append(torch.eye(3,3).cuda())
    
    Identity = torch.stack(Identity, dim=0)
    resi_R = torch.norm((torch.matmul(pred_R.transpose(2,1).contiguous(),gt_R) - Identity), dim=(1,2), keepdim=False)
    
    # Calculate Rotation and RRE Error
    R_err, geodesic_dist =  calc_rot_rre_err(pred_R, gt_R)
    
    # Calculate Translation and RTE Error
    T_err, eucl_dist = calc_tran_rte_err(pred_t, gt_t)
    
    #resi_t = torch.norm((pred_t - gt_t), dim=1, keepdim=False)
    loss_R = torch.mean(resi_R)
    loss_t = torch.mean(eucl_dist)
    loss = alpha * loss_R + loss_t

    return loss, loss_R, loss_t,  R_err , geodesic_dist, T_err, eucl_dist


def calc_rot_rre_err(pred_R, gt_R):
    
    R_error = torch.matmul(pred_R.transpose(2,1).contiguous(),gt_R)
    
    # Rotation error in euler angles
    R_err_rad = matrix_to_euler_angles(R_error, convention="XYZ")
    R_err_deg = torch.mean(torch.abs((torch.rad2deg(R_err_rad))),dim=0,keepdim=False)
    
    # Relative Rotation Error (RRE) / Geodesic distance 
    trace_R_error = torch.diagonal(R_error, dim1=-2, dim2=-1).sum(-1)
    cos_theta = (trace_R_error - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    geo_dist = torch.acos(cos_theta)
    geo_dist = torch.rad2deg(geo_dist)
    
    return R_err_deg, geo_dist

def calc_tran_rte_err(pred_t, gt_t):

    # Translation error
    T_error = pred_t - gt_t
    T_err_mean = torch.mean(torch.abs(T_error), dim=0, keepdim=False)
    
    # Relative Translation Error (RTE) / Euclidean distance
    eucl_dist = torch.norm(T_error, dim=1, keepdim=False)

    return T_err_mean, eucl_dist