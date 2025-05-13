import torch
import torch.nn as nn
import torch.nn.functional as F

class OverlapAwareCircleLoss(nn.Module):
    def __init__(self, pos_radius=5.0, safe_radius=2.5, log_scale=10.0, 
                 pos_optimal=3.0, neg_optimal=7.0, pos_margin=0.5, neg_margin=2.0, epsilon=1e-6):
        """
        Initializes the Overlap-Aware Circle Loss.

        Args:
            pos_radius (float): Distance threshold for positive pairs.
            safe_radius (float): Distance threshold for negative pairs.
            log_scale (float): Scaling factor for log-sum-exp computation.
            pos_optimal (float): Optimal feature distance for positive pairs.
            neg_optimal (float): Optimal feature distance for negative pairs.
            pos_margin (float): Margin for positive pairs.
            neg_margin (float): Margin for negative pairs.
        """
        super(OverlapAwareCircleLoss, self).__init__()
        self.pos_radius =  1.5 # if loss get stuck then increase this slightly to 1.5 - 2.0
        self.safe_radius = 0.2
        self.log_scale =   10  # if loss fluctuates a lot, then reduce it to30 or 35 to smoothen
        self.pos_optimal = 0.1
        self.neg_optimal = 1.4
        self.pos_margin =  0.1
        self.neg_margin =  1.4
        self.epsilon    =  epsilon

    def forward(self, coords_dist, feats_dist, weights=None):
        """
        Computes the Overlap-Aware Circle Loss.

        Args:
            coords_dist (torch.Tensor): Pairwise spatial distances of shape (B, N, k). kNNs spatial distances
            feats_dist (torch.Tensor): Pairwise feature distances of shape (B, N, k). kNNs feature distances
            weights (torch.Tensor, optional): Confidence weights for correspondences of shape (B, N).

        Returns:
            torch.Tensor: Computed loss value.
        """
        
        # Ensure numerical stability
        epsilon = 1e-6  # Small epsilon to avoid issues like division by zero or log of zero

        # Mask for positive and negative pairs based on spatial distance
        pos_mask = coords_dist < self.pos_radius  # Mask for positive pairs
        neg_mask = coords_dist > self.safe_radius  # Mask for negative pairs

        # print("coords_dist:", coords_dist)
        # print("self.pos_radius:", self.pos_radius)
        # print("Sample coords_dist:", coords_dist[0, 0, 0])

        
        # print("pos_mask:", pos_mask.sum(), pos_mask.shape)


        # Select anchors that have both positive and negative pairs
        row_sel = ((pos_mask.sum(-1) > 0) & (neg_mask.sum(-1) > 0)).detach()
        col_sel = ((pos_mask.sum(-2) > 0) & (neg_mask.sum(-2) > 0)).detach()

        # Compute weights for positive and negative pairs with numerical stability
        pos_weight = feats_dist - 1e5 * (~pos_mask).float()
        pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight - self.pos_optimal).detach()

        # Check the pos_weight calculation directly
        #print("pos_weight:", pos_weight)


        neg_weight = feats_dist + 1e5 * (~neg_mask).float()
        neg_weight = torch.max(torch.zeros_like(neg_weight), self.neg_optimal - neg_weight).detach()

        # Debugging pos_weight and neg_weight
        # if torch.any(torch.isnan(pos_weight)) or torch.any(torch.isinf(pos_weight)):
        #     print("NaN/inf detected in pos_weight!")
            
        # if torch.any(torch.isnan(neg_weight)) or torch.any(torch.isinf(neg_weight)):
        #     print("NaN/inf detected in neg_weight!")


        # Clamp coords_dist and feats_dist to avoid extreme values causing numerical instability
        coords_dist = torch.clamp(coords_dist, min=epsilon, max=1e6)  # Clamp the spatial distances
        feats_dist = torch.clamp(feats_dist, min=epsilon, max=1e6)    # Clamp the feature distances

        # Log-sum-exp for positive and negative pairs with safety measures
        lse_pos_row = torch.logsumexp(self.log_scale * (feats_dist - self.pos_margin) * pos_weight, dim=-1)
        lse_pos_col = torch.logsumexp(self.log_scale * (feats_dist - self.pos_margin) * pos_weight, dim=-2)

        lse_neg_row = torch.logsumexp(self.log_scale * (self.neg_margin - feats_dist) * neg_weight, dim=-1)
        lse_neg_col = torch.logsumexp(self.log_scale * (self.neg_margin - feats_dist) * neg_weight, dim=-2)

        #         # Debugging logsumexp inputs
        # print("feats_dist:", feats_dist.min(), feats_dist.max())
        # print("pos_weight:", pos_weight.min(), pos_weight.max())
        # print("neg_weight:", neg_weight.min(), neg_weight.max())

        # # Check for potential issues before logsumexp
        # logsumexp_input_pos = self.log_scale * (feats_dist - self.pos_margin) * pos_weight
        # logsumexp_input_neg = self.log_scale * (self.neg_margin - feats_dist) * neg_weight

        # print("logsumexp_input_pos:", logsumexp_input_pos.min(), logsumexp_input_pos.max())
        # print("logsumexp_input_neg:", logsumexp_input_neg.min(), logsumexp_input_neg.max())


        # Apply softplus activation and scale back
        loss_row = F.softplus(lse_pos_row + lse_neg_row) / self.log_scale
        loss_col = F.softplus(lse_pos_col + lse_neg_col) / self.log_scale

        # Compute final loss
        circle_loss = (loss_row[row_sel].mean() + loss_col[col_sel].mean()) / 2

        # Apply confidence weights if provided
        if weights is not None:
            # Normalize weights to avoid NaNs from large/small values
            weights = weights / (weights.sum(dim=-1, keepdim=True) + epsilon)  # Normalize weights
            circle_loss = (circle_loss * weights).sum() / (weights.sum() + epsilon)

        #print(f'circle loss is {circle_loss}')

        return circle_loss