import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalinfolossNet(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.c1 = nn.Conv1d(in_channels * 2, in_channels // 2, kernel_size=1, bias=False)
        self.c2 = nn.Conv1d(in_channels // 2, in_channels // 4, kernel_size=1, bias=False)
        self.c3 = nn.Conv1d(in_channels // 4, in_channels // 8, kernel_size=1, bias=False)
        self.l0 = nn.Linear(in_channels // 8, 1)
    
    def forward(self, x_global: torch.Tensor, c_global: torch.Tensor) -> torch.Tensor:
        xx = torch.cat((x_global, c_global), dim=1)
        h = xx.unsqueeze(dim=2)
        h = F.relu(self.c1(h))
        h = F.relu(self.c2(h))
        h = F.relu(self.c3(h))
        h = h.view(h.shape[0], -1)
        return self.l0(h)


class LocalinfolossNet(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels * 2, in_channels // 2, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels // 2, in_channels // 4, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(in_channels // 4, 1, kernel_size=1, bias=False)
    
    def forward(self, x_local: torch.Tensor, c_local: torch.Tensor) -> torch.Tensor:

        xx = torch.cat((x_local, c_local), dim=1)
        h = F.relu(self.conv1(xx))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = h.view(h.shape[0], -1)
        return h


class DeepMILoss(nn.Module):
    def __init__(self, global_in_channels: int = None, local_in_channels: int = None):
        super().__init__()
        self.global_d = GlobalinfolossNet(global_in_channels) if global_in_channels else None
        self.local_d = LocalinfolossNet(local_in_channels) if local_in_channels else None

        if self.global_d is None and self.local_d is None:
            raise AttributeError("MI loss not found")

    def compute_local_loss(self, x_local: torch.Tensor, x_local_prime: torch.Tensor, c_local: torch.Tensor) -> torch.Tensor:
        Ej = -F.softplus(-self.local_d(c_local, x_local)).mean()
        Em = F.softplus(self.local_d(c_local, x_local_prime)).mean()
        return 0.5 * (Em - Ej)

    def compute_global_loss(self, x_global: torch.Tensor, x_global_prime: torch.Tensor, c_global: torch.Tensor) -> torch.Tensor:
        Ej = -F.softplus(-self.global_d(c_global, x_global)).mean()
        Em = F.softplus(self.global_d(c_global, x_global_prime)).mean()
        return 0.5 * (Em - Ej)

    def forward(
        self,
        x_global: torch.Tensor = None,
        x_global_prime: torch.Tensor = None,
        x_local: torch.Tensor = None,
        x_local_prime: torch.Tensor = None,
        c_local: torch.Tensor = None,
        c_global: torch.Tensor = None,
    ) -> torch.Tensor:
        
        total_loss = 0

        if self.local_d is not None:
            total_loss += self.compute_local_loss(x_local, x_local_prime, c_local)

        if self.global_d is not None:
            total_loss += self.compute_global_loss(x_global, x_global_prime, c_global)

        return total_loss


if __name__ == "__main__":
    
    Total_MI = DeepMILoss(128,128)
    Local_MI = DeepMILoss(local_in_channels=128)
    Global_MI = DeepMILoss(global_in_channels=128)

     
    # 128 features and 1024 points
    x_global = torch.randn(2, 128) # B,C
    x_global_prime = torch.randn(2, 128) # B,C

    x_local = torch.randn(2, 128, 1024) # B,C,N
    x_local_prime = torch.randn(2, 128, 1024) # B,C,N 
    
    c_local = torch.randn(2, 128, 1024) # B,C,N
    
    c_global = torch.randn(2, 128) # B,C


    total_loss = Total_MI(x_global, x_global_prime, x_local, x_local_prime, c_local, c_global)
    local_loss = Local_MI(x_local=x_local, x_local_prime=x_local_prime, c_local=c_local)
    global_loss = Global_MI(x_global=x_global, x_global_prime=x_global_prime, c_global=c_global)
    
    print("Total loss:", total_loss.item()," Local loss:", local_loss.item()," Global loss:", global_loss.item())