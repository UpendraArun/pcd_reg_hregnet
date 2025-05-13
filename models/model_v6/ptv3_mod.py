import torch
import torch.nn as nn
import spconv.pytorch as spconv
from collections import OrderedDict
from addict import Dict
from .ptv3 import Point, PointSequential, Block, Embedding, PointModule

# --- Helper Function to compute batch indices ---
def compute_batch_indices(points):
    # points: (B, N, 3)
    B, N, _ = points.shape
    return torch.arange(B, device=points.device).repeat_interleave(N)

# --- Dummy Pooling Layer ---
from .ptv3 import PointModule

class DummyPooling(PointModule):
    """
    A dummy pooling layer that mimics the interface of SerializedPooling.
    It applies a pointwise projection to change the channel dimension while
    preserving the number of points.
    """
    def __init__(self, in_channels, out_channels, norm_layer, act_layer):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = norm_layer(out_channels)
        self.act = act_layer()
    
    def forward(self, point):
        feat = self.proj(point.feat)
        feat = self.norm(feat)
        feat = self.act(feat)
        point.feat = feat
        # Update the sparse_conv_feat so its feature channels match the new feat.
        if hasattr(point, "sparse_conv_feat") and point.sparse_conv_feat is not None:
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(feat)
        return point


# --- Modified Encoder-only PTv3 with Dummy Downsampling ---
class PointTransformerEncoder(nn.Module):
    def __init__(
        self,
        in_channels=3,  # Set to 3 if you only have XYZ coordinates.
        enc_depths=(2, 2, 2),        # Number of transformer blocks per stage.
        enc_channels=(32, 64, 128),  # Feature channels for each stage.
        enc_num_head=(2, 4, 8),      # Number of attention heads per stage.
        enc_patch_size=(1024, 1024, 1024),  # Patch size used in attention.
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        pre_norm=True,
        enable_rpe=False,       # Relative Position Encoding (optional).
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
    ):
        super().__init__()
        # The embedding layer projects the input (features) to a higher-dimensional space.
        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=lambda num_features: nn.BatchNorm1d(num_features, eps=1e-2, momentum=0.01),
            act_layer=nn.GELU,
        )
        # Build the encoder as a sequence of stages. Each stage, for s>0, begins with a
        # DummyPooling layer (mimicking the downsampling in the full model) followed by transformer blocks.
        self.enc = PointSequential()
        num_stages = len(enc_depths)
        for s in range(num_stages):
            stage = PointSequential()
            if s > 0:
                # Insert dummy pooling to match the full model’s downsampling module.
                stage.add(
                    DummyPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        norm_layer=nn.BatchNorm1d,
                        act_layer=nn.GELU,
                    ),
                    name="down"
                )
            for i in range(enc_depths[s]):
                stage.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=drop_path,
                        norm_layer=lambda normalized_shape: nn.LayerNorm(normalized_shape, eps=1e-2),
                        act_layer=nn.GELU,
                        pre_norm=pre_norm,
                        order_index=0,  # Can also use: i % len(order) if needed.
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}"
                )
            self.enc.add(stage, name=f"enc{s}")

    # After building your layers, initialize weights.
        self._init_weights()

    def _init_weights(self):
        """
        Initialize the weights of the model. 
        - Convolutional and Linear layers use Xavier uniform initialization.
        - Normalization layers (BatchNorm, LayerNorm) are initialized with constant values.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, data_dict):
        """
        Expects data_dict with the following keys:
          - "coord": (B*N, 3) tensor of point coordinates.
          - "grid_size": a scalar (e.g., 0.01) used for voxelization.
          - "batch": (B*N,) tensor of batch indices.
          - Optionally, "feat": initial features for each point.
            If not provided, "feat" is set equal to "coord".
        """
        if "feat" not in data_dict:
            data_dict["feat"] = data_dict["coord"]
        point = Point(data_dict)
        # Serialization computes a grid ordering; "order" here is set to "z".
        point.serialization(order="z", shuffle_orders=False)
        # Sparsify prepares the grid coordinates and spconv features.
        point.sparsify()
        # Apply the embedding layer.
        point = self.embedding(point)
        # Pass through the encoder (which now contains dummy downsampling and transformer blocks).
        point = self.enc(point)
        return point["feat"]


# --- Example Usage ---
if __name__ == "__main__":
    # Example input: a batch of keypoints with shape (B, N, 3)
    B, N = 2, 1000  # e.g., 2 point clouds, 1000 keypoints each.
    points = torch.rand(B, N, 3).cuda()  # Random keypoints
    feats = torch.rand(B,N,10).cuda()
    # Create a data dictionary.
    data_dict = {
        "coord": points.view(-1, 3),   # Flattened to (B*N, 3)
        "grid_size": 0.01,             # Example grid size for voxelization
        "batch": compute_batch_indices(points),  # Batch indices for each point
        "feat":feats.view(-1,10)
    }
    # Initialize the encoder; set in_channels=3 since we're using only coordinates.
    model_encoder = PointTransformerEncoder(in_channels=10).cuda()
    # Get features: output shape will be (B*N, C); reshape to (B, N, C) if desired.
    features = model_encoder(data_dict)
    features = features.view(B, N, -1)
    print("Output feature shape:", features.shape)
