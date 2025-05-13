import torch
import numpy as np 
from scipy.spatial.transform import Rotation
from typing import Tuple, List, Optional, Union, Any
from pytorch3d import transforms

def get_transformation_matrix(rot, trans) -> torch.Tensor:
    rot_matrix = transforms.euler_angles_to_matrix(rot, 'XYZ') 
    tf = get_transform_from_rotation_translation_tensor(rotation=rot_matrix, translation=trans)
    return tf

def transfrom_point_cloud(pts, trans):
    R = trans[:, :3,:3]
    T = trans[:, :3, 3]
    pts = pts @ R.mT + T
    return pts

def get_transform_from_rotation_translation(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Get rigid transform matrix from rotation matrix and translation vector.

    Args:
        rotation (array): (3, 3)
        translation (array): (3,)

    Returns:
        transform: (4, 4)
    """
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform

def get_transform_from_rotation_translation_tensor(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """Get rigid transform matrix from rotation matrix and translation vector.

    Args:
        rotation (array): (n, 3, 3)
        translation (array): (n, 3,)

    Returns:
        transform: (4, 4)
    """
    tf = torch.zeros([rotation.shape[0], 4,4], device=rotation.device)
    tf[:,:3, :3] = rotation
    tf[:,:3, 3] = translation
    tf[:,3, 3] = 1
    return tf

def get_rotation_translation_from_transform(transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get rotation matrix and translation vector from rigid transform matrix.

    Args:
        transform (array): (4, 4)

    Returns:
        rotation (array): (3, 3)
        translation (array): (3,)
    """
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return rotation, translation
def quaternion_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert a rotation matrix to quaternion.
    Args:
        matrix (torch.Tensor): [B, 4, 4] transformation matrices or [B, 3, 3] rotation matrices.

    Returns:
        torch.Tensor: shape [B, 4], normalized quaternions.
    """
    if matrix.shape[-2:] == (4, 4):
        R = matrix[:, :-1, :-1]
    elif matrix.shape[-2:] == (3, 3):
        R = matrix
    else:
        raise TypeError("Not a valid rotation matrix")
    
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    q = torch.zeros((R.size(0), 4), device=matrix.device)

    cond1 = tr > 0
    S = torch.where(cond1, (tr + 1.0).sqrt() * 2, torch.zeros_like(tr))
    q[cond1, 0] = 0.25 * S[cond1]
    q[cond1, 1] = (R[cond1, 2, 1] - R[cond1, 1, 2]) / S[cond1]
    q[cond1, 2] = (R[cond1, 0, 2] - R[cond1, 2, 0]) / S[cond1]
    q[cond1, 3] = (R[cond1, 1, 0] - R[cond1, 0, 1]) / S[cond1]

    cond2 = ~cond1 & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    S = torch.where(cond2, (1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]).sqrt() * 2, S)
    q[cond2, 0] = (R[cond2, 2, 1] - R[cond2, 1, 2]) / S[cond2]
    q[cond2, 1] = 0.25 * S[cond2]
    q[cond2, 2] = (R[cond2, 0, 1] + R[cond2, 1, 0]) / S[cond2]
    q[cond2, 3] = (R[cond2, 0, 2] + R[cond2, 2, 0]) / S[cond2]

    cond3 = ~cond1 & ~cond2 & (R[:, 1, 1] > R[:, 2, 2])
    S = torch.where(cond3, (1.0 + R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2]).sqrt() * 2, S)
    q[cond3, 0] = (R[cond3, 0, 2] - R[cond3, 2, 0]) / S[cond3]
    q[cond3, 1] = (R[cond3, 0, 1] + R[cond3, 1, 0]) / S[cond3]
    q[cond3, 2] = 0.25 * S[cond3]
    q[cond3, 3] = (R[cond3, 1, 2] + R[cond3, 2, 1]) / S[cond3]

    cond4 = ~cond1 & ~cond2 & ~cond3
    S = torch.where(cond4, (1.0 + R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1]).sqrt() * 2, S)
    q[cond4, 0] = (R[cond4, 1, 0] - R[cond4, 0, 1]) / S[cond4]
    q[cond4, 1] = (R[cond4, 0, 2] + R[cond4, 2, 0]) / S[cond4]
    q[cond4, 2] = (R[cond4, 1, 2] + R[cond4, 2, 1]) / S[cond4]
    q[cond4, 3] = 0.25 * S[cond4]

    return q / q.norm(dim=-1, keepdim=True)

def quat2mat(q: torch.Tensor) -> torch.Tensor:
    """
    Convert a quaternion to a rotation matrix.
    Args:
        q (torch.Tensor): shape [B, 4], input quaternion.

    Returns:
        torch.Tensor: [B, 4, 4] homogeneous rotation matrices.
    """
    assert q.shape[-1] == 4, "Not a valid quaternion"
    q = q / q.norm(dim=-1, keepdim=True)
    
    B = q.size(0)
    mat = torch.zeros((B, 4, 4), device=q.device)

    mat[:, 0, 0] = 1 - 2 * q[:, 2]**2 - 2 * q[:, 3]**2
    mat[:, 0, 1] = 2 * q[:, 1] * q[:, 2] - 2 * q[:, 3] * q[:, 0]
    mat[:, 0, 2] = 2 * q[:, 1] * q[:, 3] + 2 * q[:, 2] * q[:, 0]
    mat[:, 1, 0] = 2 * q[:, 1] * q[:, 2] + 2 * q[:, 3] * q[:, 0]
    mat[:, 1, 1] = 1 - 2 * q[:, 1]**2 - 2 * q[:, 3]**2
    mat[:, 1, 2] = 2 * q[:, 2] * q[:, 3] - 2 * q[:, 1] * q[:, 0]
    mat[:, 2, 0] = 2 * q[:, 1] * q[:, 3] - 2 * q[:, 2] * q[:, 0]
    mat[:, 2, 1] = 2 * q[:, 2] * q[:, 3] + 2 * q[:, 1] * q[:, 0]
    mat[:, 2, 2] = 1 - 2 * q[:, 1]**2 - 2 * q[:, 2]**2
    mat[:, 3, 3] = 1.0
    
    return mat

def tvector2mat(t: torch.Tensor) -> torch.Tensor:
    """
    Convert translation vectors to homogeneous transformation matrices.
    Args:
        t (torch.Tensor): shape=[B, 3], translation vectors.

    Returns:
        torch.Tensor: [B, 4, 4] homogeneous transformation matrices.
    """
    assert t.shape[-1] == 3, "Not a valid translation"
    
    B = t.size(0)
    mat = torch.eye(4, device=t.device).unsqueeze(0).repeat(B, 1, 1)
    mat[:, 0, 3] = t[:, 0]
    mat[:, 1, 3] = t[:, 1]
    mat[:, 2, 3] = t[:, 2]
    
    return mat

def mat2xyzrpy(rotmatrix: torch.Tensor) -> torch.Tensor:
    """
    Decompose transformation matrices into components (XYZ and Roll-Pitch-Yaw).
    Args:
        rotmatrix (torch.Tensor): [B, 4, 4] transformation matrices.

    Returns:
        torch.Tensor: shape=[B, 6], contains XYZ and Roll-Pitch-Yaw (rpy).
    """
    B = rotmatrix.size(0)
    x = rotmatrix[:, 0, 3]
    y = rotmatrix[:, 1, 3]
    z = rotmatrix[:, 2, 3]
    
    roll = torch.atan2(-rotmatrix[:, 1, 2], rotmatrix[:, 2, 2])
    pitch = torch.asin(rotmatrix[:, 0, 2])
    yaw = torch.atan2(-rotmatrix[:, 0, 1], rotmatrix[:, 0, 0])
    
    return torch.stack([x, y, z, roll, pitch, yaw], dim=-1)

def to_rotation_matrix(R: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """
    Combine rotation (from quaternion) and translation vectors into transformation matrices.
    Args:
        R (torch.Tensor): shape [B, 4], input quaternion.
        T (torch.Tensor): shape [B, 3], translation vectors.

    Returns:
        torch.Tensor: [B, 4, 4] homogeneous transformation matrices.
    """
    R_mat = quat2mat(R)
    T_mat = tvector2mat(T)
    RT = torch.bmm(T_mat, R_mat)
    return RT


def quatmultiply(q, r, device='cpu'):
    """
    Batch quaternion multiplication
    Args:
        q (torch.Tensor/np.ndarray): shape=[Nx4]
        r (torch.Tensor/np.ndarray): shape=[Nx4]
        device (str): 'cuda' or 'cpu'

    Returns:
        torch.Tensor: shape=[Nx4]
    """
    if isinstance(q, torch.Tensor):
        t = torch.zeros(q.shape[0], 4, device=device)
    elif isinstance(q, np.ndarray):
        t = np.zeros(q.shape[0], 4)
    else:
        raise TypeError("Type not supported")
    t[:, 0] = r[:, 0] * q[:, 0] - r[:, 1] * q[:, 1] - r[:, 2] * q[:, 2] - r[:, 3] * q[:, 3]
    t[:, 1] = r[:, 0] * q[:, 1] + r[:, 1] * q[:, 0] - r[:, 2] * q[:, 3] + r[:, 3] * q[:, 2]
    t[:, 2] = r[:, 0] * q[:, 2] + r[:, 1] * q[:, 3] + r[:, 2] * q[:, 0] - r[:, 3] * q[:, 1]
    t[:, 3] = r[:, 0] * q[:, 3] - r[:, 1] * q[:, 2] + r[:, 2] * q[:, 1] + r[:, 3] * q[:, 0]
    return t

def quatinv(q):
    """
    Batch quaternion inversion
    Args:
        q (torch.Tensor/np.ndarray): shape=[Nx4]

    Returns:
        torch.Tensor/np.ndarray: shape=[Nx4]
    """
    if isinstance(q, torch.Tensor):
        t = q.clone()
    elif isinstance(q, np.ndarray):
        t = q.copy()
    else:
        raise TypeError("Type not supported")
    t *= -1
    t[:, 0] *= -1
    return t


def quaternion_distance(q, r, device):
    """
    Batch quaternion distances, used as loss
    Args:
        q (torch.Tensor): shape=[Nx4]
        r (torch.Tensor): shape=[Nx4]
        device (str): 'cuda' or 'cpu'

    Returns:
        torch.Tensor: shape=[N]
    """
    t = quatmultiply(q, quatinv(r), device)
    return 2 * torch.atan2(torch.norm(t[:, 1:], dim=1), torch.abs(t[:, 0]))


def compute_angular_error(R: torch.Tensor) -> torch.Tensor:
    """
    Compute the angular error (in degrees) from a 3x3 rotation matrix R.
    The angle is computed as:
        theta = arccos((trace(R) - 1) / 2)
    Parameters:
      R (torch.Tensor): A 3x3 rotation matrix.
    Returns:
      torch.Tensor: The angular error in degrees.
    """
    trace = torch.trace(R)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)
    return theta * (180.0 / np.pi)
