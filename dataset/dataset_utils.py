import torch 
import numpy as np 
from typing import Tuple
import open3d as o3d

class ToTensor:
    def __init__(self, tensor_type: torch.dtype = torch.float):
        """ToTensor class for converting NumPy arrays to PyTorch tensors.

        Args:
            tensor_type (torch.dtype, optional): The desired data type for the converted tensor.
                Defaults to torch.float if not provided.
        """
        # Private attribute with a leading underscore for data encapsulation
        self._tensor_type = tensor_type
    
    def __call__(self, x: np.ndarray) -> torch.Tensor:
        """Convert the input NumPy array to a PyTorch tensor.

        Args:
            x (np.ndarray): Input NumPy array to be converted.

        Returns:
            torch.Tensor: PyTorch tensor created from the input NumPy array, with the specified data type.
        """
        return torch.from_numpy(x).type(self._tensor_type)

class PointCloudProjection:
    @staticmethod
    def pcd_projection(img_shape: Tuple[int, int], intrinsic: np.ndarray, pcd: np.ndarray, range_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Project a 3D point cloud onto a 2D depth image.

        Args:
            img_shape (Tuple[int, int]): Tuple (H, W) representing the height and width of the output image.
            intrinsic (np.ndarray): 3x3 intrinsic matrix.
            pcd (np.ndarray): 3xN array of 3D points to be projected.
            range_arr (np.ndarray): N-dimensional array containing the ranges of points in the point cloud.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - np.ndarray: Array of projected u-coordinates.
                - np.ndarray: Array of projected v-coordinates.
                - np.ndarray: Array of ranges (r) for each projected point.
                - np.ndarray: Boolean array indicating whether points are within the image bounds and have positive depth values (rev).
        """
        H, W = img_shape  # (943, 1980)

        proj_pcd = intrinsic @ pcd  #(3,3) @ (3,1200) = (3,12000)
       

        u, v, w = proj_pcd[0, :], proj_pcd[1, :], proj_pcd[2, :]

        #print(u.shape, v.shape, w.shape)

        # Avoid using np.asarray unnecessarily
        u = (u / w).astype(np.int64)
        v = (v / w).astype(np.int64)

        # Avoid redundant comparison (0 <= u) and (0 <= v) are unnecessary
        rev = (0 <= u) & (u < W) & (0 <= v) & (v < H) & (w > 0)
        
        # Filter arrays using the 'rev' mask
        u = u[rev]
        v = v[rev]
        r = range_arr[rev]

        return u, v, r, rev

    @staticmethod
    def binary_projection(img_shape: Tuple[int, int], intrinsic: np.ndarray, pcd: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project a 3D point cloud onto a 2D binary image.

        Args:
            img_shape (Tuple[int, int]): Tuple (H, W) representing the height and width of the output image.
            intrinsic (np.ndarray): 3x3 intrinsic matrix.
            pcd (np.ndarray): 3xN array of 3D points to be projected.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - np.ndarray: Array of projected u-coordinates.
                - np.ndarray: Array of projected v-coordinates.
                - np.ndarray: Boolean array indicating whether points are within the image bounds and have positive depth values (rev).
        """
        H, W = img_shape
        proj_pcd = intrinsic @ pcd
        u, v, w = proj_pcd[0, :], proj_pcd[1, :], proj_pcd[2, :]

        # Avoid using np.asarray unnecessarily
        u = (u / w).astype(np.int64)
        v = (v / w).astype(np.int64)

        # Avoid redundant comparison (0 <= u) and (0 <= v) are unnecessary
        rev = (0 <= u) & (u < W) & (0 <= v) & (v < H) & (w > 0)

        return u, v, rev
     
class PointCloudFilter:
    def __init__(self, voxel_size: float = 0.6, concat: str = 'none', max_range:float = 100):
        """PointCloudFilter class for processing point cloud data.

        Args:
            voxel_size (float, optional): Voxel size for downsampling. Defaults to 0.3.
            concat (str, optional): Concatenation operation for normal estimation.
                Possible values: 'none', 'xyz', or 'zero-mean'. Defaults to 'none'.
        """
        # Private attributes with leading underscore for data encapsulation
        self._voxel_size = voxel_size
        self._concat = concat
        self._max_range = max_range

    def remove_points_by_range(self, point_cloud: np.ndarray, intensity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        """ Filters the input point cloud by given range
        Args:
            point_cloud (np.ndarray): Input point cloud data as a NumPy array (N, 3).
            intensity   """

        range = np.linalg.norm(point_cloud, axis=1)

        point_cloud = point_cloud[range< self._max_range, :]
        intensity = intensity[range< self._max_range]

        return point_cloud, intensity

    def fiter_points(self, x: np.ndarray) -> np.ndarray:
        """Process the input point cloud data and return the result.

        Args:
            x (np.ndarray): Input point cloud data as a NumPy array (N, 4).

        Returns:
            np.ndarray: Processed point cloud data as a NumPy array.
                The output shape depends on the 'concat' mode:
                - 'none': (N, 3)
                - 'xyz': (N, 6)
                - 'zero-mean': (N, 6)
        """
        # Create an Open3D PointCloud object from the input data
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(x[:,:3])
        #pcd.colors = o3d.utility.Vector3dVector(x[:,3:])
        
        # Voxel downsample the PointCloud based on the specified voxel size
        pcd = pcd.voxel_down_sample(self._voxel_size)
        
        points = np.asarray(pcd.points, dtype=np.float32)
        #colors = np.asarray(pcd.colors, dtype=np.float32)

        #pcd_xyz = np.hstack([points, colors])
        pcd_xyz = points
        if self._concat == 'none':
            # Return the point cloud data as-is
            return points 
        else:
            # Estimate normals for the point cloud
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self._voxel_size*3, max_nn=30))
            pcd.normalize_normals()
            pcd_norm = np.array(pcd.normals, dtype=np.float32)
            
            if self._concat == 'xyz':
                # Concatenate XYZ coordinates and normals
                return np.hstack([pcd_xyz, pcd_norm])  # (N, 3), (N, 3) -> (N, 6)
            elif self._concat == 'zero-mean':
                # Center the point cloud and adjust normals
                center = np.mean(pcd_xyz, axis=0, keepdims=True)  # (3,)
                pcd_zero = pcd_xyz - center
                pcd_norm *= np.where(np.sum(pcd_zero * pcd_norm, axis=1, keepdims=True) < 0, -1, 1)
                return np.hstack([pcd_zero, pcd_norm])  # (N, 3), (N, 3) -> (N, 6)
            else:
                raise RuntimeError('Unknown concat mode: %s' % self._concat)
    
    def __call__(self, point_cloud, intensity)-> Tuple[np.ndarray, np.ndarray]:
        return self.remove_points_by_range(point_cloud=point_cloud, intensity=intensity)
               
class PointCloudResampler:
    def __init__(self, num_points: int = 1024):
        """PointCloudResampler class for resampling point clouds.

        Args:
            num_points (int, optional): The desired number of points in the resampled point cloud.
                Defaults to 1024 if not provided.
        """
        # Private attribute with a leading underscore for data encapsulation
        self._num_points = num_points

    def __call__(self, point_cloud: np.ndarray, intensity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Resample the input point cloud to the desired number of points.

        Args:
            point_cloud (np.ndarray): The input point cloud as a numpy array, where each row
                represents a point with its coordinates.

        Returns:
            np.ndarray: The resampled point cloud with the desired number of points.
                The output will either be a padded version of the input cloud
                or a random subsample, depending on the number of points in the input cloud.
        """
        num_points_in_cloud = point_cloud.shape[0]

        if self._num_points != -1:
            if num_points_in_cloud <= self._num_points:
                # Pad with random points if the input cloud has fewer points
                pad_count = self._num_points - num_points_in_cloud
                pad_indices = np.random.choice(num_points_in_cloud, pad_count, replace=True)
                padded_cloud = np.vstack((point_cloud, point_cloud[pad_indices]))
                if intensity is not None:
                    padded_intensity = np.hstack((intensity, intensity[pad_indices]))
                    return padded_cloud, padded_intensity
                return padded_cloud, None
            else:
                # Randomly subsample if the input cloud has more points
                selected_indices = np.random.choice(num_points_in_cloud, self._num_points, replace=False)
                resampled_cloud = point_cloud[selected_indices]
                if intensity is not None:
                    resampled_intensity = intensity[selected_indices]
                    return resampled_cloud, resampled_intensity
                return resampled_cloud, None
        else:
            if intensity is not None:
                return point_cloud, intensity
            return point_cloud, None

class PointCloud180degFilter:
    def __init__(self):
        pass

    def __call__(self, point_cloud: np.ndarray) -> np.ndarray:          

        x = point_cloud[:, 0]
        y = point_cloud[:, 1]

        # Calculate the angle theta in radians (-π to π)
        theta = np.arctan2(y, x)  # This gives the angle in radians

        # Convert theta from radians to degrees for easy comparison
        theta_degrees = np.degrees(theta)  # Convert radians to degrees

        # Filter points to keep only those within the 0-180 degree range
        # theta_degrees range between [-180, 180], so we want to keep those between -90 and 90 degrees
        
        filtered_indices = (theta_degrees >= -130) & (theta_degrees <= 50)

        # Apply the filter to get the filtered points
        filtered_points = point_cloud[filtered_indices]

        """ filtered_point_cloud = o3d.geometry.PointCloud()
        filtered_point_cloud.points = o3d.utility.Vector3dVector(point_cloud)

        all_bounds = filtered_point_cloud.get_axis_aligned_bounding_box()
        print(all_bounds.get_max_bound())
        
        filtered_point_cloud = np.asarray(filtered_point_cloud.points)
        min_x = all_bounds.get_min_bound()[0]
        max_x = all_bounds.get_max_bound()[0]
        min_y = all_bounds.get_min_bound()[1]
        max_y = all_bounds.get_max_bound()[1]
        min_z = all_bounds.get_min_bound()[2]
        max_z = all_bounds.get_max_bound()[2]
        
        filter_mask = (
            (filtered_point_cloud[:, 0] >= 0) & (filtered_point_cloud[:, 0] <= max_x) &
            (filtered_point_cloud[:, 1] >= 0) & (filtered_point_cloud[:, 1] <= max_y)
        )
        filtered_points = filtered_point_cloud[filter_mask]
        #print(f"points along x filtered") """

        return filtered_points

class MinMaxScaler:
    def __init__(self, min_val: float = 0, max_val: float = 100):
        """
        Initializes the MinMaxScaler with fixed min and max values.

        Args:
            min_val (float): Minimum value of the data range to be scaled (default: 0).
            max_val (float): Maximum value of the data range to be scaled (default: 100).
        """
        self.min_val = min_val
        self.max_val = max_val
        self.scale_ = 1 / (self.max_val - self.min_val)
        self.min_ = -self.min_val * self.scale_


    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Transform the data to the range [0, 1].

        Args:
            data (np.ndarray): Data to be scaled.

        Returns:
            np.ndarray: Scaled data in the range [0, 1].
        """
        return (data * self.scale_) + self.min_

    def inverse_scale(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform the data from the range [0, 1] back to the original range.

        Args:
            data (np.ndarray): Data to be inverse transformed.

        Returns:
            np.ndarray: Data in the original range.
        """
        return (data - self.min_) / self.scale_ + self.min_val