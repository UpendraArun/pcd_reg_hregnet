# torch
import torch 
from torchvision.transforms import transforms as Tf
from PIL import Image
from torch.utils.data import Dataset
import sys

#dataset 


# Utils 
from .dataset_utils import PointCloud180degFilter, PointCloudFilter, PointCloudProjection, PointCloudResampler, ToTensor, MinMaxScaler


# Transfroms 
from transform import SE3, UniformTransformSE3

# Config 
from config import Config

#tools 
import os 
import numpy as np 
import numpy.linalg as la
import open3d as o3d
from pyquaternion import Quaternion
from typing import Any, Tuple
import json
import pprint
from os.path import join
import glob


class A2D2Dataset(Dataset):
    def __init__(self,  config:Config):
        """Initialize the A2D2 dataset.

        Args:
            config: Configuration object containing various parameters.
        """
        # Initialize A2D2 API instance
        self.config = config
        
        with open(self.config.dataset_config.cams_lidars_json, 'r') as f:
            self.cams_lidars_json = json.load(f)
        
        self.epsilon = self.config.dataset_config.epsilon
        
        # Initialize the split and the split ratios
        self.split = self.config.dataset_config.split
        self.split_ratios = self.config.dataset_config.split_ratios
        self.root_path = self.config.dataset_config.dataset_root
        
        # Assign the sensors
        self.sensor_a = self.config.dataset_config.sensor_a_key
        self.sensor_b = self.config.dataset_config.sensor_b_key
        
        if self.config.dataset_config.mode == 'L2L':

            self.lidar_a_list = self.get_lidar_files(self.root_path, self.sensor_a) 
            self.lidar_b_list = self.get_lidar_files(self.root_path, self.sensor_b)
            
            assert len(self.lidar_a_list) == len(self.lidar_b_list), "Mismatch in Lidar files count."
            
            # Pair files from both LiDARs for splitting into train, 
            self.pairs = list(zip(self.lidar_a_list, self.lidar_b_list))
            

        elif self.config.dataset_config.mode == 'C2L':

            self.camera_list = self.get_camera_files(self.root_path, self.sensor_a) 
            self.lidar_list = self.get_lidar_files(self.root_path, self.sensor_b)
            
            assert len(self.camera_list) == len(self.lidar_list), "Mismatch in Camera Lidar files count."
            
            # Pair files from both LiDARs for splitting into train, 
            self.pairs = list(zip(self.camera_list, self.lidar_list))


        else:
            raise IOError("Invalid Sensor mode, check config")

        # Split the dataset
        self.train_pairs, self.val_pairs, self.test_pairs = self.split_dataset()
        

        # Depending on the phase, choose the correct set of pairs
        if self.split == 'train':
            self.data = self.train_pairs
        elif self.split == 'val':
            self.data = self.val_pairs
        elif self.split == 'test':
            self.data = self.test_pairs
        else:
            raise IOError("Invalid split, check config")

        #print(self.data)
        # Limit the number of scenes to the first x scenes, defined in limscenes
        """ if self.config.dataset_config.limscenes is not None:
            self.scene_tokens = self.scene_tokens[:self.config.dataset_config.limscenes] """


        # Tools      
        self.np_to_tensor = ToTensor(tensor_type=torch.float)
        self.img_to_tensor = Tf.ToTensor()
        
        self.range_scaler = MinMaxScaler(min_val=0, max_val=self.config.dataset_config.max_range)
        self.intensity_scaler = MinMaxScaler(min_val=0, max_val=self.config.dataset_config.max_intensity)
        self.point_cloud_sampler = PointCloudResampler(num_points=self.config.dataset_config.pcd_min_samples)
        self.point_cloud_filter = PointCloudFilter(voxel_size=self.config.dataset_config.voxel_size, concat='none', max_range= self.config.dataset_config.max_range)
        self.pt_projection = PointCloudProjection()
        self.point_cloud_180deg_filter = PointCloud180degFilter()

        super(A2D2Dataset, self).__init__()


    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """Get a specific sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing data for the selected sample.
        """
              
        if self.config.dataset_config.mode == "L2L":
            return self.lidar_to_lidar(idx=idx)
        elif self.config.dataset_config.mode == "C2L":
            return self.camera_to_lidar(idx=idx)
        else:
            raise IOError("Invalid mode check dataset config")
        
        
    def lidar_to_lidar(self, idx: int) -> dict: 

        # Load Lidar data(npz file) for both Lidars
        lidar_a = np.load(self.lidar_a_list[idx]) 
        lidar_b = np.load(self.lidar_b_list[idx])
        #print(list(sensor_token_a.keys()))
        
        # Get both Lidars transformed into Global(Vehicle) frame
        lidar_a_view = self.cams_lidars_json['cameras'][self.sensor_a]['view']
        lidar_b_view = self.cams_lidars_json['cameras'][self.sensor_b]['view']     
        target_view = self.cams_lidars_json['vehicle']['view']

        # Compute the transformation matrix from source to target
        extrinsic_a = self.transform_from_to(lidar_a_view, target_view)
        extrinsic_b = self.transform_from_to(lidar_b_view, target_view)

        """ sensor_token_a, extrinsic_left  = self.project_lidar_from_to(lidar_a, lidar_a_view, target_view)
        sensor_token_b, extrinsic_right = self.project_lidar_from_to(lidar_b, lidar_b_view, target_view) """
        
        extrinsic_a_inv = np.linalg.inv(extrinsic_a)
        extrinsic  = np.dot(extrinsic_a_inv, extrinsic_b)         

        # Load, transform and preprocess point clouds 
        point_cloud_left, intensity_left = self.load_lidar_point_cloud(lidar_data=lidar_a,
                                                    extrinsic_matrix=extrinsic,
                                                    transform=False)
        
        point_cloud_right, intensity_right = self.load_lidar_point_cloud(lidar_data=lidar_b,
                                                    extrinsic_matrix=extrinsic,
                                                    transform=True)
        
        return dict(pcd_left        = self.np_to_tensor(point_cloud_left),
                    intensity_left  = self.np_to_tensor(intensity_left),
                    pcd_right       = self.np_to_tensor(point_cloud_right),
                    intensity_right = self.np_to_tensor(intensity_right),
                    extrinsic       = self.np_to_tensor(extrinsic))
    
    def camera_to_lidar(self, sample_token):
         # Retrieve sample data from A2D2 API
        sample = self.trucksc.get('sample', sample_token)

        # Get tokens
        sensor_token_lidar = sample['data'][self.config.dataset_config.lidar_tokens[0]]
        sensor_token_camera = sample['data'][self.config.dataset_config.camera_tokens[0]]

        # Load calibration matrix for the camera
        intrinsic_resized, intrinsic_extended = self.get_intrinsic_matrix(sensor_token_camera)

        # Load camera front data
        image, image_size = self.load_image(sensor_token_camera)

        # Load transformation matrix
        extrinsic = self.get_extrinsic_matrix(lidar_token_a=sensor_token_camera, lidar_token_b=sensor_token_lidar)
    
        # Camera to Lidar
        point_cloud_left, intensity_left = self.load_lidar_point_cloud(token=sensor_token_lidar,
                                                    extrinsic_matrix=extrinsic,
                                                    transform=True)
        
        depth_image, pcd_range, intensity_left = self.get_depth_image(point_cloud=point_cloud_left.T,
                                                    image_size=image_size,
                                                    intrinsic_matrix=intrinsic_resized,
                                                    intensity=intensity_left)

        return dict(point_cloud=self.np_to_tensor(point_cloud_left),
                intensity = self.np_to_tensor(intensity_left),
                extrinsic=self.np_to_tensor(extrinsic),
                image=self.img_to_tensor(image),
                depth_img = self.img_to_tensor(depth_image),
                intrinsic = self.np_to_tensor(intrinsic_resized),
                pcd_range=self.np_to_tensor(pcd_range),
                image_size = self.img_to_tensor(image_size))
    
  
    def load_image(self, token: str) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Load and preprocess an image from the dataset.

        Args:
            token (str): Token of the sample data.

        Returns:
            Tuple[torch.Tensor, Tuple[int, int]]: A tuple containing the preprocessed image as a PyTorch tensor and the dimensions of the resized image.
        """
        # Retrieve sample data for the given token
        cam_data = self.trucksc.get('sample_data', token)
        image_path = os.path.join(self.trucksc.dataroot, cam_data['filename'])

        # Load and preprocess the image using PIL
        image = Image.open(image_path).convert('RGB')
        
        # Calculate resized dimensions
        resized_height = round(image.height * self.config.dataset_config.resize_ratio[0])
        resized_width = round(image.width * self.config.dataset_config.resize_ratio[1])
        
        # Resize the image using bilinear interpolation and normalize pixel values
        image = image.resize([resized_width, resized_height], Image.BILINEAR)

        # Convert the image to a PyTorch tensor and adjust dimensions
        #image_tensor = self.img_to_tensor(image)
        image_tensor = np.asarray(image)
        #print(type(image))

        return image_tensor, (resized_height, resized_width)
    
    def get_depth_image(self, point_cloud: np.ndarray, image_size: Tuple[int, int], intrinsic_matrix: np.ndarray, intensity:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates a depth image and a range image from a given point cloud using projection onto a 2D plane.

        Args:
            point_cloud (np.ndarray): 3D point cloud with shape (3, num_points) containing (x, y, z) coordinates.
            image_size (Tuple[int, int]): Size of the output images in pixels (height, width).
            intrinsic_matrix (np.ndarray): 3x3 intrinsic matrix used for projection.

        Returns:
            torch.Tensor: Generated depth image with shape (height, width), containing depths of projected points.
            torch.Tensor: Tensor containing the ranges of points in the point cloud.
        """
        # Get size
        height, width = image_size
        
        # Calculate the Euclidean distances (ranges) of each point in the point cloud
        pcd_range = np.linalg.norm(point_cloud, axis=0)
        pcd_range = self.range_scaler(pcd_range)
        
        # Perform point cloud projection using provided intrinsic matrix
        u, v, r, idx = self.pt_projection.pcd_projection((height, width), intrinsic_matrix, point_cloud, pcd_range)

        # Create the depth image as a PyTorch tensor
        depth_img = torch.zeros((height, width, 2), dtype=torch.float32)
        depth_img[v, u, 0] = torch.from_numpy(r).type(torch.float32)
        depth_img[v, u, 1] = torch.from_numpy(intensity[idx]).type(torch.float32)
        
        # Convert the ranges to a PyTorch tensor
        pcd_range = torch.from_numpy(pcd_range).type(torch.float32)

        depth_img = depth_img.numpy()
        pcd_range = pcd_range.numpy()
        
        return depth_img, pcd_range, intensity

    def load_lidar_point_cloud(self, lidar_data: dict, extrinsic_matrix: np.ndarray, transform: bool) -> np.ndarray:
        """Load and preprocess a lidar point cloud from the dataset.

        Args:
            lidar (dict): lidar dict.
            extrinsic_matrix (np.ndarray): Extrinsic matrix for the lidar.
            transform (Bool): Transforms the point cloud into a point cloud 'a' reference frame.

        Returns:
            np.ndarray: Preprocessed lidar point cloud points and intensities as a NumPy arrays.

        """

        #token = self.project_lidar_from_to(lidar=token, src_view=src_view, target_view=target_view)

        points = lidar_data['pcloud_points']        # (N,3)
        intensity = lidar_data['pcloud_attr.reflectance'] # (N, )

        # Remove close points
        #point_cloud.remove_close(radius=radius)

        # Transform point cloud
        if transform == True:
            """ point_cloud = np.dot(extrinsic_matrix[:3,:3],point_cloud)
            # Translate the point cloud
            point_cloud = point_cloud + extrinsic_matrix[:3,3].reshape(3,1)
            point_cloud = point_cloud
            print(point_cloud.shape) """
            points_hom = np.ones((points.shape[0], 4))
            points_hom[:, 0:3] = points
            points_trans = (np.dot(extrinsic_matrix, points_hom.T)).T 
            points = points_trans[:,0:3]

        points = points.T     # (3, N) np.ndarray

        # Filter points which are within the 0 to 180 deg of each lidar
        if self.config.dataset_config.mode == 'L2L':
            #points = self.point_cloud_180deg_filter(points.T).T  # () np.ndarray
            #print(points.shape)
            pass 
                
        # Filter points using self.point_cloud_filter method (assuming it's defined elsewhere)
        points, intensity = self.point_cloud_filter(points.T, intensity.T)           # (3, N) np.ndarray
       
        # Subsample the points to keep uniform dimensions using self.point_cloud_sampler method (assuming it's defined elsewhere)
        points, intensity = self.point_cloud_sampler(points, intensity)         # (N, 3) np.ndarray
        
        # Scale intensity 
        intensity = self.intensity_scaler(intensity)
   
        return points, intensity


    def split_dataset(self):
        # Total number of pairs
        total_files = len(self.pairs)

        # Calculate the split indices
        train_split = int(self.split_ratios[0] * total_files)
        val_split = int((self.split_ratios[0] + self.split_ratios[1]) * total_files)

        # Split the pairs into training, validation, and testing sets
        train_pairs = self.pairs[:train_split]
        val_pairs = self.pairs[train_split:val_split]
        test_pairs = self.pairs[val_split:]

        return train_pairs, val_pairs, test_pairs
    
    def get_lidar_files(self, root, lidar_token):
        lidar_files = []
        for dirpath, _, filenames in os.walk(root):
            for file in filenames:
                if file.endswith('.npz') and lidar_token in dirpath:
                    lidar_files.append(os.path.join(dirpath, file))
        return sorted(lidar_files)

    def get_axes_of_a_view(self, view):
        x_axis = view['x-axis']
        y_axis = view['y-axis']
        x_axis_norm = la.norm(x_axis)
        y_axis_norm = la.norm(y_axis)
        if (x_axis_norm < self.epsilon or y_axis_norm < self.epsilon):
            raise ValueError("Norm of input vector(s) too small.")
        x_axis = x_axis / x_axis_norm
        y_axis = y_axis / y_axis_norm
        y_axis = y_axis - x_axis * np.dot(y_axis, x_axis)
        z_axis = np.cross(x_axis, y_axis)
        y_axis_norm = la.norm(y_axis)
        z_axis_norm = la.norm(z_axis)
        if (y_axis_norm < self.epsilon) or (z_axis_norm < self.epsilon):
            raise ValueError("Norm of view axis vector(s) too small.")
        y_axis = y_axis / y_axis_norm
        z_axis = z_axis / z_axis_norm
        return x_axis, y_axis, z_axis

    def get_origin_of_a_view(self, view):
        return view['origin']

    def get_transform_to_global(self, view):
        x_axis, y_axis, z_axis = self.get_axes_of_a_view(view)
        origin = self.get_origin_of_a_view(view)
        transform_to_global = np.eye(4)
        transform_to_global[0:3, 0] = x_axis
        transform_to_global[0:3, 1] = y_axis
        transform_to_global[0:3, 2] = z_axis
        transform_to_global[0:3, 3] = origin
        return transform_to_global

    def get_transform_from_global(self, view):
        transform_to_global = self.get_transform_to_global(view)
        trans = np.eye(4)
        rot = np.transpose(transform_to_global[0:3, 0:3])
        trans[0:3, 0:3] = rot
        trans[0:3, 3] = np.dot(rot, -transform_to_global[0:3, 3])
        return trans

    def transform_from_to(self, src_view, target_view):
        trans_src_to_global = self.get_transform_to_global(src_view)
        trans_global_to_target = self.get_transform_from_global(target_view)
        return np.dot(trans_global_to_target, trans_src_to_global)
    
    def project_lidar_from_to(self, lidar, src_view, target_view):
        lidar = dict(lidar)
        trans = self.transform_from_to(src_view, target_view)
        points = lidar["pcloud_points"]
        points_hom = np.ones((points.shape[0], 4))
        points_hom[:, 0:3] = points
        points_trans = (np.dot(trans, points_hom.T)).T 
        lidar["pcloud_points"] = points_trans[:,0:3]
        return lidar, trans
  
class A2D2Perturbation():
    def __init__(self, dataset: A2D2Dataset, config:Config):
        """Initialize the A2D2 perturbation dataset.

        Args:
            dataset (A2D2Dataset): The base A2D2 dataset.
            config (Config) : The configuration for A2D2 dataset.
        """

        self.config = config
        if self.config.dataset_config.mode =='C2L':
            assert (self.config.dataset_config.pooling_size - 1) % 2 == 0, 'pooling size must be odd to keep image size constant'
            self.pooling = torch.nn.MaxPool2d(kernel_size=self.config.dataset_config.pooling_size, stride=1, padding=(self.config.dataset_config.pooling_size - 1) // 2)
            self.pt_projection = PointCloudProjection()

        self.se3 = SE3()
        self.dataset = dataset

        if self.config.dataset_config.perturbations_file is not None and (self.config.dataset_config.split == 'val' 
                                                                          or self.config.dataset_config.split == 'test'):
            if os.path.exists(self.config.dataset_config.perturbations_file):
                self.perturb = torch.from_numpy(np.loadtxt(self.config.dataset_config.perturbations_file, dtype=np.float32, delimiter=','))[None, ...]  # (1, N, 6)

            else:
                self.__create_perturb_file(self.config, len(self.dataset), self.config.dataset_config.perturbations_file)
                self.perturb = torch.from_numpy(np.loadtxt(self.config.dataset_config.perturbations_file, dtype=np.float32, delimiter=','))[None, ...]

        
        #elif self.config.dataset_config.perturbations_file is not None and self.config.dataset_config.split == 'test':
            #if os.path.exists(self.config.dataset_config.perturbations_file):
                #self.perturb = torch.from_numpy(np.loadtxt(self.config.dataset_config.perturbations_file, dtype=np.float32, delimiter=','))[None, ...]  # (1, N, 6)
            #else:
                #self.__create_perturb_file(self.config, len(self.dataset), self.config.dataset_config.perturbations_file)
                #self.perturb = torch.from_numpy(np.loadtxt(self.config.dataset_config.perturbations_file, dtype=np.float32, delimiter=','))[None, ...]
        
        
        else:
            # Get random transform
            self.transform = UniformTransformSE3(max_deg=self.config.dataset_config.max_rot_error,
                                                max_tran= self.config.dataset_config.max_trans_error,
                                                mag_randomly = self.config.dataset_config.mag_randomly,
                                                distribution=self.config.dataset_config.distribution)
            #print(self.transform)
            #self.transform = transform.MaxTransformSE3(max_deg, max_tran, mag_randomly)
            #self.transform = transform.RandomTransformSE3(max_deg, max_tran, mag_randomly)

    def __len__(self) -> int:
        """Get the length of the perturbation dataset.

        Returns:
            int: Number of samples in the perturbation dataset.
        """
        return len(self.dataset)

    def __create_perturb_file(self, config: Config, length:int, path_to_file:str) -> None:
        """
        Creates a perturbation file for the perturbation dataset.

        Args:
            config (Config): Configuration object containing dataset settings.
            length (int): Number of entries in the perturbation file.
            path_to_file (str): Path to save the perturbation file.
        """
        transform = UniformTransformSE3(
            max_deg=config.dataset_config.max_rot_error,
            max_tran=config.dataset_config.max_trans_error,
            distribution=config.dataset_config.distribution,
            mag_randomly=config.dataset_config.mag_randomly
        )
        perturb_arr = np.zeros([length, 6])
        for i in range(length):
            perturb_arr[i, :] = transform.generate_transform().cpu().numpy()
        np.savetxt(path_to_file, perturb_arr, delimiter=',')
    
    def __getitem__(self, index: int) -> dict:
        """Get a specific sample from the perturbation dataset.

        Args:
            index (int): Index of the sample.

        Returns:
            dict: A dictionary containing perturbed data for the selected sample.
        """
        data = self.dataset[index]

        if self.config.dataset_config.mode == 'C2L':
            return self.camera_to_lidar(data, index)
        
        elif self.config.dataset_config.mode == 'L2L':
            return self.lidar_to_lidar(data, index)
        else:
            raise IOError("Check dataset config")

    
    def camera_to_lidar(self, data:dict, index=None):
        
        # Get iamge 
        H, W = data['image_size']
        calibed_pcd_left = data['point_cloud'][:,:3]
        intrinsic = data['intrinsic']
       
        if self.config.dataset_config.split == 'mini_train': # randomly generate igt (initial guess transformation)
            igt = self.se3.exp(self.transform.generate_transform()).squeeze(0)
            # Rotate the point cloud
            _uncalibed_pcd = np.dot(calibed_pcd_left,(igt[:3,:3].numpy()).T)
            # Translate the point cloud
            _uncalibed_pcd = _uncalibed_pcd + igt[:3,3:].squeeze(1).numpy()
            igt = igt.squeeze(0)  # (4, 4)

        else:
            igt = self.se3.exp(self.perturb[:, index, :])  # (1, 6) -> (1, 4, 4)
            _uncalibed_pcd = self.se3.transform(igt, calibed_pcd_left[None, ...]).squeeze(0)  # (3, N)
            igt.squeeze_(0)  # (4, 4)

        _uncalibed_depth_img = torch.zeros_like(data['depth_img'])

        #print(_uncalibed_depth_img.shape)
       
        u, v, r, idx = self.pt_projection.pcd_projection(data["image_size"], data["intrinsic"], _uncalibed_pcd.T, data["pcd_range"])
        # intensity = data["intensity"][idx]
        # Create the depth image as a PyTorch tensor
        _uncalibed_depth_img[0, v, u] = torch.from_numpy(r).type(torch.float32)
        _uncalibed_depth_img[1, v, u] = torch.from_numpy(data["intensity"][idx]).type(torch.float32)


         # Add new item
        new_data = dict(uncalibed_pcd=_uncalibed_pcd, uncalibed_depth_img = _uncalibed_depth_img, igt=igt)
        data.update(new_data)
        data['depth_img'] = self.pooling(data['depth_img'][None, ...])
        data['uncalibed_depth_img'] = self.pooling(data['uncalibed_depth_img'][None, ...])

        return data

    def lidar_to_lidar(self, data:dict, index=None):
        
        # Get image

        calibed_pcd_right = data['pcd_right'][:,:3].T       # (N, 3)
       
        if self.config.dataset_config.split == 'train': # randomly generate igt (initial guess transformation)
            igt = self.se3.exp(self.transform.generate_transform())
            # Rotate the point cloud
            _uncalibed_pcd = self.se3.transform(igt, calibed_pcd_right[None, ...]).squeeze(0)  #np.dot(calibed_pcd_right,(igt[:3,:3].numpy()).T)
            # Translate the point cloud
            # _uncalibed_pcd = _uncalibed_pcd + igt[:3,3:].squeeze(1).numpy()
            igt = igt.squeeze(0)  # (4, 4)
        else:
            igt = self.se3.exp(self.perturb[:, index, :])  # (1, 6) -> (1, 4, 4)
            #print(f"the shape of igt is {igt.shape}")
            _uncalibed_pcd = self.se3.transform(igt, calibed_pcd_right[None, ...]).squeeze(0) 
            igt.squeeze_(0)  # (4, 4)

        # Add new item
        new_data = dict(uncalibed_pcd=_uncalibed_pcd.T, igt=igt)
        data.update(new_data)
        return data