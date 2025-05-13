# Dataloaders for TruckScenes dataset
# torch
import torch 
from torchvision.transforms import transforms as Tf
from PIL import Image
from torch.utils.data import Dataset

#dataset 
from truckscenes import TruckScenes
from truckscenes.utils.data_classes import LidarPointCloud
from truckscenes.utils.splits import create_splits_scenes

# Utils 
from .dataset_utils import PointCloud180degFilter, PointCloudFilter, PointCloudProjection, PointCloudResampler, ToTensor, MinMaxScaler


# Transfroms 
from transform import SE3, UniformTransformSE3

# Config 
from config import Config

#tools 
import os 
import numpy as np 
import open3d as o3d
from pyquaternion import Quaternion
from typing import Any, Tuple

class TruckScenesLoader:
    def __call__(self, config:Config,  verbose: bool = True) -> TruckScenes:
        """Load the TruckScenes dataset.

        Args:
            config (Config) : Configuration, Check config.json
            verbose (bool, optional): Whether to display verbose information. Defaults to False.

        Returns:
            TruckScenes: TruckScenes dataset instance.
        """
        return TruckScenes(version=config.dataset_config.version, dataroot=config.dataset_config.path, verbose=verbose)

class TruckScenesDataset(Dataset):
    def __init__(self,  trucksc: TruckScenes, config:Config):
        """Initialize the TruckScenes dataset.

        Args:
            trucksc (TruckScenes): TruckScenes API instance.
            config: Configuration object containing various parameters.
        """
        # Initialize TruckScenes API instance
        self.config = config
        self.trucksc = trucksc
        self.split = config.dataset_config.split

        # Get the scene tokens based on the split
        self.scene_tokens = self.get_scene_tokens(split=self.split)
        
        # Limit the number of scenes to the first x scenes, defined in limscenes
        if self.config.dataset_config.limscenes is not None:
            self.scene_tokens = self.scene_tokens[:self.config.dataset_config.limscenes]

        # Get the sample tokens for the entire dataset
        self.sample_tokens = self.get_sample_tokens()

        # Tools      
        self.np_to_tensor = ToTensor(tensor_type=torch.float)
        self.img_to_tensor = Tf.ToTensor()
        
        self.range_scaler = MinMaxScaler(min_val=0, max_val=self.config.dataset_config.max_range)
        self.intensity_scaler = MinMaxScaler(min_val=0, max_val=self.config.dataset_config.max_intensity)
        self.point_cloud_sampler = PointCloudResampler(num_points=self.config.dataset_config.pcd_min_samples)
        self.point_cloud_filter = PointCloudFilter(voxel_size=self.config.dataset_config.voxel_size, concat='none', max_range= self.config.dataset_config.max_range)
        self.pt_projection = PointCloudProjection()
        self.point_cloud_180deg_filter = PointCloud180degFilter()

        super(TruckScenesDataset, self).__init__()

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.sample_tokens)

    def __getitem__(self, idx: int) -> dict:
        """Get a specific sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing data for the selected sample.
        """
        
        # Get the sample token at the given index
        sample_token = self.sample_tokens[idx]

        if self.config.dataset_config.mode == "L2L":
            return self.lidar_to_lidar(sample_token=sample_token)
        elif self.config.dataset_config.mode == "C2L":
            return self.camera_to_lidar(sample_token=sample_token)
        else:
            raise IOError("Invalid mode check dataset config")


    def lidar_to_lidar(self, sample_token):

        # Retrieve sample data from TruckScenes API
        sample = self.trucksc.get('sample', sample_token)

        # Get tokens
        sensor_token_a = sample['data'][self.config.dataset_config.lidar_tokens[0]]
        sensor_token_b = sample['data'][self.config.dataset_config.lidar_tokens[1]]

        # Load transformation matrix
        extrinsic = self.get_extrinsic_matrix(lidar_token_a=sensor_token_a, lidar_token_b=sensor_token_b)

        # Lidar to Lidar
        point_cloud_left, intensity_left = self.load_lidar_point_cloud(token=sensor_token_a,
                                                    extrinsic_matrix=extrinsic,
                                                    transform=False)
        
        
        point_cloud_right, intensity_right = self.load_lidar_point_cloud(token=sensor_token_b,
                                                    extrinsic_matrix=extrinsic,
                                                    transform=True)
        
        return dict(pcd_left=self.np_to_tensor(point_cloud_left),
                    intensity_left = self.np_to_tensor(intensity_left),
                    pcd_right=self.np_to_tensor(point_cloud_right),
                    intensity_right = self.np_to_tensor(intensity_right),
                    extrinsic=self.np_to_tensor(extrinsic))
    
    def camera_to_lidar(self, sample_token):
         # Retrieve sample data from TruckScenes API
        sample = self.trucksc.get('sample', sample_token)

        # Get tokens
        sensor_token_lidar = sample['data'][self.config.dataset_config.lidar_tokens[0]]
        sensor_token_camera = sample['data'][self.config.dataset_config.camera_tokens[0]]

        # Load calibration matrix for the camera
        intrinsic_resized, intrinsic_extended = self.get_intrinsic_matrix(sensor_token_camera)

        # Load camera front data
        image, image_size = self.load_image(sensor_token_camera)
        #print(type(image_size))

        # Load transformation matrix
        extrinsic = self.get_extrinsic_matrix(lidar_token_a=sensor_token_camera, lidar_token_b=sensor_token_lidar)
    
        # Cameera to Lidar
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
                image_size = image_size)
    

    def get_scene_tokens(self, split: str) -> list:
        """Get scene tokens based on the dataset split.

        Args:
            split (str): Dataset split ('train', 'val', 'test', 'mini_train' or 'mini_val').
        Returns:
            list: List of scene tokens.
        """
        
        # Get scene splits (assuming create_splits_scenes() is defined elsewhere)
        scene_splits = create_splits_scenes()

        # Select the appropriate scene tokens based on the split
        if split == 'train':
            #print(scene_splits[split])  # Debug: Print the list of 'split' scenes to see what's currently included
            # Iterate over scene names in scene_splits[split], check if the name exists in the dataset, then retrieve the token
            valid_scene_tokens = []
            for scene_name in scene_splits['train']:
                scene_tokens = self.trucksc.field2token('scene', 'name', scene_name)
                if scene_tokens:  # This checks if the list is not empty
                    valid_scene_tokens.append(scene_tokens[0])
            scene_splits['train'] = valid_scene_tokens  # Update scene_splits['val'] with valid tokens
        
        elif split == 'val':
            #print(scene_splits[split])  # Debug: Print the list of 'split' scenes to see what's currently included
            # Iterate over scene names in scene_splits[split], check if the name exists in the dataset, then retrieve the token
            valid_scene_tokens = []
            for scene_name in scene_splits['val']:
                scene_tokens = self.trucksc.field2token('scene', 'name', scene_name)
                if scene_tokens:  # This checks if the list is not empty
                    valid_scene_tokens.append(scene_tokens[0])
            scene_splits['val'] = valid_scene_tokens  # Update scene_splits['val'] with valid tokens
            #scene_splits[split] = [self.nusc.field2token('scene', 'name', scene_name)[0] for scene_name in scene_splits[split]]
        
        # elif split == 'train_detect':
        #     valid_scene_tokens = []
        #     for scene_name in scene_splits['train_detect']:
        #         scene_tokens = self.trucksc.field2token('scene', 'name', scene_name)
        #         if scene_tokens:  
        #             valid_scene_tokens.append(scene_tokens[0])
        #     scene_splits['train_detect'] = valid_scene_tokens

        elif split == 'test':
            valid_scene_tokens = []
            for scene_name in scene_splits['test']:
                scene_tokens = self.trucksc.field2token('scene', 'name', scene_name)
                if scene_tokens:  
                    valid_scene_tokens.append(scene_tokens[0])
            scene_splits['test'] = valid_scene_tokens  
        
        # elif split == 'mini_train':
        #     valid_scene_tokens = []
        #     for scene_name in scene_splits['mini_train']:
        #         scene_tokens = self.trucksc.field2token('scene', 'name', scene_name)
        #         if scene_tokens: 
        #             valid_scene_tokens.append(scene_tokens[0])
        #     scene_splits['mini_train'] = valid_scene_tokens
        
        # elif split == 'mini_val':
        #     valid_scene_tokens = []
        #     for scene_name in scene_splits['mini_val']:
        #         scene_tokens = self.trucksc.field2token('scene', 'name', scene_name)
        #         if scene_tokens:  
        #             valid_scene_tokens.append(scene_tokens[0])
        #     scene_splits['mini_val'] = valid_scene_tokens

        return scene_splits[split]

    def get_sample_tokens(self) -> list:
        """Get a list of sample tokens for all scenes.

        Returns:
            list: List of sample tokens.
        """
        sample_tokens = []
        for scene_token in self.scene_tokens:
            token = self.get_sample_token(scene_token=scene_token)
            sample_tokens.append(token)
        return sum(sample_tokens,[])

    def get_sample_token(self, scene_token: str) -> list:
        """Get sample tokens for a specific scene.

        Args:
            scene_token (str): Token of the scene.

        Returns:
            list: List of sample tokens in the scene.
        """
        sample_tokens = []
        scene = self.trucksc.get('scene', scene_token)
        sample_token = scene['first_sample_token']
        while sample_token:
            sample_tokens.append(sample_token)
            sample = self.trucksc.get('sample', sample_token)
            sample_token = sample['next']
        
        #  # To make sure that each batch contains scenes from only one sequence 
        # cut_index = len(sample_tokens)% self.config.dataset_config.batch_size
        # if cut_index > 0:
        #     sample_tokens = sample_tokens[:-cut_index]

        return sample_tokens
  
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

    def load_lidar_point_cloud(self, token: str, extrinsic_matrix: np.ndarray, transform: bool) -> np.ndarray:
        """Load and preprocess a lidar point cloud from the dataset.

        Args:
            token (str): Token of the sample data.
            extrinsic_matrix (np.ndarray): Extrinsic matrix for the lidar.
            radius (float, optional): Radius for removing close points. Defaults to 0.1.
            transform (Bool): Transforms the point cloud into a point cloud 'a' reference frame.

        Returns:
            np.ndarray: Preprocessed lidar point cloud as a NumPy array.
        """
        
        # Get path
        lidar_data = self.trucksc.get('sample_data', token)
        lidar_path = os.path.join(self.config.dataset_config.path, lidar_data['filename'])
        #print(f"Loading LiDAR file from: {lidar_path}")

        # Read pcd.bin file for lidar
        point_cloud = LidarPointCloud.from_file(lidar_path)
        intensity = np.array(point_cloud.points[3,:])

        # Remove close points
        #point_cloud.remove_close(radius=radius)

        # Transform point cloud
        if transform == True:
            point_cloud.transform(extrinsic_matrix)

        #Get points
        points = point_cloud.points[:3,:]     # (3, N) np.ndarray

        # Filter points which are within the 0 to 180 deg of each lidar
        if self.config.dataset_config.mode == 'L2L':
            #points = self.point_cloud_180deg_filter(points.T).T  # () np.ndarray
            #print(points.shape)
            pass 
                
        # Filter points using self.point_cloud_filter method (assuming it's defined elsewhere)
        points, intensity = self.point_cloud_filter(points.T, intensity.T)           # (3, N) np.ndarray
       
        # Subsample the points to keep uniform dimensions using self.point_cloud_sampler method (assuming it's defined elsewhere)
        points, intensity = self.point_cloud_sampler(points, intensity)         #(N, 3)
        
        # Scale intensity 
        #intensity = self.intensity_scaler(intensity)
   
        return points, intensity

    def get_intrinsic_matrix(self, cam_token: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the calibration matrix for a specific camera and its extended version.

        Args:
            cam_token (str): Token of the camera data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the original calibration matrix and extended calibration matrix as numpy arrays.
        """
        # Retrieve camera data and corresponding calibrated sensor data
        cam_data = self.trucksc.get('sample_data', cam_token)
        sensor_data = self.trucksc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        
        # Obtain the camera's intrinsic calibration matrix
        calibration_matrix = np.array(sensor_data['camera_intrinsic'])

        # Adjust calibration matrix according to re-size ratio
        calibration_matrix = np.diag([self.config.dataset_config.resize_ratio[1], self.config.dataset_config.resize_ratio[0], 1]) @ calibration_matrix

        # Create an extended calibration matrix by applying extend_ratio to translation components
        calibration_matrix_extended = calibration_matrix.copy()
        calibration_matrix_extended[0, -1] *= self.config.dataset_config.extend_ratio[0]
        calibration_matrix_extended[1, -1] *= self.config.dataset_config.extend_ratio[1]

        return calibration_matrix, calibration_matrix_extended
    
    def get_extrinsic_matrix(self, lidar_token_a:str, lidar_token_b:str) -> np.ndarray:

        """
        Compute the cumulative transformation matrix to transform a point cloud from the lidar frame to the camera frame.

        Args:
            lidar_token_a (str): Token of the lidar sample associated with the point cloud 'a'.
            lidar_token_b (str): Token of the lidar sample associated with the point cloud 'b'
            trucksc (TruckScenes): An instance of the TruckScenes class providing access to dataset information.

        Returns:
            np.ndarray: A 4x4 transformation matrix that captures the cumulative transformation process.
        """
        # Get information about the lidar_a
        # lidar_data = self.nusc.get('sample_data', lidar_token)
        lidar_a = self.trucksc.get('sample_data', lidar_token_a)
        #print(f"{pointsensor}\n")
        # Get information about the lidar_b
        lidar_b = self.trucksc.get('sample_data', lidar_token_b)
        #print(f"{cam}\n")

        # Transformation matrix for the first step: Sensor frame left with respect to ego vehicle frame at the sweep timestamp
        cs_record = self.trucksc.get('calibrated_sensor', lidar_a['calibrated_sensor_token'])
        transform = np.eye(4)
        transform[:3, :3] = Quaternion(cs_record['rotation']).rotation_matrix
        transform[:3, 3] = np.array(cs_record['translation'])
        transform = np.linalg.inv(transform)

        # Transformation matrix for the second step: Ego vehicle frame at the sweep timestamp to global frame
        pose_record = self.trucksc.get('ego_pose', lidar_a['ego_pose_token'])
        ego_to_global = np.eye(4)
        ego_to_global[:3, :3] = Quaternion(pose_record['rotation']).rotation_matrix
        ego_to_global[:3, 3] = np.array(pose_record['translation'])
        ego_to_global = np.linalg.inv(ego_to_global)
        transform = np.dot(transform, ego_to_global)        # Transformation 1

        # Transformation matrix for the third step: Global frame to ego vehicle frame at the image timestamp
        pose_record = self.trucksc.get('ego_pose', lidar_b['ego_pose_token'])
        global_to_ego = np.eye(4)
        global_to_ego[:3, :3] = Quaternion(pose_record['rotation']).rotation_matrix
        global_to_ego[:3, 3] = np.array(pose_record['translation'])
        #global_to_ego = np.linalg.inv(global_to_ego)
        transform = np.dot(transform, global_to_ego)        # Transformation 2

        # Transformation matrix for the fourth step: Ego vehicle frame to camera frame
        cs_record = self.trucksc.get('calibrated_sensor', lidar_b['calibrated_sensor_token'])
        ego_to_cam = np.eye(4)
        ego_to_cam[:3, :3] = Quaternion(cs_record['rotation']).rotation_matrix
        ego_to_cam[:3, 3] = np.array(cs_record['translation'])
        #ego_to_cam = np.linalg.inv(ego_to_cam)
        transform = np.dot(transform, ego_to_cam)           # Transformation 3
        #print(f"transformation matrix is: \n {transform.shape}")

        return transform
  
class TruckScenesPerturbation():
    def __init__(self, dataset: TruckScenesDataset, config:Config):
        """Initialize the TruckScenes perturbation dataset.

        Args:
            dataset (TruckScenesDataset): The base TruckScenes dataset.
            max_deg (float): Maximum degree of perturbation for random transformation.
            max_tran (float): Maximum translation for random transformation.
            mag_randomly (bool, optional): Whether to apply random magnitude to perturbations. Defaults to True.
            pooling_size (int, optional): Size of max-pooling kernel for image pooling. Defaults to 5.
            file (str, optional): Path to a perturbation file. Defaults to None.
        """

        self.config = config
        self.split = self.config.dataset_config.split

        if self.config.dataset_config.mode =='C2L':
            assert (self.config.dataset_config.pooling_size - 1) % 2 == 0, 'pooling size must be odd to keep image size constant'
            self.pooling = torch.nn.MaxPool2d(kernel_size=self.config.dataset_config.pooling_size, stride=1, padding=(self.config.dataset_config.pooling_size - 1) // 2)
            self.pt_projection = PointCloudProjection()

        self.se3 = SE3()
        self.dataset = dataset

        if self.split != 'train' and self.config.dataset_config.perturbations_file:
            
            self.perturb_path = self.config.dataset_config.perturbations_file + self.split + '.txt'
            #print(self.perturb_path)
            if os.path.exists(self.perturb_path):
                self.perturb = torch.from_numpy(np.loadtxt(self.perturb_path, dtype=np.float32, delimiter=','))[None, ...]  # (1, N, 6)
            else:
                self.__create_perturb_file(self.config, len(self.dataset), self.perturb_path)
                self.perturb = torch.from_numpy(np.loadtxt(self.perturb_path, dtype=np.float32, delimiter=','))[None, ...]
        
        else:
            # Get random transform
            self.transform = UniformTransformSE3(max_deg=self.config.dataset_config.max_rot_error,
                                                max_tran= self.config.dataset_config.max_trans_error,
                                                mag_randomly = self.config.dataset_config.mag_randomly,
                                                distribution=self.config.dataset_config.distribution)
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
       
        if self.split == 'train': # randomly generate igt (initial guess transformation)
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
        #print(f"the shape of calibed pcd is {calibed_pcd_right.shape}")
       
        if self.split == 'train': # randomly generate igt (initial guess transformation)
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
            #print(_uncalibed_pcd.shape)
            igt.squeeze_(0)  # (4, 4)

        # Add new item
        new_data = dict(uncalibed_pcd=_uncalibed_pcd.T, igt=igt)
        data.update(new_data)
        return data