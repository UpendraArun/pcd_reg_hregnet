from dataset.dataset_utils import PointCloudProjection
import numpy as np 

import matplotlib.pyplot as plt

class VisualizeDataset:

    @staticmethod
    def plot_projected_points(data, all_points=True):

        # Get intrinsic matrix
        intrinsic_matrix = data['intrinsic'].squeeze(0).numpy()

        # Get point cloud 
        point_cloud_uncalib = data['uncalibed_pcd'].squeeze(0).numpy().T
        point_cloud_calib = data['point_cloud'].squeeze(0).numpy()
        intensity = data['intensity'].squeeze(0).numpy()

        # RGB image 
        

        image = data['image'].squeeze(0).numpy().transpose(1, 2, 0)

        # 
        projector = PointCloudProjection()

        if not all_points:
            # Get only the points on the image
            _, _, r = projector.binary_projection(img_shape=image.shape[:2], intrinsic=intrinsic_matrix, pcd=point_cloud_uncalib.T)
            point_cloud_uncalib = point_cloud_uncalib[r,:]

            # Get only the points on the image
            _, _, r = projector.binary_projection(img_shape=image.shape[:2], intrinsic=intrinsic_matrix, pcd=point_cloud_calib.T)
            point_cloud_calib = point_cloud_calib[r,:]
            intensity_calib = intensity[r]

        # Project points using the intrinsic matrix
        projected_points_uncalib = np.dot(intrinsic_matrix, point_cloud_uncalib.T).T
        projected_points_uncalib /= projected_points_uncalib[:, 2].reshape(-1, 1)  # Normalize by the third column (z

        # Project points using the intrinsic matrix
        projected_points_calib = np.dot(intrinsic_matrix, point_cloud_calib.T).T
        projected_points_calib /= projected_points_calib[:, 2].reshape(-1, 1)  # Normalize by the third column (z
    
        # Depth wise colouring
        coloring_uncalib = point_cloud_uncalib[:,2]
        coloring_calib = point_cloud_calib[:,2]


        # Plot the projected points on the image
        plt.figure(2, figsize=(16,9))
        plt.imshow(image)  # Blank image
        plt.scatter(projected_points_uncalib[:, 0], projected_points_uncalib[:, 1], c=coloring_uncalib, s =1)
        plt.title('Projected Points on Image Decalibrated')
        plt.axis('off')
        plt.show()

        # Plot the projected points on the image
        plt.figure(2, figsize=(16,9))
        plt.imshow(image)  # Blank image
        plt.scatter(projected_points_calib[:, 0], projected_points_calib[:, 1], c=coloring_calib, marker='o', s =2)
        plt.title('Projected Points on Image GT')
        plt.axis('off')
        plt.show()
        
    @staticmethod
    def plot_projected_points_decalib(data, all_points=True):

        # Get intrinsic matrix
        intrinsic_matrix = data['intrinsic'].squeeze(0).numpy()

        # Get point cloud 
        point_cloud_uncalib = data['uncalibed_pcd'].squeeze(0).numpy().T
        point_cloud_calib = data['point_cloud'].squeeze(0).numpy()

        # RGB image 
        image = data['image'].squeeze(0).numpy().transpose(1, 2, 0)

        # 
        projector = PointCloudProjection()

        if not all_points:
            # Get only the points on the image
            _, _, r = projector.binary_projection(img_shape=image.shape[:2], intrinsic=intrinsic_matrix, pcd=point_cloud_uncalib.T )
            point_cloud_uncalib = point_cloud_uncalib[r]

            # Get only the points on the image
            _, _, r = projector.binary_projection(img_shape=image.shape[:2], intrinsic=intrinsic_matrix, pcd=point_cloud_calib.T )
            point_cloud_calib = point_cloud_calib[r]

        # Project points using the intrinsic matrix
        projected_points_uncalib = np.dot(intrinsic_matrix, point_cloud_uncalib.T).T
        projected_points_uncalib /= projected_points_uncalib[:, 2].reshape(-1, 1)  # Normalize by the third column (z

        # Project points using the intrinsic matrix
        projected_points_calib = np.dot(intrinsic_matrix, point_cloud_calib.T).T
        projected_points_calib /= projected_points_calib[:, 2].reshape(-1, 1)  # Normalize by the third column (z
    
        # Depth wise colouring
        coloring_uncalib = point_cloud_uncalib[:,2]
        coloring_calib = point_cloud_calib[:,2]

        #print(intrinsic_matrix)
        #print(data['igt'])

        # Plot the projected points on the image
        plt.figure(2, figsize=(16,16), dpi = 300)
        plt.imshow(image)  # Blank image
        plt.scatter(projected_points_uncalib[:, 0], projected_points_uncalib[:, 1], c=coloring_uncalib, s =10)
        #plt.title('Projected Points on Image Decalibrated')
        plt.title('Initial Decalibration')
        plt.axis('off')
        plt.show()


    @staticmethod
    def plot_inference(data, predicted_tf, idx, all_points=True):

        def transform_point_cloud(point_cloud, transformation_matrix):
            # Ensure that the point cloud and transformation matrix are NumPy arrays.
            point_cloud = np.array(point_cloud)[0].T
            transformation_matrix = np.array(transformation_matrix)

            # Check the dimensions of the transformation matrix and the point cloud.
            if transformation_matrix.shape != (4, 4):
                raise ValueError("Transformation matrix must be a 4x4 matrix.")
            if point_cloud.shape[1] != 3:
                raise ValueError("Point cloud must have 3 columns (x, y, z).")

            # Add a column of ones to the point cloud to convert it to homogeneous coordinates.
            homogenous_point_cloud = np.hstack((point_cloud, np.ones((point_cloud.shape[0],1))))

            # Apply the transformation matrix to the homogeneous point cloud.
            transformed_point_cloud = np.dot(homogenous_point_cloud, transformation_matrix.T)

            # Divide by the fourth column to remove the homogeneous coordinate and get the transformed points.
            transformed_point_cloud = transformed_point_cloud[:, :3]

            return transformed_point_cloud
        
        image=data['img'].cpu().numpy()
        point_cloud_uncalib=data['uncalibed_pcd'].detach().numpy()
        point_cloud_calib=data['pcd'].detach().numpy()[0].T
        intrinsic_matrix= data['InTran'].detach().numpy()[0]
        predicted_tf= predicted_tf.cpu().detach().numpy()[0]


        # 
        projector = PointCloudProjection()

        # Get image 
        image = image[0].transpose(1, 2, 0)

        # Transfrom point cloud 
        point_cloud_uncalib = transform_point_cloud(point_cloud_uncalib, predicted_tf)

        if not all_points:
            u, v, r = projector.binary_projection(img_shape=image.shape[:2], intrinsic=intrinsic_matrix, pcd=point_cloud_uncalib.T )
            point_cloud_uncalib = point_cloud_uncalib[r]

            # Transfrom point cloud 
            u, v, r = projector.binary_projection(img_shape=image.shape[:2], intrinsic=intrinsic_matrix, pcd=point_cloud_calib.T )
            point_cloud_calib = point_cloud_calib[r]
        
        
        coloring = point_cloud_uncalib[:,2] 
        coloring_calib = point_cloud_calib[:,2] 

        # Project points using the intrinsic matrix
        projected_points = np.dot(intrinsic_matrix, point_cloud_uncalib.T).T
        projected_points /= projected_points[:, 2].reshape(-1, 1)  # Normalize by the third column (z)k

        # Project points using the intrinsic matrix
        projected_points_calib = np.dot(intrinsic_matrix, point_cloud_calib.T).T
        projected_points_calib /= projected_points_calib[:, 2].reshape(-1, 1)  # Normalize by the third column (z)k
        

        plt.figure(1,figsize=(16,16), dpi=300)
        plt.imshow(image)  # Blank image
        plt.scatter(projected_points[:, 0], projected_points[:, 1], c=coloring, s =10)
        plt.title('Projected Points on Image Predicted')
        plt.axis('off')
        plt.show()

        plt.figure(2,figsize=(16,16), dpi=300)
        plt.imshow(image)  # Blank image
        plt.scatter(projected_points_calib[:, 0], projected_points_calib[:, 1],c=coloring_calib, s =10)
        plt.title('Projected Points on Image GT')
        plt.axis('off')
        plt.show()
        plt.close('all')

    @staticmethod
    def plot_depth_image(data):
        plt.figure(figsize=(16,16))
        plt.subplot(1,2,1)
        plt.imshow(data['depth_img'].squeeze(0).numpy().transpose(1, 2, 0), cmap='viridis')
        plt.axis('off')
        plt.title('Calibrated Depth Image')
        plt.subplot(1,2,2)
        plt.title('Un-alibrated Depth Image')
        plt.imshow(data['uncalibed_depth_img'].squeeze(0).numpy().transpose(1, 2, 0), cmap='plasma')
        plt.axis('off')
        plt.show()

    
    @staticmethod
    def plot_training_logs(data_dict, x_label=None, y_labels=None, main_title=None):
        """
        Plot data from a dictionary in a 2x2 grid of subplots.

        Parameters:
        - data_dict: A dictionary where keys are dataset names and values are lists of data points.
        - x_label: Label for the x-axis (optional).
        - y_labels: List of labels for the y-axes (optional).
        - main_title: Title for the entire grid of subplots (optional).
        """
        num_plots = len(data_dict)
        num_rows, num_cols = 2,4 # 2x2 grid of subplots

        # Create a new figure with the specified number of rows and columns
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 8))

        # Flatten the axes into a 1D array for easier indexing
        axes = axes.flatten()

        # Iterate through the data_dict and plot each dataset in a subplot
        for i, (key, values) in enumerate(data_dict.items()):
            ax = axes[i]  # Select the current subplot
            ax.plot(values)
            ax.set_title(key)  # Use the dataset key as the subplot title
            ax.set_xlabel('Epochs')
            ax.set_ylabel(key)

        # Adjust layout and add a main title
        fig.tight_layout()
        if main_title:
            fig.suptitle(main_title, fontsize=16)

        # Show the plot
        plt.show()

    # @staticmethod
    # def plot_perturbation(data1, data2=None):
    #     if data1.shape[1] >= 6:
    #         num_cols = data1.shape[1]

    #         # Calculate the number of rows and columns for subplots
    #         num_rows = (num_cols // 2) + (num_cols % 2)  # At most 2 columns per row
    #         num_cols_per_row = 2  # Two columns per row

    #         # Create a new figure with subplots
    #         fig, axes = plt.subplots(num_rows, num_cols_per_row, figsize=(16, 16))

    #         # Flatten the axes array to make it easier to access
    #         axes = axes.flatten()

    #         y_axis = ['roll', 'pitch', 'yaw', 't_x', 't_y', 't_z']

    #         for col in range(num_cols):
    #             if col < num_cols:
    #                 # Plot points for the first dataset
    #                 axes[col].scatter(range(len(data1[:, col])), data1[:, col], label='Input Perturbation')
    #                 if data2 is not None:
    #                     # Check if a second dataset is provided and plot its points as well
    #                     axes[col].scatter(range(len(data2[:, col])), data2[:, col], label='Output Results')
    #                 axes[col].set_title(y_axis[col])
    #                 axes[col].set_xlabel('Data Samples')
    #                 axes[col].set_ylabel(y_axis[col])
    #                 axes[col].legend()

    #         # Hide any remaining empty subplots
    #         for col in range(num_cols, num_rows * num_cols_per_row):
    #             fig.delaxes(axes[col])

    #         # Adjust subplot layout to prevent overlap
    #         plt.tight_layout()
    #         plt.show()
    #     else:
    #         print("The array must have at least 6 columns to plot in subplots.")

   