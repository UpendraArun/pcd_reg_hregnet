import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from dataset.dataset_utils import PointCloudProjection

class PointCloudInferenceVisualizer:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    @staticmethod
    def transform_point_cloud(point_cloud, transformation_matrix):
        point_cloud = np.array(point_cloud)[0].T
        transformation_matrix = np.array(transformation_matrix)

        if transformation_matrix.shape != (4, 4):
            raise ValueError("Transformation matrix must be a 4x4 matrix.")
        if point_cloud.shape[1] != 3:
            raise ValueError("Point cloud must have 3 columns (x, y, z).")

        homogenous_point_cloud = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
        transformed_point_cloud = np.dot(homogenous_point_cloud, transformation_matrix.T)
        return transformed_point_cloud[:, :3]

    @staticmethod
    def project_points(point_cloud, intrinsic_matrix):
        projected_points = np.dot(intrinsic_matrix, point_cloud.T).T
        projected_points /= projected_points[:, 2].reshape(-1, 1)
        return projected_points

    @staticmethod
    def plot_image_with_points(image, points, coloring, title, output_path):
        plt.figure(figsize=(16, 16), dpi=300)
        plt.imshow(image)
        plt.scatter(points[:, 0], points[:, 1], c=coloring, s=10)
        plt.title(title)
        plt.axis('off')
        plt.savefig(output_path)
        plt.close()

    def plot_inference(self, data, predicted_tf, idx, all_points=True):
        image = data['img'].cpu().numpy()[0].transpose(1, 2, 0)
        point_cloud_uncalib = data['uncalibed_pcd'].detach().numpy()
        point_cloud_calib = data['pcd'].detach().numpy()[0].T
        intrinsic_matrix = data['InTran'].detach().numpy()[0]
        predicted_tf = predicted_tf.cpu().detach().numpy()[0]

        projector = PointCloudProjection()
        point_cloud_uncalib = self.transform_point_cloud(point_cloud_uncalib, predicted_tf)

        if not all_points:
            u, v, r = projector.binary_projection(img_shape=image.shape[:2], intrinsic=intrinsic_matrix, pcd=point_cloud_uncalib.T)
            point_cloud_uncalib = point_cloud_uncalib[r]
            u, v, r = projector.binary_projection(img_shape=image.shape[:2], intrinsic=intrinsic_matrix, pcd=point_cloud_calib.T)
            point_cloud_calib = point_cloud_calib[r]

        coloring_uncalib = point_cloud_uncalib[:, 2]
        coloring_calib = point_cloud_calib[:, 2]

        projected_points_uncalib = self.project_points(point_cloud_uncalib, intrinsic_matrix)
        projected_points_calib = self.project_points(point_cloud_calib, intrinsic_matrix)

        output_path_pred = os.path.join(self.output_dir, f"plot_pred_{idx}.png")
        output_path_gt = os.path.join(self.output_dir, f"plot_gt_{idx}.png")

        self.plot_image_with_points(image, projected_points_uncalib, coloring_uncalib, 'Projected Points on Image Predicted', output_path_pred)
        self.plot_image_with_points(image, projected_points_calib, coloring_calib, 'Projected Points on Image GT', output_path_gt)

    def generate_video(self, num_frames, video_path="output/video.avi", fps=2):
        images = [os.path.join(self.output_dir, f"plot_pred_{i}.png") for i in range(num_frames)]
        images += [os.path.join(self.output_dir, f"plot_gt_{i}.png") for i in range(num_frames)]

        frame = cv2.imread(images[0])
        height, width, _ = frame.shape
        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

        for image_path in images:
            frame = cv2.imread(image_path)
            video.write(frame)

        video.release()
        print(f"Video saved to {video_path}")
