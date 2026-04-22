# -*- coding: utf-8 -*-
"""
Created on Sun May  5 18:19:34 2024

@author: andre
"""

import numpy as np
import open3d as o3d

def downsample_point_cloud(point_cloud, num_points):
    """
    Downsample a point cloud to a specific number of points.
    
    Args:
    - point_cloud: numpy array of shape (N, 3) representing the point cloud
    - num_points: desired number of points after downsampling
    
    Returns:
    - downsampled_point_cloud: numpy array of shape (num_points, 3) representing the downsampled point cloud
    """
    num_points_original = point_cloud.shape[0]
    if num_points_original <= num_points:
        return point_cloud
    
    # Calculate the step size for downsampling
    step_size = num_points_original // num_points
    
    # Select points at regular intervals to downsample
    downsampled_point_cloud = point_cloud[::step_size]
    
    # If the last point is not included, add it
    if len(downsampled_point_cloud) < num_points:
        downsampled_point_cloud = np.vstack([downsampled_point_cloud, point_cloud[-1]])
    
    return downsampled_point_cloud

TXT_DIR_X = "point_cloud_2.txt"
TXT_DIR_Y = "output_y.txt"

# Load the contents of the TXT files
coord1 = np.loadtxt(TXT_DIR_X,delimiter=",")
coord2 = np.loadtxt(TXT_DIR_Y)

point_cloud1 = np.array(coord1)
point_cloud2 = np.array(coord2)

# Downsample the point cloud to 100 points
downsampled_point_cloud = downsample_point_cloud(point_cloud1, 10000)

# Convert numpy array to Open3D PointCloud
pcd1= o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(downsampled_point_cloud)
# Assign color to the second point cloud
pcd1.paint_uniform_color([1, 0, 0])  # green

o3d.visualization.draw_geometries([pcd1])