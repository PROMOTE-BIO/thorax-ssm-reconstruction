# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:33:06 2024

@author: andre
"""
import numpy as np
import open3d as o3

TXT_DIR_X = "point_cloud_0.txt"

# Load the contents of the TXT files and transform them into point clouds
coord1 = np.loadtxt(TXT_DIR_X, delimiter=",")
point1=np.array(coord1)
point_cloud1 = o3.geometry.PointCloud()
point_cloud1.points = o3.utility.Vector3dVector(point1)

# Define voxel size
voxel_size = 2

# Apply voxel grid downsampling
downsampled_cloud = point_cloud1.voxel_down_sample(voxel_size) #downsampling according to voxel size
downsampled_cloud.paint_uniform_color([1,0,0]) #red
print(downsampled_cloud)

o3.visualization.draw_geometries([downsampled_cloud]) #point cloud visualization

np.savetxt("C:/Users/andre/Downloads/bcpd-master/win/point_cloud_0_down.txt", downsampled_cloud.points , fmt="%.6f", delimiter=",")

print("Points saved to file txt")