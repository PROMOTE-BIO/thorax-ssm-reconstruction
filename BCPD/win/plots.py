# -*- coding: utf-8 -*-
"""
Created on Sat May 18 18:57:21 2024

@author: AndreJoao
"""

# import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Load the first point cloud from a txt file
point_cloud1 = np.loadtxt("C:/Users/AndreJoao/Desktop/interpolações/Bayesian/Down-15000/output_y.interpolated-k350.txt")
# Extract X, Y, Z coordinates
x = point_cloud1[:, 0]
y = point_cloud1[:, 1]
z = point_cloud1[:, 2]

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the points
ax.scatter(x, y, z, s=1, c=z, cmap='jet', marker='o')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Point Cloud Interpolated')

# Show the plot
plt.show()
# # Convert numpy array to Open3D PointCloud
# pcd1 = o3d.geometry.PointCloud()
# pcd1.points = o3d.utility.Vector3dVector(point_cloud1)
# # Assign color to the first point cloud
# pcd1.paint_uniform_color([1, 0, 0])  # red

# Load the second point cloud from a txt file
point_cloud2 = np.loadtxt("output_y.txt")
# Extract X, Y, Z coordinates
x = point_cloud2[:, 0]
y = point_cloud2[:, 1]
z = point_cloud2[:, 2]

# Create a figure and a 3D axis
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the points
ax.scatter(x, y, z, s=1, c=z, cmap='jet', marker='o')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Point Cloud Normal')

# Show the plot
plt.show()
# # Convert numpy array to Open3D PointCloud
# pcd2 = o3d.geometry.PointCloud()
# pcd2.points = o3d.utility.Vector3dVector(point_cloud2)
# # Assign color to the second point cloud
# pcd2.paint_uniform_color([0, 1, 0])  # green

# Load the first point cloud from a txt file
point_cloud3 = np.loadtxt("point_cloud_0.txt", delimiter=",")
# Extract X, Y, Z coordinates
x = point_cloud3[:, 0]
y = point_cloud3[:, 1]
z = point_cloud3[:, 2]

# Create a figure and a 3D axis
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the points
ax.scatter(x, y, z, s=1, c=z, cmap='jet', marker='o')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Point Cloud Target')

# Show the plot
plt.show()
# # Convert numpy array to Open3D PointCloud
# pcd3 = o3d.geometry.PointCloud()
# pcd3.points = o3d.utility.Vector3dVector(point_cloud3)
# # Assign color to the first point cloud
# pcd3.paint_uniform_color([0, 0, 1])  # blue

# # Visualize both point clouds
# o3d.visualization.draw_geometries([pcd1,pcd2,pcd3])
