# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 23:49:20 2024

@author: AndreJoao
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 17:51:49 2024

@author: andre
"""
import open3d as o3d
import numpy as np


# Load the first point cloud from a txt file
point_cloud1 = np.loadtxt('target_downsampled_3000.txt',delimiter=",")

# Convert numpy array to Open3D PointCloud
pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(point_cloud1)
# Assign color to the first point cloud
pcd1.paint_uniform_color([1, 0, 0])  # red

# Load the second point cloud from a txt file
point_cloud2 = np.loadtxt("CPDRes2-3000.txt")

# Convert numpy array to Open3D PointCloud
pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(point_cloud2)
# Assign color to the second point cloud
pcd2.paint_uniform_color([0, 1, 0])  # green

# # Create the first visualizer
# vis2 = o3d.visualization.Visualizer()
# vis2.create_window(window_name='Point Cloud 2')

# # Add geometry to the first visualizer
# vis2.add_geometry(pcd2)

# # Update the visualizer and keep the window open
# vis2.update_geometry(pcd2)
# vis2.poll_events()
# vis2.update_renderer()

# Load the first point cloud from a txt file
point_cloud3 = np.loadtxt('output_y-BCPD rigido 3000.txt')

# Convert numpy array to Open3D PointCloud
pcd3 = o3d.geometry.PointCloud()
pcd3.points = o3d.utility.Vector3dVector(point_cloud3)
# Assign color to the first point cloud
pcd3.paint_uniform_color([0, 0, 1])  # blue

# # Create the first visualizer
# vis3 = o3d.visualization.Visualizer()
# vis3.create_window(window_name='Point Cloud 2')

# # Add geometry to the first visualizer
# vis3.add_geometry(pcd3)

# # Update the visualizer and keep the window open
# vis3.update_geometry(pcd3)
# vis3.poll_events()
# vis3.update_renderer()

# Function to check for window close condition
def should_close(vis):
    return vis.poll_events() and vis.update_renderer()

#Visualize both point clouds
o3d.visualization.draw_geometries([pcd1,pcd3])


# # Run the visualizers with a loop and a key press condition
# try:
#     while True:
#         if not should_close(vis2) or not should_close(vis3):
#             break
# except KeyboardInterrupt:
#     pass
# finally:
#     vis2.destroy_window()
#     vis3.destroy_window()



