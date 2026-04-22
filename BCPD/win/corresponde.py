# -*- coding: utf-8 -*-
"""
Created on Thu May  2 20:34:34 2024

@author: andre
"""

import numpy as np
from scipy.spatial import cKDTree

def find_correspondences(reference_points, transformed_points):
    # Build KD-tree for the transformed point cloud
    tree = cKDTree(transformed_points)
    
    # Query the closest point in the transformed point cloud for each point in the reference cloud
    distances, indices = tree.query(reference_points, k=1)
    
    # Return the indices of correspondences and their distances
    return indices, distances

def check_uniqueness(correspondences):
    # Count the occurrences of each correspondence
    correspondence_counts = np.bincount(correspondences)
    
    # Check if any correspondence occurs more than once
    is_unique = np.all(correspondence_counts == 1)
    
    return is_unique

def has_duplicate(nums):
    seen = set()
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    return False

def find_duplicates(nums):
    seen = {}
    duplicates = []
    for i, num in enumerate(nums):
        if num in seen:
            duplicates.append((seen[num], i))
        seen[num] = i
    return duplicates

def compute_uniqueness_metrics(correspondences):
    # Compute the percentage of unique correspondences
    print("Numero de unicos = ", np.count_nonzero(np.bincount(correspondences) == 1) )
    unique_percentage = np.count_nonzero(np.bincount(correspondences) == 1) / len(correspondences) * 100
    
    # Compute the average number of correspondences per point
    average_correspondences = len(correspondences) / len(np.unique(correspondences))
    
    return unique_percentage, average_correspondences

import open3d as o3d

def visualize_correspondences(reference_points, transformed_points, correspondences):
    # Create Open3D point cloud objects
    ref_cloud = o3d.geometry.PointCloud()
    ref_cloud.points = o3d.utility.Vector3dVector(reference_points)
    ref_cloud.paint_uniform_color([1, 0, 0])  # Red color for reference points
    
    trans_cloud = o3d.geometry.PointCloud()
    trans_cloud.points = o3d.utility.Vector3dVector(transformed_points)
    trans_cloud.paint_uniform_color([0, 0, 1])  # Blue color for transformed points
    
    # Create a line set for correspondences
    lines = []
    for i in range(len(correspondences)):
        ref_point = reference_points[i]
        trans_point = transformed_points[correspondences[i]]
        lines.append([ref_point, trans_point])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.concatenate(lines, axis=0))
    line_set.lines = o3d.utility.Vector2iVector([[i*2, i*2+1] for i in range(len(lines))])
    line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(lines))])  # Green color for lines
    
    # Visualize
    o3d.visualization.draw_geometries([ref_cloud, trans_cloud, line_set])


TXT_DIR_X = "output__downsampledY_before_deformation-3000.txt"
TXT_DIR_Y = "output_y-GCPD default 3000.txt"

# Load the contents of the TXT files
coord1 = np.loadtxt(TXT_DIR_X)
coord2 = np.loadtxt(TXT_DIR_Y)

reference_points = np.array(coord1)
transformed_points = np.array(coord2)
correspondences, distances = find_correspondences(reference_points, transformed_points)
visualize_correspondences(coord1, coord2, correspondences)