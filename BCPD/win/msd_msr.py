# -*- coding: utf-8 -*-
"""
Created on Sun May  5 18:07:16 2024

@author: andre
"""

import numpy as np

def mean_square_root(point_cloud1, point_cloud2):
    # Check if the two point clouds have the same number of points
    if len(point_cloud1) != len(point_cloud2):
        raise ValueError("Point clouds must have the same number of points.")

    # Compute pairwise Euclidean distances
    squared_distances = np.sum((point_cloud1 - point_cloud2)**2, axis=1)

    # Compute the mean of the squared distances
    mean_squared_distance = np.mean(squared_distances)

    return np.sqrt(mean_squared_distance)


def mean_square_root_deviation(pc1, pc2):
    """
    Compute the mean square root deviation between two point clouds.

    Parameters:
    pc1 (numpy.ndarray): First point cloud of shape (n, d).
    pc2 (numpy.ndarray): Second point cloud of shape (n, d), where n is the number of points and d is the dimension.

    Returns:
    float: Mean square root deviation between the two point clouds.
    """
    assert pc1.shape == pc2.shape, "Point clouds must have the same shape"

    # Compute Euclidean distance between corresponding points in the two point clouds
    distances = np.linalg.norm(pc1 - pc2, axis=1)

    # Compute the mean square root deviation
    msd = np.mean(distances)
    
    return msd

TXT_DIR_X = "point_cloud_2_down.txt"
TXT_DIR_Y = "output_y.txt"

# Load the contents of the TXT files
coord1 = np.loadtxt(TXT_DIR_X,delimiter=",")
coord2 = np.loadtxt(TXT_DIR_Y)

point_cloud1 = np.array(coord1)
point_cloud2 = np.array(coord2)

msr = mean_square_root(point_cloud1, point_cloud2)
msd=mean_square_root_deviation(point_cloud1, point_cloud2)
print("Mean square root between the two point clouds:", msr, msd)
