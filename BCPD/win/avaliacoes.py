# -*- coding: utf-8 -*-
"""
Created on Mon May 20 09:58:20 2024

@author: andre
"""

import numpy as np
from scipy.spatial import KDTree
# import ot

# point_cloud_target = np.loadtxt("C:/Users/AndreJoao/Downloads/bcpd-master-final/bcpd-master/win/point_cloud_0.txt", delimiter=",")

# point_cloud_eval = np.loadtxt("C:/Users/AndreJoao/Desktop/Dados/Geodesic/15000/K/350/output_y.txt")

# teste_target=np.loadtxt('target_downsampled_15000.txt', delimiter=",")
# teste_source=np.loadtxt('C:/Users/AndreJoao/Desktop/CPD/15000/Lambda/1000/output_y.txt')
# corr=np.loadtxt("C:/Users/AndreJoao/Desktop/CPD/15000/Lambda/1000/output_e.txt", skiprows=1)

def chamfer_distance(A, B):
    """
    Computes the chamfer distance between two sets of points A and B.
    """
    tree = KDTree(B)
    dist_A = tree.query(A)[0]
    tree = KDTree(A)
    dist_B = tree.query(B)[0]
    return np.mean(dist_A) + np.mean(dist_B)

def hausdorff_distance(A, B):
    """
    Compute the Hausdorff distance between two point clouds A and B.

    Parameters:
    A (ndarray): First point cloud of shape (N, D)
    B (ndarray): Second point cloud of shape (M, D)

    Returns:
    float: Hausdorff distance between point cloud A and B
    """
    # Create KDTree for efficient nearest neighbor search
    tree_A = KDTree(A)
    tree_B = KDTree(B)
    
    # Compute directed Hausdorff distance from A to B
    distances_A_to_B = tree_B.query(A)[0]
    hausdorff_A_to_B = np.max(distances_A_to_B)
    
    # Compute directed Hausdorff distance from B to A
    distances_B_to_A = tree_A.query(B)[0]
    hausdorff_B_to_A = np.max(distances_B_to_A)
    
    # The Hausdorff distance is the maximum of the two directed distances
    hausdorff_distance = max(hausdorff_A_to_B, hausdorff_B_to_A)
    
    return hausdorff_distance


def rmsd(target, source, corr):
    """
    Compute the Root-Mean-Square Deviation (RMSD) between two point clouds.

    Parameters:
    point_cloud_1 (ndarray): First point cloud of shape (N, D)
    point_cloud_2 (ndarray): Second point cloud of shape (N, D)

    Returns:
    float: RMSD between point cloud 1 and point cloud 2
    """
    # Ensure the point clouds have the same number of points
    if len(target)!=len(source):
        raise ValueError("Point clouds must have the same size for RMSD calculation")

    diff=0
    for i in range(len(corr)):
        ponto_target = target[int(corr[i][0])-1] 
        # print("indice do target",i)
        ponto_source = source[int(corr[i][1]-1)]
        # print("indice da source", corr[i][1])
        diff+=(np.linalg.norm(ponto_target-ponto_source))**2
    mean=diff/len(corr)
    rmsd_value=np.sqrt(mean)    
    

    return rmsd_value


# # Calculate and print the Hausdorff distance
# hd = hausdorff_distance(teste_target, teste_source)
# print("Hausdorff Distance: ", hd)


# print("Chamfer Distance: ",chamfer_distance(teste_target, teste_source))

# # Calculate and print the RMSD
# rmsd_value = rmsd(teste_target, teste_source,corr)
# print("RMSD:", rmsd_value)

# file_path = "C:/Users/AndreJoao/Desktop/CPD/3000/Beta/0.1/output_comptime.txt"  
# with open(file_path, 'r') as file:
#     print("Time data:")
#     print(file.read())

# file_path = "C:/Users/AndreJoao/Desktop/CPD/3000/Beta/0.1/output_info.txt"
# with open(file_path, 'r') as file:
#     print("Output Info:")
#     print(file.read())