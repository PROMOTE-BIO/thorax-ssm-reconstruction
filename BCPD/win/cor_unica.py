# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 12:26:06 2024

@author: andre
"""

import numpy as np
import copy
import open3d as o3d

def extract_features(point_cloud):
    # Compute normals
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Compute FPFH features
    fpfh = o3d.registration.compute_fpfh_feature(point_cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))
    
    return fpfh

def main():
    TXT_DIR_X = "bunny-x.txt"
    TXT_DIR_Y = "bunny-y.txt"

    # Load the contents of the TXT files
    coord1 = np.loadtxt(TXT_DIR_X, delimiter=",")
    coord2 = np.loadtxt(TXT_DIR_Y, delimiter=",")

    point1=np.array(coord1)
    point2=np.array(coord2)

    point_cloud1 = o3d.geometry.PointCloud()
    point_cloud1.points = o3d.utility.Vector3dVector(point1)

    point_cloud2 = o3d.geometry.PointCloud()
    point_cloud2.points = o3d.utility.Vector3dVector(point2)

    target_cloud=copy.deepcopy(point_cloud2)
    source_cloud=copy.deepcopy(point_cloud1)

    # Extract features
    source_features = extract_features(source_cloud)
    target_features = extract_features(target_cloud)

    # Find correspondences
    correspondence = o3d.registration.registration_ransac_based_on_feature_matching(
        source_cloud, target_cloud, source_features, target_features, max_correspondence_distance=0.05,
        estimation_method=o3d.registration.TransformationEstimationPointToPoint(False), 
        ransac_n=4, checkers=[o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                              o3d.registration.CorrespondenceCheckerBasedOnDistance(0.075)],
        criteria=o3d.registration.RANSACConvergenceCriteria(4000000, 500))

    print("Number of correspondences found:", len(correspondence.correspondence_set))

if __name__ == "__main__":
    main()



