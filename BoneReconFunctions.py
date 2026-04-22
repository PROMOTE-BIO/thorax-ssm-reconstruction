# -*- coding: utf-8 -*-

"""
BoneReconFunctions.py

Collection of functions used for anatomical bone geometry reconstruction
based on bony or skin landmarks.

This script aggregates utilities for:
- Coordinate system definition
- Landmark processing
- Statistical Shape Model (SSM) handling
- Optimization-based reconstruction
"""

# =========================
# Standard library imports
# =========================
import time
import copy
import os
import pickle

# =========================
# Third-party libraries
# =========================
import numpy as np
import pandas as pd
import pygad
import trimesh
import pyvista as pv
import open3d as o3d
import skopt
from scipy.optimize import minimize, direct
from gradient_free_optimizers import PatternSearch, DirectAlgorithm, RandomRestartHillClimbingOptimizer, RandomAnnealingOptimizer, BayesianOptimizer, LipschitzOptimizer, RandomSearchOptimizer, ParallelTemperingOptimizer, GridSearchOptimizer

# =========================
# Project-specific modules
# =========================
from SSMFunctions import SSMReconstruction, SphereLandmarks
from MeshProperties import MeshDataToPolyData, SurfaceMeshSampling
from Registration import CustomCPD, trimesh_to_open3d

# =========================
# Color definitions (Open3D)
# =========================
# Colors normalized to the [0, 1] range as required by Open3D
green = [0,1,0]
black = [0,0,0]
red = [1,0,0]
blue = [0,0,1]
purple = [1,0,1]


# ============================================================
# Coordinate system and landmark transformation utilities
# ============================================================

def GetLocalReferenceFrame(df: pd.DataFrame, boneopt: str, reference: str):
    
    """
    Compute the local anatomical reference frame for a given bone.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing landmark coordinates.
        Columns represent landmark labels.
        The first three rows correspond to x, y, z coordinates
        expressed in the global coordinate system.
    boneopt : str
        Identifier of the bone (e.g., thorax, clavicle, scapula).
    reference : str
        Reference convention used for defining the local frame

    Returns
    -------
    R : np.ndarray
        Rotation matrix from global to local reference frame.
    t : np.ndarray
        Translation vector from global to local reference frame.
    H : np.ndarray
        Homogeneous transformation matrix
    """
    
    if boneopt not in ['Scapula', 'Clavicle', 'Humerus', "Thorax"]:
        
        raise Exception('Please select a bone of the shoulder joint (Scapula, Humerus or Clavicle).')
    
    if boneopt == 'Scapula':
        
        # Get the numpy arrays of the landmarks coordinates needed for the transformation
        AAvec = np.asarray(df['AA'], dtype = float)
        TSvec = np.asarray(df['TS'], dtype = float)
        AIvec = np.asarray(df['AI'], dtype = float)
        ACvec = np.asarray(df['AC'], dtype = float)
        
        if reference == 'old':
            
            PCvec = np.asarray(df['PC'], dtype = float)
        
        if reference == 'new':
        
            # Compute the unit vectors of the local coordinate frame
            # With the coordinates in the original reference frame
            # According to the ISB article
            new_z = (AAvec - TSvec) / np.linalg.norm(AAvec - TSvec)  # unitary
            new_x = np.cross(AAvec - TSvec, AIvec - TSvec)  # pointing forward
            new_x = new_x / np.linalg.norm(new_x) # unitary
            new_y = np.cross(new_z, new_x) # pointing upward and unitary
            
            # Compute the general transformation matrix (4x4)
            # General transformation matrix
            # Identity matrix
            T = np.eye(4)
            # Rotation of the axis
            T[0, :3], T[1, :3], T[2, :3] = new_x, new_y, new_z
            # Translation of the origin
            # The sign must be negative so that the formula returns (0,0,0)
            T[:3, 3] = - np.dot(T[:3, :3], AAvec)
        
        elif reference == 'old':
            
            # Compute the unit vectors of the local coordinate frame
            # With the coordinates in the original reference frame
            # According to the ISB article
            new_x = (ACvec - TSvec) / np.linalg.norm(ACvec - TSvec)  # unitary
            new_z = np.cross(TSvec - ACvec, AIvec - ACvec)  # pointing backward
            new_z = new_z / np.linalg.norm(new_z) # unitary
            new_y = np.cross(-new_x, new_z) # pointing upward and unitary
            new_y = new_y/np.linalg.norm(new_y)
            
            # Compute the general transformation matrix (4x4)
            # General transformation matrix
            # Identity matrix
            T = np.eye(4)
            # Rotation of the axis
            T[0, :3], T[1, :3], T[2, :3] = new_x, new_y, new_z
            # Translation of the origin
            # The sign must be negative so that the formula returns (0,0,0)
            T[:3, 3] = - np.dot(T[:3, :3], ACvec)
    
    #TODO make 2 new elif clauses for the Humerus and Clavicle bone transformations, according to ISB article
    elif boneopt == "Thorax":
        # Get the numpy arrays of the landmarks coordinates needed for the transformation
        C7vec = np.asarray(df['C7'], dtype = float)
        T8vec = np.asarray(df['T8'], dtype = float)
        JNvec = np.asarray(df['JN'], dtype = float)
        XPvec = np.asarray(df['XP'], dtype = float)
        R10vec = np.asarray(df['R10'], dtype = float)

        Ot = JNvec
        MXP_T8 = (XPvec + T8vec)/2
        MJN_C7 = (JNvec + C7vec)/2
        yt = MJN_C7 - MXP_T8
        Yt = yt/np.linalg.norm(yt) # normalizar o vetor
        zt = np.cross((JNvec - MXP_T8),(C7vec - MXP_T8))
        Zt = zt/np.linalg.norm(zt) # normalizar o vetor
        Xt = np.cross(Yt, Zt)

        # Compute the general transformation matrix (4x4)
        # General transformation matrix
        # Identity matrix
        T = np.eye(4)
        # Rotation of the axis
        T[0, :3], T[1, :3], T[2, :3] = Xt, Yt, Zt
        # Translation of the origin
        # The sign must be negative so that the formula returns (0,0,0)
        T[:3, 3] = - np.dot(T[:3, :3], Ot)

    # Return the rotation, translation and general matrix
    return T[:3, :3], T[:3, 3], T

def TransformToLocalReferenceFrame(arr: np.array, R: np.array, t: np.array):
    
    """
    Transform a set of 3D points from the global coordinate system
    to a local anatomical reference frame.

    Parameters
    ----------
    arr : np.ndarray
        Array of 3D points with shape (N, 3).
    R : np.ndarray
        Rotation matrix defining the local reference frame.
    t : np.ndarray
        Translation vector defining the local origin.

    Returns
    -------
    np.ndarray
        Transformed points in the local reference frame.
    """
    
    new_arr = np.asarray([np.matmul(R, point) + t for point in arr])
    
    return new_arr

def GetLocalLandmark(filepath: str, boneopt: str):
    
    """
    Load landmark coordinates from file and express them
    in the local anatomical reference frame.

    Parameters
    ----------
    filepath : str
        Path to the landmark file.
    boneopt : str
        Identifier of the bone (e.g., thorax, clavicle, scapula).

    Returns
    -------
    df: pandas.dataFrame
        Landmark coordinates in the local reference frame.
    T : np.ndarray
        Homogeneous transformation matrix
    """
    
    # Get the landmarks from a random geometry 
    tagdfcoords = pd.read_csv(filepath, sep = ',', header = None, index_col = False).T 
    
    # Make columns the landmarks and the three entries the x, y, z coordinates
    tagdfcoords.columns = tagdfcoords.iloc[0]
    tagdfcoords.drop(0, inplace = True)  # index remains 1, 2 and 3
    
    # Get translation and rotation matrixes for alignment
    R, t, T = GetLocalReferenceFrame(tagdfcoords, boneopt, 'new')
    
    # Transform data to numpy array and apply transformation
    tagcoords = np.array(tagdfcoords.T).astype(float)
    newtagcoords = TransformToLocalReferenceFrame(tagcoords, R, t)
    
    # Make data frame with landmark indicative and local coordinates
    df = pd.DataFrame(newtagcoords.T, columns = tagdfcoords.columns, index = ['xcoord', 'ycoord', 'zcoord'], 
                      dtype = float)
    
    return df, T

def GetLocalLandmarkFromCoord(tagdfcoords, boneopt: str):
    
    """
    Convert landmark coordinates already loaded in memory
    into the corresponding local reference frame.

    Parameters
    ----------
    tagdfcoords : pandas.DataFrame
        Landmark coordinates in global space.
    boneopt : str
        Bone identifier.

    Returns
    -------
    df: pandas.dataFrame
        Landmark coordinates in the local reference frame.
    T : np.ndarray
        Homogeneous transformation matrix
    """
    
    # This function is similar to GetLocalLandmark, but instead of reading coordinates
    # from a file, it already receives these as input

    # Get translation and rotation matrixes for alignment
    R, t, T = GetLocalReferenceFrame(tagdfcoords, boneopt, 'new')
    
    # Transform data to numpy array and apply transformation
    tagcoords = np.array(tagdfcoords.T).astype(float)
    newtagcoords = TransformToLocalReferenceFrame(tagcoords, R, t)
    
    # Make data frame with landmark indicative and local coordinates
    df = pd.DataFrame(newtagcoords.T, columns = tagdfcoords.columns, index = ['xcoord', 'ycoord', 'zcoord'], 
                      dtype = float)
    
    return df, T

def ReadSSMInfo(StatModel_info: dict, boneopt: str, plots: bool = False):
    
    """
    Read and parse the metadata associated with a Statistical Shape Model.

    Parameters
    ----------
    StatModel_info : dict
        Dictionary containing SSM paths, parameters and metadata.
    boneopt : str
        Bone identifier.
    plots : bool, optional
        If True, intermediate visualizations are displayed.

    Returns
    -------
    dict
        Processed Statistical Shape Model information.
    """

    # Make copy of pickle file for editing
    StatModel = copy.deepcopy(StatModel_info)

    # Compute the MVShape mesh from the vertices and triangles numpy arrays
    StatModel['MVShape']['Mesh'] = o3d.geometry.TriangleMesh()
    StatModel['MVShape']['Mesh'].vertices = o3d.cpu.pybind.utility.Vector3dVector(StatModel_info['MVShape']['Mesh']['Vertices'])
    StatModel['MVShape']['Mesh'].triangles = o3d.cpu.pybind.utility.Vector3iVector(StatModel_info['MVShape']['Mesh']['Triangles'])
    StatModel['MVShape']['Mesh'].compute_triangle_normals()
    StatModel['MVShape']['Mesh'].compute_vertex_normals()
    if plots:
        o3d.visualization.draw_geometries([StatModel['MVShape']['Mesh']])

    # Comput the MVShape point cloud from the points numpy array
    StatModel['MVShape']['Pcd'] =  o3d.geometry.PointCloud()
    StatModel['MVShape']['Pcd'].points = o3d.cpu.pybind.utility.Vector3dVector(StatModel_info['MVShape']['Pcd']['Points'])
    StatModel['MVShape']['Pcd'].estimate_normals()
    if plots:
        o3d.visualization.draw_geometries([StatModel['MVShape']['Pcd']])
    
    return StatModel

def ProcessAverageShape(StatModel: dict, boneopt: str, mean_shape_path, plots: bool = False, radius=0): 
    
    """
    Load and preprocess the mean shape of a Statistical Shape Model.

    Parameters
    ----------
    StatModel : dict
        Statistical Shape Model structure.
    boneopt : str
        Bone identifier.
    mean_shape_path : str
        Path to the mean shape geometry.
    plots : bool, optional
        Enable visualization for debugging or inspection.
    radius : float, optional
        Optional radius for landmark sphere visualization.

    Returns
    -------
    closest_points_av: np.array
        Indices of the mean shape in the SSM closest to the bony landmarks manually measured in the mean shape
    pointsAroundList: list
        Indices of the mean shape in the SSM closest to the bony landmarks manually measured in the mean shape considering an uncertainty radius
    closestIndex: list
    
    """

    ## Find closest points,in AverageShape, to the AVShape bony landmarks
    ## Might be different from the MVShape because of the added mean displacement added (it is, that is why we do it)
    ## The SD added is assumed to not modify the point closest to the landmarks 
    
    # Get average shape
    AVShape = SSMReconstruction(StatModel, [0], None, originalsize = True)
    
    # Read the avshape landmarks (already in inertia reference frame and in real size)
    filepath = mean_shape_path + "\\Mean_Bony_Landmarks.txt"
        
    tagdfcoords = pd.read_csv(filepath, sep = ',', header = None, index_col = False).T 
    tagdfcoords.columns = tagdfcoords.iloc[0]
    tagdfcoords.drop(0, inplace = True)
    if boneopt == 'Scapula':
        tagdfcoords = tagdfcoords[['AA', 'TS', 'AI', 'AC']]
    elif boneopt == "Thorax":
        tagdfcoords = tagdfcoords[['C7', 'T8', 'JN', 'XP', 'R10']]

    #TODO create 2 elif's for Clavicle and Humeruss
    avlandmarks = np.array(tagdfcoords.T).astype(float)
    
    # Submit the av shape landmarks to the same process of the SSM
    # Create pcd with landmarks from av shape
    landmarkavpcd = o3d.geometry.PointCloud(o3d.cpu.pybind.utility.Vector3dVector(avlandmarks))
    # Get points
    landmarksavpoints = copy.deepcopy(np.asarray(landmarkavpcd.points))
    
    # Find the points in av that most closely match its bony landmarks
    closest_points_av = np.zeros(len(avlandmarks), dtype = int)
    NPoints = np.asarray(AVShape['Pcd'].points).shape[0]
    avpoints = copy.deepcopy(np.asarray(AVShape['Pcd'].points))
    for j in range(len(avlandmarks)):
        
        aux = np.asarray([landmarksavpoints[j] for i in range(NPoints)])
        dist = np.linalg.norm(aux - avpoints, axis = 1)
        closest_points_av[j] = np.argmin(dist).astype(np.int64)
    
    pointsAroundList = []
    closestIndex = []
    
    for i in range(len(closest_points_av)):
        
        closest_point = avpoints[closest_points_av[i], :]
        
        for k in range(len(avpoints)):
                
            if np.linalg.norm(closest_point - avpoints[k, :]) <= radius:
                    
                pointsAroundList.append(k)
                closestIndex.append(i)
                
    
    # Make rigid registration between landmarks and closest points and apply to av shape
    AvClosestPoints = avpoints[closest_points_av]
    AvPcdClosestPoints = o3d.geometry.PointCloud(o3d.cpu.pybind.utility.Vector3dVector(AvClosestPoints))
    # Perform rigid registration
    # This makes the reconstruction match its closest points to the landmarks, besides minimizing the distance
    RigidReg = o3d.pipelines.registration.registration_icp(AvPcdClosestPoints, 
                                                           landmarkavpcd, 
                                                           2 * np.max(landmarkavpcd.compute_nearest_neighbor_distance()))
    # Apply the transformation to current geometry
    AVShape['Pcd'].transform(RigidReg.transformation)
    avpoints = copy.deepcopy(np.asarray(AVShape['Pcd'].points))
    
    # Visualize the choosen points (in blue), the landmarks (in green) 
    # and remaining model points (in red), all in the same point cloud
    colour_target = [ blue if i in closest_points_av else red for i in range(NPoints) ]
    AVShape['Pcd'].colors = o3d.utility.Vector3dVector(colour_target)
    
    if plots:
        o3d.visualization.draw_geometries([AVShape['Pcd'],
                                           landmarkavpcd.paint_uniform_color(green)])
    
    # Print the difference between the choosen points and the av landmarks 
    # These should be the lowest possible
    # dif = np.linalg.norm(avpoints[closest_points_av] - landmarksavpoints, axis = 1)
    # print('Difference between closest points in av shape and reference landmarks (mm):')
    # print(dif)
    
    return closest_points_av, pointsAroundList, closestIndex

def ReconstructFromLandmark(StatModel: dict, landmarkpcd: o3d.geometry.PointCloud, closest_points: np.array, 
                            T_local: np.array, folderpath: str, boneopt: str, plots: bool, tipo: str,
                            PC: int, opt:str, reg:str, c:float, TNC:bool, pointsList, closestIndex, reference_shape_path):
    
    
    """
    Perform bone reconstruction from sparse landmark data
    using a Statistical Shape Model and optimization strategy.

    Parameters
    ----------
    StatModel : dict
        Statistical Shape Model data.
    landmarkpcd : open3d.geometry.PointCloud
        Landmark point cloud.
    closest_points : np.ndarray
        Closest surface points corresponding to landmarks.
    T_local : np.ndarray
        Transformation matrix to local reference frame.
    folderpath : str
        Output directory for results.
    boneopt : str
        Bone identifier.
    plots : bool
        Enable plotting.
    tipo : str
        Reconstruction mode or type.
    PC : int
        Number of principal components used.
    opt : str
        Optimization method identifier.
    reg : str
        Registration method identifier.
    c : float
        Regularization or weighting coefficient.
    TNC : bool
        Flag indicating constrained optimization at the end.
    pointsList, closestIndex :
        Auxiliary reconstruction data.
    reference_shape_path : str
        Path to reference geometry.

    Returns
    -------
    OptGeom: dict
        Dictionary containing data about the reconstructed geometry
    KvalOpt: 
        Optimal principal components for the reconstruction of the geometry
    tOpt:
        Time taken for the reconstruction
    calls:
        Number of call to the objective function
    fit: 
        Objective function value
    """
    
    # Define objective/function for the GA (Very important step)
    def FitnessFuncSSMReconstruct_GA(StatModel, closest_points_, pcd, T_final, reg, c):
        
        
        def fitness_function(kvals, kvals_idx):
            # Build current geometry 
            CurGeom = SSMReconstruction(StatModel, kvals, None, originalsize = True)
            
            #TODO this next step is not advisable to be in fitness function. 
            # It will just make it more computational expensive
            # In the future, pass all the SSM code to local coordinates and this is not needed
            # For now it is necessary, as the eigenvalues and eigen vectors are defined
            # using the previously defined reference frame generated by the inertia alignment
            
            # Pass current geometry to local coordinate frame (the mesh is not needed)
            # Point Cloud
            CurGeom['Pcd'].transform(T_final)

            # Make point cloud with closest points
            closest_points_pcd = o3d.geometry.PointCloud(o3d.cpu.pybind.utility.Vector3dVector(np.asarray(CurGeom['Pcd'].points)[closest_points_]))
            # Perform rigid registration
            # This makes the reconstruction match its closest points to the landmarks, besides minimizing the distance
            # This is done even if both geometry and landmarks are in the same local reference frame, as the transformation
            # Might not be perfect for all cases
            #TODO in the future it is advisable to take this rigid registration out, as it induces some random variability for the solution
            # That is, running twice the 2 lines below come different RigidReg variables
            RigidReg = o3d.pipelines.registration.registration_icp(closest_points_pcd, 
                                                                   pcd, 
                                                                   2 * np.max(pcd.compute_nearest_neighbor_distance()))
            # Apply the transformation to the closest points set
            closest_points_pcd.transform(RigidReg.transformation)
            
            # Evaluation for current geometry 
            if reg == 'Marques':
                
                regularization = np.sum(np.exp(np.square(kvals)) - np.ones(len(kvals)))
                
            elif reg == 'Sobral':
                eigValArr = StatModel['SSM']['EigVal'][:PC]
                
                regularization = np.sum(c*(1/(eigValArr/eigValArr[0]))*np.square(kvals))
            
            f = - (np.sum(np.square(np.asarray(closest_points_pcd.points) - np.asarray(pcd.points))) + regularization)
            
            return f

        return fitness_function
    
    def FitnessFuncSSMReconstruct_NGO(StatModel, closest_points_, pcd, T_final, reg, c):
        
        
        def fitness_function(kvals):
            it = kvals.items()

            l = list(it)
            kvals = np.zeros(len(l))
            for i in range(len(l)):
                
                kvals[i] = l[i][1]
                
            # Build current geometry 
            CurGeom = SSMReconstruction(StatModel, kvals, None, originalsize = True)
            
            #TODO this next step is not advisable to be in fitness function. 
            # It will just make it more computational expensive
            # In the future, pass all the SSM code to local coordinates and this is not needed
            # For now it is necessary, as the eigenvalues and eigen vectors are defined
            # using the previously defined reference frame generated by the inertia alignment
            
            # Pass current geometry to local coordinate frame (the mesh is not needed)
            # Point Cloud
            CurGeom['Pcd'].transform(T_final)

            # Make point cloud with closest points
            closest_points_pcd = o3d.geometry.PointCloud(o3d.cpu.pybind.utility.Vector3dVector(np.asarray(CurGeom['Pcd'].points)[closest_points_]))
            # Perform rigid registration
            # This makes the reconstruction match its closest points to the landmarks, besides minimizing the distance
            # This is done even if both geometry and landmarks are in the same local reference frame, as the transformation
            # Might not be perfect for all cases
            #TODO in the future it is advisable to take this rigid registration out, as it induces some random variability for the solution
            # That is, running twice the 2 lines below come different RigidReg variables
            RigidReg = o3d.pipelines.registration.registration_icp(closest_points_pcd, 
                                                                   pcd, 
                                                                   2 * np.max(pcd.compute_nearest_neighbor_distance()))
            # Apply the transformation to the closest points set
            closest_points_pcd.transform(RigidReg.transformation)
            
            # Evaluation for current geometry 
            # The fitness function of PyGad is written as a maximization instead of  a minimization
            # this implies that the fitness must come with a negative sign
            #TODO include weight in the displacement values, for each 'landmark+coord', computed using the 
            # test error of the respective regression model (more error = less penalization of the sum of squares of the displacement)
            # vide mail Quental 08-Aug-2023
            
            if reg == 'Marques':
                
                regularization = np.sum(np.exp(np.square(kvals)) - np.ones(len(kvals)))
                
            elif reg == 'Sobral':
                eigValArr = StatModel['SSM']['EigVal'][:PC]
                
                regularization = c*np.sum((1/(eigValArr/eigValArr[0]))*np.square(kvals))
            
            f = - (np.mean(np.sqrt(np.sum(np.square(np.asarray(closest_points_pcd.points) - np.asarray(pcd.points)), axis=1))) + regularization)
            
            return f

        return fitness_function
    
    # Define objective/function for the TNC (Very important step)
    def FitnessFuncSSMReconstruct_TNC(StatModel, closest_points_, pcd, T_final, reg, c):
        
        
        def fitness_function(kvals):
            
            # Build current geometry 
            CurGeom = SSMReconstruction(StatModel, kvals, None, originalsize = True)
            
            #TODO this next step is not advisable to be in fitness function. 
            # It will just make it more computational expensive
            # In the future, pass all the SSM code to local coordinates and this is not needed
            # For now it is necessary, as the eigenvalues and eigen vectors are defined
            # using the previously defined reference frame generated by the inertia alignment
            
            # Pass current geometry to local coordinate frame (the mesh is not needed)
            # Point Cloud
            CurGeom['Pcd'].transform(T_final)
 
            # Make point cloud with closest points
            #closest_points_pcd = o3d.geometry.PointCloud(o3d.cpu.pybind.utility.Vector3dVector(np.asarray(CurGeom['Pcd'].points)[closest_points_]))
            closest_points_pcd = o3d.geometry.PointCloud(o3d.cpu.pybind.utility.Vector3dVector(np.asarray(CurGeom['Pcd'].points)[pointsList]))
            # Perform rigid registration
            # This makes the reconstruction match its closest points to the landmarks, besides minimizing the distance
            # This is done even if both geometry and landmarks are in the same local reference frame, as the transformation
            # Might not be perfect for all cases
            #TODO in the future it is advisable to take this rigid registration out, as it induces some random variability for the solution
            # That is, running twice the 2 lines below come different RigidReg variables
            RigidReg = o3d.pipelines.registration.registration_icp(closest_points_pcd, 
                                                                   pcd, 
                                                                   2 * np.max(pcd.compute_nearest_neighbor_distance()))
            # Apply the transformation to the closest points set
            closest_points_pcd.transform(RigidReg.transformation)
            
            if reg == 'Marques':
                
                regularization = np.sum(np.exp(np.square(kvals)) - np.ones(len(kvals)))
                
            elif reg == 'Sobral':
                eigValArr = StatModel['SSM']['EigVal'][:PC]
                
                regularization = c*np.sum((1/(eigValArr/eigValArr[0]))*np.square(kvals))
            
            # Evaluation for current geometry 
            #TODO include weight in the displacement values, for each 'landmark+coord', computed using the 
            # test error of the respective regression model (more error = less penalization of the sum of squares of the displacement)
            # vide mail Quental 08-Aug-2023

            f = np.mean(np.sqrt(np.sum(np.square(np.asarray(closest_points_pcd.points) - np.asarray(pcd.points)[closestIndex]), axis=1))) + regularization
            
            return f

        return fitness_function
    
    # Define objective/function for the TNC (Very important step)
    def FitnessFuncSSMReconstruct_Min(StatModel, closest_points_, pcd, T_final):
        
        def fitness_function(kvals):
            
            # Build current geometry 
            CurGeom = SSMReconstruction(StatModel, kvals, None, originalsize = True)
            
            #TODO this next step is not advisable to be in fitness function. 
            # It will just make it more computational expensive
            # In the future, pass all the SSM code to local coordinates and this is not needed
            # For now it is necessary, as the eigenvalues and eigen vectors are defined
            # using the previously defined reference frame generated by the inertia alignment
            
            # Pass current geometry to local coordinate frame (the mesh is not needed)
            # Point Cloud
            CurGeom['Pcd'].transform(T_final)
 
            # Make point cloud with closest points
            closest_points_pcd = o3d.geometry.PointCloud(o3d.cpu.pybind.utility.Vector3dVector(np.asarray(CurGeom['Pcd'].points)[closest_points_]))
            # Perform rigid registration
            # This makes the reconstruction match its closest points to the landmarks, besides minimizing the distance
            # This is done even if both geometry and landmarks are in the same local reference frame, as the transformation
            # Might not be perfect for all cases
            #TODO in the future it is advisable to take this rigid registration out, as it induces some random variability for the solution
            # That is, running twice the 2 lines below come different RigidReg variables
            RigidReg = o3d.pipelines.registration.registration_icp(closest_points_pcd, 
                                                                   pcd, 
                                                                   2 * np.max(pcd.compute_nearest_neighbor_distance()))
            # Apply the transformation to the closest points set
            closest_points_pcd.transform(RigidReg.transformation)
            
            # Evaluation for current geometry 
            #TODO include weight in the displacement values, for each 'landmark+coord', computed using the 
            # test error of the respective regression model (more error = less penalization of the sum of squares of the displacement)
            # vide mail Quental 08-Aug-2023
            
            f = np.sum(np.square(np.asarray(closest_points_pcd.points) - np.asarray(pcd.points))) + np.sum(np.exp(np.square(kvals)) - np.ones(len(kvals)))
            
            return f

        return fitness_function
    
    ### For the reconstruction, we must set the reference and fixed closest points to its bony landmarks
    ### in the same reference frame as the predicted bony landmarks
    ### The predicted bony landamrks are already in the local reference frame computed by the bony landmarks
    ### given as input. As such, we will put the reference in its local reference frame as well
    ### For this we must apply the inverse transformation made in the SSM code (inertia axes)
    ### And then apply the local transformation matrix
    
    
    ## Undo general transformation matrix of the inertia axes
    # This will undo the rotation and translation of the inertia alignment, but not the scaling
    # For every new geometry = ref + mu + k_i*sqrt(lambda_i)*phi_i
    # (scaling factor of MVShape is nullified setting originalsize = True in SSMReconstruction)
    
    ## One way to do it is like this, but with previous knowledge of the SSM Code
    T_inv = np.eye(4)
    T_inv[:3, :3] = StatModel['MVShape']['R']
    T_inv[:3, 3] = StatModel['MVShape']['t']
                                                     
    # Compute new transformation matrix from reference original coordinate frame to local coordinate frame
    # It should be the bony landmark file for the mean geometry (Ref + mu), but there is no DICOM to match
    # Since we do not have it, we should use the bony landmarks of the mean shape, but they are in the inertial referential frame
    # So we use the reference bony landmark file local reference frame transformation
    if reference_shape_path != None:
        _, T_new = GetLocalLandmark(reference_shape_path, boneopt)
        T_final = T_new
    else:
        # NEW: CQ Implementation
        cwd = os.getcwd()
        _, T_new = GetLocalLandmark(cwd + "\\ReferenceShape\\Ref_Skin_Landmarks.txt", boneopt)
        # The final transformation matrix from inertia frame to local frame 
        T_final = np.eye(4)
        # R_{local/inertia} = R_{local/original} * R_{original/inertia}
        T_final[:3, :3] = np.dot(T_new[:3, :3], T_inv[:3, :3])
        # T_{local/inertia} = T_{local/original} + T_{original/inertia}
        # Must put all translation vectors in the same reference frame (local, as is the desired one)
        T_final[:3, 3] = T_new[:3, 3] + np.dot(T_new[:3, :3], T_inv[:3, 3])
        
        """
        T_new = T_local
        # The final transformation matrix from inertia frame to local frame 
        T_final = np.eye(4)
        # R_{local/inertia} = R_{local/original} * R_{original/inertia}
        T_final[:3, :3] = np.dot(T_new[:3, :3], T_inv[:3, :3])
        # T_{local/inertia} = T_{local/original} + T_{original/inertia}
        # Must put all translation vectors in the same reference frame (local, as is the desired one)
        T_final[:3, 3] = T_new[:3, 3] + np.dot(T_new[:3, :3], T_inv[:3, 3])
        """
    

    # Define function to change initial population according to NEigVal
    # So that population can consider the best solutions found with previous PCs
    # Adapted from António's code. [I think it is ugly, but makes sense and it works, so not going to waste time here...]
    def initial_pop(previous_solution, Npopulation, num_genes):
        
        # Values between -3 and 3 that will be added to the new PC to consider
        # In total, 13 solutions, made considering the PC-1 best one, are added to the population
        # Every time a PC is added
        new_vals = np.arange(-3, 4, 1)
        
        # New population 
        pop = []
        
        # Add the previous solutions with the new PC with a given value
        for j in range(len(new_vals)):
            prev_sol = list(copy.deepcopy(previous_solution))
            current_eigval = float(new_vals[j])
            prev_sol.append(current_eigval)
            pop.append(prev_sol)
        
        # Complete the population with random solutions (normally distributed)
        # Basically, by computing random points with 'k' samples taken from normal distribution
        # we are taking guesses at minimums of the objective function...
        for i in range(Npopulation - len(new_vals)):
            new_sol = []
            for z in range(num_genes): 
                new_sol.append(np.random.normal())
            pop.append(new_sol)
            
        return pop
    
    # The optimal PCs can be computed using one of two approaches:
    #   - 'IncrementalSol' - it begins by solving the problem with a single PC and
    #                       it increases the number of PCs iteratively until all
    #                       requested PCs are used
    #   - 'SingleSol'      - it identifies all PCs in a single optimization problem
    if tipo == 'slow':
        
        sOpt = time.perf_counter()
        for NEigVal in range(1, PC + 1):
            
            # Defines the parameters for the optimization GA algorithm
            # Highly mutable, modifiable and context-dependent. The best way to choose these parameters
            # Is via trial-and-error. A lot of papers exist about the selection of GA parameters, but no use
            # Depends on the problem...
            
            num_genes = NEigVal
            num_generations = int(150 * num_genes)
            num_parents_mating = int(round(0.1 * num_generations))
            sol_per_pop = int(round(0.2 * num_generations))
            elite_num_genes = int(round(0.1 * sol_per_pop))
            # The first value is the mutation rate for the low-quality solutions.
            # The second value is the mutation rate for the high-quality solutions.
            # mutation_num_genes = [int(max(1,round(0.3 * num_genes))), int(round(0.1 * num_genes))]
            mutation_probability = [0.3, 0.1]
            gene_bounds = [{'low': -3, 'high': 3} for gene in range(num_genes)]
            # Based on articles, it is the best 
            parent_selection_type = 'rws'
            
            if num_genes == 1:
                
                # Even if kvals are ranged between -3 and 3, the fitness function + the distribution of shapes
                # (that we assume is normally distributed) will automatically exclude high absolute k vals
                # So there is no problem in defining like -2, 2, -3, 3 in the initial population
                # initial_population = [[np.random.normal() for gene in range(num_genes)] for sol in range(sol_per_pop)]
                initial_population = [[el for gene in range(num_genes)] for el in np.arange(-3, 4, 1)] + [[np.random.normal()] for el in range(sol_per_pop - len(np.arange(-3, 4, 1)))]
                KvalOpt = None
                
            else:
                
                # Population will take into account 'N-1' best solution, with the new NEigVal taking 
                # values between 3 and 3, with interval of 0.5.
                # With keep_elitism = True, this ensures that increasing the number of PCs does not produce 
                # a worst solution
                initial_population = initial_pop(KvalOpt, sol_per_pop, num_genes)
                
            # Creates the optimization problem
            ga_instance = pygad.GA(num_generations = num_generations,
                                   num_parents_mating = num_parents_mating,
                                   fitness_func = FitnessFuncSSMReconstruct_GA(StatModel, closest_points, landmarkpcd, T_final, reg, c),
                                   initial_population = initial_population,
                                   gene_space = gene_bounds,
                                   parent_selection_type = parent_selection_type,
                                   keep_elitism = elite_num_genes,
                                   mutation_type = 'adaptive',
                                   # mutation_by_replacement = True,
                                   # random_mutation_min_val = lb,
                                   # random_mutation_max_val = ub,
                                   # mutation_num_genes = mutation_num_genes,
                                   mutation_probability = mutation_probability,
                                   parallel_processing = ['thread', 24])
                
            # Runs the optmization
            ga_instance.run()
            
            # Extracts the best solution
            KvalOpt_GA, fit, _ = ga_instance.best_solution()
            
            KvalOpt_GA = [-3 if gene < -3 else 3 if gene > 3 else gene for gene in KvalOpt_GA]
          
            print('GA:')
            print(f'    Sol: {KvalOpt_GA}')
            print(f'    Fitness Value: {round(fit,4)}')
            
            # Make one iteration of the TNC with KvalOpt_GA as initial guess
            # To assert we get to a minimum, at least locally
            result = minimize(FitnessFuncSSMReconstruct_TNC(StatModel, closest_points, landmarkpcd, T_final, reg, c), 
                              x0 = KvalOpt_GA, 
                              method = 'TNC', 
                              bounds = tuple((-3,3) for i in range(NEigVal)),
                              tol = 1e-8,
                              options = {'maxfun': 1e5}) 
            
            if result.success:
                KvalOpt = copy.deepcopy(result.x)
                
                KvalOpt = [-3 if gene < -3 else 3 if gene > 3 else gene for gene in KvalOpt]
                
                print('TNC:')
                print(f'    Sol: {KvalOpt}')
                print(f'    Fitness Value: {round(result.fun,4)}')
    
            else:
                KvalOpt =  copy.deepcopy(KvalOpt_GA)
        
            # Register time of optimization procedure
            tOpt = time.perf_counter() - sOpt
            
    else:
        
        sOpt = time.perf_counter()
        
        num_genes = PC
        num_generations = int(50 * num_genes)
        num_parents_mating = int(round(0.1 * num_generations))
        sol_per_pop = int(round(0.2 * num_generations))
        elite_num_genes = int(round(0.1 * sol_per_pop))
        # The first value is the mutation rate for the low-quality solutions.
        # The second value is the mutation rate for the high-quality solutions.
        # mutation_num_genes = [int(max(1,round(0.3 * num_genes))), int(round(0.1 * num_genes))]
        mutation_probability = [0.3, 0.1]
        gene_bounds = [{'low': -3, 'high': 3} for gene in range(num_genes)]
        # Based on articles, it is the best 
        parent_selection_type = 'rws'
        
        initial_population = [[np.random.normal() for gene in range(num_genes)] for el in range(sol_per_pop)]
        

        if opt == 'genetic':
            # Creates the optimization problem
            ga_instance = pygad.GA(num_generations = num_generations,
                                   num_parents_mating = num_parents_mating,
                                   fitness_func = FitnessFuncSSMReconstruct_GA(StatModel, closest_points, landmarkpcd, T_final, reg, c),
                                   initial_population = initial_population,
                                   gene_space = gene_bounds,
                                   parent_selection_type = parent_selection_type,
                                   keep_elitism = elite_num_genes,
                                   mutation_type = 'adaptive',
                                   # mutation_by_replacement = True,
                                   # random_mutation_min_val = lb,
                                   # random_mutation_max_val = ub,
                                   # mutation_num_genes = mutation_num_genes,
                                   mutation_probability = mutation_probability,
                                   parallel_processing = ['thread', 24])
                
            # Runs the optmization
            ga_instance.run()
            
            # Extracts the best solution
            KvalOpt_GA, fit, _ = ga_instance.best_solution()
            
            KvalOpt_GA = [-3 if gene < -3 else 3 if gene > 3 else gene for gene in KvalOpt_GA]
          
            print('GA:')
            print(f'    Sol: {KvalOpt_GA}')
            print(f'    Fitness Value: {round(fit,4)}')
            
            calls = num_generations*sol_per_pop*2
            
        elif opt == 'Bayes':
            
            search_space = [skopt.space.space.Real(-3,3) for i in range(num_genes)]
            
            res = skopt.gp_minimize(FitnessFuncSSMReconstruct_Min(StatModel, closest_points, landmarkpcd, T_final),            # the function to minimize
                  search_space,      # the bounds on each dimension of x
                  x0=[0. for i in range(num_genes)],            # the starting point
                  acq_func="gp_hedge",     # the acquisition function (optional)
                  initial_point_generator='lhs',
                  n_calls=25*num_genes,         # the number of evaluations of f including at x0
                  n_random_starts=12*num_genes)
        
            KvalOpt_GA, fit, calls = res.x, res.fun, 25*num_genes
            
            print('Bayesian Optimization:')
            print(f'    Sol: {KvalOpt_GA}')
            print(f'    Fitness Value: {round(fit,4)}')
            
        elif opt == 'Pattern':
            
            search_space = {}
            for i in range(num_genes):
                
                search_space[f'PC{i+1}'] = np.arange(-3, 3, 0.01)
                
            early_stop = {'n_iter_no_change': int(10000*(PC**2)/6)}
            if PC > 9:
                
                positions = 8
                pattern = 0.5
                
            else:
                
                positions = 8
                pattern = 0.5
            
            initialize = {'random': 1000*(PC**2)}
            
            opt = PatternSearch(search_space, n_positions=positions, 
                                pattern_size = pattern)
            
            opt.search(FitnessFuncSSMReconstruct_NGO(StatModel, closest_points, landmarkpcd, T_final, reg, c),
                       n_iter= 10000*(PC**2),early_stopping=early_stop)
            
            fit = opt.search_data.tail(1)['score']
            KvalOpt_GA = opt.search_data.tail(1).to_numpy()[0][1:]
            calls = len(opt.iter_times)
            print('Pattern search:')
            print(f'    Sol: {KvalOpt_GA}')
            print(f'    Fitness Value: {round(fit,4)}')
            
        elif opt == 'DIRECT':
            
            search_space = {}
            for i in range(num_genes):
                
                search_space[f'PC{i+1}'] = np.arange(-3, 3, 0.001)
                
            #early_stop = {'n_iter_no_change': int(500*(PC**2)/3)}
            
            initialize = {'random': 30*(PC**2)}
            
            opt = DirectAlgorithm(search_space, initialize=initialize)
            
            opt.search(FitnessFuncSSMReconstruct_NGO(StatModel, closest_points, landmarkpcd, T_final, reg, c),
                       n_iter= 60*(PC**2))
            
            fit = opt.search_data.tail(1)['score']
            KvalOpt_GA = opt.search_data.tail(1).to_numpy()[0][1:]
            
            print('Direct Algorithm:')
            print(f'    Sol: {KvalOpt_GA}')
            print(f'    Fitness Value: {round(fit,4)}')
            
        elif opt == 'RandomHill':
            
            search_space = {}
            for i in range(num_genes):
                
                search_space[f'PC{i+1}'] = np.arange(-3, 3, 0.001)
                
            #early_stop = {'n_iter_no_change': int(500*(PC**2)/3)}
            
            opt = RandomRestartHillClimbingOptimizer(search_space, n_neighbours=10, n_iter_restart=20)
            
            opt.search(FitnessFuncSSMReconstruct_NGO(StatModel, closest_points, landmarkpcd, T_final, reg, c),
                       n_iter= 1000*(PC**2))
            
            fit = opt.search_data.tail(1)['score']
            KvalOpt_GA = opt.search_data.tail(1).to_numpy()[0][1:]
            calls = len(opt.iter_times)
            
            print('Random Hill:')
            print(f'    Sol: {KvalOpt_GA}')
            print(f'    Fitness Value: {round(fit,4)}')
            
        elif opt== 'RandomAna':
            
            search_space = {}
            for i in range(num_genes):
                
                search_space[f'PC{i+1}'] = np.arange(-3, 3, 0.001)
                
            #early_stop = {'n_iter_no_change': int(500*(PC**2)/3)}
            
            initialize = {'random': 100*(PC**2)}
            
            opt = RandomAnnealingOptimizer(search_space, n_neighbours=15, initialize=initialize, start_temp=25)
            
            opt.search(FitnessFuncSSMReconstruct_NGO(StatModel, closest_points, landmarkpcd, T_final, reg, c),
                       n_iter= 1000*(PC**2))
            
            fit = opt.search_data.tail(1)['score']
            KvalOpt_GA = opt.search_data.tail(1).to_numpy()[0][1:]
            calls = len(opt.iter_times)
            
            print('Random Ana:')
            print(f'    Sol: {KvalOpt_GA}')
            print(f'    Fitness Value: {round(fit,4)}')
            
        elif opt == 'Bayes2':
            
            search_space = {}
            for i in range(num_genes):
                
                search_space[f'PC{i+1}'] = np.arange(-3, 3, 0.001)
                
            #early_stop = {'n_iter_no_change': int(500*(PC**2)/3)}
            
            opt = BayesianOptimizer(search_space, max_sample_size=1000000, rand_rest_p=0.05)
            
            opt.search(FitnessFuncSSMReconstruct_NGO(StatModel, closest_points, landmarkpcd, T_final, reg, c),
                       n_iter= 100*(PC**2))
            
            fit = opt.search_data.tail(1)['score']
            KvalOpt_GA = opt.search_data.tail(1).to_numpy()[0][1:]
            
            print('Bayesian Optimizer 2:')
            print(f'    Sol: {KvalOpt_GA}')
            print(f'    Fitness Value: {round(fit,4)}')
            
        elif opt == 'Lip':
            
            search_space = {}
            for i in range(num_genes):
                
                search_space[f'PC{i+1}'] = np.arange(-3, 3, 0.001)
                
            #early_stop = {'n_iter_no_change': int(500*(PC**2)/3)}
            
            initialize = {'random': 10*(PC)}
            
            opt = LipschitzOptimizer(search_space, max_sample_size=1000000, initialize=initialize)
            
            opt.search(FitnessFuncSSMReconstruct_NGO(StatModel, closest_points, landmarkpcd, T_final, reg, c),
                       n_iter= 20*(PC))
            
            fit = opt.search_data.tail(1)['score']
            KvalOpt_GA = opt.search_data.tail(1).to_numpy()[0][1:]
            
            print('Lipschitz Optimizer:')
            print(f'    Sol: {KvalOpt_GA}')
            print(f'    Fitness Value: {round(fit,4)}')
            
        elif opt == 'Random':
            
            search_space = {}
            for i in range(num_genes):
                
                search_space[f'PC{i+1}'] = np.arange(-3, 3, 0.001)
                
            #early_stop = {'n_iter_no_change': int(500*(PC**2)/3)}
            
            opt = RandomSearchOptimizer(search_space)
            
            opt.search(FitnessFuncSSMReconstruct_NGO(StatModel, closest_points, landmarkpcd, T_final, reg, c),
                       n_iter= 1000*(PC**2))
            
            fit = opt.search_data.tail(1)['score']
            KvalOpt_GA = opt.search_data.tail(1).to_numpy()[0][1:]
            calls = len(opt.iter_times)
            
            print('Random Search:')
            print(f'    Sol: {KvalOpt_GA}')
            print(f'    Fitness Value: {round(fit,4)}')
            
        elif opt == 'Parallel':
            
            search_space = {}
            for i in range(num_genes):
                
                search_space[f'PC{i+1}'] = np.arange(-3, 3, 0.001)
                
            #early_stop = {'n_iter_no_change': int(500*(PC**2)/3)}
            initialize = {'random': 100*(PC), 'grid': 8}
            
            opt = ParallelTemperingOptimizer(search_space, rand_rest_p=0.01, initialize=initialize)
            
            opt.search(FitnessFuncSSMReconstruct_NGO(StatModel, closest_points, landmarkpcd, T_final, reg, c),
                       n_iter= 1000*(PC**2))
            
            fit = opt.search_data.tail(1)['score']
            KvalOpt_GA = opt.search_data.tail(1).to_numpy()[0][1:]
            calls = len(opt.iter_times)
            
            print('Parallel Tempering Optimization:')
            print(f'    Sol: {KvalOpt_GA}')
            print(f'    Fitness Value: {round(fit,4)}')
            
        elif opt == 'Grid':
            
            search_space = {}
            for i in range(num_genes):
                
                search_space[f'PC{i+1}'] = np.arange(-3, 3, 0.01)
                
            #early_stop = {'n_iter_no_change': int(500*(PC**2)/3)}
            
            opt = GridSearchOptimizer(search_space, rand_rest_p=0.01)
            
            opt.search(FitnessFuncSSMReconstruct_NGO(StatModel, closest_points, landmarkpcd, T_final, reg, c),
                       n_iter= 1000*(PC**2))
            
            fit = opt.search_data.tail(1)['score']
            KvalOpt_GA = opt.search_data.tail(1).to_numpy()[0][1:]
            
            print('Grid Search:')
            print(f'    Sol: {KvalOpt_GA}')
            print(f'    Fitness Value: {round(fit,4)}')
            
        elif opt == 'DIRECT2':
            
            search_space = [(-3,3) for i in range(num_genes)]
            
            if PC <= 5:
                res = direct(FitnessFuncSSMReconstruct_TNC(StatModel, closest_points, landmarkpcd, T_final, reg, c),            # the function to minimize
                  search_space, vol_tol=10**(-5*PC), maxiter=500*PC, maxfun=10000 + 1000*PC)
            else:   
                res = direct(FitnessFuncSSMReconstruct_TNC(StatModel, closest_points, landmarkpcd, T_final, reg, c),            # the function to minimize
                  search_space, vol_tol=10**(-5*PC), maxiter= 1500 + 700*PC, maxfun=70000 + 8000*PC)
        
            KvalOpt_GA, fit, calls = copy.deepcopy(res.x), res.fun, res.nfev
            
            print('Direct Algorithm 2:')
            print(f'    Sol: {KvalOpt_GA}')
            print(f'    Fitness Value: {round(fit,4)}')
            #print(res.message)       
            
        # Make one iteration of the TNC with KvalOpt_GA as initial guess
        # To assert we get to a minimum, at least locally

        if TNC:
            result = minimize(FitnessFuncSSMReconstruct_TNC(StatModel, closest_points, landmarkpcd, T_final, reg, c), 
                              x0 = KvalOpt_GA, 
                              method = 'TNC', 
                              bounds = tuple((-3,3) for i in range(PC)),
                              tol = 1e-8,
                              options = {'maxfun': 1e5})
            calls = calls + result.nfev
            
            if result.success:
                KvalOpt = copy.deepcopy(result.x)
                
                KvalOpt = [-3 if gene < -3 else 3 if gene > 3 else gene for gene in KvalOpt]
                
                print('TNC:')
                print(f'    Sol: {KvalOpt}')
                print(f'    Fitness Value: {round(result.fun,4)}')
    
            else:
                KvalOpt =  copy.deepcopy(KvalOpt_GA)
                
        else:
            
            
                KvalOpt =  copy.deepcopy(KvalOpt_GA)
        # Register time of optimization procedure
        tOpt = time.perf_counter() - sOpt

    # Compute reconstructed geometry
    OptGeom = SSMReconstruction(StatModel, KvalOpt, None, originalsize = True)
    
    # Pass current geometry to local coordinate frame (both point cloud and mesh)
    # Point Cloud
    
    # Undo transformation matrix of the inertia axes
    OptGeom['Pcd'].transform(T_final)
    # Mesh
    OptGeom['Mesh'].transform(T_final)
    OptGeom['Mesh'].compute_triangle_normals()
    OptGeom['Mesh'].compute_vertex_normals()
    
    if plots:
        # Check best solution and random landmarks
        o3d.visualization.draw_geometries([OptGeom['Pcd'].paint_uniform_color(red),
                                           landmarkpcd.paint_uniform_color(green)])

    # Get closest points choosen and plot geometries again
    # Define colors for the reconstructed model + closest points
    OptGeomCoords = np.asarray(OptGeom['Pcd'].points)

    # Make point cloud with closest points
    closest_points_pcd = o3d.geometry.PointCloud(o3d.cpu.pybind.utility.Vector3dVector(OptGeomCoords[closest_points]))
    
    # Perform rigid registration
    # This makes the reconstruction match its closest points to the landmarks, besides minimizing the distance
    RigidReg = o3d.pipelines.registration.registration_icp(closest_points_pcd, 
                                                           landmarkpcd, 
                                                           2 * np.max(landmarkpcd.compute_nearest_neighbor_distance()))
    # Apply the transformation to current geometry
    OptGeom['Pcd'].transform(RigidReg.transformation)
    
    # Get closest points choosen and plot geometries again
    OptGeomCoords = np.asarray(OptGeom['Pcd'].points)
    
    colour_opt = [ blue if i in closest_points else red for i in range(len(OptGeomCoords)) ]
    OptGeom['Pcd'].colors = o3d.utility.Vector3dVector(colour_opt)
    
    '''
    if plots:
        
        o3d.visualization.draw_geometries([OptGeom['Pcd'],
                                           landmarkpcd.paint_uniform_color(green)])
    '''
    
    if plots:
        OptGeomCoords = np.asarray(OptGeom['Pcd'].points)
        closest_points_pcd = o3d.geometry.PointCloud(o3d.cpu.pybind.utility.Vector3dVector(OptGeomCoords[closest_points]))
        
        o3d.visualization.draw_geometries([OptGeom['Pcd'].paint_uniform_color(red),
                                          landmarkpcd.paint_uniform_color(green),
                                          closest_points_pcd.paint_uniform_color(blue)])


    # Print difference
    landmarkdif = OptGeomCoords[closest_points] - np.asarray(landmarkpcd.points)
    landmarkdifavg = np.mean(np.linalg.norm(landmarkdif, axis = 1))
    print(f'\nAverage distance between reconstruction closest points and real bony landmarks : {round(landmarkdifavg,2)} mm')
    
    # Updates the mesh
    OptGeom['Mesh'].vertices = o3d.cpu.pybind.utility.Vector3dVector(copy.deepcopy(OptGeomCoords))
    OptGeom['Mesh'].compute_triangle_normals()
    OptGeom['Mesh'].compute_vertex_normals()
    
    if plots:
        o3d.visualization.draw_geometries([OptGeom['Mesh']])
    
    # Save reconstruction in local reference frame 
    PolyDataOpt = MeshDataToPolyData(OptGeom)
    PolyDataOpt = PolyDataOpt.smooth(n_iter = 100)
    PolyDataOpt.save(folderpath + '_local.stl')
    
    # Save the info for the reconstructed model
    OptData = {'Mesh' : { 'Vertices' : copy.deepcopy(np.asarray(OptGeom['Mesh'].vertices)),
                          'Triangles' : copy.deepcopy(np.asarray(OptGeom['Mesh'].triangles))
                        },
               'Solution': KvalOpt,
               'LandmarkDif' : landmarkdif,
               'LandmarkCor' : closest_points
               }
    
    with open(folderpath + '.pickle', "wb") as file:
        pickle.dump(OptData, file, pickle.HIGHEST_PROTOCOL)
    
    return OptGeom, KvalOpt, tOpt, calls, fit

def modelReconstruction(PCs, NPc, StatModel, boneopt, mode, reference_shape_path):
    
    
    """
    High-level wrapper function for model-based bone reconstruction.

    Parameters
    ----------
    PCs : np.ndarray
        Principal component coefficients.
    NPc : int
        Number of principal components.
    StatModel : dict
        Statistical Shape Model.
    boneopt : str
        Bone identifier.
    mode : str
        Reconstruction mode.
    reference_shape_path : str
        Reference geometry path.

    Returns
    -------
    OptGeom: dict
        Dictionary containing data about the reconstructed geometry
    KvalOpt: 
        Optimal principal components for the reconstruction of the geometry
    NPc: 
        Number of principal components
    """

    
    T_inv = np.eye(4)
    T_inv[:3, :3] = StatModel['MVShape']['R']
    T_inv[:3, 3] = StatModel['MVShape']['t']
    
    _, T_new = GetLocalLandmark(reference_shape_path, boneopt)
    
    # The final transformation matrix from inertia frame to local frame 
    T_final = np.eye(4)
    # R_{local/inertia} = R_{local/original} * R_{original/inertia}
    T_final[:3, :3] = np.dot(T_new[:3, :3], T_inv[:3, :3])
    # T_{local/inertia} = T_{local/original} + T_{original/inertia}
    # Must put all translation vectors in the same reference frame (local, as is the desired one)
    T_final[:3, 3] = T_new[:3, 3] + np.dot(T_new[:3, :3], T_inv[:3, 3])
    
    
    if mode == 'Model':
        KvalOpt = PCs[:NPc]
        
    elif mode == 'Random':
        
        KvalOpt = np.random.normal(loc=0, scale=1, size=NPc)
    
    OptGeom = SSMReconstruction(StatModel, KvalOpt, None, originalsize = True)
    
    OptGeom['Pcd'].transform(T_final)
    # Mesh
    OptGeom['Mesh'].transform(T_final)
    OptGeom['Mesh'].compute_triangle_normals()
    OptGeom['Mesh'].compute_vertex_normals()
    
    return OptGeom, KvalOpt, NPc, None, None, None

def CompareToSolution(stlpath: str, picklepath: str, OptGeom: dict, StatModel: dict, T: np.array, boneopt: str, center, algorithm: str, skin_landmarks, n_landmarks, landmarks, landmarks_positions, plots: bool = False, maxIter: int = 250, centerBool=False, regMethod='Cor'):
    
    
    """
    Compare reconstructed geometry against a reference solution.

    Parameters
    ----------
    stlpath : str
        Path to ground-truth STL geometry.
    picklepath : str
        Path to saved optimization results.
    OptGeom : dict
        Optimized geometry information.
    StatModel : dict
        Statistical Shape Model.
    T : np.ndarray
        Transformation matrix.
    boneopt : str
        Bone identifier.
    algorithm : str
        Optimization algorithm used.
    plots : bool, optional
        Enable visualization.
    maxIter : int, optional
        Maximum number of iterations.
    centerBool : bool, optional
        Center geometry before comparison.
    regMethod : str, optional
        Registration method used.

    Returns
    -------
    dict
        Error metrics and comparison results.
    """

    # Decimate original mesh until it is with same number of points of reconstruction
    NPoints = len(np.asarray(OptGeom['Pcd'].points))

    # mesh = SurfaceMeshSampling(trimesh.exchange.load.load(stlpath).as_open3d, int(NPoints)) # n tem as_open3d nesta versão
    o3d_mesh = trimesh_to_open3d(trimesh.exchange.load.load(stlpath))
    mesh = SurfaceMeshSampling(o3d_mesh, int(NPoints))
    pcd_new = o3d.geometry.PointCloud(o3d.cpu.pybind.utility.Vector3dVector(mesh.vertices))

    if (n_landmarks > 0 and not skin_landmarks.empty):
        NPoints = NPoints - n_landmarks
        pcd_new_array = np.asarray(pcd_new.points)
        
        for lm, pos in zip(landmarks, landmarks_positions):
            coord = (skin_landmarks[lm].iloc[0], 
                     skin_landmarks[lm].iloc[1], 
                     skin_landmarks[lm].iloc[2])
            pcd_new_array = np.insert(pcd_new_array, pos, coord, axis = 0)
        
        pcd_new.points = o3d.utility.Vector3dVector(pcd_new_array)
    
    if centerBool:
        
        NPoints = NPoints - 1
        pcd_new.points.append(center)

    # Now we have our reconstruction in the bony landmark file local reference frame
    # And the original geometry, imported from a file, in an unknow reference frame
    # So we put the original in its local reference frame as well
    
    # Note that aling the original solution with T_local is not advisable, since the bony landmarks file can have
    # a different referential frame from the original solution (because of the software for instance)
    # Indeed, this is only possible because the original .stl and the bony landmarks .txt is in the same reference frame 
    #TODO Solve this issue for the case where the above line is not valid
    
    # Local alignment for original geometry
    pcd_new.transform(T)
    mesh.transform(T)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    
    if plots:
        
        # Check the reconstruction and the original geometry
        o3d.visualization.draw_geometries([OptGeom['Pcd'].paint_uniform_color(blue),
                                           pcd_new.paint_uniform_color(red)])
        
        o3d.visualization.draw_geometries([OptGeom['Mesh'].paint_uniform_color(blue),
                                           mesh.paint_uniform_color(red)])

    # Rigid registration
    RigidReg = o3d.pipelines.registration.registration_icp(OptGeom['Pcd'], 
                                                           pcd_new, 
                                                           2 * np.max(pcd_new.compute_nearest_neighbor_distance()))
    OptGeom['Pcd'].transform(RigidReg.transformation)
    OptGeom['Mesh'].transform(RigidReg.transformation)
    OptGeom['Mesh'].compute_triangle_normals()
    OptGeom['Mesh'].compute_vertex_normals()
    
    
    if plots:
        
        # Check the reconstruction and the original geometry
        o3d.visualization.draw_geometries([OptGeom['Pcd'].paint_uniform_color(blue),
                                           pcd_new.paint_uniform_color(red)])
        
        o3d.visualization.draw_geometries([OptGeom['Mesh'].paint_uniform_color(blue),
                                           mesh.paint_uniform_color(red)])
        
        
    ## In terms of pcd and mesh visualization of the forms, this is as far as we can get (in terms of alignment)
    ## The rest of the code is to computed a distance metric between the two shapes, though not having influence on its shape
    ## That is, what is missing is correpondence between shapes
    
    # Save reconstruction and original solution in pickle
    
    # Open pickle
    with open(picklepath, "rb") as file:
        OptData = pickle.load(file)
        
    ## Add new data to pickle variable
    # The original solution provided
    OptData['OriginalMesh'] = {'Vertices' : copy.deepcopy(np.asarray(pcd_new.points)),
                               'Triangles' : copy.deepcopy(np.asarray(mesh.triangles))
                              }
    
    # And update best solution, rigidly aligned with solution
    # It is still the same reconstruction, just rigidly registered against the solution
    # cpdRes cannot be used, because the vertices are switched for elastic registration (correspondence), messing up the mesh.triangles variable
    OptData['Mesh'] = {'Vertices' : copy.deepcopy(np.asarray(OptGeom['Mesh'].vertices)),
                       'Triangles' : copy.deepcopy(np.asarray(OptGeom['Mesh'].triangles))
                      }
        
       
    # Save reconstruction and original solution in local referential frames.
    # Original solution
    OriGeom = {}
    OriGeom['Mesh'] = o3d.geometry.TriangleMesh()
    OriGeom['Mesh'].vertices = o3d.cpu.pybind.utility.Vector3dVector(OptData['OriginalMesh']['Vertices'])
    OriGeom['Mesh'].triangles = o3d.cpu.pybind.utility.Vector3iVector(OptData['OriginalMesh']['Triangles'])
    OriGeom['Mesh'].compute_triangle_normals()
    OriGeom['Mesh'].compute_vertex_normals()
    PolyDataOpt = MeshDataToPolyData(OriGeom)
    PolyDataOpt = PolyDataOpt.smooth(n_iter = 100)
    PolyDataOpt.save(stlpath.split('.stl')[0] + '_local' + '.stl')
    
    PolyDataOpt = MeshDataToPolyData(OptGeom)
    PolyDataOpt = PolyDataOpt.smooth(n_iter = 100)
    PolyDataOpt.save(picklepath.split('.pickle')[0] + '_local_aligned-with-sol' + '.stl')

    
    # Save the reconstruction with landmarks for visualization purposes
    if n_landmarks > 0:
        # Cria uma esfera à volta de cada landmark apenas para visualização
        PolyDataOpt = SphereLandmarks(OptGeom, PolyDataOpt, n_landmarks, landmarks_positions, plots)
        PolyDataOpt.save(picklepath.split('.pickle')[0] + '_local_aligned-with-sol_view' + '.stl')
    
    ## Correspondence
        
    # To obtain a distance metric, we should correspond point-to-point the two geometries
    # CPDCor is a shortcut to obtain a correspondence vector, but is not fool-proof (CQuental). 
    # It has a lot of repeated point-to-point correspondences. That is why Frederico triples the number of the target points, although that method is not the best one as well
    # cpdRes is the deformed source, which has better results than using CPDCor (CQuental)
    # We use cpdRes, because that way we guarantee a direct correspondence and, thus, we can compute a distance metric, for the reconstruction
    # The source must be OptGeom['Pcd'], so that the displacement can be used for heatmaps
    # But deformed source - source gives almost 0 error, which is not realistic.
    # Thus, Frederico's approach is considered, similarly to Validation.py in the SSM code
    # Normalization == 1 is not tested enough. It yields bad results. Use Normalization == 0 (CQuental)
    mesh_cp = SurfaceMeshSampling(mesh, int(3 * NPoints))
    pcd_new_cp = o3d.geometry.PointCloud(o3d.cpu.pybind.utility.Vector3dVector(mesh_cp.vertices))
    
    if (n_landmarks > 0 and not skin_landmarks.empty):
        pcd_new_cp_array = np.asarray(pcd_new_cp.points)

        for pos in landmarks_positions:
            coord_aligned = np.asarray(pcd_new.points)[pos, :]
            pcd_new_cp_array = np.insert(pcd_new_cp_array, pos, coord_aligned, axis = 0)
        
        pcd_new_cp.points = o3d.utility.Vector3dVector(pcd_new_cp_array)

    if centerBool:
        
        center_aligned = np.asarray(pcd_new.points)[-1, :]
        pcd_new_cp.points.append(center_aligned)
    
    clstdist = np.asarray(OptGeom['Pcd'].compute_point_cloud_distance(pcd_new))
    avgdistRMSE = np.sqrt(np.mean(np.square(clstdist)))
    avgdistMean = np.mean(clstdist)
    # Print closest average distance between shapes
    print(f'\nAverage nearest distance between reconstructed and original model : {round(avgdistMean,2)} mm')
    
    # The pcd distance between shapes
    OptData['NearestDisplacement'] = clstdist
    
    sCPD = time.perf_counter()
    
    SSMSettings = {'SamplingPoints': NPoints, 'OversamplingFactor': 3,
                   'Registration': regMethod}
    
    SSMSettings['CPDSettings'] = {'Algorithm': algorithm, 'w': 0.0,
                                             'MaxIter': maxIter, 'MaxError': 1e-6,
                                             'Normalization': 0, 'Beta': 1,
                                             'Lambda': 1, 'Invert': True}
    SSMSettings["Landmarks"] = n_landmarks
    
    cpdRes, CPDCor, _, _, _ = CustomCPD(OptGeom['Pcd'],
                                        pcd_new_cp,
                                        0,
                                        SSMSettings,               
                                        1,
                                        landmarks_positions,
                                        center=centerBool,
                                        plots= False)


    tCPD = time.perf_counter() - sCPD
    
    if plots:
        
        # Check the reconstruction and the original geometry
        o3d.visualization.draw_geometries([OptGeom['Pcd'].paint_uniform_color(blue),
                                               pcd_new_cp.paint_uniform_color(red)])
    
        #print('Result of PyCPD: ', np.asarray(cpdRes.points))
        #print('Source: ', np.asarray(OptGeom['Pcd'].points))
    
    # Get distance-metric between reconstruction and original geometries
    displacement = np.asarray(OptGeom['Pcd'].points) - np.asarray(pcd_new_cp.points)[CPDCor]
    #TODO use the line below to test direct correspondence, using source and deformed source distance metric
    # Test in InteractivePlots.py if distribution of error is good enough
    # displacement = np.asarray(OptGeom['Pcd'].points) - np.asarray(cpdRes.points)
    # Print distance - real average distance between shapes
    distanceMean = np.mean(np.linalg.norm(displacement, axis = 1))
    distanceRMSE = np.sqrt(np.mean(np.sum(np.square(displacement), axis = 1)))
    
    
    print(f'\nAverage corresponded distance distance between reconstructed and original model : {round(distanceMean,2)} mm')

    # Save non rigid regisration results
    OptData['CPDCor'] = copy.deepcopy(CPDCor)
    # The displacement between original and rigidly+non-rigidly registered reconstruction
    OptData['Displacement'] = displacement

    if (n_landmarks > 0 and not skin_landmarks.empty):
        PredictedLandmarks = np.asarray(OptGeom["Pcd"].points)[landmarks_positions, :]
        skin_landmarks_transformed = {lm: np.asarray(pcd_new.points)[pos, :] for lm, pos in zip(landmarks, landmarks_positions)}

        errors = []
        for i, lm in enumerate(landmarks):
            errors.append(np.linalg.norm(PredictedLandmarks[i] - skin_landmarks_transformed[lm]))

        mae_landmarks_error = np.mean(errors)
        rmse_landmarks_error = np.sqrt(np.mean(np.square(errors)))
    
    else:
        mae_landmarks_error = None
        rmse_landmarks_error = None
    
    OptimalGeomCenter = np.asarray(OptGeom['Pcd'].points)[-1 , : ]
    
    if centerBool:
        
        centerErrorArray = (OptimalGeomCenter - center_aligned)**2
        
        centerError = (np.sum(centerErrorArray))**(1/2)
        xSquared = centerErrorArray[0]
        ySquared = centerErrorArray[1]
        zSquared = centerErrorArray[2]
        
    else:
    
        centerError = None
        xSquared = None
        ySquared = None
        zSquared = None
    
    # Save the pickle again
    with open(picklepath, "wb") as file:
        pickle.dump(OptData, file, pickle.HIGHEST_PROTOCOL)
    
        
    return distanceMean, distanceRMSE, avgdistMean, avgdistRMSE, tCPD, centerError, xSquared, ySquared, zSquared, OptimalGeomCenter, mae_landmarks_error, rmse_landmarks_error

    