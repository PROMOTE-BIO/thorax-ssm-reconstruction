# -*- coding: utf-8 -*-
"""
Main script for geometry reconstruction using anatomical landmarks.

This script supports reconstruction based on either:
- Skin landmarks (SSM-SL-based)
- Bony landmarks (SSM-BL-based)

Landmarks are transformed into a local anatomical reference frame prior to
statistical shape model (SSM)-based reconstruction.
"""


# =========================
# Standard library imports
# =========================
import os
import time
from datetime import datetime
import copy
import pickle
import re


# =========================
# Third-party libraries
# =========================
import numpy as np
import pandas as pd
import open3d as o3d


# =========================
# Project-specific modules
# =========================
from MeshProperties import MeshDataToPolyData
from BoneReconFunctions import (
    ReadSSMInfo, ProcessAverageShape, ReconstructFromLandmark, 
    modelReconstruction, CompareToSolution, GetLocalLandmarkFromCoord
    )
from SkinToBoneMapping import organize_predictors, apply_equation


# =========================
# Working directories
# =========================
# Working directory and default paths
cwd = os.getcwd()
# change paths if moved
path = cwd + "\\MeanShape"
mean_shape_path = cwd + "\\MeanShape\\Mean_Bony_Landmarks.txt"


# ============================================================
# Core reconstruction utilities
# ============================================================


def Reconstruct(TrainedSSMData: dict, coords, T_local, skin_landmarks, 
                ReconSettings: dict, OptSettings: dict, CompSettings: dict):
    
    
    """
    Perform geometry reconstruction using a trained Statistical Shape Model.

    Depending on the selected method, this function:
    - Uses skin or bony landmarks
    - Matches them to the statistical model
    - Optimizes shape parameters (principal components)
    
    Optionally, it compares the reconstructed geometry with a ground-truth solution.
    
    Parameters
    ----------
    TrainedSSMData : dict
        Trained Statistical Shape Model data.
    coords : array-like
        Input landmark coordinates (skin or bony).
    T_local : np.ndarray
        Transformation matrix to the local anatomical reference frame.
    skin_landmarks : list or array
        List of skin landmarks used for mapping or reconstruction.
    ReconSettings : dict
        Reconstruction configuration parameters.
    OptSettings : dict
        Optimization settings (algorithm, constraints, iterations).
    CompSettings : dict
        Comparison settings for solution evaluation.

    Returns
    -------
    dict
        Reconstruction results and optimized geometry.
    """

    t1 = time.perf_counter()

    # Convert landmark dictionary to point cloud
    land = np.array([coords[lm] for lm in ReconSettings['Landmarks']], dtype=float)
    landmarkpcd = o3d.geometry.PointCloud(o3d.cpu.pybind.utility.Vector3dVector(land))

    # Load statistical shape model
    TrainedSSM = ReadSSMInfo(TrainedSSMData, ReconSettings['Bone'], 
                             ReconSettings['Plots'])

     # If using skin landmarks, their correspondence is already known
    if (ReconSettings['NLandmark'] > 0 and ReconSettings['ReconsMethod'] == 'SSM-SL-based'):
        closest_points_av = ReconSettings['LandmarkPos']
        pointsList = ReconSettings['LandmarkPos']
        closestIndex = list(range(ReconSettings['NLandmark']))
        mean_shape_path_f = None
    else:
        # Otherwise, find corresponding points on the mean shape
        (closest_points_av, pointsList, 
         closestIndex) = ProcessAverageShape(TrainedSSM, 
                                             ReconSettings['Bone'], path, 
                                             ReconSettings['Plots'], 
                                             radius = OptSettings['Radius']) 
        mean_shape_path_f = mean_shape_path

    # Perform reconstruction. The output geometry (OptGeom) is in the local reference frame.
    if (ReconSettings['ReconsAlg'] == 'SSM-Opt'):
        # Landmark-driven optimization
        (OptGeom, KvalOpt, tOpt, calls, 
         fit) = ReconstructFromLandmark(TrainedSSM, landmarkpcd, 
                                        closest_points_av, T_local, 
                                        ReconSettings['OutputName'] + '_Reconstructed', 
                                        ReconSettings['Bone'],
                                        ReconSettings['Plots'], 
                                        OptSettings['OptApproach'], 
                                        ReconSettings['PC'], 
                                        OptSettings['OptAlgorithm'], 
                                        OptSettings['RegularizationTerm'], 
                                        OptSettings['RegularizationWeight'], 
                                        OptSettings['GradAtEnd'],
                                        pointsList, closestIndex, mean_shape_path_f)
    elif (ReconSettings['ReconsAlg'] == 'SSM-Reg'):
        # Regression-based or random reconstruction. This is not implemented 
        # for the thorax reconstruction. This is only for other bones.
        (OptGeom, KvalOpt, tOpt, calls, 
         fit) = modelReconstruction(ReconSettings['PCSol'], ReconSettings['PC'], TrainedSSM, 
                                    ReconSettings['Bone'], ReconSettings['ReconsAlg'], mean_shape_path_f)
                                       
    t2, t3 = time.perf_counter(), None

     # Optional comparison with ground-truth geometry
    if (CompSettings['ExistsSol'] == 'Yes'):

        stlpath = CompSettings['StlToComp']

        picklepath = ReconSettings['OutputName'] + '_Reconstructed' + '.pickle'
        (distanceMean, distanceRMSE, avgdistMean, avgdistRMSE, tCPD, 
         centError, x, y, z, centerOpt, mae_landmarks_error, 
         rmse_landmarks_error) = CompareToSolution(stlpath, picklepath, OptGeom, 
                                                   TrainedSSM, T_local, ReconSettings['Bone'], 
                                                   CompSettings['JointCenter'], 
                                                   ReconSettings['NonRegAlg'], 
                                                   skin_landmarks, ReconSettings['NLandmark'], 
                                                   ReconSettings['Landmarks'], 
                                                   ReconSettings['LandmarkPos'], 
                                                   ReconSettings['Plots'], 
                                                   CompSettings['MaxIter'], 
                                                   CompSettings['JointCenterBool']
                                                   )
        t3 = time.perf_counter()
    else:
        # Default values when no comparison is performed
        distanceMean = 0
        distanceRMSE = 0
        avgdistMean = 0
        avgdistRMSE = 0
        tCPD = 0
        centError = 0
        x = 0
        y = 0
        z = 0
        centerOpt = 0
        mae_landmarks_error = 0
        rmse_landmarks_error = 0
        t3 = t2
    
    return [TrainedSSM, OptGeom, KvalOpt, landmarkpcd, distanceMean, 
            avgdistMean, distanceRMSE, avgdistRMSE, t1, t2, t3, tOpt, tCPD, 
            centError, x, y, z, calls, fit, centerOpt, mae_landmarks_error, 
            rmse_landmarks_error]
 
    
def print_and_save(resultname, parameters, Npc, n_landmarks, tipo, sol):

    
    """
    Print reconstruction results to console and save them to disk.

    Parameters
    ----------
    resultname : str
        Base name for result files.
    parameters : dict
        Reconstruction and optimization parameters.
    Npc : int
        Number of principal components used.
    n_landmarks : int
        Number of landmarks used in reconstruction.
    tipo : str
        Reconstruction type identifier.
    sol : dict
        Solution data to be saved.

    Returns
    -------
    dict
        Reconstruction results and optimized geometry.
    """

    [StatModel, OptGeom, KvalOpt, landmarkpcd, distanceMean, avgdistMean, distanceRMSE, avgdistRMSE, t1, t2, t3, tOpt, tCPD, error, x, y, z, calls, fit, centerOpt, mae_landmarks_error, rmse_landmarks_error] = parameters

    print('       ======================== Additional info ========================          \n')
    print(f"Number of points/vertices: {np.asarray(StatModel['MVShape']['Pcd'].points).shape[0]}")
    print(f'Number of principal components: {Npc}')
    if n_landmarks == 0:
        print(f'Number of bony landmarks: {np.asarray(landmarkpcd.points).shape[0]}')
    else:
        print(f'Number of skin landmarks: {np.asarray(landmarkpcd.points).shape[0]}')
    print('Duration of reconstruction = ' + time.strftime("%H:%M:%S", time.gmtime(t2 - t1)))
    print('Duration of GA+TNC = ' + time.strftime("%H:%M:%S", time.gmtime(tOpt)))
    print('Type of running = ' + tipo)

    if sol == 'Yes':
        print('Duration of comparison to original solution = ' + time.strftime("%H:%M:%S", time.gmtime(t3 - t2)))
        print('Duration of CustomCPD = ' + time.strftime("%H:%M:%S", time.gmtime(tCPD)))
        print(f'Corresponded distance between reconstruction and original solution: {round(distanceMean, 2)} mm')
        print(f'Nearest distance between reconstructed and original model : {round(avgdistMean,2)} mm')
        print('Total Duration = ' + time.strftime("%H:%M:%S", time.gmtime(t3 - t1)))
        
    if n_landmarks == 0:
        indxs = ['Points', 'PCs', 'Bony_landmarks', 'Total_Duration', 'Duration_Reconstruction', 'Duration_GA+TNC', 'Type_Running']
    else:
        indxs = ['Points', 'PCs', 'Skin_landmarks', 'Total_Duration', 'Duration_Reconstruction', 'Duration_GA+TNC', 'Type_Running']
    vals = [np.asarray(StatModel['MVShape']['Pcd'].points).shape[0], Npc, np.asarray(landmarkpcd.points).shape[0],
            time.strftime("%H:%M:%S", time.gmtime(t3 - t1)),
            time.strftime("%H:%M:%S", time.gmtime(t2 - t1)),
            time.strftime("%H:%M:%S", time.gmtime(tOpt)), tipo]
                            
    if sol == 'Yes':
        indxs += ['Duration_Comparison', 'Duration_CPD', 'Corresponded_distance', 'Nearest_distance', "Corresponded_RMSE_distance", "Nearest_RMSE_distance"]
        vals += [time.strftime("%H:%M:%S", time.gmtime(t3 - t2)), time.strftime("%H:%M:%S", time.gmtime(tCPD)), f'{round(distanceMean, 2)} mm', f'{round(avgdistMean, 2)} mm', f'{round(distanceRMSE, 2)} mm', f'{round(avgdistRMSE, 2)} mm']
                            
        df = pd.DataFrame(vals, columns = ['Parameter'], index = indxs)
        df.to_csv(resultname + '_Output.txt', header = None, sep = '=')
    if n_landmarks != 0:
        indxs += ["MAE_landmarks", "RMSE_landmarks"]
        vals += [mae_landmarks_error, rmse_landmarks_error]

    return [StatModel, OptGeom, KvalOpt, landmarkpcd, distanceMean, 
            avgdistMean, distanceRMSE, avgdistRMSE, t1, t2, t3, tOpt, tCPD, 
            error, x, y, z, calls, fit, centerOpt, mae_landmarks_error, 
            rmse_landmarks_error]


def ssm_thorax_reconstruction(SubjectFile: str = '', ReconstructMethod: str ='SSM-SL-based',
               nPCs = None, PlotGeometries: bool = None, CompareSol: str = None):
    
    
    """
    High-level entry point for thorax reconstruction using SSMs.

    This function:
    - Loads subject-specific landmark data
    - Selects reconstruction method (skin-based or bony-based)
    - Handles SSM projection and reconstruction
    - Optionally visualizes reconstructed geometries
    - Optionally compares results against a reference solution

    Parameters
    ----------
    SubjectFile : str
        Path to subject landmark file.
    ReconstructMethod : str
        Reconstruction method:
        - 'SSM-SL-based' (see paper for details)
        - 'SSM-BL-based' (see paper for details)
    nPCs : int or list, optional
        Number(s) of principal components to test.
    PlotGeometries : bool, optional
        Enable visualization of reconstructed geometries.
    CompareSol : str, optional
        Path to reference solution for comparison.

    Returns
    -------
    Saves the reconstructed geometry to the Results folder
    """

    
    # Defines default values for the parameters that are optional.
    if (nPCs == None):
        nPCs = -1
    if (PlotGeometries == None):
        PlotGeometries = True
    if (CompareSol == None):
        CompareSol = 'False'
        
    # Checks if the subjectfile is not empty and it the file exists. It only
    # continues if the file exists
    if ((SubjectFile != '' and os.path.exists(f"{cwd}\\InputData\\{SubjectFile}.csv")) and
        (ReconstructMethod == 'SSM-SL-based' or ReconstructMethod == 'SSM-BL-based')):
    
        start = datetime.now()
        print('\nStarting date and time: ' + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + '\n')
        start = time.process_time()
    
        # Reconstruction settings
        ReconSettings = {'Bone': "Thorax", # Bone to be processed
                         'NTrain': 60, # Number of subjects used for training the SSM
                         'NPoints': 20000, # Number of points in the base point cloud
                         'NonRegAlg': "BCPD", # Algorithm for the non-rigid registration
                         'Plots': False, # Boolean for debugging
                         'PC': int(nPCs), # Number of principal components to use
                         'ReconsAlg': "SSM-Opt", # Type of reconstruction method: SSM-Opt uses optimmization while SSM-Reg uses linear regression with SSM. For the thorax, only the SSM-Opt is available.
                         'ReconsMethod': ReconstructMethod,
                         'PCSol': None, # PC solution, if available (mainly for SSM-Reg)
                         'OutputName': f"{cwd}\\Results\\{SubjectFile}" # Name of the output file with error metrics
                         }
        # Default number of principal components
        if (ReconSettings['PC'] == -1):
            ReconSettings['PC'] = 2
        # If Bone is Thorax, ReconsMethod must always be SSM-Opt
        if (ReconSettings['Bone'] == "Thorax"):
            ReconSettings['ReconsAlg'] = "SSM-Opt"
        # Landmark names in the same order as the SSM. This code is specific of the thorax only
        if (ReconSettings['Bone'] == "Thorax"):
            ReconSettings['Landmarks'] = ['C7', 'T8', 'JN', 'XP', 'R10']
        # Defines the number of landmarks
        if (ReconSettings['ReconsMethod'] == 'SSM-BL-based'):
            ReconSettings['NLandmark'] = 0  # 0 if bony; 5 if skin
        elif (ReconSettings['ReconsMethod'] == 'SSM-SL-based'):
            ReconSettings['NLandmark'] = 5  # 0 if bony; 5 if skin                
        ReconSettings['LandmarkPos'] = list(range(ReconSettings['NPoints'], 
                                                  ReconSettings['NPoints'] + ReconSettings['NLandmark']))
            
        # Optimization settings
        OptSettings = {'OptApproach': 'fast', # Optimization approach 
                       'OptAlgorithm': 'DIRECT2', # Optimization algorithm
                       'RegularizationTerm': 'Sobral', # Type of regularization term to use. There are two options: "Marques" based on the paper "Reconstruction of Scapula Bone Shapes from Digitized Skin Landmarks Using Statistical Shape Modeling and Multiple Linear Regression"; and "Sobral", like the regularization used in the thorax paper
                       'RegularizationWeight': 0.1, # Regularization factor in the optimization problem
                       'GradAtEnd': True, # Bool to tell whether a gradient algorithm (TNC) is to be used at the end of the optimization
                       'Radius': 0 # Testing variable. Not relevant for now.
                       }
                
        # Comparison settings
        CompSettings = {'ExistsSol': "Yes", # "Yes" or "No" depending on whether a solution for comparison exists
                        'JointCenterBool': False, # This is a boolean to tell if the joint center is a point. This does not apply to the thorax
                        'JointCenter': None, # These are the coordinates of the joint center. This does not apply to the thorax
                        'MaxIter': 250
                        }
        if (CompareSol == 'False'):
            CompSettings['StlToComp'] = '' # Path to the stl to compare
            CompSettings['ExistsSol'] = "No"
        else:
            CompSettings['StlToComp'] = f"{cwd}\\InputData\\{CompareSol}.stl"
        
        " Processes the skin and subject-specific parameters "
        # Reads data from the file and organizes the data into demographics and landmark
        AllData = pd.read_csv(f"{cwd}\\InputData\\{SubjectFile}.csv", sep = ',', header = None, index_col = False).T 
        AllData.columns = AllData.iloc[0]
        AllData.drop(0, inplace = True)  # index remains 1, 2 and 3
        # Subject-specific variables input
        SubjectVar = {'Age': AllData['Age'].iloc[0],
                      'Sex': AllData['Sex'].iloc[0],
                      'Height': AllData['Height'].iloc[0],
                      'Weight': AllData['Weight'].iloc[0]}
        ## Landmark data
        skinCoord = AllData.drop(columns = ['Age', 'Sex', 'Height', 'Weight'])
        # Transforms the coordinates into the local reference frame of the bone according to ISB
        localskinCoord, T_local = GetLocalLandmarkFromCoord(skinCoord, ReconSettings['Bone'])
    
        " Reads the statistical shape model"
        # Load training SSM data
        if (ReconSettings['ReconsMethod'] == 'SSM-BL-based'):
            with open(cwd + f'\\SSMs\\SSM_BL_NModels{ReconSettings['NTrain']}_NPoints{ReconSettings['NPoints']}_{ReconSettings['NonRegAlg']}_Cor_center{CompSettings['JointCenterBool']}_nlandmarks{ReconSettings['NLandmark']}.pickle', "rb") as file:
                TrainedSSMData = pickle.load(file)
        elif (ReconSettings['ReconsMethod'] == 'SSM-SL-based'):
            with open(cwd + f'\\SSMs\\SSM_SL_NModels{ReconSettings['NTrain']}_NPoints{ReconSettings['NPoints']}_{ReconSettings['NonRegAlg']}_Cor_center{CompSettings['JointCenterBool']}_nlandmarks{ReconSettings['NLandmark']}.pickle', "rb") as file:
                TrainedSSMData = pickle.load(file)
    
            
        # Reconstructs the bone geometry using the selected method.
        if (ReconSettings['ReconsMethod'] == 'SSM-BL-based'):
        
            # Reads the regression equation data from a csv file
            equations_data = pd.read_csv(cwd + "\\SkinToBoneCoefficients.csv", usecols = ["target", "equation"])

            # Makes the equations in the csv file "usable"
            equation_funcs = {} # These are functions to be used as: result = equation_funcs["x_C7_bony"](**inputs) ; inputs = {"age": 30,"d_C7_JN_skin": 50,"d_XP_R10_skin": 60} in dictionary
            equation_vars = {} # Predictors needed for the equations

            for idx, row in equations_data.iterrows():
                target = row["target"]
                equation = row["equation"]

                expression = equation.split("=", 1)[1].strip()

                vars_in_eq = sorted(set(re.findall(r'\b[a-zA-Z_]\w*\b', expression)) - {target})

                equation_vars[target] = vars_in_eq

                def make_equation(expr):
                    return lambda **kwargs: eval(expr, {}, kwargs)
            
                equation_funcs[target] = make_equation(expression)

            # Builds the predictors
            reg_predictors = organize_predictors(equation_vars, localskinCoord, SubjectVar)
        
            # Goes through all landmarks and computes the corresponding bony landmarks
            # Makes a copy of the localSkinCoord to keep the same structure
            localboneCoord = copy.deepcopy(localskinCoord)
        
            for name in ReconSettings['Landmarks']:
            
                # Updates the x, y, and z coordinates
                localboneCoord.loc['xcoord', name] = apply_equation(equation_vars, 
                                                                    equation_funcs, 
                                                                    'x_' + name + '_bony', 
                                                                    reg_predictors)
                localboneCoord.loc['ycoord', name] = apply_equation(equation_vars, 
                                                                    equation_funcs, 
                                                                    'y_' + name + '_bony', 
                                                                    reg_predictors)
                localboneCoord.loc['zcoord', name] = apply_equation(equation_vars, 
                                                                    equation_funcs, 
                                                                    'z_' + name + '_bony', 
                                                                    reg_predictors)
            
            # Identifies the optimal PC components for the reconstruction of the thorax            
            ReconParameters = Reconstruct(TrainedSSMData, localboneCoord, T_local, None, 
                                     ReconSettings, OptSettings, CompSettings)
    
            print('Bony Landmark Reconstruction concluded.\n')
            [SSMModel, OptGeom, KvalOpt, landmarkpcd, distanceMean, 
             avgdistMean, distanceRMSE, avgdistRMSE, t1, t2, t3, tOpt, tCPD, 
             error, x, y, z, calls, fit, centerOpt, mae_landmarks_error, 
             rmse_landmarks_error] = print_and_save(ReconSettings['OutputName'], 
                                                    ReconParameters, 
                                                    ReconSettings['PC'],
                                                    ReconSettings['NLandmark'], 
                                                    OptSettings['OptApproach'], 
                                                    CompSettings['ExistsSol'])

        elif (ReconSettings['ReconsMethod'] == 'SSM-SL-based'):
        
            ReconParameters = Reconstruct(TrainedSSMData, localskinCoord, T_local, skinCoord, 
                                     ReconSettings, OptSettings, CompSettings)
    
            print('Skin Landmark Reconstruction concluded.\n')
                            
            [SSMModel, OptGeom, KvalOpt, landmarkpcd, distanceMean, 
             avgdistMean, distanceRMSE, avgdistRMSE, t1, t2, t3, tOpt, tCPD, 
             error, x, y, z, calls, fit, centerOpt, mae_landmarks_error, 
             rmse_landmarks_error] = print_and_save(ReconSettings['OutputName'], 
                                                    ReconParameters, 
                                                    ReconSettings['PC'],
                                                    ReconSettings['NLandmark'], 
                                                    OptSettings['OptApproach'], 
                                                    CompSettings['ExistsSol'])

            #print(f"MAE Landmarks error: {mae_landmarks_error}\n")
            #print(f"RMSE Landmarks error: {rmse_landmarks_error}\n")
        
        else:
            print('Reconstruction method unknown:' + ReconSettings['ReconsMethod'])
            input('Press any key to abort.')

        end = time.process_time()
        elapsed = end - start
        print("Case reconstructed!!")
        print(f"Time elapsed: {elapsed:.2f} seconds!\n")
            
        """ 
        Plots the results
        """
        # The OptGeom is in the local reference frame of the body
        if (PlotGeometries):
            # Skin landmarks
            Visskinlandmarks = np.array(localskinCoord.T).astype(float)
        
            VisskinlandmarkMesh = [OptGeom['Mesh'].paint_uniform_color([1, 0, 0])]
            for coord in Visskinlandmarks:
                
                # Creates a shpere mesh with radius 0.5
                mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5.0)

                # Compute vertex normals for proper rendering
                mesh_sphere.compute_vertex_normals()
                
                # Paint the sphere a color (optional)
                mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7]) # Blue

                # Move the sphere to global position (x, y, z)
                mesh_sphere.translate(coord)
            
                # Append the mesh in VisskinlandmarkMesh
                VisskinlandmarkMesh.append(mesh_sphere)

            # Visualization of the results
            o3d.visualization.draw_geometries(VisskinlandmarkMesh)
    
        # Saves the bone geometry to an stl file
        PolyDataOpt = MeshDataToPolyData(OptGeom)
        PolyDataOpt.save(f"{cwd}\\Results\\{SubjectFile}.stl")
    else:
        if (SubjectFile == '' or not os.path.exists(f"{cwd}\\InputData\\{SubjectFile}.csv")):
            print('Subjectfile "' + SubjectFile + '" not found. Aborting.')
        elif (ReconstructMethod != 'SSM-SL-based' and ReconstructMethod != 'SSM-BL-based'):
            print('Reconstruction method "' + ReconstructMethod + '" not defined . Aborting.')



# ============================================================
# Script execution entry point
# ============================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--filename', metavar='filename', required=True,
                        help='Subject filename (csv type), excluding extension')
    parser.add_argument('--method', metavar='method', required=True,
                        help='Reconstruction method: SSM-SL-based (skin-embedded model) or SSM-BL-based. The default is the SSM-SL-based approach.')
    parser.add_argument('--PCs', metavar='PCs', required=False,
                        help='Number of PCs to use; The default is 8')
    parser.add_argument('--plot', metavar='plot', required=False,
                        help='True for plotting the results at the end')
    parser.add_argument('--compare', metavar='compare', required=False,
                        help='If a solution exists, input the stl filename for comparison, excluding extension')
    args = parser.parse_args()
    
    ssm_thorax_reconstruction(args.filename, args.method, args.PCs, args.plot, args.compare)