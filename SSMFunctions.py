# -*- coding: utf-8 -*-
"""
Created by Carlos Quental. Adapted by Augusto
Only does SSM (not SSAM or SDM) 
and uses one of the training geomtries as reference

"""

import copy
import numpy as np
import pyvista as pv
import pickle
import open3d as o3d
from MeshProperties import UpdatesMeshData
from StatisticsFunctions import CustomPCA, VarianceAnalysis
from MeshProperties import MeshDataToPolyData


def SSMBuild(StatModel, MeshData, SSMSettings, thresh, plots=False):
    """
    This function buids the statistical shape model

    Parameters
    ----------
    SSM : TYPE
        DESCRIPTION.
    MeshData : TYPE
        DESCRIPTION.
    SSMSettings : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    # # Number of models
    NModels = len(MeshData)

    # # Allocates memory for the output
    StatModel['uTraining'] = np.zeros(
        (np.asarray(StatModel['MVShape']['Pcd'].points).shape[0] * 3, NModels))

    for i in range(NModels):

        # Update the displacement between the MVShape and the i-th instance.
        # It needs to denormalize the cpdRes data
        cpdRes = copy.deepcopy(MeshData[i]['Pcd'])

        # TODO COMPUTE DIFFERENCE BETWEEN MVSHAPE AND TARGET WITH CPDCORRESPONDENCE
        # Cu = copy.deepcopy(
        #     (np.asarray(StatModel['MVShape']['Pcd'].points) * StatModel['MVShape']['CSize']) -
        #     (np.asarray(MeshData[i]['Pcd'].points) * MeshData[i]['CSize'])[CPDCorrespondence]
        #     )
        Cu = copy.deepcopy(
            (np.asarray(cpdRes.points) * MeshData[i]['CSize']) -
            (np.asarray(StatModel['MVShape']['Pcd'].points)
             * StatModel['MVShape']['CSize'])
        )

        # Adds the data normalized by the size of the MVShape --> PCA needs normalized data
        StatModel['uTraining'][:, i] = copy.deepcopy(
            Cu.flatten() / StatModel['MVShape']['CSize'])

    # Builds the SSM model
    # Computes the mean shape
    StatModel['SSM']['Mean'] = np.mean(StatModel['uTraining'], axis=1)

    # Computes the eigenvalues and eigenvectors of the covariance matrix
    StatModel['SSM']['EigVal'], StatModel['SSM']['EigVec'], pca_object, StatModel['SSM']['Transform'] = CustomPCA(StatModel['uTraining'], 'PCA')

    # Analysis of the variance
    # Relevant eigenvalues are the ones who cumulatively explain thresh*100% variability
    # of the training shape dataset
    StatModel['SSM']['EigValRelev'] = VarianceAnalysis(StatModel['SSM']['EigVal'],
                                                       thresh,
                                                       plots)

    return StatModel, pca_object

'''def transform(StatModel, shape):
    
    print(StatModel['SSM']['Mean'].shape)
    print((shape.T.flatten()-StatModel['SSM']['Mean']).shape)
    print(np.diag(1/np.sqrt(StatModel['SSM']['EigVal'])).shape)
    print((StatModel['SSM']['EigVec'].T).shape)
    k = np.diag(1/np.sqrt(StatModel['SSM']['EigVal']))@StatModel['SSM']['EigVec'].T@(shape.T-StatModel['SSM']['Mean'].reshape((len(shape.T), 1)))
    print(k.shape)
        
    return k'''

def SSMReconstruction(StatApp, Kval, SSMSettings, originalsize):
    """


    Parameters
    ----------
    StatApp : TYPE
        DESCRIPTION.
    NEigVal : TYPE
        DESCRIPTION.
    k : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    # Number of eigenvalues
    NEigVal = len(Kval)

    # Mean data
    Mean = copy.deepcopy(StatApp['SSM']['Mean'])

    # Allocates memory for the standard deviation
    SD = np.zeros(Mean.shape[0])

    # Goes through NEigVal values and computes the displacements along the principal directions
    for i in range(NEigVal):

        SD += Kval[i] * np.sqrt(StatApp['SSM']['EigVal']
                                [i]) * StatApp['SSM']['EigVec'][:, i]


    # The total data are the mean data plus the standard deviation data
    TotData = Mean + SD

    # Makes the output mesh equal to the MV Shape and then updates its vertices
    MeshOut = UpdatesMeshData(StatApp, TotData, orisize=originalsize)

    # return MeshOutOriginal
    return MeshOut

def SSMReconstruction2(pca, StatApp, Kval, SSMSettings, originalsize):
    """


    Parameters
    ----------
    StatApp : TYPE
        DESCRIPTION.
    NEigVal : TYPE
        DESCRIPTION.
    k : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    if len(Kval) < pca.n_components_:

        new_coordinates = []

        for i in range(len(Kval)):

            new_coordinates.append(Kval[i])
            
        
        for j in range(len(Kval), pca.n_components_):
            
            new_coordinates.append(0)
    
    else: new_coordinates = Kval
    
    TotData = pca.inverse_transform(np.asarray([new_coordinates]))
    
    
    # Makes the output mesh equal to the MV Shape and then updates its vertices
    MeshOut = UpdatesMeshData(StatApp, TotData[0], orisize=originalsize)

    # return MeshOutOriginal
    return MeshOut

def CheckPC(pca, StatApp, EigRel, SSMSettings, MeshData, center, landmarks_positions, bopt, iters=1000, plots=False):
    """
    This function goes through the relevant eigenvalues and plots
    the principal components to see what they change.

    Parameters
    ----------
    SSM : TYPE
        DESCRIPTION.
    EigRel : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    # Allocates memory for the array k
    Kval = np.zeros(EigRel,)

    # Builds the mean
    MeshOutMean = SSMReconstruction(
        StatApp, Kval, SSMSettings, originalsize=True)
    PolyDataMean = MeshDataToPolyData(MeshOutMean)
    PolyDataMean = PolyDataMean.smooth(n_iter=iters)

    if SSMSettings["Landmarks"] > 0:
        # Cria uma esfera à volta de cada landmark apenas para visualização
        PolyDataMean = SphereLandmarks(MeshOutMean, PolyDataMean, SSMSettings["Landmarks"], landmarks_positions, plots)

    if center:
        # Create a sphere around the center point, for visualization proposes
        PolyDataMeanCenter = SphereCenter(MeshOutMean, PolyDataMean, plots)

        PolyDataMean = PolyDataMeanCenter

    NModels = len(MeshData)
    NPoints = SSMSettings['SamplingPoints']
    algorithm = SSMSettings['CPDSettings']['Algorithm']
    method = SSMSettings['Registration']

    PolyDataMean.save(
        f'Results\\{bopt}\\Mean_NModels{NModels}_NPoints{NPoints}_{algorithm}_{method}_center{center}_nlandmarks{SSMSettings["Landmarks"]}.stl')

    # Goes through all relevant eigenvalues
    for i in range(EigRel):

        # Applies 3 SD to the current eigrel
        Kval[i] = 3

        # Builds the mesh
        MeshOutPlus = SSMReconstruction(
            StatApp, Kval, SSMSettings, originalsize=True)


        # Applies -3 SD to the current eigrel
        Kval[i] = -3

        # Builds the mesh
        MeshOutMinus = SSMReconstruction(
            StatApp, Kval, SSMSettings, originalsize=True)

        # Saves the 2 meshes
        PolyDataPlus = MeshDataToPolyData(MeshOutPlus)
        PolyDataPlus = PolyDataPlus.smooth(n_iter=iters)

        PolyDataMinus = MeshDataToPolyData(MeshOutMinus)
        PolyDataMinus = PolyDataMinus.smooth(n_iter=iters)

        if SSMSettings["Landmarks"] > 0:
        # Cria uma esfera à volta de cada landmark apenas para visualização
            PolyDataPlus = SphereLandmarks(MeshOutPlus, PolyDataPlus, SSMSettings["Landmarks"], landmarks_positions, plots)

            PolyDataMinus = SphereLandmarks(MeshOutMinus, PolyDataMinus, SSMSettings["Landmarks"], landmarks_positions, plots)

        if center:
            # Create a sphere around the center point, for visualization proposes
            PolyDataPlusCenter = SphereCenter(MeshOutPlus, PolyDataPlus, plots)

            PolyDataPlus = PolyDataPlusCenter

            # Create a sphere around the center point, for visualization proposes
            PolyDataMinusCenter = SphereCenter(
                MeshOutMinus, PolyDataMinus, plots)

            PolyDataMinus = PolyDataMinusCenter

        PolyDataPlus.save(
            f'Results\\{bopt}\\PC{i+1}_NModels{NModels}_NPoints{NPoints}_{algorithm}_{method}_center{center}_nlandmarks{SSMSettings["Landmarks"]}_+3.stl')
       
        PolyDataMinus.save(
            f'Results\\{bopt}\\PC{i+1}_NModels{NModels}_NPoints{NPoints}_{algorithm}_{method}_center{center}_nlandmarks{SSMSettings["Landmarks"]}_-3.stl')

        Kval[i] = 0

def SphereCenter(Mesh, PolyData, plots=False):
    """
    Creates a sphere for visualization proposes around the center of the 
    glenohumeral joint. Returns Poly data with the sphere around the center

    """

    PointCouldArray = np.array(Mesh['Pcd'].points)

    cent = PointCouldArray[-1:,]

    PolyDataSphere = pv.Sphere(radius=5, center=(
        cent[0][0], cent[0][1], cent[0][2]))

    if plots:
        # Graphical representation
        pl = pv.Plotter()
        pl.add_mesh(PolyData, color='r', style='wireframe', line_width=3)
        pl.add_mesh(PolyDataSphere, color='b', style='wireframe', line_width=3)
        pl.show()

    PolyDataFinal = PolyData + PolyDataSphere

    return PolyDataFinal

def SphereLandmarks(Mesh, PolyData, n_landmarks, landmarks_positions, plots = False):
    """
    Creates a sphere for visualization proposes around each landmark.
    Return Poly data with the spheres around the landmarks
    """

    PointCloudArray = np.array(Mesh['Pcd'].points)

    landmarks = PointCloudArray[landmarks_positions, :]

    PolyDataSphereLandmarks = pv.PolyData()

    for point in landmarks:
        sphere = pv.Sphere(radius = 5, center = point)
        PolyDataSphereLandmarks = PolyDataSphereLandmarks + sphere
    
    if plots:
        # Graphical representation
        pl = pv.Plotter()
        pl.add_mesh(PolyData, color = 'r', style = 'wireframe', line_width = 3)
        pl.add_mesh(PolyDataSphereLandmarks, color = 'b', style = 'wireframe', line_width = 3)
        pl.show()
    
    PolyDataFinal = PolyData + PolyDataSphereLandmarks

    return PolyDataFinal

def SSMSaveData(StatModel, SSMSettings, MeshData, bopt, center):
    """
    Saves a StatModel.pickle file in the Results folder
    To be imported by any given application using the SSM
    Any further data is not suppose to be used 
    As it is constrained to the training data given as input to the SSM
    """

    # We can save the points of point cloud, vertices of mesh
    # and traingles of mesh, all as numpy arrays

    StatModel_cp = copy.deepcopy(StatModel)

    # Change StatModel dictionary to be pickable
    StatModel_cp['MVShape']['Mesh'], StatModel_cp['MVShape']['Pcd'] = {}, {}
    # Add the information about MVShape mesh
    StatModel_cp['MVShape']['Mesh']['Vertices'] = copy.deepcopy(
        np.asarray(StatModel['MVShape']['Mesh'].vertices))
    StatModel_cp['MVShape']['Mesh']['Triangles'] = copy.deepcopy(
        np.asarray(StatModel['MVShape']['Mesh'].triangles))
    # Add the information about MVShape point cloud
    StatModel_cp['MVShape']['Pcd']['Points'] = copy.deepcopy(
        np.asarray(StatModel['MVShape']['Pcd'].points))

    algorithm = SSMSettings['CPDSettings']['Algorithm']
    method = SSMSettings['Registration']
    NModels = len(MeshData)
    NPoints = SSMSettings['SamplingPoints']

    with open(f'Results\\{bopt}\\StatModel_NModels{NModels}_NPoints{NPoints}_{algorithm}_{method}_center{center}_nlandmarks{SSMSettings["Landmarks"]}.pickle', "wb") as file:
        pickle.dump(StatModel_cp, file, pickle.HIGHEST_PROTOCOL)
