# -*- coding: utf-8 -*-
"""
Created by Carlos Quental. Adapted by Augusto
Only does SSM (not SSAM or SDM) 
and uses one of the training geomtries as reference

"""

import os
import copy
import numpy as np
import pyvista as pv
from functools import partial
import matplotlib.pyplot as plt
import open3d as o3d
import sys
from probreg import cpd
from deformable_registration import DeformableRegistration
from MeshProperties import MeshDataToPolyData
# Current directory
cwd = os.getcwd()
sys.path.insert(0, cwd + '\\BCPD\\win')
from ExeBCPD import ExecuteBCPD
import time
red = [1,0,0]
blue = [0,0,1]
green = [0,1,0]

def trimesh_to_open3d(mesh_trimesh):
    import open3d as o3d
    import numpy as np
    
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh_trimesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh_trimesh.faces)
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh

def CustomCPD(source, target, instance, SSMSettings, Stiffness, landmarks_positions, center = False, plots = False, MeshData = None, ref = False):
    """
    

    Returns
    -------
    None.

    """
    
    # Finds correspondence using CPD
    # compute cpd registration
    target_cp = copy.deepcopy(target)
    source_cp = copy.deepcopy(source)
    
    CPDSettings = SSMSettings['CPDSettings']
    
    # Defines the cpd algorithm to use. If it equals compare, both are considered
    CpdAlgorithm = CPDSettings['Algorithm']
    
    if (CPDSettings['Normalization'] == 1):
        target_cpp = copy.deepcopy(np.asarray(target.points))
        source_cpp = copy.deepcopy(np.asarray(source.points))
        
        # Normalization of the data
        source_cpp, target_cpp, normal = CPDNormalization(source_cpp, target_cpp)
    else:    
        target_cpp = copy.deepcopy(np.asarray(target.points))
        source_cpp = copy.deepcopy(np.asarray(source.points))
    
    # Selects different algorithms
    if (CpdAlgorithm == 'ProbReg' or CpdAlgorithm == 'Compare'):
        acpd = cpd.NonRigidCPD(source_cpp, beta=2, lmd=2, use_cuda=False)
        tf_param, sigma, _ = acpd.registration(target_cpp,
                                               CPDSettings['w'], 
                                               CPDSettings['MaxIter'],
                                               CPDSettings['MaxError'])    

        # Computes the resultant cloud point
        result = copy.deepcopy(source)
        result.points = o3d.cpu.pybind.utility.Vector3dVector(copy.deepcopy(source_cpp))
        result.points = tf_param.transform(result.points)
        
        # Auxilliary function  to estimate the probability map   
        cv = lambda x: np.asarray(x.points if isinstance(x, o3d.geometry.PointCloud) else x)
        ProbabilityCor = ProbabilityCorrespondenceCPD(cv(result), target_cpp, sigma, CPDSettings['w'])
        CPDCorrespondences = np.argmax(ProbabilityCor, axis = 1)
        CPDMaxProbabilities = np.max(ProbabilityCor, axis = 1)
        # Checks results - apparently, the CPDCorrespondences contain the indices 
        # of the target that correspond to each point of the source
        #check_pointmatching_result(result, target_cp, 1, CPDCorrespondences[1])
        
        if plots:
            # draw result
            source_cp.paint_uniform_color([1, 0, 0])
            target_cp.paint_uniform_color([0, 1, 0])
            result.paint_uniform_color([0, 0, 1])
            final = [source_cp, target_cp, result]
            o3d.visualization.draw_geometries(final)
            
            PolyDatafinal = MeshDataToPolyData(final)
            p = pv.Plotter()
            p.add_mesh(PolyDatafinal,
                       scalars = final['Appearance'],
                       interpolate_before_map = False, opacity = 0.5, cmap = 'jet')
            p.export_html(f'Results\\cpd_{instance}.html')
            p.show()
        
        # Denormalization of the data if it was normalized
        if (CPDSettings['Normalization'] == 1):
            input('Denormalization still not developed')
        # For visualization purposes
        result_probreg = copy.deepcopy(result) # for visualization purposes
    
    """
    Approach using PyPCD
    """
    
    CheckEvolution = 0
    if (CpdAlgorithm == 'PyCPD' or CpdAlgorithm == 'Compare'):
        
        t_pycpd_i = time.time()
        
        if (CPDSettings['Normalization'] == 1):
            alpha = 3
            beta = 0.6
            
            tolerance = 1e-6
        else:
            # If the data are not normalized, the following parameters work best.
            # These parameters were also tested using the original algorithm of 
            # Myronenko and they provided good results.
            #beta = np.sqrt(1/source_cpp.shape[0]) * 0.5 # Based on the normalization of the original algorithm (but of course without any normalization)
            #alpha = source_cpp.shape[0] / beta
            """
            # Stiffness parameter
            # The stiffness parameter was implemented to change the rigidity of the transformation.
            # If the stiffness is 1, the source will adjust almost perfectly to the target.
            # As the stiffness increases, the adjustment becomes more constrained.
            """
            # These first beta and alpha yielded good results overall but for
            # some instances, they failed somehow. The new parameters, below,
            # seem more robust.
            #beta = np.sqrt(1/source_cpp.shape[0]) * 0.5 # Based on the normalization of the original algorithm (but of course without any normalization)
            #alpha = Stiffness * source_cpp.shape[0] / beta
            beta = np.sqrt(1/source_cpp.shape[0]) * 0.25 # Based on the normalization of the original algorithm (but of course without any normalization)
            alpha = Stiffness * 4 * source_cpp.shape[0] / beta
            tolerance = 1e-10
        
        
        # Creates the structure for the registration
        reg = DeformableRegistration(**{'X': target_cpp, 'Y': source_cpp,
                                        'max_iterations': CPDSettings['MaxIter'], 
                                        'tolerance': tolerance,
                                        'alpha': alpha,
                                        'beta': beta})
        
        
    
        if (CheckEvolution == 1):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            callback = partial(PyCPDVisualization, ax=ax)
        
            output = reg.register(callback)
            plt.show()
        else:
            output = reg.register()
        
        t_pycpd = time.time() - t_pycpd_i
    
        # Denormalization of the data if it was normalized
        if (CPDSettings['Normalization'] == 1):
            tf_points = output[0] * normal['targetscale'] + normal['targetd']
        else:
            tf_points = output[0]
        
        if SSMSettings["Landmarks"] > 0:
            for pos in landmarks_positions:
                tf_points[pos, :] = target_cpp[pos, :] # muda diretamente de acordo com a posição dos landmarks

        if center:
            tf_points[-1, :] = target_cpp[-1, :]
            
        # Computes the resultant cloud point
        result = copy.deepcopy(o3d.geometry.PointCloud(o3d.cpu.pybind.utility.Vector3dVector(tf_points)))
        
        # Auxilliary function  to estimate the probability map   
        ProbabilityCor = output[1][2]
        CPDCorrespondences = np.argmax(ProbabilityCor, axis = 1)
        CPDMaxProbabilities = np.max(ProbabilityCor, axis = 1)
        
        # For now, the variable tf_param will be defined as []
        tf_param = []
        
        # For visualization purposes
        result_pycpd = copy.deepcopy(result) # for visualization purposes
        
        
        t_bcpd_i = time.time()
        tf_points_bcpd, CPDCorrespondences_bcpd = BCPD(target_cpp, source_cpp, CPDSettings, SSMSettings["Landmarks"], landmarks_positions, center)
        
        print(CPDCorrespondences_bcpd)
        
        t_bcpd = time.time() - t_bcpd_i
        
        result_bcpd = copy.deepcopy(o3d.geometry.PointCloud(o3d.cpu.pybind.utility.Vector3dVector(tf_points_bcpd)))
        
        
        target_cp.paint_uniform_color([1, 0, 0])
        result_bcpd.paint_uniform_color([0, 1, 0])
        result_pycpd.paint_uniform_color([0, 0, 1])
        final = [target_cp, result_bcpd]
        o3d.visualization.draw_geometries([target_cp, result_bcpd, result_pycpd])

        print(t_bcpd, t_pycpd)
        
        #TargetGeom = GetGeom(o3d.geometry.PointCloud(o3d.cpu.pybind.utility.Vector3dVector(target_cpp)))
        #BCPDGeom = GetGeom(result_bcpd)
        #PyCPD = GetGeom(result_pycpd)
        
        CompareAlgorithms(MeshData, result_bcpd, result_pycpd)
        
        print(MeshData)

    elif CpdAlgorithm == 'BCPD':
        
        tf_points, CPDCorrespondences = BCPD(target_cpp, source_cpp, CPDSettings, SSMSettings["Landmarks"], landmarks_positions, center)
        
        result = copy.deepcopy(o3d.geometry.PointCloud(o3d.cpu.pybind.utility.Vector3dVector(tf_points)))
        
        tf_param, CPDMaxProbabilities, ProbabilityCor = None, None, None
        
        '''
        target_cp.paint_uniform_color([1, 0, 0])
        result.paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries([target_cp, result])'''

    if (CpdAlgorithm == 'Compare'):
        
        if plots:
            target_cp.paint_uniform_color([1, 0, 0])
            result_probreg.paint_uniform_color([0, 1, 0])
            result_pycpd.paint_uniform_color([0, 0, 1])
            final = [target_cp, result_probreg, result_pycpd]
            o3d.visualization.draw_geometries(final)
            
            PolyDatafinal = MeshDataToPolyData(final)
            p = pv.Plotter()
            p.add_mesh(PolyDatafinal,
                       scalars = final['Appearance'],
                       interpolate_before_map = False, opacity = 0.5, cmap = 'jet')
            p.export_html(f'Results\\cpd_{instance}.html')
            p.show()
    
    if ref:
        
        result = o3d.geometry.PointCloud(o3d.cpu.pybind.utility.Vector3dVector(target_cpp))
        
        if center and SSMSettings["Landmarks"] > 0:
            CPDCorrespondences = np.asarray(range(SSMSettings['SamplingPoints'] + SSMSettings["Landmarks"] + 1)) # Caso haja landmarks e center fazer a correspondencia incluindo esses pontos também
        elif center and SSMSettings["Landmarks"] == 0:
            CPDCorrespondences = np.asarray(range(SSMSettings['SamplingPoints'] + 1)) # caso só haja center
        elif not center and SSMSettings["Landmarks"] > 0:
            CPDCorrespondences = np.asarray(range(SSMSettings['SamplingPoints'] + SSMSettings["Landmarks"])) # Caso só haja landmarks
        else: CPDCorrespondences = np.asarray(range(SSMSettings['SamplingPoints']))
        
    print(CPDCorrespondences)
    print(len(CPDCorrespondences))
    print( f'Matching = {round(len(np.unique(CPDCorrespondences))/len(CPDCorrespondences)*100,2)} %')
    
    return result, CPDCorrespondences, tf_param, CPDMaxProbabilities, ProbabilityCor

def GetGeom(point_cloud):
    
    Geom = {}
    
    Geom['Mesh'] = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, 0.03)
    Geom['Pcd'] = copy.deepcopy(point_cloud)
    
    
    Geom['Mesh'].compute_triangle_normals()
    Geom['Mesh'].compute_vertex_normals()
    
    o3d.visualization.draw_geometries([Geom['Mesh']])
    
    
    return Geom

    
def GetPolyData(OriGeom: dict, iters):
        
        
    # Compute original geometry poly data
    PolyDataOri = MeshDataToPolyData(OriGeom)
    PolyDataOri = PolyDataOri.smooth(n_iter = iters)
    
    
    
    return PolyDataOri

def CPDNormalization(source, target):
    """
    

    Parameters
    ----------
    source : TYPE
        DESCRIPTION.
    target : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    # Number of points in the source and target point clouds
    NSource = source.shape[0]
    NTarget = target.shape[0]
    
    # Defines the structure that allows normalizing and denormalizing the point clouds
    normal = {'sourced': np.mean(source, axis = 0),
              'targetd': np.mean(target, axis = 0),
              'sourcescale': 0,
              'targetscale': 0
              }
    
    # Takes the mean from the point clouds
    sourcenorm = source - normal['sourced']
    targetnorm = target - normal['targetd']
    
    # Computes the scaling factors
    normal['sourcescale'] = np.sqrt(np.sum(sourcenorm**2) / NSource)
    normal['targetscale'] = np.sqrt(np.sum(targetnorm**2) / NTarget)
    
    # Computes the final normalized data
    sourcenorm /= normal['sourcescale']
    targetnorm /= normal['targetscale']
    
    return sourcenorm, targetnorm, normal

def ProbabilityCorrespondenceCPD(t_source, target, sigma2, w):
    """
    This function was taken from the probreg package. It belongs to the
    Coherent Point Drift class:
        estep_res = cpdcor.expectation_step(t_source, cv(target_cp), sigma, 0)
    Since the original function does not output the probability of correspondence,
    a copy was made to output it.
    """
    
    pmat = np.stack([np.sum(np.square(target - ts), axis=1) for ts in t_source])
    pmat = np.exp(-pmat / (2.0 * sigma2))

    c = (2.0 * np.pi * sigma2) ** (t_source.shape[1] * 0.5)
    c *= (w / (1.0 - w)) * (t_source.shape[0] / target.shape[0])
    den = np.sum(pmat, axis=0)
    den[den==0] = np.finfo(np.float32).eps
    den += c

    pmat  = np.divide(pmat, den)
    
    return pmat

def PyCPDVisualization(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:, 0],  X[:, 1], X[:, 2], color='red', label='Target')
    ax.scatter(Y[:, 0],  Y[:, 1], Y[:, 2], color='blue', label='Source')
    ax.text2D(0.87, 0.92, 'Iteration: {:d}'.format(
        iteration), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)
    
def BCPD(target, source, CPDSettings, n_landmarks, landmarks_positions, center = False):
    
    beta = CPDSettings['Beta']
    lamda = CPDSettings['Lambda']
    invert = CPDSettings['Invert']
    
    # Substitui NaN por 0
    source = np.nan_to_num(source, nan=0.0)
    target = np.nan_to_num(target, nan=0.0)
    
    np.savetxt(cwd + '\\BCPD\\win\\Source.txt', source, delimiter=',')
    np.savetxt(cwd + '\\BCPD\\win\\Target.txt', target, delimiter=',')
    
    NPoints = len(source)
    
    ExecuteBCPD(cwd, 'Target.txt', 'Source.txt', beta, lamda, invert)
    
    result = np.loadtxt(cwd + '\\BCPD\\win\\output_y.txt')
    
    if n_landmarks > 0:
        for pos in landmarks_positions:
            result[pos, :] = target[pos]

    if center:
        result[-1, :] = target[-1, :]
        
    correspondence_raw = np.loadtxt(cwd + '\\BCPD\\win\\output_e.txt', skiprows=1)
    
    correspondence = correspondence_raw[:, 1].astype(int)

    # Corrigir os índices dos landmarks e centro manualmente
    if n_landmarks > 0:
        correspondence_full = np.zeros(len(target), dtype = int)
        correspondence_full[:len(correspondence)] = correspondence
        for pos in landmarks_positions:
            correspondence[pos] = pos + 1 
            # mais 1 pq as posições dadas no array que se quer já são zero based, e depois no return já se faz -1 para tudo

    if center:
        correspondence[-1] = len(target) - 1

    #correspondence = getCorrespondence(correspondence_raw, NPoints)

    return result, correspondence-1
    
def getCorrespondence(correspondence_raw, NPoints):
    
    correspondence_n = correspondence_raw[:, 0].astype(int) - 1
    correspondence_m = correspondence_raw[:, 1].astype(int) - 1 
    correspondence_prob = correspondence_raw[:, 2]
    
    correspondence = np.zeros(NPoints, dtype = int)
    
    for i in range(NPoints):
        
        sub_n = correspondence_n[correspondence_m == i]
        sub_prob = correspondence_prob[correspondence_m == i]
        
        if len(sub_n) == 0:
            
            print(i)
            
            raise Exception('Error in correspondence')
            
        else:
        
            arg_max = np.argmax(sub_prob)
            
            correspondence[i] = sub_n[arg_max]
            
    print(f'Matching = {round(len(np.unique(correspondence))/len(correspondence)*100,2)} %')
    
    return correspondence
        
def readCenter2(filename):
    file = open(filename, "r") # abrir o ficheiro
    
    previous_line = ''
    
    data = []
    for line in file.readlines():
        separated_line = line.split(',')
        
        try:
            point = (float(separated_line[0]),
                     float(separated_line[1]),
                     float(separated_line[2]))
            caseNumber = int(previous_line)
            
            data.append((point, caseNumber))
        
        except:
            
            pass
        
        finally:
        
            previous_line = line
            
    file.close()
    return data

def readRadius(filename):
    file = open(filename, "r") # abrir o ficheiro
    
    previous_line = ''
    
    data = []
    for line in file.readlines():
        separated_line = line.split(',')
        
        try:
            radius = float(separated_line[3])
            caseNumber = int(previous_line)
            
            data.append((radius, caseNumber))
        
        except:
            
            pass
        
        finally:
        
            previous_line = line
            
    file.close()
    return data

def readRes(filename):
    file = open(filename, "r") # abrir o ficheiro
    
    previous_line = ''
    
    data = []
    for line in file.readlines():
        separated_line = line.split(',')
        
        try:
            radius = float(separated_line[0])
            caseNumber = int(previous_line)
            
            data.append((radius, caseNumber))
        
        except:
            
            pass
        
        finally:
        
            previous_line = line
            
    file.close()
    return data
    