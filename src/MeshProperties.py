# -*- coding: utf-8 -*-
"""
Created by Carlos Quental. Adapted by Augusto
Only does SSM (not SSAM or SDM) 
and uses one of the training geomtries as reference

"""
import numpy as np
import copy
import pyvista as pv
import open3d as o3d

def MeshDataToPolyData(Mesh):
    """
    This function takes elements and nodal coordinates to create a pyvista
    mesh (polydata)

    Parameters
    ----------
    StatApp : Statistical model being applied
        DESCRIPTION.
    Mesh : Dictionary containing mesh data
        DESCRIPTION.

    Returns
    -------
    None.

    """

    # For this condition, the mesh structure already contains the faces
    Faces = np.asarray(Mesh['Mesh'].triangles)
    
    # Nodes
    nodes = np.asarray(Mesh['Mesh'].vertices)
    
    NFaces = Faces.shape[0]
    pyfaces = np.hstack((3 * np.ones((NFaces, 1)), Faces)).astype(np.int32())
    # Creates a polydata mesh using pyvista. The faces must be flatten
    pysurf = pv.PolyData(nodes, pyfaces.flatten())
    
    return pysurf

def UpdatesMeshData(StatApp, UpdData, orisize):
    """
    This function updates the data of meshes and point clouds according to the
    data created from a statistical shape model
    
    Parameters
    ----------
    meshdata : TYPE
        DESCRIPTION.
    meshind : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    # Number of points in the data. If the data contains only coordinates,
    # then we just need to divide the data by 3
    NPoints = int(len(UpdData) / StatApp['Dof'])
    
    if orisize:
        # Updated vertices
        UpdVertices = (copy.deepcopy(np.asarray(StatApp['MVShape']['Pcd'].points)) + UpdData.reshape((NPoints, 3))) * StatApp['MVShape']['CSize']
        
    else:
        # Updated vertices
        UpdVertices = copy.deepcopy(np.asarray(StatApp['MVShape']['Pcd'].points)) + UpdData.reshape((NPoints, 3))

    # Makes the output mesh equal to the MV Shape 
    MeshOut = {'Mesh': copy.deepcopy(StatApp['MVShape']['Mesh']),
               'Pcd': copy.deepcopy(StatApp['MVShape']['Pcd']),
               'Appearance': np.ones((NPoints, 1))}
    
    # Updates its vertices
    MeshOut['Mesh'].vertices = o3d.cpu.pybind.utility.Vector3dVector(UpdVertices)
    
    # Updates the normals
    MeshOut['Mesh'].compute_triangle_normals()
    MeshOut['Mesh'].compute_vertex_normals()

    # Updates the point cloud
    MeshOut['Pcd'].points = o3d.cpu.pybind.utility.Vector3dVector(UpdVertices)
    
    # Updates the normals
    MeshOut['Pcd'].estimate_normals()

    return MeshOut

def SurfaceMeshSampling(mesh, target_points):
    """
    This function recieves a mesh and performs decimation or subdivision (slowly) until a 
    given number of points is achieved. The algorithm must be iterative because
    what we control is the number of target triangles
    

    Parameters
    ----------
    mesh : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    #Estimate number of triangles to have 100+10000 vertex using Euler
    numFaces = int(round(2.5 * target_points))

    # Number of current vertices
    NCurVertices = np.asarray(mesh.vertices).shape[0]
    
    # Makes a copy of the mesh
    mesh_cp = copy.deepcopy(mesh) 
    
    
    # Subdivision --> mesh with more points
    if (NCurVertices < target_points):
        
        while (NCurVertices < target_points):
            
            # Subdivides the mesh
            mesh_cp = mesh_cp.subdivide_loop()
            
            # Updates the number of vertices
            NCurVertices = np.asarray(mesh_cp.vertices).shape[0]
    
    # Decimation --> mesh with less points
    if (NCurVertices > target_points):
        
        while (NCurVertices > target_points):
            
            # Decimates the mesh
            mesh_cp = mesh_cp.simplify_quadric_decimation(numFaces)
            
            # Updates the number of vertices
            NCurVertices = np.asarray(mesh_cp.vertices).shape[0]
            
            #Refine our estimation to slowly converge to TARGET vertex number
            numFaces -= (NCurVertices - target_points)
        
    if (NCurVertices != target_points):
        print('Mesh sampling to the target number of points did not succeed\n.')
        input('Press any key to continue or stop the simulation\n')
        
    return mesh_cp




