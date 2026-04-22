# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:07:59 2024

@author: andre
"""
from scipy.spatial import cKDTree
import numpy as np

corr = np.loadtxt("output_e.txt", skiprows=1, usecols=(0,1))

target = np.loadtxt("point_cloud_0.txt", delimiter=",")
source= np.loadtxt("point_cloud_2.txt", delimiter=",")
deformada= np.loadtxt("output_y.interpolated.txt")

def distancia3D (point_1, point_2):
    return np.linalg.norm(point_2-point_1)

def closest_points_indices(point_cloud1, point_cloud2):
    tree = cKDTree(point_cloud2)
    distances, indices = tree.query(point_cloud1, k=1)
    final=[]
    for i in range(len(indices)):
        final+=[[i,indices[i]]]
    return final


def RMSD (target, deformed, corr):
    soma=0
    pontos=len(corr)
    for i in range(len(corr)):
        index=int(corr[i][1])
        norma=distancia3D(target[i], deformed[index-1])
        soma+=norma**2
    media=soma/pontos
    rmsd=np.sqrt(media)
    return rmsd