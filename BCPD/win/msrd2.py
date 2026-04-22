# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:08:28 2024

@author: andre
"""

import numpy as np

corr = np.loadtxt("output_e-origbet0.1.txt", skiprows=1, usecols=(0,1))
# corr = np.loadtxt("C:/Users/AndreJoao/Desktop/interpolações/Geodesic/Down 15000/output_e-k350.txt", skiprows=1, usecols=(0,1))

target_orig = np.loadtxt("point_cloud_0.txt", delimiter=",")
source= np.loadtxt("point_cloud_2.txt", delimiter=",")
target= np.loadtxt("output_x.txt")
deformada= np.loadtxt("output_y.txt")

# deformada= np.loadtxt("C:/Users/AndreJoao/Desktop/interpolações/Geodesic/Down 15000/output_y-k350.txt")
deformada_inter=np.loadtxt("C:/Users/AndreJoao/Desktop/interpolações/Bayesian/Down-15000/output_y.interpolated-k350.txt")

def constroi_matriz(initial, truth,deformed,cor):
    X=[] #pontos do target/ground-truth
    B=[] #pontos da deformada
    orig=[] #pontos iniciais 
    indexs=[]
    for i in range(len(cor)):
        index=int(cor[i][1])
        indexs+=[index]
        orig+=[initial[index-1]]
        X+=[truth[i]]
        B+=[deformed[index-1]]
    return X,B,orig

X, B, orig= constroi_matriz(source, target, deformada, corr)

def menos(matrizA,matrizB):
    resultado=[]
    for i in range(len(matrizA)):
        resultado+=[matrizA[i]-matrizB[i]]
    return np.array(resultado)

def r(X,B,cor):
    X_B=menos(X, B)
    numerador=0
    normas=[]
    for i in range(len(X_B)):
        norma=np.linalg.norm(X_B[i])
        normas+=[norma]
        numerador+=(norma)**2
    denominador=len(cor)
    final=(numerador/denominador)**(1/2)
    return final

def accuracy(source, target, deformada, corr):
    # modo=input("Avaliar interpolação (I), downsample (D) ou nenhum (N): ")
    # if modo=="N":
    #     X,B,orig=constroi_matriz(source, target, deformada, corr)
    #     fracao=r(X,B,corr)/r(X,orig,corr)
    # elif modo=="I":
    #     X,B,orig=constroi_matriz(source, target, deformada, corr)
    #     fracao=r(target,B,corr)
    # elif modo=="D":
    X,B,orig=constroi_matriz(source, target, deformada, corr)
    fracao=r(X,B,corr)/r(X,orig,corr)
    # else:
    #     print("Modo inválido!")
    #     return None
    return fracao

valor=accuracy(source, target, deformada, corr)
print("valor para normal: ", valor)

inter=accuracy(source, target, deformada_inter, corr)
print("valor interpolado: ", inter)



