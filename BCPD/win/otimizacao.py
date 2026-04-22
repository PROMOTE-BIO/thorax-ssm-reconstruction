# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:47:07 2024

@author: AndreJoao
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 21:37:09 2024

@author: andre
"""

import subprocess
from avaliacoes import chamfer_distance, hausdorff_distance, rmsd
import numpy as np
from scipy.optimize import minimize, differential_evolution

# Set a random seed for reproducibility
np.random.seed(42)

def algoritmo (vetor): #vetor com [beta, lambda, K]
    omg = '0.0'
    bet = str(vetor[0])
    lmd = str(vetor[1])
    gma = '0.1'
    zet = '0'
    K = str(vetor[2])
    J = '300'
    c = '1e-6'
    n = '100'
    modo1 = '-G geodesic,0.1,10,0.15'
    modo2 = '-DB,15000,0.15'
    
    # Define the command and its arguments
    command = [
        './bcpd',
        '-x', 'target_downsampled_6000.txt',
        '-y', 'source_rig-6000.txt',
        '-w' + omg,
        '-l' + lmd,
        '-b' + bet,
        '-g' + gma,
        '-z' + zet,
        '-J' + J,
        '-K' + K,
        '-n' + n,
        '-c' + c,
        '-p',
        # '-svexYP',
        # modo1,
        # modo2
    ]
        
    command+=['-sA'] #Determine the outputs we want, if we want all just put -A
    
    # Join the command list into a single string
    command_str = ' '.join(command)
    
    # Execute the command
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("Command failed with error:", e)
    
    # Read and print the content of a file
    # file_path = "output_comptime.txt"  
    # with open(file_path, 'r') as file:
    #     print("Time data:")
    #     print(file.read())
        
        
    # file_path = "output_info.txt" 
    # with open(file_path, 'r') as file:
    #     print("Output Info:")
    #     print(file.read())
    file_path = "output_y.txt"
    target = np.loadtxt('target_downsampled_6000.txt', delimiter=",")
    # print(len(target))
    resultado = np.loadtxt(file_path)
    # print(len(resultado))
    corr=np.loadtxt("output_e.txt", skiprows=1)
    print("fez função")
    return rmsd(target, resultado,corr)#, -hausdorff_distance(target, resultado)

# i=algoritmo(0.7, 100, 200, True, 3000, True, 'point_cloud_0.txt', 'point_cloud_2.txt')
# print("Chamfer distance:",i)

# Initial point
# x0 = np.array([0.1,1000,200])
  

# Define the bounds
bounds = [(0.1,2.5), (1,5000),(50,350)]  # Bounds for x[0] and x[1]

# Global optimization using Differential Evolution
result_de = differential_evolution(algoritmo, bounds)
print("Global optimization result:", result_de.x, "with RMSD:", result_de.fun)
print("DE iterations:", result_de.nit)
print("DE function evaluations:", result_de.nfev)

# Local refinement using Nelder-Mead starting from the result of Differential Evolution
result_nm = minimize(algoritmo, result_de.x, method='Nelder-Mead', bounds=bounds)
print("Local refinement result:", result_nm.x, "with RMSD:", result_nm.fun)
print("NM iterations:", result_nm.nit)
print("NM function evaluations:", result_nm.nfev)


# Tentar outro algoritmo de otimização para confirmar resultados deste e se os
# resultados são fiáveis para depois realizar os melhores casos BCPD, GCPD e CPD.
# Outras condições iniciais

    