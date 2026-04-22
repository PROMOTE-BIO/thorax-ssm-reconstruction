# -*- coding: utf-8 -*-
"""
Created on Fri May 31 22:53:28 2024

@author: AndreJoao
"""

# Define the file paths
input_file_path = 'source_rig-3000.txt'
output_file_path = 'source_rig2-3000.txt'

# Open the input file and output file
with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
    for line in input_file:
        # Replace tabs with commas
        comma_separated_line = line.replace('\t', ',')
        # Write to the output file
        output_file.write(comma_separated_line)

print(f"Comma-separated coordinates have been saved to {output_file_path}")

Optimization terminated successfully.
         Current function value: 0.048900
         Iterations: 65
         Function evaluations: 153
Optimal solution: [2.33280127e-01 1.18261382e+02 2.74210679e+02]
Function value at optimal solution: 0.0489004420193417

#3000 - geodesico
Global optimization result: [3.14163305e-01 1.34342507e+00 3.42213351e+02] with RMSD: 0.045746212829989155

Local refinement result: [3.29503275e-01 1.26976201e+00 3.49110103e+02] with RMSD: 0.04533166302156806

Global optimization result: [2.61802566e-01 1.36548435e+01 3.45170938e+02] with RMSD: 0.04579399977051208

Local refinement result: [2.57062957e-01 1.42840316e+01 3.56981725e+02] with RMSD: 0.04573849513946286


#6000 - geodesico
Global optimization result: [  0.35832177  11.68273382 324.4827132 ] with RMSD: 0.03643389195313429

Local refinement result: [2.97526209e-01 1.23415762e+01 3.75575586e+02] with RMSD: 0.03592022938472363

Global optimization result: [2.99379945e-01 3.37830568e+01 3.47354004e+02] with RMSD: 0.03646955653663981

Global optimization result: [2.80554810e-01 1.73825265e+01 3.45368285e+02] with RMSD: 0.03646399271085796
DE iterations: 17
DE function evaluations: 866

Local refinement result: [3.07830138e-01 1.66100511e+01 3.47941366e+02] with RMSD: 0.03643916001090498
NM iterations: 167
NM function evaluations: 600


#3000 bayesian
Global optimization result: [2.62339828e-01 1.09815979e+01 3.43873791e+02] with RMSD: 0.04577267950557407
DE iterations: 18
DE function evaluations: 947
Local refinement result: [2.60637796e-01 1.10517245e+01 3.44263563e+02] with RMSD: 0.045745477593138476
NM iterations: 72
NM function evaluations: 154


#6000 bayesian
Global optimization result: [  0.37420246   3.92838365 347.10114744] with RMSD: 0.0365191345033966
DE iterations: 19
DE function evaluations: 988

Local refinement result: [  0.36971504   3.97055229 347.3305743 ] with RMSD: 0.035949695543627244
NM iterations: 169
NM function evaluations: 600

#CPD 3000
Local refinement result: [2.2678403  1.15432807] with RMSD: 0.06716338978568244
NM iterations: 50
NM function evaluations: 111

Local refinement result: [2.2678403  1.15432807] with RMSD: 0.06716338978568244
NM iterations: 50
NM function evaluations: 111

Local refinement result: [4.36742422e+00 1.00000000e-06] with RMSD: 0.00921616724221631
NM iterations: 38
NM function evaluations: 86

