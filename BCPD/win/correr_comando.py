# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 21:37:09 2024

@author: andre
"""
import numpy as np
import subprocess

omg = '0.0'
bet = '10'
lmd = '100'
gma = '0.1'
zet = '0.0'
K = '200'
J = '300'
c = '1e-3'
n = '100'
modo1 = '-G geodesic,0.1,10,0.15'
modo2 = '-DB,3000,0.1'

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
    '-sA',
    modo1,
    # modo2
]

# Join the command list into a single string
command_str = ' '.join(command)


# Execute the command
try:
    subprocess.run(command, check=True)
except subprocess.CalledProcessError as e:
    print("Command failed with error:", e)

# Read and print the content of a file
file_path = "output_comptime.txt"  
with open(file_path, 'r') as file:
    print("Time data:")
    print(file.read())
    
    
file_path = "output_info.txt" 
with open(file_path, 'r') as file:
    print("Output Info:")
    print(file.read())
    



    

    