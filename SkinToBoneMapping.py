# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 09:53:08 2026

@author: CQuental
"""

import numpy as np

" Auxiliary functions"
# Function to apply the regression equations and compute bone landmarks
def apply_equation(equation_vars, equation_funcs, coord, id_row):
    inputs = {}

    for var in equation_vars[coord]:
        inputs[var] = id_row[var]
    
    result = equation_funcs[coord](**inputs)
    return result

def get_bony_landmark_coordinates(name, row):
    return tuple(apply_equation(f"{axis}_{name}_bony", row) for axis in ['x', 'y', 'z'])

# Function to compute the predicts needed for the regression equations.
def organize_predictors(equation_vars, localSkinCoord, SubjectVar):
    
    # Begins the output dictionary as empty and adds predictors
    Predictors = {}

    # Goes through all equations in equation_vars
    for eq in equation_vars:
        
        for pred in equation_vars[eq]:
            
            # Adds the predictor if it is not yet in the Predictors dictionary
            if (pred == 'BMI'):
                Predictors['BMI'] = SubjectVar['Weight'] / SubjectVar['Height']**2
            elif (pred == 'age'):
                Predictors['age'] = SubjectVar['Age']
            elif (pred == 'sex'):
                Predictors['sex'] = SubjectVar['Sex']
            elif (pred == 'height'):
                Predictors['height'] = SubjectVar['Height']
            elif (pred == 'weight'):
                Predictors['weight'] = SubjectVar['Weight']
            elif ('x_' in pred):
                # Gets the landmark name
                _, landmarkname, _ = pred.split("_", 2)
                
                # Saves the data to the dictionary
                Predictors[pred] = localSkinCoord[landmarkname]['xcoord']
            elif ('y_' in pred):
                # Gets the landmark name
                _, landmarkname, _ = pred.split("_", 2)

                # Saves the data to the dictionary
                Predictors[pred] = localSkinCoord[landmarkname]['ycoord']                
            elif ('z_' in pred):
                # Gets the landmark name
                _, landmarkname, _ = pred.split("_", 2)
                    
                # Saves the data to the dictionary
                Predictors[pred] = localSkinCoord[landmarkname]['zcoord']
                
            elif ('d_' in pred):
                
                # Gets the landmark names
                _, landmarkname1, landmarkname2, _ = pred.split("_", 3)
                
                # Saves the distance between the landmarks 
                Predictors[pred] = np.linalg.norm(localSkinCoord[landmarkname1] - localSkinCoord[landmarkname2])
                
            else:
                print('Predictor not found: ' + pred)
                input('Press any key to continue...')
                
    
    # Returns the output
    return Predictors