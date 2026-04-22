# -*- coding: utf-8 -*-
"""
Created by Carlos Quental. Adapted by Augusto
Only does SSM (not SSAM or SDM) 
and uses one of the training geomtries as reference

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Statistical auxilliary functions
def CovFunction(Data, KernelType):
    """
    

    Parameters
    ----------
    Data : TYPE
        DESCRIPTION.
    KernelType : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    if (KernelType == 'Empirical'):
        kssm = np.cov(Data)
    else:
        kssm = []
        print('CovFunction: Kernel type not defined')
        input('Press any key to continue...')
        
    return kssm

def CheckSymmetry(a, rtol=1e-06, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

"""
 PCA analysis - computation of eigenvectors and eigenvalues of the covariance matrix
"""
def CustomPCA(X, Type):
    """
    

    Parameters
    ----------
    X : TYPE
        Matrix with displacement. Each line represents a different coordinate.
        Each column represents a different geometry (observation).

    Returns
    -------
    None.

    """
    
    if (Type == 'Empirical'):
        
        # Computes the covariance matrix
        CovMat = CovFunction(X, Type)
    
        # Check if the matrix is symmetric. If it is not, it issues an error
        if (CheckSymmetry(CovMat) == False):
            print('CustomPCA: Assumption of a symmetric matrix not valid')
            input('Press any key to continue')
            eigvalues = []
            eigvectors = []
        else:
            eigvalues, eigvectors = np.linalg.eigh(CovMat)
        
            # The data are output is ascending order, but it is more relevant in descending order. 
            # For that
            eigvalues = eigvalues[::-1]
            eigvectors = eigvectors[:,::-1]    
    elif (Type == 'PCA'):
        
        # Number of components equals number of models - 1
        pca = PCA(svd_solver='full', whiten=True)
        pca.fit(X.T) # The X matrix is transposed because the lines
        # must be the observations and the columns the features (coordinates)
        
        new_coords = pca.transform(X.T)
        
        # Definition of the eigen values and eigenvectors
        eigvalues = np.asarray(pca.explained_variance_)
        eigvectors = np.asarray(pca.components_).T # Because PCA outputs the 
        # eigenvectors as lines, it is transposed so that they become the 
        # columns
    
        
    else:
        print('Type of PCA not defined')
        input('Press any key to continue...')
        
    return eigvalues, eigvectors, pca, new_coords

""" 
Analysis of the variance
"""
def VarianceAnalysis(EigVal, Thresh, PlotFig = False):
    """
    

    Parameters
    ----------
    EigVal : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    # Number of eigenvalues
    NEig = len(EigVal)
    
    # Sum of all the eigenvalues
    EigSum = np.sum(EigVal)
    
    # Cummulative variance
    CumVar = 0
    for i in range(NEig):
        
        # Updates the cummulative variance
        CumVar += EigVal[i] / EigSum
        
        # If the CumVar is greater or equal than the threshold
        # we have found the number of relevant eigenvalues
        if (CumVar >= Thresh):
            RelevEig = i + 1
            break
            
    # Plots the results if asked
    if (PlotFig):
        # Number of eigenvalues
        #NEig = RelevEig + 10
    
        # Individual and cumulative variance
        IndVar = np.zeros(NEig)
        CumVar = np.zeros(NEig)
        for i in range(NEig):
            IndVar[i] = EigVal[i] / EigSum
            if (i == 0):
                CumVar[i] = IndVar[i]
            else:
                CumVar[i] = CumVar[i-1] + IndVar[i]
    
        # Plots the data
        plt.figure()          
        ind = np.arange(NEig)
        plt.ylim(0.0, 1.0)
        plt.bar(ind, IndVar, color='r')
        plt.ylabel('Individual variance')      
        
        x = np.linspace(0, NEig, NEig)
        axes2 = plt.twinx()
        axes2.plot(x, CumVar, color='k')
        axes2.set_ylim(0, 1)
        axes2.set_ylabel('Cummulative variance')
        plt.savefig('Results\\Cumulative_variance.png', dpi = 300)
        plt.show()
    
    return RelevEig

