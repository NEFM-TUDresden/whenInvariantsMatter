'''
This is a supplementary material to the paper

"The role of the invariants in neural network-based modelling of incompressible hyperelasticity"
by Franz Dammass, Karl A. Kalina and Markus Kästner.

The code is provided under the CC BY-SA 4.0 license, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
When you find this code useful, please cite the corresponding paper.
'''


import numpy as np

def lamGamToF(lamGam : np.array, loadCase : str) -> np.array:
    ''' generate full deformation gradient tensor for given simple loadCase according to the "loads" lamGam
        -> lamGam: np.array of shape (nTest,)'''

    nTest = lamGam.shape[0]
    Fs = np.zeros((nTest,3,3))

    if not loadCase == 'simpleShear':
        lambda1s = lamGam

    if loadCase == 'uniAx' or loadCase == 'uniAxCompr': # uniaxial tension
        lambda2s = 1/(np.sqrt(lambda1s))
        lambda3s = lambda2s
    elif loadCase == 'eqBiAx':
        lambda2s = lambda1s
        lambda3s = 1/(np.power(lambda1s,2))
    elif loadCase == 'pureShear':
        lambda2s = np.zeros(lambda1s.shape)+1.0
        lambda3s = 1/lambda1s

    elif loadCase == 'simpleShear':
        gammas = lamGam

    if loadCase != 'simpleShear':
        for iTest in range(nTest):		
            Fs[iTest,:,:] = np.diag((lambda1s[iTest],lambda2s[iTest],lambda3s[iTest]),0)
    else:
        Fs[:,0,1]=gammas
        Fs[:,0,0]=1
        Fs[:,1,1]=1
        Fs[:,2,2]=1

    return Fs
