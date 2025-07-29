'''
This is a supplementary material to the paper

"The role of the invariants in neural network-based modelling of incompressible hyperelasticity"
by Franz Dammass, Karl A. Kalina and Markus Kästner.

The code is provided under the CC BY-SA 4.0 license, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
When you find this code useful, please cite the corresponding paper.
'''


import PANNIncompModel as pann
import FNNIncompModel as fnn
from lamGamToF import lamGamToF
import tensorflow as tf
import numpy as np
import os

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

# experimental results of H. Alexander, ‘A constitutive relation for rubber-like materials’, International Journal of Engineering Science, Sep. 1968.
dataAlexander={} #lam	P
dataAlexander['uniAx'] = np.array([
    [1.07769051, 0.24720321],
    [1.20212378, 0.3106125 ],
    [1.39983408, 0.37412539],
    [1.54665325, 0.40692966],
    [1.8110115 , 0.46559806],
    [2.09942129, 0.506223  ],
    [2.28936862, 0.56147712],
    [2.64461717, 0.63933216],
    [2.98752678, 0.70352491],
    [3.485607  , 0.78756471],
    [4.87364959, 0.97593963],
    [5.77107022, 1.26645985],
    [6.35778544, 1.51555073],
    [6.80040967, 1.77724456],
    [7.29547732, 2.10475371],
    [7.67578665, 2.47745913],
    [7.98910645, 2.83843699],
    [8.16834137, 3.10100387],
    [8.36002442, 3.35241575],
    [8.51350968, 3.64605472]])

dataAlexander['eqBiAx']  = np.array([
    [1.24809362, 0.50416417],
    [1.44673925, 0.60702087],
    [1.85027998, 0.69777505],
    [2.22581272, 0.82308704],
    [2.66326693, 1.04966475],
    [2.83699107, 1.13321342],
    [3.14092232, 1.31800894],
    [3.47575698, 1.5545308 ],
    [3.64311697, 1.69145395],
    [4.15110242, 2.19648786],
    [4.53493913, 2.62568509],
    [5.02948915, 3.40236067],
    [5.27896199, 4.11631536],
    [5.50341779, 4.85701447]])



##### choose a model type #####

onlyInvar = 0 # 0 .. \tilde I1 and \tilde I2, 1 .. only \tilde I1, 2 .. only \tilde I2
nnType = 'PANN' # 'PANN' or unconstrained 'FNN' 
# nnType = 'FNN'


##### Load NN model #####
invarSubDirs = ['both','i1','i2']

thisDir= os.path.dirname(os.path.realpath(__file__))
dirName ='Network_Parameters_Alexander' + '/' + invarSubDirs[onlyInvar] + '/' # path to the specific network parameter sub-directory
dirPath = os.path.join(thisDir,dirName)

modelName={'PANN':'FICNN','FNN':'FNN'}

if onlyInvar ==0:
    archString = '-3-3'
else:
    archString = '-2-2'

nnParameterFile = f'alexLC_param_{modelName[nnType]}_{archString}' # network parameter file


if nnType == 'PANN':
    model_manualImpl = pann.PANNIncompModel(dirPath,nnParameterFile,onlyInvar=onlyInvar)
elif nnType == 'FNN':
    # unconstrained FNN
    model_manualImpl = fnn.FNNIncompModel(dirPath,nnParameterFile,onlyInvar=onlyInvar)
else:
     raise(ValueError('An invalid NN type has been chosen!'))

model_manualImpl.printParams()

##### Load experimental data used for visualisation #####
loadCases=['uniAx','eqBiAx','pureShear']
loadCasesExp =['uniAx','eqBiAx']

loadCaseLabels=['UT','BT','PS']
iLoadCase = 0

lamsExp=[]
PK1Exp=[]
FsExp=[] #deformation gradients for exp defo states, needed for NN stress computation

lams_Visu=[]
nTestVisu = int(2e2) # number of data points for model visualisation
FsVisu=[]

invarColors = ['tab:blue','tab:orange','tab:green']

iLoadCase = 0
for loadCase in loadCases:
    if loadCase in loadCasesExp:
        lams_i=dataAlexander[loadCase][:,0]
        PK1_i = dataAlexander[loadCase][:,1]

        lamsExp.append(np.array(lams_i))
        PK1Exp.append(np.array(PK1_i))

        ### generation of deformation gradients ###
        nTest = dataAlexander[loadCase].shape[0]
        FsExp_LC = lamGamToF(np.array(lams_i),loadCase)
        FsExp.append(tf.convert_to_tensor(FsExp_LC,dtype=tf.float64))

    # additional deformation gradients for visualisation
    if loadCase in loadCasesExp:
        lams_Visu_LC=np.linspace(1.0,lamsExp[iLoadCase][-1],nTestVisu,endpoint=True)
    else:
        if loadCase == 'pureShear':
            lamGamVisuMax = 7.0
        elif loadCase == 'eqBiAx':
            lamGamVisuMax = 6.0
        else:
            raise(ValueError)
        lams_Visu_LC=np.linspace(1.0,lamGamVisuMax,nTestVisu,endpoint=True)
    lams_Visu.append(lams_Visu_LC)
    FsVisu_LC = lamGamToF(lams_Visu_LC,loadCase)
    FsVisu.append(tf.convert_to_tensor(FsVisu_LC,dtype=tf.float64))

    iLoadCase += 1

### VISUALISATION ###
markers=['x','+','o'] #UT, BT, PS

fig, axs = plt.subplots(1,3, figsize=(8,8/3))
PK1_112_NN=[]
PK1_1_NN_Visu=[]

ytop=[4.15,5.4,0.]
xyLabel=np.array([[1.05,3.5],[1.05,4.5],[1.08,2.6]])

invarI=0
corCoeffs = []
for iLoadCase in range(len(loadCases)):

	_,stressNN_Visu = model_manualImpl.psi_and_PK1(FsVisu[iLoadCase])
	PK1_1_NN_Visu.append(np.array(stressNN_Visu[:,0,0]- stressNN_Visu[:,2,2] * FsVisu[iLoadCase][:,2,2] * tf.pow(FsVisu[iLoadCase][:,0,0],-1)))

	if loadCases[iLoadCase] in loadCasesExp:
		_, stressNN = model_manualImpl.psi_and_PK1(FsExp[iLoadCase])
		PK1_112_NN.append(stressNN[:,0,0]- stressNN[:,2,2] * FsExp[iLoadCase][:,2,2] * tf.pow(FsExp[iLoadCase][:,0,0],-1))
		axs[iLoadCase].plot(lamsExp[iLoadCase], PK1Exp[iLoadCase], marker=markers[iLoadCase], linestyle='None',color= 'gray')
          
		coefficient_of_dermination_skl = r2_score(PK1Exp[iLoadCase],PK1_112_NN[iLoadCase].numpy())
		axs[iLoadCase].text(xyLabel[iLoadCase,0],ytop[iLoadCase],r'$r^2 = %.3f $'%(coefficient_of_dermination_skl))

	axs[iLoadCase].plot(lams_Visu[iLoadCase], PK1_1_NN_Visu[iLoadCase], '-', color=invarColors[onlyInvar])

	axs[iLoadCase].set_ylabel(r'$P_{11} \, / \, (\mathrm{N}/\mathrm{mm}^2)$')
	axs[iLoadCase].set_xlabel(r'$\lambda_1$')

	axs[iLoadCase].text(xyLabel[iLoadCase,0],xyLabel[iLoadCase,1],loadCaseLabels[iLoadCase])
	axs[iLoadCase].grid()

axs[0].set_xlim((0.9,9))
axs[0].set_ylim((-0.05,4))
axs[0].set_xticks((1,3,5,7,9))
axs[1].set_xlim((0.9,6))
axs[1].set_ylim((-0.1,5.2))
axs[1].set_yticks((0,1,2,3,4,5))
axs[1].set_xticks((1,2,3,4,5,6))
axs[2].set_xlim((0.9,8))
axs[2].set_ylim((-0.05,3.))
axs[2].set_xticks((1,3,5,7))

plt.subplots_adjust(left=0.1, bottom=0.3, right=0.95, wspace=0.5, hspace = 0.4)

fig.savefig(os.path.join(thisDir,f'visualisation_Alexander.pdf'))
