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


##### choose a model type #####

onlyInvar = 0 # 0 .. \tilde I1 and \tilde I2, 1 .. only \tilde I1, 2 .. only \tilde I2
nnType = 'PANN' # 'PANN' or unconstrained 'FNN' 
# nnType = 'FNN'

FitUTOnly = False # model parameterised from UT data only?
# FitUTOnly = True # model parameterised from UT data only?


##### Load NN model #####
invarSubDirs = ['both','i1','i2']

thisDir= os.path.dirname(os.path.realpath(__file__))
dirName ='Network_Parameters_Treloar' + '/' 
if FitUTOnly:
    dirName += 'Fit_UT_'
else:
    dirName += 'Fit_UT_BT_'
dirName +=invarSubDirs[onlyInvar] + '/' # path to the specific network parameter sub-directory
dirPath = os.path.join(thisDir,dirName)

modelName={'PANN':'FICNN','FNN':'FNN'}

if onlyInvar ==0:
    archString = '-4-4'
else:
    archString = '-2-2'

nnParameterFile = f'treLC_param_{modelName[nnType]}_{archString}' # network parameter file


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
expData={}

dirPathExpFiles = os.path.join(thisDir,'treloarData') #load data of Treloar: ‘Stress-strain data for vulcanised rubber under various types of deformation’, Trans. Faraday Soc., 1944.

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
    reFile = 'treloar_%i.csv'%(iLoadCase)
    reFilePath = os.path.join(dirPathExpFiles,reFile)
    reFilePath = str(reFilePath)
    expData = np.loadtxt(open(reFilePath, "r"), delimiter=",")#,

    lamsExp.append(np.array(expData[:,0]))
    PK1Exp.append(np.array(expData[:,1]))

    ### generation of deformation gradients ###
    nTest = expData.shape[0]
    FsExp_LC = lamGamToF(np.array(expData[:,0]),loadCase)
    FsExp.append(tf.convert_to_tensor(FsExp_LC,dtype=tf.float64))

    # additional deformation gradients for visualisation
    lams_Visu_LC=np.linspace(1.0,lamsExp[iLoadCase][-1]*1.05,nTestVisu,endpoint=True)
    lams_Visu.append(lams_Visu_LC)
    FsVisu_LC = lamGamToF(lams_Visu_LC,loadCase)
    FsVisu.append(tf.convert_to_tensor(FsVisu_LC,dtype=tf.float64))

    iLoadCase += 1
    
### VISUALISATION ###
markers=['x','+','o'] #UT, BT, PS

fig, axs = plt.subplots(1,3, figsize=(8,8/3))
PK1_112_NN=[]
PK1_1_NN_Visu=[]

ytop=[6.55,2.625,2.02]
xyLabel=np.array([[1.007,5.88],[1.005,2.35],[1.005,1.8]])

invarI=0
corCoeffs = []
for iLoadCase in range(len(loadCases)):

	# NN stress prediction
	energy_NN, stressNN = model_manualImpl.psi_and_PK1(FsExp[iLoadCase])

	PK1_112_NN.append(stressNN[:,0,0]- stressNN[:,2,2] * FsExp[iLoadCase][:,2,2] * tf.pow(FsExp[iLoadCase][:,0,0],-1))


	_,stressNN_Visu = model_manualImpl.psi_and_PK1(FsVisu[iLoadCase])
	PK1_1_NN_Visu.append(np.array(stressNN_Visu[:,0,0]- stressNN_Visu[:,2,2] * FsVisu[iLoadCase][:,2,2] * tf.pow(FsVisu[iLoadCase][:,0,0],-1)))


	axs[iLoadCase].plot(lamsExp[iLoadCase], PK1Exp[iLoadCase], marker=markers[iLoadCase], linestyle='None',color= 'gray')

	axs[iLoadCase].plot(lams_Visu[iLoadCase], PK1_1_NN_Visu[iLoadCase], '-', color=invarColors[onlyInvar])

	axs[iLoadCase].set_ylabel(r'$P_{11} \, / \, (\mathrm{N}/\mathrm{mm}^2)$')
	axs[iLoadCase].set_xlabel(r'$\lambda_1$')
	try:
		coefficient_of_dermination_skl = r2_score(PK1Exp[iLoadCase],PK1_112_NN[iLoadCase].numpy())
		axs[iLoadCase].text(xyLabel[iLoadCase,0],ytop[iLoadCase],r'$r^2 = %.3f $'%(coefficient_of_dermination_skl))
	except:
		axs[iLoadCase].text(xyLabel[iLoadCase,0],ytop[iLoadCase],r'$r^2 =$ ?')

	axs[iLoadCase].text(xyLabel[iLoadCase,0],xyLabel[iLoadCase,1],loadCaseLabels[iLoadCase])
	axs[iLoadCase].grid()

axs[0].set_ylim((-0.1,6.5))
axs[0].set_xticks((1,3,5,7))
axs[1].set_ylim((-0.05,2.6))
axs[1].set_yticks((0,0.5,1.0,1.5,2.0,2.5))
axs[1].set_xticks((1,2,3,4,5))
axs[2].set_ylim((-0.05,2))
axs[2].set_yticks((0,0.5,1.0,1.5,2.0))
axs[2].set_xticks((1,2,3,4,5))

plt.subplots_adjust(left=0.1, bottom=0.3, right=0.95, wspace=0.5, hspace = 0.4)

fig.savefig(os.path.join(thisDir,f'visualisation_Treloar.pdf'))
