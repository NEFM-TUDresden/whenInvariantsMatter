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

# pseudo-elastic stress-deformation tuples and uncertainty for human cortex tissue (gray matter)
# taken from Fig. 11, Budday et al., ‘Mechanical characterization of human brain tissue’, Acta Biomaterialia, Jan. 2017.
dataBudday={} #lam	up	mean	lowCalc
dataBudday['uniAx'] = np.array([[1.0, 0.0, 0.0, 0.0],
    [1.01238865, 0.061951036, 0.043709773, 0.02546851],
    [1.024951077, 0.113389521, 0.082245402, 0.051101283],
    [1.037428974, 0.157370164, 0.115301332, 0.0732325],
    [1.049948503, 0.200239705, 0.148702241, 0.097164778],
    [1.062502917, 0.250807361, 0.184429066, 0.11805077],
    [1.0749755, 0.329357808, 0.234415461, 0.139473115],
    [1.087528235, 0.458502023, 0.311768755, 0.165035488],
    [1.100013367, 0.644140303, 0.411394595, 0.178648887]])
dataBudday['uniAxCompr']  = np.array([[1.0, 0.0, 0.0, 0.0],
    [0.987656225, -0.096492024, -0.072258065, -0.048024105],
    [0.974949328, -0.202415245, -0.14368729, -0.084959335],
    [0.962576251, -0.325784914, -0.234689817, -0.14359472],
    [0.950107047, -0.47600067, -0.344991349, -0.213982029],
    [0.937654877, -0.661151787, -0.479212683, -0.297273579],
    [0.925029689, -0.895963987, -0.654613876, -0.413263764],
    [0.912757944, -1.206451613, -0.881963807, -0.557476002],
    [0.900205855, -1.581935484, -1.134196283, -0.686457082]])
dataBudday['simpleShear'] = np.array([[0.0, 0.0, 0.0, 0.0],
    [0.0249, 0.0497, 0.0363, 0.0229],
    [0.0498, 0.0926, 0.0681, 0.0436],
    [0.0751, 0.1401, 0.10215, 0.0642],
    [0.1, 0.1964, 0.1451, 0.0938],
    [0.1249, 0.2751, 0.20145, 0.1278],
    [0.1502, 0.3717, 0.27235, 0.173],
    [0.1747, 0.5307, 0.38225, 0.2338],
    [0.1989, 0.7634, 0.54685, 0.3303]])

##### choose a model type #####

onlyInvar = 0# 0 .. \tilde I1 and \tilde I2, 1 .. only \tilde I1, 2 .. only \tilde I2

nnType = 'PANN' # 'PANN' or unconstrained 'FNN' 
# nnType = 'FNN'


##### Load NN model #####
invarSubDirs = ['both','i1','i2']

thisDir= os.path.dirname(os.path.realpath(__file__))
dirName ='Network_Parameters_Budday' + '/' + invarSubDirs[onlyInvar] + '/' # path to the specific network parameter sub-directory
dirPath = os.path.join(thisDir,dirName)

# modelName={'PANN':'FICNN','FNN':'FNN'}
modelName={'PANN':'FICNNoHL','FNN':'FNN'}


if nnType == 'PANN':
    archString = '-3'
else:
    archString = '-2-2'

nnParameterFile = f'brain_param_{modelName[nnType]}_{archString}' # network parameter file


if modelName[nnType] == 'FICNN':
    model_manualImpl = pann.PANNIncompModel(dirPath,nnParameterFile,onlyInvar=onlyInvar)
    # model_manualImpl = pannOHL.FICNNoHLIncompModel(dirPath,nnParameterFile)
elif modelName[nnType] == 'FNN' or modelName[nnType] == 'FICNNoHL':
    # unconstrained FNN
    model_manualImpl = fnn.FNNIncompModel(dirPath,nnParameterFile,onlyInvar=onlyInvar)
else:
     raise(ValueError('An invalid NN type has been chosen!'))

# model_manualImpl.printParams()


##### Load experimental data used for visualisation #####
loadCases=['uniAx','uniAxCompr','simpleShear']
loadCasesExp =loadCases

loadCaseLabels=['UT','UC','SS']

invarColors = ['tab:blue','tab:orange','tab:green']

lamsExp=[]
PK1Exp=[]
FsExp=[] #deformation gradients for exp defo states, needed for NN stress computation

lams_Visu=[]
nTestVisu = int(2e2) # number of data points for model visualisation
FsVisu=[]

dirPathExpFiles = os.path.join(thisDir,'brainData') # pseudo-elastic stress-deformation tuples and uncertainty for human cortex tissue (gray matter) 
# of Budday et al., ‘Mechanical characterization of human brain tissue’, Acta Biomaterialia, Jan. 2017 
# as provided in Table 1, Linka et al., ‘Automated model discovery for human brain using Constitutive Artificial Neural Networks’, Acta Biomaterialia, Apr. 2023.
iLoadCase = 0
for loadCase in loadCases:
    reFile = 'cortexGrayMatter_%i.csv'%(iLoadCase)
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
    lams_Visu_LC=np.linspace(lamsExp[iLoadCase][0],lamsExp[iLoadCase][-1],nTestVisu,endpoint=True)
    lams_Visu.append(lams_Visu_LC)
    FsVisu_LC = lamGamToF(lams_Visu_LC,loadCase)
    FsVisu.append(tf.convert_to_tensor(FsVisu_LC,dtype=tf.float64))

    iLoadCase += 1

### VISUALISATION ###
markers=['x','+','1'] #UT, BT, SS

fig, axs = plt.subplots(1,3, figsize=(8,8/3))
PK1_112_NN=[]
PK1_1_NN_Visu=[]

ytop=[0.68,0.15,0.83]
xyLabel=np.array([[0.993,0.55],[0.893,-0.15],[-0.005,0.7]])

invarI=0
corCoeffs = []
for iLoadCase in range(len(loadCases)):

	_,stressNN_Visu = model_manualImpl.psi_and_PK1(FsVisu[iLoadCase])
	if loadCases[iLoadCase] != 'simpleShear':
		PK1_1_NN_Visu.append(stressNN_Visu[:,0,0]- stressNN_Visu[:,2,2] * FsVisu[iLoadCase][:,2,2] * tf.pow(FsVisu[iLoadCase][:,0,0],-1))
	else:
		PK1_1_NN_Visu.append(stressNN_Visu[:,0,1])
          
	if loadCases[iLoadCase] in loadCasesExp:
		_, stressNN = model_manualImpl.psi_and_PK1(FsExp[iLoadCase])
		
		if loadCases[iLoadCase] != 'simpleShear':
			PK1_112_NN.append(stressNN[:,0,0]- stressNN[:,2,2] * FsExp[iLoadCase][:,2,2] * tf.pow(FsExp[iLoadCase][:,0,0],-1))
		else:
			PK1_112_NN.append(stressNN[:,0,1])
               
		axs[iLoadCase].plot(lamsExp[iLoadCase], PK1Exp[iLoadCase], marker=markers[iLoadCase], linestyle='None',color= 'gray')
          
		# fill range of experimental data
		axs[iLoadCase].fill_between(dataBudday[loadCases[iLoadCase]][:,0],dataBudday[loadCases[iLoadCase]][:,1],dataBudday[loadCases[iLoadCase]][:,3],color='gray',alpha=0.4,edgecolor=None)
          
		coefficient_of_dermination_skl = r2_score(PK1Exp[iLoadCase],PK1_112_NN[iLoadCase].numpy())
		axs[iLoadCase].text(xyLabel[iLoadCase,0],ytop[iLoadCase],r'$r^2 = %.3f $'%(coefficient_of_dermination_skl))

	axs[iLoadCase].plot(lams_Visu[iLoadCase], PK1_1_NN_Visu[iLoadCase], '-', color=invarColors[onlyInvar])

	axs[iLoadCase].set_ylabel(r'$P_{11} \, / \, (\mathrm{N}/\mathrm{mm}^2)$')
	axs[iLoadCase].set_xlabel(r'$\lambda_1$')

	axs[iLoadCase].text(xyLabel[iLoadCase,0],xyLabel[iLoadCase,1],loadCaseLabels[iLoadCase])
	axs[iLoadCase].grid()

axs[0].set_xlim((0.99,1.11))
axs[0].set_ylim((-0.01,0.65))
axs[1].set_xlim((0.89,1.01))
axs[2].set_xlim((-0.01,0.21))
axs[2].set_ylim((-0.01,0.8))

plt.subplots_adjust(left=0.1, bottom=0.3, right=0.95, wspace=0.5, hspace = 0.4)

fig.savefig(os.path.join(thisDir,f'visualisation_Budday.pdf'))
