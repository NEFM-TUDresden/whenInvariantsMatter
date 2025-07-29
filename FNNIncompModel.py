'''
This is a supplementary material to the paper

"The role of the invariants in neural network-based modelling of incompressible hyperelasticity"
by Franz Dammass, Karl A. Kalina and Markus Kästner.

The code is provided under the CC BY-SA 4.0 license, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
When you find this code useful, please cite the corresponding paper.
'''

import tensorflow as tf
import save_store_variables as ssv
import copy
from typing import List

class FNNIncompModel():
	'''isochoric Free energy -- unconstrained feed forward neural network'''
	
	def __init__(self,parameterPath:str, nnParameterFile: str, onlyInvar = 0) -> None:
		'''
		parameter onlyInvar : only Consider \tilde I_1 or  \tilde I_2
		'''
	
		
		weightsHL_mainPath,biasesHL,weightsOL_mainPath,biasOL = ssv.load_from_file(parameterPath,nnParameterFile)

		self.nHL = len(weightsHL_mainPath)

		self.onlyInv = onlyInvar

		# parameters
		self.weightsHL_mainPath = weightsHL_mainPath # list of tf.Tensor, one element per HL
		self.biasesHL =  biasesHL # list of tf.Tensor, one element per HL
		self.weightsOL_mainPath = tf.reshape(weightsOL_mainPath,shape=(weightsOL_mainPath.shape[0],)) # tf.Tensor
		self.biasOL = tf.reshape(biasOL,shape=()) # tf.Tensor

		# consistency check
		if len(self.weightsHL_mainPath) != (len(self.biasesHL)):
			raise(ValueError('There is something wrong in the data structures!'))

		unitTens = tf.eye(num_rows=3,num_columns=3,batch_shape=[1],dtype=tf.float64)
		self.psiNNZero = copy.deepcopy(tf.constant(self.psi_NN(unitTens),dtype=tf.float64))

	def printParams(self):
		'''print values of the weights and biases of the model'''
		nHL = len(self.weightsHL_mainPath)
		for iHL in range(nHL):
			print(f'---------- HIDDEN LAYER # {iHL} ----------')

			for dataEl in [self.weightsHL_mainPath,self.biasesHL]:
				print(dataEl[iHL].shape)
				print(dataEl[iHL].numpy())

		print(f'---------- OUTPUT LAYER ----------')
		iData=0
		for datael in [self.weightsOL_mainPath,self.biasOL]:
			print(datael.shape)
			print(datael.numpy())
			iData +=1
	
	def activFu(self,var : tf.Tensor) -> tf.Tensor:
		return tf.math.log(1.0 + tf.exp(var)) #softplus

	def invariants(self,F:tf.Tensor) -> tf.Tensor:
		'''calculate the polyconvex invariants \tilde I_1 and \tilde I_2 of the isochoric right Cauchy-Green deformation tensor'''
		p = tf.constant(3/2,dtype=tf.float64)

		C = tf.einsum('bkL,bkM->bLM',F,F)

		# compressible invariants
		I1C = tf.linalg.trace(C)
		I2C = 0.5*(tf.pow(I1C,2) - tf.einsum('bLM,bLM->b',C,C))
		I3C = tf.linalg.det(C)

		# polyconvex incompressible invariants
		I1Tilde = tf.pow(I3C,tf.constant(-1/3,dtype=tf.float64))*I1C
		I2Tilde = tf.pow(tf.pow(I3C,tf.constant(-2/3,dtype=tf.float64))*I2C,p)

		if self.onlyInv == 1:
			invars = tf.reshape(I1Tilde,shape=(I1Tilde.shape[0],1))
		elif self.onlyInv == 2:
			invars =  tf.reshape(I2Tilde,shape=(I2Tilde.shape[0],1))
		elif self.onlyInv == 0:
			invars = tf.stack([I1Tilde, I2Tilde], axis=-1)
		else:
			raise(ValueError)

		return invars

	def psi(self, F: tf.Tensor) -> tf.Tensor:
		'''isochoric psi_PANN'''
		return self.psi_NN(F) - 1.0*self.psiNNZero

	def psi_NN(self,F:tf.Tensor) -> tf.Tensor:
		'''isochoric psi_NN'''

		invars = self.invariants(F)

		# first HL
		f_iHL = self.activFu(tf.einsum('iN,bi->bN',self.weightsHL_mainPath[0],invars) + self.biasesHL[0])
		
		# other HL
		for iHL in range(1,self.nHL):						
			f_iHL = self.activFu(tf.einsum('nN,bn->bN',self.weightsHL_mainPath[iHL],f_iHL) + self.biasesHL[iHL])
		
		# OL
		psi_NN = tf.einsum('n,bn->b',self.weightsOL_mainPath,f_iHL) +self.biasOL

		return psi_NN
	
	def psi_and_PK1(self, F: tf.Tensor) -> List[tf.Tensor]:
		'''isochoric psi_PANN and first PK stress related to isochoric deformation'''

		with tf.GradientTape() as tape:
			tape.watch(F)
			psi = self.psi(F)
		P = tape.gradient(psi, F)
		return psi, P