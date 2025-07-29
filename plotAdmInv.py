'''
This is a supplementary material to the paper

"The role of the invariants in neural network-based modelling of incompressible hyperelasticity"
by Franz Dammass, Karl A. Kalina and Markus Kästner.

The code is provided under the CC BY-SA 4.0 license, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
When you find this code useful, please cite the corresponding paper.
'''


import numpy as np
import os
import matplotlib.pyplot as plt
import cmath


def f_low(K):
    '''
		f_low as defined in the manuscript: lower bound of \overline I_2 for given \overline I_1
		K ... \overline I_1
    '''
    complex_expr = K**6 - 540*K**3 - 5832 + 24*cmath.sqrt(-3*K**9 + 243*K**6 - 6561*K**3 + 59049)
    cube_root = cmath.exp(cmath.log(complex_expr) / 3)
    term1 = -cube_root / 24
    term2 = 6 * (-3/2 * K - 1/144 * K**4) / cube_root
    term3 = K**2 / 12
    term4 = -cmath.sqrt(3) * (cube_root / 12 + 12 * (-3/2 * K - 1/144 * K**4) / cube_root) * (1j / 2)
    return term1 + term2 + term3 + term4

def f_up(K):
    '''
        f_up as defined in the manuscript: upper bound of \overline I_2 for given \overline I_1
        K ... \overline I_1
    '''
    sqrt_inner = -3 * K**9 + 243 * K**6 - 6561 * K**3 + 59049
    sqrt_term = cmath.sqrt(sqrt_inner)
    
    cube_root_inner = K**6 - 540 * K**3 - 5832 + 24 * sqrt_term
    cube_root_term = cube_root_inner**(1/3)
    
    if cube_root_term == 0:
        raise ValueError("Cube root term results in division by zero.")
    
    term1 = cube_root_term / 12
    term2 = -12 * (-3/2 * K - K**4 / 144) / cube_root_term
    term3 = K**2 / 12
    
    result = term1 + term2 + term3
    return result

def f_out(K):
    '''
        f_out as defined in the manuscript
		K ... \overline I_1
    '''
    sqrt_inner = -3 * K**9 + 243 * K**6 - 6561 * K**3 + 59049
    sqrt_term = cmath.sqrt(sqrt_inner) 
    
    cube_root_inner = K**6 - 540 * K**3 - 5832 + 24 * sqrt_term
    cube_root_term = cube_root_inner**(1/3) 
    
    if cube_root_term == 0:
        raise ValueError("Cube root term results in division by zero.")
    
    term1 = -1 / 24 * cube_root_term
    term2 = 6 * (-3/2 * K - K**4 / 144) / cube_root_term
    term3 = K**2 / 12
    
    term4 = (cmath.sqrt(3) / 2) * (
        cube_root_term / 12 + 12 * (-3/2 * K - K**4 / 144) / cube_root_term
    ) * 1j
    
    result = term1 + term2 + term3 + term4
    return result

colors=['tab:pink','tab:brown','tab:cyan','navy']
lineStyles = ['solid','dashed','dashdot','dotted']

def cm2inch(arg):
	inch = 2.54
	return arg/inch # 1 inch = 2.54 cm

thisDir = os.path.dirname(os.path.realpath(__file__))

fig,(sp2,sp1) =plt.subplots(1,2, figsize=(cm2inch(2*7),cm2inch(7)))


##### limiting curves and estimates and pure shear  ######
# estimates: limiting curves for invariants I (Lemma A.2, Hartmann & Neff 2003)
I1s = np.linspace(3.0,int(301),int(1e3))
I2_upper = 1/3 * np.power(I1s,2.0)
I2_lower = np.power(3.0 * I1s,0.5)

sp1.plot(I1s,I2_upper,ls='dashdot',color='tab:gray')
sp1.plot(I1s,I2_lower,ls='dashdot',color='tab:gray')

sp2.plot(I1s,(I2_upper),ls='dashdot',color='tab:gray')
sp2.plot(I1s,I2_lower,ls='dashdot',color='tab:gray')


# invariant sets for loadcases
lam1s_ut = np.linspace(1.0,20,int(1e3))

I1s_UT = np.power(lam1s_ut,2.0)+2.0*np.power(lam1s_ut,-1)
I2s_UT = np.power(lam1s_ut,-2.0)+2.0*lam1s_ut
sp1.plot(I1s_UT,I2s_UT,ls=lineStyles[0],color=colors[0])
sp2.plot(I1s_UT,(I2s_UT),ls=lineStyles[0],color=colors[0],label='UT')

lam1s_bt = np.linspace(1.0,10,int(1e4))
I1s_BT = 2.*np.power(lam1s_bt,2.0)+np.power(lam1s_bt,-4.0)
I2s_BT = 2.*np.power(lam1s_bt,-2.0)+np.power(lam1s_bt,4.0)
sp1.plot(I1s_BT,I2s_BT,ls=lineStyles[0],color=colors[1])
sp2.plot(I1s_BT,(I2s_BT),ls=lineStyles[0],color=colors[1],label='BT')

lam1s_ps = np.linspace(1.0,20,int(1e3))
I1s_PS = np.power(lam1s_ps,2.0) + np.power(lam1s_ps,-2.0) + 1
I2s_PS = I1s_PS
sp1.plot(I1s_PS,I1s_PS,ls=lineStyles[-1],color=colors[2])
sp2.plot(I1s_PS,I2s_PS,ls=lineStyles[-1],color=colors[2],label='PS')


### Exemplary deformation states ###

loadCases = ['uniAx','eqBiAx','pureShear','simpleShear']

# exemplary deformation states: huge defos
lam1Gamexampl= np.array([[8,],[4,],[8,],[7,]])
iLoadCase = 0
for loadCase in loadCases:
	lambda1Gams=lam1Gamexampl[iLoadCase,:]
	
	if loadCase == 'uniAx':
		I1_exampl = np.power(lambda1Gams,2.0)+2.0/lambda1Gams
		I2_exampl = np.power(lambda1Gams,-2.0)+2.0*lambda1Gams
	elif loadCase == 'eqBiAx':
		I1_exampl = 2*np.power(lambda1Gams,2.0)+np.power(lambda1Gams,-4.0)
		I2_exampl = 2*np.power(lambda1Gams,-2.0)+np.power(lambda1Gams,4.0)
	elif loadCase == 'pureShear':
		I2_exampl = np.power(lambda1Gams,-2.0)+np.power(lambda1Gams,2.0) + 1
		I1_exampl=I2_exampl
	elif loadCase =='simpleShear':
		I2_exampl = 3 + np.power(lambda1Gams,2.0)
		I1_exampl=I2_exampl

	sp1.plot(I1_exampl,I2_exampl,'x',color=colors[iLoadCase])
	
	print(f'{loadCase} : lamGam = {lambda1Gams} I_1 = {I1_exampl} and I_2 = {I2_exampl}')

	iLoadCase +=1

# exemplary deformation states: small defos
lam1Gamexampl= np.array([[1.1,],[1.05,],[1.1,],[0.1,]])
iLoadCase = 0
for loadCase in loadCases:
	lambda1Gams=lam1Gamexampl[iLoadCase,:]
	
	if loadCase == 'uniAx':
		I1_exampl = np.power(lambda1Gams,2.0)+2.0/lambda1Gams
		I2_exampl = np.power(lambda1Gams,-2.0)+2.0*lambda1Gams
	elif loadCase == 'eqBiAx':
		I1_exampl = 2*np.power(lambda1Gams,2.0)+np.power(lambda1Gams,-4.0)
		I2_exampl = 2*np.power(lambda1Gams,-2.0)+np.power(lambda1Gams,4.0)
	elif loadCase == 'pureShear':
		I2_exampl = np.power(lambda1Gams,-2.0)+np.power(lambda1Gams,2.0) + 1
		I1_exampl=I2_exampl
	elif loadCase =='simpleShear':
		I2_exampl = 3 + np.power(lambda1Gams,2.0)
		I1_exampl=I2_exampl

	sp2.plot(I1_exampl,I2_exampl,'o',color=colors[iLoadCase])
	
	print(f'{loadCase} : lamGam = {lambda1Gams} I_1 = {I1_exampl} and I_2 = {I2_exampl}')
	
	iLoadCase +=1



# Create a range of values for ici
ici_vals = np.linspace(3.0, 300, int(1e4))  # Use real values for simplicity here
ici_vals = ici_vals + 1j * 0  # Treat as complex numbers
# low_bound = f_low(ici_vals)

low_bound = []
up_bound = []
out = []
for iciV in ici_vals:
	low_bound.append(f_low(iciV))
	up_bound.append(f_up(iciV))
	out.append(f_out(iciV))

# sp1.plot(ici_vals,low_bound,ls='dotted',color='red')
for spI in (sp1,sp2):
	spI.plot(ici_vals,low_bound,ls='dotted',color='red')
	spI.plot(ici_vals,up_bound,ls='dotted',color='red')
	spI.plot(ici_vals,out,ls='dotted',color='tab:gray')

sp1.grid()
sp1.set_xlim((0,300)) # Trel: max. 60
sp1.set_ylim((0,300))
sp1.set_xlabel(r'$\overline{I}_1$')
sp1.set_ylabel(r'$\overline{I}_2$')
sp1.yaxis.tick_right()
sp1.yaxis.set_label_position("right")

limZoomLow=2.995
limZoomUp=3.055
sp2.grid()
sp2.set_xlim((limZoomLow,limZoomUp))

sp2.set_ylim((limZoomLow,limZoomUp))
sp2.set_xlabel(r'$\overline{I}_1$')
sp2.set_ylabel(r'$\overline{I}_2$')

plt.tight_layout(h_pad=10)

plt.subplots_adjust(left=0.1, bottom=0.3, right=0.95, wspace=0.4, hspace = 2)

fig.savefig(os.path.join(thisDir,'admInv.pdf'),bbox_inches = "tight")

fig2,spp1 =plt.subplots(1,1, figsize=(cm2inch(7),cm2inch(7)))
spp1.plot(ici_vals,low_bound,ls='solid',color='tab:pink',label='f_low')
spp1.plot(ici_vals,up_bound,ls='solid',color='tab:brown',label='f_up')
spp1.plot(ici_vals,out,ls='dotted',color='tab:gray',label='f_out')
spp1.grid()
spp1.set_ylim((-50,300))
spp1.set_xlabel(r'$\overline{I}_1$')
spp1.set_ylabel(r'$\overline{I}_2$')
spp1.legend(loc='best')
fig2.savefig(os.path.join(thisDir,'admInv_fout.pdf') ,bbox_inches = "tight")