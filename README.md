This is a supplementary material to the paper

"The role of the invariants in neural network-based modelling of incompressible hyperelasticity"
by Franz Dammass, Karl A. Kalina and Markus Kästner.

The code is provided under the CC BY-SA 4.0 license, see https://creativecommons.org/licenses/by-nc-sa/4.0/
When you find this code useful, please cite the corresponding paper.

-----

Running the scripts requires python3, numpy and tensorflow. The software has been tested with 3.11.7.

-----
First,
the admissibility conditions for the deformation I1 and I2
and typical values of I1 and I2
can be visualised running the script

plotAdmInv.py

-----
Second,
by running one of the scripts

pannVisu_Alexander.py
pannVisu_Budday.py
pannVisu_Treloar.py

you can visualise the stress predictions of the neural network based models compared to the experimental data as shown in the paper.
By modyfing the variables 

onlyInvar

or

nnType

in the section "choose a model type", you can select between PANN and unconstrained FNN-based models and vary whether the model should depend on both \tilde I_1 and \tilde I_2 or only one of the invariants.


-----
