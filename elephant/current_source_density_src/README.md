Here, are CSD methods for different electrode configurations.

Keywords: Local field potentials; Current-source density; CSD;
Multielectrode; Laminar electrode; Barrel cortex

1D - laminar probe like electrodes. 
2D - Microelectrode Array like
3D - UtahArray or multiple laminar probes.

The following methods have been implemented here, for, 

1D - StandardCSD, DeltaiCSD, SplineiCSD, StepiCSD, KCSD1D
2D - KCSD2D, MoIKCSD (Saline layer on top of slice)
3D - KCSD3D

Each of these methods listed have some advantages - except StandardCSD which is
not recommended. The KCSD methods can handle broken or irregular electrode
configurations electrode

iCSD
----
Python-implementation of the inverse current source density (iCSD) methods from
http://software.incf.org/software/csdplotter

The Python iCSD toolbox lives on GitHub as well:
https://github.com/espenhgn/iCSD

The methods were originally developed by Klas H. Pettersen, as described in:
Klas H. Pettersen, Anna Devor, Istvan Ulbert, Anders M. Dale, Gaute
T. Einevoll, Current-source density estimation based on inversion of
electrostatic forward solution: Effects of finite extent of neuronal activity
and conductivity discontinuities, Journal of Neuroscience Methods, Volume 154,
Issues 1Ð2, 30 June 2006, Pages 116-133, ISSN 0165-0270,
http://dx.doi.org/10.1016/j.jneumeth.2005.12.005.
(http://www.sciencedirect.com/science/article/pii/S0165027005004541)

To see an example of usage of the methods, see
[demo_icsd.py](https://github.com/espenhgn/iCSD/blob/master/demo_icsd.py)

KCSD 
---- 
This is 1.0 version of kCSD inverse method proposed in

J. Potworowski, W. Jakuczun, S. Łęski, D. K. Wójcik
"Kernel Current Source Density Method"
Neural Computation 24 (2012), 541–575

Some key advantages for KCSD methods are
-- irregular grid of electrodes - accepts arbitrary electrode placement.
-- crossvalidation to ensure no over fitting
-- CSD is not limited to electrode positions - it can obtained at any location

For citation purposes, 
If you use this software in published research please cite the following work
- kCSD1D - [1, 2]
- kCSD2D - [1, 3]
- kCSD3D - [1, 4]
- MoIkCSD - [1, 3, 5]

[1] Potworowski, J., Jakuczun, W., Łęski, S. & Wójcik, D. (2012) 'Kernel
current source density method.' Neural Comput 24(2), 541-575.

[2] Pettersen, K. H., Devor, A., Ulbert, I., Dale, A. M. & Einevoll,
G. T. (2006) 'Current-source density estimation based on inversion of
electrostatic forward solution: effects of finite extent of neuronal activity
and conductivity discontinuities.' J Neurosci Methods 154(1-2), 116-133.

[3] Łęski, S., Pettersen, K. H., Tunstall, B., Einevoll, G. T., Gigg, J. &
Wójcik, D. K. (2011) 'Inverse Current Source Density method in two dimensions:
Inferring neural activation from multielectrode recordings.' Neuroinformatics
9(4), 401-425.

[4] Łęski, S., Wójcik, D. K., Tereszczuk, J., Świejkowski, D. A., Kublik, E. &
Wróbel, A. (2007) 'Inverse current-source density method in 3D: reconstruction
fidelity, boundary effects, and influence of distant sources.' Neuroinformatics
5(4), 207-222.

[5] Ness, T. V., Chintaluri, C., Potworowski, J., Łeski, S., Głabska, H.,
Wójcik, D. K. & Einevoll, G. T. (2015) 'Modelling and Analysis of Electrical
Potentials Recorded in Microelectrode Arrays (MEAs).' Neuroinformatics 13(4),
403-426.

For your research interests of Kernel methods of CSD please see,
https://github.com/Neuroinflab/kCSD-python 

Contact: Prof. Daniel K. Wojcik

Here (https://github.com/Neuroinflab/kCSD-python/tree/master/tests), are
scripts to compare different KCSD methods with different CSD sources. You can
play around with the different parameters of the methods.

The implentation is based on the Matlab version at INCF
(http://software.incf.org/software/kcsd), which is now out-dated. A python
version based on this was developed by Grzegorz Parka
(https://github.com/INCF/pykCSD), which is also not supported at this
point. This current version of KCSD methods in elephant is a mirror of
https://github.com/Neuroinflab/kCSD-python/commit/8e2ae26b00da7b96884f2192ec9ea612b195ec30
