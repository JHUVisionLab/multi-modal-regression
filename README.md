# multi-modal-regression
Models and Loss functions for our Mixed Classification-Regression framework for 3D Pose Estimation

Shorthand notation for my models:-\
r0 : Euclidean Regression\
r1 : Geodesic Regression\
c0 : Classification\
m0 : Simple Bin & Delta\
m1 : Geodesic Bin & Delta\
m2 : Riemannian Bin & Delta\
m3 : Probabilistic Bin & Delta\
m0+, m1+, m2+, m3+ : One delta per pose bin models

Please create the following folders: data, plots, results, models\
I setup symbolic links in the data folder. For example: ln -s /path/to/rendered/data data/renderforcnn

The code is largely self-explanatory. But if you encounter any problems, please contact me at siddharthm@jhu.edu.

If you use this work please cite our paper : 
<a href="https://arxiv.org/abs/1805.03225"> A Mixed Classification-Regression Framework for 3D Pose Estimation from 2D Images </a>

