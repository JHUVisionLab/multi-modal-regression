# -*- coding: utf-8 -*-
"""
Function to learn a kmeans dictionary starting from images
"""

from dataGenerators import ImagesAll
from helperFunctions import parse_name, rotation_matrix
from axisAngle import get_y

import numpy as np
from sklearn.mixture import GaussianMixture
import pickle
import progressbar
import sys

# relevant paths
image_path = 'data/renderforcnn'

# relevant variables
num_clusters = int(sys.argv[1])
print('num_clusters: ', num_clusters)
gmm_file = 'data/gmm_dictionary_axis_angle_' + str(num_clusters) + '.pkl'

# setup data loader to access the images
train_data = ImagesAll(image_path, 'render')
image_names = np.concatenate(train_data.list_image_names)

# get pose targets from training data
bar = progressbar.ProgressBar()
ydata = []
for i in bar(range(len(image_names))):
	image_name = image_names[i]
	_, _, az, el, ct, _ = parse_name(image_name)
	R = rotation_matrix(az, el, -ct)
	y = get_y(R)
	ydata.append(y)
ydata = np.stack(ydata)
print('\nData size: ', ydata.shape)

# run kmeans
gmm = GaussianMixture(num_clusters, covariance_type='full', verbose=1, n_init=10)
gmm.fit(ydata)
print(gmm.means_)

# save output
fid = open(gmm_file, 'wb')
pickle.dump(gmm, fid)

del gmm

# load and check
gmm = pickle.load(open(gmm_file, 'rb'))
print(gmm.means_)
