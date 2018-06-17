# -*- coding: utf-8 -*-
"""
Numpy and Scipy script files that are common to both Keras+TF and PyTorch
"""

import numpy as np
import re
from scipy.spatial.distance import cdist

__all__ = ['classes', 'eps', 'parse_name', 'rotation_matrix', 'get_gamma']


# object categories of interest
classes = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor']


# numeric precision for my experiments
eps = 1e-6


# parse the name of the image to get model and pose parameters
def parse_name(image_name):
	ind = [match.start() for match in re.finditer('_', image_name)]
	synset_str = image_name[:ind[0]]
	model_str = image_name[ind[0]+1:ind[1]]
	az = float(image_name[ind[1]+2:ind[2]])
	el = float(image_name[ind[2]+2:ind[3]])
	ct = float(image_name[ind[3]+2:ind[4]])
	d = float(image_name[ind[4]+2:])
	return synset_str, model_str, az, el, ct, d


# get rotation matrix R(az, el, ct) given the three euler angles :
# azimuth az, elevation el, camera-tilt ct
def rotation_matrix(az, el, ct):
	ca = np.cos(np.radians(az))
	sa = np.sin(np.radians(az))
	cb = np.cos(np.radians(el))
	sb = np.sin(np.radians(el))
	cc = np.cos(np.radians(ct))
	sc = np.sin(np.radians(ct))
	Ra = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
	Rb = np.array([[1, 0, 0], [0, cb, -sb], [0, sb, cb]])
	Rc = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
	R = np.dot(np.dot(Rc, Rb), Ra)
	return R


def get_gamma(kmeans_dict):
	N = kmeans_dict.shape[0]
	D = cdist(kmeans_dict, kmeans_dict, 'sqeuclidean')
	d = np.zeros(N)
	for i in range(N):
		d[i] = np.amin(D[i, np.arange(N) != i])
	gamma = 1/(2*np.amin(d))
	return gamma
