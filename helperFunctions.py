# -*- coding: utf-8 -*-
"""
Numpy and Scipy script files that are common to both Keras+TF and PyTorch
"""

import numpy as np
import re
from scipy.spatial.distance import cdist
import torch
from torch.optim import Optimizer

__all__ = ['classes', 'eps', 'parse_name', 'rotation_matrix', 'get_gamma', 'get_accuracy']


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


# Implements variation of SGD (optionally with momentum)
class mySGD(Optimizer):

	def __init__(self, params, c, alpha1=1e-6, alpha2=1e-8, momentum=0, dampening=0, weight_decay=0, nesterov=False):
		defaults = dict(alpha1=alpha1, alpha2=alpha2, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
		super(mySGD, self).__init__(params, defaults)
		self.c = c

	def __setstate__(self, state):
		super(mySGD, self).__setstate__(state)
		for group in self.param_groups:
			group.setdefault('nesterov', False)

	def step(self, closure=None):
		loss = None
		if closure is not None:
			loss = closure()

		for group in self.param_groups:
			weight_decay = group['weight_decay']
			momentum = group['momentum']
			dampening = group['dampening']
			nesterov = group['nesterov']

			for p in group['params']:
				if p.grad is None:
					continue
				d_p = p.grad.data

				state = self.state[p]

				# State initialization
				if len(state) == 0:
					state['step'] = 0
				state['step'] += 1

				if weight_decay != 0:
					d_p.add_(weight_decay, p.data)
				if momentum != 0:
					param_state = self.state[p]
					if 'momentum_buffer' not in param_state:
						buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
						buf.mul_(momentum).add_(d_p)
					else:
						buf = param_state['momentum_buffer']
						buf.mul_(momentum).add_(1 - dampening, d_p)
					if nesterov:
						d_p = d_p.add(momentum, buf)
					else:
						d_p = buf

				# cyclical learning rate
				t = (np.fmod(state['step']-1, self.c)+1)/self.c
				if t <= 0.5:
					step_size = (1-2*t)*group['alpha1'] + 2*t*group['alpha2']
				else:
					step_size = 2*(1-t)*group['alpha2'] + (2*t-1)*group['alpha1']
				p.data.add_(-step_size, d_p)

		return loss


def get_accuracy(ytrue, ypred, num_classes):
	# print(ytrue.shape, ypred.shape)
	acc = np.zeros(num_classes)
	for i in range(num_classes):
		acc[i] = np.sum((ytrue == i)*(ypred == i))/np.sum(ytrue == i)
	return np.mean(acc)

