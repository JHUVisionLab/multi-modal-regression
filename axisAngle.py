# -*- coding: utf-8 -*-
"""
Pose models and loss functions for axis-angle representation
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from helperFunctions import eps


"""
Numpy functions
"""


# axis angle y = \theta v where theta is angle of rotation about axis v
# y = logm(R) where R is a rotation matrix. I use Rodriguez' rotation formula
def get_y(R):
	tR = 0.5*(np.trace(R)-1)
	theta = np.arccos(np.clip(tR, -1., 1.))
	tmp = 0.5*(R - R.T)
	v = np.array([tmp[2, 1], tmp[0, 2], tmp[1, 0]])
	if np.linalg.norm(v) > eps:
		v = v/np.linalg.norm(v)
	else:
		v = np.zeros(3)
	y = theta*v
	return y


# function to get rotation matrix given axis-angle representation
def get_R(v):
	theta = np.linalg.norm(v)
	if theta < eps:
		R = np.eye(3)
	else:
		v = v / theta
		V = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
		R = np.eye(3) + np.sin(theta) * V + (1 - np.cos(theta)) * np.dot(V, V)
	return R


# function to get angle error between gt and predicted viewpoints
def get_error(ygt, yhat):
	N = ygt.shape[0]
	az_error = np.zeros(N)
	for i in range(N):
		# read the 3-dim axis-angle vectors
		v1 = ygt[i]
		v2 = yhat[i]
		# get correponding rotation matrices
		R1 = get_R(v1)
		R2 = get_R(v2)
		# compute \|log(R_1^T R_2)\|_F/\sqrt(2) using Rodrigues' formula
		R = np.dot(R1.T, R2)
		tR = np.trace(R)
		theta = np.arccos(np.clip(0.5*(tR-1), -1.0, 1.0))   # clipping to avoid numerical issues
		atheta = np.abs(theta)
		# print('i:{0}, tR:{1}, theta:{2}'.format(i, tR, theta, atheta))
		az_error[i] = np.rad2deg(atheta)
	medErr = np.median(az_error)
	maxErr = np.amax(az_error)
	acc = 100 * np.sum(az_error < 30) / az_error.size
	print('Error stats- Median: {0}, Max: {1}, <30: {2}'.format(medErr, maxErr, acc))
	return acc, medErr, az_error


# function to get angle error between gt and predicted viewpoints
def get_error2(ygt, yhat, labels, num):
	N = ygt.shape[0]
	err = np.zeros(N)
	for i in range(N):
		# read the 3-dim axis-angle vectors
		v1 = ygt[i]
		v2 = yhat[i]
		# get correponding rotation matrices
		R1 = get_R(v1)
		R2 = get_R(v2)
		# compute \|log(R_1^T R_2)\|_F/\sqrt(2) using Rodrigues' formula
		R = np.dot(R1.T, R2)
		tR = np.trace(R)
		theta = np.arccos(np.clip(0.5*(tR-1), -1.0, 1.0))   # clipping to avoid numerical issues
		atheta = np.abs(theta)
		# print('i:{0}, tR:{1}, theta:{2}'.format(i, tR, theta, atheta))
		err[i] = np.rad2deg(atheta)
	# print(labels.shape, err.shape)
	labels = np.squeeze(labels)
	medErr = np.zeros(num)
	for i in range(num):
		ind = (labels == i)
		# print(ind.shape)
		medErr[i] = np.median(err[ind])
	# print(medErr)
	return np.mean(medErr)


"""
Loss function
"""


class geodesic_loss(nn.Module):

	def __init__(self, reduce=True):
		super().__init__()
		self.eps = eps
		self.reduce = reduce

	def forward(self, ypred, ytrue):
		angle_pred = torch.norm(ypred, 2, 1)
		angle_true = torch.norm(ytrue, 2, 1)
		axis_pred = F.normalize(ypred)
		axis_true = F.normalize(ytrue)
		tmp = torch.abs(torch.cos(angle_true/2)*torch.cos(angle_pred/2) + torch.sin(angle_true/2)*torch.sin(angle_pred/2)*torch.sum(axis_true*axis_pred, dim=1))
		theta = 2.0*torch.acos(torch.clamp(tmp, -1+self.eps, 1-self.eps))
		if self.reduce:
			return torch.mean(theta)
		else:
			return theta
