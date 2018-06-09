# -*- coding: utf-8 -*-
"""
Pose models and loss functions for quaternion representation
"""
import torch
from torch import nn
import torch.nn.functional as F
from helperFunctions import eps
import numpy as np

"""
Numpy functions
"""


# y = (\cos \theta/2, \sin \theta/2 v) where \theta is angle of rotation about axis v
# first get \theta and v given rotation matrix R and then compute y
def get_y(R):
	tR = 0.5 * (np.trace(R) - 1)
	theta = np.arccos(np.clip(tR, -1., 1.))
	tmp = 0.5 * (R - R.T)
	v = np.array([tmp[2, 1], tmp[0, 2], tmp[1, 0]])
	if np.linalg.norm(v) > eps:
		v = v / np.linalg.norm(v)
	else:
		theta = 0
		v = np.zeros(3)
	y = np.array([np.cos(theta / 2.), np.sin(theta / 2.) * v[0], np.sin(theta / 2.) * v[1], np.sin(theta / 2.) * v[2]])
	return y


# function to get angle error between gt and predicted quaternions
def get_error(ygt, yhat):
	N = ygt.shape[0]
	az_error = np.zeros(N)
	for i in range(N):
		# read the quaternion vectors in (c, v) format
		c1 = ygt[i, 0]
		v1 = ygt[i, 1:]
		c2 = yhat[i, 0]
		v2 = yhat[i, 1:]
		# get correponding angle difference - (c1, -v1) . (c2, v2)
		tmp = np.clip(c1 * c2 + np.sum(v1 * v2), -1.0, 1.0)
		theta = 2.0 * np.arccos(np.abs(tmp))
		# convert into degrees
		az_error[i] = np.rad2deg(theta)
	medErr = np.median(az_error)
	maxErr = np.amax(az_error)
	acc = 100 * np.sum(az_error < 30) / az_error.size
	print('Error stats- Median: {0}, Max: {1}, <30: {2}'.format(medErr, maxErr, acc))
	return acc, medErr, az_error


# function to get angle error between gt and predicted quaternions
def get_error2(ygt, yhat, labels, num):
	N = ygt.shape[0]
	err = np.zeros(N)
	for i in range(N):
		# read the quaternion vectors in (c, v) format
		c1 = ygt[i, 0]
		v1 = ygt[i, 1:]
		c2 = yhat[i, 0]
		v2 = yhat[i, 1:]
		# get correponding angle difference - (c1, -v1) . (c2, v2)
		tmp = np.clip(c1 * c2 + np.sum(v1 * v2), -1.0, 1.0)
		theta = 2.0 * np.arccos(np.abs(tmp))
		# convert into degrees
		err[i] = np.rad2deg(theta)
	labels = np.squeeze(labels)
	medErr = np.zeros(num)
	for i in range(num):
		ind = (labels == i)
		# print(ind.shape)
		medErr[i] = np.median(err[ind])
	# print('Median Angle Error: ', medErr)
	return np.mean(medErr)


def convert_dictionary(axisangle_dict):
	N = axisangle_dict.shape[0]
	quaternion_dict = np.zeros((N, 4))
	for i in range(N):
		x = axisangle_dict[i]
		angle = np.linalg.norm(x)
		if angle > eps:
			axis = x/angle
		else:
			axis = np.zeros(3)
		y = np.array([np.cos(angle/2.), np.sin(angle/2.)*axis[0], np.sin(angle/2.)*axis[1], np.sin(angle/2.)*axis[2]])
		y = y / np.linalg.norm(y)
		quaternion_dict[i] = y
	return quaternion_dict


"""
Pose models
"""


# 3 fully connected layers
class model_3layer(nn.Module):

	def __init__(self, N0, N1, N2):
		super().__init__()
		self.fc1 = nn.Linear(N0, N1, bias=False)
		self.bn1 = nn.BatchNorm1d(N1)
		self.fc2 = nn.Linear(N1, N2, bias=False)
		self.bn2 = nn.BatchNorm1d(N2)
		self.fc3 = nn.Linear(N2, 4)

	def forward(self, x):
		x = F.relu(self.bn1(self.fc1(x)))
		x = F.relu(self.bn2(self.fc2(x)))
		x = F.normalize(F.tanh(self.fc3(x)))
		return x


# 2 fully connected layers
class model_2layer(nn.Module):

	def __init__(self, N0, N1):
		super().__init__()
		self.fc1 = nn.Linear(N0, N1, bias=False)
		self.bn1 = nn.BatchNorm1d(N1)
		self.fc2 = nn.Linear(N1, 4)

	def forward(self, x):
		x = F.relu(self.bn1(self.fc1(x)))
		x = F.normalize(F.tanh(self.fc2(x)))
		return x


# 1 fully connected layer
class model_1layer(nn.Module):

	def __init__(self, N0):
		super().__init__()
		self.fc = nn.Linear(N0, 4)

	def forward(self, x):
		x = F.normalize(F.tanh(self.fc(x)))
		return x

"""
Loss function
"""


class geodesic_loss(nn.Module):

	def __init__(self, reduce=True):
		super().__init__()
		self.eps = eps
		self.reduce = reduce

	def forward(self, ypred, ytrue):
		ypred = F.normalize(ypred)
		tmp = torch.abs(torch.sum(ytrue*ypred, dim=1))
		theta = 2.0*torch.acos(torch.clamp(tmp, -1+self.eps, 1-self.eps))
		if self.reduce:
			return torch.mean(theta)
		else:
			return theta
