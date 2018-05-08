# -*- coding: utf-8 -*-
"""
Pose models and loss functions for multibin representation
"""
import pickle
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

"""
Pose models
"""
class bin_1layer(nn.Module):
	def __init__(self, N0, num_clusters):
		super().__init__()
		self.fc = nn.Linear(N0, num_clusters)
	def forward(self, x):
		x = self.fc(x)
		return x

class res_1layer(nn.Module):
	def __init__(self, N0, ndim):
		super().__init__()
		self.fc = nn.Linear(N0, ndim)
	def forward(self, x):
		x = self.fc(x)
		return x

class bin_2layer(nn.Module):
	def __init__(self, N0, N1, num_clusters):
		super().__init__()
		self.fc1 = nn.Linear(N0, N1, bias=False)
		self.bn1 = nn.BatchNorm1d(N1)
		self.fc2 = nn.Linear(N1, num_clusters)
	def forward(self, x):
		x = F.relu(self.bn1(self.fc1(x)))
		x = self.fc2(x)
		return x

class res_2layer(nn.Module):
	def __init__(self, N0, N1, ndim):
		super().__init__()
		self.fc1 = nn.Linear(N0, N1, bias=False)
		self.bn1 = nn.BatchNorm1d(N1)
		self.fc2 = nn.Linear(N1, ndim)
	def forward(self, x):
		x = F.relu(self.bn1(self.fc1(x)))
		x = self.fc2(x)
		return x

class bin_3layer(nn.Module):
	def __init__(self, N0, N1, N2, num_clusters):
		super().__init__()
		self.fc1 = nn.Linear(N0, N1, bias=False)
		self.bn1 = nn.BatchNorm1d(N1)
		self.fc2 = nn.Linear(N1, N2, bias=False)
		self.bn2 = nn.BatchNorm1d(N2)
		self.fc3 = nn.Linear(N2, num_clusters)
	def forward(self, x):
		x = F.relu(self.bn1(self.fc1(x)))
		x = F.relu(self.bn2(self.fc2(x)))
		x = self.fc3(x)
		return x

class res_3layer(nn.Module):
	def __init__(self, N0, N1, N2, ndim):
		super().__init__()
		self.fc1 = nn.Linear(N0, N1, bias=False)
		self.bn1 = nn.BatchNorm1d(N1)
		self.fc2 = nn.Linear(N1, N2, bias=False)
		self.bn2 = nn.BatchNorm1d(N2)
		self.fc3 = nn.Linear(N2, ndim)
	def forward(self, x):
		x = F.relu(self.bn1(self.fc1(x)))
		x = F.relu(self.bn2(self.fc2(x)))
		x = self.fc3(x)
		return x


"""
Loss function
"""


class loss_m0(nn.Module):
	def __init__(self, alpha):
		super().__init__()
		self.alpha = alpha
		self.mse = nn.MSELoss().cuda()
		self.ce = nn.CrossEntropyLoss().cuda()

	def forward(self, ypred, ytrue):
		# ytrue = [ydata_bin, ydata_res]
		# ypred = [score, residual]
		l1 = self.ce(ypred[0], ytrue[0])
		l2 = self.mse(ypred[1], ytrue[1])
		return torch.add(l1, self.alpha, l2)


class loss_m1(nn.Module):
	def __init__(self, alpha, kmeans_file, my_loss=None):
		super().__init__()
		self.alpha = alpha
		kmeans = pickle.load(open(kmeans_file, 'rb'))
		self.cluster_centers_ = Variable(torch.from_numpy(kmeans.cluster_centers_).float()).cuda()
		if my_loss is None:
			self.mse = nn.MSELoss().cuda()
		else:
			self.mse = my_loss
		self.ce = nn.CrossEntropyLoss().cuda()

	def forward(self, ypred, ytrue):
		# ytrue = (ydata_label, ydata)
		# ypred = (score, residual)
		l1 = self.ce(ypred[0], ytrue[0])
		_, ind = torch.max(ypred[0], dim=1)
		y = torch.index_select(self.cluster_centers_, 0, ind)
		l2 = self.mse(y+ypred[1], ytrue[1])
		return torch.add(l1, self.alpha, l2)


# loss
class loss_m2(nn.Module):
	def __init__(self, alpha, num_clusters):
		super().__init__()
		self.alpha = alpha
		self.num_clusters = num_clusters
		self.mse = nn.MSELoss().cuda()
		self.ce = nn.CrossEntropyLoss().cuda()

	def forward(self, ypred, ytrue):
		# ytrue = (ydata_label, ydata_res)
		# ypred = (score, residual)
		l1 = self.ce(ypred[0], ytrue[0])
		_, label = torch.max(ypred[0], dim=1)
		label = torch.zeros(label.size(0), self.num_clusters).scatter_(1, label.unsqueeze(1).data.cpu(), 1.0)
		label = Variable(label.unsqueeze(2).cuda())
		yres = torch.squeeze(torch.bmm(ytrue[1].permute(0, 2, 1), label), 2)
		l2 = self.mse(ypred[1], yres)
		return torch.add(l1, self.alpha, l2)


class loss_m3(nn.Module):
	def __init__(self, alpha, kmeans_file, my_loss=None):
		super().__init__()
		self.alpha = alpha
		kmeans = pickle.load(open(kmeans_file, 'rb'))
		self.cluster_centers = Variable(torch.from_numpy(kmeans.cluster_centers_).float()).cuda()
		self.n_clusters = kmeans.n_clusters
		if my_loss is None:
			self.mse = nn.MSELoss(reduce=False).cuda()
		else:
			self.mse = my_loss
		self.kl = nn.KLDivLoss().cuda()

	def forward(self, ypred, ytrue):
		# ytrue = (ydata_prob, ydata)
		# ypred = (score, residual)
		l1 = self.kl(F.log_softmax(ypred[0], dim=1), ytrue[0])
		l2 = torch.stack([self.mse(ytrue[1], torch.add(ypred[1], 1.0, self.cluster_centers.index_select(0, Variable(i*torch.ones(1).long().cuda()))))
		                  for i in range(self.n_clusters)])
		l2 = torch.mean(torch.sum(torch.mul(F.softmax(ypred[0], dim=1), torch.t(l2)), dim=1))
		return torch.add(l1, self.alpha, l2)


class loss_m4(loss_m3):
	def __init__(self, alpha, kmeans_file, my_loss=None):
		super().__init__(alpha, kmeans_file, my_loss)

	def forward(self, ypred, ytrue):
		# ytrue = (ydata_prob, ydata)
		# ypred = (score, residual)
		l1 = self.kl(F.log_softmax(ypred[0], dim=1), ytrue[0])
		y = self.cluster_centers + ypred[1]
		l2 = torch.stack([self.mse(ytrue[1], torch.squeeze(y.index_select(1, Variable(i*torch.ones(1).long().cuda())))) for i in range(self.n_clusters)])
		l2 = torch.mean(torch.sum(torch.mul(F.softmax(ypred[0], dim=1), torch.t(l2)), dim=1))
		return torch.add(l1, self.alpha, l2)
