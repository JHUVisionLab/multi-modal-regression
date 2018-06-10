import pickle
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from quaternion import convert_dictionary
from helperFunctions import eps


"""
Loss function
"""


class SimpleLoss(nn.Module):
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


class GeodesicLoss(nn.Module):
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


class GeodesicLossQ(nn.Module):
	def __init__(self, alpha, kmeans_file, my_loss=None):
		super().__init__()
		self.alpha = alpha
		kmeans = pickle.load(open(kmeans_file, 'rb'))
		self.cluster_centers_ = Variable(torch.from_numpy(convert_dictionary(kmeans.cluster_centers_)).float()).cuda()
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


class SimpleRelaXedLoss(nn.Module):
	def __init__(self, alpha):
		super().__init__()
		self.alpha = alpha
		self.mse = nn.MSELoss().cuda()
		self.kl = nn.KLDivLoss().cuda()

	def forward(self, ypred, ytrue):
		# ytrue = [ydata_bin, ydata_res]
		# ypred = [score, residual]
		l1 = self.kl(F.log_softmax(ypred[0], dim=1), ytrue[0])
		l2 = self.mse(ypred[1], ytrue[1])
		return torch.add(l1, self.alpha, l2)


class RelaXedLoss(nn.Module):
	def __init__(self, alpha, kmeans_file, my_loss):
		super().__init__()
		self.alpha = alpha
		kmeans = pickle.load(open(kmeans_file, 'rb'))
		self.cluster_centers_ = Variable(torch.from_numpy(kmeans.cluster_centers_).float()).cuda()
		self.my_loss = my_loss
		self.kl = nn.KLDivLoss().cuda()

	def forward(self, ypred, ytrue):
		# ytrue = (ydata_label, ydata)
		# ypred = (score, residual)
		l1 = self.kl(F.log_softmax(ypred[0], dim=1), ytrue[0])
		_, ind = torch.max(ypred[0], dim=1)
		y = torch.index_select(self.cluster_centers_, 0, ind)
		l2 = self.my_loss(y+ypred[1], ytrue[1])
		return torch.add(l1, self.alpha, l2)


class RelaXedProbabilisticLoss(nn.Module):
	def __init__(self, alpha, kmeans_file, my_loss):
		super().__init__()
		self.alpha = alpha
		kmeans = pickle.load(open(kmeans_file, 'rb'))
		self.cluster_centers = Variable(torch.from_numpy(kmeans.cluster_centers_).float()).cuda()
		self.n_clusters = kmeans.n_clusters
		self.my_loss = my_loss
		self.kl = nn.KLDivLoss().cuda()

	def forward(self, ypred, ytrue):
		# ytrue = (ydata_prob, ydata)
		# ypred = (score, residual)
		l1 = self.kl(F.log_softmax(ypred[0], dim=1), ytrue[0])
		l2 = torch.stack([self.my_loss(ytrue[1], torch.add(ypred[1], 1.0, self.cluster_centers.index_select(0, Variable(i*torch.ones(1).long().cuda()))))
		                  for i in range(self.n_clusters)])
		l2 = torch.mean(torch.sum(torch.mul(F.softmax(ypred[0], dim=1), torch.t(l2)), dim=1))
		return torch.add(l1, self.alpha, l2)


class ProbabilisticLoss(nn.Module):
	def __init__(self, alpha, kmeans_file, my_loss):
		super().__init__()
		self.alpha = alpha
		kmeans = pickle.load(open(kmeans_file, 'rb'))
		self.cluster_centers = Variable(torch.from_numpy(kmeans.cluster_centers_).float()).cuda()
		self.n_clusters = kmeans.n_clusters
		self.my_loss = my_loss
		self.ce = nn.CrossEntropyLoss().cuda()

	def forward(self, ypred, ytrue):
		# ytrue = (ydata_prob, ydata)
		# ypred = (score, residual)
		l1 = self.ce(ypred[0], ytrue[0])
		l2 = torch.stack([self.my_loss(ytrue[1], torch.add(ypred[1], 1.0, self.cluster_centers.index_select(0, Variable(i*torch.ones(1).long().cuda()))))
		                  for i in range(self.n_clusters)])
		l2 = torch.mean(torch.sum(torch.mul(F.softmax(ypred[0], dim=1), torch.t(l2)), dim=1))
		return torch.add(l1, self.alpha, l2)


class RelaXedProbabilisticLossQ(nn.Module):
	def __init__(self, alpha, kmeans_file, my_loss):
		super().__init__()
		self.alpha = alpha
		kmeans = pickle.load(open(kmeans_file, 'rb'))
		self.cluster_centers = Variable(torch.from_numpy(convert_dictionary(kmeans.cluster_centers_)).float()).cuda()
		self.n_clusters = kmeans.n_clusters
		self.my_loss = my_loss
		self.kl = nn.KLDivLoss().cuda()

	def forward(self, ypred, ytrue):
		# ytrue = (ydata_prob, ydata)
		# ypred = (score, residual)
		l1 = self.kl(F.log_softmax(ypred[0], dim=1), ytrue[0])
		l2 = torch.stack([self.my_loss(ytrue[1], torch.add(ypred[1], 1.0, self.cluster_centers.index_select(0, Variable(i*torch.ones(1).long().cuda()))))
		                  for i in range(self.n_clusters)])
		l2 = torch.mean(torch.sum(torch.mul(F.softmax(ypred[0], dim=1), torch.t(l2)), dim=1))
		return torch.add(l1, self.alpha, l2)


class RelaXedProbabilisticMultiresLoss(RelaXedProbabilisticLoss):
	def __init__(self, alpha, kmeans_file, my_loss):
		super().__init__(alpha, kmeans_file, my_loss)

	def forward(self, ypred, ytrue):
		# ytrue = (ydata_prob, ydata)
		# ypred = (score, residual)
		l1 = self.kl(F.log_softmax(ypred[0], dim=1), ytrue[0])
		y = self.cluster_centers + ypred[1]
		l2 = torch.stack([self.my_loss(ytrue[1], torch.squeeze(y.index_select(1, Variable(i*torch.ones(1).long().cuda())))) for i in range(self.n_clusters)])
		l2 = torch.mean(torch.sum(torch.mul(F.softmax(ypred[0], dim=1), torch.t(l2)), dim=1))
		return torch.add(l1, self.alpha, l2)


class ProbabilisticMultiresLoss(ProbabilisticLoss):
	def __init__(self, alpha, kmeans_file, my_loss):
		super().__init__(alpha, kmeans_file, my_loss)

	def forward(self, ypred, ytrue):
		# ytrue = (ydata_prob, ydata)
		# ypred = (score, residual)
		l1 = self.ce(ypred[0], ytrue[0])
		y = self.cluster_centers + ypred[1]
		l2 = torch.stack([self.my_loss(ytrue[1], torch.squeeze(y.index_select(1, Variable(i*torch.ones(1).long().cuda())))) for i in range(self.n_clusters)])
		l2 = torch.mean(torch.sum(torch.mul(F.softmax(ypred[0], dim=1), torch.t(l2)), dim=1))
		return torch.add(l1, self.alpha, l2)


class RelaXedProbabilisticMultiresLossQ(RelaXedProbabilisticLossQ):
	def __init__(self, alpha, kmeans_file, my_loss):
		super().__init__(alpha, kmeans_file, my_loss)

	def forward(self, ypred, ytrue):
		# ytrue = (ydata_prob, ydata)
		# ypred = (score, residual)
		l1 = self.kl(F.log_softmax(ypred[0], dim=1), ytrue[0])
		y = self.cluster_centers + ypred[1]
		l2 = torch.stack([self.my_loss(ytrue[1], torch.squeeze(y.index_select(1, Variable(i*torch.ones(1).long().cuda())))) for i in range(self.n_clusters)])
		l2 = torch.mean(torch.sum(torch.mul(F.softmax(ypred[0], dim=1), torch.t(l2)), dim=1))
		return torch.add(l1, self.alpha, l2)


class RiemannianLoss(nn.Module):
	def __init__(self, alpha, pose_dict):
		super().__init__()
		self.alpha = alpha
		self.key_poses = torch.from_numpy(pose_dict).float().cuda()
		self.ce = nn.CrossEntropyLoss().cuda()
		proj = np.array([[0,0,0,0,0,-1,0,1,0], [0,0,1,0,0,0,-1,0,0], [0,-1,0,1,0,0,0,0,0]])
		self.proj = torch.from_numpy(proj).float().cuda()
		self.Id = torch.eye(3).float().cuda()

	def my_loss(self, ypred, ytrue):
		# geodesic loss between predicted and gt rotations
		tmp = torch.stack([torch.trace(torch.mm(ypred[i].t(), ytrue[i])) for i in range(ytrue.size(0))])
		angle = torch.acos(torch.clamp((tmp - 1.0) / 2, -1 + eps, 1 - eps))
		return torch.mean(angle)

	def forward(self, ypred, ytrue):
		# ytrue = (ydata_bin, ydata_rot)
		# ypred = (score, residual)
		l1 = self.ce(ypred[0], ytrue[0])
		_, ind = torch.max(ypred[0], dim=1)
		yres = ypred[1]
		angle = torch.norm(yres, 2, 1)
		axis = F.normalize(yres)
		axis = torch.mm(axis, self.proj).view(-1, 3, 3)
		y = torch.stack([self.Id + torch.sin(angle[i])*axis[i] + (1.0 - torch.cos(angle[i]))*torch.mm(axis[i], axis[i]) for i in range(angle.size(0))])
		y = torch.bmm(torch.index_select(self.key_poses, 0, ind), y)
		l2 = self.my_loss(y, ytrue[1])
		return torch.add(l1, self.alpha, l2)


# loss
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
