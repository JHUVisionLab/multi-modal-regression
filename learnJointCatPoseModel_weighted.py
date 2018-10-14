# -*- coding: utf-8 -*-
"""
Joint Cat & Pose model (Weighted) with Geodesic Bin and Delta model for the axis-angle representation
"""

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataGenerators import TestImages, my_collate
from binDeltaGenerators import GBDGenerator
from binDeltaModels import OneBinDeltaModel, OneDeltaPerBinModel
from axisAngle import get_error2, geodesic_loss
from helperFunctions import classes

import numpy as np
import math
import scipy.io as spio
import gc
import os
import time
import progressbar
import pickle
import argparse
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Geodesic Bin & Delta Model')
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--save_str', type=str)
parser.add_argument('--dict_size', type=int, default=200)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--feature_network', type=str, default='resnet')
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--multires', type=bool, default=False)
parser.add_argument('--db_type', type=str, default='clean')
parser.add_argument('--init_lr', type=float, default=1e-5)
args = parser.parse_args()
print(args)
# assign GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

# save stuff here
init_model_file = os.path.join('models', args.save_str + '_cat.tar')
model_file = os.path.join('models', args.save_str + '_wgt.tar')
results_file = os.path.join('results', args.save_str + '_wgt_' + args.db_type)
plots_file = os.path.join('plots', args.save_str + '_wgt_' + args.db_type)
log_dir = os.path.join('logs', args.save_str + '_wgt_' + args.db_type)

# kmeans data
kmeans_file = 'data/kmeans_dictionary_axis_angle_' + str(args.dict_size) + '.pkl'
kmeans = pickle.load(open(kmeans_file, 'rb'))
kmeans_dict = kmeans.cluster_centers_
cluster_centers_ = Variable(torch.from_numpy(kmeans_dict).float()).cuda()
num_clusters = kmeans.n_clusters

# relevant variables
ndim = 3
N0, N1, N2, N3 = 2048, 1000, 500, 100
num_classes = len(classes)
if args.db_type == 'clean':
	db_path = 'data/flipped_new'
else:
	db_path = 'data/flipped_all'
num_classes = len(classes)
train_path = os.path.join(db_path, 'train')
test_path = os.path.join(db_path, 'test')

# loss
ce_loss = nn.CrossEntropyLoss().cuda()
gve_loss = geodesic_loss().cuda()

# DATA
# datasets
train_data = GBDGenerator(train_path, 'real', kmeans_file)
test_data = TestImages(test_path)
# setup data loaders
train_loader = DataLoader(train_data, batch_size=args.num_workers, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=my_collate)
test_loader = DataLoader(test_data, batch_size=32)
print('Train: {0} \t Test: {1}'.format(len(train_loader), len(test_loader)))

# my_model
if not args.multires:
	orig_model = OneBinDeltaModel(args.feature_network, num_classes, num_clusters, N0, N1, N2, ndim)
else:
	orig_model = OneDeltaPerBinModel(args.feature_network, num_classes, num_clusters, N0, N1, N2, N3, ndim)


class JointCatPoseModel(nn.Module):
	def __init__(self, oracle_model):
		super().__init__()
		# old stuff
		self.num_classes = oracle_model.num_classes
		self.num_clusters = oracle_model.num_clusters
		self.ndim = oracle_model.ndim
		self.feature_model = oracle_model.feature_model
		self.bin_models = oracle_model.bin_models
		self.res_models = oracle_model.res_models
		# new stuff
		self.fc = nn.Linear(N0, num_classes).cuda()

	def forward(self, x):
		x = self.feature_model(x)
		y0 = self.fc(x)
		label = torch.unsqueeze(F.softmax(y0, dim=1), dim=2)
		if not args.multires:
			y1 = torch.stack([self.bin_models[i](x) for i in range(self.num_classes)]).permute(1, 2, 0)
			y2 = torch.stack([self.res_models[i](x) for i in range(self.num_classes)]).permute(1, 2, 0)
			y1 = torch.squeeze(torch.bmm(y1, label), 2)
			y2 = torch.squeeze(torch.bmm(y2, label), 2)
		else:
			y1 = torch.stack([self.bin_models[i](x) for i in range(self.num_classes)]).permute(1, 2, 0)
			y2 = torch.stack([self.res_models[i](x) for i in range(self.num_classes * self.num_clusters)])
			y2 = y2.view(self.num_classes, self.num_clusters, -1, self.ndim).permute(1, 2, 3, 0)
			y1 = torch.squeeze(torch.bmm(y1, label), 2)
			y2 = torch.squeeze(torch.matmul(y2, label), 3)
			pose_label = torch.argmax(y1, dim=1, keepdim=True)
			pose_label = torch.zeros(pose_label.size(0), self.num_clusters).scatter_(1, pose_label.data.cpu(), 1.0)
			pose_label = Variable(pose_label.unsqueeze(2).cuda())
			y2 = torch.squeeze(torch.bmm(y2.permute(1, 2, 0), pose_label), 2)
		return [y0, y1, y2]   # cat, pose_bin, pose_delta


model = JointCatPoseModel(orig_model)
model.load_state_dict(torch.load(init_model_file))
# print(model)
optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda ep: 1/(1+ep))
writer = SummaryWriter(log_dir)
count = 0
s = 0
val_err = []
val_acc = []


def training():
	global count, s, val_acc, val_err
	model.train()
	bar = progressbar.ProgressBar(max_value=len(train_loader))
	for i, sample in enumerate(train_loader):
		# forward steps
		# output
		label = Variable(sample['label'].squeeze().cuda())
		ydata_bin = Variable(sample['ydata_bin'].cuda())
		ydata = Variable(sample['ydata'].cuda())
		xdata = Variable(sample['xdata'].cuda())
		output = model(xdata)
		output_cat = output[0]
		output_bin = output[1]
		output_res = output[2]
		# loss
		Lc_cat = ce_loss(output_cat, label)
		Lc_pose = ce_loss(output_bin, ydata_bin)
		ind = torch.argmax(output_bin, dim=1)
		y = torch.index_select(cluster_centers_, 0, ind) + output_res
		Lr = gve_loss(y, ydata)
		loss = Lc_cat + Lc_pose + math.exp(-s)*Lr + s
		# parameter updates
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		s = math.log(Lr)
		# store
		count += 1
		writer.add_scalar('train_loss', loss.item(), count)
		writer.add_scalar('alpha', math.exp(-s), count)
		if i % 500 == 0:
			ytrue_cat, ytrue_pose, ypred_cat, ypred_pose = testing()
			spio.savemat(results_file, {'ytrue_cat': ytrue_cat, 'ytrue_pose': ytrue_pose, 'ypred_cat': ypred_cat, 'ypred_pose': ypred_pose})
			tmp_acc = get_accuracy(ytrue_cat, ypred_cat, num_classes)
			tmp_err = get_error2(ytrue_pose, ypred_pose, ytrue_cat, num_classes)
			writer.add_scalar('val_acc', tmp_acc, count)
			writer.add_scalar('val_err', tmp_err, count)
			val_acc.append(tmp_acc)
			val_err.append(tmp_err)
		# cleanup
		del xdata, label, output, loss, output_cat, output_bin, output_res
		bar.update(i+1)
	train_loader.dataset.shuffle_images()


def testing():
	model.eval()
	ytrue_cat, ytrue_pose = [], []
	ypred_cat, ypred_pose = [], []
	for i, sample in enumerate(test_loader):
		xdata = Variable(sample['xdata'].cuda())
		output = model(xdata)
		output_cat = output[0]
		output_bin = output[1]
		output_res = output[2]
		tmp_labels = np.argmax(output_cat.data.cpu().numpy(), axis=1)
		ypred_cat.append(tmp_labels)
		label = Variable(sample['label'])
		ytrue_cat.append(sample['label'].squeeze().numpy())
		ypred_bin = np.argmax(output_bin.data.cpu().numpy(), axis=1)
		ypred_res = output_res.data.cpu().numpy()
		ypred_pose.append(kmeans_dict[ypred_bin, :] + ypred_res)
		ytrue_pose.append(sample['ydata'].numpy())
		del xdata, label, output, sample, output_cat, output_bin, output_res
		gc.collect()
	ytrue_cat = np.concatenate(ytrue_cat)
	ypred_cat = np.concatenate(ypred_cat)
	ytrue_pose = np.concatenate(ytrue_pose)
	ypred_pose = np.concatenate(ypred_pose)
	model.train()
	return ytrue_cat, ytrue_pose, ypred_cat, ypred_pose


def save_checkpoint(filename):
	torch.save(model.state_dict(), filename)


def get_accuracy(ytrue, ypred, num_classes):
	acc = np.zeros(num_classes)
	for i in range(num_classes):
		acc[i] = np.sum((ytrue == i)*(ypred == i))/np.sum(ytrue == i)
	return np.mean(acc)


ytrue_cat, ytrue_pose, ypred_cat, ypred_pose = testing()
spio.savemat(results_file, {'ytrue_cat': ytrue_cat, 'ytrue_pose': ytrue_pose, 'ypred_cat': ypred_cat, 'ypred_pose': ypred_pose})
tmp_acc = get_accuracy(ytrue_cat, ypred_cat, num_classes)
tmp_err = get_error2(ytrue_pose, ypred_pose, ytrue_cat, num_classes)
print('Acc: {0} \t Err: {1}'.format(tmp_acc, tmp_err))

for epoch in range(args.num_epochs):
	tic = time.time()
	scheduler.step()
	# training step
	training()
	# save model at end of epoch
	save_checkpoint(model_file)
	# validation
	ytrue_cat, ytrue_pose, ypred_cat, ypred_pose = testing()
	spio.savemat(results_file, {'ytrue_cat': ytrue_cat, 'ytrue_pose': ytrue_pose, 'ypred_cat': ypred_cat, 'ypred_pose': ypred_pose})
	tmp_acc = get_accuracy(ytrue_cat, ypred_cat, num_classes)
	tmp_err = get_error2(ytrue_pose, ypred_pose, ytrue_cat, num_classes)
	print('Acc: {0} \t Err: {1}'.format(tmp_acc, tmp_err))
	writer.add_scalar('val_acc', tmp_acc, count)
	writer.add_scalar('val_err', tmp_err, count)
	val_acc.append(tmp_acc)
	val_err.append(tmp_err)
	# time and output
	toc = time.time() - tic
	print('Epoch: {0} done in time {1}s'.format(epoch, toc))
	# cleanup
	gc.collect()
writer.close()
val_acc = np.stack(val_acc)
val_err = np.stack(val_err)
spio.savemat(plots_file, {'val_acc': val_acc, 'val_err': val_err})
