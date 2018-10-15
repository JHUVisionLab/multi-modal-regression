# -*- coding: utf-8 -*-
"""
Category given Pose model starting with Geodesic Bin and Delta model for the axis-angle representation
"""

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataGenerators import TestImages, my_collate
from binDeltaGenerators import GBDGenerator
from binDeltaModels import OneBinDeltaModel, OneDeltaPerBinModel
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
parser.add_argument('--init_lr', type=float, default=1e-3)
args = parser.parse_args()
print(args)
# assign GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

# save stuff here
init_model_file = os.path.join('models', args.save_str + '.tar')
model_file = os.path.join('models', args.save_str + '_cat.tar')
results_file = os.path.join('results', args.save_str + '_cat_' + args.db_type)
plots_file = os.path.join('plots', args.save_str + '_cat_' + args.db_type)
log_dir = os.path.join('logs', args.save_str + '_cat_' + args.db_type)

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
orig_model.load_state_dict(torch.load(init_model_file))


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
		y = self.fc(x)
		return y


model = JointCatPoseModel(orig_model)
# freeze the feature+pose part
model.feature_model.eval()
for param in model.feature_model.parameters():
	param.requires_grad = False
model.bin_models.eval()
for param in model.bin_models.parameters():
	param.requires_grad = False
model.res_models.eval()
for param in model.res_models.parameters():
	param.requires_grad = False
# print(model)


def my_schedule(ep):
	return 10**-(ep//10)/(1 + ep % 10)


optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.init_lr)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, my_schedule)
writer = SummaryWriter(log_dir)
count = 0
val_acc = []


def training():
	global count, val_acc
	# model.train()
	bar = progressbar.ProgressBar(max_value=len(train_loader))
	for i, sample in enumerate(train_loader):
		# forward steps
		# output
		xdata = Variable(sample['xdata'].cuda())
		label = Variable(sample['label'].squeeze().cuda())
		output = model(xdata)
		# loss
		loss = ce_loss(output, label)
		# parameter updates
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# store
		count += 1
		writer.add_scalar('train_loss', loss.item(), count)
		# if i % 500 == 0:
		# 	gt_labels, pred_labels = testing()
		# 	spio.savemat(results_file, {'gt_labels': gt_labels, 'pred_labels': pred_labels})
		# 	tmp_acc = get_accuracy(gt_labels, pred_labels, num_classes)
		# 	writer.add_scalar('val_acc', tmp_acc, count)
		# 	val_acc.append(tmp_acc)
		# cleanup
		del xdata, label, output, loss
		bar.update(i)
	train_loader.dataset.shuffle_images()


def testing():
	# model.eval()
	gt_labels = []
	pred_labels = []
	for i, sample in enumerate(test_loader):
		xdata = Variable(sample['xdata'].cuda())
		output = model(xdata)
		tmp_labels = np.argmax(output.data.cpu().numpy(), axis=1)
		pred_labels.append(tmp_labels)
		label = Variable(sample['label'])
		gt_labels.append(sample['label'].squeeze().numpy())
		del xdata, label, output, sample
		gc.collect()
	gt_labels = np.concatenate(gt_labels)
	pred_labels = np.concatenate(pred_labels)
	# model.train()
	return gt_labels, pred_labels


def save_checkpoint(filename):
	torch.save(model.state_dict(), filename)


def get_accuracy(ytrue, ypred, num_classes):
	# print(ytrue.shape, ypred.shape)
	acc = np.zeros(num_classes)
	for i in range(num_classes):
		acc[i] = np.sum((ytrue == i)*(ypred == i))/np.sum(ytrue == i)
	# print(acc)
	# print('Mean: {0}'.format(np.mean(acc)))
	return np.mean(acc)


gt_labels, pred_labels = testing()
spio.savemat(results_file, {'gt_labels': gt_labels, 'pred_labels': pred_labels})
tmp_acc = get_accuracy(gt_labels, pred_labels, num_classes)
print('Acc: {0}'.format(tmp_acc))

for epoch in range(args.num_epochs):
	tic = time.time()
	scheduler.step()
	# training step
	training()
	# save model at end of epoch
	save_checkpoint(model_file)
	# validation
	gt_labels, pred_labels = testing()
	spio.savemat(results_file, {'gt_labels': gt_labels, 'pred_labels': pred_labels})
	tmp_acc = get_accuracy(gt_labels, pred_labels, num_classes)
	print('\nAcc: {0}'.format(tmp_acc))
	writer.add_scalar('val_acc', tmp_acc, count)
	val_acc.append(tmp_acc)
	# time and output
	toc = time.time() - tic
	print('Epoch: {0} done in time {1}s'.format(epoch, toc))
	# cleanup
	gc.collect()
writer.close()
val_acc = np.stack(val_acc)
spio.savemat(plots_file, {'val_acc': val_acc})
