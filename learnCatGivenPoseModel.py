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
args = parser.parse_args()
print(args)
# assign GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

# save stuff here
init_model_file = os.path.join('models', args.save_str + '.tar')
model_file = os.path.join('models', args.save_str + '_cat.tar')
results_dir = os.path.join('results', args.save_str + '_cat_' + args.db_type)
plots_file = os.path.join('plots', args.save_str + '_cat_' + args.db_type)
log_dir = os.path.join('logs', args.save_str + '_cat_' + args.db_type)
if not os.path.exists(results_dir):
	os.mkdir(results_dir)

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
orig_model.load_state_dict(torch.load(model_file))


class JointCatPoseModel(nn.Module):
	def __init__(self, oracle_model):
		super().__init__()
		self.oracle_model = oracle_model
		self.fc = nn.Linear(N0, num_classes)

	def forward(self, x):
		x = self.oracle_model.feature_model(x)
		y = self.fc(x)
		return y


model = JointCatPoseModel(orig_model)
# freeze the feature+pose part
for param in model.oracle_model.parameters():
	param.requires_grad = False
# print(model)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda ep: 1/(1+ep))
writer = SummaryWriter(log_dir)
count = 0
val_acc = []


def training():
	global count, val_loss
	model.train()
	bar = progressbar.ProgressBar(max_value=len(train_loader))
	for i, sample in enumerate(train_loader):
		# forward steps
		# output
		xdata = Variable(sample['xdata'].cuda())
		label = Variable(sample['label'].cuda())
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
		if i % 500 == 0:
			gt_labels, pred_labels = testing()
			tmp_acc = get_accuracy(gt_labels, pred_labels)
			writer.add_scalar('val_acc', tmp_acc, count)
			val_acc.append(tmp_acc)
		# cleanup
		del xdata, label, output, loss
		bar.update(i)
	train_loader.dataset.shuffle_images()


def testing():
	model.eval()
	gt_labels = []
	pred_labels = []
	for i, sample in enumerate(test_loader):
		xdata = Variable(sample['xdata'].cuda())
		output = model(xdata)
		tmp_labels = np.argmax(output.data.cpu().numpy(), axis=1)
		pred_labels.append(tmp_labels)
		label = Variable(sample['label'].cuda())
		gt_labels.append(sample['label'].numpy())
		del xdata, label, output, sample
		gc.collect()
	gt_labels = np.concatenate(gt_labels)
	pred_labels = np.concatenate(pred_labels)
	model.train()
	return gt_labels, pred_labels


def save_checkpoint(filename):
	torch.save(model.state_dict(), filename)


def get_accuracy(ytrue, ypred):
	# print(ytrue.shape, ypred.shape)
	acc = np.zeros(num_classes)
	for i in range(num_classes):
		acc[i] = np.sum((ytrue == i)*(ypred == i))/np.sum(ytrue == i)
	# print(acc)
	print('Mean: {0}'.format(np.mean(acc)))
	return np.mean(acc)[0]


gt_labels, pred_labels = testing()
tmp_acc = get_accuracy(gt_labels, pred_labels)
print('\nAcc: {0}'.format(tmp_acc))

for epoch in range(args.num_epochs):
	tic = time.time()
	scheduler.step()
	# training step
	training()
	# validation
	gt_labels, pred_labels = testing()
	tmp_acc = get_accuracy(gt_labels, pred_labels)
	print('\nAcc: {0}'.format(tmp_acc))
	writer.add_scalar('val_acc', tmp_acc, count)
	val_acc.append(tmp_acc)
	# time and output
	toc = time.time() - tic
	print('Epoch: {0} done in time {1}s'.format(epoch, toc))
	# cleanup
	gc.collect()
writer.close()
val_loss = np.stack(val_loss)
spio.savemat(plots_file, {'val_loss': val_loss})
