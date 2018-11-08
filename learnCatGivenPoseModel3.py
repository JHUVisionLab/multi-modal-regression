# -*- coding: utf-8 -*-
"""
Category given Pose model starting with Geodesic Regression model for the axis-angle representation
"""

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataGenerators import TestImages, my_collate, ImagesAll
from poseModels import model_3layer
from featureModels import resnet_model
from helperFunctions import classes

import numpy as np
import scipy.io as spio
import gc
import os
import time
import progressbar
import argparse
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Geodesic Regression Model')
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--save_str', type=str)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--db_type', type=str, default='clean')
parser.add_argument('--init_lr', type=float, default=1e-4)
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

# relevant variables
ndim = 3
N0, N1, N2 = 2048, 1000, 500
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
train_data = ImagesAll(train_path, 'real')
test_data = TestImages(test_path)
# setup data loaders
train_loader = DataLoader(train_data, batch_size=args.num_workers, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=my_collate)
test_loader = DataLoader(test_data, batch_size=32)
print('Train: {0} \t Test: {1}'.format(len(train_loader), len(test_loader)))


# my_model
class RegressionModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.num_classes = num_classes
		self.ndim = ndim
		self.feature_model = resnet_model('resnet50', 'layer4').cuda()
		self.pose_models = nn.ModuleList([model_3layer(N0, N1, N2, ndim) for i in range(self.num_classes)]).cuda()

	def forward(self, x, label):
		x = self.feature_model(x)
		x = torch.stack([self.pose_models[i](x) for i in range(self.num_classes)]).permute(1, 2, 0)
		label = torch.zeros(label.size(0), self.num_classes).scatter_(1, label.data.cpu(), 1.0)
		label = Variable(label.unsqueeze(2).cuda())
		y = torch.squeeze(torch.bmm(x, label), 2)
		y = np.pi*F.tanh(y)
		del x, label
		return y


orig_model = RegressionModel()
orig_model.load_state_dict(torch.load(init_model_file))


class JointCatPoseModel(nn.Module):
	def __init__(self, oracle_model):
		super().__init__()
		# old stuff
		self.num_classes = oracle_model.num_classes
		self.ndim = oracle_model.ndim
		self.feature_model = oracle_model.feature_model
		self.pose_models = oracle_model.pose_models
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
model.pose_models.eval()
for param in model.pose_models.parameters():
	param.requires_grad = False
# print(model)


def my_schedule(ep):
	return 1. / (1. + ep)


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
	print(acc)
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
