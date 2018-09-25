# -*- coding: utf-8 -*-
"""
Function that learns feature model + 3layer pose models x 12 object categories
in an end-to-end manner by minimizing the mean squared error for axis-angle representation
"""

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Optimizer

from featureModels import resnet_model
from axisAngle import get_error2, geodesic_loss
from poseModels import model_3layer
from helperFunctions import classes
from dataGenerators import ImagesAll, TestImages, my_collate

import numpy as np
import scipy.io as spio
import gc
import os
import re
import progressbar
import argparse
from tensorboardX import SummaryWriter
import time

parser = argparse.ArgumentParser(description='Pure Regression Models')
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--db_type', type=str, default='clean')
parser.add_argument('--save_str', type=str)
parser.add_argument('--ydata_type', type=str, default='axis_angle')
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--nonlinearity', type=str, default='valid')
parser.add_argument('--num_epochs', type=int, default=9)
args = parser.parse_args()
print(args)
# assign GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

model_file = os.path.join('models', args.save_str + '.tar')
results_dir = os.path.join('results', args.save_str)
plots_file = os.path.join('plots', args.save_str)
log_dir = os.path.join('logs', args.save_str)
os.mkdir(results_dir)


# to handle synset_str with _ in it
def parse_name2(image_name):
	ind = [match.start() for match in re.finditer('_', image_name)]
	synset_str = image_name[:ind[-5]]
	model_str = image_name[ind[-5]+1:ind[-4]]
	az = float(image_name[ind[-4]+2:ind[-3]])
	el = float(image_name[ind[-3]+2:ind[-2]])
	ct = float(image_name[ind[-2]+2:ind[-1]])
	d = float(image_name[ind[-1]+2:])
	return synset_str, model_str, az, el, ct, d


def myProj(x):
	angle = torch.norm(x, 2, 1)
	axis = F.normalize(x)
	angle = torch.fmod(angle, 2*np.pi)
	return angle*axis


# my model for pose estimation: feature model + 1layer pose model x 12
class my_model(nn.Module):
	def __init__(self):
		super().__init__()
		self.num_classes = num_classes
		self.feature_model = resnet_model('resnet50', 'layer4').cuda()
		self.pose_models = nn.ModuleList([model_3layer(N0, N1, N2, ndim) for i in range(self.num_classes)]).cuda()

	def forward(self, x, label):
		x = self.feature_model(x)
		x = torch.stack([self.pose_models[i](x) for i in range(self.num_classes)]).permute(1, 2, 0)
		label = torch.zeros(label.size(0), self.num_classes).scatter_(1, label.data.cpu(), 1.0)
		label = Variable(label.unsqueeze(2).cuda())
		y = torch.squeeze(torch.bmm(x, label), 2)
		if args.nonlinearity == 'valid':
			y = np.pi*F.tanh(y)
		elif args.nonlinearity == 'correct':
			y = myProj(y)
		else:
			pass
		del x, label
		return y


# Implements variation of SGD (optionally with momentum)
class mySGD(Optimizer):

	def __init__(self, params, c, alpha1=1e-6, alpha2=1e-7, momentum=0, dampening=0, weight_decay=0, nesterov=False):
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


if args.db_type == 'clean':
	db_path = 'data/flipped_new/test'
else:
	db_path = 'data/flipped_all/test'
	parse_name = parse_name2
num_classes = len(classes)

# DATA
train_data = ImagesAll(db_path, 'real', args.ydata_type)
test_data = TestImages(db_path, args.ydata_type)
train_loader = DataLoader(train_data, batch_size=args.num_workers, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=my_collate)
test_loader = DataLoader(test_data, batch_size=32)
print('Train: {0} \t Test: {1}'.format(len(train_loader), len(test_loader)))

# MODEL
N0, N1, N2 = 2048, 1000, 500
if args.ydata_type == 'axis_angle':
	ndim = 3
else:
	ndim = 4
model = my_model()
model.load_state_dict(torch.load(model_file))
# print(model)
criterion = geodesic_loss().cuda()
optimizer = mySGD(model.parameters(), c=2*len(train_loader))
# store stuff
writer = SummaryWriter(log_dir)
count = 0
val_loss = []
num_ensemble = 0


def training():
	global count, val_loss, num_ensemble
	model.train()
	bar = progressbar.ProgressBar(max_value=len(train_loader))
	for i, sample in enumerate(train_loader):
		# forward steps
		xdata = Variable(sample['xdata'].cuda())
		label = Variable(sample['label'].cuda())
		ydata = Variable(sample['ydata'].cuda())
		output = model(xdata, label)
		loss = criterion(output, ydata)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# store
		writer.add_scalar('train_loss', loss.item(), count)
		if i % 500 == 0:
			ytest, yhat_test, test_labels = testing()
			tmp_val_loss = get_error2(ytest, yhat_test, test_labels, num_classes)
			writer.add_scalar('val_loss', tmp_val_loss, count)
			val_loss.append(tmp_val_loss)
		if count % optimizer.c == optimizer.c/2:
			ytest, yhat_test, test_labels = testing()
			num_ensemble += 1
			results_file = os.path.join(results_dir, 'num'+str(num_ensemble))
			spio.savemat(results_file, {'ytest': ytest, 'yhat_test': yhat_test, 'test_labels': test_labels})
		count += 1
		# cleanup
		del xdata, label, ydata, output, loss, sample
		bar.update(i)
	train_loader.dataset.shuffle_images()


def testing():
	model.eval()
	bar = progressbar.ProgressBar(max_value=len(test_loader))
	ypred = []
	ytrue = []
	labels = []
	for i, sample in enumerate(test_loader):
		xdata = Variable(sample['xdata'].cuda())
		label = Variable(sample['label'].cuda())
		output = model(xdata, label)
		ypred.append(output.data.cpu().numpy())
		ytrue.append(sample['ydata'].numpy())
		labels.append(sample['label'].numpy())
		bar.update(i)
		del xdata, label, output, sample
		gc.collect()
	ypred = np.concatenate(ypred)
	ytrue = np.concatenate(ytrue)
	labels = np.concatenate(labels)
	model.train()
	return ytrue, ypred, labels


ytest, yhat_test, test_labels = testing()
print('\nMedErr: {0}'.format(get_error2(ytest, yhat_test, test_labels, num_classes)))

for epoch in range(args.num_epochs):
	tic = time.time()
	# training step
	training()
	# validation
	ytest, yhat_test, test_labels = testing()
	tmp_val_loss = get_error2(ytest, yhat_test, test_labels, num_classes)
	print('\nMedErr: {0}'.format(tmp_val_loss))
	writer.add_scalar('val_loss', tmp_val_loss, count)
	val_loss.append(tmp_val_loss)
	# time and output
	toc = time.time() - tic
	print('Epoch: {0} done in time {1}s'.format(epoch, toc))
	# cleanup
	gc.collect()
writer.close()
val_loss = np.stack(val_loss)
spio.savemat(plots_file, {'val_loss': val_loss})
