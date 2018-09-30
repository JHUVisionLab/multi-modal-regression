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
# from axisAngle import get_error2, geodesic_loss
from quaternion import get_error2, geodesic_loss
from poseModels import model_3layer
from helperFunctions import classes
from dataGenerators import ImagesAll, TestImages, my_collate

import numpy as np
import scipy.io as spio
import gc
import os
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
results_dir = os.path.join('results', args.save_str + '_' + args.db_type)
plots_file = os.path.join('plots', args.save_str + '_' + args.db_type)
log_dir = os.path.join('logs', args.save_str + '_' + args.db_type)
if not os.path.exists(results_dir):
	os.mkdir(results_dir)


def myProj(x):
	angle = torch.norm(x, 2, 1, True)
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
			# y = myProj(y)
			y = F.normalize(y)
		else:
			pass
		del x, label
		return y


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
				writer.add_scalar('lr', step_size, state['step'])
				p.data.add_(-step_size, d_p)

		return loss


if args.db_type == 'clean':
	db_path = 'data/flipped_new'
else:
	db_path = 'data/flipped_all'
num_classes = len(classes)
train_path = os.path.join(db_path, 'train')
test_path = os.path.join(db_path, 'test')
render_path = 'data/renderforcnn/'

# DATA
real_data = ImagesAll(train_path, 'real', args.ydata_type)
render_data = ImagesAll(render_path, 'render', args.ydata_type)
test_data = TestImages(test_path, args.ydata_type)
real_loader = DataLoader(real_data, batch_size=args.num_workers, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=my_collate)
render_loader = DataLoader(render_data, batch_size=args.num_workers, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=my_collate)
test_loader = DataLoader(test_data, batch_size=32)
print('Real: {0} \t Render: {1} \t Test: {2}'.format(len(real_loader), len(render_loader), len(test_loader)))

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
# criterion = nn.MSELoss().cuda()
optimizer = mySGD(model.parameters(), c=2*len(real_loader))
# store stuff
writer = SummaryWriter(log_dir)
count = 0
val_loss = []
num_ensemble = 0


def training():
	global count, val_loss, num_ensemble
	model.train()
	bar = progressbar.ProgressBar(max_value=len(real_loader))
	for i, (sample_real, sample_render) in enumerate(zip(real_loader, render_loader)):
		# forward steps
		xdata_real = Variable(sample_real['xdata'].cuda())
		label_real = Variable(sample_real['label'].cuda())
		ydata_real = Variable(sample_real['ydata'].cuda())
		output_real = model(xdata_real, label_real)
		loss_real = criterion(output_real, ydata_real)
		xdata_render = Variable(sample_render['xdata'].cuda())
		label_render = Variable(sample_render['label'].cuda())
		ydata_render = Variable(sample_render['ydata'].cuda())
		output_render = model(xdata_render, label_render)
		loss_render = criterion(output_render, ydata_render)
		loss = loss_real + loss_render
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
		count += 1
		if count % optimizer.c == optimizer.c/2:
			ytest, yhat_test, test_labels = testing()
			num_ensemble += 1
			results_file = os.path.join(results_dir, 'num'+str(num_ensemble))
			spio.savemat(results_file, {'ytest': ytest, 'yhat_test': yhat_test, 'test_labels': test_labels})
		# cleanup
		del xdata_real, xdata_render, label_real, label_render, ydata_real, ydata_render
		del output_real, output_render, loss_real, loss_render, sample_real, sample_render, loss
		bar.update(i)
	render_loader.dataset.shuffle_images()
	real_loader.dataset.shuffle_images()


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
results_file = os.path.join(results_dir, 'num'+str(num_ensemble))
spio.savemat(results_file, {'ytest': ytest, 'yhat_test': yhat_test, 'test_labels': test_labels})

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
