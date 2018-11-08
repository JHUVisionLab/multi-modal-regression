# -*- coding: utf-8 -*-
"""
Independent model based on Geodesic Regression model R_G
"""

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataGenerators import ImagesAll, TestImages, my_collate
from axisAngle import get_error2, geodesic_loss
from poseModels import model_3layer
from helperFunctions import classes
from featureModels import resnet_model

import numpy as np
import scipy.io as spio
import gc
import os
import time
import progressbar
import argparse
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Pure Regression Models')
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--render_path', type=str, default='data/renderforcnn/')
parser.add_argument('--augmented_path', type=str, default='data/augmented2/')
parser.add_argument('--pascal3d_path', type=str, default='data/flipped_new/test/')
parser.add_argument('--save_str', type=str)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--feature_network', type=str, default='resnet')
parser.add_argument('--N0', type=int, default=2048)
parser.add_argument('--N1', type=int, default=1000)
parser.add_argument('--N2', type=int, default=500)
parser.add_argument('--init_lr', type=float, default=1e-4)
parser.add_argument('--num_epochs', type=int, default=3)
args = parser.parse_args()
print(args)
# assign GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

# save stuff here
results_file = os.path.join('results', args.save_str)
model_file = os.path.join('models', args.save_str + '.tar')
plots_file = os.path.join('plots', args.save_str)
log_dir = os.path.join('logs', args.save_str)

# relevant variables
ydata_type = 'axis_angle'
ndim = 3
num_classes = len(classes)

mse_loss = nn.MSELoss().cuda()
gve_loss = geodesic_loss().cuda()
ce_loss = nn.CrossEntropyLoss().cuda()

# DATA
# datasets
real_data = ImagesAll(args.augmented_path, 'real', ydata_type)
render_data = ImagesAll(args.render_path, 'render', ydata_type)
test_data = TestImages(args.pascal3d_path, ydata_type)
# setup data loaders
real_loader = DataLoader(real_data, batch_size=args.num_workers, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=my_collate)
render_loader = DataLoader(render_data, batch_size=args.num_workers, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=my_collate)
test_loader = DataLoader(test_data, batch_size=32)
print('Real: {0} \t Render: {1} \t Test: {2}'.format(len(real_loader), len(render_loader), len(test_loader)))
max_iterations = min(len(real_loader), len(render_loader))


# my_model
class IndependentModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.num_classes = num_classes
		self.feature_model = resnet_model('resnet50', 'layer4').cuda()
		self.pose_model = model_3layer(args.N0, args.N1, args.N2, ndim).cuda()

	def forward(self, x):
		x = self.feature_model(x)
		x = self.pose_model(x)
		x = np.pi*F.tanh(x)
		return x


model = IndependentModel()
# print(model)
# loss and optimizer
optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
# store stuff
writer = SummaryWriter(log_dir)
count = 0
val_loss = []


# OPTIMIZATION functions
def training_init():
	global count, val_loss
	model.train()
	bar = progressbar.ProgressBar(max_value=max_iterations)
	for i, (sample_real, sample_render) in enumerate(zip(real_loader, render_loader)):
		# forward steps
		xdata_real = Variable(sample_real['xdata'].cuda())
		ydata_real = Variable(sample_real['ydata'].cuda())
		output_real = model(xdata_real)
		xdata_render = Variable(sample_render['xdata'].cuda())
		ydata_render = Variable(sample_render['ydata'].cuda())
		output_render = model(xdata_render)
		output_pose = torch.cat((output_real, output_render))
		gt_pose = torch.cat((ydata_real, ydata_render))
		loss = mse_loss(output_pose, gt_pose)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# store
		count += 1
		writer.add_scalar('train_loss', loss.item(), count)
		if i % 1000 == 0:
			ytest, yhat_test, test_labels = testing()
			spio.savemat(results_file, {'ytest': ytest, 'yhat_test': yhat_test, 'test_labels': test_labels})
			tmp_val_loss = get_error2(ytest, yhat_test, test_labels, num_classes)
			writer.add_scalar('val_loss', tmp_val_loss, count)
			val_loss.append(tmp_val_loss)
		# cleanup
		del xdata_real, xdata_render, ydata_real, ydata_render
		del output_real, output_render, sample_real, sample_render, loss, output_pose, gt_pose
		bar.update(i)
		# stop
		if i == max_iterations:
			break
	render_loader.dataset.shuffle_images()
	real_loader.dataset.shuffle_images()


def training():
	global count, val_loss
	model.train()
	bar = progressbar.ProgressBar(max_value=max_iterations)
	for i, (sample_real, sample_render) in enumerate(zip(real_loader, render_loader)):
		# forward steps
		xdata_real = Variable(sample_real['xdata'].cuda())
		ydata_real = Variable(sample_real['ydata'].cuda())
		output_real = model(xdata_real)
		xdata_render = Variable(sample_render['xdata'].cuda())
		ydata_render = Variable(sample_render['ydata'].cuda())
		output_render = model(xdata_render)
		output_pose = torch.cat((output_real, output_render))
		gt_pose = torch.cat((ydata_real, ydata_render))
		loss = gve_loss(output_pose, gt_pose)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# store
		count += 1
		writer.add_scalar('train_loss', loss.item(), count)
		if i % 1000 == 0:
			ytest, yhat_test, test_labels = testing()
			spio.savemat(results_file, {'ytest': ytest, 'yhat_test': yhat_test, 'test_labels': test_labels})
			tmp_val_loss = get_error2(ytest, yhat_test, test_labels, num_classes)
			writer.add_scalar('val_loss', tmp_val_loss, count)
			val_loss.append(tmp_val_loss)
		# cleanup
		del xdata_real, xdata_render, ydata_real, ydata_render
		del output_real, output_render, sample_real, sample_render, loss, output_pose, gt_pose
		bar.update(i)
		# stop
		if i == max_iterations:
			break
	render_loader.dataset.shuffle_images()
	real_loader.dataset.shuffle_images()


def testing():
	model.eval()
	ypred = []
	ytrue = []
	labels = []
	for i, sample in enumerate(test_loader):
		xdata = Variable(sample['xdata'].cuda())
		label = Variable(sample['label'].cuda())
		output = model(xdata)
		ypred.append(output.data.cpu().numpy())
		ytrue.append(sample['ydata'].numpy())
		labels.append(sample['label'].numpy())
		del xdata, label, output, sample
		gc.collect()
	ypred = np.concatenate(ypred)
	ytrue = np.concatenate(ytrue)
	labels = np.concatenate(labels)
	model.train()
	return ytrue, ypred, labels


def save_checkpoint(filename):
	torch.save(model.state_dict(), filename)


# initialization
training_init()
ytest, yhat_test, test_labels = testing()
print('\nMedErr: {0}'.format(get_error2(ytest, yhat_test, test_labels, num_classes)))

for epoch in range(args.num_epochs):
	tic = time.time()
	scheduler.step()
	# training step
	training()
	# save model at end of epoch
	save_checkpoint(model_file)
	# validation
	ytest, yhat_test, test_labels = testing()
	print('\nMedErr: {0}'.format(get_error2(ytest, yhat_test, test_labels, num_classes)))
	# time and output
	toc = time.time() - tic
	print('Epoch: {0} done in time {1}s'.format(epoch, toc))
	# cleanup
	gc.collect()
writer.close()
val_loss = np.stack(val_loss)
spio.savemat(plots_file, {'val_loss': val_loss})

# evaluate the model
ytest, yhat_test, test_labels = testing()
print('\nMedErr: {0}'.format(get_error2(ytest, yhat_test, test_labels, num_classes)))
spio.savemat(results_file, {'ytest': ytest, 'yhat_test': yhat_test, 'test_labels': test_labels})
