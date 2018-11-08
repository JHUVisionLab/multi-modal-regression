# -*- coding: utf-8 -*-
"""
Elhoseiny model based on Geodesic Regression model R_G
"""

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataGenerators import ImagesAll, TestImages, my_collate
from axisAngle import get_error2, geodesic_loss
from poseModels import model_3layer
from helperFunctions import classes, get_accuracy
from featureModels import resnet_model, vgg_model

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
class ElhoseinyModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.num_classes = num_classes
		if args.feature_network == 'resnet':
			self.feature_model = resnet_model('resnet50', 'layer4').cuda()
		elif args.feature_network == 'vgg':
			self.feature_model = vgg_model('vgg13', 'fc6').cuda()
		self.pose_model = model_3layer(args.N0, args.N1, args.N2, ndim).cuda()
		self.category_model = nn.Linear(args.N0, num_classes)

	def forward(self, x):
		x = self.feature_model(x)
		y0 = self.category_model(x)
		y1 = self.pose_model(x)
		y1 = np.pi*F.tanh(y1)
		del x
		return [y0, y1]     # cat, pose


model = ElhoseinyModel()
# print(model)
# loss and optimizer
optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
# store stuff
writer = SummaryWriter(log_dir)
count = 0
val_err, val_acc = [], []


# OPTIMIZATION functions
def training_init():
	global count, val_err, val_acc
	model.train()
	bar = progressbar.ProgressBar(max_value=max_iterations)
	for i, (sample_real, sample_render) in enumerate(zip(real_loader, render_loader)):
		# forward steps
		xdata_real = Variable(sample_real['xdata'].cuda())
		label_real = Variable(sample_real['label'].cuda())
		ydata_real = Variable(sample_real['ydata'].cuda())
		output_real = model(xdata_real)
		xdata_render = Variable(sample_render['xdata'].cuda())
		label_render = Variable(sample_render['label'].cuda())
		ydata_render = Variable(sample_render['ydata'].cuda())
		output_render = model(xdata_render)
		output_pose = torch.cat((output_real[1], output_render[1]))
		gt_pose = torch.cat((ydata_real, ydata_render))
		Lr = mse_loss(output_pose, gt_pose)
		Lc = ce_loss(output_real[0], label_real.squeeze())
		loss = 0.1*Lc + Lr
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# store
		count += 1
		writer.add_scalar('train_loss', loss.item(), count)
		if i % 1000 == 0:
			ytrue_cat, ytrue_pose, ypred_cat, ypred_pose = testing()
			spio.savemat(results_file, {'ytrue_cat': ytrue_cat, 'ytrue_pose': ytrue_pose, 'ypred_cat': ypred_cat, 'ypred_pose': ypred_pose})
			tmp_acc = get_accuracy(ytrue_cat, ypred_cat, num_classes)
			tmp_err = get_error2(ytrue_pose, ypred_pose, ytrue_cat, num_classes)
			writer.add_scalar('val_acc', tmp_acc, count)
			writer.add_scalar('val_err', tmp_err, count)
			val_acc.append(tmp_acc)
			val_err.append(tmp_err)
		# cleanup
		del xdata_real, xdata_render, label_real, label_render, ydata_real, ydata_render, Lr, Lc
		del output_real, output_render, sample_real, sample_render, loss, output_pose, gt_pose
		bar.update(i)
		# stop
		if i == max_iterations:
			break
	render_loader.dataset.shuffle_images()
	real_loader.dataset.shuffle_images()


def training():
	global count, val_err, val_acc
	model.train()
	bar = progressbar.ProgressBar(max_value=max_iterations)
	for i, (sample_real, sample_render) in enumerate(zip(real_loader, render_loader)):
		# forward steps
		xdata_real = Variable(sample_real['xdata'].cuda())
		label_real = Variable(sample_real['label'].cuda())
		ydata_real = Variable(sample_real['ydata'].cuda())
		output_real = model(xdata_real, label_real)
		xdata_render = Variable(sample_render['xdata'].cuda())
		label_render = Variable(sample_render['label'].cuda())
		ydata_render = Variable(sample_render['ydata'].cuda())
		output_render = model(xdata_render, label_render)
		output_pose = torch.cat((output_real[1], output_render[1]))
		gt_pose = torch.cat((ydata_real, ydata_render))
		Lr = gve_loss(output_pose, gt_pose)
		Lc = ce_loss(output_real[0], label_real.squeeze())
		loss = 0.1*Lc + Lr
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# store
		count += 1
		writer.add_scalar('train_loss', loss.item(), count)
		if i % 1000 == 0:
			ytrue_cat, ytrue_pose, ypred_cat, ypred_pose = testing()
			spio.savemat(results_file, {'ytrue_cat': ytrue_cat, 'ytrue_pose': ytrue_pose, 'ypred_cat': ypred_cat, 'ypred_pose': ypred_pose})
			tmp_acc = get_accuracy(ytrue_cat, ypred_cat, num_classes)
			tmp_err = get_error2(ytrue_pose, ypred_pose, ytrue_cat, num_classes)
			writer.add_scalar('val_acc', tmp_acc, count)
			writer.add_scalar('val_err', tmp_err, count)
			val_acc.append(tmp_acc)
			val_err.append(tmp_err)
		# cleanup
		del xdata_real, xdata_render, label_real, label_render, ydata_real, ydata_render, Lr, Lc
		del output_real, output_render, sample_real, sample_render, loss, output_pose, gt_pose
		bar.update(i)
		# stop
		if i == max_iterations:
			break
	render_loader.dataset.shuffle_images()
	real_loader.dataset.shuffle_images()


def testing():
	model.eval()
	ytrue_cat, ytrue_pose = [], []
	ypred_cat, ypred_pose = [], []
	for i, sample in enumerate(test_loader):
		xdata = Variable(sample['xdata'].cuda())
		output = model(xdata)
		output_cat = output[0]
		output_pose = output[1]
		tmp_labels = np.argmax(output_cat.data.cpu().numpy(), axis=1)
		ypred_cat.append(tmp_labels)
		label = Variable(sample['label'])
		ytrue_cat.append(sample['label'].squeeze().numpy())
		ypred_pose.append(output_pose.data.cpu().numpy())
		ytrue_pose.append(sample['ydata'].numpy())
		del xdata, label, output, sample, output_cat, output_pose
		gc.collect()
	ytrue_cat = np.concatenate(ytrue_cat)
	ypred_cat = np.concatenate(ypred_cat)
	ytrue_pose = np.concatenate(ytrue_pose)
	ypred_pose = np.concatenate(ypred_pose)
	model.train()
	return ytrue_cat, ytrue_pose, ypred_cat, ypred_pose


def save_checkpoint(filename):
	torch.save(model.state_dict(), filename)


# initialization
training_init()
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
	# time and output
	toc = time.time() - tic
	print('Epoch: {0} done in time {1}s'.format(epoch, toc))
	# cleanup
	gc.collect()
writer.close()
val_acc = np.stack(val_acc)
val_err = np.stack(val_err)
spio.savemat(plots_file, {'val_acc': val_acc, 'val_err': val_err})

# evaluate the model
ytrue_cat, ytrue_pose, ypred_cat, ypred_pose = testing()
spio.savemat(results_file, {'ytrue_cat': ytrue_cat, 'ytrue_pose': ytrue_pose, 'ypred_cat': ypred_cat, 'ypred_pose': ypred_pose})
tmp_acc = get_accuracy(ytrue_cat, ypred_cat, num_classes)
tmp_err = get_error2(ytrue_pose, ypred_pose, ytrue_cat, num_classes)
print('Acc: {0} \t Err: {1}'.format(tmp_acc, tmp_err))
