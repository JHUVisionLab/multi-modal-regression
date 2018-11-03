# -*- coding: utf-8 -*-
"""
Joint Cat & Pose model (Top1) with Geodesic Regression model for the axis-angle representation
"""

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import models

from dataGenerators import TestImages, my_collate, ImagesAll
from featureModels import resnet_model
from poseModels import model_3layer
from axisAngle import get_error2, geodesic_loss
from helperFunctions import classes, get_accuracy

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
parser.add_argument('--feature_network', type=str, default='resnet')
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--db_type', type=str, default='clean')
parser.add_argument('--init_lr', type=float, default=1e-4)
args = parser.parse_args()
print(args)
# assign GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

# save stuff here
init_model_file = os.path.join('models', args.save_str + '.tar')
model_file = os.path.join('models', args.save_str + '_top1.tar')
results_file = os.path.join('results', args.save_str + '_top1_' + args.db_type)
plots_file = os.path.join('plots', args.save_str + '_top1_' + args.db_type)
log_dir = os.path.join('logs', args.save_str + '_top1_' + args.db_type)

# relevant variables
ndim = 3
N0, N1, N2, N3 = 1024, 1000, 500, 2048
num_classes = len(classes)
if args.db_type == 'clean':
	db_path = 'data/flipped_new'
else:
	db_path = 'data/flipped_all'
num_classes = len(classes)
real_path = os.path.join(db_path, 'train')
render_path = 'data/renderforcnn'
test_path = os.path.join(db_path, 'test')

# loss
ce_loss = nn.CrossEntropyLoss().cuda()
gve_loss = geodesic_loss().cuda()

# DATA
# datasets
real_data = ImagesAll(real_path, 'real')
render_data = ImagesAll(render_path, 'render')
test_data = TestImages(test_path)
# setup data loaders
real_loader = DataLoader(real_data, batch_size=args.num_workers, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=my_collate)
render_loader = DataLoader(render_data, batch_size=args.num_workers, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=my_collate)
test_loader = DataLoader(test_data, batch_size=32)
print('Real: {0} \t Render: {1} \t Test: {2}'.format(len(real_loader), len(render_loader), len(test_loader)))
max_iterations = min(len(real_loader), len(render_loader))


# my_model
class PoseGivenCatModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.num_classes = num_classes
		self.feature_model = resnet_model('resnet50', 'layer3').cuda()
		self.pose_models = nn.ModuleList([model_3layer(N0, N1, N2, 3) for i in range(self.num_classes)]).cuda()

	def forward(self, x, label):
		x = self.feature_model(x)
		x = torch.stack([self.pose_models[i](x) for i in range(self.num_classes)]).permute(1, 2, 0)
		label = torch.zeros(label.size(0), self.num_classes).scatter_(1, label.data.cpu(), 1.0)
		label = Variable(label.unsqueeze(2).cuda())
		y = torch.squeeze(torch.bmm(x, label), 2)
		y = np.pi*F.tanh(y)
		del x, label
		return y


class JointCatPoseModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.num_classes = num_classes
		init_model = PoseGivenCatModel()
		init_model.load_state_dict(torch.load(init_model_file))
		self.features = init_model.feature_model.features
		self.avgpool1 = init_model.feature_model.avgpool
		self.pose_models = init_model.pose_models
		resnet50 = models.resnet50(pretrained=True).cuda()
		self.layer4 = resnet50.layer4
		self.avgpool2 = resnet50.avgpool
		self.fc = nn.Linear(N3, num_classes)

	def forward(self, x):
		x = self.features(x)
		x1 = self.layer4(x)
		x1 = self.avgpool2(x1)
		x1 = x1.view(x1.size(0), -1)
		y1 = self.fc(x1)
		x2 = self.avgpool1(x)
		x2 = x2.view(x2.size(0), -1)
		y2 = torch.stack([self.pose_models[i](x2) for i in range(self.num_classes)]).permute(1, 2, 0)
		_, pose_label = torch.max(y1, dim=1, keepdim=True)
		pose_label = torch.zeros(pose_label.size(0), self.num_classes).scatter_(1, pose_label.data.cpu(), 1.0)
		pose_label = Variable(pose_label.unsqueeze(2).cuda())
		y2 = torch.squeeze(torch.bmm(y2, pose_label), 2)
		y2 = np.pi*F.tanh(y2)
		return [y1, y2]


model = JointCatPoseModel()
# print(model)


def my_schedule(ep):
	return 1. / (1. + ep)


optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, my_schedule)
writer = SummaryWriter(log_dir)
count = 0
val_err = []
val_acc = []


def training():
	global count, val_acc, val_err
	model.train()
	bar = progressbar.ProgressBar(max_value=max_iterations)
	for i, (sample_real, sample_render) in enumerate(zip(real_loader, render_loader)):
		# forward steps
		# output
		label_real = Variable(sample_real['label'].squeeze().cuda())
		ydata_real = Variable(sample_real['ydata'].cuda())
		xdata_real = Variable(sample_real['xdata'].cuda())
		output_real = model(xdata_real)
		output_cat_real = output_real[0]
		ypred_real = output_real[1]
		label_render = Variable(sample_render['label'].squeeze().cuda())
		ydata_render = Variable(sample_render['ydata'].cuda())
		xdata_render = Variable(sample_render['xdata'].cuda())
		output_render = model(xdata_render)
		output_cat_render = output_render[0]
		ypred_render = output_render[1]
		ydata = torch.cat((ydata_real, ydata_render))
		# loss
		Lc_cat = ce_loss(output_cat_real, label_real)   # use only real images for category loss
		y = torch.cat((ypred_real, ypred_render))
		Lr = gve_loss(y, ydata)                         # gve loss on final pose
		loss = 0.1*Lc_cat + Lr
		# parameter updates
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
		del label_real, ydata_real, xdata_real, output_real, output_cat_real, ypred_real
		del label_render, ydata_render, xdata_render, output_render, output_cat_render, ypred_render
		del	ydata, Lc_cat, Lr, loss, y
		bar.update(i+1)
	real_loader.dataset.shuffle_images()
	render_loader.dataset.shuffle_images()


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
