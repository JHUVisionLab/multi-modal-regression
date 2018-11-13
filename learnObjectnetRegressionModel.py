# -*- coding: utf-8 -*-
"""
Learn models using ObjectNet3D images from setupDataFlipped_objectnet3d
"""

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F

from dataGenerators import my_collate
from axisAngle import get_error2, geodesic_loss, get_y
from featureModels import resnet_model
from poseModels import model_3layer
from helperFunctions import parse_name, rotation_matrix

import numpy as np
import scipy.io as spio
import gc
import os
import time
import progressbar
import argparse
from tensorboardX import SummaryWriter
from PIL import Image


parser = argparse.ArgumentParser(description='Objectnet Models')
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--save_str', type=str)
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

# constants
N0, N1, N2, N3, ndim = 2048, 1000, 500, 100, 3
init_lr = 1e-4
num_workers = 1

# paths
db_path = 'data/objectnet'
train_path = os.path.join(db_path, 'train')
test_path = os.path.join(db_path, 'test')

# classes
tmp = spio.loadmat(os.path.join(db_path, 'dbinfo'), squeeze_me=True)
classes = tmp['classes']
num_classes = len(classes)

# data
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(), normalize])


class TrainImages(Dataset):
	def __init__(self):
		self.db_path = train_path
		self.classes = classes
		self.num_classes = len(self.classes)
		self.list_image_names = []
		for i in range(self.num_classes):
			tmp = spio.loadmat(os.path.join(self.db_path, self.classes[i] + '_info'), squeeze_me=True)
			image_names = tmp['image_names']
			self.list_image_names.append(image_names)
		self.num_images = np.array([len(self.list_image_names[i]) for i in range(self.num_classes)])
		self.image_names = self.list_image_names

	def __len__(self):
		return np.amax(self.num_images)

	def __getitem__(self, idx):
		# return sample with xdata, ydata, label
		xdata, ydata, label = [], [], []
		for i in range(self.num_classes):
			image_name = self.image_names[i][idx % self.num_images[i]]
			label.append(i*torch.ones(1).long())
			# read image
			img_pil = Image.open(os.path.join(self.db_path, self.classes[i], image_name + '.png'))
			xdata.append(preprocess(img_pil))
			# parse image name to get correponding target
			_, _, az, el, ct, _ = parse_name(image_name)
			R = rotation_matrix(az, el, ct)
			tmpy = get_y(R)
			ydata.append(torch.from_numpy(tmpy).float())
		xdata = torch.stack(xdata)
		ydata = torch.stack(ydata)
		label = torch.stack(label)
		sample = {'xdata': xdata, 'ydata': ydata, 'label': label}
		return sample

	def shuffle_images(self):
		self.image_names = [np.random.permutation(self.list_image_names[i]) for i in range(self.num_classes)]


class TestImages(Dataset):
	def __init__(self):
		self.db_path = test_path
		self.classes = classes
		self.num_classes = len(self.classes)
		self.list_image_names = []
		self.list_labels = []
		for i in range(self.num_classes):
			tmp = spio.loadmat(os.path.join(self.db_path, self.classes[i] + '_info'), squeeze_me=True)
			image_names = tmp['image_names']
			self.list_image_names.append(image_names)
			self.list_labels.append(i*np.ones(len(image_names), dtype='int'))
		self.image_names = np.concatenate(self.list_image_names)
		self.labels = np.concatenate(self.list_labels)

	def __len__(self):
		return len(self.image_names)

	def __getitem__(self, idx):
		# return sample with xdata, ydata, label
		image_name = self.image_names[idx]
		label = self.labels[idx]
		# read image
		img_pil = Image.open(os.path.join(self.db_path, self.classes[label], image_name + '.png'))
		xdata = preprocess(img_pil)
		# parse image name to get correponding target
		_, _, az, el, ct, _ = parse_name(image_name)
		R = rotation_matrix(az, el, ct)
		tmpy = get_y(R)
		ydata = torch.from_numpy(tmpy).float()
		label = label*torch.ones(1).long()
		sample = {'xdata': xdata, 'ydata': ydata, 'label': label}
		# print(xdata.size(), ydata.size(), label.size())
		return sample


class RegressionModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.num_classes = num_classes
		self.feature_model = resnet_model('resnet50', 'layer4').cuda()
		self.pose_model = model_3layer(N0+num_classes, N1, N2, ndim).cuda()

	def forward(self, x, label):
		x = self.feature_model(x)
		label = torch.zeros(label.size(0), self.num_classes).scatter_(1, label.data.cpu(), 1.0)
		label = Variable(label.cuda())
		x = torch.cat((x, label), dim=1)
		x = self.pose_model(x)
		x = np.pi * F.tanh(x)
		return x


# loss
mse_loss = nn.MSELoss().cuda()
ce_loss = nn.CrossEntropyLoss().cuda()
gve_loss = geodesic_loss().cuda()

# DATA
# datasets
train_data = TrainImages()
test_data = TestImages()
# setup data loaders
train_loader = DataLoader(train_data, batch_size=num_workers, shuffle=True, num_workers=4, pin_memory=True, collate_fn=my_collate)
test_loader = DataLoader(test_data, batch_size=32)
print('Train: {0} \t Test: {1}'.format(len(train_loader), len(test_loader)))

# my_model
model = RegressionModel()
# print(model)
# loss and optimizer
optimizer = optim.Adam(model.parameters(), lr=init_lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
# store stuff
writer = SummaryWriter(log_dir)
count = 0
val_loss = []


# OPTIMIZATION functions
def training_init():
	global count, val_loss
	model.train()
	bar = progressbar.ProgressBar(max_value=len(train_loader))
	for i, sample in enumerate(train_loader):
		# forward steps
		# outputs
		xdata = Variable(sample['xdata'].cuda())
		ydata = Variable(sample['ydata']).cuda()
		output = model(xdata)
		# loss
		loss = mse_loss(output, ydata)
		# parameter updates
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# store
		count += 1
		writer.add_scalar('train_loss', loss.item(), count)
		if i % 7000 == 0:
			ytest, yhat_test, test_labels = testing()
			spio.savemat(results_file, {'ytest': ytest, 'yhat_test': yhat_test, 'test_labels': test_labels})
			tmp_val_loss = get_error2(ytest, yhat_test, test_labels, num_classes)
			writer.add_scalar('val_loss', tmp_val_loss, count)
			val_loss.append(tmp_val_loss)
		# cleanup
		del output, loss, sample, xdata, ydata
		bar.update(i+1)
	train_loader.dataset.shuffle_images()


def training():
	global count, val_loss
	model.train()
	bar = progressbar.ProgressBar(max_value=len(train_loader))
	for i, sample in enumerate(train_loader):
		# forward steps
		# output
		xdata = Variable(sample['xdata'].cuda())
		ydata = Variable(sample['ydata']).cuda()
		output = model(xdata)
		# loss
		loss = gve_loss(output, ydata)
		# parameter updates
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# store
		count += 1
		writer.add_scalar('train_loss', loss.item(), count)
		if i % 7000 == 0:
			ytest, yhat_test, test_labels = testing()
			spio.savemat(results_file, {'ytest': ytest, 'yhat_test': yhat_test, 'test_labels': test_labels})
			tmp_val_loss = get_error2(ytest, yhat_test, test_labels, num_classes)
			writer.add_scalar('val_loss', tmp_val_loss, count)
			val_loss.append(tmp_val_loss)
		# cleanup
		del output, sample, loss, xdata, ydata
		bar.update(i+1)
	train_loader.dataset.shuffle_images()


def testing():
	model.eval()
	ypred = []
	ytrue = []
	labels = []
	bar = progressbar.ProgressBar(max_value=len(test_loader))
	for i, sample in enumerate(test_loader):
		xdata = Variable(sample['xdata'].cuda())
		label = Variable(sample['label'].cuda())
		output = model(xdata)
		ypred.append(output.data.cpu().numpy())
		ytrue.append(sample['ydata'].numpy())
		labels.append(sample['label'].numpy())
		del xdata, label, output, sample
		gc.collect()
		bar.update(i+1)
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
	# scheduler.step()
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
