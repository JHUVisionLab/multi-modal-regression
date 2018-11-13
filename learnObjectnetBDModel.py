# -*- coding: utf-8 -*-
"""
Learn models using ObjectNet3D images from setupDataFlipped_objectnet3d
"""

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dataGenerators import my_collate
from axisAngle import get_error2, geodesic_loss, get_y
from featureModels import resnet_model
from binDeltaModels import bin_3layer, res_3layer, res_2layer
from helperFunctions import parse_name, rotation_matrix

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
from PIL import Image


parser = argparse.ArgumentParser(description='Objectnet Models')
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--save_str', type=str)
parser.add_argument('--dict_size', type=int, default=200)
parser.add_argument('--num_epochs', type=int, default=3)
parser.add_argument('--multires', type=bool, default=False)
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

# kmeans data
kmeans_file = 'data/kmeans_dictionary_axis_angle_' + str(args.dict_size) + '.pkl'
kmeans = pickle.load(open(kmeans_file, 'rb'))
kmeans_dict = kmeans.cluster_centers_
cluster_centers_ = Variable(torch.from_numpy(kmeans_dict).float()).cuda()
num_clusters = kmeans.n_clusters

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
		ydata_bin = kmeans.predict(ydata.numpy())
		ydata_res = ydata.numpy() - kmeans_dict[ydata_bin, :]
		ydata_bin = torch.from_numpy(ydata_bin).long()
		ydata_res = torch.from_numpy(ydata_res).float()
		label = torch.stack(label)
		sample = {'xdata': xdata, 'ydata': ydata, 'label': label, 'ydata_bin': ydata_bin, 'ydata_res': ydata_res}
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
		return sample


class OneBinDeltaModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.num_clusters = args.dict_size
		self.feature_model = resnet_model('resnet50', 'layer4').cuda()
		self.bin_model = bin_3layer(N0, N1, N2, self.num_clusters).cuda()
		self.res_model = res_3layer(N0, N1, N2, ndim).cuda()

	def forward(self, x):
		x = self.feature_model(x)
		y1 = self.bin_model(x)
		y2 = self.res_model(x)
		del x
		return [y1, y2]


class OneDeltaPerBinModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.ndim = ndim
		self.num_classes = num_classes
		self.num_clusters = args.dict_size
		self.feature_model = resnet_model('resnet50', 'layer4').cuda()
		self.bin_model = bin_3layer(N0+num_classes, N1, N2, num_clusters).cuda()
		self.res_models = nn.ModuleList([res_2layer(N0+num_classes, N3, ndim) for i in range(self.num_clusters)]).cuda()

	def forward(self, x, label):
		x = self.feature_model(x)
		label = torch.zeros(label.size(0), self.num_classes).scatter_(1, label.data.cpu(), 1.0)
		label = Variable(label.cuda())
		x = torch.cat((x, label), dim=1)
		y1 = self.bin_model(x)
		y2 = torch.stack([self.res_models[i](x) for i in range(self.num_clusters)])
		y2 = y2.view(self.num_clusters, -1, self.ndim).permute(1, 2, 0)
		pose_label = torch.argmax(y1, dim=1, keepdim=True)
		pose_label = torch.zeros(pose_label.size(0), self.num_clusters).scatter_(1, pose_label.data.cpu(), 1.0)
		pose_label = Variable(pose_label.unsqueeze(2).cuda())
		y2 = torch.squeeze(torch.bmm(y2, pose_label), 2)
		del x, pose_label
		return [y1, y2]


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
if not args.multires:
	model = OneBinDeltaModel()
else:
	model = OneDeltaPerBinModel()
# print(model)
# loss and optimizer
optimizer = optim.Adam(model.parameters(), lr=init_lr)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
# store stuff
writer = SummaryWriter(log_dir)
count = 0
val_loss = []
s = 0


# OPTIMIZATION functions
def training_init():
	global count, val_loss, s
	model.train()
	bar = progressbar.ProgressBar(max_value=len(train_loader))
	for i, sample in enumerate(train_loader):
		# forward steps
		# outputs
		xdata = Variable(sample['xdata'].cuda())
		ydata_bin = Variable(sample['ydata_bin']).cuda()
		ydata_res = Variable(sample['ydata_res']).cuda()
		output = model(xdata)
		# loss
		Lc = ce_loss(output[0], ydata_bin)
		Lr = mse_loss(output[1], ydata_res)
		loss = Lc + 0.5*math.exp(-2*s)*Lr + s
		# parameter updates
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		s = 0.5*math.log(Lr)
		# store
		count += 1
		writer.add_scalar('train_loss', loss.item(), count)
		writer.add_scalar('alpha', 0.5*math.exp(-2*s), count)
		if i % 7000 == 0:
			ytest, yhat_test, test_labels = testing()
			spio.savemat(results_file, {'ytest': ytest, 'yhat_test': yhat_test, 'test_labels': test_labels})
			tmp_val_loss = get_error2(ytest, yhat_test, test_labels, num_classes)
			writer.add_scalar('val_loss', tmp_val_loss, count)
			val_loss.append(tmp_val_loss)
		# cleanup
		del xdata, ydata_bin, ydata_res, output, loss, Lc, Lr
		bar.update(i+1)
	train_loader.dataset.shuffle_images()


def training():
	global count, val_loss, s
	model.train()
	bar = progressbar.ProgressBar(max_value=len(train_loader))
	for i, sample in enumerate(train_loader):
		# forward steps
		# output
		xdata = Variable(sample['xdata'].cuda())
		ydata_bin = Variable(sample['ydata_bin']).cuda()
		ydata = Variable(sample['ydata']).cuda()
		output = model(xdata)
		# loss
		ind = torch.argmax(output[0], dim=1)
		y = torch.index_select(cluster_centers_, 0, ind) + output[1]
		Lc = ce_loss(output[0], ydata_bin)
		Lr = gve_loss(y, ydata)
		loss = Lc + math.exp(-s)*Lr + s
		# parameter updates
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		s = math.log(Lr)
		# store
		count += 1
		writer.add_scalar('train_loss', loss.item(), count)
		writer.add_scalar('alpha', math.exp(-s), count)
		if i % 7000 == 0:
			ytest, yhat_test, test_labels = testing()
			spio.savemat(results_file, {'ytest': ytest, 'yhat_test': yhat_test, 'test_labels': test_labels})
			tmp_val_loss = get_error2(ytest, yhat_test, test_labels, num_classes)
			writer.add_scalar('val_loss', tmp_val_loss, count)
			val_loss.append(tmp_val_loss)
		# cleanup
		del xdata, ydata_bin, ydata, output, y, Lr, Lc, loss, ind
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
		ypred_bin = np.argmax(output[0].data.cpu().numpy(), axis=1)
		ypred_res = output[1].data.cpu().numpy()
		ypred.append(kmeans_dict[ypred_bin, :] + ypred_res)
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

s = 0  # reset
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
