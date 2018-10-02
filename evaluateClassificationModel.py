# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataGenerators import ImagesAll, TestImages, my_collate
from featureModels import resnet_model
from axisAngle import get_error2
from binDeltaModels import bin_3layer
from helperFunctions import classes, mySGD

import numpy as np
import scipy.io as spio
import gc
import os
import time
import progressbar
import pickle
from tensorboardX import SummaryWriter
import argparse

parser = argparse.ArgumentParser(description='Evaluate Classification Model')
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--save_str', type=str)
parser.add_argument('--dict_size', type=int, default=200)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--feature_network', type=str, default='resnet')
parser.add_argument('--db_type', type=str, default='clean')
parser.add_argument('--num_epochs', type=int, default=9)
args = parser.parse_args()
print(args)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

# kmeans info
kmeans_file = 'data/kmeans_dictionary_axis_angle_' + str(args.dict_size) + '.pkl'
kmeans = pickle.load(open(kmeans_file, 'rb'))

# relevant paths and files
model_file = os.path.join('models', args.save_str + '.tar')
results_dir = os.path.join('results', args.save_str + '_' + args.db_type)
plots_file = os.path.join('plots', args.save_str + '_' + args.db_type)
log_dir = os.path.join('logs', args.save_str + '_' + args.db_type)
if not os.path.exists(results_dir):
	os.mkdir(results_dir)

# relevant variables
N0, N1, N2 = 2048, 1000, 500
num_classes = len(classes)
num_clusters = kmeans.n_clusters
kmeans_dict = kmeans.cluster_centers_

if args.db_type == 'clean':
	db_path = 'data/flipped_new'
else:
	db_path = 'data/flipped_all'
num_classes = len(classes)
train_path = os.path.join(db_path, 'train')
test_path = os.path.join(db_path, 'test')
render_path = 'data/renderforcnn/'

# DATA
real_data = ImagesAll(train_path, 'real', 'axis_angle')
render_data = ImagesAll(render_path, 'render', 'axis_angle')
test_data = TestImages(test_path, 'axis_angle')
real_loader = DataLoader(real_data, batch_size=args.num_workers, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=my_collate)
render_loader = DataLoader(render_data, batch_size=args.num_workers, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=my_collate)
test_loader = DataLoader(test_data, batch_size=32)
print('Real: {0} \t Render: {1} \t Test: {2}'.format(len(real_loader), len(render_loader), len(test_loader)))


# MODEL
# my model for pose estimation: feature model + 1layer pose model x 12
class my_model(nn.Module):
	def __init__(self):
		super().__init__()
		self.num_classes = num_classes
		self.feature_model = resnet_model('resnet50', 'layer4').cuda()
		self.pose_models = nn.ModuleList([bin_3layer(N0, N1, N2, num_clusters) for i in range(self.num_classes)]).cuda()

	def forward(self, x, label):
		x = self.feature_model(x)
		x = torch.stack([self.pose_models[i](x) for i in range(self.num_classes)]).permute(1, 2, 0)
		label = torch.zeros(label.size(0), self.num_classes).scatter_(1, label.data.cpu(), 1.0)
		label = Variable(label.unsqueeze(2).cuda())
		y = torch.squeeze(torch.bmm(x, label), 2)
		del x, label
		return y


# my_model
model = my_model()
model.load_state_dict(torch.load(model_file))
# print(model)
# loss and optimizer
criterion = nn.CrossEntropyLoss().cuda()
optimizer = mySGD(model.parameters(), c=2*len(real_loader))
writer = SummaryWriter(log_dir)
val_loss = []
count = 0
num_ensemble = 0


# OPTIMIZATION functions
def training():
	global count, val_loss, num_ensemble
	model.train()
	bar = progressbar.ProgressBar(max_value=len(real_loader))
	for i, (sample_real, sample_render) in enumerate(zip(real_loader, render_loader)):
		# forward steps
		xdata_real = Variable(sample_real['xdata'].cuda())
		label_real = Variable(sample_real['label'].cuda())
		ydata = sample_real['ydata'].numpy()
		ydata_real = Variable(torch.from_numpy(kmeans.predict(ydata)).long().cuda())
		output_real = model(xdata_real, label_real)
		loss_real = criterion(output_real, ydata_real)
		xdata_render = Variable(sample_render['xdata'].cuda())
		label_render = Variable(sample_render['label'].cuda())
		ydata = sample_render['ydata'].numpy()
		ydata_render = Variable(torch.from_numpy(kmeans.predict(ydata)).long().cuda())
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
		if count % optimizer.c == optimizer.c / 2:
			ytest, yhat_test, test_labels = testing()
			num_ensemble += 1
			results_file = os.path.join(results_dir, 'num' + str(num_ensemble))
			spio.savemat(results_file, {'ytest': ytest, 'yhat_test': yhat_test, 'test_labels': test_labels})
		# cleanup
		del xdata_real, xdata_render, label_real, label_render, ydata_real, ydata_render
		del output_real, output_render, loss_real, loss_render, sample_real, sample_render, loss
		bar.update(i)
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
		output = model(xdata, label)
		ypred_bin = np.argmax(output.data.cpu().numpy(), axis=1)
		ypred.append(kmeans_dict[ypred_bin, :])
		ytrue.append(sample['ydata'].numpy())
		labels.append(sample['label'].numpy())
		del xdata, label, output, sample
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
