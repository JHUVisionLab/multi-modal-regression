# -*- coding: utf-8 -*-
"""
Function that learns feature model + 3layer pose models x 12 object categories
in an end-to-end manner by minimizing the mean squared error for axis-angle representation
"""

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

from featureModels import resnet_model
from axisAngle import model_3layer, get_error2, get_y
from helperFunctions import classes, parse_name, rotation_matrix

from PIL import Image
import numpy as np
import scipy.io as spio
import gc
import os
import re
import progressbar
import argparse

parser = argparse.ArgumentParser(description='Pure Regression Models')
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--db_type', type=str, default='clean')
parser.add_argument('--save_str', type=str)
args = parser.parse_args()
print(args)
# assign GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id


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


if args.db_type == 'clean':
	db_path = 'data/flipped_new/test'
	results_file = os.path.join('results_clean', args.save_str)
else:
	db_path = 'data/flipped_all/test'
	results_file = os.path.join('results_all', args.save_str)
	parse_name = parse_name2
model_file = os.path.join('models', args.save_str + '.tar')

num_classes = len(classes)


# DATA
class ImagesAll(Dataset):
	def __init__(self, db_path):
		self.db_path = db_path
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
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		self.preprocess = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(), normalize])

	def __len__(self):
		return len(self.image_names)

	def __getitem__(self, idx):
		# return sample with xdata, ydata, label
		image_name = self.image_names[idx]
		label = self.labels[idx]
		# read image
		img_pil = Image.open(os.path.join(self.db_path, self.classes[label], image_name + '.png'))
		xdata = self.preprocess(img_pil)
		# parse image name to get correponding target
		_, _, az, el, ct, _ = parse_name(image_name)
		R = rotation_matrix(az, el, ct)
		tmpy = get_y(R)
		# get torch data
		ydata = torch.from_numpy(tmpy).float()
		label = label*torch.ones(1).long()
		sample = {'xdata': xdata, 'ydata': ydata, 'label': label}
		return sample


test_data = ImagesAll(db_path)
test_loader = DataLoader(test_data, batch_size=32)


# MODEL
N0, N1, N2 = 2048, 1000, 500


# my model for pose estimation: feature model + 1layer pose model x 12
class my_model(nn.Module):
	def __init__(self):
		super().__init__()
		self.num_classes = num_classes
		self.feature_model = resnet_model('resnet50', 'layer4').cuda()
		self.pose_models = nn.ModuleList([model_3layer(N0, N1, N2) for i in range(self.num_classes)]).cuda()

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
# print(model)


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


# evaluate the model
ytest, yhat_test, test_labels = testing()
get_error2(ytest, yhat_test, test_labels, num_classes)
spio.savemat(results_file, {'ytest': ytest, 'yhat_test': yhat_test, 'test_labels': test_labels})
