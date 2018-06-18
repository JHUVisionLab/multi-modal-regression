import torch
from torch.utils.data import Dataset
from torchvision import transforms

from helperFunctions import parse_name, rotation_matrix, classes, eps
from axisAngle import get_y
from quaternion import convert_dictionary
from quaternion import get_y as get_quaternion

from PIL import Image
import numpy as np
import scipy.io as spio
import os
import pickle
from scipy.spatial.distance import cdist


class ImagesAll(Dataset):
	def __init__(self, db_path, db_type, ydata_type='axis_angle'):
		self.db_path = db_path
		self.classes = classes
		self.num_classes = len(self.classes)
		self.db_type = db_type
		self.ydata_type = ydata_type
		self.list_image_names = []
		for i in range(self.num_classes):
			if self.db_type == 'real':
				tmp = spio.loadmat(os.path.join(self.db_path, self.classes[i] + '_train_info'), squeeze_me=True)
			elif self.db_type == 'render':
				tmp = spio.loadmat(os.path.join(self.db_path, self.classes[i] + '_info'), squeeze_me=True)
			else:
				raise NameError('Unknown db_type passed')
			image_names = tmp['image_names']
			self.list_image_names.append(image_names)
		self.num_images = np.array([len(self.list_image_names[i]) for i in range(self.num_classes)])
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		self.preprocess = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(), normalize])
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
			xdata.append(self.preprocess(img_pil))
			# parse image name to get correponding target
			_, _, az, el, ct, _ = parse_name(image_name)
			if self.db_type == 'real':
				R = rotation_matrix(az, el, ct)
			elif self.db_type == 'render':
				R = rotation_matrix(az, el, -ct)
			else:
				raise NameError('Unknown db_type passed')
			if self.ydata_type == 'axis_angle':
				tmpy = get_y(R)
			elif self.ydata_type == 'quaternion':
				tmpy = get_quaternion(R)
			else:
				raise NameError('Uknown ydata_type passed')
			ydata.append(torch.from_numpy(tmpy).float())
		xdata = torch.stack(xdata)
		ydata = torch.stack(ydata)
		label = torch.stack(label)
		sample = {'xdata': xdata, 'ydata': ydata, 'label': label}
		return sample

	def shuffle_images(self):
		self.image_names = [np.random.permutation(self.list_image_names[i]) for i in range(self.num_classes)]


class GBDGenerator(ImagesAll):
	def __init__(self, db_path, db_type, kmeans_file):
		# initialize the renderedImages dataset first
		super().__init__(db_path, db_type)
		# add the kmeans part
		self.kmeans = pickle.load(open(kmeans_file, 'rb'))
		self.num_clusters = self.kmeans.n_clusters

	def __len__(self):
		return np.amax(self.num_images)

	def __getitem__(self, idx):
		# run the item handler of the renderedImages dataset
		sample = super().__getitem__(idx)
		# update the ydata target using kmeans dictionary
		ydata = sample['ydata'].numpy()
		# bin part
		ydata_bin = self.kmeans.predict(ydata)
		sample['ydata_bin'] = torch.from_numpy(ydata_bin).long()
		# residual part
		ydata_res = ydata - self.kmeans.cluster_centers_[ydata_bin, :]
		sample['ydata_res'] = torch.from_numpy(ydata_res).float()
		return sample


class GBDGeneratorQ(ImagesAll):
	def __init__(self, db_path, db_type, kmeans_file):
		# initialize the renderedImages dataset first
		super().__init__(db_path, db_type, 'quaternion')
		# add the kmeans part
		self.kmeans = pickle.load(open(kmeans_file, 'rb'))
		self.num_clusters = self.kmeans.n_clusters
		self.kmeans.cluster_centers_ = convert_dictionary(self.kmeans.cluster_centers_)

	def __len__(self):
		return np.amax(self.num_images)

	def __getitem__(self, idx):
		# run the item handler of the renderedImages dataset
		sample = super().__getitem__(idx)
		# update the ydata target using kmeans dictionary
		ydata = sample['ydata'].numpy()
		# bin part
		ydata_bin = self.kmeans.predict(ydata)
		sample['ydata_bin'] = torch.from_numpy(ydata_bin).long()
		# residual part
		ydata_res = ydata - self.kmeans.cluster_centers_[ydata_bin, :]
		sample['ydata_res'] = torch.from_numpy(ydata_res).float()
		return sample


class XPBDGenerator(ImagesAll):
	def __init__(self, db_path, db_type, kmeans_file, gamma):
		# initialize the renderedImages dataset first
		super().__init__(db_path, db_type)
		self.gamma = gamma
		# add the kmeans part
		self.kmeans = pickle.load(open(kmeans_file, 'rb'))
		self.num_clusters = self.kmeans.n_clusters

	def __len__(self):
		return np.amax(self.num_images)

	def __getitem__(self, idx):
		# run the item handler of the renderedImages dataset
		sample = super().__getitem__(idx)
		# update the ydata target using kmeans dictionary
		ydata = sample['ydata'].numpy()
		# bin part
		ydata_bin = np.exp(-self.gamma*cdist(ydata, self.kmeans.cluster_centers_, 'sqeuclidean'))
		ydata_bin = ydata_bin/np.sum(ydata_bin, axis=1, keepdims=True)
		sample['ydata_bin'] = torch.from_numpy(ydata_bin).float()
		# residual part
		ydata_res = ydata - np.dot(ydata_bin, self.kmeans.cluster_centers_)     # need to think more about m4
		sample['ydata_res'] = torch.from_numpy(ydata_res).float()
		return sample

