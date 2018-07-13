function setupDataDetection_maskrcnn
clear; clc; close all;

classes = {'aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', ...
	'chair', 'diningtable', 'motorcycle', 'sofa', 'train', 'tvmonitor'};
num_classes = length(classes);

% relevant paths
pascal3d_path = 'data/pascal3d';
db_path = fullfile(pascal3d_path, 'PASCAL/VOCdevkit/VOC2012');
mat_path = 'data/maskrcnn_dets_nofinetune';
if ~exist(mat_path, 'dir'), mkdir(mat_path); end
img_path = fullfile(db_path, 'JPEGImages');
sets_file = fullfile(db_path, 'ImageSets/Main/val.txt');
dest_path = fullfile(mat_path, 'all');
if ~exist(dest_path, 'dir'), mkdir(dest_path); end
patch_size = [224, 224];
results_path = 'data/X-101-64x4d-FPN_1x_results_nofinetune/';

% get list of all test images
fid = fopen(sets_file, 'r');
tmp = textscan(fid, '%s');
image_names = tmp{1};
fclose(fid);
num_images = length(image_names);

% process detected results returned
dets = cell(1, num_classes);
for i = 1:num_classes
	cls = classes{i};
	filename = fullfile(results_path, sprintf('results_%s.txt', cls));
	fid = fopen(filename, 'r');
	tmp = textscan(fid, '%s %f %f %f %f %f\n');
	fclose(fid);
	det_images = tmp{1};
	det_boxes = [tmp{2}, tmp{3}, tmp{4}, tmp{5}];
	det_scores = tmp{6};
	cls_dets = cell(1, num_images);
	for j = 1:num_images
		ind = strcmp(det_images, image_names{j});
		cls_dets{j} = [det_boxes(ind, :), det_scores(ind)];
	end
	dets{i} = cls_dets;
end

% get images
for i = 1:length(image_names)
	image_name = image_names{i};
	bboxes = cell(1, num_classes); 
	labels = cell(1, num_classes);
	for j = 1:num_classes
		bboxes{j} = dets{j}{i};
		labels{j} = j*ones(size(dets{j}{i}, 1), 1);
	end
	bboxes = cat(1, bboxes{:});
	labels = cat(1, labels{:});
	xdata = cell(1, size(bboxes, 1));
	% read image
	img = imread(fullfile(img_path, sprintf('%s.jpg', image_name)));
	for k = 1:size(bboxes, 1)
		patch = get_patch(bboxes(k, :), img, patch_size);
		xdata{k} = shiftdim(patch, -1); 
	end
	xdata = cat(1, xdata{:});
	fprintf('image: %d \t num_boxes: %d \n', i, length(labels));
	save(fullfile(dest_path, image_name), 'xdata', 'bboxes', 'labels');
end

function patch = get_patch(bbox, img, patch_size)
% function to get patch inside gt-bbox
[nR, nC, ~] = size(img);
% extract patch inside bounding box
x1 = max(1, round(bbox(1))); x2 = min(nC, round(bbox(3)));
y1 = max(1, round(bbox(2))); y2 = min(nR, round(bbox(4)));
patch = img(y1:y2, x1:x2, :);
% resize patch to canonical size
patch = imresize(patch, patch_size) ;
