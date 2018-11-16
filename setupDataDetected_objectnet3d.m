function setupDataDetected_objectnet3d
% function to setup objectnet3d data with flips and 90-rotations

clear; clc;

% paths and variables
db_path = 'data/objectnet3d';
save_dir = fullfile(db_path, 'detected');
image_dir = fullfile(db_path, 'Images');
det_path = fullfile(db_path, 'vgg16_fast_rcnn_view_objectnet3d_selective_search_iter_160000');

% get category info
cls_file = fullfile(db_path, 'Image_sets/classes.txt');
fid = fopen(cls_file, 'r');
tmp = textscan(fid, '%s');
fclose(fid);
classes = tmp{1};
num_classes = length(classes);

% run through all classes
for i = 1:num_classes
	cls = classes{i};
	fid = fopen(fullfile(det_path, sprintf('detections_%s.txt', cls)), 'r');
	tmp = textscan(fid, '%s %f %f %f %f %f %f %f %f');
	image_names = tmp{1};
	bboxes = [tmp{2}, tmp{3}, tmp{4}, tmp{5}];
	det_scores = tmp{6};
	ypred = [tmp{7}, tmp{8}, tmp{9}];
	% make dir and store this
	save_path = fullfile(save_dir, cls);
	if ~exist(save_path, 'dir'), mkdir(save_path); end
	save(fullfile(save_dir, sprintf('%s_detinfo', cls)), 'image_names', 'bboxes', 'det_scores', 'ypred');
	% run through all detected bboxes to create images
	num_images = length(image_names);
	for j = 1:num_images
		fprintf('i: %d/%d \t j: %d/%d \n', i, num_classes, j, num_images);
		img = imread(fullfile(image_dir, [image_names{j}, '.JPEG']));
		patch = get_patch(bboxes(j, :), img);
		imwrite(patch, fullfile(save_path, sprintf('%s_%08d.png', cls, j)));
	end
end
 
function patch = get_patch(bbox, img)
% function to get patch inside gt-bbox
[nR, nC, ~] = size(img);
% extract patch inside bounding box
x1 = max(1, round(bbox(1))); x2 = min(nC, round(bbox(3)));
y1 = max(1, round(bbox(2))); y2 = min(nR, round(bbox(4)));
patch = img(y1:y2, x1:x2, :);
if size(patch, 3)==1, patch=cat(3, patch, patch, patch); end
% resize if necessary
scale = [size(patch, 1)/224, size(patch, 2)/224];
if any(scale>1), patch = imresize(patch, 1/max(scale)); end
