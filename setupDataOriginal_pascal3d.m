function setupDataOriginal_pascal3d(cls, db_path, voc_dir)
% function to setup pascal3d+ data. Read the data from PASCAL3D+_release1.1
% stored in db_path. Save resized patch inside ground truth bounding box.
% Usage: setupDataOriginal_pascal3d(cls, db_path, voc_dir);
% cls: class of interest
% db_path: location of the pascal3d+ data. eg: 'D:/datasets/pascal3d/PASCAL3D+_release1.1/';
% voc_dir: location of VOC2012 devkit to get train+val sets. eg: 'D:/datasets/VOCdevkit/VOC2012';

clc;
% paths and variables
save_dir = 'data/original';		% where all this will be stored. change or setup a symbolic link if necessary
save_location = fullfile(save_dir, cls);
if ~exist(save_location, 'dir'), mkdir(save_location); end
patch_size = [224, 224];
anno_dir = fullfile(db_path, 'Annotations');
image_dir = fullfile(db_path, 'Images');
sets_path = fullfile(db_path, 'Image_sets');

% start parallel processing
poolobj = parpool(16);

% get imagenet data: Train + Val
fprintf('**********************Imagenet-Train+Val: \n');
imagenet_train = read_file(fullfile(sets_path, sprintf('%s_imagenet_train.txt', cls))); 
imagenet_val = read_file(fullfile(sets_path, sprintf('%s_imagenet_val.txt', cls)));
image_path = fullfile(image_dir, sprintf('%s_imagenet', cls));
anno_path = fullfile(anno_dir, sprintf('%s_imagenet', cls));
image_extn = '.JPEG';
% imagenet-train
ind = zeros(length(imagenet_train), 1);
parfor i = 1:length(imagenet_train)
	fprintf('i: %d \n', i);
	ind(i) = process_image(cls, imagenet_train{i}, image_path, anno_path, image_extn, save_location, patch_size);
end
imagenet_train = imagenet_train(ind>0);
% imagenet-val
ind = zeros(length(imagenet_val), 1);
parfor i = 1:length(imagenet_val)
	fprintf('i: %d \n', i);
	ind(i) = process_image(cls, imagenet_val{i}, image_path, anno_path, image_extn, save_location, patch_size);
end
imagenet_val = imagenet_val(ind>0);

% get pascal data : Train + Val
fprintf('**********************Pascal-Train: \n');
pascal_train = read_file2(fullfile(voc_dir, 'ImageSets/Main',sprintf('%s_train.txt', cls)));
pascal_val = read_file2(fullfile(voc_dir, 'ImageSets/Main', sprintf('%s_val.txt', cls)));
image_path = fullfile(image_dir, sprintf('%s_pascal', cls));
anno_path = fullfile(anno_dir, sprintf('%s_pascal', cls));
image_extn = '.jpg';
% pascal-train
ind = zeros(length(pascal_train), 1);
parfor i = 1:length(pascal_train)
	fprintf('i: %d \n', i);
	ind(i) = process_image(cls, pascal_train{i}, image_path, anno_path, image_extn, save_location, patch_size);
end
pascal_train = pascal_train(ind>0);
% pascal-val
ind = zeros(length(pascal_val), 1);
parfor i = 1:length(pascal_val)
	fprintf('i: %d \n', i);
	ind(i) = process_image(cls, pascal_val{i}, image_path, anno_path, image_extn, save_location, patch_size);
end
pascal_val = pascal_val(ind>0);

% close parallel threads
delete(poolobj);

% save the train and val data
save(fullfile(save_dir, sprintf('%s_info', cls)), 'imagenet_train', 'imagenet_val', 'pascal_train', 'pascal_val');


function ind = process_image(cls, image_name, image_path, anno_path, image_extn, save_location, patch_size)
% function to save the resized extracted patch for future processing
ind = 0;
% load image
img = imread(fullfile(image_path, [image_name, image_extn]));
[nR, nC, d] = size(img);
if(d ~= 3), return; end
% load annotation
tmp = load(fullfile(anno_path, image_name));
objects = tmp.record.objects;
% run through all annotated objects in image
xdata = cell(1, length(objects));
ydata = cell(1, length(objects));
for j = 1:length(objects)
	object = objects(j);

	% ignore if object not of desired class
	if(~strcmp(object.class, cls)), continue; end
	% ignore if object is truncated or occluded
	if(object.truncated > 0 || object.occluded > 0), continue; end
	% ignore if fine viewpoint has not been annotated
	if(object.viewpoint.distance == 0), continue; end

	% get bbox and pose info
	az = object.viewpoint.azimuth;
	el = object.viewpoint.elevation;
	ct = object.viewpoint.theta;
% 	d = object.viewpoint.distance;
% 	f = object.viewpoint.focal * object.viewpoint.viewport;
% 	px = object.viewpoint.px;
% 	py = object.viewpoint.py;
	bbox = object.bbox;
	if(bbox(1) > nC || bbox(2) > nR), continue; end		% bad bbox
	
	% get patch inside bbox without any augmentation
	patch = get_patch(bbox, img, patch_size);
	% get rotation matrix and it's corresponding 3-dim angle-axis rep
	R1 = get_R(az, el, ct);
	y1 = get_v(R1);
	xdata{j} = shiftdim(patch, -1);
	ydata{j} = y1';
end

% collapse into single variables
xdata = cat(1, xdata{:});
ydata = cat(1, ydata{:});

% save results
if(~isempty(xdata))
	save(fullfile(save_location, image_name), 'xdata', 'ydata');
	ind = 1;
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


function image_list = read_file(filename)
% function to read file in filename and retrieve text info stored in it
disp(filename);
fid=fopen(filename,'r');
tmp=textscan(fid,'%s');
image_list=tmp{1};
fclose(fid);


function image_list=read_file2(filename)
% function to read pascal test-train splits
disp(filename);
fid=fopen(filename,'r');
tmp=textscan(fid,'%s %d');
image_list=tmp{1};
is_class=tmp{2};
image_list=image_list(is_class>0);

