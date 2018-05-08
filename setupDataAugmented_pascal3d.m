function setupDataAugmented_pascal3d(cls)

% paths and variables
save_dir = 'data/augmented2';	% where all this will be stored. change or setup a symbolic link if necessary
db_path = 'data/pascal3d';
voc_dir = 'data/pascal3d/PASCAL/VOCdevkit/VOC2012';
save_location = fullfile(save_dir, cls);
if ~exist(save_location, 'dir'), mkdir(save_location); end
anno_dir = fullfile(db_path, 'Annotations');
image_dir = fullfile(db_path, 'Images');
sets_path = fullfile(db_path, 'Image_sets');
cad_path = fullfile(db_path, 'CAD');

% load the models
tmp = load(fullfile(cad_path, cls));
models = tmp.(cls);

%poolobj = parpool(8);

% get imagenet data: Train + Val
fprintf('**********************Imagenet-Train+Val: \n');
imagenet_train = read_file(fullfile(sets_path, sprintf('%s_imagenet_train.txt', cls)));
imagenet_val = read_file(fullfile(sets_path, sprintf('%s_imagenet_val.txt', cls)));
image_path = fullfile(image_dir, sprintf('%s_imagenet', cls));
anno_path = fullfile(anno_dir, sprintf('%s_imagenet', cls));
image_extn = '.JPEG';
% imagenet-train
for i = 1:length(imagenet_train)
	fprintf('i: %d/%d \n', i, length(imagenet_train));
 	process_image(cls, imagenet_train{i}, image_path, anno_path, image_extn, save_location, models);
end
% imagenet-val
for i = 1:length(imagenet_val)
 	fprintf('i: %d/%d \n', i, length(imagenet_val));
 	process_image(cls, imagenet_val{i}, image_path, anno_path, image_extn, save_location, models);
end
% pascal-train
pascal_train = read_file2(fullfile(voc_dir, 'ImageSets/Main',sprintf('%s_train.txt', cls)));
image_path = fullfile(image_dir, sprintf('%s_pascal', cls));
anno_path = fullfile(anno_dir, sprintf('%s_pascal', cls));
image_extn = '.jpg';
for i = 1:length(pascal_train)
	fprintf('i: %d/%d \n', i, length(pascal_train));
	process_image(cls, pascal_train{i}, image_path, anno_path, image_extn, save_location, models);
end

%delete(poolobj);

function process_image(cls, image_name, image_path, anno_path, image_extn, save_location, models)
% function to save the resized extracted patch for future processing

% load image
img = imread(fullfile(image_path, [image_name, image_extn]));
[nR, nC, d] = size(img);
if(d ~= 3), return; end
% load annotation
tmp = load(fullfile(anno_path, image_name));
objects = tmp.record.objects;
% run through all annotated objects in image
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
	d = object.viewpoint.distance;
	f = object.viewpoint.focal * object.viewpoint.viewport;
	px = object.viewpoint.px;
	py = object.viewpoint.py;
	bbox = object.bbox;
	if(bbox(1) > nC || bbox(2) > nR), continue; end		% bad bbox
	
	% get vertices
	cad_index = object.cad_index;
	vertices = models(cad_index).vertices;

	% get patch inside bbox without any augmentation
	patch = get_patch(bbox, img);

	% get patches with jittered bounding box and data augmentation
	try [patches, targets] = get_augmented_patches(bbox, img, vertices, az, el, ct, d, f, px, py);
	catch err
		patches = {patch};
		targets = {[az, el, ct]};
	end
	
	% save everything
	imageid = get_id(image_name);
	for k = 1:length(patches)
		patch = patches{k};
		if isempty(patch), continue; end
		az = correct_angle(targets{k}(1));
		el = correct_angle(targets{k}(2));
		ct = correct_angle(targets{k}(3));
		imwrite(patch, fullfile(save_location, sprintf('%s_%sobject%d_a%03.1f_e%03.1f_t%03.1f_d%03.1f.png', cls, imageid, j, az, el, ct, d)));
	end
end

function patch = get_patch(bbox, img)
% function to get patch inside gt-bbox
[nR, nC, ~] = size(img);
% extract patch inside bounding box
x1 = max(1, round(bbox(1))); x2 = min(nC, round(bbox(3)));
y1 = max(1, round(bbox(2))); y2 = min(nR, round(bbox(4)));
patch = img(y1:y2, x1:x2, :);
scale = [size(patch, 1)/224, size(patch, 2)/224];
if any(scale>1), patch = imresize(patch, 1/max(scale)); end

function [patches, targets] = get_augmented_patches(bbox, img, vertices, az, el, ct, d, f, px, py)
% function to get augmented patches around gt-anno
% augmentation ranges - HARD CODED FOR NOW
az_range = -1:1;
el_range = -1:1;
ct_range = -4:2:4;
% image size
[nR, nC, ~] = size(img);
% get basic mask
x1 = max(1, round(bbox(1))); x2 = min(nC, round(bbox(3)));
y1 = max(1, round(bbox(2))); y2 = min(nR, round(bbox(4)));
mask = zeros(nR, nC);
mask(y1:y2, x1:x2) = 1;
mask = (mask>0);
% get visible vertices at annotated viewpoint
vis = get_visibility(vertices, az, el, ct, d);
[x, y] = project(vertices(vis, :), az, el, ct, d, px, py, f);
% perturb around az and ct to get new patches
patches = cell(length(az_range), length(el_range), length(ct_range), 2);
targets = cell(length(az_range), length(el_range), length(ct_range), 2);
for i = 1:length(az_range)
	for j = 1:length(el_range)
		for k = 1:length(ct_range)
			% pose for augmented image
			az_new = az + az_range(i);
			el_new = el + el_range(j);
			ct_new = ct + ct_range(k);
			% project model at new desired pose
			[x_tform, y_tform] = project(vertices(vis, :), az_new, el_new, ct_new, d, px, py, f);
			% fit homography between projections at original pose and new one
			tform = fitgeotrans([x, y], [x_tform, y_tform], 'projective');
			% transform image rectangle under homography
			[tmpx, tmpy] = transformPointsForward(tform, [1, nC], [1, nR]);
			% if homography causes extreme shape change, ignore it
			if(abs(diff(tmpx))>10*nC && abs(diff(tmpy))>10*nR), continue; end
			% transform image and mask under homography
			new_img = imwarp(img, tform);
			new_mask = imwarp(mask, tform);
			% get bbox in new image
			new_bbox = [find(sum(new_mask), 1, 'first'), find(sum(new_mask, 2), 1, 'first'), ...
				find(sum(new_mask), 1, 'last'), find(sum(new_mask, 2), 1, 'last')];
			if(isempty(new_bbox)), continue; end
			% extract the patch and it's flipped version in transformed image
			patch = new_img(new_bbox(2):new_bbox(4), new_bbox(1):new_bbox(3), :);
			scale = [size(patch, 1)/224, size(patch, 2)/224];
			if any(scale>1), patch = imresize(patch, 1/max(scale)); end
			patches{i, j, k, 1} = patch;
			targets{i, j, k, 1} = [az_new, el_new, ct_new];
			patches{i, j, k, 2} = fliplr(patch);
			targets{i, j, k, 2} = [-az_new, el_new, -ct_new];
		end
	end
end
patches = reshape(patches, 1, numel(patches));
targets = reshape(targets, 1, numel(targets));


function ind = get_visibility(P, a, b, c, d)
% get closest 25% of points which are assumed to be visible

% convert from object coordinate system to camera coordinate system
a = -a; b = 90+b; c = -c;
% compute cos and sin of each
sa = sind(a);
ca = cosd(a);
sb = sind(b);
cb = cosd(b);
sc = sind(c);
cc = cosd(c);
% get rotation R and translation T
R = [cc -sc 0; sc cc 0; 0 0 1] * [1 0 0; 0 cb -sb; 0 sb cb] * [ca -sa 0; sa ca 0; 0 0 1];
T = [0; 0; d];
% transform points in 3D space
Pn = bsxfun(@plus, P*R', T');
% compute distance from origin
distances = sqrt(sum(Pn.^2, 2));
sorted = sort(distances);
% get roughly 25% of closest points
th = sorted(ceil(0.25*length(distances)));
ind = (distances < th);


function [x, y] = project(P, a, b, c, d, u, v, f)
% project the points using the paramters a,b,c,d,u,v.
% convert from object coordinate system to camera coordinate system
a = -a; b = 90+b; c = -c;
% compute cos and sin of each
sa = sind(a);
ca = cosd(a);
sb = sind(b);
cb = cosd(b);
sc = sind(c);
cc = cosd(c);
% get rotation R and translation T
R = [cc -sc 0; sc cc 0; 0 0 1] * [1 0 0; 0 cb -sb; 0 sb cb] * [ca -sa 0; sa ca 0; 0 0 1];
T = [0; 0; d];
% transform the points in 3D
Pn = bsxfun(@plus, P*R', T');
% projection matrix
M = [f 0 0; 0 f 0; 0 0 1];
% project onto image plane and add 2D displacement
p = Pn*M';
x = p(:, 1)./p(:, 3) + u;
y = p(:, 2)./p(:, 3) + v;


function imageid = get_id(image_name)
ind = strfind(image_name, '_');
imageid = image_name(setdiff(1:length(image_name), ind));


function y = correct_angle(x)
if x < 0
	y = x + 360; 
else
	y = x; 
end


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
