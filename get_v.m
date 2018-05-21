function v=get_v(R)
% get 3-dim axis-angle representation of a rotation matrix
tR = 0.5*(trace(R)-1);
theta = acos(min(1, max(tR, -1)));	% angle of rotation
% get axis of rotation
tmp = 0.5*(R-R');
y = [tmp(3,2), tmp(1,3), tmp(2,1)]';
if(norm(y))
    u = y/norm(y);
else
    u = zeros(3,1);
end
% final axis-angle representation
v = theta*u;
