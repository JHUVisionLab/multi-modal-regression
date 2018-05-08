function theta = computeGeodesicError(ytest, yhat)
% function to compute viewpoint angle between the gt and predicted pose
N = size(ytest, 1);
eps = 1e-10;
proj = [0, 0, 0, 0, 0, 1, 0, -1, 0; 0, 0, -1, 0, 0, 0, 1, 0, 0; 0, 1, 0, -1, 0, 0, 0, 0, 0];
theta = zeros(N, 1);
for n = 1:N
	v1 = ytest(n, :);
	v2 = yhat(n, :);
	t1 = norm(v1); t2 = norm(v2);
	nv1 = v1/max(t1, eps);
	nv2 = v2/max(t2, eps);
	sv1 = reshape(nv1 * proj, [3, 3]);
	sv2 = reshape(nv2 * proj, [3, 3]);
	R1 = eye(3) + sin(t1) * sv1 + (1-cos(t1)) * sv1 * sv1;
	R2 = eye(3) + sin(t2) * sv2 + (1-cos(t2)) * sv2 * sv2;
	R = R1'*R2;
	tmp = min(1-eps, max(-1+eps, 0.5*(trace(R)-1)));
	theta(n) = abs(acosd(tmp));
end
