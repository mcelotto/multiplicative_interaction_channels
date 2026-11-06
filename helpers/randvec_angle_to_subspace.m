% draw a vector uniformly from the unit sphere in the subspace making angle theta
% to the column space of UA
function x = randvec_angle_to_subspace(UA, theta)
% UA: n x k matrix whose columns span the subspace
% theta: desired angle in radians, between 0 and pi/2

[n,~] = size(UA);

% Orthonormalize basis for the subspace
Q = orth(UA);

% Random unit vector in the subspace
a = randn(size(Q,2),1);
u = Q*a;
u = u / norm(u);

% Random unit vector in the orthogonal complement
b = randn(n,1);
v = b - Q*(Q'*b);   % remove subspace component
v = v / norm(v);

% Combine to get desired angle
x = cos(theta)*u + sin(theta)*v;
end
