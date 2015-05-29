function X = SimplexProj(Y)
% X = SimplexProj(Y) Projection onto the probability simplex
%
% It projects each row of matrix Y onto the probability simplex. That is,
% if y is a 1xD vector, this function solves
%   min_x |x-y|   s.t.  x_1+...+x_D = 1, x_i>=0 for i=1,...,D.
%
% The runtime for a vector of 1xD is O(D.log(D)).
% The code is efficient and fully vectorised.
%
% References:
% - W. Wang and M. A. Carreira-Perpinan: "Projection onto the probability
%   simplex: an efficient algorithm with a simple proof, and an application".
%   arXiv:1309.1541, Sep. 3, 2013.
%
% Input:
%   Y: NxD matrix.
% Output:
%   X: NxD matrix.
%
% Copyright (c) 2013 by Weiran Wang and Miguel A. Carreira-Perpinan.
%
[N,D] = size(Y);
X = sort(Y,2,'descend');
Xtmp = (cumsum(X,2)-1)*diag(sparse(1./(1:D)));

% numerical underflow
tmp = sum(X>Xtmp,2);
tmp = max(tmp, 1); 

X = max(bsxfun(@minus,Y,Xtmp(sub2ind([N,D],(1:N)',tmp))),0);

% numerical underflow
idx = find(sum(X,2) ==0); 
[~, idx2] = max(Y,[],2);
X(sub2ind(size(X), idx, idx2(idx))) = 1;
end
