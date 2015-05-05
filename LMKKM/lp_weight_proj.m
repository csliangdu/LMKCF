function a = lp_weight_proj(h, gamma)
% This function solve the following problem
%
% 	\min_{a} \quad \sum_i a_i^gamma h_i = h^T a^gamma
%	 s.t.    a_i >=0, \sum_i a_i = 1
%
% [1]Multi-View K-Means Clustering on Big Data, IJCAI, 2013
%

assert(gamma > 1, 'gamma should be (1, inf)');
tmp = 1 / ( 1 - gamma);
a = (gamma * h) .^tmp;
a = a / sum(a);
end