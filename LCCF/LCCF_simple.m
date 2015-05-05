function [U_final, V_final, nIter_final, objhistory_final] = LCCF_simple(K, nCluster, k, options, U, V)
% Locally Consistant Concept Factorization (LCCF)
W = Kernel2KNN(K, k);
[U_final, V_final, nIter_final, objhistory_final] = LCCF_Multi(K, nCluster, W, options, U, V);
