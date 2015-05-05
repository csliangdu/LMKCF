function [U_final, V_final, nIter_final, objhistory_final] = MACF(K, nCluster, mak_options, options, U, V)
% Clustering analysis using manifold kernel concept factorization
%
% Notation:
% K ... (nSmp x nSmp) kernel matrix 
% k ... number of hidden factors
% options ... Structure holding all settings
%
% You only need to provide the above four inputs.
%
%
% References:
% [1] Wei Xu, Yihong Gong. Document Clustering by Concept Factorization, SIGIR 2004
% [2] Deng Cai, Xiaofei He, Jiawei Han, "Locally Consistent Concept
%     Factorization for Document Clustering", IEEE Transactions on Knowledge
%     and Data Engineering, Vol. 23, No. 6, pp. 902-913, 2011.   
% [3] Ping Li, Chun Chen, Jiajun Bu. Clustering analysis using manifold kernel concept factorization, Neurocomputing.
%
%**************************************************
%     Author: Liang Du <csliangdu@gmail.com>
%     Version: 1.0
%     Last modified: 2015-05-02 18:36:44
%**************************************************

K = ManifoldKernel(K, mak_options.reg, mak_options.W, mak_options.X, mak_options.k, mak_options.isBinary);
[U_final, V_final, nIter_final, objhistory_final] = KCF_Multi(K, nCluster, options, U, V);
end