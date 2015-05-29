function [U_final, w_final, nIter_final, objhistory_final] = GMKFCM(Ks, c, degree, options, U)
% Globalized Multiple Kernel Fuzzy c-means (LMKFCM)
%
% Input
%      Ks : (nSmp x nSmp x nKernel) kernel matrices
%       c : number of clusters
%  degree : fuzzification degree
% options : structure holding all settings
%       U : initial membership matrix
%
% Output
% U_final : membership matrix
% w_final : weight assignment to affinity matrices
% nIter_final 
% objhistory_final
%  weight : weight assignment to affinity matrices
%
% [1] Hsin-Chien Huang, Yung-Yu Chuang and Chu-Song Chen. Multiple Kernel Fuzzy Clustering, 
%    IEEE TRANSACTIONS ON FUZZY SYSTEMS
%
%**************************************************
%     Author: Liang Du <csliangdu@gmail.com>
%     Version: 1.0
%     Last modified: 2015-05-23 15:32:06
%**************************************************

if ~exist('options', 'var')
    options = [];
end

differror = 1e-5;
if isfield(options,'error')
    differror = options.error;
end

maxIter = 100;
if isfield(options, 'maxIter')
    maxIter = options.maxIter;
end

nRepeat = 1;
if isfield(options,'nRepeat')
    nRepeat = options.nRepeat;
end

minIterOrig = 10;
if isfield(options,'minIter')
    minIterOrig = options.minIter;
end
minIter = minIterOrig-1;

nSmp = size(Ks,1);
nKernel = size(Ks,3);

if isempty(U)
    U = abs(rand(nSmp,c));
    U = bsxfun(@rdivide, U, max(sum(U,2), eps));
else
    nRepeat = 1;
end

w = ones(nKernel,1)/nKernel;
K = calculate_kernel_w(Ks, w.^2);

selectInit = 1;
if nRepeat == 1
    selectInit = 0;
    minIterOrig = 0;
    minIter = 0;
    if isempty(maxIter)
        objhistory = CalculateObj(Ks, U, w, degree);
    else
        if isfield(options,'Converge') && options.Converge
            objhistory = CalculateObj(Ks, U, w, degree);
        end
    end
else
    if isfield(options,'Converge') && options.Converge
        error('Not implemented!');
    end
end

tryNo = 0;
while tryNo < nRepeat
    tryNo = tryNo+1;
    nIter = 0;
    maxErr = 1;
    
    while(maxErr > differror)
        % ===================== update U ========================
        U = KFCM(K, c, degree, struct('maxIter', [], 'nRepeat', 1), U);
        
        % ===================== update w ========================
        e = zeros(nKernel, 1);
        for iKernel = 1:nKernel
            e(iKernel) = CalculateObj(Ks(:,:,iKernel), U, 1, degree);
        end
        w = lp_weight_proj(e, 2);
        
        % ===================== update K ========================
        K = calculate_kernel_w(Ks, w.^2);
        
        nIter = nIter + 1;
        if nIter > minIter
            if selectInit
                objhistory = CalculateObj(Ks, U, w, degree);
                maxErr = 0;
            else
                if isempty(maxIter)
                    newobj = CalculateObj(Ks, U, w, degree);
                    objhistory = [objhistory; newobj]; %#ok<AGROW>
                    maxErr = abs(objhistory(end) - objhistory(end-1)) / abs(objhistory(end-1));
                else
                    if isfield(options,'Converge') && options.Converge
                        newobj = CalculateObj(Ks, U, w, degree);
                        objhistory = [objhistory newobj]; %#ok<AGROW>
                    end
                    maxErr = 1;
                    if nIter >= maxIter
                        maxErr = 0;
                        if isfield(options,'Converge') && options.Converge
                        else
                            objhistory = 0;
                        end
                    end
                end
            end
        end
    end
    
    if tryNo == 1
        U_final = U;
        w_final = w;
        K_final = K;
        nIter_final = nIter;
        objhistory_final = objhistory;
    else
        if objhistory(end) < objhistory_final(end)
            U_final = U;
            w_final = w;
            K_final = K;
            nIter_final = nIter;
            objhistory_final = objhistory;
        end
    end
    
    if selectInit
        if tryNo < nRepeat
            %re-start
            U = abs(rand(nSmp,c));
            U = bsxfun(@rdivide, U, max(sum(U,2), eps));
            w = ones(nKernel,1)/nKernel;
            K = calculate_kernel_w(Ks, w.^2);
        else
            tryNo = tryNo - 1;
            minIter = 0;
            selectInit = 0;
            U = U_final;
            w = w_final;
            K = K_final;
            objhistory = objhistory_final;
        end
    end
end

nIter_final = nIter_final + minIterOrig;

end

function obj = CalculateObj(Ks, U, w, degree)

nKernel = size(Ks,3);
if nKernel > 1
    K = calculate_kernel_w(Ks, w);
else
    K = Ks;
    clear Ks;
end

Um = U .^ degree;
d = sum(Um);
V = bsxfun(@rdivide, Um, max(d, eps));
KV = K * V;  % n^2 k
VKV = V' * KV; % k^2n

dist = ones(size(K, 1), 1) * diag(VKV)' - 2 * KV + diag(K) * ones(1, size(U, 2));

obj = sum(sum( dist .* Um ));
end

function K_w = calculate_kernel_w(Ks, w)
K_w = zeros(size(Ks(:, :, 1)));
for m = 1:size(Ks, 3)
    K_w = K_w + w(m) * Ks(:, :, m);
end
end

function a = lp_weight_proj(h, gamma)
% This function solve the following problem
%
%   \min_{a} \quad \sum_i a_i^gamma h_i = h^T a^gamma
%    s.t.    a_i >=0, \sum_i a_i = 1
%
% [1]Multi-View K-Means Clustering on Big Data, IJCAI, 2013
%

assert(gamma > 1, 'gamma should be (1, inf)');
tmp = 1 / ( 1 - gamma);
a = (gamma * h) .^tmp;
a = a / sum(a);
end