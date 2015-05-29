function [U_final, V_final, w_final, nIter_final, objhistory_final] = GMKCF_lp_simplex_Multi(Ks, c, p, options, U, V)
% Globalized Multiple Kernel Concept Factorization (GMKCF)
%
% where
%   X
% Notation:
% K ... (nSmp x nSmp) kernel matrix
% c ... number of hidden factors
%
% options ... Structure holding all settings
%
% You only need to provide the above three inputs.
%
%
%**************************************************
%     Author: Liang Du <csliangdu@gmail.com>
%     Version: 1.0
%     Last modified: 2015-04-15 21:49:42
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

minIterOrig = 30;
if isfield(options,'minIter')
    minIterOrig = options.minIter;
end
minIter = minIterOrig-1;

meanFitRatio = 0.1;
if isfield(options,'meanFitRatio')
    meanFitRatio = options.meanFitRatio;
end

Norm = 2;
NormV = 1;

nSmp = size(Ks,1);
nKernel = size(Ks,3);

if isempty(U)
    U = abs(rand(nSmp,c));
    V = abs(rand(nSmp,c));
else
    nRepeat = 1;
end

w = lp_simplex_proj(ones(nKernel,1), p);
K = calculate_kernel_theta(Ks, w);
[U,V] = NormalizeUV(K, U, V, NormV, Norm);

selectInit = 1;
if nRepeat == 1
    selectInit = 0;
    minIterOrig = 0;
    minIter = 0;
    if isempty(maxIter)
        objhistory = CalculateObj(Ks, U, V, w);
        meanFit = objhistory*10;
    else
        if isfield(options,'Converge') && options.Converge
            objhistory = CalculateObj(Ks, U, V, w);
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
        % ===================== update U/V ========================
        [U, V] = KCF_Multi(K, c, struct('maxIter', [], 'nRepeat', 1), U, V);
        
        % ===================== update w ========================
        
        q = zeros(nKernel, 1);
        for iKernel = 1:nKernel
            q(iKernel) = CalculateObj(Ks(:,:,iKernel), U, V, 1);
        end
        w = lp_simplex_proj(q, p);
        K = calculate_kernel_theta(Ks, w);
        
        nIter = nIter + 1;
        if nIter > minIter
            [U,V] = NormalizeUV(K, U, V, NormV, Norm);
            if selectInit
                objhistory = CalculateObj(Ks, U, V, w);
                maxErr = 0;
            else
                if isempty(maxIter)
                    newobj = CalculateObj(Ks, U, V, w);
                    objhistory = [objhistory; newobj]; %#ok<AGROW>
                    meanFit = meanFitRatio*meanFit + (1-meanFitRatio)*newobj;
                    maxErr = (meanFit-newobj)/meanFit;
                else
                    if isfield(options,'Converge') && options.Converge
                        newobj = CalculateObj(Ks, U, V, w);
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
        V_final = V;
        w_final = w;
        nIter_final = nIter;
        objhistory_final = objhistory;
    else
        if objhistory(end) < objhistory_final(end)
            U_final = U;
            V_final = V;
            w_final = w;
            nIter_final = nIter;
            objhistory_final = objhistory;
        end
    end
    
    if selectInit
        if tryNo < nRepeat
            %re-start
            U = abs(rand(nSmp,c));
            V = abs(rand(nSmp,c));
            w = lp_simplex_proj(ones(nKernel,1), p);
            K = calculate_kernel_theta(Ks, w);
            [U,V] = NormalizeUV(K, U, V, NormV, Norm);
        else
            tryNo = tryNo - 1;
            minIter = 0;
            selectInit = 0;
            U = U_final;
            V = V_final;
            w = w_final;
            K = calculate_kernel_theta(Ks, w);
            objhistory = objhistory_final;
            meanFit = objhistory*10;
        end
    end
end

nIter_final = nIter_final + minIterOrig;

Norm = 2;
NormV = 0;

[U_final,V_final] = NormalizeUV(K, U_final, V_final, NormV, Norm);

end

%==========================================================================

function [obj] = CalculateObj(Ks, U, V, w)
nKernel = size(Ks,3);

if nKernel > 1
    K = calculate_kernel_theta(Ks, w);
else
    K = Ks;
    clear Ks;
end
UK = U'*K; % n^2k
UKU = UK*U; % nk^2
VUK = V*UK; % n^2k
VV = V'*V; % nk^2
obj = sum(diag(K))-2*sum(diag(VUK))+sum(sum(UKU.*VV));
end


function [U, V] = NormalizeUV(K, U, V, NormV, Norm)
k = size(U,2);
if Norm == 2
    if NormV
        norms = max(1e-15,sqrt(sum(V.^2,1)))';
        V = V*spdiags(norms.^-1,0,k,k);
        U = U*spdiags(norms,0,k,k);
    else
        norms = max(1e-15,sqrt(sum(U.*(K*U),1)))';
        U = U*spdiags(norms.^-1,0,k,k);
        V = V*spdiags(norms,0,k,k);
    end
else
    if NormV
        norms = max(1e-15,sum(abs(V),1))';
        V = V*spdiags(norms.^-1,0,k,k);
        U = U*spdiags(norms,0,k,k);
    else
        norms = max(1e-15,sum(U.*(K*U),1))';
        U = U*spdiags(norms.^-1,0,k,k);
        V = V*spdiags(norms,0,k,k);
    end
end
end

function K_theta = calculate_kernel_theta(Ks, theta)
K_theta = zeros(size(Ks(:, :, 1)));
for m = 1:size(Ks, 3)
    K_theta = K_theta + theta(m) * Ks(:, :, m);
end
end

function a = lp_simplex_proj(h, alpha)
% This function solve the following problem
%
%   \min_{a} \quad \sum_i a_i h_i = a^T h
%    s.t.    a_i >=0, \sum_i a_i^alpha = 1
%
% [1]Weighted Feature Subset Non-negative Matrix Factorization and Its Applications to Document Understanding. ICDM 2010
%

assert( (alpha > 0) && (alpha < 1), 'alpha should be (0, 1)');
t1 = 1 / alpha;
t2 = 1 / (alpha - 1);
t3 = alpha / (alpha - 1);

t4 = sum(h .^ t3);
t5 = (1 / t4)^t1;
t6 = h .^ t2;
a = t5 * t6;
end