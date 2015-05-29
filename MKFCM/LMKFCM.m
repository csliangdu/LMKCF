function [U_final, W_final, nIter_final, objhistory_final] = LMKFCM(Ks, c, degree, options, U)
% Localized Multiple Kernel Fuzzy c-means (LMKFCM)
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
% W_final : weight assignment to affinity matrices
% nIter_final 
% objhistory_final
%  weight : weight assignment to affinity matrices
%
%**************************************************
%     Author: Liang Du <csliangdu@gmail.com>
%     Version: 1.0
%     Last modified: 2015-05-23 14:29:01
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

W = ones(nSmp, nKernel)/nKernel;
K = calculate_localized_kernel_W(Ks, W);

selectInit = 1;
if nRepeat == 1
    selectInit = 0;
    minIterOrig = 0;
    minIter = 0;
    if isempty(maxIter)
        objhistory = CalculateObj(Ks, U, W, degree);
    else
        if isfield(options,'Converge') && options.Converge
            objhistory = CalculateObj(Ks, U, W, degree);
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
        
        % ===================== update W ========================
        Um = U .^ degree;
        d1 = sum(Um);
        V = bsxfun(@rdivide, Um, max(d1, eps));
        d2 = sum(Um, 2);
        M = diag(d2) + V * diag(d1) * V' - 2 * V * Um';

        clear Um d1 d2 V;
        Km = zeros(nSmp, nSmp, nKernel);
        for iKernel = 1:nKernel
            Km(:,:,iKernel) = M .* Ks(:,:,iKernel);
        end
        W = update_W_apg(Km, W);
        
        % ===================== update K ========================
        K = calculate_localized_kernel_W(Ks, W);
        
        nIter = nIter + 1;
        if nIter > minIter
            if selectInit
                objhistory = CalculateObj(Ks, U, W, degree);
                maxErr = 0;
            else
                if isempty(maxIter)
                    newobj = CalculateObj(Ks, U, W, degree);
                    objhistory = [objhistory; newobj]; %#ok<AGROW>
                    maxErr = abs(objhistory(end) - objhistory(end-1)) / abs(objhistory(end-1));
                else
                    if isfield(options,'Converge') && options.Converge
                        newobj = CalculateObj(Ks, U, W, degree);
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
        W_final = W;
        K_final = K;
        nIter_final = nIter;
        objhistory_final = objhistory;
    else
        if objhistory(end) < objhistory_final(end)
            U_final = U;
            W_final = W;
            K_final = K;
            nIter_final = nIter;
            objhistory_final = objhistory;
        end
    end
    
    if selectInit
        if tryNo < nRepeat
            %re-start
            U = abs(rand(nSmp,c));
            W = ones(nSmp, nKernel)/nKernel;
            K = calculate_localized_kernel_W(Ks, W);
        else
            tryNo = tryNo - 1;
            minIter = 0;
            selectInit = 0;
            U = U_final;
            W = W_final;
            K = K_final;
            objhistory = objhistory_final;
        end
    end
end

nIter_final = nIter_final + minIterOrig;

end

function obj = CalculateObj(Ks, U, W, degree)

nKernel = size(Ks,3);
if nKernel > 1
    K = calculate_localized_kernel_W(Ks, W);
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



function K_W = calculate_localized_kernel_W(K, W)
K_W = zeros(size(K(:, :, 1)));
for m = 1:size(K, 3)
    K_W = K_W + (W(:, m) * W(:, m)') .* K(:, :, m);
end
end

function [X_new, objHistory] = update_W_apg(Ks, X_old)
%
% [1] An Accelerated Gradient Method for Trace Norm Minimization, ICML 09, Algorithm 2
%

a = 1;
gamma = 1.05;
Lf = 100;
maxiter_pg = 100;
myeps = 1e-8;
epsilon = inf;

objHistory = [];

iter = 0;
Z_new = X_old;
a_new = a;
while epsilon > myeps && iter < maxiter_pg
    
    %%----------------------------------------------------------
    % search the valid Lipschitz constant, which satisfies
    %   Obj( proj(Z_old, Lf)) < Obj_pg( proj(Z_old, Lf), Z_old)
    %%----------------------------------------------------------
    Lf_candi = Lf;
    is_valid_Lf = true;
    Z_old = Z_new;
    % extract non-loop parts
    [obj_pg_1, grad_Z] = obj_W(Ks, Z_old);
    while is_valid_Lf
        % compute objective value
        Y = Z_old - 1 ./ Lf_candi * grad_Z;
        X_candi = SimplexProj(Y);
        obj_candi = obj_W(Ks, X_candi);

        % compute objective value of auxilary function
        X_diff = (X_candi - Z_old);
        obj_pg_2 = sum(sum(X_diff .* grad_Z));
        obj_pg_3 = sum(sum(X_diff .* X_diff));
        obj_pg = obj_pg_1 + obj_pg_2 + .5 * Lf_candi * obj_pg_3;
        if obj_candi > obj_pg
            Lf_candi = Lf_candi * gamma;
        else
            is_valid_Lf = false;
        end
    end
    Lf = Lf_candi; % get the valid Lipschitz constant
    if iter > 1
        X_old = X_new;
    end
    X_new = X_candi; % get the result from proximal operator with valid Lipschitz constant
    a_old = a_new;
    a_new = ( 1 + sqrt(1 + 4 * a_old^2)) / 2;
    Z_new = X_new + (a_old - 1) / a_new * (X_new - X_old);
    
    obj = obj_W(Ks, X_new);
    objHistory = [objHistory; obj]; %#ok
    
    epsilon = sum(sum((X_new - X_old).^2));
    
    if iter > 1 && objHistory(end) > objHistory(end-1)
        epsilon = 0;
        X_new = X_old;
        objHistory = objHistory(1:end-1);
    end
    iter = iter + 1;
end
end

function [obj, grad_W] = obj_W(Ks, W)
[n, ~, p] = size(Ks);
obj = 0;
grad_W = zeros(n,p);
for i = 1:p
    grad_W(:,i) = Ks(:,:,i) * W(:,i);
    obj = obj + W(:,i)' * grad_W(:,i);
end
end