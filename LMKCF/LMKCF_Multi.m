function [U_final, V_final, Z_final, nIter_final, objhistory_final] = LMKCF_Multi(Ks, c, options, U, V)
% Localized Multiple Kernel Concept Factorization (LMKCF)
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

Z = ones(nSmp, nKernel)/nKernel;
K = calculate_localized_kernel_theta(Ks, Z);

if isempty(U)
    U = abs(rand(nSmp,c));
    V = abs(rand(nSmp,c));
    [U,V] = NormalizeUV(K, U, V, NormV, Norm);
else
    nRepeat = 1;
end



selectInit = 1;
if nRepeat == 1
    selectInit = 0;
    minIterOrig = 0;
    minIter = 0;
    if isempty(maxIter)
        objhistory = CalculateObj(Ks, U, V, Z);
        meanFit = objhistory*10;
    else
        if isfield(options,'Converge') && options.Converge
            objhistory = CalculateObj(Ks, U, V, Z);
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
        
        % ===================== update Z ========================
        UV = U * V';       % n^2k
        M = eye(nSmp) - UV;
        M = M * M';        % n^3
        M = (M + M') / 2;
        clear UV;
        Km = zeros(nSmp, nSmp, nKernel);
        for iKernel = 1:nKernel
            Km(:,:,iKernel) = M .* Ks(:,:,iKernel);
        end
        Z = update_W_apg( Km, Z);
        
        K = calculate_localized_kernel_theta(Ks, Z);
        
        nIter = nIter + 1;
        if nIter > minIter
            [U,V] = NormalizeUV(K, U, V, NormV, Norm);
            if selectInit
                objhistory = CalculateObj(Ks, U, V, Z);
                maxErr = 0;
            else
                if isempty(maxIter)
                    newobj = CalculateObj(Ks, U, V, Z);
                    objhistory = [objhistory; newobj]; %#ok<AGROW>
                    meanFit = meanFitRatio*meanFit + (1-meanFitRatio)*newobj;
                    maxErr = (meanFit-newobj)/meanFit;
                else
                    if isfield(options,'Converge') && options.Converge
                        newobj = CalculateObj(Ks, U, V, Z);
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
        Z_final = Z;
        nIter_final = nIter;
        objhistory_final = objhistory;
    else
        if objhistory(end) < objhistory_final(end)
            U_final = U;
            V_final = V;
            Z_final = Z;
            nIter_final = nIter;
            objhistory_final = objhistory;
        end
    end
    
    if selectInit
        if tryNo < nRepeat
            %re-start
            U = abs(rand(nSmp,c));
            V = abs(rand(nSmp,c));
            Z = ones(nSmp, nKernel)/nKernel;
            K = calculate_localized_kernel_theta(Ks, Z);
            [U,V] = NormalizeUV(K, U, V, NormV, Norm);
        else
            tryNo = tryNo - 1;
            minIter = 0;
            selectInit = 0;
            U = U_final;
            V = V_final;
            Z = Z_final;
            K = calculate_localized_kernel_theta(Ks, Z);
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

function [obj] = CalculateObj(Ks, U, V, Z)
nKernel = size(Ks,3);
if nKernel > 1
    K = calculate_localized_kernel_theta(Ks, Z);
else
    K = Ks;
    clear Ks;
end
UK = U'*K;        % n^2k
UKU = UK*U;    % nk^2
VUK = V*UK;    % n^2k
VV = V'*V;        % nk^2
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

function K_Theta = calculate_localized_kernel_theta(K, Theta)
K_Theta = zeros(size(K(:, :, 1)));
for m = 1:size(K, 3)
    K_Theta = K_Theta + (Theta(:, m) * Theta(:, m)') .* K(:, :, m);
end
end

function [X_new, objHistory] = update_W_apg(Ks, X_old)
%
% [1] An Accelerated Gradient Method for Trace Norm Minimization, ICML 09, Algorithm 2
%

[n, ~, p] = size(Ks);

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
