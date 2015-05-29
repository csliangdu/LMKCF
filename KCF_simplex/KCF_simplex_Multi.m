function [U_final, V_final, nIter_final, objhistory_final] = KCF_simplex_Multi(K, k, options, U, V)
% Document Clustering by Concept Factorization
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
%
%
%   version 2.0 --April/2010
%   version 1.0 --Dec./2008
%
%   Written by Deng Cai (dengcai AT gmail.com)
%
%**************************************************
%     Author: Liang Du <csliangdu@gmail.com>
%     Version: 1.0
%     Last modified: 2015-05-02 17:09:28
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

lambda = 10.^3;
if isfield(options, 'lambda')
    lambda = options.lambda;
end

nSmp = size(K,1);

if isempty(U)
    U = abs(rand(nSmp,k));
    V = abs(rand(nSmp,k));
    V = bsxfun(@rdivide, V, max(sum(V,2), eps));
else
    nRepeat = 1;
end


selectInit = 1;
if nRepeat == 1
    selectInit = 0;
    minIterOrig = 0;
    minIter = 0;
    if isempty(maxIter)
        objhistory = CalculateObj(K, U, V);
        meanFit = objhistory*10;
    else
        if isfield(options,'Converge') && options.Converge
            objhistory = CalculateObj(K, U, V);
        end
    end
else
    if isfield(options,'Converge') && options.Converge
        error('Not implemented!');
    end
end

nnFlag = 1;
if min(min(K)) < 0
    nnFlag = 0;
    Kp = (abs(K) + K)/2;
    Kn = (abs(K) - K)/2;
end

tryNo = 0;
while tryNo < nRepeat
    tryNo = tryNo+1;
    nIter = 0;
    maxErr = 1;
    objhistory_debug = [];
    while(maxErr > differror)
        % ===================== update U ========================
        if nnFlag
            KV = K*V;               % n^2k
            VV = V'*V;              % nk^2
            KUVV = K*U*VV;          % n^2k
            
            U = U.*(KV./max(KUVV,1e-10));
            clear KV VV KUVV;
        else
            VV = V'*V;                      % nk^2
            UVV = U * VV;                % nk^2
            KpUVV = Kp*UVV;         % n^2k
            KnUVV = Kn*UVV;          % n^2k
            
            KV = K*V;                % n^2k
            
            U = U.*( (KV + sqrt(KV.^2 + 4 * KpUVV .* KnUVV)) ./ max(2 * KpUVV,1e-10) );
            
            clear VV UVV KpUVV KnUVV KV;
        end
        
        % ===================== update V ========================
        if 0
            if nnFlag
                B = K * U;    % n^2 k
                A = U' * B;   % k^2 n
                
                VA = V * A;         % k^2 n
                
                %t1 = B + lambda * ones(size(V,1), size(V,2));
                %t2 = max(VA + lambda * V * ones(size(V,2), size(V,2)), eps);
                t1 = B;
                t2 = max(VA + lambda *  ones(size(V,1), size(V,2)), eps);
                
                V = V .* (t1 ./ t2).^(1/2);
                clear B A VA t1 t2;
                
                V = bsxfun(@rdivide, V, max(sum(V,2), eps));
            else
                % ===================== update V ========================
                Bp = Kp * U;    % n^2 k
                Bn = Kn * U;    % n^2 k
                Ap = U' * Bp;   % k^2 n
                An = U' * Bn;   % k^2 n
                
                VAp = V * Ap;     % k^2 n
                VAn = V * An;     % k^2 n
                
                %            t1 = VAn + Bp + lambda * ones(size(V,1), size(V,2));
                %            t2 = max(VAp + Bn + lambda * V * ones(size(V,2), size(V,2)), eps);
                t1 = VAn + Bp;
                t2 = max(VAp + Bn + lambda  * ones(size(V,1), size(V,2)), eps);
                V = V .* (t1 ./ t2).^(1/2);
                clear Bp Bn Ap An VAp VAn t1 t2;
                
                V = bsxfun(@rdivide, V, max(sum(V,2), eps));
            end
        else
            
            V = update_V_apg(K, U, V);
        end
        %         newobj = CalculateObj(K, U, V);
        %         objhistory_debug = [objhistory_debug; newobj]; %#ok<AGROW>
        
        nIter = nIter + 1;
        if nIter > minIter
            if selectInit
                objhistory = CalculateObj(K, U, V);
                maxErr = 0;
            else
                if isempty(maxIter)
                    newobj = CalculateObj(K, U, V);
                    objhistory = [objhistory newobj]; %#ok<AGROW>
                    meanFit = meanFitRatio*meanFit + (1-meanFitRatio)*newobj;
                    maxErr = (meanFit-newobj)/meanFit;
                else
                    if isfield(options,'Converge') && options.Converge
                        newobj = CalculateObj(K, U, V);
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
        nIter_final = nIter;
        objhistory_final = objhistory;
    else
        if objhistory(end) < objhistory_final(end)
            U_final = U;
            V_final = V;
            nIter_final = nIter;
            objhistory_final = objhistory;
        end
    end
    
    if selectInit
        if tryNo < nRepeat
            %re-start
            U = abs(rand(nSmp,k));
            V = abs(rand(nSmp,k));
            V = bsxfun(@rdivide, V, max(sum(V,2), eps));
        else
            tryNo = tryNo - 1;
            minIter = 0;
            selectInit = 0;
            U = U_final;
            V = V_final;
            objhistory = objhistory_final;
            meanFit = objhistory*10;
        end
    end
end

nIter_final = nIter_final + minIterOrig;

end


function obj_KCF = CalculateObj(K, U, V)
UK = U'*K;  % n^2k
UKU = UK*U; % nk^2
VUK = V*UK; % n^2k
VV = V'*V;  % nk^2
obj_KCF = sum(diag(K))-2*sum(diag(VUK))+sum(sum(UKU.*VV));
end



function [X_new, objHistory] = update_U_apg(K, X_old, V)
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
    obj_pg_1 = CalculateObj(K, Z_old, V);
    tmp = Z_old * (V' * V) - V;
    grad_Z = 2 * K * tmp;
    while is_valid_Lf
        % compute objective value
        Y = Z_old - 1 ./ Lf_candi * grad_Z;
        X_candi = SimplexProj(Y);
        obj_candi = CalculateObj(K, X_candi, V);
        
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
    
    obj = CalculateObj(K, X_new, V); % obj_candi;
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

function [X_new, objHistory] = update_V_apg(K, U, X_old)
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
    obj_pg_1 = CalculateObj(K, U, Z_old);
    KU = K * U;
    UKU = U' * KU;
    grad_Z = 2 * Z_old * UKU - 2 * KU;
    while is_valid_Lf
        % compute objective value
        Y = Z_old - 1 ./ Lf_candi * grad_Z;
        X_candi = SimplexProj(Y);
        obj_candi = CalculateObj(K, U, X_candi);
        
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
    
    obj = CalculateObj(K, U, X_new); % obj_candi;
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

function [U] = update_U_nn(K, U, V)
nnFlag = 1;
if min(min(K)) < 0
    nnFlag = 0;
    Kp = (abs(K) + K)/2;
    Kn = (abs(K) - K)/2;
end
if nnFlag
    % ===================== update U ========================
    KV = K*V;               % n^2k
    VV = V'*V;              % nk^2
    KUVV = K*U*VV;          % n^2k
    
    U = U.*(KV./max(KUVV,1e-10));
    clear KV VV KUVV;
else
    % ===================== update U ========================
    VV = V'*V;               % nk^2
    UVV = U * VV;            % nk^2
    KpUVV = Kp*UVV;          % n^2k
    KnUVV = Kn*UVV;          % n^2k
    
    KV = K*V;                % n^2k
    
    U = U.*( (KV + sqrt(KV.^2 + 4 * KpUVV .* KnUVV)) ./ max(2 * KpUVV,1e-10) );
    
    clear VV UVV KpUVV KnUVV KV;
end
end


function [V] = update_V_nn(K, U, V)
nnFlag = 1;
if min(min(K)) < 0
    nnFlag = 0;
    Kp = (abs(K) + K)/2;
    Kn = (abs(K) - K)/2;
end

if nnFlag
    % ===================== update V ========================
    KU = K*U;               % n^2k
    UKU = U'*KU;            % nk^2
    VUKU = V*UKU;           % nk^2
    
    V = V.*(KU./max(VUKU,1e-10));
    clear WV DV KU UKU VUKU;
    
else
    % ===================== update V ========================
    KpU = Kp*U;                 % n^2k
    UKpU = U'*KpU;              % nk^2
    VUKpU = V*UKpU;             % nk^2
    
    KnU = Kn*U;                 % n^2k
    UKnU = U'*KnU;              % nk^2
    VUKnU = V*UKnU;             % nk^2
    
    KU = K*U;                   % n^2k
    V = V.*( (KU + sqrt(KU.^2 + 4 * VUKpU .* VUKnU)) ./ max(2 * VUKpU,1e-10) );
    
    clear KpU UKpU VUKpU KnU UKnU VUKnU KU;
end
end


function obj = CalculateObj_simplex(K, U, V, lambda)
obj = CalculateObj(K, U, V);
obj2 = sum(V,2) - ones(size(V,1),1);
obj2 = sum(obj2.^2);
obj = obj + lambda * obj2;
end