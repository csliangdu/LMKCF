function [U_final, V_final, nIter_final, objhistory_final] = KCF_orth_Multi(K, k, options, U, V)
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

lambda = 10.^6;
if isfield(options, 'lambda')
    lambda = options.lambda;
end

nSmp = size(K,1);

if isempty(U)
    U = abs(rand(nSmp,k));
    V = abs(rand(nSmp,k));
else
    nRepeat = 1;
end

selectInit = 1;
if nRepeat == 1
    selectInit = 0;
    minIterOrig = 0;
    minIter = 0;
    if isempty(maxIter)
        objhistory = CalculateObj_orth(K, U, V, lambda);
        meanFit = objhistory*10;
    else
        if isfield(options,'Converge') && options.Converge
            objhistory = CalculateObj_orth(K, U, V, lambda);
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
    while(maxErr > differror)
        
        if nnFlag
            % ===================== update U ========================
            KV = K*V;               % n^2k
            VV = V'*V;              % nk^2
            KUVV = K*U*VV;          % n^2k
            
            U = U.*(KV./max(KUVV,1e-10));
            clear KV VV KUVV;
            
            % ===================== update V ========================
            B = K * U;    % n^2 k
            A = U' * B;   % k^2 n
            
            VTV = V' * V;       % k^2 n
            VVTV = V * VTV; % k^2 n
            VA = V * A;         % k^2 n
            
            t1 = B + 2 * lambda * V;
            %t2 = max(VA + 2 * lambda * VVTV, eps);
            t2 = max( 2 * lambda * VVTV, eps);
            V = V .* (t1 ./ t2).^(1/4);
            %V = V * diag(sqrt(1./(diag(V'*V)+eps)));
            clear B A VTV VVTV VA t1 t2;
        else
            % ===================== update U ========================
            VV = V'*V;                      % nk^2
            UVV = U * VV;                % nk^2
            KpUVV = Kp*UVV;         % n^2k
            KnUVV = Kn*UVV;          % n^2k
            
            KV = K*V;                % n^2k
            
            U = U.*( (KV + sqrt(KV.^2 + 4 * KpUVV .* KnUVV)) ./ max(2 * KpUVV,1e-10) );
            
            clear VV UVV KpUVV KnUVV KV;
            
            % ===================== update V ========================
            Bp = Kp * U;    % n^2 k
            Bn = Kn * U;    % n^2 k
            Ap = U' * Bp;   % k^2 n
            An = U' * Bn;   % k^2 n
            
            VTV = V' * V;       % k^2 n
            VVTV = V * VTV; % k^2 n
            VAp = V * Ap;     % k^2 n
            VAn = V * An;     % k^2 n
            
            %t1 = VAn + Bp + 2 * lambda * V;
            %t2 = max(VAp + Bn + 2 * lambda * VVTV, eps);
            t1 = Bp + 2 * lambda * V;
            t2 = max(Bn + 2 * lambda * VVTV, eps);          
            V = V .* (t1 ./ t2).^(1/4);
            % V = V * diag(sqrt(1./(diag(V'*V)+eps)));
            clear Bp Bn Ap An VTV VVTV VAp VAn t1 t2;
        end

        nIter = nIter + 1;
        if nIter > minIter
            if selectInit
                objhistory = CalculateObj_orth(K, U, V, lambda);
                maxErr = 0;
            else
                if isempty(maxIter)
                    newobj = CalculateObj_orth(K, U, V, lambda);
                    objhistory = [objhistory newobj]; %#ok<AGROW>
                    meanFit = meanFitRatio*meanFit + (1-meanFitRatio)*newobj;
                    maxErr = (meanFit-newobj)/meanFit;
                else
                    if isfield(options,'Converge') && options.Converge
                        newobj = CalculateObj_orth(K, U, V, lambda);
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

function obj = CalculateObj_orth(K, U, V, lambda)
obj = CalculateObj(K, U, V);
obj2 = V'*V - diag(ones(size(V,2),1));
obj2 = sum(sum(obj2.^2));
obj = obj + lambda * obj2;
end