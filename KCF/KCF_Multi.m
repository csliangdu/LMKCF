function [U_final, V_final, nIter_final, objhistory_final] = KCF_Multi(K, k, options, U, V)
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


Norm = 2;
NormV = 1;

nSmp = size(K,1);


if isempty(U)
    U = abs(rand(nSmp,k));
    V = abs(rand(nSmp,k));
else
    nRepeat = 1;
end
[U,V] = NormalizeUV(K, U, V, NormV, Norm);

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
    while(maxErr > differror)
        if nnFlag
            % ===================== update V ========================
            KU = K*U;               % n^2k
            UKU = U'*KU;            % nk^2
            VUKU = V*UKU;           % nk^2
            
            V = V.*(KU./max(VUKU,1e-10));
            clear WV DV KU UKU VUKU;
            % ===================== update U ========================
            KV = K*V;               % n^2k
            VV = V'*V;              % nk^2
            KUVV = K*U*VV;          % n^2k
            
            U = U.*(KV./max(KUVV,1e-10));
            clear KV VV KUVV;
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
            % ===================== update U ========================
            VV = V'*V;               % nk^2
            UVV = U * VV;            % nk^2
            KpUVV = Kp*UVV;          % n^2k
            KnUVV = Kn*UVV;          % n^2k

            KV = K*V;                % n^2k
            
            U = U.*( (KV + sqrt(KV.^2 + 4 * KpUVV .* KnUVV)) ./ max(2 * KpUVV,1e-10) );
            
            clear VV UVV KpUVV KnUVV KV;          
        end
        
        nIter = nIter + 1;
        if nIter > minIter
            [U,V] = NormalizeUV(K, U, V, NormV, Norm);
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
            
            [U,V] = NormalizeUV(K, U, V, NormV, Norm);
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

Norm = 2;
NormV = 0;

[U_final,V_final] = NormalizeUV(K, U_final, V_final, NormV, Norm);

end


function obj_KCF = CalculateObj(K, U, V)
UK = U'*K;  % n^2k
UKU = UK*U; % nk^2
VUK = V*UK; % n^2k
VV = V'*V;  % nk^2
obj_KCF = sum(diag(K))-2*sum(diag(VUK))+sum(sum(UKU.*VV));
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
