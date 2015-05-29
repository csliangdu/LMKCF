function [U_final, nIter_final, objhistory_final] = KFCM(K, c, degree, options, U)
% Kernel Fuzzy c-means (KFCM)
%
% Input
%       K: (nSmp x nSmp) kernel matrix
%       c : number of clusters
%  degree : fuzzification degree
% options : structure holding all settings
%       U : initial membership matrix
%
% Output
% U_final : membership matrix
% nIter_final
% objhistory_final
%
%**************************************************
%     Author: Liang Du <csliangdu@gmail.com>
%     Version: 1.0
%     Last modified: 2015-05-23 11:45:27
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

nSmp = size(K,1);

if isempty(U)
    U = abs(rand(nSmp, c));
    U = bsxfun(@rdivide, U, max(sum(U,2), eps));
else
    nRepeat = 1;
end

selectInit = 1;
if nRepeat == 1
    selectInit = 0;
    minIterOrig = 0;
    minIter = 0;
    if isempty(maxIter)
        objhistory = CalculateObj(K, U, degree);
    else
        if isfield(options,'Converge') && options.Converge
            objhistory = CalculateObj(K, U, degree);
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
        % calculate normalized memberships
        Um = U .^ degree;
        V = Um * diag(1 ./ sum(Um) );
        
        % calculate distances D
        KV = K * V;
        dist = ones(nSmp, 1) * diag(V' * KV )' - 2 * KV + diag(K) * ones(1, c);
        
        % update memberships U
        U = dist .^ ( -1 / ( degree - 1) );
        U = U ./ ( sum(U, 2) * ones(1, c) );
        
        nIter = nIter + 1;
        if nIter > minIter
            if selectInit
                objhistory = CalculateObj(K, U, degree);
                maxErr = 0;
            else
                if isempty(maxIter)
                    newobj = CalculateObj(K, U, degree);
                    objhistory = [objhistory; newobj]; %#ok<AGROW>
                    maxErr = abs(objhistory(end) - objhistory(end-1)) / abs(objhistory(end-1));
                else
                    if isfield(options,'Converge') && options.Converge
                        newobj = CalculateObj(K, U, degree);
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
        nIter_final = nIter;
        objhistory_final = objhistory;
    else
        if objhistory(end) < objhistory_final(end)
            U_final = U;
            nIter_final = nIter;
            objhistory_final = objhistory;
        end
    end
    
    if selectInit
        if tryNo < nRepeat
            %re-start
            U = abs(rand(nSmp,c));
            U = bsxfun(@rdivide, U, max(sum(U,2), eps));
        else
            tryNo = tryNo - 1;
            minIter = 0;
            selectInit = 0;
            U = U_final;
            objhistory = objhistory_final;
        end
    end
end

nIter_final = nIter_final + minIterOrig;

end

function obj = CalculateObj(K, U, degree)
Um = U .^ degree;
d = sum(Um);
V = bsxfun(@rdivide, Um, max(d, eps));
KV = K * V;  % n^2 k
VKV = V' * KV; % k^2n

dist = ones(size(K, 1), 1) * diag(VKV)' - 2 * KV + diag(K) * ones(1, size(U, 2));

obj = sum(sum( dist .* Um ));
end