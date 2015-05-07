function [ outG0, outFCell, outAlpha, outAlpha_r, outObj, outNumIter ] = weighted_robust_multi_kmeans( inXCell, inPara, inG0 )
% solve the following problem
% min_{F^(v), G0, alpha^{v}} sum_v {(alpha^{v})^r*||X^(v) - F^(v)G0^T)^T||_2,1}
% s.t. G0 is a cluster indicator, sum_v{alpha^{v}) = 1, alpha^{v} >= 0
% 
% input: 
%       inXcell: v by 1 cell, and the size of each cell is d_v by n
%       inPara: parameter cell
%               inPara.maxIter: max number of iterator
%               inPara.thresh:  the convergence threshold
%               inPara.numCluster: the number cluster
%               inPara.r: the parameter to control the distribution of the
%                         weights for each view
%       inG0: init common cluster indicator
% output:
%       outG0: the output cluster indicator (n by c)
%       outFcell: the cluster centroid for each view (d by c by v)
%       outObj: obj value
%       outNumIter: number of iterator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ref:
% Xiao Cai, Feiping Nie, Heng Huang. 
% Multi-View K-Means Clustering on Big Data. 
% The 23rd International Joint Conference on Artificial Intelligence (IJCAI), 2013.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% parameter settings
maxIter = inPara.maxIter;
thresh = inPara.thresh;
c = inPara.numCluster;
r = inPara.r;
n = size(inXCell{1}, 2);
numView = length(inXCell);
% inti alpha
alpha = ones(numView, 1)/numView; 
% inti common indicator D3
G0 = inG0;
for v = 1: numView    
    D4{v} = sparse(diag(ones(n, 1))* alpha(v)^r);
end
% % Fix D3{v}, G0, alpha, update F{v}
% for v = 1: numView
%     M = G0'*D4{v}*G0;
%     N = inXCell{v}*D4{v}*G0;
%     F{v} = N/M;
% end
% clear M N;
tmp = 1/(1-r);
obj = zeros(maxIter, 1);
% loop
for t = 1: maxIter
    fprintf('processing iteration %d...\n', t);
    
    % Fix D3{v}, G0, alpha, update F{v}
    for v = 1: numView
        M = (G0'*D4{v}*G0);
        N = inXCell{v}*D4{v}*G0;
        F{v} = N/M;
    end
        
    % Fix D3{v}, F{v}, update G0    
    for i = 1:n
        dVec = zeros(numView, 1);
        for v = 1: numView
            xVec{v} = inXCell{v}(:,i);
            tt = diag(D4{v});
            dVec(v, 1) = tt(i);
        end
        G0(i,:) = searchBestIndicator(dVec, xVec, F);
    end
    
    % Fix F{v}, G0, D4{v}, update alpha
    h = zeros(numView, 1);
    for v = 1: numView
        E{v} = (inXCell{v} - F{v}*G0')';
        Ei2{v} = sqrt(sum(E{v}.*E{v}, 2));
        h(v) = sum(Ei2{v});
    end
    alpha = ((r*h).^tmp)/(sum(((r*h).^tmp)));
    
    % Fix F{v}, G0, update D4{v}
    for v = 1: numView
        E{v} = (inXCell{v} - F{v}*G0')';
        Ei2{v} = sqrt(sum(E{v}.*E{v}, 2) + eps);                
        D4{v} = sparse(diag(0.5./Ei2{v}*(alpha(v)^r)));
    end
    
    % calculate the obj
    obj(t) = 0;
    for v = 1: numView
        obj(t) = obj(t) + (alpha(v)^r)*sum(Ei2{v});
    end
    if(t > 1)
        diff = obj(t-1) - obj(t);
        if(diff < thresh)
            break;
        end
    end
end
% debug
% figure, plot(1: length(obj), obj);

outObj = obj;
outNumIter = t;
outFCell = F;
outG0 = G0;
outAlpha = alpha;
outAlpha_r = alpha.^r;

end
%% function searchBestIndicator
function outVec = searchBestIndicator(dVec, xCell, F)
% solve the following problem,
numView = length(F);
c = size(F{1}, 2);
tmp = eye(c);
obj = zeros(c, 1);
for j = 1: c
    for v = 1: numView
        obj(j,1) = obj(j,1) + dVec(v) * (norm(xCell{v} - F{v}(:,j))^2);
    end
end
[min_val, min_idx] = min(obj);
outVec = tmp(:, min_idx);
end

