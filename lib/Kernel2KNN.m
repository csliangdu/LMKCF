function W = Kernel2KNN(K, k)
nSmp = size(K,1);
% extract W from top-k of K
[~, idx] = sort(K, 2, 'descend');
if exist('k','var') && k > 0
    k = min(nSmp, k);
else
    k = 5;
end
idx5 = idx(:, 2:k+1);

W = zeros(nSmp);
rIdx = repmat((1:nSmp)', 1, k);
cIdx = idx5(:);
indx = sub2ind(size(W), rIdx(:), cIdx);
W(indx) = 1;
W = max(W, W');
