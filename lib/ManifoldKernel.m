function K = ManifoldKernel(K, reg, W, X, k, isBinary)

if ~exist('isBinary', 'var')
    isBinary = 1;
end

nSmp = size(K,1);

if exist('reg','var') && reg > 0
    if exist('W','var') && ~isempty(W)
        % use pre-specified W;
    elseif exist('X','var') && ~isempty(X)
        % use W from X and k
        Woptions = [];
        if exist('k','var') && k > 0
            Woptions.k = k;
        else
            Woptions.k = 5;
        end
        if nSmp > 3000
            tmpD = EuDist2(X(randsample(nSmp,3000),:));
        else
            tmpD = EuDist2(X);
        end
        Woptions.t = mean(mean(tmpD));
        W = constructW(X, Woptions);
    else
        % extract W from top-k of K
        [Kval, idx] = sort(K, 2, 'descend');
        if exist('k','var') && k > 0
            k = min(nSmp, k);
        else
            k = 5;
        end
        idx5 = idx(:, 2:k+1);
        Kval5 = Kval(:, 2:k+1);
        W = zeros(nSmp);
        rIdx = repmat((1:nSmp)', 1, k);
        cIdx = idx5(:);
        indx = sub2ind(size(W), rIdx(:), cIdx);
        W(indx) = Kval5(:);
        W = (W + W') / 2;
    end
    if isBinary
        W = real(W > 0);
    end
    D = full(sum(W,2));
    L = spdiags(D,0,nSmp,nSmp)-W;
    
    K=(speye(size(K,1))+reg*K*L)\K;
    K = max(K,K');
end
end