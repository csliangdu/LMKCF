function Ustar = coreg_centroid_on_multi_kernel(Ks, nCluster, options)
if ~exist('options', 'var')
    options = [];
end
lambda = 1;
if isfield(options,'lambda')
    lambda = options.lambda;
end

maxiter = 30;
if isfield(options, 'maxiter')
    maxiter = options.maxiter;
end
[nSmp, ~, nKernel] = size(Ks);
if length(lambda) == 1
    lambda = ones(nKernel,1) * lambda;
end
opts.disp = 0;
L = zeros(nSmp, nSmp, nKernel);
U = zeros(nSmp, nCluster, nKernel);
for iter = 1:maxiter
    if iter == 1
        for iKernel = 1:nKernel
            D = diag(sum(Ks(:,:,iKernel),1));
            L(:,:,iKernel) = sqrt(inv(D))*Ks(:,:,iKernel)*sqrt(inv(D));
            L(:,:,iKernel) = (L(:,:,iKernel)+L(:,:,iKernel)')/2;
            [eigvec, eigval] = eig(L(:,:,iKernel));
            eigval = diag(eigval);
            [~, eig_idx] = sort(eigval, 'descend');
            U(:,:,iKernel) = eigvec(:, eig_idx(1:nCluster));
            % [U(:,:,iKernel), ~] = eigs(L(:,:,iKernel),nCluster,'LA',opts);
        end
    else
        L_ustar = Ustar*Ustar';
        L_ustar = (L_ustar+L_ustar')/2;
        for iKernel = 1:nKernel
            [eigvec, eigval] = eig(L(:,:,iKernel) + lambda(iKernel)*L_ustar);
            eigval = diag(eigval);
            [~, eig_idx] = sort(eigval, 'descend');
            U(:,:,iKernel) = eigvec(:, eig_idx(1:nCluster));
            % [U(:,:,iKernel), ~] = eigs(L(:,:,iKernel) + lambda(iKernel)*L_ustar, nCluster,'LA',opts);
        end
    end
    
    L_ustar(1:nSmp,1:nSmp) = 0;
    for iKernel = 1:nKernel
        L_ustar = L_ustar + lambda(iKernel)*U(:,:,iKernel)*U(:,:,iKernel)';
    end
    L_ustar = (L_ustar+L_ustar')/2;
    [eigvec, eigval] = eig(L_ustar);
    eigval = diag(eigval);
    [~, eig_idx] = sort(eigval, 'descend');
    Ustar= eigvec(:, eig_idx(1:nCluster));
    %     [Ustar, ~] = eigs(L_ustar, nCluster,'LA',opts);
end
Ustar_norm = max(eps, sqrt(sum(Ustar.^2, 2)));
Ustar = bsxfun(@rdivide, Ustar, Ustar_norm);
end