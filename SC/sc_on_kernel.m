function U = sc_on_kernel(K, nCluster)
    numEV = nCluster*1.5;
    
    D = diag(sum(K,1));
    inv_sqrt_D = sqrt(inv(abs(D)));
    L = inv_sqrt_D*K*inv_sqrt_D;
    L = (L+L')/2;
    opts.disp = 0;
    eigvec = eigs(L,ceil(numEV),'LA',opts);  
    U = eigvec(:,1:nCluster);
    feaNorm = max(1e-14,full(sum(U.^2,2)));
    U = bsxfun(@rdivide, U, feaNorm);