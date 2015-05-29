function result = run_mkfc_on_multiple_kernel(Ks, y, nRepeats, paramCell)
nP = length(paramCell);
nK = length(Ks);
nData = max(size(y));
K = zeros(nData, nData, nK);
for iKernel = 1:nK
    K(:, :, iKernel) = Ks{iKernel};
end
clear Ks;
nClusters = length(unique(y));
if ~exist('nRepeats', 'var')
	nRepeats = 1;
end
result = [];
result.acc_obj_v = zeros(nP, 1);
result.purity_obj_v = zeros(nP, 1);
result.nmi_obj_v = zeros(nP, 1);
result.acc_avg_v = zeros(nP, 1);
result.purity_avg_v = zeros(nP, 1);
result.nmi_avg_v = zeros(nP, 1);
result.acc_std_v = zeros(nP, 1);
result.purity_std_v = zeros(nP, 1);
result.nmi_std_v = zeros(nP, 1);
result.detail = struct('acc_v', zeros(nP, nRepeats), 'purity_v', zeros(nP, nRepeats), 'nmi_v', zeros(nP, nRepeats),...
	'label_v', zeros(nP, max(size(y)), nRepeats), 'obj_v', zeros(nP, nRepeats));

for p = 1 : nP
    str = sprintf( 'Trying paramter set %d out of %d', p, nP );
    disp( str );
    
    param = paramCell{p};
    
    acc_v = zeros(1, nRepeats);
    purity_v = zeros(1, nRepeats);
    nmi_v = zeros(1, nRepeats);
    obj_v = zeros(1, nRepeats);
    label_v = zeros(nData, nRepeats);
    
    rng('default');
    for iRepeat = 1:nRepeats
        disp(['MKFC ',  num2str(iRepeat), ' of ' num2str(nRepeats), ' iterations begin ...']);
		U = rand(nData, nClusters);
		[~, uidx] = max(U, [], 2);
		U = zeros(size(U));
		U(sub2ind(size(U), [1:size(U,1)]', uidx)) = 1;
		[label, kw, obj_final] = MKFC(U, nClusters, param.degree, 1e-5, K);
        res = ClusteringMeasure(y, label);
        acc_v(iRepeat) = res(1);
        purity_v(iRepeat) = res(2);
        nmi_v(iRepeat) = res(3);
        obj_v(iRepeat) = obj_final;
        label_v(:, iRepeat) = label;
    end
    result.detail.acc_v(p,:) = acc_v;
    result.detail.purity_v(p,:) = purity_v;
    result.detail.nmi_v(p,:) = nmi_v;
    result.detail.obj_v(p,:) = obj_v;
    result.detail.label_v(p,:,:) = label_v;
    
    [~, oidx] = min(obj_v);
    result.acc_obj_v( p ) = acc_v(oidx);
    result.nmi_obj_v( p ) = nmi_v(oidx);
    result.purity_obj_v( p) = purity_v(oidx);
    result.acc_avg_v( p ) = mean(acc_v);
    result.nmi_avg_v( p ) = mean(nmi_v);
    result.purity_avg_v( p) = mean(purity_v);
    result.acc_std_v( p ) = std(acc_v);
    result.nmi_std_v( p ) = std(nmi_v);
    result.purity_std_v( p) = std(purity_v);
end
end