function res_kcf_orth_v_nn = KCF_orth_single_kernel(K, y, prefix_file_name, nRepeat)

nSmp = size(K, 1);
nClass = length(unique(y));
assert(nSmp == length(y), 'The size of K and y should be the same!');

if ~exist('prefix_file_name', 'var') || isempty('prefix_file_name')
    prefix_file_name = 'some_kernel';
end

disp(['KenelKmeans ', num2str(nRepeat), 'iterations!']);
res_file = strcat(prefix_file_name, '_res_kcf_orth_v_nn_single_kernel.mat');
if exist(res_file, 'file')
    load(res_file, 'res_kcf_orth_v_nn');
else
    res_kcf_orth_v_nn = [];
    rng('default')
    for iRepeat = 1:nRepeat
        t_start = clock;
        disp(['KCF orth ', num2str(iRepeat), ' of ' num2str(nRepeat), ' iterations begin ...']);
        [~, V, nIter, objhistory] = KCF_orth_Multi(K, nClass, struct('maxIter', [], 'nRepeat', 1), [], []);
        label_kcf = litekmeans(V, nClass, 'maxIter', 1000, 'Replicates', 20);
        [~, label_kcf_max] = max(V, [], 2);
        res_kcf_orth_v_nn = [res_kcf_orth_v_nn; [ClusteringMeasure(y, label_kcf), ClusteringMeasure(y, label_kcf_max)]];%#ok<AGROW> 
        t_end = clock;
        disp(['KCF orth ', num2str(iRepeat), ' of ' num2str(nRepeat), ' iterations done.']);
        disp(['KCF orth exe time: ', num2str(etime(t_end, t_start))]);
    end
    save(res_file, 'res_kcf_orth_v_nn');
end
save(res_file, 'res_kcf_orth_v_nn');


end