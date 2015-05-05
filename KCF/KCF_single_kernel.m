function [res_kcf] = KCF_single_kernel(K, y, prefix_file_name, nRepeat)

nSmp = size(K, 1);
nClass = length(unique(y));
assert(nSmp == length(y), 'The size of K and y should be the same!');

if ~exist('prefix_file_name', 'var') || isempty('prefix_file_name')
    prefix_file_name = 'some_kernel';
end

disp(['KenelKmeans ', num2str(nRepeat), 'iterations!']);
res_file = strcat(prefix_file_name, '_res_kcf_single_kernel.mat');
if exist(res_file, 'file')
    load(res_file, 'res_kcf');
else
    res_kcf = [];
    rng('default');
    for iRepeat = 1:nRepeat
        t_start = clock;
        disp(['KCF ', num2str(iRepeat), ' of ' num2str(nRepeat), ' iterations begin ...']);
        [~, V, nIter, objhistory] = KCF_Multi(K, nClass, struct('maxIter', [], 'nRepeat', 1), [], []);
        label_kcf = litekmeans(V, nClass, 'maxIter', 1000, 'Replicates', 20);
        res_kcf = [res_kcf; ClusteringMeasure(y, label_kcf)];%#ok<AGROW>
        t_end = clock;
        disp(['KCF ', num2str(iRepeat), ' of ' num2str(nRepeat), ' iterations done.']);
        disp(['KCF exe time: ', num2str(etime(t_end, t_start))]);
    end
    save(res_file, 'res_kcf');
end
save(res_file, 'res_kcf');
end
