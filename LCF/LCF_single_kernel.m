function [res_lcf] = LCF_single_kernel(K, y, prefix_file_name, nRepeat)

nSmp = size(K, 1);
nClass = length(unique(y));
assert(nSmp == length(y), 'The size of K and y should be the same!');

if ~exist('prefix_file_name', 'var') || isempty('prefix_file_name')
    prefix_file_name = 'some_kernel';
end

disp(['KenelKmeans ', num2str(nRepeat), 'iterations!']);
res_file = strcat(prefix_file_name, '_res_lcf_single_kernel.mat');
if exist(res_file, 'file')
    load(res_file, 'res_lcf');
else
    res_lcf = [];
    rng('default');
    for iRepeat = 1:nRepeat
        t_start = clock;
        disp(['LCF ', num2str(iRepeat), ' of ' num2str(nRepeat), ' iterations begin ...']);
        [~, V] = LCF_Multi(K, nClass, struct('maxIter', 100, 'nRepeat', 1, 'lambda', 0.3), [], []);
        label_lcf = litekmeans(V, nClass, 'maxIter', 1000, 'Replicates', 10);
        res_lcf = [res_lcf; ClusteringMeasure(y, label_lcf)];%#ok<AGROW>
        t_end = clock;
        disp(['LCF ', num2str(iRepeat), ' of ' num2str(nRepeat), ' iterations done.']);
        disp(['LCF exe time: ', num2str(etime(t_end, t_start))]);
    end
    save(res_file, 'res_lcf');
end
save(res_file, 'res_lcf');
end
