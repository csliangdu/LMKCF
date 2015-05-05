function [res_lccf] = LCCF_single_kernel(K, y, prefix_file_name, nRepeat)

nSmp = size(K, 1);
nClass = length(unique(y));
assert(nSmp == length(y), 'The size of K and y should be the same!');

if ~exist('prefix_file_name', 'var') || isempty('prefix_file_name')
    prefix_file_name = 'some_kernel';
end

disp(['KenelKmeans ', num2str(nRepeat), 'iterations!']);
res_file = strcat(prefix_file_name, '_res_lccf_single_kernel.mat');
if exist(res_file, 'file')
    load(res_file, 'res_lccf');
else
    res_lccf = [];
    rng('default');
    for iRepeat = 1:nRepeat
        t_start = clock;
        disp(['LCCF ', num2str(iRepeat), ' of ' num2str(nRepeat), ' iterations begin ...']);
        [~, V] = LCCF_simple(K, nClass, 5, struct('maxIter', [], 'nRepeat', 1, 'alpha', 10), [], []);
        label_lccf = litekmeans(V, nClass, 'maxIter', 1000, 'Replicates', 10);
        res_lccf = [res_lccf; ClusteringMeasure(y, label_lccf)];%#ok<AGROW>
        t_end = clock;
        disp(['LCCF ', num2str(iRepeat), ' of ' num2str(nRepeat), ' iterations done.']);
        disp(['LCCF exe time: ', num2str(etime(t_end, t_start))]);
    end
    save(res_file, 'res_lccf');
end
save(res_file, 'res_lccf');
end
