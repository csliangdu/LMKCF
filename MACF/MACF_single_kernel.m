function [res_macf] = MACF_single_kernel(K, y, prefix_file_name, nRepeat)

nSmp = size(K, 1);
nClass = length(unique(y));
assert(nSmp == length(y), 'The size of K and y should be the same!');

if ~exist('prefix_file_name', 'var') || isempty('prefix_file_name')
    prefix_file_name = 'some_kernel';
end

disp(['KenelKmeans ', num2str(nRepeat), 'iterations!']);
res_file = strcat(prefix_file_name, '_res_macf_single_kernel.mat');
if exist(res_file, 'file')
    load(res_file, 'res_macf');
else
    res_macf = [];
    rng('default');
    for iRepeat = 1:nRepeat
        t_start = clock;
        disp(['MACF ', num2str(iRepeat), ' of ' num2str(nRepeat), ' iterations begin ...']);
        [~, V] = MACF(K, nClass, struct('reg', 1, 'W', [], 'X', [], 'k', 5, 'isBinary', 1), struct('maxIter', [], 'nRepeat', 1), [], []);
        label_macf = litekmeans(V, nClass, 'maxIter', 1000, 'Replicates', 20);
        res_macf = [res_macf; ClusteringMeasure(y, label_macf)];%#ok<AGROW>
        t_end = clock;
        disp(['MACF ', num2str(iRepeat), ' of ' num2str(nRepeat), ' iterations done.']);
        disp(['MACF exe time: ', num2str(etime(t_end, t_start))]);
    end
    save(res_file, 'res_macf');
end
save(res_file, 'res_macf');
end
