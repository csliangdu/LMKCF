function [res_sc, obj_final] = SC_single_affinity(K, y, prefix_file_name, nRepeat)

nSmp = size(K, 1);
nClass = length(unique(y));
assert(nSmp == length(y), 'The size of K and y should be the same!');

if ~exist('prefix_file_name', 'var') || isempty('prefix_file_name')
    prefix_file_name = 'some_affinity';
end

res_sc = [];
obj_final = [];

disp(['SC on ', prefix_file_name, ' ', num2str(nRepeat), 'iterations!']);

res_file = strcat(prefix_file_name, '_res_sc_single_affinity.mat');

if exist(res_file, 'file')
    load(res_file, 'res_sc', 'obj_final');
else
	rng('default');
    try
        V = sc_on_kernel(K, nClass);
    catch
        V = SC(K, nClass);
        V = NormalizeFea(V, 1);
    end
	for iRepeat = 1:nRepeat
		t_start = clock;
		disp(['SC ',  num2str(iRepeat), ' of ' num2str(nRepeat), ' iterations begin ...']);
		label_sc = litekmeans(V, nClass, 'maxIter', 100, 'Replicates', 20);
		res_sc = [res_sc; ClusteringMeasure(y, label_sc)];%#ok<AGROW>
		obj_final = [obj_final; 0];%#ok<AGROW>
		t_end = clock;
		disp(['SC ',  num2str(iRepeat), ' of ' num2str(nRepeat), ' iterations done.']);
		disp(['SC exe time: ', num2str(etime(t_end, t_start))]);
	end
    save(res_file, 'res_sc', 'obj_final');
end