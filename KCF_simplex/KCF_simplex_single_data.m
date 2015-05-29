function KCF_simplex_single_data(dataset, kernel_type, nRepeat)

dir1 = pwd;
addpath(fullfile(pwd, '..', 'lib'));
addpath(fullfile(pwd, '..', 'data'));

cd(fullfile(dir1, '..', 'data'));
BuildKernels(dataset, kernel_type);

cd(dir1);
KCF_simplex_all_kernel(dataset, kernel_type, nRepeat);
% KCF_equal_weight_multi_kernel(dataset, kernel_type, nRepeat);