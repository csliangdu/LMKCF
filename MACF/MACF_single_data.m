function MACF_single_data(dataset, kernel_type, nRepeat)

dir1 = pwd;
addpath(fullfile(pwd, '..', 'lib'));
addpath(fullfile(pwd, '..', 'data'));

cd(fullfile(dir1, '..', 'data'));
BuildKernels(dataset, kernel_type);

cd(dir1);
MACF_all_kernel(dataset, kernel_type, nRepeat);
MACF_equal_weight_multi_kernel(dataset, kernel_type, nRepeat);