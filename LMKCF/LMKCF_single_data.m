function LMKCF_single_data(dataset, kernel_type, nRepeat)

dir1 = pwd;
addpath(fullfile(pwd, '..', 'lib'));
addpath(fullfile(pwd, '..', 'data'));
addpath('C:\Program Files\Mosek\7\toolbox\r2013a');
cd(fullfile(dir1, '..', 'data'));
BuildKernels(dataset, kernel_type);

cd(dir1);
LMKCF_multi_kernel(dataset, kernel_type, nRepeat);
rmpath('C:\Program Files\Mosek\7\toolbox\r2013a');