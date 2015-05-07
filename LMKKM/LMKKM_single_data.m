function LMKKM_single_data(dataset, kernel_type, nRepeat)

dir1 = pwd;
addpath(fullfile(pwd, '..', 'lib'));
addpath(fullfile(pwd, '..', 'data'));
mosek_dir = 'C:\Program Files\Mosek\7\toolbox\r2013a';
if exist(mosek_dir, 'dir') == 7
	addpath(mosek_dir);
end
cd(fullfile(dir1, '..', 'data'));
BuildKernels(dataset, kernel_type);

cd(dir1);
LMKKM_multi_kernel(dataset, kernel_type, nRepeat);
if exist(mosek_dir, 'dir') == 7
	rmpath(mosek_dir);
end