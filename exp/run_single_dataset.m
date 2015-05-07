function run_single_dataset(dataset, myusername, mypassword)
addpath(fullfile(pwd, '..', 'lib'));
addpath(fullfile(pwd, '..', 'data'));
addpath(fullfile(pwd, '..', 'KernelKmeans'));
addpath(fullfile(pwd, '..', 'SC'));
addpath(fullfile(pwd, '..', 'KCF'));
addpath(fullfile(pwd, '..', 'LCCF'));
% addpath(fullfile(pwd, '..', 'LCF'));
addpath(fullfile(pwd, '..', 'MACF'));
addpath(fullfile(pwd, '..', 'AASC'));
addpath(fullfile(pwd, '..', 'Coreg'));
% addpath(fullfile(pwd, '..', 'MKKM'));
addpath(fullfile(pwd, '..', 'LMKKM'));
addpath(fullfile(pwd, '..', 'RMKKM'));
addpath(fullfile(pwd, '..', 'GMKCF'));
addpath(fullfile(pwd, '..', 'LMKCF'));

kernel_type = {'NCW-SCALE-SYM'};
nRepeat = 20;
diary on;
logfile = fullfile(pwd, ['log_exp_', dataset, '_', datestr(now,30)]);
diary(logfile);

KernelKmeans_single_data(dataset, kernel_type, nRepeat);
LMKKM_single_data(dataset, kernel_type, nRepeat);
RMKKM_single_data(dataset, kernel_type, nRepeat);

KCF_single_data(dataset, kernel_type, nRepeat);
LCCF_single_data(dataset, kernel_type, nRepeat);
% LCF_single_data(dataset, kernel_type, nRepeat);
MACF_single_data(dataset, kernel_type, nRepeat);
GMKCF_single_data(dataset, kernel_type, nRepeat);
LMKCF_single_data(dataset, kernel_type, nRepeat);

SC_single_data(dataset, kernel_type, nRepeat);
coreg_centroid_single_data(dataset, kernel_type, nRepeat);
AASC_single_data(dataset, kernel_type, nRepeat);

% send email 
if ~exist('myusername', 'var') || isempty(myusername)
    myusername = 'xxx';
end
if ~exist('mypassword', 'var') || isempty(mypassword)
    mypassword = 'xxx';
end
email_notify(myusername, mypassword,[], ['ALL on ', dataset, ' done!']);
diary off;

aggregate_baseline_tables(dataset, kernel_type);

% delete kernerls created on disk
data_dir = fullfile(pwd, '..', 'data');
kernel_dir = fullfile(data_dir, [dataset, '_kernel']);
if exist(kernel_dir, 'dir') == 7
    rmdir(kernel_dir, 's');
end

rmpath(fullfile(pwd, '..', 'lib'));
rmpath(fullfile(pwd, '..', 'data'));
rmpath(fullfile(pwd, '..', 'KernelKmeans'));
rmpath(fullfile(pwd, '..', 'SC'));
rmpath(fullfile(pwd, '..', 'KCF'));
rmpath(fullfile(pwd, '..', 'LCCF'));
% rmpath(fullfile(pwd, '..', 'LCF'));
rmpath(fullfile(pwd, '..', 'MACF'));
rmpath(fullfile(pwd, '..', 'AASC'));
% rmpath(fullfile(pwd, '..', 'MKKM'));
rmpath(fullfile(pwd, '..', 'Coreg'));
rmpath(fullfile(pwd, '..', 'LMKKM'));
rmpath(fullfile(pwd, '..', 'RMKKM'));
rmpath(fullfile(pwd, '..', 'GMKCF'));
rmpath(fullfile(pwd, '..', 'LMKCF'));
