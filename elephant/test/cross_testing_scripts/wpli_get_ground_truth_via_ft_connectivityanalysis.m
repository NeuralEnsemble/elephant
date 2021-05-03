%% use FieldTrips' ft_connectivityanalysis() to calculate WPLI ground-truth
%% BEFORE using this script: change current folder to cross_testing_scripts
addpath '/path/to/FieldTrip_toolbox'

% ARTIFICIAL DATA
dataset1 = load('artificial_LFPs_1.mat');
dataset2 = load('artificial_LFPs_2.mat');

siz = size(dataset1.lfp_matrix);
ntrials = siz(1);
nsamples = siz(2);

% create a data-structure, which is compatible with FieldTrip
joined_data.label = {'sig_1', 'sig_2'};
joined_data.trial = cell(1, ntrials);
for t = 1:ntrials
    joined_data.trial(1, t) = {[dataset1.lfp_matrix(t, :); dataset2.lfp_matrix(t, :)]};
end
joined_data.time = cell(1, ntrials);
for t = 1:ntrials
    joined_data.time(1, t) = {dataset1.time};
end
joined_data.fsample = double(dataset1.sf);

% do frequency-analysis
cfg_freq            = [];
cfg_freq.output     = 'fourier';
% mtmfft
cfg_freq.method     = 'mtmfft';
cfg_freq.keeptrials = 'yes';
cfg_freq.tapsmofrq  = 0.4;
cfg_freq.taper      = 'dpss';
cfg_freq.polyremoval= -1;

freqfourier = ft_freqanalysis(cfg_freq, joined_data);

% do WPLI-calculation
cfg_wpli            = [];
cfg_wpli.method     = 'wpli';
ft_wpli      = ft_connectivityanalysis(cfg_wpli, freqfourier);

% take absolute value (optional)
wpli_2 = ft_wpli.wplispctrm(1, 2, 1:(end-1));
wpli_2 = reshape(wpli_2, [1, length(wpli_2)]);
% wpli_2 = abs(wpli_2);

writematrix(wpli_2, 'ground_truth_WPLI_from_ft_connectivityanalysis_with_artificial_LFPs_multitaped.csv');

exit;
