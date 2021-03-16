%% use FieldTrips' ft_connectivity_wpli() to calculate WPLI ground-truth
%% BEFORE using this script: change current folder to cross_testing_scripts

% ARTIFICIAL DATA more complex
dataset1 = load('artificial_LFPs_1.mat');
dataset2 = load('artificial_LFPs_2.mat');

lfp1 = dataset1.lfp_matrix;
lfp2 = dataset2.lfp_matrix;
Fs = double(dataset1.sf);

siz = size(lfp1)
trial = siz(1)
tlength = siz(2)

fft1 = realfft(lfp1, tlength, 2);
fft2 = realfft(lfp2, tlength, 2);
freq = 0:double(Fs/tlength):(Fs/2); %-0.1 to stay lower Fs/2

cs = fft1 .* conj(fft2);
length_fft = floor(tlength/2) +1
cs = reshape(cs, [trial, 1, length_fft]);

[wpli_1, v, n] = ft_connectivity_wpli(cs, 'feedback', 'text', 'dojack', 0, 'debias', 0);
% wpli_1 = abs(wpli_1); %(optional)

writematrix(wpli_1, 'ground_truth_WPLI_from_ft_connectivity_wpli_with_artificial_LFPs.csv');


figure(1);
plot(freq, wpli_1);

function realfft = realfft(a, tlength, dim)
     fft_all = fft(a, tlength, dim);
     realfft = fft_all(:, 1:(floor(length(fft_all)/2)+1));
end