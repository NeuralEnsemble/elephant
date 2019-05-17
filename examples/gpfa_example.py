import numpy as np
import matplotlib.pyplot as plt
import neo
import quantities as pq

from elephant.gpfa.neural_trajectory import neural_trajectory
from elephant.gpfa.plotting import plot3D, plotEachDimVsTime


# =====
# TIPS
# =====
# - For exploratory analysis using GPFA, we often run only Section 1
#   below, and not Section 2 (which finds the optimal latent
#   dimensionality).  This can provide a substantial savings in running
#   time, since running Section 2 takes roughly K times as long as
#   Section 1, where K is the number of cross-validation folds.  As long
#   as we use a latent dimensionality that is 'large enough' in Section 1,
#   we can roughly estimate the latent dimensionality by looking at
#   the plot produced by plotEachDimVsTime.m.  The optimal latent
#   dimensionality is approximately the number of top dimensions that
#   have 'meaningful' temporal structure.  For visualization purposes,
#   this rough dimensionality estimate is usually sufficient.
#
# - For exploratory analysis with the two-stage methods, we MUST run
#   Section 2 to obtain the optimal smoothing kernel width.  There is
#   no easy way estimate the optimal smoothing kernel width from the
#   results of Section 1.

# ===========================================
# 1) Basic extraction of neural trajectories
# ===========================================
dat_mat = np.load("sample_data.npy", allow_pickle=True)

# convert the data to neo.SpikeTrain objects
dat = []
for tr in range(dat_mat['trialId'].size):
    sts = []
    for neu in range(dat_mat['spikes'][0].shape[0]):
        spike_times = np.where(dat_mat['spikes'][tr][neu] == 1)[0]*pq.ms
        sts.append(neo.SpikeTrain(spike_times, t_start=0, t_stop=400, units=1*pq.ms))
    dat.append((tr, sts))

# Select method to extract neural trajectories:
# 'gpfa' -- Gaussian-process factor analysis
# 'fa'   -- Smooth and factor analysis (not implemented yet)
# 'ppca' -- Smooth and probabilistic principal components analysis (not implemented yet)
# 'pca'  -- Smooth and principal components analysis (not implemented yet)
method = 'gpfa'

# Select the time step of neural trajectories
bin_size = 20*pq.ms

# Select number of latent dimensions
x_dim = 8
# NOTE: The optimal dimensionality should be found using
#       cross-validation (Section 2) below (not implemented yet).

# If using a two-stage method ('fa', 'ppca', or 'pca'), select
# standard deviation (in msec) of Gaussian smoothing kernel.
kernSD = 30.0
# NOTE: The optimal kernel width should be found using
#       cross-validation (Section 2) below (not implemented yet).

# Extract neural trajectories
params_est, seqs_train, seqs_test, fit_info = neural_trajectory(dat, method, bin_size, x_dim)

# Plot neural trajectories in 3D space
plot3D(seqs_train, 'xorth', [0, 1, 2])
# NOTES:
# - This figure shows the time-evolution of neural population
#   activity on a single-trial basis.  Each trajectory is extracted from
#   the activity of all units on a single trial.
# - This particular example is based on multi-electrode recordings
#   in premotor and motor cortices within a 400 ms period starting 300 ms
#   before movement onset.  The extracted trajectories appear to
#   follow the same general path, but there are clear trial-to-trial
#   differences that can be related to the physical arm movement.
# - Analogous to Figure 8 in Yu et al., J Neurophysiol, 2009.
# WARNING:
# - If the optimal dimensionality (as assessed by cross-validation in
#   Section 2) is greater than 3, then this plot may mask important
#   features of the neural trajectories in the dimensions not plotted.
#   This motivates looking at the next plot, which shows all latent
#   dimensions.

# Plot each dimension of neural trajectories versus time
plotEachDimVsTime(seqs_train, 'xorth', fit_info['bin_size'])
# NOTES:
# - These are the same neural trajectories as in the previous figure.
#   The advantage of this figure is that we can see all latent
#   dimensions (one per panel), not just three selected dimensions.
#   As with the previous figure, each trajectory is extracted from the
#   population activity on a single trial.  The activity of each unit
#   is some linear combination of each of the panels.  The panels are
#   ordered, starting with the dimension of greatest covariance
#   (in the case of 'gpfa' and 'fa') or variance (in the case of
#   'ppca' and 'pca').
# - From this figure, we can roughly estimate the optimal
#   dimensionality by counting the number of top dimensions that have
#   'meaningful' temporal structure.   In this example, the optimal
#   dimensionality appears to be about 5.  This can be assessed
#   quantitatively using cross-validation in Section 2.
# - Analogous to Figure 7 in Yu et al., J Neurophysiol, 2009.

plt.show()

# fprintf('\n');
# fprintf('Basic extraction and plotting of neural trajectories is complete.\n');
# fprintf('Press any key to start cross-validation...\n');
# fprintf('[Depending on the dataset, this can take many minutes to hours.]\n');
# pause;


# TODO: include cross-validation
"""
% ========================================================
% 2) Full cross-validation to find:
%  - optimal state dimensionality for all methods
        %  - optimal smoothing kernel width for two-stage methods
% ========================================================

% Select number of cross-validation folds
numFolds = 4;

% Perform cross-validation for different state dimensionalities.
% Results are saved in mat_results/runXXX/, where XXX is runIdx.
for xDim = [2 5 8]
neuralTraj(runIdx, dat, 'method',  'pca', 'xDim', xDim, 'numFolds', numFolds);
neuralTraj(runIdx, dat, 'method', 'ppca', 'xDim', xDim, 'numFolds', numFolds);
neuralTraj(runIdx, dat, 'method',   'fa', 'xDim', xDim, 'numFolds', numFolds);
neuralTraj(runIdx, dat, 'method', 'gpfa', 'xDim', xDim, 'numFolds', numFolds);
end
fprintf('\n');
% NOTES:
% - These function calls are computationally demanding.  Cross-validation
%   takes a long time because a separate model has to be fit for each
%   state dimensionality and each cross-validation fold.

% Plot prediction error versus state dimensionality.
% Results files are loaded from mat_results/runXXX/, where XXX is runIdx.
kernSD = 30; % select kernSD for two-stage methods
plotPredErrorVsDim(runIdx, kernSD);
% NOTES:
% - Using this figure, we i) compare the performance (i.e,,
%   predictive ability) of different methods for extracting neural
%   trajectories, and ii) find the optimal latent dimensionality for
%   each method.  The optimal dimensionality is that which gives the
%   lowest prediction error.  For the two-stage methods, the latent
%   dimensionality and smoothing kernel width must be jointly
%   optimized, which requires looking at the next figure.
% - In this particular example, the optimal dimensionality is 5. This
%   implies that, even though the raw data are evolving in a
%   53-dimensional space (i.e., there are 53 units), the system
%   appears to be using only 5 degrees of freedom due to firing rate
%   correlations across the neural population.
% - Analogous to Figure 5A in Yu et al., J Neurophysiol, 2009.

% Plot prediction error versus kernelSD.
% Results files are loaded from mat_results/runXXX/, where XXX is runIdx.
xDim = 5; % select state dimensionality
plotPredErrorVsKernSD(runIdx, xDim);
% NOTES:
% - This figure is used to find the optimal smoothing kernel for the
%   two-stage methods.  The same smoothing kernel is used for all units.
% - In this particular example, the optimal standard deviation of a
%   Gaussian smoothing kernel with FA is 30 ms.
% - Analogous to Figures 5B and 5C in Yu et al., J Neurophysiol, 2009.
"""
