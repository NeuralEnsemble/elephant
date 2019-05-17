import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot3D(seq, xspec, dimsToPlot=[0,1,2], nPlotMax=20, redTrials=[], **extraOpts):
    # function plot3D(seq, xspec, varargin)
    # %
    # % plot3D(seq, xspec, ...)
    # %
    # % Plot neural trajectories in a three-dimensional space.
    # %
    # % INPUTS:
    # %
    # % seq        - data structure containing extracted trajectories
    # % xspec      - field name of trajectories in 'seq' to be plotted
    # %              (e.g., 'xorth' or 'xsm')
    # %
    # % OPTIONAL ARGUMENTS:
    # %
    # % dimsToPlot - selects three dimensions in seq.(xspec) to plot
    # %              (default: 1:3)
    # % nPlotMax   - maximum number of trials to plot (default: 20)
    # % redTrials  - vector of trialIds whose trajectories are plotted in red
    # %              (default: [])
    # %
    # % @ 2009 Byron Yu -- byronyu@stanford.edu

    if seq[0][xspec].shape[0] < 3:
        print("ERROR: Trajectories have less than 3 dimensions.\n")
        return

    f = plt.figure()
    ax = f.gca(projection='3d', aspect=1)

    for n in range(min(len(seq), nPlotMax)):
        dat = seq[n][xspec][dimsToPlot, :]
        if seq[n]['trialId'] in redTrials:
            col = 'red'
            lw  = 3
        else:
            col = 'gray'
            lw = 0.5
        ax.plot(dat[0], dat[1], dat[2], '.-', linewidth=lw, color=col)

    if xspec == 'xorth':
        str1 = r"$\tilde{{\mathbf{{x}}}}_{{{},:}}$".format(dimsToPlot[0])
        str2 = r"$\tilde{{\mathbf{{x}}}}_{{{},:}}$".format(dimsToPlot[1])
        str3 = r"$\tilde{{\mathbf{{x}}}}_{{{},:}}$".format(dimsToPlot[2])
    else:
        str1 = r"${{\mathbf{{x}}}}_{{{},:}}$".format(dimsToPlot[0])
        str2 = r"${{\mathbf{{x}}}}_{{{},:}}$".format(dimsToPlot[1])
        str3 = r"${{\mathbf{{x}}}}_{{{},:}}$".format(dimsToPlot[2])
    ax.set_xlabel(str1, fontsize=24)
    ax.set_ylabel(str2, fontsize=24)
    ax.set_zlabel(str3, fontsize=24)

    plt.tight_layout()


def plotEachDimVsTime(seq, xspec, binWidth, nPlotMax=20, redTrials=[], nCols = 4, **extraOpts):
    # function plotEachDimVsTime(seq, xspec, binWidth, varargin)
    # %
    # % plotEachDimVsTime(seq, xspec, binWidth, ...)
    # %
    # % Plot each state dimension versus time in a separate panel.
    # %
    # % INPUTS:
    # %
    # % seq       - data structure containing extracted trajectories
    # % xspec     - field name of trajectories in 'seq' to be plotted
    # %             (e.g., 'xorth' or 'xsm')
    # % binWidth  - spike bin width used when fitting model
    # %
    # % OPTIONAL ARGUMENTS:
    # %
    # % nPlotMax  - maximum number of trials to plot (default: 20)
    # % redTrials - vector of trialIds whose trajectories are plotted in red
    # %             (default: [])
    # % nCols     - number of subplot columns (default: 4)
    # %
    # % @ 2009 Byron Yu -- byronyu@stanford.edu

    f = plt.figure()

    Xall = np.array([x for x in seq[xspec]])
    xMax = np.ceil(10 * np.abs(Xall).max()) / 10  # round max value to next highest 1e-1

    Tmax    = seq['T'].max()
    xtkStep = np.ceil(Tmax/25.)*5
    xtk = np.arange(1, Tmax+1, xtkStep)
    xtkl = np.arange(0, Tmax*binWidth, xtkStep*binWidth, dtype=np.int)
    ytk = [-xMax, 0, xMax]

    nRows = int(np.ceil(Xall.shape[1] / nCols))

    for n in range(min(len(seq), nPlotMax)):
        dat = seq[xspec][n]
        T = seq['T'][n]

        for k in range(dat.shape[0]):
            plt.subplot(nRows, nCols, k+1)

            if seq['trialId'][n] in redTrials:
                col = 'red'
                lw = 3
            else:
                col = 'gray'
                lw = 0.5

            plt.plot(np.arange(1, T+1), dat[k, :], linewidth=lw, color=col)

    for k in range(dat.shape[0]):
        plt.subplot(nRows, nCols, k+1)
        plt.axis([1, Tmax, 1.1*min(ytk), 1.1*max(ytk)])

        if xspec is 'xorth':
            # str = r'$\tilde{{\mathbf x}}_{{{},:}}$'.format(k)
            str = r'$\tilde{{\mathbf{{x}}}}_{{{},:}}$'.format(k)
        else:
            str = r"${{\mathbf{{x}}}}_{{{},:}}$".format(k)
        plt.title(str, fontsize=16)

        plt.xticks(xtk, xtkl)
        plt.yticks(ytk, ytk)
        plt.xlabel('Time (ms)')

    plt.tight_layout()

