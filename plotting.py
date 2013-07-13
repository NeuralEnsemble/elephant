"""
NeuroTools.plotting
===================

This module contains a collection of tools for plotting and image processing that 
shall facilitate the generation and handling of NeuroTools data visualizations.
It utilizes the Matplotlib and the Python Imaging Library (PIL) packages.


Classes
-------

SimpleMultiplot     - object that creates and handles a figure consisting of multiple panels, all with the same datatype and the same x-range.


Functions
---------

get_display         - returns a pylab object with a plot() function to draw the plots.
progress_bar        - prints a progress bar to stdout, filled to the given ratio.
pylab_params        - returns a dictionary with a set of parameters that help to nicely format figures by updating the pylab run command parameters dictionary 'pylab.rcParams'.
set_axis_limits     - defines the axis limits in a plot.
set_labels          - defines the axis labels of a plot.
set_pylab_params    - updates a set of parameters within the the pylab run command parameters dictionary 'pylab.rcParams' in order to achieve nicely formatted figures.
save_2D_image       - saves a 2D numpy array of gray shades between 0 and 1 to a PNG file.
save_2D_movie       - saves a list of 2D numpy arrays of gray shades between 0 and 1 to a zipped tree of PNG files.
"""

import sys, numpy
from NeuroTools import check_dependency


# Check availability of pylab (essential!)
if check_dependency('pylab'):
    import pylab
if check_dependency('matplotlib'):
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Check availability of PIL
PILIMAGEUSE = check_dependency('PIL')
if PILIMAGEUSE:
    import PIL.Image as Image



########################################################
# UNIVERSAL FUNCTIONS AND CLASSES FOR NORMAL PYLAB USE #
########################################################



def get_display(display):
    """
    Returns a pylab object with a plot() function to draw the plots.
    
    Inputs:
        display - if True, a new figure is created. Otherwise, if display is a
                  subplot object, this object is returned.
    """
    if display is False:
        return None
    elif display is True:
        pylab.figure()
        return pylab
    else:
        return display



def progress_bar(progress):
    """
    Prints a progress bar to stdout.

    Inputs:
        progress - a float between 0. and 1.
        
    Example:
        >> progress_bar(0.7)
            |===================================               |
    """
    progressConditionStr = "ERROR: The argument of function NeuroTools.plotting.progress_bar(...) must be a float between 0. and 1.!"
    assert (type(progress) == float) and (progress >= 0.) and (progress <= 1.), progressConditionStr
    length = 50
    filled = int(round(length*progress))
    print "|" + "=" * filled + " " * (length-filled) + "|\r",
    sys.stdout.flush()



def pylab_params(fig_width_pt=246.0,
                ratio=(numpy.sqrt(5)-1.0)/2.0,# Aesthetic golden mean ratio by default
                text_fontsize=10, tick_labelsize=8, useTex=False):
    """
    Returns a dictionary with a set of parameters that help to nicely format figures.
    The return object can be used to update the pylab run command parameters dictionary 'pylab.rcParams'.

    Inputs:
        fig_width_pt   - figure width in points. If you want to use your figure inside LaTeX,
                         get this value from LaTeX using '\\showthe\\columnwidth'.
        ratio          - ratio between the height and the width of the figure.
        text_fontsize  - size of axes and in-pic text fonts.
        tick_labelsize - size of tick label font.
        useTex         - enables or disables the use of LaTeX for all labels and texts
                         (for details on how to do that, see http://www.scipy.org/Cookbook/Matplotlib/UsingTex).
    """
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*ratio            # height in inches
    fig_size =  [fig_width,fig_height]

    params = {
            'axes.labelsize'  : text_fontsize,
            'text.fontsize'   : text_fontsize,
            'xtick.labelsize' : tick_labelsize,
            'ytick.labelsize' : tick_labelsize,
            'text.usetex'     : useTex,
            'figure.figsize'  : fig_size}
            
    return params



def set_axis_limits(subplot, xmin, xmax, ymin, ymax):
    """
    Defines the axis limits of a plot.
    
    Inputs:
        subplot     - the targeted plot
        xmin, xmax  - the limits of the x axis
        ymin, ymax  - the limits of the y axis
        
    Example:
        >> x = range(10)
        >> y = []
        >> for i in x: y.append(i*i)
        >> pylab.plot(x,y)
        >> plotting.set_axis_limits(pylab, 0., 10., 0., 100.)
    """
    if hasattr(subplot, 'xlim'):
        subplot.xlim(xmin, xmax)
        subplot.ylim(ymin, ymax)
    elif hasattr(subplot, 'set_xlim'):
        subplot.set_xlim(xmin, xmax)
        subplot.set_ylim(ymin, ymax)
    else: 
        raise Exception('ERROR: The plot passed to function NeuroTools.plotting.set_axis_limits(...) does not provide limit defining functions.')



def set_labels(subplot, xlabel, ylabel):
    """
    Defines the axis labels of a plot.
    
    Inputs:
        subplot - the targeted plot
        xlabel  - a string for the x label
        ylabel  - a string for the y label
        
    Example:
        >> x = range(10)
        >> y = []
        >> for i in x: y.append(i*i)
        >> pylab.plot(x,y)
        >> plotting.set_labels(pylab, 'x', 'y=x^2')
    """
    if hasattr(subplot, 'xlabel'):
        subplot.xlabel(xlabel)
        subplot.ylabel(ylabel)
    elif hasattr(subplot, 'set_xlabel'):
        subplot.set_xlabel(xlabel)
        subplot.set_ylabel(ylabel)
    else: 
        raise Exception('ERROR: The plot passed to function NeuroTools.plotting.set_label(...) does not provide labelling functions.')



def set_pylab_params(fig_width_pt=246.0,
                    ratio=(numpy.sqrt(5)-1.0)/2.0,# Aesthetic golden mean ratio by default
                    text_fontsize=10, tick_labelsize=8, useTex=False):
    """
    Updates a set of parameters within the the pylab run command parameters dictionary 'pylab.rcParams' 
    in order to achieve nicely formatted figures.

    Inputs:
        fig_width_pt   - figure width in points. If you want to use your figure inside LaTeX,
                         get this value from LaTeX using '\showthe\columnwidth'
        ratio          - ratio between the height and the width of the figure
        text_fontsize  - size of axes and in-pic text fonts
        tick_labelsize - size of tick label font
        useTex         - enables or disables the use of LaTeX for all labels and texts
                         (for details on how to do that, see http://www.scipy.org/Cookbook/Matplotlib/UsingTex)
    """
    pylab.rcParams.update(pylab_params(fig_width_pt=fig_width_pt, ratio=ratio, text_fontsize=text_fontsize, \
        tick_labelsize=tick_labelsize, useTex=useTex))



####################################################################
# SPECIAL PLOTTING FUNCTIONS AND CLASSES FOR SPECIFIC REQUIREMENTS #
####################################################################



def save_2D_image(mat, filename):
    """
    Saves a 2D numpy array of gray shades between 0 and 1 to a PNG file.

    Inputs:
        mat      - a 2D numpy array of floats between 0 and 1
        filename - string specifying the filename where to save the data, has to end on '.png'
    
    Example:
        >> import numpy
        >> a = numpy.random.random([100,100]) # creates a 2D numpy array with random values between 0. and 1.
        >> save_2D_image(a,'randomarray100x100.png')
    """
    assert PILIMAGEUSE, "ERROR: Since PIL has not been detected, the function NeuroTools.plotting.save_2D_image(...) is not supported!"
    matConditionStr = "ERROR: First argument of function NeuroTools.plotting.imsave(...) must be a 2D numpy array of floats between 0. and 1.!"
    filenameConditionStr = "ERROR: Second argument of function NeuroTools.plotting.imsave(...) must be a string ending on \".png\"!"
    assert (type(mat) == numpy.ndarray) and (mat.ndim == 2) and (mat.min() >= 0.) and (mat.max() <= 1.), matConditionStr
    assert (type(filename) == str) and (len(filename) > 4) and (filename[-4:].lower() == '.png'), filenameConditionStr
    mode = 'L'
    # PIL asks for a permuted (col,line) shape coresponding to the natural (x,y) space
    pilImage = Image.new(mode, (mat.shape[1], mat.shape[0]))
    data = numpy.floor(numpy.ravel(mat) * 256.)
    pilImage.putdata(data)
    pilImage.save(filename)



def save_2D_movie(frame_list, filename, frame_duration):
    """
    Saves a list of 2D numpy arrays of gray shades between 0 and 1 to a zipped tree of PNG files.
    
    Inputs:
        frame_list     - a list of 2D numpy arrays of floats between 0 and 1
        filename       - string specifying the filename where to save the data, has to end on '.zip'
        frame_duration - specifier for the duration per frame, will be stored as additional meta-data
        
    Example:
        >> import numpy
        >> framelist = []
        >> for i in range(100): framelist.append(numpy.random.random([100,100])) # creates a list of 2D numpy arrays with random values between 0. and 1.
        >> save_2D_movie(framelist, 'randommovie100x100x100.zip', 0.1)
    """
    try:
        import zipfile
    except ImportError:
        raise ImportError("ERROR: Python module zipfile not found! Needed by NeuroTools.plotting.save_2D_movie(...)!")
    try:
        import StringIO
    except ImportError:
        raise ImportError("ERROR: Python module StringIO not found! Needed by NeuroTools.plotting.save_2D_movie(...)!")
    assert PILIMAGEUSE, "ERROR: Since PIL has not been detected, the function NeuroTools.plotting.save_2D_movie(...) is not supported!"
    filenameConditionStr = "ERROR: Second argument of function NeuroTools.plotting.save_2D_movie(...) must be a string ending on \".zip\"!"
    assert (type(filename) == str) and (len(filename) > 4) and (filename[-4:].lower() == '.zip'), filenameConditionStr
    zf = zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED)
    container = filename[:-4] # remove .zip
    frame_name_format = "frame%s.%dd.png" % ("%", pylab.ceil(pylab.log10(len(frame_list))))
    for frame_num, frame in enumerate(frame_list):
        frame_data = [(p,p,p) for p in frame.flat]
        im = Image.new('RGB', frame.shape, 'white')
        im.putdata(frame_data)
        io = StringIO.StringIO()
        im.save(io, format='png')
        pngname = frame_name_format % frame_num
        arcname = "%s/%s" % (container, pngname)
        io.seek(0)
        zf.writestr(arcname, io.read())
        progress_bar(float(frame_num)/len(frame_list))

    # add 'parameters' and 'frames' files to the zip archive
    zf.writestr("%s/parameters" % container,
                'frame_duration = %s' % frame_duration)
    zf.writestr("%s/frames" % container,
                '\n'.join(["frame%.3d.png" % i for i in range(len(frame_list))]))
    zf.close()



class SimpleMultiplot(object):
    """
    A figure consisting of multiple panels, all with the same datatype and
    the same x-range.
    """
    def __init__(self, nrows, ncolumns, title="", xlabel=None, ylabel=None,
                 scaling=('linear','linear')):
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.axes = []
        self.all_panels = self.axes
        self.nrows = nrows
        self.ncolumns = ncolumns
        self.n = nrows*ncolumns
        self._curr_panel = 0
        self.title = title
        topmargin = 0.06
        rightmargin = 0.02
        bottommargin = 0.1
        leftmargin=0.1
        v_panelsep = 0.1*(1 - topmargin - bottommargin)/nrows #0.05
        h_panelsep = 0.1*(1 - leftmargin - rightmargin)/ncolumns
        panelheight = (1 - topmargin - bottommargin - (nrows-1)*v_panelsep)/nrows
        panelwidth = (1 - leftmargin - rightmargin - (ncolumns-1)*h_panelsep)/ncolumns
        assert panelheight > 0
        
        bottomlist = [bottommargin + i*v_panelsep + i*panelheight for i in range(nrows)]
        leftlist = [leftmargin + j*h_panelsep + j*panelwidth for j in range(ncolumns)]
        bottomlist.reverse()
        for j in range(ncolumns):
            for i in range(nrows):
                ax = self.fig.add_axes([leftlist[j],bottomlist[i],panelwidth,panelheight])
                self.set_frame(ax,[True,True,False,False])
                ax.xaxis.tick_bottom()
                ax.yaxis.tick_left()
                self.axes.append(ax)
        if xlabel:
            self.axes[self.nrows-1].set_xlabel(xlabel)
        if ylabel:
            self.fig.text(0.5*leftmargin,0.5,ylabel,
                          rotation='vertical',
                          horizontalalignment='center',
                          verticalalignment='center')
        if scaling == ("linear","linear"):
            self.plot_function = "plot"
        elif scaling == ("log", "log"):
            self.plot_function = "loglog"
        elif scaling == ("log", "linear"):
            self.plot_function = "semilogx"
        elif scaling == ("linear", "log"):
            self.plot_function = "semilogy"
        else:
            raise Exception("Invalid value for scaling parameter")

    def finalise(self):
        """Adjustments to be made after all panels have been plotted."""
        # Turn off tick labels for all x-axes except the bottom one
        self.fig.text(0.5, 0.99, self.title, horizontalalignment='center',
                      verticalalignment='top')
        for ax in self.axes[0:self.nrows-1]+self.axes[self.nrows:]:
            ax.xaxis.set_ticklabels([])

    def save(self, filename):
        """Saves/prints the figure to file.
        
        Inputs:
            filename - string specifying the filename where to save the data
        """
        self.finalise()
        self.canvas.print_figure(filename)

    def next_panel(self):
        """Changes to next panel within figure."""
        ax = self.axes[self._curr_panel]
        self._curr_panel += 1
        if self._curr_panel >= self.n:
            self._curr_panel = 0
        ax.plot1 = getattr(ax, self.plot_function)
        return ax

    def panel(self, i):
        """Returns panel i."""
        ax = self.axes[i]
        ax.plot1 = getattr(ax, self.plot_function)
        return ax

    def set_frame(self, ax, boollist, linewidth=2):
        """
        Defines frames for the chosen axis.

        Inputs:
            as        - the targeted axis
            boollist  - a list 
            linewidth - the limits of the y axis
        """
        assert type(boollist) in [list, numpy.ndarray]
        assert len(boollist) == 4
        if boollist != [True,True,True,True]:
            bottom = Line2D([0, 1], [0, 0], transform=ax.transAxes, linewidth=linewidth, color='k')
            left   = Line2D([0, 0], [0, 1], transform=ax.transAxes, linewidth=linewidth, color='k')
            top    = Line2D([0, 1], [1, 1], transform=ax.transAxes, linewidth=linewidth, color='k')
            right  = Line2D([1, 0], [1, 1], transform=ax.transAxes, linewidth=linewidth, color='k')
            ax.set_frame_on(False)
            for side,draw in zip([left,bottom,right,top],boollist):
                if draw:
                    ax.add_line(side)
