# -*- coding: utf-8 -*-
"""
NeuroTools
==========

NeuroTools is not a neural simulator, but a collection of tools
to support all tasks associated with a neural simulation project which
are not handled by the simulation engine.

For more information see:
http://neuralensemble.org/NeuroTools


Available subpackages
---------------------

NeuroTools functionality is modularized as follows: 

signals    - provides core classes for manipulation of spike trains and analog signals. 
spike2     - offers an easy way for reading data from CED's Spike2 Son files. 
parameters - contains classes for managing large, hierarchical parameter sets. 
analysis   - cross-correlation, tuning curves, frequency spectrum, etc.
stgen      - various stochastic process generators relevant for Neuroscience 
             (OU, poisson, inhomogenous gamma, ...). 
utilities  - miscellaneous stuff, like SRB access.
io         - NeuroTools support for reading and writing of files in various formats. 
plotting   - routines for plotting and visualization.
datastore  - a consistent interface for persistent data storage (e.g. for caching intermediate results).
random     - a set of classes representing statistical distributions

Sub-package specific documentation is available by importing the
sub-package, and requesting help on it:

>>> import NeuroTools.signals
>>> help(NeuroTools.signals)
"""

__all__ = ['analysis', 'parameters', 'plotting', 'signals', 'stgen', 'io', 'datastore', 'utilities', 'spike2', 'random', 'optimize', 'tisean']
__version__ = "0.1.0 (Asynchronous Astrocyte)"
import warnings
import platform
from operator import __or__

#########################################################
## ALL DEPENDENCIES SHOULD BE GATHERED HERE FOR CLARITY
#########################################################

# The nice thing would be to gather every non standard
# dependency here, in order to centralize the warning
# messages and the check
dependencies = {'pylab' : {'website' : 'http://matplotlib.sourceforge.net/', 'is_present' : False, 'check':False},
                'matplotlib': {'website' : 'http://matplotlib.sourceforge.net/', 'is_present' : False, 'check':False},
                'tables': {'website' : 'http://www.pytables.org/moin' , 'is_present' : False, 'check':False},
                'psyco' : {'website' : 'http://psyco.sourceforge.net/', 'is_present' : False, 'check':False},
                'pygsl' : {'website' : 'http://pygsl.sourceforge.net/', 'is_present' : False, 'check':False},
                'PIL'   : {'website' : 'http://www.pythonware.com/products/pil/', 'is_present':False, 'check':False},
                'scipy' : {'website' : 'http://numpy.scipy.org/' , 'is_present' : False, 'check':False},
                'NeuroTools.facets.hdf5' : {'website' : None, 'is_present' : False, 'check':False},
                'srblib'  : {'website' : 'http://www.sdsc.edu/srb/index.php/Python', 'is_present' : False, 'check':False},
                'rpy'     : {'website' : 'http://rpy.sourceforge.net/', 'is_present' : False, 'check':False},
                'rpy2'     : {'website' : 'http://rpy.sourceforge.net/rpy2.html', 'is_present' : False, 'check':False},

                'django'  : {'website': 'http://www.djangoproject.com', 'is_present': False, 'check': False},
                'IPython' : {'website': 'http://ipython.scipy.org/', 'is_present': False, 'check': False},
                'interval': {'website': 'http://pypi.python.org/pypi/interval/1.0.0', 'is_present': False, 'check': False},
                'TableIO' : {'website': 'http://kochanski.org/gpk/misc/TableIO.html', 'is_present': False, 'check': False},
                ## Add here your extensions ###
               }


# Don't raise warnings for psyco on non-32bit systems
if platform.machine() != 'i386':
    dependencies['psyco']['check']=True

#########################################################
## Function to display error messages on the dependencies
#########################################################


class DependencyWarning(UserWarning):
    pass

def get_import_warning(name):
    return '''** %s ** package is not installed. 
To have functions using %s please install the package.
website : %s
''' %(name, name, dependencies[name]['website'])

def get_runtime_warning(name, errmsg):
    return '** %s ** package is installed but cannot be imported. The error message is: %s' %(name, errmsg)

def check_numpy_version():
    import numpy
    numpy_version = numpy.__version__.split(".")[0:2]
    numpy_version = float(".".join(numpy_version))
    if numpy_version >= 1.2:
        return True
    else:
        return False

def check_pytables_version():
   #v = [int(s) for s in __version__.split('.')]
   if tables.__version__<= 2: #1.4: #v[0] < 1 or (v[0] == 1 and v[1] < 4):
       raise Exception('PyTables version must be >= 1.4, installed version is %s' % __version__)

def check_dependency(name):
    if dependencies[name]['check']:
        return dependencies[name]['is_present']
    else:
        try:
            exec("import %s" %name)
            dependencies[name]['is_present'] = True
        except ImportError:
            warnings.warn(get_import_warning(name), DependencyWarning)
        except RuntimeError, errmsg:
            warnings.warn(get_runtime_warning(name, errmsg), DependencyWarning)
        dependencies[name]['check'] = True
        return dependencies[name]['is_present']



# Setup fancy logging

red     = 0010; green  = 0020; yellow = 0030; blue = 0040;
magenta = 0050; cyan   = 0060; bright = 0100
try:
    import ll.ansistyle
    def colour(col, text):
        try:
            return unicode(ll.ansistyle.Text(col, unicode(text)))
        except UnicodeDecodeError, e:
            raise UnicodeDecodeError("%s. text was %s" % (e, text))
except ImportError:
    def colour(col, text):
            return text
        
import logging

# Add a header() level to logging
logging.HEADER = (logging.WARNING + logging.ERROR)/2 # higher than warning, lower than error
logging.addLevelName(logging.HEADER, 'HEADER')

root = logging.getLogger()

def root_header(msg, *args, **kwargs):
    if len(root.handlers) == 0:
        basicConfig()
    apply(root.header, (msg,)+args, kwargs)

def logger_header(self, msg, *args, **kwargs):
    if self.manager.disable >= logging.HEADER:
        return
    if logging.HEADER >= self.getEffectiveLevel():
        apply(self._log, (logging.HEADER, msg, args), kwargs)

logging.Logger.header = logger_header
logging.header = root_header

class FancyFormatter(logging.Formatter):
    """
    A log formatter that colours and indents the log message depending on the level.
    """
    
    DEFAULT_COLOURS = {
        'CRITICAL': bright+red,
        'ERROR': red,
        'WARNING': magenta,
        'HEADER': bright+yellow,
        'INFO': cyan,
        'DEBUG': green
    }
    
    DEFAULT_INDENTS = {
        'CRITICAL': "",
        'ERROR': "",
        'WARNING': "",
        'HEADER': "",
        'INFO': "  ",
        'DEBUG': "    ",
    }
    
    def __init__(self, fmt=None, datefmt=None, colours=DEFAULT_COLOURS, mpi_rank=None):
        logging.Formatter.__init__(self, fmt, datefmt)
        self._colours = colours
        self._indents = FancyFormatter.DEFAULT_INDENTS
        if mpi_rank is None:
            self.prefix = ""
        else:
            self.prefix = "%-3d" % mpi_rank
    
    def format(self, record):
        s = logging.Formatter.format(self, record)
        if record.levelname == "HEADER":
            s = "=== %s ===" % s
        if self._colours:
            s = colour(self._colours[record.levelname], s)
        return self.prefix + self._indents[record.levelname] + s


class NameOrLevelFilter(logging.Filter):
    """
    Logging filter which allows messages that either have an approved name, or
    have a level >= the level specified.
    
    The intended use is when you want to receive most messages at a high level,
    but receive certain named messages at a lower level, e.g. for debugging a
    particular component.
    """
    def __init__(self, names=[], level=logging.INFO):
        self.names = names
        self.level = level
        
    def filter(self, record):
        if len(self.names) == 0:
            allow_by_name = True
        else:
            allow_by_name = record.name in self.names
        allow_by_level = record.levelno >= self.level
        return (allow_by_name or allow_by_level)


def init_logging(filename, file_level=logging.INFO, console_level=logging.WARNING, mpi_rank=None):
    if mpi_rank is None:
        mpi_fmt = ""
    else:
        mpi_fmt = "%3d " % mpi_rank
    logging.basicConfig(level=file_level,
                        format='%%(asctime)s %s%%(name)-10s %%(levelname)-6s %%(message)s [%%(pathname)s:%%(lineno)d]' % mpi_fmt,
                        filename=filename,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(FancyFormatter('%(message)s', mpi_rank=mpi_rank))
    logging.getLogger('').addHandler(console)
    return console
