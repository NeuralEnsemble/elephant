"""
This module implements a class and decorator to inspect and extract
events associated with file input and output. Two callback objects can be
used for the monitor to send information regarding IO events to another
class or function.
"""

from functools import wraps
import inspect
import builtins


def savefig_callback(function):
    @wraps(function)
    def get_save(*args, **kwargs):
        callback = set()
        print("Figure.savefig:", args, kwargs)
        old_input, old_output = \
            monitor.set_callbacks(None, callback, save=True)
        result = function(*args, **kwargs)
        print("Figure.savefig IO:", callback)
        monitor.set_callbacks(old_input, old_output)
        del callback
        return result
    return get_save


def open_decorator(function):
    @wraps(function)
    def wrap_open(*args, **kwargs):
        mode = kwargs['mode'] if 'mode' in kwargs else args[1]
        filename = kwargs['file'] if 'file' in kwargs else args[0]
        result = function(*args, **kwargs)
        print(inspect.getsourcefile(inspect.currentframe().f_back))
        if 'r' in mode:
            monitor.input(filename)
        if any(x in mode for x in ['w', 'x', 'a']):
            monitor.output(filename)

        return result
    return wrap_open


class IOMonitor(object):
    """
    Class to monitor file IO and send the operations to the `Provenance`
    class decorator.
    """

    input_callback = None
    output_callback = None

    @classmethod
    def set_callbacks(cls, input_list=None, output_list=None, save=False):
        old_inputs, old_outputs = None, None
        if input_list is not None:
            if save:
                old_inputs = cls.input_callback
            cls.input_callback = input_list
        if output_list is not None:
            if save:
                old_outputs = cls.output_callback
            cls.output_callback = output_list
        return old_inputs, old_outputs

    @classmethod
    def input(cls, filename):
        if cls.input_callback is not None:
            cls.input_callback.add(filename)

    @classmethod
    def output(cls, filename):
        if cls.output_callback is not None:
            cls.output_callback.add(filename)


def activate():
    """
    Activates IO tracking.
    """
    builtins.open = open_decorator(builtins.open)


def deactivate():
    """
    Deactivates IO tracking.
    """
    builtins.open = inspect.unwrap(builtins.open)


monitor = IOMonitor()
