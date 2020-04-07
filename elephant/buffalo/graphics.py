# -*- coding: utf-8 -*-
"""
This module implements objects that performs analyses that produce histograms.

These classes support a standardized flow for processing a data input, producing standardized outputs and provenance
information.

:copyright: Copyright 2014-2019 by the Elephant team, see `doc/authors.rst`.
:license: BSD, see LICENSE.txt for details.
"""
from .base import Analysis
import matplotlib.pyplot as plt


class BuffaloGraphic(Analysis):
    """
    Generic class to work with graphics within Buffalo.

    Parameters
    ----------
    params: dict
        Input parameters for the graphic. May contain the following:

            1. `title`: str
                Title of the graph

            2. `xlabel`: str
                Label for the X axis

            3. `ylabel`: str
                Label for the Y axis

            4. `axes`: list
                List of X and Y axes bounds.

            5. `name`: str
                Name of matplotlib Figure object.

    kwargs: dict
        Additional parameters for the matplotlib .figure() function

    Attributes
    ----------
    title
    xlabel
    ylabel
    figure_name
    axes
    figure

    Methods
    -------
    activate
    """

    _name = "BuffaloGraphic"
    _description = "Object that stores and manipulates graphic output"

    _figure = None

    _required_types = {'title': (str,),
                       'xlabel': (str,),
                       'ylabel': (str,),
                       'axes': (list,),
                       'name': (str,)}

    def __init__(self, *args, params={}, **kwargs):
        """
        Initializes a matplotlib figure object within the class.
        The Analysis Buffalo superclass is called to store the parameters used by the child class to work on the
        graphic.

        Parameters
        ----------
        args: list
            Input data

        params: dict
            Dictionary of input parameters

        kwargs: dict
            matplotlib .figure() kwargs, that will be forwarded
        """
        super().__init__(*args, params=params, **kwargs)
        name = self.figure_name
        if name is not None:
            self._figure = plt.figure(name, **kwargs)
        else:
            self._figure = plt.figure(**kwargs)

    def _process(self, *args, **kwargs):
        pass

    def _set_xlabel(self, value):
        if value is not None:
            plt.xlabel(value)

    def _set_ylabel(self, value):
        if value is not None:
            plt.ylabel(value)

    def _set_title(self, value):
        if value is not None:
            plt.title(value)

    def _set_axes(self, value):
        if value is not None:
            plt.axes(value)

    def set_text_and_axes(self):
        self.activate()
        self._set_xlabel(self.xlabel)
        self._set_ylabel(self.ylabel)
        self._set_title(self.title)
        self._set_axes(self.axes)

    def activate(self):
        """
        Set the figure in this object as current figure in matplotlib.
        """
        plt.figure(self.figure.number)

    @property
    def title(self):
        """
        Title of the graphic.

        Returns
        -------
        str, None
        """
        return self.get_input_parameter('title')

    @property
    def xlabel(self):
        """
        Label for the X axis.

        Returns
        -------
        str, None
        """
        return self.get_input_parameter('xlabel')

    @property
    def ylabel(self):
        """
        Label for the Y axis.

        Returns
        -------
        str, None
        """
        return self.get_input_parameter('ylabel')

    @property
    def figure_name(self):
        """
        Name of the Figure object.

        Returns
        -------
        str, None
        """
        return self.get_input_parameter('name')

    @property
    def axes(self):
        """
        Bounds of X and y axes.

        Returns
        -------
        list, None
        """
        return self.get_input_parameter('axes')

    @property
    def figure(self):
        """
        Matplotlib Figure object created and used by this object.

        Returns
        -------
        matplotlib.Figure
        """
        return self._figure
