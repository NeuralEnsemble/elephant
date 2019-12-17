# -*- coding: utf-8 -*-
"""
This module implements base super classes from which all Buffalo objects are derived.

These classes support objects to produce standardized outputs and provenance capture during the analysis workflow in
Elephant.

:copyright: Copyright 2014-2019 by the Elephant team, see `doc/authors.rst`.
:license: BSD, see LICENSE.txt for details.
"""

from __future__ import print_function
from datetime import datetime
from copy import deepcopy


class AnalysisObject(object):
    """
    This is the superclass for the Buffalo analysis objects.
    """

    _timestamp = None           # UTC time of instance creation
    _annotations = None        # Dictionary with custom annotations

    def __init__(self):
        self._timestamp = datetime.utcnow()

    @property
    def annotations(self):
        return self._annotations

    def set_annotations(self, annotation):
        if not isinstance(annotation, dict):
            raise TypeError("Annotations must be a dictionary")
        self._annotations = deepcopy(annotation)
