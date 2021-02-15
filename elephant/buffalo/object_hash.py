"""
This module implements classes for hashing Python object and files, that is
used by the `Provenance` class decorator to track unique objects during the
script execution.
"""

import inspect
import joblib
import hashlib

from dill._dill import save_function, save_property

from collections import namedtuple
import numpy as np
from pathlib import Path

from copy import copy

# Need to use `dill` pickling function to support lambdas
# The dispatch table of the `joblib.Hasher` object is updated
joblib.hashing.Hasher.dispatch[type(save_function)] = save_function
# joblib.hashing.Hasher.dispatch[type(property)] = save_property


ObjectInfo = namedtuple('ObjectInfo', ('hash', 'type', 'id', 'details'))
FileInfo = namedtuple('FileInfo', ('hash', 'hash_type', 'path', 'details'))


class _HashMemoization(object):
    """
    Class for memoization of hashes during provenance tracking.
    It should not be used outside the `Provenance` decorator.
    """

    def __init__(self):
        self.items = dict()

    def memoize(self, obj, value):
        """
        Add an object hash to the memoization dictionary.

        Parameters
        ----------
        obj : object
            Python object whose hash is being memoized.
        value : int
            Python object hash of value and type.
        """
        self.items[id(obj)] = value

    def check(self, obj):
        """
        Check if a given object has a hash memoized. If a hash is present,
        it will be retrieved.

        Parameters
        ----------
        obj : object
            Python object whose hash is being checked.

        Returns
        -------
        str or None
            If no hash was memoized for `obj`, returns None.
            Otherwise, returns the hash value.

        """
        return self.items.get(id(obj))

    def clear(self):
        """
        Clears the memoization dictionary.
        """
        self.items.clear()


hash_memoizer = _HashMemoization()


class BuffaloFileHash(object):

    HASH_TYPES = {'md5': hashlib.md5,
                  'sha256': hashlib.sha256}

    def _get_file_hash(self, file_path, hash_type='sha256',
                       block_size=4096 * 1024):
        file_hash = self.HASH_TYPES[hash_type]()

        with open(file_path, 'rb') as file:
            for block in iter(lambda: file.read(block_size), b""):
                file_hash.update(block)

        return file_hash.hexdigest(), hash_type

    def __init__(self, file_path):
        self._file_path = file_path
        self._hash, self._hash_type = self._get_file_hash(file_path)
        self._details = {}

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if isinstance(other, BuffaloFileHash):
            return hash(self) == hash(other) and \
                   self._hash_type == other._hash_type
        else:
            raise TypeError("Cannot compare different objects")

    def __repr__(self):
        return "{}: [{}] {}".format(Path(self._file_path).name,
                                    self._hash_type, self._hash)

    def info(self):
        return FileInfo(self._hash, self._hash_type,
                        self._file_path, self._details)


class BuffaloObjectHash(object):
    """
    Class for hashing Python objects.

    The object hash value is obtaining by hashing a tuple consisting of the
    object type, and the MD5 hash of its value.

    The type is a string produced by the combination of the module where it
    was defined plus the type name (both returned by :func:`type`).
    Value hash is calculated using :func:`joblib.hash` for NumPy arrays and
    other container types, or the builtin :func:`hash` function for
    matplotlib objects.

    The method `info` is called to obtain the provenance information
    associated with the object during tracking.

    Parameters
    ----------
    obj : object
        A Python object that will be hashed with respect to type and content.
    """

    @staticmethod
    def _get_object_package(obj):
        # Returns the string with the name of the package where the object
        # is defined
        module = inspect.getmodule(obj)
        package = ""
        if module is not None:
            package = module.__package__.split(".")[0]
        return package

    def __init__(self, obj):
        self.package = self._get_object_package(obj)
        self.type = "{}.{}".format(type(obj).__module__, type(obj).__name__)
        self.id = id(obj)
        self.value = obj

    def __hash__(self):

        # If we already computed the hash for the object during this function
        # call, retrieve it from the memoized values
        print(self.type, self.id)
        memoized = hash_memoizer.check(self.value)
        if memoized is not None:
            return memoized

        print("Hashing")
        array_of_matplotlib = False
        if (isinstance(self.value, np.ndarray) and
                self.value.dtype == np.dtype('object')):
            if (len(self.value) and
                    self._get_object_package(self.value.flat[0]) ==
                    'matplotlib'):
                array_of_matplotlib = True

        if self.package in ['matplotlib']:
            # For matplotlib objects, we need to use the builtin hashing
            # function instead of joblib's. Multiple object hashes are
            # generated, since each time something is plotted the object
            # changes.
            value_hash = hash(self.value)
        elif array_of_matplotlib:  # or isinstance(self.value, list):
            # We also have to use an exception for NumPy arrays with Axes
            # objects, as those also change when the plot changes.
            # These are usually return by the `plt.subplots()` call.
            value_hash = id(self.value)
        else:
            # Other objects, like Neo, Quantity and NumPy arrays, use joblib
            value_hash = joblib.hash(self.value)

        # Compute final hash by type and value
        object_hash = hash((self.type, value_hash))

        # Memoize the hash
        hash_memoizer.memoize(self.value, object_hash)

        return object_hash

    def __repr__(self):
        return "{}: {}".format(self.id, self.type)

    def info(self):
        """
        Returns provenance information for the object.

        Returns
        -------
        ObjectInfo
            A named tuple with the following attributes:
            * hash : int
                Hash of the object.
            * type: str
                Type of the object.
            * id : int
                Reference for the object.
            * details : dict
                Extended information (metadata) on the object.
        """
        # Here we can extract specific metadata to record
        details = {}

        # Currently fetching the whole instance dictionary
        if hasattr(self.value, '__dict__'):
            # Need to copy otherwise the hashes change
            details = copy(self.value.__dict__)

        # Store specific attributes that are relevant for arrays, quantities
        # Neo objects, and AnalysisObjects
        for attr in ('units', 'shape', 'dtype', 't_start', 't_stop',
                     'id', 'nix_name', 'dimensionality', 'pid',
                     'create_time'):
            if hasattr(self.value, attr):
                details[attr] = getattr(self.value, attr)

        return ObjectInfo(hash(self), self.type, self.id, details)
