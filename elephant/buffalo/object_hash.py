"""
This module implements classes for hashing Python object and files, that is
used by the `Provenance` class decorator to track unique objects during the
script execution.
"""

import hashlib
import inspect
import uuid
from collections import namedtuple
from copy import copy
from pathlib import Path

import joblib
import numpy as np
from dill._dill import save_function

# Need to use `dill` pickling function to support lambdas.
# Some objects may have attributes that are lambdas. One example is the
# test case of Nose. When running Elephant unit tests that access variables
# in the class (e.g., `self.spiketrains`), the hashing of the `self` object
# fails.
# Here we update the dispatch table of the `joblib.Hasher` object to use
# the function from `dill` that supports these attributes.
joblib.hashing.Hasher.dispatch[type(save_function)] = save_function


ObjectInfo = namedtuple('ObjectInfo', ('hash', 'type', 'id', 'details'))
FileInfo = namedtuple('FileInfo', ('hash', 'hash_type', 'path'))


class BuffaloFileHash(object):
    """
    Class for hashing files.

    The SHA256 hash and file path are captured.

    The method `info` is called to obtain these provenance information as the
    `FileInfo` named tuple.

    Easy comparison between files can be done using the equality operator.

    Parameters
    ----------
    file_path : str or path-like
        The path to the file that is being hashed.
    """

    @staticmethod
    def _get_file_hash(file_path, block_size=4096 * 1024):
        file_hash = hashlib.sha256()

        with open(file_path, 'rb') as file:
            for block in iter(lambda: file.read(block_size), b""):
                file_hash.update(block)

        return file_hash.hexdigest()

    def __init__(self, file_path):
        self.file_path = file_path
        self._hash_type = 'sha256'
        self._hash = self._get_file_hash(file_path)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if isinstance(other, BuffaloFileHash):
            return hash(self) == hash(other) and \
                   self._hash_type == other._hash_type
        else:
            raise TypeError("Cannot compare different objects")

    def __repr__(self):
        return f"{Path(self.file_path).name}: " \
               f"[{self._hash_type}] {self._hash}"

    def info(self):
        """
        Returns provenance information for the file.

        Returns
        -------
        FileInfo
            A named tuple with the following attributes:
            * hash : int
                SHA256 hash of the file.
            * hash_type: str
                String storing the hash type (='sha256').
            * path : str or path-like
                The path to the file that was hashed.
        """
        return FileInfo(hash=self._hash,
                        hash_type=self._hash_type,
                        path=self.file_path)


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

    # TODO: don't make it static
    _hash_memoizer = dict()

    @classmethod
    def clear_memoization(cls):
        cls._hash_memoizer.clear()

    @classmethod
    def memoize(cls, id, hash_value):
        cls._hash_memoizer[id] = hash_value

    @classmethod
    def get_memoized(cls, id):
        return cls._hash_memoizer.get(id)

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
        self.type = f"{type(obj).__module__}.{type(obj).__name__}"
        self.id = id(obj)
        self.value = obj

    def __hash__(self):

        # If we already computed the hash for the object during this function
        # call, retrieve it from the memoized values
        print(self.type, self.id)
        memoized = self.get_memoized(self.id)
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
        self.memoize(self.id, object_hash)

        return object_hash

    def __repr__(self):
        return f"{self.id}: {self.type}"

    def info(self):
        """
        Returns provenance information for the object. If the object is None,
        then the hash is replaced by a unique id for the object.

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
        # All Nones will have the same hash. Use UUID instead
        if self.value is None:
            unique_id = uuid.uuid4()
            return ObjectInfo(unique_id, self.type, self.id, {})

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
