"""
This module implements a hashing object that is used by the `Provenance`
class decorator to track unique objects during the script execution.
"""

import joblib
import hashlib
from dill._dill import save_function
from collections import namedtuple
from pathlib import Path
import inspect
import numpy as np


# Need to use `dill` pickling function to support lambdas
# The dispatch table of the `joblib.Hasher` object is updated
joblib.hashing.Hasher.dispatch[type(save_function)] = save_function


ObjectInfo = namedtuple('ObjectInfo', ('hash', 'type', 'id', 'details'))
FileInfo = namedtuple('FileInfo', ('hash', 'hash_type', 'path', 'details'))


class BuffaloFileHash(object):

    HASH_TYPES = {'md5': hashlib.md5,
                  'sha256': hashlib.sha256}

    def _get_file_hash(self, file_path, hash_type='sha256',
                       block_size=4096*1024):
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
    Python object hash.

    Methods to use with equal to operators and :func:`hash` builtin function
    are implemented.

    The object hash value is obtaining by hashing a tuple consisting of: the
    object reference, the object type, and the SHA-1 hash of its value.
    The object reference is returned by :func:`id`. The type is a string
    produced by the combination of the module where it was defined plus the
    type name (both returned by :func:`type`). Value hash is calculated using
    :func:`joblib.hash`.

    Parameters
    ----------
    obj : object
        A Python object that will be hashed with respect to type, content and
        reference.
    """

    def _get_object_package(self, obj):
        module = inspect.getmodule(obj)
        package = ""
        if module is not None:
            package = module.__package__.split(".")[0]
        return package

    def _get_object_info(self, obj):
        package = self._get_object_package(obj)
        class_name = "{}.{}".format(type(obj).__module__,
                                    type(obj).__name__)
        return id(obj), class_name, obj, package

    def __init__(self, obj):
        self.id, self.type, self.value, self.package = \
            self._get_object_info(obj)

    def __hash__(self):
        # For matplotlib objects, we need to use the builtin hashing function
        # instead of the joblib's
        # Multiple objects are generated, since each time something is plotted
        # the object changes.
        # We also have to use an exception for NumPy arrays with Axes objects,
        # as those also change when the plot changes. These are usually return
        # by the `plt.subplots()` call

        array_of_matplotlib = False
        if (isinstance(self.value, np.ndarray) and
            self.value.dtype == 'O' and
            self._get_object_package(self.value) == 'matplotlib'):
                array_of_matplotlib = True

        if self.package in ['matplotlib']:
            value_hash = hash(self.value)
        elif array_of_matplotlib:
            value_hash = id(self.value)
        else:
            value_hash = joblib.hash(self.value)

        return hash((self.type, value_hash))

    def __eq__(self, other):
        if isinstance(other, BuffaloObjectHash):
            return hash(self) == hash(other)
        else:
            object_id, class_name, value, package = self._get_object_info(other)
            if value is self.value:
                return True
            else:
                return (class_name, value) == (
                    self.type, self.value
                )

    def __repr__(self):
        return "{}: {}".format(self.id, self.type)

    def get_label(self):
        return self.type

    def get_prov_entity_string(self, namespace):
        return "{}:{}:{}".format(namespace, self.type, hash(self))

    def info(self):
        # Here we can extract specific metadata to record
        # Currently fetching the whole class dictionary
        details = {}
        if hasattr(self.value, '__dict__'):
            details = self.value.__dict__
        return ObjectInfo(hash(self), self.type, self.id, details)
