"""
This module implements a hashing object that is used by the `Provenance`
class decorator to track unique objects during the script execution.
"""

import joblib
from dill._dill import save_function
from collections import namedtuple


# Need to use `dill` pickling function to support lambdas
# The dispatch table of the `joblib.Hasher` object is updated
joblib.hashing.Hasher.dispatch[type(save_function)] = save_function


ObjectInfo = namedtuple('ObjectInfo', ('hash', 'type', 'id', 'details'))


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

    @staticmethod
    def _get_object_info(obj):
        class_name = "{}.{}".format(type(obj).__module__,
                                    type(obj).__name__)
        return id(obj), class_name, obj

    def __init__(self, obj):
        self.id, self.type, self.value = self._get_object_info(obj)

    def __hash__(self):
        return hash((self.id, self.type, joblib.hash(self.value)))

    def __eq__(self, other):
        if isinstance(other, BuffaloObjectHash):
            return hash(self) == hash(other)
        else:
            object_id, class_name, value = self._get_object_info(other)
            if value is self.value:
                return True
            else:
                return (object_id, class_name, value) == (
                    self.id, self.type, self.value
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
