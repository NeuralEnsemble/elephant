"""
This module implements a hashing object that is used by the `Provenance`
class decorator to track unique objects during the script execution.
"""

import joblib


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

    _id = None
    _type = None
    _value = None

    @staticmethod
    def _get_object_info(obj):
        class_name = "{}.{}".format(type(obj).__module__,
                                    type(obj).__name__)
        return id(obj), class_name, obj

    def __init__(self, obj):
        self._id, self._type, self._value = self._get_object_info(obj)

    def __hash__(self):
        return hash((self._id, self._type, joblib.hash(self._value)))

    def __eq__(self, other):
        if isinstance(other, BuffaloObjectHash):
            return hash(self) == hash(other)
        else:
            object_id, class_name, value = self._get_object_info(other)
            if value is self._value:
                return True
            else:
                return (object_id, class_name, value) == (
                    self._id, self._type, self._value
                )

    def __repr__(self):
        return "{}: {} = {}".format(self._id, self._type, self._value)

    @property
    def value(self):
        """
        The actual Python object associated with this hash.
        """
        return self._value

    def get_md_string(self):
        # TODO: use other solution for better plots
        value = "id: {}".format(self._id)
        return '"{}"["{}<br>{}"]'.format(hash(self), self._type, value)
