import io

from nixio.data_array import DataArray
from nixio.compression import Compression
from nixio.section import Section
from nixio import File, FileMode

import quantities as pq
from neo import AnalogSignal, NixIO
from neo.io.nixio import units_to_string

from uuid import uuid4
from copy import copy as copy_fn
from collections import UserDict


class BuffaloAnnotations(UserDict):

    def __init__(self, metadata):
        super(BuffaloAnnotations, self).__init__()
        self._metadata = metadata

    def __setitem__(self, key, value):
        super(BuffaloAnnotations, self).__setitem__(key, value)
        for k, v in self.items():
            if k in self._metadata.props:
                del self._metadata.props[k]
            # TODO: check if reimplementing the function is better
            NixIO._write_property(self, self._metadata, k, v)


class NixAnalysisObject(DataArray):
    """
    Base analysis object class.

    Inherited from a NIX DataArray, supports storage of basic provenance
    information.
    """

    _nix_name = None

    def _create_new(cls, nixparent, h5parent, name, type_, data_type, shape,
                    compression):

        # If not passing a Nix parent, create a file and block
        if nixparent is None:
            storage = File("object.nix", FileMode.Overwrite)
            if "block" in storage.blocks:
                del storage.blocks["block"]
            nixparent = storage.create_block("block", name, compression)

        # If no parent group specified, open a new h5group called
        # "data_arrays"
        if h5parent is None:
            data_arrays = nixparent._h5group.open_group("data_arrays")
            if name in data_arrays:
                raise Exception("duplicate name")
        else:
            data_arrays = h5parent

        obj = DataArray._create_new(nixparent=nixparent, h5parent=data_arrays,
                                    name=name, type_=type_,
                                    data_type=data_type, shape=shape,
                                    compression=compression)

        return obj

    def __init__(self, nixparent, h5group):
        if h5group is None:
            h5group = self._h5group
        if nixparent is None:
            nixparent = self._parent

        super(NixAnalysisObject, self).__init__(nixparent, h5group)

        # Properties to store flag about warnings in the function
        self.warnings_raised = False

        # Create metadata section
        self.metadata = Section._create_new(self, self._h5group, "metadata",
                                            f"{self.__class__.__name__}"
                                            f".metadata")

        # Create custom dictionary that supports storage in the NIX file
        self.annotations = BuffaloAnnotations(self.metadata)

        # Create array dimensions
        self._add_dimensions()

    def _add_dimensions(self):
        pass

    @property
    def name(self):
        return super(NixAnalysisObject, self).name

    @name.setter
    def name(self, value):
        # Neo initialization needs to set the name. Create a setter to
        # prevent AttributeError
        self._nix_name = value

    @property
    def nix_array(self):
        return self._parent.data_arrays[0]

    @property
    def pid(self):
        return self.id


class BuffaloAnalogSignal(AnalogSignal, NixAnalysisObject):
    """
    Class for analysis output in Elephant based on NIX backend.

    This class supports storage of basic provenance information.
    """

    def __new__(cls, signal, units=None, dtype=None, copy=True,
                t_start=0 * pq.s, sampling_rate=None, sampling_period=None,
                name=None, file_origin=None, description=None,
                array_annotations=None, nixparent=None, h5group=None,
                compression=Compression.Auto, **annotations):

        class_name = cls.__name__
        nix_name = f"{class_name}.{uuid4()}" if name is None else name

        array = NixAnalysisObject._create_new(cls, nixparent=nixparent,
                                              h5parent=None, name=nix_name,
                                              type_=class_name,
                                              data_type=signal.dtype,
                                              shape=signal.shape,
                                              compression=compression)

        signal_object = AnalogSignal.__new__(cls, signal=signal,
                                             units=units, dtype=dtype,
                                             copy=copy, t_start=t_start,
                                             sampling_rate=sampling_rate,
                                             sampling_period=sampling_period,
                                             name=name,
                                             file_origin=file_origin,
                                             description=description,
                                             array_annotations=
                                             array_annotations)

        # Deepcopy does not work. Create a new dict and do shallow copy on
        # the main items of the object __dict__
        array_dict = dict()
        for k, v in array.__dict__.items():
            array_dict[k] = copy_fn(v)

        # Update the dict with DataArray attributes
        signal_object.__dict__.update(array_dict)

        return signal_object

    def __init__(self, signal, units=None, dtype=None, copy=True,
                 t_start=0 * pq.s, sampling_rate=None, sampling_period=None,
                 name=None, file_origin=None, description=None,
                 array_annotations=None, nixparent=None, h5group=None,
                 compression=Compression.Auto, **annotations):

        # Init AnalogSignal.
        # Annotations are NOT passed before the creation of the custom
        # dictionary.
        AnalogSignal.__init__(self, signal, units=units, dtype=dtype,
                              copy=copy, t_start=t_start,
                              sampling_rate=sampling_rate,
                              sampling_period=sampling_period, name=name,
                              file_origin=file_origin, description=description,
                              array_annotations=array_annotations)

        NixAnalysisObject.__init__(self, nixparent, h5group)

        # Store AnalogSignal data in DataArray
        self.write_direct(signal)

        # Add units to DataArray
        self.unit = units_to_string(self.units)

        # Insert annotations
        self.annotate(**annotations)

    def _add_dimensions(self):
        time_dimension = self.append_sampled_dimension(
            self.sampling_period.magnitude.item()
        )
        time_dimension.unit = units_to_string(self.sampling_period.units)
        time_dimension.offset = self.t_start.rescale(
            time_dimension.unit).magnitude.item()
        time_dimension.label = "Time"

    def __setitem__(self, key, value):
        super(BuffaloAnalogSignal, self).__setitem__(key, value)

        if isinstance(value, pq.Quantity):
            if value.dimensionality != self.units.dimensionality:
                value = value.rescale(self.units)

        self._write_data(value.magnitude.T, key)
