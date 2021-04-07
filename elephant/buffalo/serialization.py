"""
This module implements functionality to serialize the provenance track using
the W3C Provenance Data Model (PROV).

:copyright: Copyright 2014-2021 by the Elephant team, see `doc/authors.rst`.
:license: BSD, see LICENSE.txt for details.
"""

import pathlib
from io import BytesIO
from itertools import product

from prov.model import ProvDocument, PROV, PROV_TYPE, Namespace

from elephant.buffalo.object_hash import BuffaloFileHash
from elephant.buffalo.types import ObjectInfo


# Values that are used to compose the URNs

NID_ELEPHANT = "elephant"
NSS_FUNCTION = "function"
NSS_FILE = "file"
NSS_DATA = "object"
NSS_CONTAINER = "container"
NSS_SCRIPT = "script"


# Other namespaces used

RDFS = Namespace("rdfs", "http://www.w3.org/2000/01/rdf-schema#")


class BuffaloProvDocument(ProvDocument):

    def __init__(self, script_file_name):
        super().__init__(records=None, namespaces=None)

        # Set default namespace to avoid repetitions of the prefix
        self.add_namespace(Namespace("", f"urn:{NID_ELEPHANT}:"))

        # Add other namespaces
        self.add_namespace(RDFS)

        # Create an Agent to represent the script
        # We are using the SoftwareAgent subclass for the description
        # URN defined as :script:script_name:file_hash
        script_hash = BuffaloFileHash(script_file_name).info().hash
        script_name = pathlib.Path(script_file_name).name
        script_uri = f":{NSS_SCRIPT}:{script_name}:{script_hash}"
        script_attributes = {PROV_TYPE: PROV["SoftwareAgent"],
                             "rdfs:label": script_file_name}
        self._script_agent = self.agent(script_uri, script_attributes)

    def _create_entity(self, info):
        # Create a PROV Entity based on ObjectInfo/FileInfo information
        if isinstance(info, ObjectInfo):
            cur_uri = f":{NSS_DATA}:{info.type}:{info.hash}"
        else:
            cur_uri = f":{NSS_FILE}:{info.hash_type}:{info.hash}"

        cur_entity = self.entity(cur_uri)
        return cur_entity

    def _add_analysis_step(self, step):
        # Add one `AnalysisStep` record and generate all the provenance
        # semantic relationships
        activity_uri = f":{NSS_FUNCTION}:" \
                       f"{step.function.module}.{step.function.name}"
        cur_activity = self.activity(activity_uri)

        # Add all the inputs as entities, and create a `used` association with
        # the activity. URNs differ when the input is a file or Python object
        input_entities = []
        for key, value in step.input.items():
            cur_entity = self._create_entity(value)
            input_entities.append(cur_entity)
            self.used(cur_activity, cur_entity)

        # Add all the outputs as entities, and create the `wasGenerated`
        # relationship
        output_entities = []
        for key, value in step.output.items():
            cur_entity = self._create_entity(value)
            output_entities.append(cur_entity)
            self.wasGeneratedBy(cur_entity, cur_activity)

        # Iterate over the input/output pairs to add the `wasDerived`
        # relationship
        for input_entity, output_entity in \
                product(input_entities, output_entities):
            self.wasDerivedFrom(output_entity, input_entity)

        # Attribute the activity to the script
        self.wasAttributedTo(cur_activity, self._script_agent)

    @classmethod
    def read_records(cls, file_name, file_format=None):
        """
        Reads PROV data that was previously serialized.

        Parameters
        ----------
        file_name : str or path-like
            Location of the file with PROV data to be read.
        file_format : {'json', 'rdf', 'prov', 'xml'}
            Format used in the file that is being read.
            If None, the format will be inferred from the extension.
            Default: None

        Returns
        -------
        BuffaloProvDocument
            Instance with the loaded provenance data, that can be further
            used for plotting/querying.

        Raises
        ------
        ValueError
            If `file_format` is None and `file_name` has no extension to infer
            the format.
            If `file_format` is not 'rdf', 'json', 'prov', or 'xml'.
        """
        file_location = pathlib.Path(file_name)

        if file_format is None:
            extension = file_location.suffix
            if not extension.startswith('.'):
                raise ValueError("File has no extension and no format was "
                                 "provided")
            file_format = extension[1:]

        if file_format not in ['rdf', 'json', 'prov', 'xml']:
            raise ValueError("Unsupported serialization format")

        with open(file_name, "r") as source:
            return cls.deserialize(source, format=file_format)

    def process_analysis_steps(self, analysis_steps):
        for step in analysis_steps:
            self._add_analysis_step(step)
