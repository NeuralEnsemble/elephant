"""
This module implements functionality to serialize the provenance track using
the W3C Provenance Data Model (PROV).

:copyright: Copyright 2014-2021 by the Elephant team, see `doc/authors.rst`.
:license: BSD, see LICENSE.txt for details.
"""

from prov.model import ProvDocument, ProvAgent, ProvActivity, ProvEntity
from elephant.buffalo.object_hash import BuffaloFileHash
import pathlib


class BuffaloProvDocument(ProvDocument):

    def __init__(self, script_file_name, records=None, namespaces=None):
        super(ProvDocument, self).__init__(records=records,
                                           namespaces=namespaces)
        # Create an Agent to represent the script
        # We are using the SoftwareAgent subclass for the description
        # URI defined as the script name + file hash
        script_hash = BuffaloFileHash(script_file_name).info().hash
        script_name = pathlib.Path(script_file_name).name.replace('.', '_')
        script_uri = f"urn:{script_name}_{script_hash}"
        script_attributes = {'prov:type': 'prov:SoftwareAgent'}
        self.script_agent = ProvAgent(self, script_uri,
                                      attributes=script_attributes)

    def add_analysis_step(self, step):
        pass


def generate_prov_representation(script_file_name, analysis_steps):
    buffalo_prov = BuffaloProvDocument(script_file_name)
    for step in analysis_steps:
        buffalo_prov.add_analysis_step(step)
    return buffalo_prov