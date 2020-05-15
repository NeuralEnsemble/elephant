from prov.model import ProvDocument
from prov.dot import prov_to_dot


class BuffaloProvDocument(ProvDocument):
    # FIX: refactor class and namespaces

    def __init__(self, records=None, namespaces=None):
        super(BuffaloProvDocument, self).__init__(records, namespaces)
        self.add_namespace("functions", "functions")
        self.add_namespace("data", "data")
        self.add_namespace("params", "parameters")
        self.add_namespace("description", "descriptions")

    def add_step(self, analysis_step):
        activity = self.activity("functions:" + analysis_step.function.name)

        function_params = self._get_attributes_from_dict(analysis_step.params)
        output_type = type(analysis_step.output)
        function_params.append(('description:type', str(output_type)))
        output_entity = self.entity("data:" + str(id(analysis_step.output)))
        output_entity.wasGeneratedBy(activity)

        for key, value in analysis_step.input.items():
            input_type = type(value)
            input_id = id(value)
            input_entity = self.entity("data:{}".format(input_id))
            output_entity.wasDerivedFrom(input_entity)
            activity.used(input_entity)

    @staticmethod
    def _get_attributes_from_dict(attributes, namespace="params"):
        prov_attributes = []
        for key, value in attributes.items():
            prov_attributes.append((namespace + ":" + key, value))
        return prov_attributes

    def get_dot_graph(self, **kwargs):
        return prov_to_dot(self, **kwargs)
