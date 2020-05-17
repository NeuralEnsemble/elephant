from prov.model import ProvDocument
from prov.dot import prov_to_dot


class BuffaloProvDocument(ProvDocument):
    # TODO: refactor class and namespaces

    def __init__(self, records=None, namespaces=None, history=None,
                 objects=None):
        if history is None or objects is None:
            raise ValueError("Need provenance history and objects to generate"
                             "graph")

        super(BuffaloProvDocument, self).__init__(records, namespaces)
        self.add_namespace("functions", "functions")
        self.add_namespace("data", "data")
        self.add_namespace("params", "parameters")
        self.add_namespace("description", "descriptions")

        for item in history:
            self.add_step(item)

    def _get_function_representation(self, function_definition,
                                     namespace="functions"):
        return "{}:{}:{}".format(namespace, function_definition.module,
                                 function_definition.name)

    def add_step(self, analysis_step):
        activity = self.activity(self._get_function_representation(
                                 analysis_step.function))

        function_params = self._get_attributes_from_dict(analysis_step.params)
        output_type = analysis_step.output.type
        function_params.append(('description:type', output_type))
        output_entity = self.entity(
            analysis_step.output.get_prov_entity_string("data"))
        output_entity.wasGeneratedBy(activity)

        for key, value in analysis_step.input.items():
            input_entity = self.entity(value.get_prov_entity_string("data"))
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
