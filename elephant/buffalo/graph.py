import networkx as nx
from pyvis.network import Network

class BuffaloProvenanceGraph(nx.DiGraph):

    def add_input_to_output(self, analysis_step, input_obj, edge_label, multi_output, **attrs):
        self.add_node(input_obj, label=input_obj.type)

        if multi_output:
            for output_key, output_obj in analysis_step.output.items():
                self.add_edge(input_obj, output_obj, label=edge_label,
                              **attrs)
        else:
            self.add_edge(input_obj, analysis_step.output[0], label=edge_label,
                          **attrs)

    def add_step(self, analysis_step, **attr):
        from elephant.buffalo.provenance import VarArgs

        for key, obj in analysis_step.output.items():
            self.add_node(obj, label=obj.type)
        multi_output = len(list(analysis_step.output.keys())) > 1
        edge_attr = analysis_step.params
        edge_label = analysis_step.function.name
        if analysis_step.function.module:
            edge_label = analysis_step.function.module + "." + edge_label

        for key, obj in analysis_step.input.items():
            if isinstance(obj, VarArgs):
                for var_arg in obj.value:
                    self.add_input_to_output(analysis_step, var_arg,
                                             edge_label, multi_output,
                                             **edge_attr, **attr)
            else:
                self.add_input_to_output(analysis_step, obj, edge_label,
                                         multi_output, **edge_attr, **attr)

    def to_pyvis(self, filename, show=False, layout=True):
        """
        This method takes an exisitng Networkx graph and translates
        it to a PyVis graph format that can be accepted by the VisJs
        API in the Jinja2 template.

        Parameters
        ----------
        filename : str
            Destination where to save the file.
        show : bool
            If True, display the graph in the browser after saving.
            Default: False.
        layout : bool
            If True, use hierarchical layout if this is set.
            Default: True.

        """
        def _get_last_name(name):
            return name.split(".")[-1]

        def add_node(node_id):
            attr = nodes[node_id]
            level = attr.get('level', 1)
            net.add_node(hash(node_id), title=attr['label'],
                         label=_get_last_name(attr['label']))#, level=level)

        edges = self.edges.data()
        nodes = self.nodes
        # for node, degree in self.in_degree:
        #     if degree == 0:
        #         nodes[node]['level'] = 0

        net = Network(height="960px", width="1280px", directed=True, layout=layout)
        for v, u, edge_attr in edges:
            add_node(v)
            add_node(u)
            net.add_edge(hash(v), hash(u), title=edge_attr['label'],
                         label=_get_last_name(edge_attr['label']))
        net.save_graph(filename)
        if show:
            net.show(name=filename)
