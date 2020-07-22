import networkx as nx
from pyvis.network import Network


class BuffaloProvenanceGraph(nx.DiGraph):

    def _add_input_to_output(self, analysis_step, input_obj, edge_label,
                             edge_title, multi_output, **attrs):
        obj_type = input_obj.type
        obj_label = obj_type.split(".")[-1]
        self.add_node(input_obj, label=obj_label, title=obj_type)

        if multi_output:
            for output_key, output_obj in analysis_step.output.items():
                self.add_edge(input_obj, output_obj, label=edge_label,
                              title=edge_title, **attrs)
        else:
            self.add_edge(input_obj, analysis_step.output[0], label=edge_label,
                          title=edge_title, **attrs)

    @staticmethod
    def _get_edge_attrs_and_labels(analysis_step):
        edge_attr = analysis_step.params
        edge_label = analysis_step.function.name
        edge_title = edge_label

        if edge_label in ['attribute', 'subscript']:
            if edge_label == 'attribute':
                edge_label = ".{}".format(edge_attr['name'])
            elif edge_label == 'subscript':
                if 'slice' in edge_attr:
                    edge_label = str(edge_attr['slice'])
                else:
                    edge_label = "[{}]".format(edge_attr['index'])
        else:
            if analysis_step.function.module:
                edge_title = analysis_step.function.module + "." + edge_title
            edge_label = edge_title.split(".")[-1]

        return edge_attr, edge_label, edge_title

    def add_step(self, analysis_step, **attr):
        from elephant.buffalo.provenance import VarArgs

        for key, obj in analysis_step.output.items():
            obj_type = obj.type
            obj_label = obj_type.split(".")[-1]
            self.add_node(obj, label=obj_label, title=obj_type)
        multi_output = len(list(analysis_step.output.keys())) > 1

        edge_attr, edge_label, edge_title = \
            self._get_edge_attrs_and_labels(analysis_step)

        for key, obj in analysis_step.input.items():
            if isinstance(obj, VarArgs):
                for var_arg in obj.value:
                    self._add_input_to_output(analysis_step, var_arg,
                                              edge_label, edge_title,
                                              multi_output, **edge_attr,
                                              **attr)
            else:
                self._add_input_to_output(analysis_step, obj, edge_label,
                                          edge_title, multi_output,
                                          **edge_attr, **attr)

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

        def add_node(node_id):
            attr = nodes[node_id]
            level = attr.get('level', None)
            net.add_node(hash(node_id), title=attr['title'],
                         label=attr['label'], level=level)

        edges = self.edges.data()
        nodes = self.nodes

        # Go through the graph from the root(s), to set the levels
        roots = [node for node, degree in self.in_degree() if degree == 0]
        for root in roots:
            nodes[root]['level'] = 0
            level = 1
            children = list(self.succ[root])
            while len(children) > 0:
                next_children = list()
                for node in children:
                    if nodes[node].get('level', 0) < level:
                        nodes[node]['level'] = level
                    next_children += list(self.succ[node])
                level += 1
                children = next_children

        net = Network(height="960px", width="1280px", directed=True,
                      layout=layout)
        for v, u, edge_attr in edges:
            add_node(v)
            add_node(u)
            net.add_edge(hash(v), hash(u), title=edge_attr['title'],
                         label=edge_attr['label'])
        net.save_graph(filename)
        if show:
            net.show(name=filename)
