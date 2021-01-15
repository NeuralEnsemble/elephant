import networkx as nx
from pyvis.network import Network
import uuid


class BuffaloProvenanceGraph(nx.DiGraph):

    def _add_input_to_output(self, analysis_step, input_obj, edge_label,
                             edge_title, multi_output, function_edge, **attrs):

        def _connect_edge(input_obj, output_obj, function_edge):
            if function_edge:
                self.add_node(function_edge, label=edge_label,
                              title=edge_title, type='function',
                              params=analysis_step.params)
                if input_obj is not None:
                    self.add_edge(input_obj.hash, function_edge, type='input',
                                  **attrs)
                if output_obj is not None:
                    self.add_edge(function_edge, output_obj.hash, type='output',
                                  **attrs)
            else:
                self.add_edge(input_obj.hash, output_obj.hash, label=edge_label,
                              title=edge_title, params=analysis_step.params,
                              type='static', **attrs)

        if input_obj is not None:
            obj_type = input_obj.type
            obj_label = obj_type.split(".")[-1]
            self.add_node(input_obj.hash, label=obj_label, title=obj_type,
                          type='data')

        if multi_output:
            for output_key, output_obj in analysis_step.output.items():
                _connect_edge(input_obj, output_obj, function_edge)
        else:
            if len(analysis_step.output):
                output_obj = analysis_step.output[0]
            else:
                output_obj = None
            _connect_edge(input_obj, output_obj, function_edge)

    @staticmethod
    def _get_edge_attrs_and_labels(analysis_step):
        edge_attr = analysis_step.params
        edge_label = analysis_step.function.name
        edge_title = edge_label

        function_edge = None
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
            function_edge = hash(str(uuid.uuid4()))

        return edge_label, edge_title, function_edge

    def add_step(self, analysis_step, **attr):
        from elephant.buffalo.provenance import VarArgs

        for key, obj in analysis_step.output.items():
            obj_type = obj.type
            obj_label = obj_type.split(".")[-1]
            self.add_node(obj.hash, label=obj_label, title=obj_type, type='data')
        multi_output = len(list(analysis_step.output.keys())) > 1

        edge_label, edge_title, function_edge = \
            self._get_edge_attrs_and_labels(analysis_step)

        if len(analysis_step.input.keys()):
            for key, obj in analysis_step.input.items():
                if isinstance(obj, VarArgs):
                    for var_arg in obj.args:
                        self._add_input_to_output(analysis_step, var_arg,
                                                  edge_label, edge_title,
                                                  multi_output, function_edge,
                                                  **attr)
                else:
                    self._add_input_to_output(analysis_step, obj, edge_label,
                                              edge_title, multi_output,
                                              function_edge, **attr)
        else:
            # Function without input
            self._add_input_to_output(analysis_step, None, edge_label,
                                      edge_title, multi_output,
                                      function_edge, **attr)

    def to_pyvis(self, filename, show=False, layout=True):
        """
        This method takes an exisitng Networkx graph and translates
        it to a PyVis graph format that can be accepted by the VisJs
        API in the Jinja2 template.

        Parameters
        ----------
        filename : str
            Destination where to save the file.
        show : bool, optional
            If True, display the graph in the browser after saving.
            Default: False.
        layout : bool, optional
            If True, use hierarchical layout if this is set.
            Default: True.

        """

        def add_node(node_id):
            attr = nodes[node_id]
            level = attr.get('level', None)
            node_type = attr.get('type', 'unknown')
            shape = shape_types[node_type]
            color = color_types[node_type]
            net.add_node(hash(node_id), level=level, shape=shape,
                         color=color, label=attr['label'], title=attr['title'])

        shape_types = {'data': 'dot', 'function': 'square',
                       'unknown': 'triangle'}
        color_types = {'data': 'blue', 'function': 'red', 'unknown': 'green'}

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
            labels = {key: edge_attr[key] for key in ('label', 'title')
                      if key in edge_attr}
            net.add_edge(hash(v), hash(u), **labels)

        net.show_buttons()
        net.save_graph(filename)
        if show:
            net.show(name=filename)
