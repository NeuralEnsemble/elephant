import networkx as nx
from pyvis.network import Network


class BuffaloProvenanceGraph(nx.DiGraph):

    def add_step(self, analysis_step, **attr):
        self.add_node(analysis_step.output, label=analysis_step.output.type)
        edge_attr = analysis_step.params
        edge_label = analysis_step.function.name
        if analysis_step.function.module:
            edge_label = analysis_step.function.module + "." + edge_label

        for key, obj in analysis_step.input.items():
            self.add_node(obj, label=obj.type)
            self.add_edge(obj, analysis_step.output, label=edge_label,
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
            net.add_node(hash(node_id), title=attr['label'])

        edges = self.edges.data()
        nodes = self.nodes
        net = Network(height="960px", width="1280px", directed=True, layout=layout)
        for v, u, edge_attr in edges:
            add_node(v)
            add_node(u)
            net.add_edge(hash(v), hash(u), title=edge_attr['label'])
        net.save_graph(filename)
        if show:
            net.show(name=filename)
