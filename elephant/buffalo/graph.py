import md_mermaid as md


class BuffaloProvGraph(object):
    _nodes = None
    _edges = None

    def __init__(self, objects, history):
        self._nodes = objects
        self._edges = history

    def print_graph(self):
        graph_definition = "graph LR\n"
        for key, value in self._nodes.items():
            graph_definition += value.get_md_string()

        for entry in self._edges:
            for key, value in entry.input.items():
                graph_definition += '{} -->|"{}<br>{}"|{};\n'.format(hash(value),
                                                             entry.function.name,
                                                             "<br>".join(["{}:{}".format(key, value) for key, value in entry.params.items()]),
                                                             hash(entry.output))

        print(graph_definition)
