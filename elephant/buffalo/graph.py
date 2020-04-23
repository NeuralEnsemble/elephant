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
                function_name = entry.function.name
                function_name = entry.function.module + "." + function_name
                function_name = function_name.replace(".", ".<br>")
                graph_definition += '{} -->|"{}<br>{}"|{};\n'.format(hash(value),
                                                             function_name,
                                                             "<br>".join(["{}:{}".format(key, value) for key, value in entry.params.items()]),
                                                             hash(entry.output))

        print(graph_definition)
