

class BuffaloProvenanceGraph(object):
    _nodes = None
    _edges = None

    def __init__(self, objects, history):
        self._nodes = objects
        self._edges = history

    def save_graph(self, filename):
        graph_definition = []
        for key, value in self._nodes.items():
            graph_definition.append(value.get_md_string())

        for entry in self._edges:
            for key, value in entry.input.items():
                function_name = entry.function.name
                if entry.function.module:
                    function_name = entry.function.module + "." + function_name
                function_name = function_name.replace(".", ".<br>")
                graph_definition.append('"{}" -->|"{}<br>{}"|"{}";'.format(
                    hash(value), function_name,
                    "<br>".join(["{}:{}".format(key, value) for key, value in
                                 entry.params.items()]),
                    hash(entry.output))
                )

        # TODO: Remove duplicates
        with open(filename, 'w') as f:
            f.writelines("{}\n".format(line) for line in
                         ["graph LR"] + list(set(graph_definition)))
