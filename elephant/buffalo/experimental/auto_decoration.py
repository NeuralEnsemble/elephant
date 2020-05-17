import inspect
import ast
import numpy as np
from functools import wraps


class MockContainer(object):
    """
    Mock class to mimic Neo objects with nested elements.
    """

    def __init__(self, default=None):
        if default is None:
            self.segments = list()
            self.segments.append(MockContainer('spiketrains'))
        elif isinstance(default, str):
            setattr(self, default, list())
            self.__dict__[default].append([0])


# This is the decorator that should wrap every function in the main script
def tag_function(tag=None):
    def wrap_function(function):
        #@wraps(function)
        def decorated_function(*args, **kwargs):
            frame = inspect.getouterframes(inspect.currentframe())[1]
            tree = ast.parse(frame.code_context[0].strip())
            # pprint(ast.dump(tree))
            print("{} tagged with {}".format(function.__name__, tag))
            return function(*args, **kwargs)

        return decorated_function

    return wrap_function


class DecorateCalls(ast.NodeVisitor):
    """
    Class to identify Call AST Nodes, and apply `tag_function` decorator
    where needed.
    """

    # Do not decorate print, the auto-decorating function, and functions
    # that are already decorated
    EXCLUSIONS = ['print', 'decorate_functions', 'tag_function']

    # Lists of auto-decorated functions
    FUNCTIONS = []

    @staticmethod
    def _do_decorate(func):
        globals()[func.__name__] = tag_function("auto tag")(func)

    def decorate(self, node):
        if not node in self.EXCLUSIONS:
            x = globals().get(node, None)
            if x is not None and x not in self.FUNCTIONS:
                self.FUNCTIONS.append(x)
                self._do_decorate(x)

    def visit_Call(self, tree_node):
        # Direct call to function
        if hasattr(tree_node.func, 'id'):
            self.decorate(tree_node.func.id)
        # TODO: calls of functions in modules (e.g., np.array)


def decorate_functions():
    """
    Function that does the automatic decoration.
    It should be called at the scope where other functions should be decorated.
    """
    # Retrieve code of the frame calling this function
    source_code = inspect.getsourcelines(inspect.currentframe().f_back)

    # Build AST tree
    tree = ast.parse("".join(source_code[0]))

    # Analyze AST tree to identify Call entries, and decorate if needed
    analyzer = DecorateCalls()
    analyzer.visit(tree)

    # Return functions that were decorated
    result = analyzer.FUNCTIONS
    return result


# SCRIPT CODE

def transform(source, exponent=2):
    return [i ** exponent for i in source]


@tag_function("manual tag")
def generate_list(number):
    return [i for i in range(number)]


def get_item(source, order):
    return source[order]


def load_data():
    return MockContainer()


def main():
    functions = decorate_functions()

    block = load_data()

    spike_train = block.segments[0].spiketrains[0]
    test4 = np.array(spike_train)
    test = generate_list(5)
    test2 = generate_list(6)

    transformed = transform(test, 3)
    transformed2 = [i ** 2 for i in test2]
    transformed3 = transform(test2)

    random_var = get_item(transformed, 2)
    random_var2 = transformed[2]

    print("\n\nScript output\n")
    print(transformed)
    print(transformed2)
    print(transformed3)


if __name__ == "__main__":
    main()
