from functools import wraps


def _no_provenance(*dec_args, **dec_kwargs):
    """
    This function will be imported if the user has not configured
    the environment to use provenance tracking. It will wrap the function
    without performing any actions.
    """
    def function_decorator(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            return function(*args, **kwargs)

        return wrapped
    return function_decorator
