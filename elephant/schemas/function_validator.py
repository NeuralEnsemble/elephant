from functools import wraps
from inspect import signature
from pydantic import BaseModel

skip_validation = False

def validate_with(model_class: type[BaseModel]):
    """
    A decorator that validates the inputs of a function using a Pydantic model.
    Works for both positional and keyword arguments.
    """
    def decorator(func):
        sig = signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):

            if not skip_validation:
                # Bind args & kwargs to function parameters
                bound = sig.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                data = bound.arguments

                # Validate using Pydantic
                model_class(**data)

            # Call function
            return func(*args, **kwargs)
        return wrapper
    return decorator

def activate_validation():
    global skip_validation
    skip_validation = False

def deactivate_validation():
    global skip_validation
    skip_validation = True