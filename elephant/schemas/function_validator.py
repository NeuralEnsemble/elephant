from functools import wraps
from inspect import signature
from pydantic import BaseModel

def validate_with(model_class: type[BaseModel]):
    """
    A decorator that validates the inputs of a function using a Pydantic model.
    Works for both positional and keyword arguments.
    """
    def decorator(func):
        sig = signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Bind args & kwargs to function parameters
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            data = bound.arguments

            # Validate using Pydantic
            validated = model_class(**data)

            # Call function with validated data unpacked
            return func(**validated.model_dump())

        return wrapper
    return decorator