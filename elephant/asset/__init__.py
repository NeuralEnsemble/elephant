try:
    from .asset import *
except ImportError:  # pragma: no cover
    #  requirements-extras are missing
    pass
