try:
    from .gpfa import GPFA
except ImportError:
    # please run command `pip install -r requirements-extras.txt`
    pass

__all__ = [
    "GPFA"
]
