try:
    from .gpfa import GPFA
except ImportError as error:  # pragma: no cover
    # please run command `pip install -r requirements-extras.txt`
    error.msg += ', consider installing elephant with extras: run command ' \
                 '`pip install -r requirements-extras.txt or pip install ' \
                 'elephant[extras]'
    print(error.msg)
    pass

__all__ = [
    "GPFA"
]
