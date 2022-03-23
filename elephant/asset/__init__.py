try:
    from .asset import *
except ImportError as error:  # pragma: no cover
    # requirements-extras are missing
    error.msg += ', consider installing elephant with extras: run command ' \
                 '`pip install -r requirements-extras.txt or pip install ' \
                 'elephant[extras]'
    print(error.msg)
    pass
