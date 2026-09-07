import warnings

try:
    from .asset import *
except ImportError as err:
    # requirements - extras are missing
    warnings.warn(
        "elephant.asset requires 'extras' optional dependencies "
        f"Do `pip install elephant[extras]`; import failed with: {err} "
    )
