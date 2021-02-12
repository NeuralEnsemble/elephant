"""
This module contains utility functions to generate file names.
"""
from pathlib import Path


def get_file_name(source, output_dir=None, extension=None):
    """
    Function that generates a file name with extension `extension` and the
    same base name as in `source`. The full path is based on `output_dir`.

    Parameters
    ----------
    source : str
        Source path or file name to generate the new file name. The base name
        will be considered.
    output_dir : str, optional
        If not None, the generated file name will have this path.
        Default: None
    extension : str, optional
        If not None, the extension of the generated file name will be changed
        to `extension`. If None, the same extension as `source` will be used.
        The extension must start with period.

    Returns
    -------
    str
        File name, according to the parameters selected. If both `output_dir`
        and `extension` are None, the result will be equal to `source`.
    """
    source_file = Path(source)
    if extension is not None:
        base_name = Path(source_file.stem + extension)
    else:
        base_name = source_file

    if output_dir is not None:
        return str(Path(output_dir) / base_name)
    return str(base_name)
