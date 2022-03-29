import hashlib
import tempfile
import warnings

from elephant import _get_version
from pathlib import Path
from urllib.request import urlretrieve, urlopen
from urllib.error import HTTPError
from zipfile import ZipFile
from os import environ, getenv

from tqdm import tqdm

ELEPHANT_TMP_DIR = Path(tempfile.gettempdir()) / "elephant"

warnings.simplefilter('once', DeprecationWarning)
warnings.warn("download module will be removed in Elephant v0.12.x, please use"
              "elephant.data_utils",
              DeprecationWarning)


class TqdmUpTo(tqdm):
    """
    Provides `update_to(n)` which uses `tqdm.update(delta_n)`.
    Original implementation:
    https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b : int, optional
            Number of blocks transferred so far [default: 1].
        bsize : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def calculate_md5(filepath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def check_integrity(filepath, md5):
    if not Path(filepath).exists() or md5 is None:
        return False
    return calculate_md5(filepath) == md5


def download(url, filepath=None, checksum=None, verbose=True):
    if filepath is None:
        filename = url.split('/')[-1]
        filepath = ELEPHANT_TMP_DIR / filename
    filepath = Path(filepath)
    if check_integrity(filepath, md5=checksum):
        return filepath
    folder = filepath.absolute().parent
    folder.mkdir(exist_ok=True)
    desc = f"Downloading {url} to '{filepath}'"
    with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                  desc=desc, disable=not verbose) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to)
    return filepath


def unzip(filepath, outdir=ELEPHANT_TMP_DIR, verbose=True):
    with ZipFile(filepath) as zfile:
        zfile.extractall(path=outdir)
    if verbose:
        print(f"Extracted {filepath} to {outdir}")
