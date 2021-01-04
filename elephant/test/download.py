import hashlib
import os
import tempfile
from zipfile import ZipFile

from tqdm import tqdm

from urllib.request import urlretrieve

ELEPHANT_TMP_DIR = os.path.join(tempfile.gettempdir(), "elephant")


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


def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def check_integrity(fpath, md5):
    if not os.path.exists(fpath) or md5 is None:
        return False
    return calculate_md5(fpath) == md5


def download(url, filepath=None, checksum=None, verbose=True):
    if filepath is None:
        filename = url.split('/')[-1]
        filepath = os.path.join(ELEPHANT_TMP_DIR, filename)
    if check_integrity(filepath, md5=checksum):
        return filepath
    folder = os.path.dirname(os.path.abspath(filepath))
    if not os.path.exists(folder):
        os.mkdir(folder)
    desc = "Downloading '{url}' to '{filepath}'".format(url=url,
                                                        filepath=filepath)
    with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                  desc=desc, disable=not verbose) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to)
    return filepath


def unzip(filepath, outdir=ELEPHANT_TMP_DIR, verbose=True):
    with ZipFile(filepath) as zfile:
        zfile.extractall(path=outdir)
    if verbose:
        print("Extracted {filepath} to {outdir}".format(filepath=filepath,
                                                        outdir=outdir))
