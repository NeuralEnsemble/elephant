import hashlib
import tempfile
import warnings
import ssl

from elephant import _get_version
from pathlib import Path
from urllib.request import urlretrieve, urlopen
from urllib.error import HTTPError, URLError
from zipfile import ZipFile
from os import environ, getenv

from tqdm import tqdm

ELEPHANT_TMP_DIR = Path(tempfile.gettempdir()) / "elephant"


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
        try:
            urlretrieve(url, filename=filepath, reporthook=t.update_to)
        except URLError:
            urlretrieve(url, filename=filepath, reporthook=t.update_to,
                        data=ssl._create_unverified_context())

    return filepath


def download_datasets(repo_path, filepath=None, checksum=None,
                      verbose=True):
    r"""
        This function can be used to download files from elephant-data using
        only the path relative to the root of the elephant-data repository.
        The default URL used, points to elephants corresponding release of
        elephant-data.
        Different versions of the elephant package may require different
        versions of elephant-data.
        e.g. the follwoing URLs:
        -  https://web.gin.g-node.org/INM-6/elephant-data/raw/0.0.1
           points to release v0.0.1.
        -  https://we.gin.g-node.org/INM-6/elephant-data/raw/master
           always points to the latest state of elephant-data.
        -  http://datasets.python-elephant.org/
           points to the root of elephant data

        To change this URL, use the environment variable `ELEPHANT_DATA_URL`.
        When using data, which is not yet contained in the master branch or a
        release of elephant data, e.g. during development, this variable can
        be used to change the default URL.
        For example to use data on branch `multitaper`, change the
        `ELEPHANT_DATA_URL` to
        https://web.gin.g-node.org/INM-6/elephant-data/raw/multitaper.
        For a complete example, see Examples section.

        Parameters
        ----------
        repo_path : str
            String denoting the path relative to elephant-data repository root
        filepath : str, optional
            Path to temporary folder where the downloaded files will be stored
        checksum : str, otpional
            Checksum to verify dara integrity after download
        verbose : bool, optional
            Whether to disable the entire progressbar wrapper [].
            If set to None, disable on non-TTY.
            Default: True

        Returns
        -------
        filepath : pathlib.Path
            Path to downloaded files.


        Notes
        -----
        The default URL always points to elephant-data. Please
        do not change its value. For development purposes use the environment
        variable 'ELEPHANT_DATA_URL'.

        Examples
        --------
        The following example downloads a file from elephant-data branch
        'multitaper', by setting the environment variable to the branch URL:

        >>> import os
        >>> from elephant.datasets import download_datasets
        >>> os.environ["ELEPHANT_DATA_URL"] = "https://web.gin.g-node.org/INM-6/elephant-data/raw/multitaper" # noqa
        >>> download_datasets("unittest/spectral/multitaper_psd/data/time_series.npy")
        PosixPath('/tmp/elephant/time_series.npy')
        """

    # this url redirects to the current location of elephant-data
    url_to_root = "http://datasets.python-elephant.org/"

    # get URL to corresponding version of elephant data
    # (version elephant is equal to version elephant-data)
    default_url = url_to_root + f"raw/v{_get_version()}"

    if 'ELEPHANT_DATA_URL' not in environ:  # user did not set URL
        # is 'version-URL' available? (not for elephant development version)
        try:
            urlopen(default_url+'/README.md')

        except HTTPError as error:
            # if corresponding elephant-data version is not found,
            # use latest commit of elephant-data
            default_url = url_to_root + f"raw/master"

            warnings.warn(f"No corresponding version of elephant-data found.\n"
                          f"Elephant version: {_get_version()}. "
                          f"Data URL:{error.url}, error: {error}.\n"
                          f"Using elephant-data latest instead (This is "
                          f"expected for elephant development versions).")

        except URLError as error:
            # if verification of SSL certificate fails, do not verify cert
            try:  # try again without certificate verification
                urlopen(default_url + '/README.md',
                        context=ssl._create_unverified_context())
            except HTTPError as http_error:  # e.g. 404:
                default_url = url_to_root + f"raw/master"

            warnings.warn(f"Data URL:{default_url}, error: {error}."
                          f"{error.reason}")

    url = f"{getenv('ELEPHANT_DATA_URL', default_url)}/{repo_path}"

    return download(url, filepath, checksum, verbose)


def unzip(filepath, outdir=ELEPHANT_TMP_DIR, verbose=True):
    with ZipFile(filepath) as zfile:
        zfile.extractall(path=outdir)
    if verbose:
        print(f"Extracted {filepath} to {outdir}")
