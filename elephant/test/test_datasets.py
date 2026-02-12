import unittest
import os
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
import hashlib
import urllib

from numpy.testing import assert_allclose
from neo import NixIO
from elephant.datasets import download_datasets, load_data, ELEPHANT_DATA


# Most tests are done with mocking the environment variable to test different
# scenarios. However, in CI tests, ELEPHANT_DATA_LOCATION is set to a local
# cache path of the "elephant-data" repository, which is expected to be a
# folder. We use this for additional tests with the cache directory.
# HAS_CACHE_DIR is used as a flag to conditionally run tests that require the
# cache directory.
HAS_CACHE_DIR = ("ELEPHANT_DATA_LOCATION" in os.environ
                 and Path(os.environ["ELEPHANT_DATA_LOCATION"]).is_dir())


class TestDownloadDatasets(unittest.TestCase):

    @patch.dict(os.environ, {'ELEPHANT_DATA_LOCATION': '/invalid/path'},
                clear=True)
    def test_invalid_path(self):
        repo_path = 'some/repo/path'
        with self.assertRaises(ValueError) as error:
            download_datasets(repo_path)
        exception_msg = str(error.exception)
        self.assertIn("ELEPHANT_DATA_LOCATION must be set to either",
                      exception_msg)
        self.assertIn("/invalid/path", exception_msg)

    @patch.dict(os.environ, {'ELEPHANT_DATA_LOCATION': 'ftp://invalid.url'},
                clear=True)
    def test_invalid_url(self):
        repo_path = 'some/repo/path'
        with self.assertRaises(ValueError) as error:
            download_datasets(repo_path)
        exception_msg = str(error.exception)
        self.assertIn("ELEPHANT_DATA_LOCATION must be set to either",
                      exception_msg)
        self.assertIn("ftp://invalid.url", exception_msg)

    @patch.dict(os.environ, {'ELEPHANT_DATA_LOCATION': 'http://valid.url'},
                clear=True)
    def test_valid_url(self):
        repo_path = 'some/repo/path'
        self.assertRaises(urllib.error.URLError, download_datasets, repo_path)

    @patch.dict(os.environ, {'ELEPHANT_DATA_LOCATION': ''}, clear=True)
    def test_invalid_data(self):
        repo_path = 'some/repo/path'
        self.assertRaises(urllib.error.URLError, download_datasets, repo_path)

    @patch.dict(os.environ, {'ELEPHANT_DATA_LOCATION': 'invalid_path_or_url'},
                clear=True)
    def test_invalid_value(self):
        repo_path = 'some/repo/path'
        with self.assertRaises(ValueError) as error:
            download_datasets(repo_path)
        exception_msg = str(error.exception)
        self.assertIn("ELEPHANT_DATA_LOCATION must be set to either",
                      exception_msg)
        self.assertIn("invalid_path_or_url", exception_msg)

    @patch.dict(os.environ, {'ELEPHANT_DATA_LOCATION': ''},
                clear=True)
    def test_valid_data(self):
        # This is expected to download a file with the same name in GIN to a
        # folder 'elephant' within the current system temporary folder
        repo_path = 'dataset-1/dataset-1.h5'

        # Create a dummy file path to simulate the downloaded dataset
        dummy_file_path = Path(gettempdir()) / 'elephant' / 'dataset-1.h5'

        downloaded_file = download_datasets(repo_path,
                                            filepath=None,
                                            checksum=None)
        self.assertTrue(Path(downloaded_file).is_file())
        self.assertEqual(dummy_file_path, downloaded_file)

    @patch.dict(os.environ, {'ELEPHANT_DATA_LOCATION': ''},
                clear=True)
    def test_valid_data_with_path(self):
        # This is expected to download a file to the provided path
        # (using a temporary directory specific to the test)
        repo_path = 'dataset-1/dataset-1.h5'
        with TemporaryDirectory() as temp_dir:
            # Create a dummy file to store the downloaded dataset
            dummy_file_path = Path(temp_dir) / 'dummy'
            downloaded_file = download_datasets(repo_path,
                                                filepath=dummy_file_path,
                                                checksum=None)
            self.assertTrue(Path(downloaded_file).is_file())
            self.assertEqual(dummy_file_path, downloaded_file)

    @patch.dict(os.environ, {'ELEPHANT_DATA_LOCATION': ''},
                clear=True)
    def test_valid_data_with_integrity_check(self):
        # This is expected to check the integrity of the downloaded file by
        # comparing the MD5 hash. We manually download the file to the current
        # system temporary folder to compute the expected checksum, and then
        # we download to a different file in a temporary folder to test the
        # function.
        repo_path = 'dataset-1/dataset-1.h5'

        # Download a copy to compute the checksum
        expected_dataset = download_datasets(repo_path)
        expected_checksum = hashlib.md5(
            open(expected_dataset, 'rb').read()
        ).hexdigest()

        with TemporaryDirectory() as temp_dir:
            # Create a dummy file to simulate the downloaded dataset
            dummy_file_path = Path(temp_dir) / 'dummy_checksum'
            downloaded_file = download_datasets(repo_path,
                                                filepath=dummy_file_path,
                                                checksum=expected_checksum)
            self.assertTrue(Path(downloaded_file).is_file())
            self.assertEqual(dummy_file_path, downloaded_file)

    def test_valid_data_with_failed_integrity_check(self):
        # This forces a failure by setting an invalid checksum for a dataset
        # in GIN.
        repo_path = 'dataset-1/dataset-1.h5'
        with TemporaryDirectory() as temp_dir:
            # Create a dummy file to store the downloaded dataset
            dummy_file_path = Path(temp_dir) / 'dummy_checksum_fail'
            with self.assertRaises(ValueError) as error:
                download_datasets(repo_path,
                                  filepath=dummy_file_path,
                                  checksum="aaaaaa")
            exception_msg = str(error.exception)
            self.assertIn(repo_path, exception_msg)
            self.assertIn("does not agree with MD5 hash aaaaaa", exception_msg)

    @unittest.skipIf(not HAS_CACHE_DIR,
                     "Not testing since ELEPHANT_DATA_LOCATION is not set to "
                     "a folder")
    def test_download_with_cache_dir(self):
        # This test is expected in the GitHub Actions environment where the
        # variable is set to a local cache path. This tests that the function
        # correctly uses the cache directory and returns the expected file
        # path without re-downloading.
        repo_path = 'dataset-1/dataset-1.h5'

        # Expected path of the downloaded dataset
        expected_file_path = (
                Path(os.environ['ELEPHANT_DATA_LOCATION']) /
                "dataset-1" / "dataset-1.h5")

        downloaded_file = download_datasets(repo_path,
                                            filepath=None,
                                            checksum=None)
        self.assertTrue(Path(downloaded_file).is_file())
        self.assertEqual(expected_file_path, downloaded_file)

    @unittest.skipIf(not HAS_CACHE_DIR,
                     "Not testing since ELEPHANT_DATA_LOCATION is not set to "
                     "a folder")
    def test_download_with_cache_dir_target_path(self):
        # This test is expected in the GitHub Actions environment where the
        # variable is set to a local cache path. This tests if the function
        # uses the cache directory and correctly copies the file to the
        # provided target path.
        repo_path = 'dataset-1/dataset-1.h5'
        with TemporaryDirectory() as temp_dir:
            target_file_path = Path(temp_dir) / 'target_cache'
            downloaded_file = download_datasets(repo_path,
                                                filepath=target_file_path,
                                                checksum=None)
            self.assertTrue(Path(downloaded_file).is_file())
            self.assertEqual(target_file_path, downloaded_file)

    @unittest.skipIf(not HAS_CACHE_DIR,
                     "Not testing since ELEPHANT_DATA_LOCATION is not set to "
                     "a folder")
    def test_download_with_cache_dir_invalid_file(self):
        # This test is expected in the GitHub Actions environment where the
        # variable is set to a local cache path. This test the behavior when
        # usinga cache dir and requesting a file that does not exist.
        repo_path = 'dataset-1/not-existent'
        with self.assertRaises(ValueError) as error:
            download_datasets(repo_path)
        exception_msg = str(error.exception)
        self.assertIn("ELEPHANT_DATA_LOCATION is set to the local path",
                      exception_msg)
        self.assertIn(f"'{repo_path}' does not exist in that path",
                      exception_msg)

class TestLoadData(unittest.TestCase):

    def test_load_data_invalid(self):
        with self.assertRaises(ValueError) as error:
            load_data('invalid_dataset')
        exception_msg = str(error.exception)
        self.assertIn("not available as downloadable datasets or "
                      "generated data", exception_msg)
        self.assertIn("invalid_dataset", exception_msg)

    def test_asset(self):
        # Load the Segment with ASSET data using the interface function
        asset_data = load_data('asset')

        # Manually download and load the Segment with data
        # Do not use checksums to detect changes in the files
        asset_repo_path = ELEPHANT_DATA['asset']['repo_path']
        download_file = download_datasets(asset_repo_path)
        downloaded_asset_block = NixIO(str(download_file)).read_block()
        downloaded_asset_data = downloaded_asset_block.segments[0]

        # 500 spike trains are expected
        self.assertEqual(len(asset_data.spiketrains), 500)

        # Compare spike times
        for st1, st2 in zip(asset_data.spiketrains,
                            downloaded_asset_data.spiketrains):
            assert_allclose(st1.magnitude, st2.magnitude, atol=1e-8)

        # Compare annotations
        for annotation in ('nix_name', 'spiketrain_ordering'):
            self.assertEqual(downloaded_asset_data.annotations[annotation],
                             asset_data.annotations[annotation])


if __name__ == '__main__':
    unittest.main()
