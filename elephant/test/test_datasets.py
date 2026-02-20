import unittest
import os
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
import hashlib
from urllib.error import URLError

import numpy as np
from numpy.testing import assert_allclose
import neo
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
        self.assertRaises(URLError, download_datasets, repo_path)

    @patch.dict(os.environ, {'ELEPHANT_DATA_LOCATION': ''}, clear=True)
    def test_invalid_data(self):
        repo_path = 'some/repo/path'
        self.assertRaises(URLError, download_datasets, repo_path)

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

        # Expected path of the downloaded dataset
        expected_file_path = Path(gettempdir()) / 'elephant' / 'dataset-1.h5'

        downloaded_file = download_datasets(repo_path,
                                            filepath=None,
                                            checksum=None)
        self.assertTrue(Path(downloaded_file).is_file())
        self.assertEqual(expected_file_path, downloaded_file)

    @patch.dict(os.environ, {'ELEPHANT_DATA_LOCATION': ''},
                clear=True)
    def test_valid_data_with_path(self):
        # This is expected to download a file to the provided path
        # (using a temporary directory specific to the test)
        repo_path = 'dataset-1/dataset-1.h5'
        with TemporaryDirectory() as temp_dir:
            target_file_path = Path(temp_dir) / 'target'
            downloaded_file = download_datasets(repo_path,
                                                filepath=target_file_path,
                                                checksum=None)
            self.assertTrue(Path(downloaded_file).is_file())
            self.assertEqual(target_file_path, downloaded_file)

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
            # Download the dataset to a different file in the system
            # (into temporary directory specific to the test)
            target_file_path = Path(temp_dir) / 'target_checksum'
            downloaded_file = download_datasets(repo_path,
                                                filepath=target_file_path,
                                                checksum=expected_checksum)
            self.assertTrue(Path(downloaded_file).is_file())
            self.assertEqual(target_file_path, downloaded_file)

    @patch.dict(os.environ, {'ELEPHANT_DATA_LOCATION': ''},
                clear=True)
    def test_valid_data_existing(self):
        # This is expected to avoid downloading the file two times, since
        # the file has been previously downloaded and a checksum is provided.
        repo_path = 'dataset-1/dataset-1.h5'

        with TemporaryDirectory() as temp_dir:
            # Download the dataset to a file in the system
            # (into temporary directory specific to the test)
            target_file_path = Path(temp_dir) / 'target_existing'
            downloaded_file = download_datasets(repo_path,
                                                filepath=target_file_path,
                                                checksum=None)
            expected_checksum = hashlib.md5(
                open(downloaded_file, 'rb').read()).hexdigest()

            existing_file = download_datasets(repo_path,
                                              filepath=target_file_path,
                                              checksum=expected_checksum)
            self.assertTrue(Path(existing_file).is_file())
            self.assertEqual(target_file_path, existing_file)
            self.assertEqual(downloaded_file, existing_file)

    @patch.dict(os.environ, {'ELEPHANT_DATA_LOCATION': ''},
                clear=True)
    def test_valid_data_with_failed_integrity_check(self):
        # This forces a failure by setting an invalid checksum for a dataset
        # in GIN.
        repo_path = 'dataset-1/dataset-1.h5'
        with TemporaryDirectory() as temp_dir:
            # Download dataset to a temporary directory
            target_file_path = Path(temp_dir) / 'target_checksum_fail'
            with self.assertRaises(ValueError) as error:
                download_datasets(repo_path,
                                  filepath=target_file_path,
                                  checksum="aaaaaa")
            exception_msg = str(error.exception)
            self.assertIn(repo_path, exception_msg)
            self.assertIn("Data at", exception_msg)
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
    def test_download_with_cache_dir_with_integrity_check(self):
        # This test is expected in the GitHub Actions environment where the
        # variable is set to a local cache path. This tests that the function
        # correctly uses the cache directory and returns the expected file
        # path without re-downloading while performing the integrity check.
        repo_path = 'dataset-1/dataset-1.h5'

        # Expected path of the downloaded dataset
        expected_file_path = (
                Path(os.environ['ELEPHANT_DATA_LOCATION']) /
                "dataset-1" / "dataset-1.h5")

        expected_checksum = hashlib.md5(
            open(expected_file_path, 'rb').read()).hexdigest()

        downloaded_file = download_datasets(repo_path,
                                            filepath=None,
                                            checksum=expected_checksum)
        self.assertTrue(Path(downloaded_file).is_file())
        self.assertEqual(expected_file_path, downloaded_file)


    @unittest.skipIf(not HAS_CACHE_DIR,
                     "Not testing since ELEPHANT_DATA_LOCATION is not set to "
                     "a folder")
    def test_download_with_cache_dir_with_failed_integrity_check(self):
        # This test is expected in the GitHub Actions environment where the
        # variable is set to a local cache path. This forces a failure by
        # setting an invalid checksum for a dataset in a local cache folder.
        repo_path = 'dataset-1/dataset-1.h5'
        with TemporaryDirectory() as temp_dir:
            # Download dataset to a temporary directory
            target_file_path = Path(temp_dir) / 'target_checksum_fail'
            with self.assertRaises(ValueError) as error:
                download_datasets(repo_path,
                                  filepath=target_file_path,
                                  checksum="aaaaaa")
            exception_msg = str(error.exception)
            self.assertIn(repo_path, exception_msg)
            self.assertIn("Local file at", exception_msg)
            self.assertIn("does not agree with MD5 hash aaaaaa", exception_msg)

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
    def test_download_with_cache_dir_target_path_with_integrity_check(self):
        # This test is expected in the GitHub Actions environment where the
        # variable is set to a local cache path. This tests if the function
        # uses the cache directory and correctly copies the file to the
        # provided target path, while checking the file integrity.
        repo_path = 'dataset-1/dataset-1.h5'

        # Expected checksum of the dataset
        expected_file_path = (
                Path(os.environ['ELEPHANT_DATA_LOCATION']) /
                "dataset-1" / "dataset-1.h5")
        expected_checksum = hashlib.md5(
            open(expected_file_path, 'rb').read()).hexdigest()

        with TemporaryDirectory() as temp_dir:
            target_file_path = Path(temp_dir) / 'target_cache'
            downloaded_file = download_datasets(repo_path,
                                                filepath=target_file_path,
                                                checksum=expected_checksum)
            self.assertTrue(Path(downloaded_file).is_file())
            self.assertEqual(target_file_path, downloaded_file)

    @unittest.skipIf(not HAS_CACHE_DIR,
                     "Not testing since ELEPHANT_DATA_LOCATION is not set to "
                     "a folder")
    def test_download_with_cache_dir_invalid_file(self):
        # This test is expected in the GitHub Actions environment where the
        # variable is set to a local cache path. This test the behavior when
        # using a cache dir and requesting a file that does not exist.
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
        # Test to validate the ASSET dataset loading and integrity.

        # Load the Segment with ASSET data using the interface function
        asset_data = load_data('asset')
        self.assertIsInstance(asset_data, neo.Segment)

        # 500 spike trains are expected
        self.assertEqual(len(asset_data.spiketrains), 500)

        # Manually download and load the Segment with data
        asset_repo_path = ELEPHANT_DATA['asset']['repo_path']
        download_file = download_datasets(asset_repo_path)
        downloaded_asset_block = neo.NixIO(str(download_file)).read_block()
        downloaded_asset_data = downloaded_asset_block.segments[0]

        # Compare spike times
        for load_st, expected_st in zip(asset_data.spiketrains,
                                        downloaded_asset_data.spiketrains):
            assert_allclose(load_st.magnitude, expected_st.magnitude,
                            atol=1e-8)

        # Compare annotations
        for annotation in ('nix_name', 'spiketrain_ordering'):
            self.assertEqual(downloaded_asset_data.annotations[annotation],
                             asset_data.annotations[annotation])

    def test_unitary_events(self):
        # Test to validate the Unitary Events dataset loading and integrity.

        # Load the spike trains for the Unitary Events tutorial using the
        # interface function
        ue_data = load_data('unitary_events')

        # 36 trials with 2 spike trains are expected
        self.assertIsInstance(ue_data, list)
        self.assertEqual(len(ue_data), 36)
        self.assertTrue(all(len(sts) == 2 for sts in ue_data))
        self.assertTrue(all(all(isinstance(st, neo.SpikeTrain) for st in sts)
                            for sts in ue_data))

        # Manually download and load the spike trains
        ue_repo_path = ELEPHANT_DATA['unitary_events']['repo_path']
        download_file = download_datasets(ue_repo_path)
        downloaded_ue_block = neo.NixIO(str(download_file)).read_block()
        downloaded_ue_data = [[st for st in segment.spiketrains]
                               for segment in downloaded_ue_block.segments]

        # Compare spike times and annotations
        for load_trial, expected_trial in zip(ue_data, downloaded_ue_data):
            for load_st, download_st in zip(load_trial, expected_trial):
                assert_allclose(load_st.magnitude, load_st.magnitude,
                                atol=1e-8)
                self.assertDictEqual(load_st.annotations,
                                     download_st.annotations)

    def test_granger_causality_indirect(self):
        # Test to validate the generation of the Granger Causality tutorial
        # data with only indirect causality.
        granger_indirect = load_data('granger_causality_indirect')

        # Validate type and shape of the data (10000 samples, 3 time series)
        self.assertEqual(granger_indirect.shape, (10000, 3))
        self.assertIsInstance(granger_indirect, np.ndarray)

    def test_granger_causality_both(self):
        # Test to validate the generation of the Granger Causality tutorial
        # data with both direct and indirect causality.
        granger_both = load_data('granger_causality_both')

        # Validate type and shape of the data (10000 samples, 3 time series)
        self.assertEqual(granger_both.shape, (10000, 3))
        self.assertIsInstance(granger_both, np.ndarray)


if __name__ == '__main__':
    unittest.main()
