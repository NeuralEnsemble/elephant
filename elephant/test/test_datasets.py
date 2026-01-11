import unittest
import os
from unittest.mock import patch
from pathlib import Path
import urllib

from elephant.datasets import download_datasets


class TestDownloadDatasets(unittest.TestCase):
    @patch.dict(os.environ, {"ELEPHANT_DATA_LOCATION": "/valid/path"}, clear=True)
    @patch("os.path.exists", return_value=True)
    def test_valid_path(self, mock_exists):
        repo_path = "some/repo/path"
        expected = Path("/valid/path/some/repo/path")
        result = download_datasets(repo_path)
        self.assertEqual(result, expected)

    @patch.dict(os.environ, {"ELEPHANT_DATA_LOCATION": "http://valid.url"}, clear=True)
    @patch("os.path.exists", return_value=False)
    def test_valid_url(self, mock_exists):
        repo_path = "some/repo/path"
        self.assertRaises(urllib.error.URLError, download_datasets, repo_path)

    @patch.dict(
        os.environ, {"ELEPHANT_DATA_LOCATION": "invalid_path_or_url"}, clear=True
    )
    @patch("os.path.exists", return_value=False)
    def test_invalid_value(self, mock_exists):
        repo_path = "some/repo/path"
        with self.assertRaises(ValueError) as cm:
            download_datasets(repo_path)
        self.assertIn("invalid_path_or_url", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
