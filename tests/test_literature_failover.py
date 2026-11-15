"""Regression checks for upstream Europe PMC failure handling.

Run: python -m unittest discover -s tests -p 'test_literature_failover.py'
"""
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
import app


class FakeResponse:
    def __enter__(self):
        return self
    def __exit__(self, *_):
        return False
    def read(self):
        return b'{"hitCount": 0, "resultList": {"result": []}}'


class LiteratureFailoverTests(unittest.TestCase):
    def test_retries_after_temporary_503(self):
        temporary = HTTPError("https://example.org", 503, "unavailable", {}, None)
        with patch.object(app, "urlopen", side_effect=[temporary, FakeResponse()]) as call:
            data = app.fetch_europe_pmc("query=enuresis&format=json")
        self.assertEqual(data["hitCount"], 0)
        self.assertEqual(call.call_count, 2)

    def test_permanent_bad_request_not_retried(self):
        invalid = HTTPError("https://example.org", 400, "bad query", {}, None)
        with patch.object(app, "urlopen", side_effect=invalid) as call:
            with self.assertRaises(HTTPError):
                app.fetch_europe_pmc("query=bad")
        self.assertEqual(call.call_count, 1)

    def test_persistent_503_does_not_become_no_results(self):
        temporary = HTTPError("https://example.org", 503, "unavailable", {}, None)
        with patch.object(app, "urlopen", side_effect=temporary):
            with patch.object(app.time, "sleep"):
                with self.assertRaises(HTTPError):
                    app.fetch_europe_pmc("query=enuresis")


if __name__ == "__main__":
    unittest.main()
