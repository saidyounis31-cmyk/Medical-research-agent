"""Regression checks for readable Europe PMC abstracts and safe summary behavior.

Run: python -m unittest discover -s tests -p 'test_publication_display.py'
"""
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from app import app, clean_abstract


class PublicationDisplayTests(unittest.TestCase):
    def test_html_headings_and_entities_are_readable(self):
        source = "<h4>Materials and methods</h4><p>Children &amp; adults <b>included</b>.</p>"
        clean = clean_abstract(source)
        self.assertIn("Materials and methods", clean)
        self.assertIn("Children & adults included.", clean)
        self.assertNotIn("<h4>", clean)
        self.assertNotIn("</h4>", clean)

    def test_no_untrusted_html_markup(self):
        clean = clean_abstract('<p>Results</p><script>alert(1)</script><i>safe</i>')
        self.assertNotIn("<script>", clean)
        self.assertNotIn("<i>", clean)

    def test_summary_fails_explicitly_without_ai_key(self):
        client = TestClient(app)
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            response = client.post("/api/article-summary", json={
                "title": "A clinical research article",
                "abstract": "This clinical study evaluated an intervention in a defined population and reported its methods and outcomes.",
                "language": "ar"
            })
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "AI_NOT_CONFIGURED")

    def test_summary_rejects_unsupported_language(self):
        client = TestClient(app)
        response = client.post("/api/article-summary", json={
            "title": "A research article",
            "abstract": "A sufficiently long example abstract describing a study of clinical research.",
            "language": "xx"
        })
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
