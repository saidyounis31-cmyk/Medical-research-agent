"""Regression tests for scientific abstract formatting and multilingual summaries.
Run: python -m unittest discover -s tests -v
"""
import os
import unittest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException

from app import clean_abstract, article_summary, ArticleSummaryRequest


class AbstractFormattingTests(unittest.TestCase):
    def test_html_headings_become_readable_lines(self):
        text = clean_abstract("Intro<h4>Materials and methods</h4><p>Study &amp; results.</p>")
        self.assertEqual(text, "Intro\nMaterials and methods\nStudy & results.")
        self.assertNotIn("<h4>", text)

    def test_entities_are_decoded(self):
        self.assertEqual(clean_abstract("Sana&lt;sup&gt;a &amp; Yemen"), "Sanaa & Yemen")

    def test_missing_abstract(self):
        self.assertEqual(clean_abstract(None), "")

    def test_no_key_does_not_invent_translation(self):
        req = ArticleSummaryRequest(title="Example study", abstract="A study with sufficient abstract text for a summary request.", language="he")
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with self.assertRaises(HTTPException) as caught:
                article_summary(req)
        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(caught.exception.detail, "AI_NOT_CONFIGURED")

    def test_language_passed_to_summarizer(self):
        req = ArticleSummaryRequest(title="Example study", abstract="A study with sufficient abstract text for a summary request.", language="de")
        mock_response = MagicMock()
        mock_response.output_text = "Zusammenfassung auf Deutsch."
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only"}):
            with patch("app.OpenAI") as client:
                client.return_value.responses.create.return_value = mock_response
                result = article_summary(req)
                arguments = client.return_value.responses.create.call_args.kwargs
        self.assertEqual(result["language"], "de")
        self.assertEqual(result["basis"], "abstract_only")
        self.assertIn("Language: de", arguments["input"])


if __name__ == "__main__":
    unittest.main()
