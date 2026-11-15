import os
import unittest
from unittest.mock import patch
from evidence_answer import answer

class EvidenceAnswerTests(unittest.TestCase):
    def test_requires_server_side_api_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with self.assertRaisesRegex(RuntimeError, "AI_NOT_CONFIGURED"):
                answer("Does vitiligo respond to herbal oils?", "en")
    def test_rejects_unknown_language(self):
        with self.assertRaises(ValueError):
            answer("Does vitiligo respond to herbal oils?", "xx")
    def test_no_evidence_does_not_fabricate_answer(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only"}):
            with patch("evidence_answer.retrieve", return_value={"query": "rare disorder", "records": []}):
                result = answer("What is known about this rare disorder?", "en")
        self.assertEqual(result["status"], "no_evidence")
        self.assertIsNone(result["answer"])
        self.assertEqual(result["sources"], [])

if __name__ == "__main__":
    unittest.main()
