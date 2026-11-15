import os
import unittest
from unittest.mock import patch
from multilingual_search import normalize_query

class MultilingualSearchTests(unittest.TestCase):
    def test_full_sentence_preserved_when_disabled(self):
        with patch.dict(os.environ, {"ENABLE_MULTILINGUAL_QUERY_REWRITE": "false"}):
            q = "عندي وجع بأعصاب إيدي من فترة شو الأسباب"
            self.assertEqual(normalize_query(q)["normalized"], q)
    def test_hebrew_phrase_preserved_when_disabled(self):
        with patch.dict(os.environ, {"ENABLE_MULTILINGUAL_QUERY_REWRITE": "false"}):
            self.assertEqual(normalize_query("יש לי כאבים בעצבים"), {"original":"יש לי כאבים בעצבים","normalized":"יש לי כאבים בעצבים","rewritten":False})
    def test_no_key_safe_fallback(self):
        with patch.dict(os.environ, {"ENABLE_MULTILINGUAL_QUERY_REWRITE":"true","OPENAI_API_KEY":""}):
            self.assertFalse(normalize_query("Was hilft gegen Nervenschmerzen?")["rewritten"])
    def test_reject_oversized_query(self):
        with self.assertRaises(ValueError):
            normalize_query("a"*501)

if __name__ == "__main__":
    unittest.main()
