import os
import unittest
from unittest.mock import patch
from multilingual_search import normalize_query

class SearchNormalizationTests(unittest.TestCase):
    def check(self, text, expected):
        with patch.dict(os.environ, {"ENABLE_MULTILINGUAL_QUERY_REWRITE": "false"}):
            self.assertIn(expected, normalize_query(text)["normalized"])
    def test_vitiligo_arabic(self):
        self.check("ابنتي عندها البهاق في الوجه واليدين وأريد علاج بالنباتات", "vitiligo")
    def test_vitiligo_colloquial(self):
        self.check("بنتي عندها بهاق في وجهها", "vitiligo")
    def test_diabetes_arabic(self):
        self.check("ما هي الدراسات عن السكري النوع الثاني", "diabetes mellitus")
    def test_diabetes_english(self):
        self.check("Diabetic type 2", "diabetes mellitus")
    def test_hebrew(self):
        self.check("הילדה שלי עם ויטיליגו בפנים", "vitiligo")
    def test_german(self):
        self.check("Behandlung von Asthma bei Kindern", "asthma")
    def test_unknown_preserved(self):
        phrase = "نبات له أوراق صغيرة ورائحة قوية لا أعرف اسمه"
        with patch.dict(os.environ, {"ENABLE_MULTILINGUAL_QUERY_REWRITE": "false"}):
            self.assertEqual(normalize_query(phrase)["normalized"], phrase)
    def test_colloquial_bedwetting(self):
        self.check("ابني عمره 4.5 سنوات يعاني من التبول الليلي أثناء النوم حتى اذا نام في النهار يتبول على نفسه، ما الحل وكيف اعالجه", "nocturnal enuresis")
    def test_bedwetting_english(self):
        self.check("My four-year-old child has bedwetting at night", "nocturnal enuresis")
    def test_bedwetting_hebrew(self):
        self.check("הילד סובל מהרטבת לילה", "nocturnal enuresis")
    def test_invalid_length(self):
        with self.assertRaises(ValueError):
            normalize_query("x" * 501)

if __name__ == "__main__":
    unittest.main()
