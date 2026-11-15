"""Safe deterministic multilingual biomedical search expansion.

The lexical fallback supports selected medical concepts, not arbitrary translation.
When a concept is unknown, preserve original words rather than inventing a diagnosis.
"""
import os
import re
import unicodedata
from openai import OpenAI

CONCEPTS = (
    (r"\b(?:بهاق|البهاق|vitiligo|בהקת|ויטיליגו)\b", "vitiligo"),
    (r"\b(?:سكري|السكري|سكر|diabetes|diabetic|סוכרת|diabetes mellitus|diabetes typ)\b", "diabetes mellitus"),
    (r"\b(?:النوع الثاني|نوع ثاني|type 2|typ 2|סוג 2)\b", "type 2"),
    (r"\b(?:روماتويد|الروماتويد|التهاب المفاصل الروماتويدي|rheumatoid arthritis|דלקת מפרקים שגרונית)\b", "rheumatoid arthritis"),
    (r"\b(?:صداع|الصداع|headache|כאב ראש|kopfschmerzen)\b", "headache"),
    (r"\b(?:ربو|الربو|asthma|אסתמה)\b", "asthma"),
    (r"\b(?:زهايمر|الزهايمر|alzheimer|אלצהיימר)\b", "Alzheimer disease"),
    (r"\b(?:باركنسون|الباركنسون|parkinson|פרקינסון)\b", "Parkinson disease"),
    (r"\b(?:حبة البركة|الحبة السوداء|nigella sativa|קצח)\b", "Nigella sativa"),
    (r"\b(?:مورينجا|المورينجا|moringa)\b", "Moringa oleifera"),
    (r"\b(?:زعتر|الزعتر|thyme|thymus vulgaris)\b", "Thymus vulgaris"),
)
INTENTS = (
    (r"(?:طفل|طفلة|ابني|ابنتي|ولدي|بنتي|أطفال|ילד|ילדה|children|child|pediatric|kinder)", "child"),
    (r"(?:علاج|أعالج|دواء|أدوية|معالجة|טיפול|treatment|therapy|behandlung)", "treatment"),
    (r"(?:زيت|زيوت|أعشاب|نباتات|عشبي|צמח|שמנים|herbal|herbs|essential oils|pflanzen)", "herbal medicine"),
    (r"(?:الوجه|وجهي|وجهها|وجهه|פנים|face|gesicht)", "face"),
    (r"(?:اليدين|اليد|يدي|ידיים|hands|hand|hände)", "hands"),
    (r"(?:أعراض|ألم|symptoms|symptom|תסמינים|symptome)", "symptoms"),
)
def _fold(s):
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[\u064b-\u065f\u0670]", "", s)
    s = s.translate(str.maketrans("أإآٱىة", "ااااه"))
    return re.sub(r"\s+", " ", s).strip()

def normalize_query(query):
    query = re.sub(r"\s+", " ", str(query or "")).strip()
    if not 2 <= len(query) <= 500:
        raise ValueError("Search query must contain 2–500 characters")
    if os.getenv("ENABLE_MULTILINGUAL_QUERY_REWRITE", "false").lower() == "true" and os.getenv("OPENAI_API_KEY"):
        try:
            response = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=12).responses.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
                instructions=("Rewrite this Arabic, Hebrew, German or English medical question as concise English "
                              "biomedical search keywords. Do not diagnose. Preserve uncertainty. "
                              "Return only keywords without quotes or Boolean syntax."),
                input=query, max_output_tokens=90)
            normalized = (response.output_text or "").strip().replace("\n", " ")[:300]
            if normalized and re.search(r"[A-Za-z]", normalized):
                return {"original": query, "normalized": normalized, "rewritten": True, "method": "llm"}
        except Exception:
            pass
    folded = _fold(query)
    concepts = []
    for pattern, term in CONCEPTS:
        if re.search(pattern, folded, re.IGNORECASE):
            if term not in concepts:
                concepts.append(term)
    if concepts:
        for pattern, term in INTENTS:
            if re.search(pattern, folded, re.IGNORECASE) and term not in concepts:
                concepts.append(term)
        return {"original": query, "normalized": " ".join(concepts[:5]), "rewritten": True, "method": "lexicon"}
    return {"original": query, "normalized": query, "rewritten": False, "method": "original"}
