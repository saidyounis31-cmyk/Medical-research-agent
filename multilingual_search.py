"""Safe deterministic multilingual biomedical search expansion.

The lexical fallback supports selected medical concepts, not arbitrary translation.
When a concept is unknown, preserve original words rather than inventing a diagnosis.
"""
import os
import re
import unicodedata
from openai import OpenAI

CONCEPTS = (
    (r"(?:بقع حمراء|بقع حمرا|احمرار|طفح جلدي|حساسيه|تحسس|التهاب الجلد|red spots|skin rash|dermatitis|allergic rash|פריחה|אדמומיות|דרמטיטיס|haut ausschlag|hautausschlag|rote flecken)", "skin rash"),
    (r"(?:لسعه|لدغه|قرصه|نحله|بعوضه|حشره|sting|insect bite|bee sting|mosquito bite|עקיצה|דבורה|יתוש|insektenstich|mueckenstich)", "insect bite"),
    (r"(?:تبول|يتبول|بيتبول|بول|التبول|سلس البول|بلل الفراش|تبليل الفراش|تبول لا ارادي|التبول اللاارادي|enuresis|bedwetting|bed wetting|nocturnal incontinence|הרטבת לילה|אנורזיס|bettnassen|einnassen)", "nocturnal enuresis"),

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
    (r"(?:طفل|طفلة|ابني|ابنتي|ولدي|بنتي|اطفال|ילד|ילדה|children|child|pediatric|kinder)", "child"),
    (r"(?:علاج|اعالج|اعالجه|دواء|ادوية|معالجة|حل|טיפול|treatment|therapy|behandlung)", "treatment"),
    (r"(?:زيت|زيوت|اعشاب|نباتات|عشبي|צמח|שמנים|herbal|herbs|essential oils|pflanzen|heilpflanzen|krauter|kräuter)", "herbal medicine"),
    (r"(?:الوجه|وجهي|وجهها|وجهه|פנים|face|gesicht)", "face"),
    (r"(?:اليدين|اليد|يدي|ידיים|hands|hand|hände)", "hands"),
    (r"(?:اعراض|الم|symptoms|symptom|תסמינים|symptome)", "symptoms"),
)
def _fold(s):
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[\u064b-\u065f\u0670]", "", s)
    s = s.translate(str.maketrans("أإآٱىة", "اااااه"))
    return re.sub(r"\s+", " ", s).strip()

def normalize_query(query):
    query = re.sub(r"\s+", " ", str(query or "")).strip()
    if not 2 <= len(query) <= 500:
        raise ValueError("Search query must contain 2–500 characters")
    folded = _fold(query)
    concepts = []
    for pattern, term in CONCEPTS:
        if re.search(pattern, folded, re.IGNORECASE):
            if term not in concepts:
                concepts.append(term)
    if concepts:
        if "skin rash" in concepts:
            return {"original": query, "normalized": '("skin rash" OR dermatitis OR urticaria OR "contact dermatitis" OR "insect bite")', "rewritten": True, "method": "lexicon"}
        if "nocturnal enuresis" in concepts:
            return {"original": query, "normalized": '"nocturnal enuresis" OR bedwetting', "rewritten": True, "method": "lexicon"}
        for pattern, term in INTENTS:
            if re.search(pattern, folded, re.IGNORECASE) and term not in concepts:
                concepts.append(term)
        # Search for the disease first; use interventions for relevance ranking rather than
        # allowing generic intervention terms to dominate Europe PMC retrieval.
        disease = concepts[0]
        return {"original": query, "normalized": f'TITLE_ABS:"{disease}"', "rewritten": True, "method": "lexicon", "concepts": concepts}
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
    return {"original": query, "normalized": query, "rewritten": False, "method": "original"}


def rank_research_results(question, results):
    """Transparent lexical relevance triage, not a clinical quality or efficacy score."""
    folded = _fold(question)
    concepts = [term for pattern, term in CONCEPTS if re.search(pattern, folded, re.IGNORECASE)]
    herb_intent = any(re.search(pattern, folded, re.IGNORECASE) and term == "herbal medicine" for pattern, term in INTENTS)
    pediatric = any(re.search(pattern, folded, re.IGNORECASE) and term == "child" for pattern, term in INTENTS)
    plant_terms = re.compile(r"herbal|herb\\b|botanical|phytotherap|plant extract|medicinal plant|essential oil|traditional chinese medicine|\\btcm\\b|salvia|artemisia|tribulus|nigella|curcumin|plant-derived", re.I)
    evidence_terms = re.compile(r"randomi[sz]ed|clinical trial|controlled trial|systematic review|meta-analysis|patient|participants|human subjects", re.I)
    disease = concepts[0] if concepts else ""
    for item in results:
        title = str(item.get("title") or "")
        abstract = str(item.get("abstract") or "")
        combined = title + " " + abstract
        title_match = bool(disease and disease.lower() in title.lower())
        disease_match = bool(disease and disease.lower() in combined.lower())
        plant_match = bool(plant_terms.search(combined))
        clinical_match = bool(evidence_terms.search(combined))
        score = (4 if title_match else 2 if disease_match else 0)
        if herb_intent:
            score += 4 if plant_match else 0
        if pediatric and re.search(r"child|pediatric|adolescent", combined, re.I):
            score += 1
        if clinical_match:
            score += 1
        # Do not call a generic dermatology review direct treatment evidence.
        if disease and not disease_match:
            level = "low"
        elif herb_intent and not plant_match:
            level = "low"
        elif herb_intent and disease_match and plant_match:
            level = "direct_topic"
        else:
            level = "background"
        item["relevance"] = level
        item["relevance_score"] = score
        item["relevance_reason"] = ("Disease and botanical intervention mentioned; clinical efficacy not established"
                                    if level == "direct_topic" else
                                    "Disease context only; not direct evidence of requested intervention"
                                    if level == "background" else
                                    "Insufficient matching terms for the requested disease and intervention")
    return sorted(results, key=lambda x: x["relevance_score"], reverse=True)
