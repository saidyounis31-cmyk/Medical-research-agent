"""Normalize multilingual colloquial and long medical search phrases before Europe PMC lookup.
This module never claims translation when the LLM is unavailable.
"""
import os
import re
from openai import OpenAI

def normalize_query(query):
    query = re.sub(r"\s+", " ", str(query or "")).strip()
    if len(query) < 2 or len(query) > 500:
        raise ValueError("Search query must contain 2–500 characters")
    if os.getenv("ENABLE_MULTILINGUAL_QUERY_REWRITE", "false").lower() != "true":
        return {"original": query, "normalized": query, "rewritten": False}
    if not os.getenv("OPENAI_API_KEY"):
        return {"original": query, "normalized": query, "rewritten": False}
    try:
        response = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=12).responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            instructions=("Convert this search phrase, which may be colloquial Arabic, Hebrew, German or English, "
                          "into a concise English biomedical literature search query. Preserve the medical meaning, "
                          "drug names, and uncertainty. Do not answer the question. Output only keywords or a short "
                          "search phrase, no quotes, no Boolean operators or commentary."),
            input=query, max_output_tokens=90)
        normalized = (response.output_text or "").strip().replace("\n", " ")[:300]
        if not normalized or not re.search(r"[A-Za-z]", normalized):
            return {"original": query, "normalized": query, "rewritten": False}
        return {"original": query, "normalized": normalized, "rewritten": True}
    except Exception:
        return {"original": query, "normalized": query, "rewritten": False}
