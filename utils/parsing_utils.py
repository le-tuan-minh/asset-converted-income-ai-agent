"""Tiện ích parse dùng chung: JSON output từ LLM, chuẩn hoá text tiếng Việt."""
from __future__ import annotations
import json
import re
import unicodedata


def parse_json_safe(raw: str) -> dict:
    raw = (raw or "").strip()
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
    if match:
        raw = match.group(1)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        brace_match = re.search(r"\{[\s\S]+\}", raw)
        if brace_match:
            return json.loads(brace_match.group(0))
        raise


def strip_accents(text: str) -> str:
    """Bỏ dấu tiếng Việt + uppercase, giúp keyword matching bền hơn với lỗi OCR dấu."""
    normalized = unicodedata.normalize("NFD", text)
    no_accents = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    no_accents = no_accents.replace("Đ", "D").replace("đ", "d")
    return no_accents.upper()
