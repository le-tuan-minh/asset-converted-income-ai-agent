"""
Cấu hình LLM nền tảng (foundation LLM) dùng chung cho toàn bộ pipeline
(document_classifier, asset grouping, asset extraction, web agent...).

Muốn đổi model/provider nền tảng → chỉ sửa ở FILE NÀY, không sửa rải rác
trong từng node.
"""
from __future__ import annotations
import os

from langchain_groq import ChatGroq

# Cho phép override qua biến môi trường LLM_MODEL mà không cần sửa code.
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = 0


def get_llm(**overrides) -> ChatGroq:
    """
    Tạo instance LLM nền tảng. `overrides` cho phép ghi đè tham số (model,
    temperature, ...) cho những lần gọi cần cấu hình riêng.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY chưa được set trong .env")

    params = {
        "model": LLM_MODEL,
        "temperature": LLM_TEMPERATURE,
        "api_key": api_key,
    }
    params.update(overrides)
    return ChatGroq(**params)
