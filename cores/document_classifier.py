"""
Document Classifier — phân loại loại giấy tờ từ nội dung text đã OCR/extract.

Chiến lược 2 tầng:
  1. Rule-based keyword matching trên phần đầu văn bản (nhanh, rẻ, không tốn LLM call).
  2. Nếu rule-based không đủ tin cậy (confidence thấp / không match) → fallback
     gọi Groq LLM để phân loại dựa trên nội dung.
"""
from __future__ import annotations
import os
import re

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from schemas import DocumentType
from utils.llm_config import get_llm
from utils.parsing_utils import strip_accents

# Ngưỡng confidence rule-based tối thiểu để KHÔNG cần gọi LLM fallback
RULE_BASED_CONFIDENCE_THRESHOLD = 0.5

# Vùng "tiêu đề" — nơi văn bản chính thức VN luôn nêu rõ loại giấy tờ ngay đầu
# (Quốc hiệu + tên loại giấy tờ). Chỉ dùng để match PRIMARY keywords, tránh việc
# các trường dữ liệu tham chiếu (vd "Số CMND chủ sử dụng: ...") nằm sâu trong thân
# văn bản làm lệch kết quả phân loại.
HEADER_WINDOW_CHARS = 600

# Vùng "toàn văn" (đầu văn bản) dùng để match SECONDARY keywords — tín hiệu yếu,
# chỉ có tác dụng bổ trợ/tie-break, KHÔNG được để lấn át kết quả PRIMARY của loại khác.
CLASSIFY_WINDOW_CHARS = 2000

# Trọng số: khớp cụm từ tiêu đề đầy đủ (PRIMARY) đáng tin hơn nhiều so với khớp
# một từ viết tắt ngắn (SECONDARY) có thể xuất hiện lẫn trong bất kỳ văn bản nào.
_PRIMARY_WEIGHT = 5
_SECONDARY_WEIGHT = 1

# keyword mạnh — chỉ tin khi xuất hiện trong vùng tiêu đề đầu văn bản
_PRIMARY_KEYWORDS: dict[DocumentType, list[str]] = {
    DocumentType.CCCD: [
        "CAN CUOC CONG DAN",
        "THE CAN CUOC",
        "CHUNG MINH NHAN DAN",
    ],
    DocumentType.GCN: [
        "GIAY CHUNG NHAN QUYEN SU DUNG DAT",
        "GIAY CHUNG NHAN QUYEN SO HUU NHA O",
        "QUYEN SU DUNG DAT, QUYEN SO HUU",
    ],
    DocumentType.HOP_DONG_THE_CHAP: [
        "HOP DONG THE CHAP",
        "HOP DONG BAO DAM",
    ],
    DocumentType.XAC_NHAN_THE_CHAP: [
        "XAC NHAN THE CHAP",
        "GIAY XAC NHAN DANG KY THE CHAP",
        "DON YEU CAU DANG KY THE CHAP",
    ],
    DocumentType.VAN_BAN_CHUYEN_NHUONG: [
        "VAN BAN CHUYEN NHUONG",
        "VAN BAN THOA THUAN CHUYEN NHUONG",
    ],
    DocumentType.XAC_NHAN_CHUYEN_NHUONG: [
        "XAC NHAN CHUYEN NHUONG",
        "GIAY XAC NHAN CHUYEN NHUONG",
    ],
    DocumentType.HOP_DONG_MUA_BAN: [
        "HOP DONG MUA BAN",
        "HOP DONG CHUYEN NHUONG QUYEN SU DUNG DAT",
        "HOP DONG TANG CHO",
    ],
}

# keyword yếu — viết tắt/nhãn trường dữ liệu, có thể xuất hiện lồng trong văn bản
# khác loại (vd GCN có ghi "Số CMND chủ sử dụng"), chỉ dùng để bổ trợ/tie-break
_SECONDARY_KEYWORDS: dict[DocumentType, list[str]] = {
    DocumentType.CCCD: ["SO CCCD", "CCCD", "CMND"],
    DocumentType.GCN: ["GCNQSDD", "SO GCN", "SO VAO SO CAP GCN"],
    DocumentType.HOP_DONG_THE_CHAP: ["THE CHAP QUYEN SU DUNG DAT"],
    DocumentType.XAC_NHAN_THE_CHAP: [],
    DocumentType.VAN_BAN_CHUYEN_NHUONG: [],
    DocumentType.XAC_NHAN_CHUYEN_NHUONG: [],
    DocumentType.HOP_DONG_MUA_BAN: [],
}


def classify_rule_based(text: str, filename: str = "") -> tuple[DocumentType, float]:
    """
    Phân loại dựa trên keyword matching có trọng số:
      - PRIMARY (tiêu đề đầy đủ, vd "GIẤY CHỨNG NHẬN QUYỀN SỬ DỤNG ĐẤT") chỉ được tìm
        trong vùng tiêu đề đầu văn bản (HEADER_WINDOW_CHARS) — tín hiệu mạnh.
      - SECONDARY (viết tắt/nhãn trường, vd "CMND", "SO GCN") tìm trong vùng rộng hơn
        (CLASSIFY_WINDOW_CHARS) — tín hiệu yếu, chỉ bổ trợ/tie-break, không được lấn át
        PRIMARY của loại khác (vd GCN có ghi "Số CMND chủ sử dụng" không được khiến nó
        bị phân loại nhầm thành CCCD).

    Trả về (doc_type, confidence) với confidence trong [0, 1].
    """
    header = strip_accents((text or "")[:HEADER_WINDOW_CHARS])
    body = strip_accents((text or "")[:CLASSIFY_WINDOW_CHARS])
    filename_hint = strip_accents(filename).replace(" ", "")

    scores: dict[DocumentType, int] = {}
    for doc_type in DocumentType:
        if doc_type == DocumentType.KHONG_XAC_DINH:
            continue
        primary_hits = sum(1 for kw in _PRIMARY_KEYWORDS.get(doc_type, []) if kw in header)
        secondary_hits = sum(1 for kw in _SECONDARY_KEYWORDS.get(doc_type, []) if kw in body)
        # Tên file trùng khớp cũng cộng thêm tín hiệu yếu (vd: "cccd_kh.jpg")
        all_keywords = _PRIMARY_KEYWORDS.get(doc_type, []) + _SECONDARY_KEYWORDS.get(doc_type, [])
        filename_hits = sum(1 for kw in all_keywords if kw.replace(" ", "") in filename_hint)

        score = (
            primary_hits * _PRIMARY_WEIGHT
            + secondary_hits * _SECONDARY_WEIGHT
            + filename_hits * _SECONDARY_WEIGHT
        )
        if score > 0:
            scores[doc_type] = score

    if not scores:
        return DocumentType.KHONG_XAC_DINH, 0.0

    best_type = max(scores, key=lambda dt: scores[dt])
    confidence = min(1.0, scores[best_type] / _PRIMARY_WEIGHT)
    return best_type, confidence


_CLASSIFY_SYSTEM_PROMPT = """Bạn là hệ thống phân loại giấy tờ ngân hàng Việt Nam.
Nhiệm vụ: đọc đoạn trích văn bản và xác định đây là loại giấy tờ nào trong danh sách sau:
- CCCD: Căn cước công dân / CMTND
- GCN: Giấy chứng nhận quyền sử dụng đất / quyền sở hữu nhà
- HOP_DONG_MUA_BAN: Hợp đồng mua bán / chuyển nhượng quyền sử dụng đất
- VAN_BAN_CHUYEN_NHUONG: Văn bản chuyển nhượng
- XAC_NHAN_CHUYEN_NHUONG: Giấy xác nhận chuyển nhượng
- HOP_DONG_THE_CHAP: Hợp đồng thế chấp
- XAC_NHAN_THE_CHAP: Giấy xác nhận đăng ký thế chấp
- KHONG_XAC_DINH: Không thuộc các loại trên / không đủ thông tin để xác định

CHỈ trả về đúng MỘT từ khóa (một trong các mã ở trên), không giải thích, không markdown.
"""


def classify_llm(text: str, llm: ChatGroq | None = None) -> tuple[DocumentType, float]:
    """
    Fallback: dùng Groq LLM để phân loại khi rule-based không đủ tin cậy.
    Trả về (doc_type, confidence). Nếu không có API key hoặc lỗi gọi API,
    trả về (KHONG_XAC_DINH, 0.0) thay vì raise, để không chặn cả pipeline.
    """
    if not os.getenv("GROQ_API_KEY"):
        return DocumentType.KHONG_XAC_DINH, 0.0

    snippet = (text or "")[:CLASSIFY_WINDOW_CHARS]
    if not snippet.strip():
        return DocumentType.KHONG_XAC_DINH, 0.0

    try:
        llm = llm or get_llm()
        response = llm.invoke([
            SystemMessage(content=_CLASSIFY_SYSTEM_PROMPT),
            HumanMessage(content=f"Đoạn trích văn bản:\n{snippet}"),
        ])
        raw = (response.content or "").strip().upper()
        # Lấy token đầu tiên khớp với 1 trong các mã enum
        match = re.search(r"[A-Z_]+", raw)
        candidate = match.group(0) if match else raw
        for doc_type in DocumentType:
            if doc_type.value == candidate:
                return doc_type, 0.75  # confidence cố định cho kết quả LLM
        return DocumentType.KHONG_XAC_DINH, 0.0
    except Exception as exc:
        print(f"[Classifier] Lỗi gọi LLM classify: {exc}")
        return DocumentType.KHONG_XAC_DINH, 0.0


def classify_document(
    text: str,
    filename: str = "",
    llm: ChatGroq | None = None,
) -> tuple[DocumentType, float, str]:
    """
    Entry point: rule-based trước, fallback LLM nếu confidence thấp.
    Trả về (doc_type, confidence, method) với method ∈ {"rule", "llm"}.
    """
    doc_type, confidence = classify_rule_based(text, filename)
    if confidence >= RULE_BASED_CONFIDENCE_THRESHOLD:
        return doc_type, confidence, "rule"

    llm_doc_type, llm_confidence = classify_llm(text, llm)
    if llm_confidence > 0:
        return llm_doc_type, llm_confidence, "llm"

    # LLM cũng không xác định được → giữ kết quả rule-based (có thể là KHONG_XAC_DINH)
    return doc_type, confidence, "rule"