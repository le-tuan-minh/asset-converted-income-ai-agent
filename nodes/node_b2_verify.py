"""
B2 - Verify Node (Reasoning AI)
Gửi OCR raw text lên Groq LLM để:
  - Extract thông tin chủ tài sản từ CCCD, GCN, Hợp đồng
  - So khớp chủ sở hữu
  - Xác định tặng cho / thừa kế
  - Xác định ngày hình thành tài sản
  - Phân loại mục đích sử dụng đất và diện tích
"""
from __future__ import annotations
import json
import os
import re

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from schemas import (
    GraphState, OwnerInfo, AssetInfo,
    IdentityCheckResult, LandPurposeResult,
)

# ─── LLM setup ───────────────────────────────
def _get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY chưa được set trong .env")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=api_key,
    )


SYSTEM_PROMPT = """Bạn là chuyên gia thẩm định tín dụng ngân hàng Việt Nam.
Nhiệm vụ: phân tích văn bản OCR từ hồ sơ vay vốn và trích xuất thông tin chính xác.
Luôn trả về JSON hợp lệ, không có markdown, không có giải thích thêm.
Với các trường không tìm thấy trong văn bản, để chuỗi rỗng "".
"""

EXTRACT_PROMPT_TEMPLATE = """Dưới đây là nội dung OCR từ 3 tài liệu:

=== CCCD (Căn cước công dân) ===
{cccd_text}

=== GCN (Giấy chứng nhận quyền sử dụng đất) ===
{gcn_text}

=== Hợp đồng mua bán ===
{hop_dong_text}

Hãy trích xuất và trả về JSON với cấu trúc sau:

{{
  "owner_info": {{
    "ho_ten": "Họ tên đầy đủ trên CCCD",
    "so_cccd": "Số CCCD 12 chữ số",
    "so_cmtnd_cu": "Số CMTND 9 chữ số nếu có",
    "ngay_sinh": "DD/MM/YYYY",
    "dia_chi_thuong_tru": "Địa chỉ thường trú"
  }},
  "asset_info": {{
    "so_gcn": "Số seri/mã GCN",
    "chu_su_dung": "Họ tên chủ sử dụng trên GCN",
    "ngay_cap_gcn": "Ngày cấp GCN (DD/MM/YYYY)",
    "ngay_chuyen_nhuong": "Ngày chuyển nhượng trong hợp đồng (DD/MM/YYYY)",
    "muc_dich_su_dung": "Đất ở / Nhà ở / Đất nông nghiệp / TMDV",
    "dien_tich_tong": "Tổng diện tích m2",
    "dien_tich_dat_o": "Diện tích đất ở m2",
    "dien_tich_nha_o": "Diện tích nhà ở m2",
    "dien_tich_nn": "Diện tích nông nghiệp m2",
    "dien_tich_tmdv": "Diện tích thương mại dịch vụ m2",
    "co_thong_tin_tang_cho": false,
    "thuoc_du_an": null,
    "nguon_goc_tai_san": "Mô tả nguồn gốc: mua bán / được cấp / tặng cho / thừa kế"
  }},
  "identity_check": {{
    "owner_matched": true,
    "mismatch_fields": ["danh sách trường không khớp nếu có"],
    "is_tang_cho": false,
    "is_thua_ke": false,
    "asset_formation_date": "Ngày hình thành tài sản (ưu tiên ngày cấp GCN, nếu có chuyển nhượng thì dùng ngày chuyển nhượng)",
    "asset_formation_note": "Ghi chú về thời điểm hình thành"
  }},
  "land_purpose": {{
    "muc_dich": "Đất ở / Nhà ở / Đất nông nghiệp / TMDV",
    "dien_tich_du_dieu_kien": "Diện tích đủ điều kiện quy đổi (chỉ tính đất ở + nhà ở)",
    "is_tmdv": false,
    "thuoc_du_an": null,
    "warning_tmdv": "Cảnh báo nếu là TMDV không thuộc dự án"
  }}
}}

Lưu ý quan trọng:
- owner_matched = true nếu họ tên / số CCCD trên GCN hoặc hợp đồng TRÙNG với CCCD
- is_tang_cho = true nếu trong GCN hoặc hợp đồng có chữ "tặng cho", "cho tặng", "thừa kế", "di chúc"
- dien_tich_du_dieu_kien chỉ bao gồm đất ở + nhà ở, KHÔNG tính nông nghiệp và TMDV
- Nếu TMDV: kiểm tra có từ "dự án", "khu đô thị", "khu công nghiệp" không để xác định thuoc_du_an
"""


def _parse_json_safe(raw: str) -> dict:
    """Extract JSON từ response LLM, bỏ qua markdown fences."""
    raw = raw.strip()
    # Bỏ ```json ... ``` nếu có
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
    if match:
        raw = match.group(1)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Thử extract phần {} đầu tiên
        brace_match = re.search(r"\{[\s\S]+\}", raw)
        if brace_match:
            return json.loads(brace_match.group(0))
        raise


def node_b2_verify(state: GraphState) -> GraphState:
    """
    LangGraph node B2.
    Gửi OCR text lên Groq LLM, nhận lại structured JSON,
    parse thành domain models và lưu vào state.
    """
    print("\n" + "="*60)
    print("B2 · VERIFY NODE — Gửi OCR text lên Groq LLM")
    print("="*60)

    notes = list(state.processing_notes)
    flags = list(state.flags)

    if not state.ocr_raw.cccd_text and not state.ocr_raw.gcn_text:
        notes.append("B2 bỏ qua: không có OCR text đầu vào.")
        print("[B2] Không có OCR text, bỏ qua.")
        return state.model_copy(update={"processing_notes": notes})

    # Cắt ngắn text nếu quá dài (Groq có context limit)
    def truncate(text: str, max_chars: int = 4000) -> str:
        return text[:max_chars] + "\n...[đã cắt ngắn]" if len(text) > max_chars else text

    prompt_text = EXTRACT_PROMPT_TEMPLATE.format(
        cccd_text=truncate(state.ocr_raw.cccd_text),
        gcn_text=truncate(state.ocr_raw.gcn_text),
        hop_dong_text=truncate(state.ocr_raw.hop_dong_text),
    )

    llm = _get_llm()
    print("[B2] Đang gọi Groq LLM...")
    try:
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt_text),
        ])
        raw_output = response.content
        print(f"[B2] Nhận response từ Groq ({len(raw_output)} ký tự)")
    except Exception as exc:
        error_msg = f"[B2] Lỗi gọi Groq API: {exc}"
        print(error_msg)
        notes.append(error_msg)
        return state.model_copy(update={
            "processing_notes": notes,
            "error": str(exc),
        })

    # Parse JSON response
    try:
        data = _parse_json_safe(raw_output)
    except Exception as exc:
        error_msg = f"[B2] Không parse được JSON từ LLM: {exc}\nRaw: {raw_output[:500]}"
        print(error_msg)
        notes.append(error_msg)
        return state.model_copy(update={
            "processing_notes": notes,
            "error": str(exc),
        })

    # Map vào domain models
    owner_info      = OwnerInfo(**data.get("owner_info", {}))
    asset_info      = AssetInfo(**data.get("asset_info", {}))
    identity_check  = IdentityCheckResult(**data.get("identity_check", {}))
    land_purpose    = LandPurposeResult(**data.get("land_purpose", {}))

    notes.append("B2 hoàn thành: LLM extract thành công.")
    print(f"[B2] owner_matched={identity_check.owner_matched} | muc_dich={land_purpose.muc_dich}")
    print("[B2] Hoàn thành.\n")

    return state.model_copy(update={
        "owner_info": owner_info,
        "asset_info": asset_info,
        "identity_check": identity_check,
        "land_purpose": land_purpose,
        "flags": flags,
        "processing_notes": notes,
    })