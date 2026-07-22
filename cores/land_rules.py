"""
Land Rules — bộ quy tắc rule-based dùng làm lưới an toàn (safety net) cho việc
LLM phân loại đất Thương mại - Dịch vụ (TMDV) ở node B2.

Căn cứ pháp lý:
  - Điều 9 Luật Đất đai 2024: phân loại đất theo mục đích sử dụng
    (nhóm đất nông nghiệp / phi nông nghiệp / chưa sử dụng).
  - Thông tư 08/2024/TT-BTNMT (hiệu lực 01/08/2024), Phụ lục II - Mục A:
    quy định mã ký hiệu từng loại đất trên bản đồ địa chính / GCN.

Vai trò: đây KHÔNG phải bộ phân loại chính (LLM ở B2 mới là bộ phân loại chính,
vì cần đọc hiểu ngữ cảnh câu chữ tự nhiên trong văn bản pháp lý VN). Rule-based
ở đây chỉ đóng vai trò:
  1. Cross-check: nếu văn bản gốc có tín hiệu rõ ràng là đất TMD mà LLM lại bỏ sót
     (is_tmdv=False) → ưu tiên an toàn, tự nâng flag lên WARNING thay vì im lặng bỏ qua.
  2. Trích xuất số quyết định phê duyệt dự án (nếu có) bằng regex, làm căn cứ tham khảo
     bổ sung cho LLM/cán bộ tín dụng — KHÔNG dùng để tự động khẳng định thuoc_du_an=True.

Nguyên tắc thiết kế: rule-based chỉ được phép làm hệ thống THẬN TRỌNG HƠN (tăng cảnh báo),
không bao giờ được dùng để tự động HẠ thấp rủi ro hay khẳng định thay LLM.
"""
from __future__ import annotations
import re

from utils.parsing_utils import strip_accents

# ─────────────────────────────────────────────
# Bảng mã ký hiệu loại đất liên quan (trích từ Phụ lục II - Mục A,
# Thông tư 08/2024/TT-BTNMT) — chỉ giữ các loại đất liên quan tới nghiệp vụ
# thẩm định TSBĐ cá nhân (đất ở, TMDV, SXKD phi NN, nông nghiệp phổ biến).
# ─────────────────────────────────────────────
LAND_USE_CODE_REFERENCE: dict[str, str] = {
    "ODT": "Đất ở tại đô thị",
    "ONT": "Đất ở tại nông thôn",
    "TMD": "Đất thương mại, dịch vụ",
    "SKC": "Đất cơ sở sản xuất phi nông nghiệp",
    "SKK": "Đất khu công nghiệp",
    "SCC": "Đất khu công nghiệp, cụm công nghiệp",
    "LUC": "Đất chuyên trồng lúa",
    "LUK": "Đất trồng lúa còn lại",
    "CLN": "Đất trồng cây lâu năm",
    "NTS": "Đất nuôi trồng thủy sản",
    "NKH": "Đất nông nghiệp khác",
}

# Mã đất coi là "đất ở" hợp lệ để tính diện tích đủ điều kiện quy đổi (cùng nhà ở)
RESIDENTIAL_LAND_CODES = {"ODT", "ONT"}
# Mã đất coi là TMDV
TMDV_LAND_CODES = {"TMD"}

# ─────────────────────────────────────────────
# Tín hiệu nhận diện đất TMDV trong văn bản OCR (đã strip dấu, uppercase)
# ─────────────────────────────────────────────
_TMD_PATTERNS = [
    re.compile(r"DAT THUONG MAI,?\s*DICH VU"),
    re.compile(r"THUONG MAI\s*-\s*DICH VU"),
    re.compile(r"\bTMD\b"),  # mã ký hiệu chuẩn theo TT 08/2024/TT-BTNMT
]

# Từ khóa cho biết hồ sơ có nhắc tới "dự án" (chưa chắc đã là căn cứ đủ mạnh,
# chỉ dùng để khoanh vùng đoạn văn bản cần cán bộ đọc kỹ thêm)
_PROJECT_KEYWORDS = [
    "DU AN", "KHU DO THI", "KHU CONG NGHIEP", "KHU DAN CU",
    "CHU TRUONG DAU TU", "PHE DUYET DU AN", "GIAY CHUNG NHAN DAU TU",
]

# Từ khóa phủ định rõ ràng — đất TMDV KHÔNG thuộc dự án
_NEGATIVE_PROJECT_KEYWORDS = [
    "KHONG THUOC DU AN",
    "NGOAI QUY HOACH DU AN",
    "DAT XEN KET",
    "TU CHUYEN MUC DICH SU DUNG DAT",
    "KHONG NAM TRONG DU AN",
]

# Regex trích số quyết định phê duyệt dự án / chủ trương đầu tư, dạng phổ biến
# VN: "Quyết định số 1234/QĐ-UBND ngày 01/01/2020", "QĐ số 56/QĐ-TTg"...
_PROJECT_DECISION_PATTERN = re.compile(
    r"(?:QUYET\s*DINH|QD)[^\n]{0,40}?(?:SO)?\s*[:\.\-]?\s*"
    r"(\d{1,6}\s*[/\-]\s*[A-ZĐ\-]{2,15}(?:\s*[/\-]\s*[A-ZĐ\-]{2,15})?)"
)


def detect_tmdv_rule_based(text: str) -> dict:
    """
    Quét text OCR (GCN + hợp đồng/văn bản chuyển nhượng) để tìm tín hiệu
    rule-based liên quan đến đất TMDV và căn cứ dự án.

    Trả về dict:
      - is_tmdv_signal: bool — có tìm thấy mã/cụm từ "đất thương mại, dịch vụ" / "TMD" không
      - project_keyword_hit: bool — có nhắc tới "dự án"/"khu đô thị"/... không (tín hiệu yếu)
      - negative_project_signal: bool — có câu phủ định rõ ràng "không thuộc dự án" không
      - decision_numbers_found: list[str] — các số quyết định trích được (tối đa 3, không trùng)
    """
    stripped = strip_accents(text or "")

    is_tmdv_signal = any(p.search(stripped) for p in _TMD_PATTERNS)
    project_keyword_hit = any(kw in stripped for kw in _PROJECT_KEYWORDS)
    negative_hit = any(kw in stripped for kw in _NEGATIVE_PROJECT_KEYWORDS)

    raw_matches = _PROJECT_DECISION_PATTERN.findall(stripped)
    decision_numbers = list(dict.fromkeys(m.replace(" ", "") for m in raw_matches))[:3]

    return {
        "is_tmdv_signal": is_tmdv_signal,
        "project_keyword_hit": project_keyword_hit,
        "negative_project_signal": negative_hit,
        "decision_numbers_found": decision_numbers,
    }