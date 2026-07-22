"""
Identity Rules — bộ quy tắc rule-based dùng làm lưới an toàn (safety net) cho việc
LLM so khớp tên chủ tài sản giữa các nguồn khác nhau trong hồ sơ:
  - CCCD (owner_info.ho_ten)
  - GCN, mục biến động gần nhất (asset_info.chu_su_dung_hien_tai / chu_su_dung_goc)
  - Hợp đồng mua bán / văn bản chuyển nhượng (asset_info.ben_mua_hop_dong)

Vấn đề cần chặn: LLM có xu hướng coi 2 tên "gần giống nhau" (cùng bộ âm tiết,
khác thứ tự — vd "Nguyễn Minh Tuấn" vs "Nguyễn Tuấn Minh") là TRÙNG NHAU, hoặc
"tự đồng bộ" tên giữa các văn bản khi trích xuất mà không báo hiệu sai lệch.
Với tên người Việt, thứ tự các âm tiết là một phần định danh — hoán vị nghĩa là
MỘT TÊN KHÁC, không phải lỗi chính tả/OCR.

LƯU Ý QUAN TRỌNG VỀ DANH XƯNG (Ông/Bà/Anh/Chị...): Groq LLM không đảm bảo output
hoàn toàn tất định giữa các lần gọi dù temperature=0 — cùng một hồ sơ, có lần
LLM trích "Nguyễn Minh Tuấn", có lần lại trích "Ông Nguyễn Minh Tuấn" (kèm danh
xưng, vì văn bản gốc ghi trang trọng "Ông Nguyễn Văn A..."). Danh xưng KHÔNG
phải một phần định danh cá nhân (không phân biệt được 2 người khác nhau), nên
PHẢI được loại bỏ trước khi so sánh — nếu không, hệ thống sẽ báo false positive
không nhất quán giữa các lần chạy trên CÙNG một hồ sơ, gây mất tin cậy.

Vai trò: đây KHÔNG phải bộ so khớp chính (LLM ở B2 mới là bộ so khớp chính, vì
cần đọc hiểu ngữ cảnh trong văn bản pháp lý VN, biến động qua nhiều lần chuyển
nhượng...). Rule-based ở đây chỉ đóng vai trò lưới an toàn CUỐI CÙNG: khi phát
hiện 2 chuỗi tên (đã bỏ danh xưng) KHÔNG khớp tuyệt đối, hệ thống PHẢI chuyển
sang trạng thái thận trọng hơn (mismatch), không được để LLM tự tin kết luận
"khớp" một cách chủ quan.

Nguyên tắc thiết kế (giống land_rules.py): rule-based chỉ được phép làm hệ
thống THẬN TRỌNG HƠN, không bao giờ dùng để tự động HẠ thấp rủi ro hay ghi đè
một kết luận "mismatch" của LLM thành "match".
"""
from __future__ import annotations
import difflib

from utils.parsing_utils import strip_accents

# Danh xưng/tiền tố xã giao thường xuất hiện trước tên trong văn bản pháp lý VN
# (GCN, hợp đồng...). Đây KHÔNG phải một phần định danh cá nhân — 2 người khác
# nhau vẫn có thể cùng được gọi là "Ông"/"Bà" — nên phải loại bỏ trước khi so
# sánh, để tránh false positive khi 1 nguồn ghi kèm danh xưng còn nguồn kia thì
# không (rất phổ biến do cách hành văn khác nhau giữa các văn bản, hoặc do LLM
# trích xuất không nhất quán giữa các lần gọi).
_HONORIFIC_PREFIXES = {
    "ONG", "BA", "ANH", "CHI", "CO", "CHU", "BAC", "NGAI",
    "QUYONG", "QUYBA",
}


def _strip_honorifics(tokens: list[str]) -> list[str]:
    """
    Bỏ các token danh xưng ở ĐẦU tên (vd "ONG", "BA"), lặp lại cho tới khi
    token đầu tiên không còn là danh xưng. Chỉ xử lý ở đầu chuỗi để tránh lỡ
    tay xoá nhầm một âm tiết trong tên thật trùng với danh xưng ở giữa/cuối tên
    (dù trường hợp này gần như không xảy ra ở tên người Việt).
    """
    result = list(tokens)
    while result and result[0] in _HONORIFIC_PREFIXES:
        result.pop(0)
    return result


def normalize_name(name: str) -> str:
    """
    Chuẩn hoá tên: bỏ dấu tiếng Việt, uppercase, gộp khoảng trắng thừa, và bỏ
    danh xưng ở đầu tên (Ông/Bà/Anh/Chị...). Dùng để so sánh mà không bị ảnh
    hưởng bởi lỗi OCR về dấu câu/khoảng trắng hay cách hành văn trang trọng
    khác nhau giữa các văn bản, NHƯNG vẫn giữ nguyên thứ tự âm tiết của TÊN
    THẬT (vì thứ tự là một phần định danh).
    """
    if not name:
        return ""
    tokens = strip_accents(name).split()
    tokens = _strip_honorifics(tokens)
    return " ".join(tokens)


def _is_honorific(token: str) -> bool:
    """So khớp danh xưng KHÔNG phân biệt dấu (vd 'ÔNG' vẫn nhận diện được như
    'ONG'), dùng riêng cho normalize_name_keep_diacritics() bên dưới — vì
    _HONORIFIC_PREFIXES lưu dạng đã bỏ dấu."""
    return strip_accents(token) in _HONORIFIC_PREFIXES


def _strip_honorifics_keep_diacritics(tokens: list[str]) -> list[str]:
    """Giống _strip_honorifics(), nhưng giữ nguyên dấu của các token còn lại."""
    result = list(tokens)
    while result and _is_honorific(result[0]):
        result.pop(0)
    return result


def normalize_name_keep_diacritics(name: str) -> str:
    """
    Chuẩn hoá tên nhưng GIỮ NGUYÊN dấu tiếng Việt — chỉ uppercase, gộp khoảng
    trắng thừa, và bỏ danh xưng ở đầu tên. KHÔNG dùng để quyết định
    owner_matched (vẫn dùng normalize_name() bỏ dấu, để không tạo false
    positive khi lệch dấu do OCR) — hàm này CHỈ dùng để tính similarity hiển
    thị tham khảo cho cán bộ tín dụng, vì cần giữ dấu mới phát hiện được các
    trường hợp OCR đọc sai 1 ký tự có dấu (vd "Á" != "Ấ") — nếu bỏ dấu trước
    khi so sánh thì 2 ký tự này sẽ bị coi là giống hệt nhau, che mất sai khác
    thật sự giữa 2 nguồn.
    """
    if not name:
        return ""
    tokens = name.strip().upper().split()
    tokens = _strip_honorifics_keep_diacritics(tokens)
    return " ".join(tokens)


def compare_names(name_a: str, name_b: str) -> dict:
    """
    So sánh 2 tên đã chuẩn hoá.

    Trả về dict:
      - exact_match: bool — khớp tuyệt đối sau chuẩn hoá BỎ DẤU/khoảng trắng
        (dùng để quyết định owner_matched — cố tình bỏ dấu để không tạo false
        positive khi lỗi OCR làm sai 1 dấu câu, xem normalize_name()).
      - same_tokens_diff_order: bool — cùng bộ âm tiết nhưng khác thứ tự
        (dấu hiệu rất đáng ngờ: dễ bị LLM/nghiệp vụ nhầm là "khớp" nhưng
        thực chất là 2 tên khác nhau, vd hoán vị họ/tên đệm/tên)
      - similarity: float [0,1] — ĐÃ SỬA: độ giống chuỗi (difflib ratio) tính
        trên chuỗi GỐC CÒN DẤU (chỉ bỏ danh xưng + khoảng trắng thừa, xem
        normalize_name_keep_diacritics()), KHÔNG dùng chuỗi đã bỏ dấu. Lý do:
        nếu tính trên chuỗi đã bỏ dấu, các lỗi OCR đọc sai 1 ký tự có dấu (vd
        "Á" bị đọc thành "Ấ") sẽ bị che mất — vì sau khi bỏ dấu 2 ký tự này
        trông giống hệt nhau. Field này CHỈ dùng để hiển thị tham khảo cho cán
        bộ tín dụng, KHÔNG ảnh hưởng tới exact_match/owner_matched ở trên.
      - has_data: bool — cả 2 chuỗi đầu vào đều không rỗng (đủ để so sánh)
    """
    a, b = normalize_name(name_a), normalize_name(name_b)

    if not a or not b:
        return {
            "exact_match": False,
            "same_tokens_diff_order": False,
            "similarity": 0.0,
            "has_data": False,
        }

    exact = a == b
    tokens_a, tokens_b = a.split(), b.split()
    same_tokens_diff_order = (not exact) and (sorted(tokens_a) == sorted(tokens_b))

    # ĐÃ SỬA: similarity tính trên chuỗi GỐC CÒN DẤU, không phải a/b (đã bỏ dấu)
    a_raw = normalize_name_keep_diacritics(name_a)
    b_raw = normalize_name_keep_diacritics(name_b)
    similarity = round(difflib.SequenceMatcher(None, a_raw, b_raw).ratio(), 3)

    return {
        "exact_match": exact,
        "same_tokens_diff_order": same_tokens_diff_order,
        "similarity": similarity,
        "has_data": True,
    }


def describe_mismatch_reason(result: dict) -> str:
    """Sinh mô tả ngắn gọn lý do 2 tên không khớp, dùng trong thông báo flag."""
    if result["same_tokens_diff_order"]:
        return "cùng bộ âm tiết nhưng khác thứ tự (hoán vị họ/tên đệm/tên) — đây là 2 tên khác nhau"
    return f"độ giống nhau giữa 2 chuỗi chỉ đạt {result['similarity']:.0%}"