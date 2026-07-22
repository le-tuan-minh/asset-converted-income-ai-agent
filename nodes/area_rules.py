"""
Area Rules — parse số diện tích kiểu VN + rule-based cross-check diện tích.

Vấn đề cần chặn: các trường diện tích (dien_tich_tong, dien_tich_nn, ...) hiện
là string tự do do LLM trả về. Cùng một hồ sơ, 2 lần gọi Groq LLM (dù
temperature=0) có thể ra 2 con số khác nhau nếu để LLM tự "làm toán" — vì LLM
không tất định 100% ở việc này, và ranh giới giữa các loại đất (vd đất nuôi
trồng thủy sản có tính vào "đất nông nghiệp" hay không) không được chốt rõ
trong prompt.

Nguyên tắc thiết kế (giống land_rules.py / identity_rules.py):
  - Rule-based ở đây có 2 vai trò:
      1. Cung cấp hàm parse số kiểu VN tất định (dấu '.' phân cách nghìn, ','
         thập phân) để mọi so sánh/tính toán về sau dùng chung 1 nguồn chân lý.
      2. Cross-check tổng diện tích các thành phần với dien_tich_tong — CHỈ
         được phép sinh flag cảnh báo khi lệch, KHÔNG được tự "sửa" số liệu
         nào của LLM (không đủ căn cứ để biết số nào đúng, số nào sai).
  - "Diện tích đủ điều kiện quy đổi": ĐÃ SỬA (fix #2) — KHÔNG còn cộng gộp đất
    ở + nhà ở thành 1 con số nữa, vì đây là 2 loại tài sản khác bản chất
    (đất theo m² thửa đất, nhà theo m² sàn xây dựng), thường có đơn giá định
    giá khác nhau ở bước sau. Thay vào đó, code tính tất định lại RIÊNG TỪNG
    con số (xem compute_dien_tich_du_dieu_kien_parts), không lấy trực tiếp số
    LLM tự trả trong JSON output.
"""
from __future__ import annotations
import re

_NUM_PATTERN = re.compile(r"-?\d[\d.,]*")


def parse_vn_area(raw: str) -> float | None:
    """
    Parse chuỗi diện tích kiểu Việt Nam thành float.
    Quy ước: nếu chuỗi có dấu ',' thì ',' là phân cách thập phân và '.' là phân
    cách nghìn (vd "1.234,5" = 1234.5). Nếu KHÔNG có dấu ',' thì mọi dấu '.'
    được coi là phân cách nghìn (vd "9.112" = 9112, KHÔNG phải 9.112).

    Trả về None nếu không tìm thấy số nào trong chuỗi.
    """
    if not raw:
        return None
    match = _NUM_PATTERN.search(raw.replace(" ", ""))
    if not match:
        return None
    token = match.group(0)

    if "," in token:
        token = token.replace(".", "").replace(",", ".")
    else:
        token = token.replace(".", "")

    try:
        return float(token)
    except ValueError:
        return None


def format_area(value: float) -> str:
    """Format lại float thành string gọn (bỏ .0 nếu là số nguyên)."""
    if value == int(value):
        return str(int(value))
    return str(value)


def compute_dien_tich_du_dieu_kien_parts(
    dien_tich_dat_o: str, dien_tich_nha_o: str
) -> tuple[str, str]:
    """
    ĐÃ SỬA (fix #2 — không còn cộng gộp): tính TẤT ĐỊNH lại 2 con số đất ở và
    nhà ở RIÊNG BIỆT (chỉ parse + format lại cho nhất quán giữa các lần chạy,
    KHÔNG cộng chúng với nhau).

    Lý do tách: đất ở (m² thửa đất) và nhà ở (m² sàn xây dựng, có thể lớn hơn
    diện tích đất nếu nhà nhiều tầng) là 2 đại lượng khác bản chất, thường
    được định giá theo đơn giá riêng ở bước tính giá trị BĐS (B4+). Gộp lại
    thành 1 số làm bước sau mất khả năng áp đúng đơn giá cho từng loại.

    Trả về: (dien_tich_dat_o_formatted, dien_tich_nha_o_formatted)
    """
    dat_o = parse_vn_area(dien_tich_dat_o) or 0.0
    nha_o = parse_vn_area(dien_tich_nha_o) or 0.0
    return format_area(dat_o), format_area(nha_o)


def cross_check_area_totals(
    dien_tich_tong: str,
    dien_tich_dat_o: str,
    dien_tich_nn: str,
    dien_tich_nts: str,
    dien_tich_tmdv: str,
    tolerance_ratio: float = 0.02,
) -> dict:
    """
    So tổng các thành phần diện tích ĐẤT (không gồm dien_tich_nha_o, vì nhà ở
    nằm TRÊN đất ở, không cộng dồn thêm vào diện tích thửa đất) với
    dien_tich_tong ghi trên GCN.

    Trả về dict:
      - ok: bool
      - tong, tong_thanh_phan, lech: float (chỉ có khi tong parse được)
      - message: str — mô tả để ghi vào notes/flag khi lệch
    """
    tong = parse_vn_area(dien_tich_tong)
    if tong is None:
        return {"ok": True, "message": "Không có dien_tich_tong để đối chiếu."}

    dat_o = parse_vn_area(dien_tich_dat_o) or 0.0
    nn = parse_vn_area(dien_tich_nn) or 0.0
    nts = parse_vn_area(dien_tich_nts) or 0.0
    tmdv = parse_vn_area(dien_tich_tmdv) or 0.0

    tong_thanh_phan = dat_o + nn + nts + tmdv
    lech = abs(tong - tong_thanh_phan)
    threshold = max(tong * tolerance_ratio, 1.0)

    if lech > threshold:
        return {
            "ok": False,
            "tong": tong,
            "tong_thanh_phan": tong_thanh_phan,
            "lech": lech,
            "message": (
                f"Tổng diện tích ghi trên GCN ({format_area(tong)} m²) KHÔNG khớp với tổng "
                f"các thành phần đã trích xuất (đất ở {format_area(dat_o)} + NN {format_area(nn)} "
                f"+ NTS {format_area(nts)} + TMDV {format_area(tmdv)} = "
                f"{format_area(tong_thanh_phan)} m², lệch {format_area(lech)} m²). Có thể LLM bỏ "
                f"sót hoặc gộp nhầm một loại đất (vd gộp đất nuôi trồng thủy sản vào đất nông "
                f"nghiệp). Cần cán bộ tín dụng đối chiếu lại bản gốc GCN."
            ),
        }
    return {"ok": True, "tong": tong, "tong_thanh_phan": tong_thanh_phan, "lech": lech}