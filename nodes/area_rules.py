"""
Area Rules — parse số diện tích kiểu VN + rule-based cross-check diện tích.

Vấn đề cần chặn: các trường diện tích (dien_tich_tong, dien_tich_nn, ...) hiện
là string tự do do LLM trả về, và LLM tự thực hiện phép CỘNG (vd
dien_tich_du_dieu_kien = dat_o + nha_o) ngay trong lúc extract JSON. Điều này
khiến cùng một hồ sơ, 2 lần gọi Groq LLM (dù temperature=0) có thể ra 2 con số
khác nhau — vì LLM không tất định 100% ở việc "làm toán", và ranh giới giữa các
loại đất (vd đất nuôi trồng thủy sản có tính vào "đất nông nghiệp" hay không)
không được chốt rõ trong prompt.

Nguyên tắc thiết kế (giống land_rules.py / identity_rules.py):
  - Rule-based ở đây có 2 vai trò:
      1. Cung cấp hàm parse số kiểu VN tất định (dấu '.' phân cách nghìn, ','
         thập phân) để mọi so sánh/tính toán về sau dùng chung 1 nguồn chân lý.
      2. Cross-check tổng diện tích các thành phần với dien_tich_tong — CHỈ
         được phép sinh flag cảnh báo khi lệch, KHÔNG được tự "sửa" số liệu
         nào của LLM (không đủ căn cứ để biết số nào đúng, số nào sai).
  - RIÊNG "dien_tich_du_dieu_kien": đây là phép cộng ĐƠN GIẢN, CÓ ĐỊNH NGHĨA RÕ
    RÀNG (đất ở ODT/ONT + nhà ở), không có yếu tố suy luận ngữ nghĩa nào cần
    LLM — do đó việc này PHẢI do code tính tất định, không lấy trực tiếp con số
    LLM tự cộng trong JSON output.
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


def compute_dien_tich_du_dieu_kien(dien_tich_dat_o: str, dien_tich_nha_o: str) -> str:
    """
    Tính TẤT ĐỊNH diện tích đủ điều kiện quy đổi = đất ở (ODT/ONT) + nhà ở.
    Đây là con số duy nhất dùng để tính giá trị BĐS ở bước sau — PHẢI nhất
    quán 100% giữa các lần chạy trên cùng 1 hồ sơ, nên KHÔNG được lấy từ số
    LLM tự cộng trong JSON (dễ thiếu nhất quán giữa các lần gọi).
    """
    dat_o = parse_vn_area(dien_tich_dat_o) or 0.0
    nha_o = parse_vn_area(dien_tich_nha_o) or 0.0
    return format_area(dat_o + nha_o)


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