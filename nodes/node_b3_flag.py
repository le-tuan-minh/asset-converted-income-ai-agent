"""
B3 - Flag & Alert Engine
Rule-based engine dựa trên kết quả từ B2:
  1. Flag tài sản không trùng chủ
  2. Flag tài sản tặng cho / thừa kế
  3. Cảnh báo tài sản mới hình thành > 70% trong 2 năm gần nhất
  4. Cảnh báo đất TMDV không thuộc dự án
  5. Xác định diện tích đủ điều kiện quy đổi
"""
from __future__ import annotations
from datetime import date, datetime

from schemas import GraphState, FlagItem


def _parse_date(date_str: str) -> date | None:
    """Parse ngày từ nhiều định dạng phổ biến trong văn bản Việt Nam."""
    if not date_str:
        return None
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%y"):
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _months_ago(d: date) -> int:
    """Số tháng tính từ ngày d đến hôm nay."""
    today = date.today()
    return (today.year - d.year) * 12 + (today.month - d.month)


def node_b3_flag(state: GraphState) -> GraphState:
    """
    LangGraph node B3.
    Đọc identity_check, asset_info, land_purpose từ state,
    sinh ra danh sách FlagItem và warnings.
    """
    print("\n" + "="*60)
    print("B3 · FLAG ENGINE — Kiểm tra điều kiện ràng buộc")
    print("="*60)

    flags  = list(state.flags)
    warnings = list(state.warnings)
    notes  = list(state.processing_notes)

    ic = state.identity_check
    ai = state.asset_info
    lp = state.land_purpose

    # ── Rule 1: Chủ tài sản không khớp ─────────────────────────
    if not ic.owner_matched:
        mismatch_desc = ", ".join(ic.mismatch_fields) if ic.mismatch_fields else "không rõ trường nào"
        flags.append(FlagItem(
            flag_type="CHU_TAI_SAN_LECH",
            severity="ERROR",
            description=(
                f"Chủ sử dụng trên GCN/Hợp đồng KHÔNG khớp với CCCD khách hàng. "
                f"Trường lệch: {mismatch_desc}"
            ),
            affected_field="ho_ten / so_cccd",
        ))
        warnings.append(
            f"⛔ CHỦ TÀI SẢN LỆCH: {mismatch_desc}. "
            "Cần xác minh lại hồ sơ nhân thân."
        )
        print(f"[B3] ⛔ Flag: CHU_TAI_SAN_LECH — {mismatch_desc}")
    else:
        print("[B3] ✅ Chủ tài sản khớp CCCD.")

    # ── Rule 2: Tặng cho / Thừa kế ──────────────────────────────
    if ic.is_tang_cho or ic.is_thua_ke or ai.co_thong_tin_tang_cho:
        loai = "tặng cho" if ic.is_tang_cho else "thừa kế"
        flags.append(FlagItem(
            flag_type="TANG_CHO_THUA_KE",
            severity="WARNING",
            description=(
                f"Tài sản có nguồn gốc {loai}. "
                "Loại khỏi tài sản quy đổi, chỉ dùng tính thanh lý."
            ),
            affected_field="nguon_goc_tai_san",
        ))
        warnings.append(
            f"⚠️ TÀI SẢN {loai.upper()}: Không được dùng làm tài sản quy đổi. "
            "Chỉ tính vào tài sản thanh lý."
        )
        print(f"[B3] ⚠️ Flag: TANG_CHO_THUA_KE — {loai}")

    # ── Rule 3: Tài sản mới hình thành trong 2 năm gần nhất ─────
    formation_date = _parse_date(ic.asset_formation_date)
    if formation_date:
        months_old = _months_ago(formation_date)
        if months_old <= 24:
            flags.append(FlagItem(
                flag_type="TAI_SAN_MOI_HINH_THANH",
                severity="WARNING",
                description=(
                    f"Tài sản hình thành ngày {ic.asset_formation_date} "
                    f"({months_old} tháng trước). "
                    "Cần làm rõ nguồn gốc tiền hình thành tài sản."
                ),
                affected_field="asset_formation_date",
            ))
            warnings.append(
                f"⚠️ TÀI SẢN MỚI HÌNH THÀNH ({months_old} tháng): "
                f"{ic.asset_formation_date}. "
                "Cần làm rõ nguồn gốc vốn mua tài sản trong 2 năm gần nhất."
            )
            print(f"[B3] ⚠️ Flag: TAI_SAN_MOI_HINH_THANH — {months_old} tháng")
        else:
            print(f"[B3] ✅ Tài sản hình thành {months_old} tháng trước, không cảnh báo.")
    else:
        print("[B3] ℹ️ Không xác định được ngày hình thành tài sản.")

    # ── Rule 4: Đất TMDV không thuộc dự án ──────────────────────
    if lp.is_tmdv:
        if lp.thuoc_du_an is False:
            flags.append(FlagItem(
                flag_type="TMDV_NGOAI_DU_AN",
                severity="ERROR",
                description=(
                    "Đất TMDV KHÔNG thuộc dự án được phê duyệt. "
                    "Không đủ điều kiện làm TSBĐ theo quy định."
                ),
                affected_field="muc_dich_su_dung / thuoc_du_an",
            ))
            warnings.append(
                "⛔ ĐẤT TMDV NGOÀI DỰ ÁN: Không đủ điều kiện TSBĐ. "
                "Cần xem xét loại khỏi danh mục tài sản đảm bảo."
            )
            print("[B3] ⛔ Flag: TMDV_NGOAI_DU_AN")
        elif lp.thuoc_du_an is True:
            print("[B3] ✅ Đất TMDV thuộc dự án, đủ điều kiện.")
        else:
            warnings.append(
                "⚠️ ĐẤT TMDV: Chưa xác định được có thuộc dự án không. "
                "Cần kiểm tra thêm."
            )
            print("[B3] ⚠️ TMDV — chưa xác định thuộc dự án.")

    # ── Rule 5: Tổng hợp diện tích đủ điều kiện ─────────────────
    dien_tich_note = (
        f"Diện tích đủ điều kiện quy đổi: {lp.dien_tich_du_dieu_kien or 'chưa xác định'}"
    )
    notes.append(dien_tich_note)
    print(f"[B3] {dien_tich_note}")

    # ── Xác định có flag nghiêm trọng không ──────────────────────
    has_critical = any(f.severity == "ERROR" for f in flags)

    notes.append(f"B3 hoàn thành: {len(flags)} flag(s), {len(warnings)} cảnh báo.")
    print(f"[B3] Tổng: {len(flags)} flag(s) | has_critical={has_critical}")
    print("[B3] Hoàn thành.\n")

    return state.model_copy(update={
        "flags": flags,
        "warnings": warnings,
        "has_critical_flags": has_critical,
        "processing_notes": notes,
    })