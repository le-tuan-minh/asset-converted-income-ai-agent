"""B3 — Flag & Alert Engine (rule-based, cho 1 tài sản)."""
from __future__ import annotations
from datetime import date, datetime

from schemas import OwnerInfo, AssetInfo, IdentityCheckResult, LandPurposeResult, FlagItem
from cores.identity_rules import compare_names


def _parse_date(date_str: str) -> date | None:
    if not date_str:
        return None
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%y"):
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _months_ago(d: date) -> int:
    """Số tháng TRÒN đã trôi qua kể từ ngày d tới hôm nay (có tính ngày trong tháng)."""
    today = date.today()
    months = (today.year - d.year) * 12 + (today.month - d.month)
    if today.day < d.day:
        months -= 1
    return max(months, 0)


def _determine_asset_formation_date(owner_info: OwnerInfo, asset_info: AssetInfo) -> tuple[date | None, str]:
    """
    Xác định NGÀY HÌNH THÀNH TÀI SẢN — theo đúng yêu cầu nghiệp vụ gốc: "dựa
    vào ngày cấp giấy chứng nhận, ngày chuyển nhượng CHO KHÁCH HÀNG ghi nhận
    thời điểm hình thành tài sản".

    Rule-based, TẤT ĐỊNH, ưu tiên theo thứ tự:
      1. Trong bien_dong_lich_su, tìm biến động có chu_moi KHỚP TUYỆT ĐỐI
         (sau chuẩn hoá — compare_names) với owner_info.ho_ten, ưu tiên biến
         động GẦN NHẤT (cuối danh sách) nếu có nhiều biến động khớp. Đây mới
         đúng là "ngày chuyển nhượng cho khách hàng" — KHÔNG nhất thiết là
         biến động cuối cùng trong GCN (biến động cuối có thể là thế chấp,
         không phải chuyển nhượng chủ quyền).
      2. Nếu GCN không ghi chi tiết bien_dong_lich_su (hoặc không tìm được
         khớp) → dùng asset_info.ngay_chuyen_nhuong (do LLM tổng hợp).
      3. Nếu vẫn không có → dùng asset_info.ngay_cap_gcn (trường hợp khách
         hàng là chủ sử dụng gốc, GCN cấp thẳng, tài sản chưa từng biến động).

    KHÔNG dùng identity_check.asset_formation_date làm nguồn chính vì đây là
    câu trả lời tự do do LLM tự diễn giải/tổng hợp — không đảm bảo tất định
    giữa các lần gọi và không có cấu trúc rõ ràng để tin cậy làm căn cứ tính
    toán chính. Trường đó chỉ dùng làm GHI CHÚ tham khảo (asset_formation_note).
    """
    if owner_info.ho_ten:
        for bd in reversed(asset_info.bien_dong_lich_su):
            if not bd.chu_moi:
                continue
            result = compare_names(owner_info.ho_ten, bd.chu_moi)
            if result["has_data"] and result["exact_match"]:
                d = _parse_date(bd.ngay)
                if d:
                    return d, f"bien_dong_lich_su (khớp chu_moi='{bd.chu_moi}', ngày={bd.ngay})"

    d = _parse_date(asset_info.ngay_chuyen_nhuong)
    if d:
        return d, "asset_info.ngay_chuyen_nhuong"

    d = _parse_date(asset_info.ngay_cap_gcn)
    if d:
        return d, "asset_info.ngay_cap_gcn (không có biến động — có thể KH là chủ sử dụng gốc)"

    return None, ""


def flag_asset(
    owner_info: OwnerInfo, asset_info: AssetInfo,
    identity_check: IdentityCheckResult, land_purpose: LandPurposeResult,
    flags: list[FlagItem], warnings: list[str], notes: list[str],
) -> tuple[list[FlagItem], list[str], list[str]]:
    """B3 cho 1 tài sản — nhận flags/warnings/notes đã có từ B2, bổ sung thêm."""
    print("[B3] Flag engine — kiểm tra điều kiện ràng buộc cho tài sản này.")

    # Rule 1: Chủ tài sản không khớp
    if not identity_check.owner_matched:
        mismatch_desc = ", ".join(identity_check.mismatch_fields) if identity_check.mismatch_fields else "không rõ trường nào"
        flags.append(FlagItem(
            flag_type="CHU_TAI_SAN_LECH", severity="ERROR",
            description=(
                f"Chủ sử dụng trên GCN/Hợp đồng KHÔNG khớp với CCCD khách hàng. Trường lệch: {mismatch_desc}"
            ),
            affected_field="ho_ten / so_cccd",
        ))
        warnings.append(f"⛔ CHỦ TÀI SẢN LỆCH: {mismatch_desc}. Cần xác minh lại hồ sơ nhân thân.")
        print(f"[B3] ⛔ Flag: CHU_TAI_SAN_LECH — {mismatch_desc}")
    else:
        print("[B3] ✅ Chủ tài sản khớp CCCD.")

    # Rule 2: Tặng cho / thừa kế
    if identity_check.is_tang_cho or identity_check.is_thua_ke or asset_info.co_thong_tin_tang_cho:
        loai = "tặng cho" if (identity_check.is_tang_cho or asset_info.co_thong_tin_tang_cho) else "thừa kế"
        flags.append(FlagItem(
            flag_type="TANG_CHO_THUA_KE", severity="WARNING",
            description=f"Tài sản có nguồn gốc {loai}. Loại khỏi nguồn tài sản quy đổi, chỉ dùng cho tính toán tài sản thanh lý.",
            affected_field="nguon_goc_tai_san",
        ))
        warnings.append(f"⚠️ Tài sản có nguồn gốc {loai} — không dùng để quy đổi giá trị tài sản đảm bảo chính.")
        print(f"[B3] ⚠️ Flag: TANG_CHO_THUA_KE ({loai})")

    # Rule 3: Tài sản mới hình thành (< 24 tháng)
    formation_date, formation_source = _determine_asset_formation_date(owner_info, asset_info)

    if formation_date:
        notes.append(
            f"[B3] Ngày hình thành tài sản xác định = {formation_date.strftime('%d/%m/%Y')} "
            f"(nguồn: {formation_source})."
        )
        # Đối chiếu tham khảo với asset_formation_date do LLM tự diễn giải (nếu có) —
        # CHỈ ghi chú khi lệch, KHÔNG dùng để ghi đè kết quả rule-based ở trên.
        llm_date = _parse_date(identity_check.asset_formation_date)
        if llm_date and llm_date != formation_date:
            notes.append(
                f"[B3] Lưu ý: LLM tự ghi asset_formation_date='{identity_check.asset_formation_date}' "
                f"khác với ngày rule-based tính được ({formation_date.strftime('%d/%m/%Y')}). "
                f"Ưu tiên dùng kết quả rule-based (nguồn: {formation_source})."
            )

        months = _months_ago(formation_date)
        if months < 24:
            flags.append(FlagItem(
                flag_type="TAI_SAN_MOI_HINH_THANH", severity="WARNING",
                description=(
                    f"Tài sản hình thành ngày {formation_date.strftime('%d/%m/%Y')}, cách đây {months} tháng "
                    f"(< 24 tháng). Nguồn xác định: {formation_source}. Cần làm rõ nguồn gốc tiền hình thành tài sản."
                ),
                affected_field="nguon_goc_tai_san",
            ))
            warnings.append(f"⚠️ Tài sản mới hình thành ({months} tháng) — cần làm rõ nguồn gốc tiền.")
            print(f"[B3] ⚠️ Flag: TAI_SAN_MOI_HINH_THANH ({months} tháng, nguồn: {formation_source})")
    else:
        flags.append(FlagItem(
            flag_type="NGAY_HINH_THANH_KHONG_XAC_DINH", severity="WARNING",
            description="Không xác định được ngày hình thành tài sản từ hồ sơ hiện có (không có bien_dong_lich_su khớp tên KH, ngày_chuyen_nhuong, hoặc ngay_cap_gcn hợp lệ). Cần xác minh thủ công.",
            affected_field="nguon_goc_tai_san",
        ))
        warnings.append("⚠️ Không xác định được ngày hình thành tài sản — cần xác minh thủ công.")
        print("[B3] ⚠️ Flag: NGAY_HINH_THANH_KHONG_XAC_DINH")

    # Rule 4: TMDV ngoài dự án / cần xác minh
    if land_purpose.is_tmdv:
        if land_purpose.thuoc_du_an is False:
            flags.append(FlagItem(
                flag_type="TMDV_NGOAI_DU_AN", severity="ERROR",
                description="Đất thương mại dịch vụ (TMD) KHÔNG thuộc dự án được phê duyệt.",
                affected_field="asset_info.thuoc_du_an",
            ))
            warnings.append("⛔ Đất TMDV không thuộc dự án được phê duyệt.")
            print("[B3] ⛔ Flag: TMDV_NGOAI_DU_AN")
        elif land_purpose.thuoc_du_an is None:
            flags.append(FlagItem(
                flag_type="TMDV_CAN_XAC_MINH_THU_CONG", severity="WARNING",
                description=land_purpose.warning_tmdv or "Đất TMDV chưa xác định thuộc dự án hay không, cần xác minh thủ công.",
                affected_field="asset_info.thuoc_du_an",
            ))
            warnings.append("⚠️ Đất TMDV cần cán bộ tín dụng xác minh thủ công có thuộc dự án hay không.")
            print("[B3] ⚠️ Flag: TMDV_CAN_XAC_MINH_THU_CONG")
        if land_purpose.nguon_xac_dinh_du_an == "web_search":
            flags.append(FlagItem(
                flag_type="TMDV_DU_AN_XAC_MINH_WEB", severity="WARNING",
                description="Thông tin thuộc dự án được xác định qua tra cứu web bổ sung, chỉ mang tính tham khảo.",
                affected_field="asset_info.thuoc_du_an",
            ))

    print("[B3] Hoàn thành.")
    return flags, warnings, notes
