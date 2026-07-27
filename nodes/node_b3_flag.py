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
) -> tuple[IdentityCheckResult, list[FlagItem], list[str], list[str]]:   # ← thêm IdentityCheckResult vào return type
    """B3 cho 1 tài sản — nhận flags/warnings/notes đã có từ B2, bổ sung thêm."""
    print("[B3] Flag engine — kiểm tra điều kiện ràng buộc cho tài sản này.")

    # ... Rule 1, Rule 2 giữ nguyên ...

    # Rule 3: Tài sản mới hình thành (< 24 tháng)
    formation_date, formation_source = _determine_asset_formation_date(owner_info, asset_info)

    if formation_date:
        formation_date_str = formation_date.strftime('%d/%m/%Y')
        original_llm_value = identity_check.asset_formation_date

        notes.append(
            f"[B3] Ngày hình thành tài sản xác định = {formation_date_str} "
            f"(nguồn: {formation_source})."
        )

        llm_date = _parse_date(original_llm_value)
        note_suffix = ""
        if llm_date and llm_date != formation_date:
            notes.append(
                f"[B3] Lưu ý: LLM tự ghi asset_formation_date='{original_llm_value}' "
                f"khác với ngày rule-based tính được ({formation_date_str}). "
                f"Ưu tiên dùng kết quả rule-based (nguồn: {formation_source})."
            )
            note_suffix = f" (LLM trước đó ghi: {original_llm_value})"

        # ★ FIX: ghi đè identity_check.asset_formation_date bằng kết quả
        # rule-based — đây là trường hiển thị ra UI/kết quả cuối cùng cho
        # cán bộ tín dụng, KHÔNG được để giá trị tự do do LLM diễn giải.
        identity_check = identity_check.model_copy(update={
            "asset_formation_date": formation_date_str,
            "asset_formation_note": (
                f"Xác định bởi rule-based (nguồn: {formation_source}).{note_suffix}"
            ),
        })

        months = _months_ago(formation_date)
        if months < 24:
            flags.append(FlagItem(
                flag_type="TAI_SAN_MOI_HINH_THANH", severity="WARNING",
                description=(
                    f"Tài sản hình thành ngày {formation_date_str}, cách đây {months} tháng "
                    f"(< 24 tháng). Nguồn xác định: {formation_source}. Cần làm rõ nguồn gốc tiền hình thành tài sản."
                ),
                affected_field="nguon_goc_tai_san",
            ))
            warnings.append(f"⚠️ Tài sản mới hình thành ({months} tháng) — cần làm rõ nguồn gốc tiền.")
            print(f"[B3] ⚠️ Flag: TAI_SAN_MOI_HINH_THANH ({months} tháng, nguồn: {formation_source})")
    else:
        # ★ FIX: không xác định được bằng rule-based → cảnh báo luôn trong note
        # hiển thị, tránh để người dùng tưởng nhầm giá trị LLM là đáng tin.
        identity_check = identity_check.model_copy(update={
            "asset_formation_note": (
                "Không xác định được bằng rule-based — giá trị (nếu có) chỉ do LLM tự diễn giải, cần xác minh thủ công."
            ),
        })
        flags.append(FlagItem(
            flag_type="NGAY_HINH_THANH_KHONG_XAC_DINH", severity="WARNING",
            description="Không xác định được ngày hình thành tài sản từ hồ sơ hiện có (không có bien_dong_lich_su khớp tên KH, ngày_chuyen_nhuong, hoặc ngay_cap_gcn hợp lệ). Cần xác minh thủ công.",
            affected_field="nguon_goc_tai_san",
        ))
        warnings.append("⚠️ Không xác định được ngày hình thành tài sản — cần xác minh thủ công.")
        print("[B3] ⚠️ Flag: NGAY_HINH_THANH_KHONG_XAC_DINH")

    # Rule 4: Đất TMDV — không thuộc dự án / thiếu căn cứ pháp lý.
    # ★ FIX: rule này từng bị rơi mất trong 1 lần refactor trước (chỉ còn lại
    # comment giữ chỗ) — TMDV_NGOAI_DU_AN và TMDV_DU_AN_XAC_MINH_WEB được khai
    # báo trong schema/README nhưng KHÔNG có nơi nào thực sự tạo ra flag này.
    # Khôi phục lại, đồng thời bổ sung nhánh mới: sau khi B2b tra cứu web kết
    # luận "thuộc dự án" nhưng KHÔNG tìm được căn cứ pháp lý chính thức (chỉ
    # có mô tả marketing) — cũng phải raise flag, không chỉ ghi warning_tmdv.
    if land_purpose.is_tmdv:
        if land_purpose.thuoc_du_an is False:
            flags.append(FlagItem(
                flag_type="TMDV_NGOAI_DU_AN", severity="ERROR",
                description=(
                    "Đất TMDV KHÔNG thuộc dự án được phê duyệt. "
                    "Không đủ điều kiện làm TSBĐ theo quy định."
                ),
                affected_field="land_purpose.thuoc_du_an",
            ))
            warnings.append(
                "⛔ ĐẤT TMDV NGOÀI DỰ ÁN: Không đủ điều kiện TSBĐ. "
                "Cần xem xét loại khỏi danh mục tài sản đảm bảo."
            )
            print("[B3] ⛔ Flag: TMDV_NGOAI_DU_AN")
        elif (
            land_purpose.thuoc_du_an is True
            and land_purpose.nguon_xac_dinh_du_an == "web_search"
            and not land_purpose.can_cu_phap_ly_du_an.strip()
        ):
            flags.append(FlagItem(
                flag_type="TMDV_DU_AN_XAC_MINH_WEB", severity="WARNING",
                description=(
                    "Đất TMDV được xác định 'thuộc dự án' qua tra cứu web bổ sung, nhưng KHÔNG "
                    "tìm được số quyết định/căn cứ pháp lý chính thức (chỉ có mô tả marketing/tham "
                    "khảo từ nguồn thương mại). Cần cán bộ tín dụng xác minh thủ công trước khi kết luận."
                ),
                affected_field="land_purpose.can_cu_phap_ly_du_an",
            ))
            warnings.append(
                "⚠️ ĐẤT TMDV: Web search kết luận thuộc dự án nhưng chưa có căn cứ pháp lý "
                "chính thức — cần xác minh thủ công."
            )
            print("[B3] ⚠️ Flag: TMDV_DU_AN_XAC_MINH_WEB (thiếu căn cứ pháp lý sau web search)")
        elif land_purpose.thuoc_du_an is True:
            print("[B3] ✅ Đất TMDV thuộc dự án, có căn cứ pháp lý, đủ điều kiện.")
        else:
            if not any(f.flag_type == "TMDV_CAN_XAC_MINH_THU_CONG" for f in flags):
                flags.append(FlagItem(
                    flag_type="TMDV_CAN_XAC_MINH_THU_CONG", severity="WARNING",
                    description=(
                        "Đất TMDV chưa xác định được có thuộc dự án được phê duyệt hay không, kể cả "
                        "sau khi đã tra cứu web bổ sung (nếu có). Cần cán bộ tín dụng xác minh thủ công."
                    ),
                    affected_field="land_purpose.thuoc_du_an",
                ))
            warnings.append(
                "⚠️ ĐẤT TMDV: Chưa xác định được có thuộc dự án không. Cần kiểm tra thêm."
            )
            print("[B3] ⚠️ TMDV — chưa xác định thuộc dự án.")

    print("[B3] Hoàn thành.")
    return identity_check, flags, warnings, notes   # ← thêm identity_check
