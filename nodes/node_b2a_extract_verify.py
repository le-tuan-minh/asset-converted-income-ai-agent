"""
B2a — Gọi Groq LLM trích xuất owner_info/asset_info/identity_check/land_purpose
từ raw_text của 1 tài sản, sau đó đối chiếu rule-based (TMDV, diện tích, nhân thân).

Async (llm.ainvoke thay vì llm.invoke) để node_b2_process_assets có thể chạy
song song (asyncio.gather) việc xử lý nhiều tài sản trong cùng 1 hồ sơ, thay
vì gọi Groq API tuần tự từng tài sản một.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from schemas import DocumentItem, DOCUMENT_CATEGORY_MAP, OwnerInfo, AssetInfo, IdentityCheckResult, LandPurposeResult, FlagItem
from cores.land_rules import detect_tmdv_rule_based
from cores.identity_rules import compare_names, describe_mismatch_reason
from cores.area_rules import compute_dien_tich_du_dieu_kien_parts, cross_check_area_totals
from utils.llm_config import get_llm
from utils.parsing_utils import parse_json_safe

SYSTEM_PROMPT = """Bạn là chuyên gia thẩm định tín dụng ngân hàng Việt Nam, nắm vững
Luật Đất đai 2024 và các văn bản hướng dẫn thi hành (Thông tư 08/2024/TT-BTNMT về
mã ký hiệu loại đất). Bạn sẽ nhận nội dung OCR (có thể có lỗi ký tự) của các
giấy tờ pháp lý CỦA MỘT (1) TÀI SẢN DUY NHẤT: giấy tờ nhân thân (CCCD/CMTND),
Giấy chứng nhận QSDĐ (GCN), và (nếu có) Hợp đồng mua bán/văn bản chuyển
nhượng/hợp đồng thế chấp liên quan CHỈ tới tài sản này.

Nhiệm vụ của bạn:
1. Trích xuất owner_info: họ tên, số CCCD, số CMTND cũ, ngày sinh, địa chỉ
   thường trú — LẤY TỪ CCCD/CMTND.
2. Trích xuất asset_info: số GCN, chủ sử dụng gốc, chủ sử dụng hiện tại (sau
   biến động gần nhất), lịch sử biến động, ngày cấp GCN, ngày chuyển nhượng
   gần nhất, mục đích sử dụng đất + mã ký hiệu (ONT/ODT/CLN/LUC/LUK/NKH/NTS/
   TMD/SKC...), địa chỉ tài sản, các loại diện tích (tổng/đất ở/nhà ở/NN/NTS/
   TMDV), có thông tin tặng cho hay không, thuộc dự án hay không (nếu là
   TMDV), nguồn gốc hình thành tài sản.
   QUAN TRỌNG: cũng trích xuất ĐỘC LẬP (không suy luận/đồng bộ theo GCN) tên
   bên mua/bên bán và số CCCD bên mua ghi TRÊN CHÍNH hợp đồng mua bán/văn bản
   chuyển nhượng (nếu có) vào ben_mua_hop_dong, ben_mua_so_cccd_hop_dong,
   ben_ban_hop_dong.
3. Trích xuất identity_check: so khớp chủ tài sản (ưu tiên so với chủ sử dụng
   hiện tại, nếu không rõ thì so với chủ gốc — ghi rõ matched_against là
   "chu_hien_tai" hoặc "chu_goc" hoặc "khong_ro"), phát hiện tặng cho/thừa kế,
   xác định ngày hình thành tài sản.
4. Trích xuất land_purpose: mục đích sử dụng, mã ký hiệu, is_tmdv, thuoc_du_an
   (chỉ set true/false nếu có căn cứ RÕ RÀNG trong hồ sơ, nếu không đủ căn cứ
   thì để null). KHÔNG cần tự tính diện tích đủ điều kiện quy đổi — hệ thống
   sẽ tự lấy từ asset_info.dien_tich_dat_o / dien_tich_nha_o.

QUAN TRỌNG — PHÂN BIỆT ĐẤT NÔNG NGHIỆP (NN) VÀ ĐẤT NUÔI TRỒNG THỦY SẢN (NTS):
Đây là 2 field diện tích RIÊNG BIỆT, dễ bị nhầm lẫn nhất khi trích xuất:
  - dien_tich_nn: CHỈ gồm đất trồng cây lâu năm (CLN) + lúa (LUC/LUK) + đất
    nông nghiệp khác (NKH). TUYỆT ĐỐI KHÔNG cộng đất nuôi trồng thủy sản vào
    field này, dù trên GCN 2 dòng này có thể nằm gần nhau trong cùng 1 bảng.
  - dien_tich_nts: diện tích đất nuôi trồng thủy sản (mã NTS), PHẢI ghi RIÊNG,
    đọc TỪNG DÒNG trong bảng diện tích của GCN, không gộp 2 dòng lại.

TỰ KIỂM TRA TRƯỚC KHI TRẢ JSON: sau khi điền xong các trường diện tích, hãy tự
cộng thử dien_tich_dat_o + dien_tich_nn + dien_tich_nts + dien_tich_tmdv rồi so
với dien_tich_tong ghi trên GCN. Nếu 2 số này lệch nhau đáng kể, RÀ SOÁT LẠI
xem có phải bạn đã gộp nhầm đất nuôi trồng thủy sản vào dien_tich_nn, hoặc bỏ
sót 1 dòng nào đó trong bảng diện tích — sửa lại cho đúng TRƯỚC khi trả JSON.

CHỈ trả về JSON đúng theo cấu trúc (không thêm chữ nào khác, không markdown):
{
  "owner_info": {"ho_ten": "", "so_cccd": "", "so_cmtnd_cu": "", "ngay_sinh": "", "dia_chi_thuong_tru": ""},
  "asset_info": {
    "so_gcn": "", "chu_su_dung_goc": "", "chu_su_dung_hien_tai": "",
    "bien_dong_lich_su": [{"ngay": "", "noi_dung": "", "chu_moi": ""}],
    "ngay_cap_gcn": "", "ngay_chuyen_nhuong": "", "muc_dich_su_dung": "", "ma_ky_hieu_dat": "",
    "dia_chi_tai_san": "", "dien_tich_tong": "", "dien_tich_dat_o": "", "dien_tich_nha_o": "",
    "dien_tich_nn": "", "dien_tich_nts": "", "dien_tich_tmdv": "", "co_thong_tin_tang_cho": false,
    "thuoc_du_an": null, "ten_du_an": "", "can_cu_phap_ly_du_an": "", "nguon_goc_tai_san": "",
    "ben_mua_hop_dong": "", "ben_mua_so_cccd_hop_dong": "", "ben_ban_hop_dong": ""
  },
  "identity_check": {
    "owner_matched": false, "matched_against": "khong_ro", "mismatch_fields": [],
    "is_tang_cho": false, "is_thua_ke": false, "asset_formation_date": "", "asset_formation_note": ""
  },
  "land_purpose": {
    "muc_dich": "", "ma_ky_hieu_dat": "", "is_tmdv": false,
    "thuoc_du_an": null, "ten_du_an": "", "can_cu_phap_ly_du_an": "",
    "nguon_xac_dinh_du_an": "chua_xac_dinh", "warning_tmdv": ""
  }
}
Nếu không tìm thấy thông tin, để chuỗi rỗng "" (không phải null), trừ các
field kiểu boolean/Optional đã nêu rõ ở trên.
"""


def _build_grouped_text(documents: list[DocumentItem]) -> dict[str, str]:
    """Gom raw_text theo nhóm nghiệp vụ, CHỈ trong phạm vi tài liệu của 1 tài sản."""
    groups: dict[str, list[str]] = {"nhan_than": [], "gcn": [], "hop_dong": []}
    for doc in documents:
        category = DOCUMENT_CATEGORY_MAP[doc.doc_type]
        if category == "nhan_than":
            groups["nhan_than"].append(f"[{doc.filename}]\n{doc.raw_text}")
        elif category == "gcn":
            groups["gcn"].append(f"[{doc.filename}]\n{doc.raw_text}")
        elif category in ("chuyen_nhuong", "the_chap"):
            groups["hop_dong"].append(f"[{doc.filename}]\n{doc.raw_text}")
    return {
        "nhan_than_text": "\n\n".join(groups["nhan_than"]),
        "gcn_text": "\n\n".join(groups["gcn"]),
        "hop_dong_text": "\n\n".join(groups["hop_dong"]),
    }


def _normalize_matched_against(raw_value) -> str:
    v = str(raw_value or "").strip().lower()
    if "hien" in v:
        return "chu_hien_tai"
    if "goc" in v:
        return "chu_goc"
    return "khong_ro"


def _safe_build(model_cls, raw_data, label, notes):
    """
    Build 1 model Pydantic từ dict do LLM trả về.

    QUAN TRỌNG: nếu 1-2 field bị LLM trả sai kiểu (vd field enum nhận nhầm
    câu trả lời tự do), KHÔNG được vứt bỏ toàn bộ object — chỉ bỏ đúng
    (các) field lỗi đó và build lại, giữ nguyên mọi field khác đã trích xuất
    đúng. Đây là lỗi thực tế đã gặp: LLM trả sai `nguon_xac_dinh_du_an`
    khiến cả `land_purpose` (kể cả muc_dich, is_tmdv, dien_tich_dat_o_du_dieu_kien
    đã đúng) bị mất trắng vì Pydantic v2 reject nguyên object khi có 1 field
    sai kiểu.

    Với các field enum đã biết dễ vỡ (nguon_xac_dinh_du_an, matched_against),
    schemas.py đã có field_validator(mode="before") coerce trước — cơ chế ở
    đây là lớp phòng vệ THỨ HAI, phòng khi LLM trả sai kiểu ở field khác mà
    ta chưa lường trước.
    """
    from pydantic import ValidationError

    data = dict(raw_data or {})
    removed_fields: list[str] = []

    for _ in range(5):  # tối đa 5 vòng gỡ field lỗi, tránh vòng lặp vô hạn
        try:
            model = model_cls(**data)
            if removed_fields:
                warn_msg = (
                    f"[B2a] Trường {removed_fields} trong '{label}' bị LLM trả sai kiểu/giá trị, "
                    f"đã bỏ (các) trường này và dùng mặc định RIÊNG cho chúng — "
                    f"các trường khác trong '{label}' vẫn giữ nguyên giá trị LLM trích xuất."
                )
                print(warn_msg)
                return model, warn_msg
            return model, None
        except ValidationError as exc:
            bad_fields = list({err["loc"][0] for err in exc.errors() if err.get("loc")})
            if not bad_fields or all(f in removed_fields for f in bad_fields):
                break  # không gỡ được thêm gì mới → dừng, rơi xuống fallback bên dưới
            for f in bad_fields:
                data.pop(f, None)
                if f not in removed_fields:
                    removed_fields.append(f)
        except Exception:
            break

    # Fallback cuối cùng: vẫn lỗi dù đã gỡ field → dùng default toàn bộ (như cũ)
    try:
        return model_cls(**data), None
    except Exception as exc:
        warn_msg = (
            f"[B2a] Dữ liệu '{label}' từ LLM không hợp lệ ({exc}). "
            f"Dùng giá trị mặc định cho '{label}', cần kiểm tra thủ công."
        )
        print(warn_msg)
        return model_cls(), warn_msg


def _cross_check_tmdv_rule_based(land_purpose, asset_info, grouped, flags, notes):
    # detect_tmdv_rule_based nhận 1 CHUỖI text (không phải dict) — gộp toàn bộ
    # text của GCN + hợp đồng/thế chấp thuộc tài sản này trước khi quét.
    full_text = f"{grouped.get('gcn_text', '')}\n{grouped.get('hop_dong_text', '')}"
    signal = detect_tmdv_rule_based(full_text)

    if signal["is_tmdv_signal"] and not land_purpose.is_tmdv:
        # ĐÃ SỬA (fix #1): trước đây chỉ flag cảnh báo rồi để nguyên is_tmdv=False
        # đi tiếp, khiến B2b (web search) và các rule TMDV ở B3 bị bỏ qua oan dù
        # văn bản GCN ghi rõ "Đất thương mại, dịch vụ" — field này là string-match
        # trực tiếp trên văn bản gốc, đáng tin cậy hơn phán đoán của LLM, nên giờ
        # tự động ghi đè thay vì chỉ cảnh báo.
        land_purpose = land_purpose.model_copy(update={"is_tmdv": True})
        flags.append(FlagItem(
            flag_type="TMDV_KHONG_KHOP_RULE_BASED",
            severity="WARNING",
            description=(
                "Rule-based phát hiện tín hiệu đất thương mại dịch vụ (TMD) trong văn bản, "
                "nhưng LLM không gắn is_tmdv=True. Đã tự động ghi đè is_tmdv=True dựa trên "
                "rule-based (string-match trực tiếp trên văn bản gốc, đáng tin cậy hơn cho "
                "field này). Cần cán bộ tín dụng đối chiếu lại để xác nhận."
            ),
            affected_field="land_purpose.is_tmdv",
        ))
        notes.append(
            "[B2a] Rule-based override: is_tmdv=True (LLM đã bỏ sót, rule-based "
            "phát hiện tín hiệu TMD rõ ràng trong văn bản)."
        )

    if land_purpose.is_tmdv and land_purpose.thuoc_du_an is None:
        if signal["project_keyword_hit"] and not signal["negative_project_signal"]:
            notes.append(
                "[B2a] Rule-based tìm thấy tín hiệu liên quan dự án, nhưng chưa đủ khẳng định. "
                "Giữ thuoc_du_an=null, cần xác minh thêm (vd bước tra cứu bổ sung ở B2b)."
            )
        elif signal["negative_project_signal"]:
            notes.append(
                "[B2a] Rule-based phát hiện cụm từ phủ định dự án (vd 'không thuộc dự án', "
                "'đất xen kẹt') trong văn bản, nhưng LLM chưa gắn cờ thuoc_du_an=False. "
                "Cần cán bộ tín dụng đối chiếu lại — hệ thống KHÔNG tự động set False."
            )
        if signal["decision_numbers_found"] and not land_purpose.can_cu_phap_ly_du_an:
            notes.append(
                "[B2a] Rule-based tìm thấy số quyết định có thể liên quan tới dự án: "
                f"{signal['decision_numbers_found']}. Chỉ mang tính tham khảo, KHÔNG tự động "
                "gán vào can_cu_phap_ly_du_an — cần cán bộ tín dụng xác minh."
            )

    if land_purpose.thuoc_du_an is not None and land_purpose.nguon_xac_dinh_du_an == "chua_xac_dinh":
        land_purpose = land_purpose.model_copy(update={"nguon_xac_dinh_du_an": "ho_so_noi_bo"})
    
    # ĐÃ SỬA (fix #4): LLM đôi khi kết luận thuoc_du_an=True chỉ vì địa chỉ
    # tài sản có tên "nghe giống" dự án (vd "Golden Hills City" nằm trong địa
    # chỉ), dù KHÔNG có căn cứ pháp lý cụ thể (số quyết định phê duyệt chủ
    # trương đầu tư / GCN đầu tư) trong hồ sơ. Tên địa chỉ KHÔNG phải căn cứ
    # pháp lý — hạ về null và bắt buộc xác minh bổ sung (qua B2b/web search
    # hoặc cán bộ tín dụng), thay vì chấp nhận kết luận thiếu căn cứ của LLM.
    if (
        land_purpose.is_tmdv
        and land_purpose.thuoc_du_an is True
        and land_purpose.nguon_xac_dinh_du_an == "ho_so_noi_bo"
        and not land_purpose.can_cu_phap_ly_du_an.strip()
        and not signal["decision_numbers_found"]
    ):
        flags.append(FlagItem(
            flag_type="TMDV_CAN_XAC_MINH_THU_CONG",
            severity="WARNING",
            description=(
                "LLM kết luận 'thuộc dự án' chỉ dựa vào tên gọi trong địa chỉ tài sản "
                "(vd tên khu đô thị/dự án xuất hiện trong địa chỉ), KHÔNG có căn cứ pháp lý "
                "cụ thể (số quyết định phê duyệt chủ trương đầu tư / giấy chứng nhận đầu tư) "
                "trong hồ sơ. Đã hạ về 'chưa xác định' để bắt buộc xác minh bổ sung."
            ),
            affected_field="land_purpose.thuoc_du_an",
        ))
        notes.append(
            "[B2a] Downgrade: thuoc_du_an True→None vì thiếu căn cứ pháp lý cụ thể "
            "(chỉ dựa vào tên gọi trong địa chỉ, chưa đủ để khẳng định thuộc dự án)."
        )
        land_purpose = land_purpose.model_copy(update={
            "thuoc_du_an": None,
            "nguon_xac_dinh_du_an": "chua_xac_dinh",
        })

    if land_purpose.ma_ky_hieu_dat and not asset_info.ma_ky_hieu_dat:
        asset_info = asset_info.model_copy(update={"ma_ky_hieu_dat": land_purpose.ma_ky_hieu_dat})
    elif asset_info.ma_ky_hieu_dat and not land_purpose.ma_ky_hieu_dat:
        land_purpose = land_purpose.model_copy(update={"ma_ky_hieu_dat": asset_info.ma_ky_hieu_dat})

    if land_purpose.is_tmdv and land_purpose.thuoc_du_an is None and not land_purpose.warning_tmdv:
        land_purpose = land_purpose.model_copy(update={
            "warning_tmdv": (
                "Đất thương mại, dịch vụ (mã TMD) nhưng chưa xác định được có thuộc dự án "
                "được phê duyệt hay không từ hồ sơ hiện có. Cần xác minh bổ sung."
            )
        })

    return land_purpose, asset_info


def _cross_check_area_rule_based(asset_info, land_purpose, flags, notes):
    # ĐÃ SỬA (fix #2): không còn cộng gộp — tính lại tất định 2 con số riêng.
    dat_o_computed, nha_o_computed = compute_dien_tich_du_dieu_kien_parts(
        asset_info.dien_tich_dat_o, asset_info.dien_tich_nha_o
    )
    notes.append(
        f"[B2a] Diện tích đủ điều kiện quy đổi (tính tất định, KHÔNG cộng gộp): "
        f"đất ở = {dat_o_computed} m², nhà ở = {nha_o_computed} m²."
    )
    land_purpose = land_purpose.model_copy(update={
        "dien_tich_dat_o_du_dieu_kien": dat_o_computed,
        "dien_tich_nha_o_du_dieu_kien": nha_o_computed,
    })

    result = cross_check_area_totals(
        asset_info.dien_tich_tong,
        asset_info.dien_tich_dat_o,
        asset_info.dien_tich_nn,
        asset_info.dien_tich_nts,
        asset_info.dien_tich_tmdv,
    )
    if not result["ok"]:
        flags.append(FlagItem(
            flag_type="DIEN_TICH_KHONG_KHOP",
            severity="WARNING",
            description=result["message"],
            affected_field="asset_info.dien_tich_*",
        ))
        notes.append(f"[B2a] {result['message']}")
    return land_purpose


def _cross_check_identity_rule_based(identity_check, owner_info, asset_info, flags, notes, warnings):
    """LỚP 1: CCCD (owner_info.ho_ten) vs GCN (chu_su_dung_hien_tai/goc)."""
    if not owner_info.ho_ten:
        return identity_check

    candidate_name = (
        asset_info.chu_su_dung_hien_tai
        if identity_check.matched_against != "chu_goc"
        else asset_info.chu_su_dung_goc
    ) or asset_info.chu_su_dung_hien_tai or asset_info.chu_su_dung_goc

    if not candidate_name:
        return identity_check

    # ĐÃ SỬA (fix #7): luôn tính + lưu similarity, kể cả khi LLM đã báo khớp —
    # để cán bộ tín dụng thấy được đây là "khớp tuyệt đối" hay "khớp gần đúng"
    # (vd do lỗi OCR sai 1 ký tự có dấu), thay vì chỉ có True/False không phân
    # biệt mức độ. Việc này CHỈ để hiển thị minh bạch, KHÔNG thay đổi ngưỡng
    # quyết định owner_matched hiện có (vẫn dựa trên exact_match tuyệt đối).
    result = compare_names(owner_info.ho_ten, candidate_name)
    if result["has_data"]:
        identity_check = identity_check.model_copy(update={
            "owner_name_similarity": result["similarity"],
        })

    if identity_check.owner_matched and result["has_data"] and not result["exact_match"]:
        # compare_names là lưới an toàn: CHỈ tin "khớp" khi exact_match sau chuẩn
        # hoá (bỏ dấu + bỏ danh xưng). Nếu LLM tự tin owner_matched=True nhưng rule
        # based không thấy khớp tuyệt đối → hạ xuống False, không được tự nới lỏng.
        reason = describe_mismatch_reason(result)
        identity_check = identity_check.model_copy(update={
            "owner_matched": False,
            "mismatch_fields": list(set(identity_check.mismatch_fields + ["ho_ten (CCCD vs GCN)"])),
        })
        flags.append(FlagItem(
            flag_type="CHU_TAI_SAN_LECH_RULE_BASED",
            severity="ERROR",
            description=(
                f"Rule-based phát hiện lệch tên: CCCD ghi '{owner_info.ho_ten}' nhưng GCN ghi "
                f"'{candidate_name}' ({reason}). LLM đã kết luận sai owner_matched=True."
            ),
            affected_field="owner_info.ho_ten / asset_info.chu_su_dung",
        ))
        notes.append(f"[B2a] Rule-based hạ owner_matched True→False: {reason}")
        warnings.append(f"⛔ Rule-based phát hiện lệch tên chủ tài sản: {reason}")
    return identity_check


def _cross_check_contract_identity_rule_based(identity_check, asset_info, flags, notes, warnings):
    """LỚP 2: GCN (chu_su_dung_hien_tai) vs Hợp đồng mua bán (ben_mua_hop_dong)."""
    if not asset_info.ben_mua_hop_dong or not asset_info.chu_su_dung_hien_tai:
        return identity_check

    result = compare_names(asset_info.chu_su_dung_hien_tai, asset_info.ben_mua_hop_dong)
    if result["has_data"] and not result["exact_match"]:
        reason = describe_mismatch_reason(result)
        flags.append(FlagItem(
            flag_type="CHU_TAI_SAN_KHONG_DONG_NHAT_GIUA_HO_SO",
            severity="WARNING",
            description=(
                f"Tên chủ sử dụng trên GCN ('{asset_info.chu_su_dung_hien_tai}') không khớp bên mua "
                f"trên hợp đồng ('{asset_info.ben_mua_hop_dong}'): {reason}."
            ),
            affected_field="asset_info.chu_su_dung_hien_tai / ben_mua_hop_dong",
        ))
        notes.append(f"[B2a] Lệch tên GCN vs Hợp đồng: {reason}")
        warnings.append(f"⚠️ Tên trên GCN và Hợp đồng mua bán không khớp: {reason}")
    return identity_check


async def _extract_llm(documents: list[DocumentItem]) -> tuple[dict, dict, list[str]]:
    """Gọi Groq LLM (async) extract JSON cho 1 tài sản. Trả về (data, grouped_text, notes)."""
    grouped = _build_grouped_text(documents)
    notes = []
    user_content = (
        f"### GIẤY TỜ NHÂN THÂN (CCCD/CMTND)\n{grouped['nhan_than_text'] or '(không có)'}\n\n"
        f"### GIẤY CHỨNG NHẬN QSDĐ (GCN)\n{grouped['gcn_text'] or '(không có)'}\n\n"
        f"### HỢP ĐỒNG / VĂN BẢN CHUYỂN NHƯỢNG / THẾ CHẤP (nếu có)\n{grouped['hop_dong_text'] or '(không có)'}"
    )
    llm = get_llm()
    resp = await llm.ainvoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_content)])
    data = parse_json_safe(resp.content)
    return data, grouped, notes


async def verify_asset_async(documents: list[DocumentItem]) -> tuple[
    OwnerInfo, AssetInfo, IdentityCheckResult, LandPurposeResult, list[FlagItem], list[str], list[str], str | None
]:
    """
    B2a cho MỘT tài sản (async, gọi Groq qua llm.ainvoke để nhiều tài sản có
    thể được xử lý song song ở node_b2_process_assets). Trả về:
    (owner_info, asset_info, identity_check, land_purpose, flags, warnings, notes, error)
    """
    flags: list[FlagItem] = []
    warnings: list[str] = []
    notes: list[str] = []

    try:
        data, grouped, _ = await _extract_llm(documents)
    except Exception as exc:
        error_msg = f"[B2a] Lỗi gọi/parse LLM: {exc}"
        print(error_msg)
        notes.append(error_msg)
        return OwnerInfo(), AssetInfo(), IdentityCheckResult(), LandPurposeResult(), flags, warnings, notes, str(exc)

    ic_data = data.get("identity_check", {}) or {}
    if "matched_against" in ic_data:
        ic_data["matched_against"] = _normalize_matched_against(ic_data.get("matched_against"))

    owner_info, w1 = _safe_build(OwnerInfo, data.get("owner_info"), "owner_info", notes)
    asset_info, w2 = _safe_build(AssetInfo, data.get("asset_info"), "asset_info", notes)
    identity_check, w3 = _safe_build(IdentityCheckResult, ic_data, "identity_check", notes)
    land_purpose, w4 = _safe_build(LandPurposeResult, data.get("land_purpose"), "land_purpose", notes)

    for w in (w1, w2, w3, w4):
        if w:
            notes.append(w)
            flags.append(FlagItem(
                flag_type="OCR_THIEU_DU_LIEU", severity="WARNING",
                description=w, affected_field="B2a_llm_output",
            ))

    land_purpose, asset_info = _cross_check_tmdv_rule_based(land_purpose, asset_info, grouped, flags, notes)
    land_purpose = _cross_check_area_rule_based(asset_info, land_purpose, flags, notes)
    identity_check = _cross_check_identity_rule_based(identity_check, owner_info, asset_info, flags, notes, warnings)
    identity_check = _cross_check_contract_identity_rule_based(identity_check, asset_info, flags, notes, warnings)

    notes.append("B2a hoàn thành: LLM extract thành công.")
    print(
        f"[B2a] owner_matched={identity_check.owner_matched} "
        f"(similarity={identity_check.owner_name_similarity}) | "
        f"muc_dich={land_purpose.muc_dich} ({land_purpose.ma_ky_hieu_dat or 'N/A'}) | "
        f"is_tmdv={land_purpose.is_tmdv} | thuoc_du_an={land_purpose.thuoc_du_an} "
        f"(nguồn={land_purpose.nguon_xac_dinh_du_an}) | "
        f"dien_tich_dat_o_du_dieu_kien={land_purpose.dien_tich_dat_o_du_dieu_kien} | "
        f"dien_tich_nha_o_du_dieu_kien={land_purpose.dien_tich_nha_o_du_dieu_kien}"
    )
    return owner_info, asset_info, identity_check, land_purpose, flags, warnings, notes, None
