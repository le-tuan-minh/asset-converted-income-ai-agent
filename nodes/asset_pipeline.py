"""
Pipeline xử lý cấp TÀI SẢN: B2 (Groq LLM extract & verify) → B2c (web search
TMDV nếu cần) → B3 (rule-based flag engine).

Được gọi LẶP LẠI cho từng AssetGroupCandidate đã được con người xác nhận ở
B1c — mỗi lần gọi CHỈ thấy raw_text của các file thuộc về đúng 1 tài sản
(GCN + hợp đồng/thế chấp liên quan) + các file dùng chung (CCCD), hoàn toàn
tách biệt dữ liệu giữa các tài sản để tránh LLM nhầm lẫn/pha trộn thông tin
giữa 2 thửa đất khác nhau.
"""
from __future__ import annotations
import json
import os
import re
from datetime import date, datetime

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from schemas import (
    DocumentItem, DocumentType, DOCUMENT_CATEGORY_MAP, AssetGroupCandidate, AssetResult,
    OwnerInfo, AssetInfo, IdentityCheckResult, LandPurposeResult, FlagItem,
)
from nodes.land_rules import detect_tmdv_rule_based, LAND_USE_CODE_REFERENCE
from nodes.identity_rules import compare_names, describe_mismatch_reason
from nodes.area_rules import compute_dien_tich_du_dieu_kien_parts, cross_check_area_totals
from nodes.document_classifier import _strip_accents

MAX_TOOL_CALLS = 4


# ─────────────────────────────────────────────
# LLM setup
# ─────────────────────────────────────────────

def _get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY chưa được set trong .env")
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=api_key)


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


def _parse_json_safe(raw: str) -> dict:
    raw = (raw or "").strip()
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
    if match:
        raw = match.group(1)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        brace_match = re.search(r"\{[\s\S]+\}", raw)
        if brace_match:
            return json.loads(brace_match.group(0))
        raise


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
                    f"[B2] Trường {removed_fields} trong '{label}' bị LLM trả sai kiểu/giá trị, "
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
            f"[B2] Dữ liệu '{label}' từ LLM không hợp lệ ({exc}). "
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
        flags.append(FlagItem(
            flag_type="TMDV_KHONG_KHOP_RULE_BASED",
            severity="WARNING",
            description=(
                "Rule-based phát hiện tín hiệu đất thương mại dịch vụ (TMD) trong văn bản, "
                "nhưng LLM không gắn is_tmdv=True. Cần cán bộ tín dụng đối chiếu lại."
            ),
            affected_field="land_purpose.is_tmdv",
        ))
        notes.append("[B2] Rule-based: có tín hiệu TMDV nhưng LLM bỏ sót is_tmdv.")

    if land_purpose.is_tmdv and land_purpose.thuoc_du_an is None:
        if signal["project_keyword_hit"] and not signal["negative_project_signal"]:
            notes.append(
                "[B2] Rule-based tìm thấy tín hiệu liên quan dự án, nhưng chưa đủ khẳng định. "
                "Giữ thuoc_du_an=null, cần xác minh thêm (vd bước tra cứu bổ sung ở B2c)."
            )
        elif signal["negative_project_signal"]:
            notes.append(
                "[B2] Rule-based phát hiện cụm từ phủ định dự án (vd 'không thuộc dự án', "
                "'đất xen kẹt') trong văn bản, nhưng LLM chưa gắn cờ thuoc_du_an=False. "
                "Cần cán bộ tín dụng đối chiếu lại — hệ thống KHÔNG tự động set False."
            )
        if signal["decision_numbers_found"] and not land_purpose.can_cu_phap_ly_du_an:
            notes.append(
                "[B2] Rule-based tìm thấy số quyết định có thể liên quan tới dự án: "
                f"{signal['decision_numbers_found']}. Chỉ mang tính tham khảo, KHÔNG tự động "
                "gán vào can_cu_phap_ly_du_an — cần cán bộ tín dụng xác minh."
            )

    if land_purpose.thuoc_du_an is not None and land_purpose.nguon_xac_dinh_du_an == "chua_xac_dinh":
        land_purpose = land_purpose.model_copy(update={"nguon_xac_dinh_du_an": "ho_so_noi_bo"})

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
        f"[B2] Diện tích đủ điều kiện quy đổi (tính tất định, KHÔNG cộng gộp): "
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
        notes.append(f"[B2] {result['message']}")
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
        notes.append(f"[B2] Rule-based hạ owner_matched True→False: {reason}")
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
        notes.append(f"[B2] Lệch tên GCN vs Hợp đồng: {reason}")
        warnings.append(f"⚠️ Tên trên GCN và Hợp đồng mua bán không khớp: {reason}")
    return identity_check


def _extract_llm(documents: list[DocumentItem]) -> tuple[dict, dict, list[str]]:
    """Gọi Groq LLM extract JSON cho 1 tài sản. Trả về (data, grouped_text, notes)."""
    grouped = _build_grouped_text(documents)
    notes = []
    user_content = (
        f"### GIẤY TỜ NHÂN THÂN (CCCD/CMTND)\n{grouped['nhan_than_text'] or '(không có)'}\n\n"
        f"### GIẤY CHỨNG NHẬN QSDĐ (GCN)\n{grouped['gcn_text'] or '(không có)'}\n\n"
        f"### HỢP ĐỒNG / VĂN BẢN CHUYỂN NHƯỢNG / THẾ CHẤP (nếu có)\n{grouped['hop_dong_text'] or '(không có)'}"
    )
    llm = _get_llm()
    resp = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_content)])
    data = _parse_json_safe(resp.content)
    return data, grouped, notes


def verify_asset(documents: list[DocumentItem]) -> tuple[
    OwnerInfo, AssetInfo, IdentityCheckResult, LandPurposeResult, list[FlagItem], list[str], list[str], str | None
]:
    """
    B2 cho MỘT tài sản. Trả về:
    (owner_info, asset_info, identity_check, land_purpose, flags, warnings, notes, error)
    """
    flags: list[FlagItem] = []
    warnings: list[str] = []
    notes: list[str] = []

    try:
        data, grouped, _ = _extract_llm(documents)
    except Exception as exc:
        error_msg = f"[B2] Lỗi gọi/parse LLM: {exc}"
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
                description=w, affected_field="B2_llm_output",
            ))

    land_purpose, asset_info = _cross_check_tmdv_rule_based(land_purpose, asset_info, grouped, flags, notes)
    land_purpose = _cross_check_area_rule_based(asset_info, land_purpose, flags, notes)
    identity_check = _cross_check_identity_rule_based(identity_check, owner_info, asset_info, flags, notes, warnings)
    identity_check = _cross_check_contract_identity_rule_based(identity_check, asset_info, flags, notes, warnings)

    notes.append("B2 hoàn thành: LLM extract thành công.")
    print(
        f"[B2] owner_matched={identity_check.owner_matched} "
        f"(similarity={identity_check.owner_name_similarity}) | "
        f"muc_dich={land_purpose.muc_dich} ({land_purpose.ma_ky_hieu_dat or 'N/A'}) | "
        f"is_tmdv={land_purpose.is_tmdv} | thuoc_du_an={land_purpose.thuoc_du_an} "
        f"(nguồn={land_purpose.nguon_xac_dinh_du_an}) | "
        f"dien_tich_dat_o_du_dieu_kien={land_purpose.dien_tich_dat_o_du_dieu_kien} | "
        f"dien_tich_nha_o_du_dieu_kien={land_purpose.dien_tich_nha_o_du_dieu_kien}"
    )
    return owner_info, asset_info, identity_check, land_purpose, flags, warnings, notes, None


# ─────────────────────────────────────────────
# B2c — Web search bổ sung cho đất TMDV (Tavily, nếu có TAVILY_API_KEY)
# ─────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """Bạn là trợ lý tra cứu thông tin quy hoạch/dự án bất động sản
tại Việt Nam. Bạn có công cụ tavily_search để tìm kiếm trên web. Nhiệm vụ: xác
định xem thửa đất/dự án được mô tả có thuộc một dự án đầu tư đã được phê duyệt
hay không. Sau khi tra cứu, trả lời CHỈ bằng JSON:
{"thuoc_du_an": true/false/null, "ten_du_an": "", "can_cu_phap_ly_du_an": "", "tom_tat": ""}
Nếu không tìm thấy căn cứ đủ tin cậy, để thuoc_du_an=null."""

FINAL_VERDICT_INSTRUCTION = (
    "Dựa trên các kết quả tra cứu ở trên, hãy trả lời CHỈ bằng JSON đúng định dạng đã nêu, "
    "không thêm chữ nào khác, không dùng markdown."
)


def _build_tavily_tool():
    from langchain_core.tools import tool

    api_key = os.getenv("TAVILY_API_KEY")

    @tool
    def tavily_search(query: str) -> str:
        """Tìm kiếm thông tin dự án bất động sản/quy hoạch trên web qua Tavily."""
        if not api_key:
            return json.dumps([])
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=api_key)
            res = client.search(query, max_results=5)
            results = [{"url": r.get("url", ""), "content": r.get("content", "")[:500]} for r in res.get("results", [])]
            return json.dumps(results, ensure_ascii=False)
        except Exception as exc:
            return json.dumps([{"error": str(exc)}])

    return tavily_search


def _build_task_message(asset_info: AssetInfo, land_purpose: LandPurposeResult) -> str:
    return (
        f"Thửa đất địa chỉ: {asset_info.dia_chi_tai_san or '(không rõ)'}\n"
        f"Mục đích sử dụng: {land_purpose.muc_dich or asset_info.muc_dich_su_dung} "
        f"(mã {land_purpose.ma_ky_hieu_dat or asset_info.ma_ky_hieu_dat})\n"
        f"Tên dự án (nếu hồ sơ có nêu): {asset_info.ten_du_an or '(không có)'}\n"
        "Hãy tra cứu xem thửa đất/khu vực này có thuộc 1 dự án đầu tư đã được phê duyệt hay không."
    )


def _run_web_agent(asset_info: AssetInfo, land_purpose: LandPurposeResult) -> tuple[dict, list[str]]:
    tavily_tool = _build_tavily_tool()
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=os.getenv("GROQ_API_KEY"))
    llm_with_tools = llm.bind_tools([tavily_tool])

    messages = [
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        HumanMessage(content=_build_task_message(asset_info, land_purpose)),
    ]

    urls_seen: list[str] = []
    tool_calls_used = 0

    while tool_calls_used < MAX_TOOL_CALLS:
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)
        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        if not tool_calls:
            break
        for tc in tool_calls:
            if tool_calls_used >= MAX_TOOL_CALLS:
                break
            tool_calls_used += 1
            print(f"[B2c] Tool call #{tool_calls_used}: tavily_search({tc['args']})")
            result_str = tavily_tool.invoke(tc["args"])
            try:
                parsed = json.loads(result_str)
                if isinstance(parsed, list):
                    urls_seen.extend(r.get("url", "") for r in parsed if r.get("url"))
            except Exception:
                pass
            messages.append(ToolMessage(content=result_str, tool_call_id=tc["id"]))

    messages.append(HumanMessage(content=FINAL_VERDICT_INSTRUCTION))
    final_msg = llm.invoke(messages)
    verdict = _parse_json_safe(final_msg.content)
    urls_unique = list(dict.fromkeys(u for u in urls_seen if u))
    return verdict, urls_unique


def tmdv_websearch_asset(asset_info: AssetInfo, land_purpose: LandPurposeResult, notes: list[str]) -> LandPurposeResult:
    """B2c cho 1 tài sản — chỉ chạy nếu is_tmdv=True và thuoc_du_an chưa xác định."""
    if not (land_purpose.is_tmdv and land_purpose.thuoc_du_an is None):
        return land_purpose

    if not os.getenv("TAVILY_API_KEY"):
        notes.append("[B2c] Bỏ qua tra cứu web: chưa cấu hình TAVILY_API_KEY.")
        return land_purpose

    print("[B2c] Đất TMDV chưa xác định thuộc dự án — tra cứu web bổ sung qua Tavily...")
    try:
        verdict, urls = _run_web_agent(asset_info, land_purpose)
    except Exception as exc:
        notes.append(f"[B2c] Lỗi tra cứu web: {exc}")
        return land_purpose

    thuoc_du_an = verdict.get("thuoc_du_an", None)
    if thuoc_du_an is not None:
        land_purpose = land_purpose.model_copy(update={
            "thuoc_du_an": bool(thuoc_du_an),
            "ten_du_an": verdict.get("ten_du_an", "") or land_purpose.ten_du_an,
            "can_cu_phap_ly_du_an": verdict.get("can_cu_phap_ly_du_an", "") or land_purpose.can_cu_phap_ly_du_an,
            "nguon_xac_dinh_du_an": "web_search",
            "web_verification_sources": urls,
            "web_verification_summary": verdict.get("tom_tat", ""),
        })
        notes.append(f"[B2c] Web search kết luận thuoc_du_an={thuoc_du_an}.")
    else:
        notes.append("[B2c] Web search không tìm được căn cứ đủ tin cậy — giữ thuoc_du_an=null.")
    return land_purpose


# ─────────────────────────────────────────────
# B3 — Flag & Alert Engine (rule-based, cho 1 tài sản)
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# Entry point: xử lý 1 tài sản (B2 → B2c → B3) → AssetResult
# ─────────────────────────────────────────────

def process_single_asset(group: AssetGroupCandidate, all_documents: list[DocumentItem]) -> AssetResult:
    """
    Xử lý B2 → B2c → B3 cho MỘT tài sản. Documents đưa vào LLM CHỈ gồm các
    file thuộc group.filenames + group.shared_filenames (CCCD dùng chung) —
    KHÔNG bao giờ trộn text của tài sản khác vào.
    """
    print("\n" + "#" * 60)
    print(f"# XỬ LÝ TÀI SẢN: {group.asset_id}  (GCN gợi ý: {group.so_gcn_goi_y or 'N/A'})")
    print("#" * 60)

    wanted_filenames = set(group.filenames) | set(group.shared_filenames)
    documents = [d for d in all_documents if d.filename in wanted_filenames]

    if not documents:
        return AssetResult(
            asset_id=group.asset_id,
            document_filenames=[],
            has_critical_flags=True,
            error="Không có file nào được gán cho tài sản này.",
            flags=[FlagItem(
                flag_type="OCR_THIEU_DU_LIEU", severity="ERROR",
                description=f"Nhóm tài sản '{group.asset_id}' không có file nào — không thể xử lý B2/B3.",
                affected_field="asset_groups",
            )],
        )

    owner_info, asset_info, identity_check, land_purpose, flags, warnings, notes, error = verify_asset(documents)

    if error:
        return AssetResult(
            asset_id=group.asset_id,
            document_filenames=[d.filename for d in documents],
            owner_info=owner_info, asset_info=asset_info,
            identity_check=identity_check, land_purpose=land_purpose,
            flags=flags, warnings=warnings, processing_notes=notes,
            has_critical_flags=True, error=error,
        )

    land_purpose = tmdv_websearch_asset(asset_info, land_purpose, notes)
    flags, warnings, notes = flag_asset(owner_info, asset_info, identity_check, land_purpose, flags, warnings, notes)

    has_critical = any(f.severity == "ERROR" for f in flags)

    return AssetResult(
        asset_id=group.asset_id,
        document_filenames=[d.filename for d in documents],
        owner_info=owner_info,
        asset_info=asset_info,
        identity_check=identity_check,
        land_purpose=land_purpose,
        flags=flags,
        warnings=warnings,
        processing_notes=notes,
        has_critical_flags=has_critical,
        error=None,
    )