"""
B1b - Asset Grouping Node (Reasoning AI + rule-based hỗ trợ)

Hồ sơ khách hàng trong 1 folder có thể chứa NHIỀU tài sản bảo đảm khác nhau
(vd 2 Giấy chứng nhận QSDĐ của 2 thửa đất riêng biệt, mỗi thửa có thể kèm
theo hợp đồng mua bán/thế chấp riêng). Node này:

  1. Rule-based: với MỖI file GCN, cố gắng trích số GCN / thửa đất / tờ bản
     đồ / địa chỉ bằng regex để làm "vân tay" nhận diện tài sản.
  2. Nếu chỉ có 1 GCN → chỉ có 1 tài sản, khỏi cần LLM (fallback_single).
  3. Nếu có ≥2 GCN → gọi Groq LLM (Reasoning AI), cho đọc TOÀN BỘ raw_text của
     các file GCN/hợp đồng/thế chấp cùng lúc, yêu cầu trả về JSON gom nhóm:
     file nào (hợp đồng, thế chấp, xác nhận...) thuộc về tài sản/GCN nào, dựa
     trên số thửa, số tờ bản đồ, địa chỉ, số GCN được nhắc tới chéo giữa các
     văn bản.
  4. CCCD/CMTND (nhân thân) được coi là DÙNG CHUNG cho mọi tài sản trong hồ
     sơ (giả định 1 khách hàng, nhiều tài sản) — gắn vào shared_filenames của
     mọi nhóm.
  5. Nếu LLM bỏ sót file nào (không gán được vào nhóm nào) → tự tạo nhóm
     "cần xác minh" riêng cho file đó và sinh flag để cán bộ tín dụng chú ý
     tại bước xác nhận (human-in-the-loop, B1c).

  KẾT QUẢ CỦA NODE NÀY LUÔN LÀ ĐỀ XUẤT (candidate) — không tự động chạy tiếp
  B2/B3 cho tới khi con người xác nhận/chỉnh sửa ở bước sau (node_human_confirm_grouping).
"""
from __future__ import annotations
import re

from langchain_core.messages import HumanMessage, SystemMessage

from schemas import GraphState, DocumentType, AssetGroupCandidate, FlagItem, DOCUMENT_CATEGORY_MAP
from utils.llm_config import get_llm
from utils.parsing_utils import parse_json_safe

# ─────────────────────────────────────────────
# Rule-based: trích "vân tay" tài sản từ 1 file GCN
# ─────────────────────────────────────────────

_SO_GCN_PATTERNS = [
    r"s[oố]\s*(?:ph[aá]t\s*h[aà]nh)?\s*[:\-]?\s*([A-Z]{1,3}\s?\d{5,10})",
    r"s[oố]\s*v[aà]o\s*s[oổ]\s*(?:c[aấ]p\s*gcn)?\s*[:\-]?\s*([\w\.\/]{3,20})",
]
_THUA_DAT_PATTERN = r"th[uử]a\s*(?:đ[aấ]t)?\s*s[oố]\s*[:\-]?\s*(\d+)"
_TO_BAN_DO_PATTERN = r"t[oờ]\s*b[aả]n\s*đ[oồ]\s*s[oố]\s*[:\-]?\s*(\d+)"


def _extract_gcn_fingerprint(text: str) -> dict:
    """Trích số GCN / thửa đất / tờ bản đồ bằng regex — best-effort, có thể rỗng."""
    so_gcn = ""
    for pattern in _SO_GCN_PATTERNS:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            so_gcn = m.group(1).strip()
            break

    thua = re.search(_THUA_DAT_PATTERN, text, flags=re.IGNORECASE)
    to_bd = re.search(_TO_BAN_DO_PATTERN, text, flags=re.IGNORECASE)

    return {
        "so_gcn": so_gcn,
        "thua_dat_so": thua.group(1) if thua else "",
        "to_ban_do_so": to_bd.group(1) if to_bd else "",
    }


def _build_dia_chi_goi_y(fp: dict) -> str:
    parts = []
    if fp.get("thua_dat_so"):
        parts.append(f"Thửa {fp['thua_dat_so']}")
    if fp.get("to_ban_do_so"):
        parts.append(f"Tờ BĐ {fp['to_ban_do_so']}")
    return ", ".join(parts)


# ─────────────────────────────────────────────
# LLM grouping (khi có ≥ 2 GCN)
# ─────────────────────────────────────────────

_GROUPING_SYSTEM_PROMPT = """Bạn là chuyên gia thẩm định tín dụng ngân hàng Việt Nam.
Bạn sẽ nhận được nội dung OCR của nhiều file giấy tờ pháp lý về bất động sản
(Giấy chứng nhận QSDĐ - GCN, Hợp đồng mua bán, Văn bản/Xác nhận chuyển nhượng,
Hợp đồng/Xác nhận thế chấp). Hồ sơ này có thể ứng với NHIỀU tài sản (thửa đất)
KHÁC NHAU của cùng một khách hàng.

Nhiệm vụ: xác định mỗi file (ngoại trừ CCCD/CMTND) thuộc về TÀI SẢN nào, dựa
trên các căn cứ đối chiếu chéo: số GCN, số thửa đất, số tờ bản đồ, địa chỉ/vị
trí thửa đất, tên các bên trong hợp đồng, ngày tháng liên quan.

Mỗi tài sản PHẢI có ĐÚNG 1 file GCN làm gốc (không được gộp 2 GCN vào cùng 1
tài sản, trừ khi bạn có bằng chứng RÕ RÀNG rằng 1 GCN đã bị cấp lại/thay thế
cho GCN kia — nêu rõ trong "ly_do" nếu vậy).

CHỈ trả về JSON, không thêm chữ nào khác, đúng định dạng:
{
  "assets": [
    {
      "asset_index": 1,
      "so_gcn_goi_y": "...",
      "dia_chi_goi_y": "...",
      "filenames": ["giay_chung_nhan_1.pdf", "hop_dong_mua_ban_1.pdf"],
      "ly_do": "Cùng số thửa 123, tờ bản đồ 45, cùng nhắc tên bên mua Nguyễn Văn A",
      "do_tin_cay": 0.9
    }
  ],
  "khong_gan_duoc": ["file_khong_ro.pdf"]
}

"do_tin_cay" từ 0 đến 1: mức độ tự tin của bạn rằng nhóm file này thực sự
thuộc cùng 1 tài sản. Nếu không đủ căn cứ để gán 1 file vào tài sản nào, đưa
file đó vào mảng "khong_gan_duoc" thay vì đoán bừa.
"""


def _call_llm_grouping(gcn_docs, contract_docs) -> dict:
    llm = get_llm()

    doc_blocks = []
    for d in gcn_docs + contract_docs:
        doc_blocks.append(f"### FILE: {d.filename} (loại: {d.doc_type.value})\n{d.raw_text[:3000]}")
    user_content = "\n\n".join(doc_blocks)

    resp = llm.invoke([
        SystemMessage(content=_GROUPING_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ])
    return parse_json_safe(resp.content)


# ─────────────────────────────────────────────
# Node chính
# ─────────────────────────────────────────────

def node_b1b_group_assets(state: GraphState) -> GraphState:
    """
    LangGraph node B1b — đề xuất gom nhóm tài sản (AI + rule-based).
    KHÔNG chạy nếu B1 đã có has_critical_flags=True (thiếu giấy tờ bắt buộc).
    """
    print("\n" + "=" * 60)
    print("B1b · ASSET GROUPING — AI gom nhóm tài sản (Reasoning AI)")
    print("=" * 60)

    flags = list(state.flags)
    notes = list(state.processing_notes)

    docs = state.documents
    gcn_docs = [d for d in docs if d.doc_type == DocumentType.GCN]
    contract_docs = [
        d for d in docs
        if DOCUMENT_CATEGORY_MAP[d.doc_type] in ("chuyen_nhuong", "the_chap")
    ]
    shared_docs = [d for d in docs if d.doc_type == DocumentType.CCCD]
    unclassified_docs = [d for d in docs if d.doc_type == DocumentType.KHONG_XAC_DINH]

    shared_filenames = [d.filename for d in shared_docs]

    # ── Trường hợp đơn giản: chỉ 1 GCN → chỉ 1 tài sản, khỏi cần LLM ─────
    if len(gcn_docs) <= 1:
        filenames = [d.filename for d in gcn_docs + contract_docs]
        fp = _extract_gcn_fingerprint(gcn_docs[0].raw_text) if gcn_docs else {}
        group = AssetGroupCandidate(
            asset_id="asset_1",
            so_gcn_goi_y=fp.get("so_gcn", ""),
            dia_chi_goi_y=_build_dia_chi_goi_y(fp),
            filenames=filenames,
            shared_filenames=shared_filenames,
            grouping_method="fallback_single",
            grouping_confidence=1.0,
            grouping_reason="Chỉ phát hiện 1 GCN trong hồ sơ → toàn bộ file thuộc về 1 tài sản.",
        )
        if unclassified_docs:
            notes.append(
                f"[B1b] {len(unclassified_docs)} file không phân loại được sẽ cần cán bộ "
                "gán thủ công (không tự động gộp vào tài sản)."
            )
        notes.append("[B1b] Chỉ 1 tài sản được phát hiện — bỏ qua bước gọi LLM gom nhóm.")
        print("[B1b] Chỉ 1 GCN → 1 tài sản duy nhất.")
        return state.model_copy(update={
            "asset_groups": [group],
            "flags": flags,
            "processing_notes": notes,
        })

    # ── Trường hợp nhiều GCN → gọi LLM để gom nhóm ───────────────────────
    print(f"[B1b] Phát hiện {len(gcn_docs)} GCN → gọi LLM để gom nhóm tài sản.")
    try:
        result = _call_llm_grouping(gcn_docs, contract_docs)
    except Exception as exc:
        msg = f"[B1b] Lỗi gọi LLM gom nhóm tài sản: {exc}. Fallback: mỗi GCN = 1 tài sản riêng, không gán hợp đồng kèm theo."
        print(msg)
        notes.append(msg)
        flags.append(FlagItem(
            flag_type="GOM_NHOM_TAI_SAN_DO_TIN_CAY_THAP",
            severity="WARNING",
            description=msg,
            affected_field="asset_groups",
        ))
        fallback_groups = []
        for i, d in enumerate(gcn_docs, start=1):
            fp = _extract_gcn_fingerprint(d.raw_text)
            fallback_groups.append(AssetGroupCandidate(
                asset_id=f"asset_{i}",
                so_gcn_goi_y=fp.get("so_gcn", ""),
                dia_chi_goi_y=_build_dia_chi_goi_y(fp),
                filenames=[d.filename],
                shared_filenames=shared_filenames,
                grouping_method="fallback_single",
                grouping_confidence=0.3,
                grouping_reason="LLM gom nhóm lỗi — fallback mỗi GCN 1 tài sản, cần người xác nhận kỹ.",
            ))
        return state.model_copy(update={
            "asset_groups": fallback_groups,
            "flags": flags,
            "processing_notes": notes,
        })

    assets_raw = result.get("assets", []) or []
    unassigned = set(result.get("khong_gan_duoc", []) or [])

    all_gcn_filenames = {d.filename for d in gcn_docs}
    assigned_filenames: set[str] = set()
    groups: list[AssetGroupCandidate] = []

    for i, item in enumerate(assets_raw, start=1):
        filenames = [f for f in (item.get("filenames") or []) if f]
        assigned_filenames.update(filenames)
        confidence = float(item.get("do_tin_cay", 0.5) or 0.5)
        group = AssetGroupCandidate(
            asset_id=f"asset_{i}",
            so_gcn_goi_y=str(item.get("so_gcn_goi_y", "") or ""),
            dia_chi_goi_y=str(item.get("dia_chi_goi_y", "") or ""),
            filenames=filenames,
            shared_filenames=shared_filenames,
            grouping_method="llm",
            grouping_confidence=confidence,
            grouping_reason=str(item.get("ly_do", "") or ""),
        )
        groups.append(group)
        if confidence < 0.6:
            flags.append(FlagItem(
                flag_type="GOM_NHOM_TAI_SAN_DO_TIN_CAY_THAP",
                severity="WARNING",
                description=(
                    f"Nhóm tài sản '{group.asset_id}' (GCN gợi ý: {group.so_gcn_goi_y or 'N/A'}) "
                    f"được AI gom với độ tin cậy thấp ({confidence:.2f}). Cần cán bộ xác nhận kỹ."
                ),
                affected_field="asset_groups",
            ))

    # ── File không được gán vào nhóm nào (kể cả GCN!) → tạo nhóm riêng để lộ ra cho người xác nhận ──
    all_contract_gcn_filenames = {d.filename for d in gcn_docs + contract_docs}
    truly_unassigned = (all_contract_gcn_filenames - assigned_filenames) | (
        unassigned & all_contract_gcn_filenames
    )

    if truly_unassigned:
        msg = (
            f"[B1b] {len(truly_unassigned)} file chưa được AI gán vào tài sản nào: "
            f"{sorted(truly_unassigned)}. Đã tạo nhóm riêng để cán bộ xác nhận thủ công."
        )
        print(msg)
        notes.append(msg)
        flags.append(FlagItem(
            flag_type="GOM_NHOM_TAI_SAN_CON_FILE_LE",
            severity="WARNING",
            description=msg,
            affected_field="asset_groups",
        ))
        groups.append(AssetGroupCandidate(
            asset_id=f"asset_{len(groups) + 1}_can_xac_minh",
            so_gcn_goi_y="",
            dia_chi_goi_y="",
            filenames=sorted(truly_unassigned),
            shared_filenames=shared_filenames,
            grouping_method="llm",
            grouping_confidence=0.0,
            grouping_reason="AI không đủ căn cứ gán các file này vào tài sản nào — cần cán bộ xử lý thủ công.",
        ))

    print(f"[B1b] Đề xuất {len(groups)} nhóm tài sản (bao gồm cả nhóm 'cần xác minh' nếu có).")
    print("[B1b] Hoàn thành — chờ xác nhận của cán bộ tín dụng (human-in-the-loop).\n")

    return state.model_copy(update={
        "asset_groups": groups,
        "flags": flags,
        "processing_notes": notes,
    })