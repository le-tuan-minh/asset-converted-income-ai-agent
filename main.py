"""
Main entrypoint — chạy luồng thẩm định B1 → B1b (AI gom nhóm tài sản) →
B1c (human-in-the-loop xác nhận) → B2 → B2c → B3 cho TỪNG tài sản.

Cách chạy:
    python main.py
    python main.py --folder input_data/test_input_multi
    python main.py --folder input_data/test_input_multi --auto-confirm   (bỏ qua hỏi, tự xác nhận AI)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command

from schemas import GraphState
from graph import build_graph

# GraphState chứa các Pydantic model + Enum tự định nghĩa trong schemas.py
# (DocumentType, DocumentItem, AssetGroupCandidate, AssetResult, ...). Vì
# node B1c dùng interrupt(), toàn bộ state phải được checkpoint (msgpack) mỗi
# lần dừng/resume. Không khai báo allowed_msgpack_modules sẽ chỉ ra warning ở
# bản LangGraph hiện tại, nhưng THEO TÀI LIỆU sẽ bị CHẶN CỨNG (raise lỗi) ở
# bản tương lai — nên khai báo tường minh ngay từ bây giờ.
_ALLOWED_MSGPACK_MODULES = [
    ("schemas", "DocumentType"),
    ("schemas", "DocumentItem"),
    ("schemas", "AssetGroupCandidate"),
    ("schemas", "AssetResult"),
    ("schemas", "OwnerInfo"),
    ("schemas", "AssetInfo"),
    ("schemas", "BienDongItem"),
    ("schemas", "IdentityCheckResult"),
    ("schemas", "LandPurposeResult"),
    ("schemas", "FlagItem"),
]


# ─────────────────────────────────────────────
# Human-in-the-loop: hỏi cán bộ tín dụng xác nhận/chỉnh sửa nhóm tài sản
# ─────────────────────────────────────────────

def _print_grouping_proposal(payload: dict) -> None:
    print("\n" + "=" * 60)
    print("🧑‍💼  CẦN XÁC NHẬN: AI ĐỀ XUẤT GOM NHÓM TÀI SẢN")
    print("=" * 60)
    print(payload.get("message", ""))
    for g in payload.get("asset_groups", []):
        print(f"\n  📌 {g['asset_id']}  (độ tin cậy: {g['grouping_confidence']:.2f}, "
              f"phương pháp: {g['grouping_method']})")
        print(f"     Số GCN gợi ý : {g['so_gcn_goi_y'] or 'N/A'}")
        print(f"     Địa chỉ gợi ý: {g['dia_chi_goi_y'] or 'N/A'}")
        print(f"     File riêng   : {g['filenames']}")
        print(f"     File dùng chung (CCCD): {g['shared_filenames']}")
        print(f"     Lý do AI gom : {g['grouping_reason'] or 'N/A'}")


def _ask_human_confirmation(payload: dict, auto_confirm: bool = False) -> dict:
    """
    Console-based human-in-the-loop. Trả về dict theo đúng format node
    node_human_confirm_grouping mong đợi: {"action": "confirm"} hoặc
    {"action": "edit", "asset_groups": [...], "note": "..."}.
    """
    _print_grouping_proposal(payload)

    if auto_confirm:
        print("\n[--auto-confirm] Tự động xác nhận theo đề xuất của AI (không hỏi).")
        return {"action": "confirm"}

    print("\nCán bộ tín dụng vui lòng kiểm tra lại việc gom nhóm ở trên.")
    choice = input("Xác nhận đề xuất của AI? (Enter/'ok' = xác nhận, 'edit' = chỉnh sửa thủ công): ").strip().lower()

    if choice not in ("edit",):
        return {"action": "confirm"}

    print("\n--- CHỈNH SỬA NHÓM TÀI SẢN THỦ CÔNG ---")
    print("Nhập lại danh sách nhóm. Với mỗi nhóm, liệt kê các file (phân cách bằng dấu phẩy).")
    print("Để trống tên nhóm khi được hỏi để dừng nhập.")

    edited_groups = []
    shared_filenames = payload["asset_groups"][0]["shared_filenames"] if payload["asset_groups"] else []
    idx = 1
    while True:
        asset_id = input(f"Tên tài sản #{idx} (Enter để dừng): ").strip()
        if not asset_id:
            break
        filenames_raw = input(f"  Danh sách file thuộc '{asset_id}' (vd: gcn_1.pdf,hop_dong_1.pdf): ").strip()
        filenames = [f.strip() for f in filenames_raw.split(",") if f.strip()]
        so_gcn = input("  Số GCN (tuỳ chọn): ").strip()
        edited_groups.append({
            "asset_id": asset_id,
            "so_gcn_goi_y": so_gcn,
            "dia_chi_goi_y": "",
            "filenames": filenames,
            "shared_filenames": shared_filenames,
            "grouping_method": "human_edited",
            "grouping_confidence": 1.0,
            "grouping_reason": "Cán bộ tín dụng nhập thủ công.",
        })
        idx += 1

    note = input("Ghi chú lý do chỉnh sửa (tuỳ chọn): ").strip()
    return {"action": "edit", "asset_groups": edited_groups, "note": note}


# ─────────────────────────────────────────────
# Báo cáo kết quả
# ─────────────────────────────────────────────

def print_report(final_state: GraphState) -> None:
    print("\n" + "=" * 60)
    print("📋  KẾT QUẢ THẨM ĐỊNH TÀI SẢN BẢO ĐẢM (NHIỀU TÀI SẢN)")
    print("=" * 60)

    print(f"\n📂 GIẤY TỜ ĐÃ NHẬN DIỆN ({len(final_state.documents)} file)")
    for d in final_state.documents:
        print(
            f"   • {d.filename:30s} → {d.doc_type.value:28s} "
            f"[{d.extraction_source or 'N/A':11s} | {d.classify_method}/{d.classify_confidence:.2f} | {d.char_count} ký tự]"
        )

    print(f"\n🗂️  NHÓM TÀI SẢN ĐÃ XÁC NHẬN ({len(final_state.asset_groups)} tài sản)")
    for g in final_state.asset_groups:
        print(f"   • {g.asset_id}: {g.filenames} (+ dùng chung: {g.shared_filenames})")

    print(f"\n🏠 KẾT QUẢ TỪNG TÀI SẢN ({len(final_state.asset_results)} tài sản)")
    for r in final_state.asset_results:
        oi, ai, ic, lp = r.owner_info, r.asset_info, r.identity_check, r.land_purpose
        status = "🔴 Cần rà soát" if r.has_critical_flags else "🟢 Ổn"
        print(f"\n   ─── {r.asset_id} [{status}] ───")
        print(f"   File                    : {r.document_filenames}")

        print(f"\n   👤 CHỦ TÀI SẢN (từ CCCD)")
        print(f"      Họ tên               : {oi.ho_ten or 'N/A'}")
        print(f"      Số CCCD              : {oi.so_cccd or 'N/A'}")
        print(f"      Số CMTND cũ          : {oi.so_cmtnd_cu or 'N/A'}")
        print(f"      Ngày sinh            : {oi.ngay_sinh or 'N/A'}")
        print(f"      Địa chỉ thường trú   : {oi.dia_chi_thuong_tru or 'N/A'}")

        print(f"\n   🏠 THÔNG TIN TÀI SẢN (GCN + Hợp đồng)")
        print(f"      Số GCN               : {ai.so_gcn or 'N/A'}")
        print(f"      Địa chỉ tài sản      : {ai.dia_chi_tai_san or 'N/A'}")
        print(f"      Chủ SD gốc           : {ai.chu_su_dung_goc or 'N/A'}")
        print(f"      Chủ SD hiện tại      : {ai.chu_su_dung_hien_tai or 'N/A'}")
        print(f"      Ngày cấp GCN         : {ai.ngay_cap_gcn or 'N/A'}")
        print(f"      Ngày chuyển nhượng   : {ai.ngay_chuyen_nhuong or 'N/A'}")
        print(f"      Nguồn gốc tài sản    : {ai.nguon_goc_tai_san or 'N/A'}")
        print(f"      Có tặng cho          : {'Có' if ai.co_thong_tin_tang_cho else 'Không'}")
        if ai.bien_dong_lich_su:
            print(f"      Lịch sử biến động    :")
            for bd in ai.bien_dong_lich_su:
                print(f"        - {bd.ngay or 'N/A'}: {bd.noi_dung or 'N/A'} (chủ mới: {bd.chu_moi or 'N/A'})")
        print(f"      Bên mua (hợp đồng)   : {ai.ben_mua_hop_dong or 'N/A'} (CCCD: {ai.ben_mua_so_cccd_hop_dong or 'N/A'})")
        print(f"      Bên bán (hợp đồng)   : {ai.ben_ban_hop_dong or 'N/A'}")

        print(f"\n   📐 DIỆN TÍCH")
        print(f"      Tổng                 : {ai.dien_tich_tong or 'N/A'} m²")
        print(f"      Đất ở                : {ai.dien_tich_dat_o or 'N/A'} m²")
        print(f"      Nhà ở                : {ai.dien_tich_nha_o or 'N/A'} m²")
        print(f"      Nông nghiệp (NN)     : {ai.dien_tich_nn or 'N/A'} m²")
        print(f"      Nuôi trồng TS (NTS)  : {ai.dien_tich_nts or 'N/A'} m²")
        print(f"      Thương mại DV (TMDV) : {ai.dien_tich_tmdv or 'N/A'} m²")
        # ĐÃ SỬA (fix #2): tách 2 dòng riêng, không còn cộng gộp đất ở + nhà ở
        print(f"      Đất ở (đủ ĐK quy đổi): {lp.dien_tich_dat_o_du_dieu_kien or 'N/A'} m² (tính tất định)")
        print(f"      Nhà ở (đủ ĐK quy đổi): {lp.dien_tich_nha_o_du_dieu_kien or 'N/A'} m² (tính tất định)")

        print(f"\n   🧾 MỤC ĐÍCH SỬ DỤNG ĐẤT")
        print(f"      Mục đích             : {lp.muc_dich or ai.muc_dich_su_dung or 'N/A'}")
        print(f"      Mã ký hiệu đất       : {lp.ma_ky_hieu_dat or ai.ma_ky_hieu_dat or 'N/A'}")
        print(f"      Là đất TMDV          : {'Có' if lp.is_tmdv else 'Không'}")
        if lp.is_tmdv:
            print(f"      Thuộc dự án          : {lp.thuoc_du_an if lp.thuoc_du_an is not None else 'Chưa xác định'}")
            print(f"      Tên dự án            : {lp.ten_du_an or 'N/A'}")
            print(f"      Căn cứ pháp lý DA    : {lp.can_cu_phap_ly_du_an or 'N/A'}")
            print(f"      Nguồn xác định DA    : {lp.nguon_xac_dinh_du_an}")
            if lp.warning_tmdv:
                print(f"      Cảnh báo TMDV        : {lp.warning_tmdv}")
            if lp.web_verification_summary:
                print(f"      Tóm tắt tra cứu web  : {lp.web_verification_summary}")
                print(f"      Nguồn web            : {lp.web_verification_sources}")

        print(f"\n   🔎 KIỂM TRA CHỦ TÀI SẢN")
        # ĐÃ SỬA (fix #7): hiển thị similarity score để phân biệt "khớp tuyệt
        # đối" và "khớp gần đúng" (vd do lỗi OCR sai dấu), thay vì chỉ True/False.
        sim = ic.owner_name_similarity
        if sim is None:
            sim_note = ""
        elif sim >= 0.999:
            sim_note = " (khớp tuyệt đối)"
        else:
            sim_note = f" (similarity: {sim:.0%} — khớp gần đúng, cần đối chiếu bản gốc)"
        print(f"      Khớp CCCD            : {'✅ Có' if ic.owner_matched else '❌ Không'} (so với: {ic.matched_against}){sim_note}")
        if ic.mismatch_fields:
            print(f"      Trường lệch          : {ic.mismatch_fields}")
        print(f"      Tặng cho / thừa kế   : tặng cho={ic.is_tang_cho}, thừa kế={ic.is_thua_ke}")
        print(f"      Ngày hình thành (LLM): {ic.asset_formation_date or 'N/A'} — {ic.asset_formation_note or ''}")

        print(f"\n   🚩 FLAGS ({len(r.flags)})")
        for f in r.flags:
            icon = "⛔" if f.severity == "ERROR" else "⚠️"
            print(f"      {icon} [{f.flag_type}] {f.description}")

        if r.warnings:
            print(f"\n   ⚠️  CẢNH BÁO CHO CÁN BỘ TÍN DỤNG")
            for w in r.warnings:
                print(f"      {w}")

        if r.processing_notes:
            print(f"\n   📝 GHI CHÚ XỬ LÝ (debug)")
            for n in r.processing_notes:
                print(f"      {n}")

        if r.error:
            print(f"\n   ❌ Lỗi xử lý: {r.error}")

    print(f"\n🔀 ROUTING TỔNG THỂ")
    status = "🔴 Human Review" if final_state.has_critical_flags else "🟢 Tiếp tục quy trình (B4)"
    print(f"   Kết quả: {status}")

    if final_state.error:
        print(f"\n❌ LỖI XỬ LÝ: {final_state.error}")

    print("\n" + "=" * 60)


def save_output(final_state: GraphState, output_path: str = "output/result.json") -> None:
    """Lưu kết quả ra file JSON (bỏ raw_text trong documents để file gọn hơn)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data = final_state.model_dump(exclude={"documents": {"__all__": {"raw_text"}}})
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Kết quả đã lưu: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Thẩm định tín dụng AI Agent — hỗ trợ nhiều tài sản")
    parser.add_argument(
        "--folder",
        default="input_data/test_input_1",
        help="Folder chứa các file giấy tờ đầu vào (số lượng file bất kỳ: pdf/jpg/png/...); "
             "có thể chứa NHIỀU tài sản (nhiều GCN).",
    )
    parser.add_argument("--output", default="output/result.json")
    parser.add_argument(
        "--auto-confirm", action="store_true",
        help="Bỏ qua bước hỏi console, tự động xác nhận đề xuất gom nhóm của AI "
             "(dùng cho test tự động — KHÔNG khuyến khích dùng trong môi trường thật).",
    )
    parser.add_argument(
        "--thread-id", default="cli-session",
        help="Thread ID cho checkpointer LangGraph (mỗi hồ sơ nên có 1 thread_id riêng).",
    )
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY chưa được set.")
        sys.exit(1)

    checkpointer = MemorySaver(
        serde=JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_MSGPACK_MODULES)
    )
    graph = build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": args.thread_id}}

    initial_state = GraphState(input_folder=args.folder)

    result = graph.invoke(initial_state, config=config)

    # ── Xử lý interrupt (human-in-the-loop xác nhận gom nhóm tài sản) ────
    while "__interrupt__" in result:
        interrupt_obj = result["__interrupt__"][0]
        payload = interrupt_obj.value
        human_response = _ask_human_confirmation(payload, auto_confirm=args.auto_confirm)
        result = graph.invoke(Command(resume=human_response), config=config)

    final_state = GraphState(**result)

    print_report(final_state)
    save_output(final_state, args.output)


if __name__ == "__main__":
    main()