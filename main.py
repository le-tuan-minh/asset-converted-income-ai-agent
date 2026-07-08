"""
Main entrypoint — chạy luồng thẩm định B1→B2→B3.

Cách chạy:
    python main.py
    python main.py --cccd path/to/cccd.jpg --gcn path/to/gcn.pdf --hop-dong path/to/hd.pdf
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env (GROQ_API_KEY)
load_dotenv()

from schemas import GraphState
from graph import build_graph


def print_report(final_state: GraphState) -> None:
    """In báo cáo kết quả ra console."""
    sep = "─" * 60

    print("\n" + "=" * 60)
    print("📋  KẾT QUẢ THẨM ĐỊNH TÀI SẢN BẢO ĐẢM")
    print("=" * 60)

    # Thông tin chủ tài sản
    oi = final_state.owner_info
    print(f"\n👤 CHỦ TÀI SẢN (từ CCCD)")
    print(f"   Họ tên         : {oi.ho_ten or 'N/A'}")
    print(f"   Số CCCD        : {oi.so_cccd or 'N/A'}")
    print(f"   Số CMTND cũ    : {oi.so_cmtnd_cu or 'N/A'}")
    print(f"   Ngày sinh       : {oi.ngay_sinh or 'N/A'}")
    print(f"   Địa chỉ TT     : {oi.dia_chi_thuong_tru or 'N/A'}")

    # Thông tin tài sản
    ai = final_state.asset_info
    print(f"\n🏠 TÀI SẢN BẢO ĐẢM (từ GCN + HĐ)")
    print(f"   Số GCN         : {ai.so_gcn or 'N/A'}")
    print(f"   Chủ sử dụng    : {ai.chu_su_dung or 'N/A'}")
    print(f"   Ngày cấp GCN   : {ai.ngay_cap_gcn or 'N/A'}")
    print(f"   Ngày CN        : {ai.ngay_chuyen_nhuong or 'N/A'}")
    print(f"   Mục đích SD    : {ai.muc_dich_su_dung or 'N/A'}")
    print(f"   DT tổng        : {ai.dien_tich_tong or 'N/A'} m²")
    print(f"   DT đất ở       : {ai.dien_tich_dat_o or 'N/A'} m²")
    print(f"   DT nhà ở       : {ai.dien_tich_nha_o or 'N/A'} m²")
    print(f"   DT NN          : {ai.dien_tich_nn or 'N/A'} m²")
    print(f"   DT TMDV        : {ai.dien_tich_tmdv or 'N/A'} m²")
    print(f"   Nguồn gốc      : {ai.nguon_goc_tai_san or 'N/A'}")

    # Kết quả B2
    ic = final_state.identity_check
    lp = final_state.land_purpose
    print(f"\n🔍 KẾT QUẢ KIỂM TRA")
    print(f"   Chủ TS khớp CCCD  : {'✅ Có' if ic.owner_matched else '❌ Không'}")
    print(f"   Tặng cho/TK       : {'⚠️ Có' if (ic.is_tang_cho or ic.is_thua_ke) else '✅ Không'}")
    print(f"   Ngày hình thành   : {ic.asset_formation_date or 'N/A'}")
    print(f"   DT đủ đk quy đổi : {lp.dien_tich_du_dieu_kien or 'N/A'} m²")

    # Flags
    print(f"\n🚩 FLAGS & CẢNH BÁO ({len(final_state.flags)} flag(s))")
    if not final_state.flags:
        print("   Không có flag.")
    for f in final_state.flags:
        icon = "⛔" if f.severity == "ERROR" else "⚠️"
        print(f"   {icon} [{f.flag_type}] {f.description}")

    # Warnings
    if final_state.warnings:
        print(f"\n⚠️  WARNINGS")
        for w in final_state.warnings:
            print(f"   {w}")

    # Routing
    print(f"\n🔀 ROUTING")
    status = "🔴 Human Review" if final_state.has_critical_flags else "🟢 Tiếp tục quy trình (B4)"
    print(f"   Kết quả: {status}")

    if final_state.error:
        print(f"\n❌ LỖI XỬ LÝ: {final_state.error}")

    print("\n" + "=" * 60)


def save_output(final_state: GraphState, output_path: str = "output/result.json") -> None:
    """Lưu kết quả ra file JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data = final_state.model_dump(exclude={"ocr_raw"})  # Bỏ raw OCR để JSON gọn hơn
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Kết quả đã lưu: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Thẩm định tín dụng AI Agent B1-B3")
    parser.add_argument("--cccd",     default="input_data/test_input_1/cccd_kh.jpg")
    parser.add_argument("--gcn",      default="input_data/test_input_1/giay_chung_nhan.pdf")
    parser.add_argument("--hop-dong", default="input_data/test_input_1/hop_dong_mua_ban.pdf")
    parser.add_argument("--output",   default="output/result.json")
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY chưa được set. Tạo file .env với GROQ_API_KEY=<key>")
        sys.exit(1)

    # Khởi tạo state ban đầu
    initial_state = GraphState(
        cccd_path=args.cccd,
        gcn_path=args.gcn,
        hop_dong_path=args.hop_dong,
    )

    print(f"\n🚀 Bắt đầu luồng thẩm định")
    print(f"   CCCD     : {args.cccd}")
    print(f"   GCN      : {args.gcn}")
    print(f"   Hợp đồng : {args.hop_dong}")

    # Build và chạy graph
    graph = build_graph()
    final_state = graph.invoke(initial_state)

    # In báo cáo
    print_report(final_state)

    # Lưu output
    save_output(final_state, args.output)


if __name__ == "__main__":
    main()