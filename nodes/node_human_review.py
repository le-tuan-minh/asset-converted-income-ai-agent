"""
Node terminal: đánh dấu hồ sơ (hoặc 1/nhiều tài sản trong hồ sơ) cần cán bộ
tín dụng rà soát thủ công trước khi đi tiếp bước B4 (kiểm tra CIC TSBĐ).
Không thực hiện thêm xử lý AI nào — chỉ là điểm dừng rõ ràng trong luồng.
"""
from __future__ import annotations

from schemas import GraphState


def node_human_review(state: GraphState) -> GraphState:
    print("\n" + "=" * 60)
    print("🔴 HUMAN REVIEW — Hồ sơ cần cán bộ tín dụng rà soát thủ công")
    print("=" * 60)

    if state.error:
        print(f"Lý do: lỗi xử lý — {state.error}")
    elif not state.documents:
        print("Lý do: không có tài liệu nào được xử lý ở B1.")
    else:
        n_critical_assets = sum(1 for r in state.asset_results if r.has_critical_flags)
        print(f"Lý do: {n_critical_assets}/{len(state.asset_results)} tài sản có flag mức ERROR.")

    return state