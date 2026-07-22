"""
B2/B2c/B3 (cấp hồ sơ) — lặp qua TỪNG tài sản đã được con người xác nhận ở B1c
(state.asset_groups) và chạy pipeline B2→B2c→B3 riêng cho từng tài sản
(nodes/asset_pipeline.py::process_single_asset), tổng hợp kết quả vào
state.asset_results.

has_critical_flags cấp HỒ SƠ = True nếu B1 đã lỗi, HOẶC có bất kỳ tài sản nào
có has_critical_flags=True → toàn bộ hồ sơ sẽ được route sang human_review để
cán bộ tín dụng rà soát (dù có thể vẫn còn tài sản khác trong hồ sơ không có
vấn đề gì).
"""
from __future__ import annotations

from schemas import GraphState
from nodes.asset_pipeline import process_single_asset


def node_b2_process_assets(state: GraphState) -> GraphState:
    print("\n" + "=" * 60)
    print(f"B2→B2c→B3 · XỬ LÝ {len(state.asset_groups)} TÀI SẢN ĐÃ XÁC NHẬN")
    print("=" * 60)

    notes = list(state.processing_notes)
    asset_results = []

    for group in state.asset_groups:
        result = process_single_asset(group, state.documents)
        asset_results.append(result)
        notes.append(
            f"[B2-B3] Tài sản '{group.asset_id}' xử lý xong — "
            f"{len(result.flags)} flag(s), has_critical_flags={result.has_critical_flags}."
        )

    has_critical = state.has_critical_flags or any(r.has_critical_flags for r in asset_results)

    print(f"\n[B2-B3] Hoàn thành xử lý {len(asset_results)} tài sản.")
    n_critical = sum(1 for r in asset_results if r.has_critical_flags)
    if n_critical:
        print(f"[B2-B3] ⛔ {n_critical}/{len(asset_results)} tài sản có flag ERROR — cần Human Review.")
    else:
        print("[B2-B3] ✅ Không có tài sản nào có flag ERROR.")

    return state.model_copy(update={
        "asset_results": asset_results,
        "has_critical_flags": has_critical,
        "processing_notes": notes,
    })