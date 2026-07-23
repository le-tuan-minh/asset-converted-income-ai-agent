"""
B1c - Human-in-the-loop: xác nhận / chỉnh sửa nhóm tài sản do AI đề xuất ở B1b.

Dùng cơ chế `interrupt()` của LangGraph: khi graph chạy tới node này, thực thi
sẽ DỪNG LẠI và trả điều khiển về cho caller (main.py) kèm payload mô tả các
nhóm tài sản đang đề xuất. Cán bộ tín dụng xem qua console (hoặc UI khác nếu
tích hợp sau này), sau đó:
  - Gõ "ok" / Enter  → xác nhận gom nhóm như AI đề xuất.
  - Gõ "edit"        → nhập lại thủ công danh sách file cho từng tài sản.
Main.py sẽ resume graph bằng `Command(resume=...)` với lựa chọn của cán bộ.

Vì đây là bước bắt buộc do nghiệp vụ yêu cầu "AI đề xuất, người xác nhận" nên
node này KHÔNG có đường tắt bỏ qua — mọi hồ sơ nhiều tài sản đều phải qua đây
trước khi chạy B2/B2c/B3.
"""
from __future__ import annotations

from langgraph.types import interrupt

from schemas import GraphState, AssetGroupCandidate, FlagItem


def node_b1c_confirm_grouping(state: GraphState) -> GraphState:
    print("\n" + "=" * 60)
    print("B1c · HUMAN-IN-THE-LOOP — Xác nhận nhóm tài sản")
    print("=" * 60)

    payload = {
        "message": (
            "AI đã đề xuất gom nhóm tài sản dưới đây. Vui lòng xác nhận hoặc "
            "chỉnh sửa trước khi hệ thống tiếp tục xử lý B2-B3 cho từng tài sản."
        ),
        "asset_groups": [g.model_dump() for g in state.asset_groups],
    }

    # Thực thi DỪNG tại đây cho tới khi caller resume bằng Command(resume=...)
    human_response: dict = interrupt(payload)

    flags = list(state.flags)
    notes = list(state.processing_notes)

    action = (human_response or {}).get("action", "confirm")

    if action == "edit":
        edited_raw = human_response.get("asset_groups", [])
        new_groups = [AssetGroupCandidate(**g) for g in edited_raw]
        edit_note = human_response.get("note", "")
        notes.append(
            f"[B1c] Cán bộ tín dụng đã CHỈNH SỬA nhóm tài sản do AI đề xuất. "
            f"Ghi chú: {edit_note or '(không có)'}"
        )
        flags.append(FlagItem(
            flag_type="GOM_NHOM_TAI_SAN_DA_CHINH_SUA",
            severity="WARNING",
            description=(
                f"Cán bộ tín dụng đã chỉnh sửa gom nhóm tài sản so với đề xuất của AI. "
                f"Ghi chú: {edit_note or '(không có)'}"
            ),
            affected_field="asset_groups",
        ))
        for g in new_groups:
            g_obj = g.model_copy(update={"grouping_method": "human_edited"})
        new_groups = [g.model_copy(update={"grouping_method": "human_edited"}) for g in new_groups]
        print(f"[B1c] Đã nhận {len(new_groups)} nhóm tài sản SAU khi chỉnh sửa.")
        return state.model_copy(update={
            "asset_groups": new_groups,
            "grouping_confirmed": True,
            "grouping_human_notes": edit_note,
            "flags": flags,
            "processing_notes": notes,
        })

    # action == "confirm" (mặc định) → giữ nguyên đề xuất của AI
    notes.append("[B1c] Cán bộ tín dụng đã XÁC NHẬN nhóm tài sản do AI đề xuất, không chỉnh sửa.")
    print("[B1c] Đã xác nhận gom nhóm như AI đề xuất, không chỉnh sửa.")
    return state.model_copy(update={
        "grouping_confirmed": True,
        "flags": flags,
        "processing_notes": notes,
    })