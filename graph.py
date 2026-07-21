"""
LangGraph Graph definition cho luồng thẩm định tín dụng B1 → B2 → B2c → B3.

Luồng:
  START → b1_input → (router B1) → human_review (nếu thiếu giấy tờ bắt buộc)
                                  → b2_verify → b2c_tmdv_websearch → b3_flag
                                                → (router B3) → human_review | END

  Router sau B1: nếu hồ sơ thiếu giấy tờ nhân thân (CCCD/CMTND) hoặc thiếu Giấy
  chứng nhận QSDĐ (GCN) — 2 loại giấy tờ BẮT BUỘC theo nghiệp vụ — thì dừng ngay,
  chuyển sang human_review, KHÔNG chạy B2/B2c/B3. Lý do: nếu thiếu 1 trong 2 nhóm
  này, owner_info hoặc asset_info sẽ rỗng/không đầy đủ, đưa vào B2 sẽ khiến LLM
  phải "so khớp" trên dữ liệu vốn không đủ căn cứ, dễ sinh kết luận owner_matched
  sai/giả (false positive CHU_TAI_SAN_LECH), gây nhiễu báo cáo và không giúp ích
  cho cán bộ tín dụng — bản chất vấn đề là "thiếu hồ sơ" chứ không phải "phát hiện
  sai khác thực sự".

  b2c_tmdv_websearch là bước bổ sung: chỉ thực sự tra cứu web khi B2 (kể cả sau
  rule-based cross-check) vẫn chưa xác định được đất TMDV có thuộc dự án hay không.
  Các trường hợp khác, node này pass-through gần như tức thời (không tốn API call).

Lưu ý LangGraph:
  - .invoke() với Pydantic BaseModel state trả về dict (không phải model instance)
  - Mỗi node nhận GraphState, trả về GraphState
  - main.py sẽ convert dict → GraphState sau khi invoke
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from schemas import GraphState
from nodes.node_b1_input import node_b1_input
from nodes.node_b2_verify import node_b2_verify
from nodes.node_b2c_tmdv_websearch import node_b2c_tmdv_websearch
from nodes.node_b3_flag import node_b3_flag


def route_after_b1(state: GraphState) -> str:
    """
    Conditional edge sau B1:
    - Thiếu giấy tờ bắt buộc (nhan_than/GCN) hoặc lỗi input (has_critical_flags=True
      hoặc error) → dừng lại, chuyển sang human_review ngay, KHÔNG chạy B2/B2c/B3.
    - Đủ điều kiện → tiếp tục B2.
    """
    # Handle cả dict lẫn GraphState (LangGraph có thể pass dict giữa nodes)
    if isinstance(state, dict):
        has_critical = state.get("has_critical_flags", False)
        has_error = state.get("error") is not None
    else:
        has_critical = state.has_critical_flags
        has_error = state.error is not None

    if has_critical or has_error:
        return "human_review"
    return "continue"


def route_after_b3(state: GraphState) -> str:
    """
    Conditional edge sau B3:
    - Có flag ERROR → "human_review"
    - Không có → "end"
    """
    # Handle cả dict lẫn GraphState (LangGraph có thể pass dict giữa nodes)
    if isinstance(state, dict):
        has_critical = state.get("has_critical_flags", False)
        has_error = state.get("error") is not None
    else:
        has_critical = state.has_critical_flags
        has_error = state.error is not None

    if has_critical or has_error:
        return "human_review"
    return "end"


def node_human_review(state: GraphState) -> GraphState:
    """
    Placeholder node: tài sản có flag nghiêm trọng (bao gồm cả trường hợp thiếu
    giấy tờ bắt buộc ngay từ B1) → chờ cán bộ xem xét.
    Trong production có thể gửi notification, tạo task trên hệ thống.
    """
    print("\n" + "="*60)
    print("🔴 HUMAN REVIEW — Hồ sơ có flag nghiêm trọng, cần xét duyệt thủ công")
    print("="*60)

    # Handle cả dict lẫn GraphState
    if isinstance(state, dict):
        state = GraphState(**state)

    notes = list(state.processing_notes)
    notes.append("Hồ sơ được chuyển sang Human Review do có flag ERROR.")
    return state.model_copy(update={"processing_notes": notes})


def build_graph() -> StateGraph:
    """Build và compile LangGraph StateGraph."""
    builder = StateGraph(GraphState)

    # Thêm nodes
    builder.add_node("b1_input",            node_b1_input)
    builder.add_node("b2_verify",           node_b2_verify)
    builder.add_node("b2c_tmdv_websearch",  node_b2c_tmdv_websearch)
    builder.add_node("b3_flag",             node_b3_flag)
    builder.add_node("human_review",        node_human_review)

    builder.set_entry_point("b1_input")

    # Conditional edge NGAY SAU B1: dừng sớm nếu thiếu giấy tờ bắt buộc
    builder.add_conditional_edges(
        "b1_input",
        route_after_b1,
        {
            "human_review": "human_review",
            "continue": "b2_verify",
        },
    )

    # Edges tuyến tính B2 → B2c → B3
    builder.add_edge("b2_verify",          "b2c_tmdv_websearch")
    builder.add_edge("b2c_tmdv_websearch", "b3_flag")

    # Conditional edge sau B3
    builder.add_conditional_edges(
        "b3_flag",
        route_after_b3,
        {
            "human_review": "human_review",
            "end": END,
        },
    )
    builder.add_edge("human_review", END)

    return builder.compile()