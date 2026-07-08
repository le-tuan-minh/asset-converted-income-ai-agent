"""
LangGraph Graph definition cho luồng thẩm định tín dụng B1 → B2 → B3.

Luồng:
  START → b1_input → b2_verify → b3_flag → (router) → human_review | END
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from schemas import GraphState
from nodes.node_b1_input import node_b1_input
from nodes.node_b2_verify import node_b2_verify
from nodes.node_b3_flag import node_b3_flag


def route_after_b3(state: GraphState) -> str:
    """
    Conditional edge sau B3:
    - Có flag ERROR → "human_review"
    - Không có → "end"
    """
    if state.has_critical_flags or state.error:
        return "human_review"
    return "end"


def node_human_review(state: GraphState) -> GraphState:
    """
    Placeholder node: tài sản có flag nghiêm trọng → chờ cán bộ xem xét.
    Trong production có thể gửi notification, tạo task trên hệ thống.
    """
    print("\n" + "="*60)
    print("🔴 HUMAN REVIEW — Hồ sơ có flag nghiêm trọng, cần xét duyệt thủ công")
    print("="*60)
    notes = list(state.processing_notes)
    notes.append("Hồ sơ được chuyển sang Human Review do có flag ERROR.")
    return state.model_copy(update={"processing_notes": notes})


def build_graph() -> StateGraph:
    """Build và compile LangGraph StateGraph."""
    builder = StateGraph(GraphState)

    # Thêm nodes
    builder.add_node("b1_input",      node_b1_input)
    builder.add_node("b2_verify",     node_b2_verify)
    builder.add_node("b3_flag",       node_b3_flag)
    builder.add_node("human_review",  node_human_review)

    # Edges tuyến tính
    builder.set_entry_point("b1_input")
    builder.add_edge("b1_input",  "b2_verify")
    builder.add_edge("b2_verify", "b3_flag")

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