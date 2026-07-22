"""
LangGraph Graph definition — luồng thẩm định tín dụng hỗ trợ NHIỀU TÀI SẢN
trong cùng 1 hồ sơ (folder input), với bước AI gom nhóm tài sản + xác nhận
của con người (human-in-the-loop) trước khi chạy B2/B2c/B3 cho từng tài sản.

Luồng:

  START
    │
    ▼
  b1_input                     (OCR hybrid + phân loại giấy tờ — cấp hồ sơ)
    │
    ├─ route_after_b1: thiếu CCCD hoặc thiếu GCN ở cấp hồ sơ → human_review → END
    │
    ▼ (đủ điều kiện)
  b1b_group_assets              (Reasoning AI đề xuất gom nhóm tài sản;
                                  rule-based hỗ trợ trích số GCN/thửa đất/tờ BĐ;
                                  nếu chỉ có 1 GCN thì bỏ qua bước gọi LLM)
    │
    ▼
  b1c_confirm_grouping           (HUMAN-IN-THE-LOOP — interrupt(): dừng graph,
                                  trả đề xuất gom nhóm ra ngoài cho cán bộ tín
                                  dụng xác nhận hoặc chỉnh sửa, rồi resume)
    │
    ▼
  b2_process_assets              (Lặp qua TỪNG tài sản đã xác nhận, chạy B2
                                  Groq LLM extract & verify → B2c web search
                                  TMDV nếu cần → B3 rule-based flag engine.
                                  Documents của tài sản này được lọc riêng,
                                  KHÔNG lẫn dữ liệu giữa các tài sản.)
    │
    ├─ route_after_processing: có ≥1 tài sản có flag ERROR → human_review → END
    │
    ▼
  END                             (Sẵn sàng cho B4 - kiểm tra CIC TSBĐ)

Ghi chú kỹ thuật:
  - Graph PHẢI được compile với 1 checkpointer (vd MemorySaver) để interrupt()/
    Command(resume=...) hoạt động — xem build_graph(checkpointer=...).
  - .invoke() với Pydantic BaseModel state trả về dict (không phải model
    instance); main.py sẽ convert dict → GraphState sau khi invoke.
  - Khi graph dừng tại interrupt, kết quả invoke() trả về sẽ chứa key
    "__interrupt__" thay vì state đầy đủ — main.py xử lý việc này.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from schemas import GraphState
from nodes.node_b1_input import node_b1_input
from nodes.node_b1b_group_assets import node_b1b_group_assets
from nodes.node_human_confirm_grouping import node_human_confirm_grouping
from nodes.node_b2_process_assets import node_b2_process_assets
from nodes.node_human_review import node_human_review


def route_after_b1(state: GraphState) -> str:
    """
    Thiếu giấy tờ bắt buộc (CCCD/CMTND hoặc GCN) ở cấp hồ sơ, hoặc lỗi input
    → dừng ngay, sang human_review, KHÔNG chạy B1b/B1c/B2/B3.
    """
    if state.has_critical_flags or state.error:
        return "human_review"
    return "continue"


def route_after_processing(state: GraphState) -> str:
    """Có ít nhất 1 tài sản (hoặc chính B1) có flag ERROR → human_review."""
    if state.has_critical_flags:
        return "human_review"
    return "end"


def build_graph(checkpointer=None):
    """
    Build và compile LangGraph.

    checkpointer: BẮT BUỘC truyền vào (vd langgraph.checkpoint.memory.MemorySaver())
    nếu muốn dùng human-in-the-loop (node b1c_confirm_grouping dùng interrupt()).
    Không truyền checkpointer, graph vẫn build được nhưng KHÔNG thể dừng/resume
    ở bước xác nhận gom nhóm — chỉ phù hợp cho test nhanh logic B1/B1b thuần.
    """
    graph = StateGraph(GraphState)

    graph.add_node("b1_input", node_b1_input)
    graph.add_node("b1b_group_assets", node_b1b_group_assets)
    graph.add_node("b1c_confirm_grouping", node_human_confirm_grouping)
    graph.add_node("b2_process_assets", node_b2_process_assets)
    graph.add_node("human_review", node_human_review)

    graph.set_entry_point("b1_input")

    graph.add_conditional_edges(
        "b1_input",
        route_after_b1,
        {"human_review": "human_review", "continue": "b1b_group_assets"},
    )
    graph.add_edge("b1b_group_assets", "b1c_confirm_grouping")
    graph.add_edge("b1c_confirm_grouping", "b2_process_assets")

    graph.add_conditional_edges(
        "b2_process_assets",
        route_after_processing,
        {"human_review": "human_review", "end": END},
    )
    graph.add_edge("human_review", END)

    if checkpointer is not None:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()