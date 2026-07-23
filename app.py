"""
Gradio web UI cho luồng thẩm định TSBĐ (nodes B1a → B1b → B1c → B2 → B3).

Chỉ là một lớp giao diện web bọc quanh graph.py / main.py — KHÔNG thay đổi
logic xử lý nào. Người dùng nhập đường dẫn folder chứa giấy tờ, bấm chạy,
nếu graph dừng ở bước human-in-the-loop (B1c) thì xác nhận/chỉnh sửa nhóm
tài sản ngay trên web thay vì qua console, rồi xem báo cáo kết quả.

Chạy:
    python app.py
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import uuid

import gradio as gr
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command

from schemas import GraphState
from graph import build_graph
from main import (
    _ALLOWED_MSGPACK_MODULES,
    _print_grouping_proposal,
    print_report,
    save_output,
)


def _capture(fn, *args, **kwargs) -> str:
    """Chạy fn(...) và trả về mọi thứ nó print() ra, thay vì in ra console."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args, **kwargs)
    return buf.getvalue()


def _interrupt_outputs(payload: dict, message: str, state: dict):
    proposal_text = _capture(_print_grouping_proposal, payload)
    groups_json = json.dumps(payload.get("asset_groups", []), ensure_ascii=False, indent=2)
    return (
        message,
        gr.update(visible=True),
        proposal_text,
        groups_json,
        gr.update(visible=False),
        "",
        None,
        state,
    )


def _final_outputs(result: dict, output_path: str, message: str, state: dict):
    final_state = GraphState(**result)
    report_text = _capture(print_report, final_state)
    save_output(final_state, output_path)
    return (
        message,
        gr.update(visible=False),
        "",
        "[]",
        gr.update(visible=True),
        report_text,
        output_path,
        state,
    )


def _error_outputs(message: str, state: dict, keep_group_visible: bool = False, groups_json: str | None = None):
    return (
        message,
        gr.update(visible=keep_group_visible),
        gr.update(),
        groups_json if groups_json is not None else gr.update(),
        gr.update(visible=False),
        "",
        None,
        state,
    )


def run_start(folder_path: str, output_path: str, state: dict):
    folder_path = (folder_path or "").strip()
    output_path = (output_path or "output/result.json").strip()

    if not folder_path:
        return _error_outputs("❌ Vui lòng nhập đường dẫn folder chứa giấy tờ đầu vào.", state)
    if not os.path.isdir(folder_path):
        return _error_outputs(f"❌ Không tìm thấy folder: {folder_path}", state)
    if not os.getenv("GROQ_API_KEY"):
        return _error_outputs("❌ GROQ_API_KEY chưa được set (kiểm tra file .env).", state)

    checkpointer = MemorySaver(
        serde=JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_MSGPACK_MODULES)
    )
    graph = build_graph(checkpointer=checkpointer)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = graph.invoke(GraphState(input_folder=folder_path), config=config)
    except Exception as e:  # noqa: BLE001 - hiển thị lỗi ra UI thay vì crash server
        return _error_outputs(f"❌ Lỗi khi xử lý: {e}", state)

    new_state = {"graph": graph, "config": config}

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        return _interrupt_outputs(
            payload, "⏸️ Đang chờ cán bộ tín dụng xác nhận gom nhóm tài sản.", new_state
        )

    return _final_outputs(result, output_path, "✅ Hoàn tất xử lý (không có bước cần xác nhận).", new_state)


def run_confirm(action: str, edited_json: str, note: str, output_path: str, state: dict):
    graph = (state or {}).get("graph")
    config = (state or {}).get("config")
    output_path = (output_path or "output/result.json").strip()

    if graph is None or config is None:
        return _error_outputs(
            "❌ Chưa có phiên xử lý nào đang chờ xác nhận. Vui lòng bấm 'Bắt đầu xử lý' trước.",
            state or {},
        )

    if action == "edit":
        try:
            groups = json.loads(edited_json)
        except json.JSONDecodeError as e:
            return _error_outputs(
                f"❌ JSON nhóm tài sản không hợp lệ: {e}", state, keep_group_visible=True, groups_json=edited_json
            )
        human_response = {"action": "edit", "asset_groups": groups, "note": note or ""}
    else:
        human_response = {"action": "confirm"}

    try:
        result = graph.invoke(Command(resume=human_response), config=config)
    except Exception as e:  # noqa: BLE001
        return _error_outputs(
            f"❌ Lỗi khi xử lý xác nhận: {e}", state, keep_group_visible=True, groups_json=edited_json
        )

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        return _interrupt_outputs(payload, "⏸️ Vẫn đang chờ xác nhận (vòng tiếp theo).", state)

    return _final_outputs(result, output_path, "✅ Hoàn tất xử lý.", state)


with gr.Blocks(title="Thẩm định TSBĐ — AI Agent") as demo:
    gr.Markdown("# 🏦 AI Agent Thẩm định tín dụng — Tài sản bảo đảm")
    gr.Markdown(
        "Nhập đường dẫn folder chứa giấy tờ đầu vào (CCCD, GCN, hợp đồng, ...). "
        "Có thể chứa nhiều tài sản trong cùng 1 hồ sơ."
    )

    with gr.Row():
        folder_input = gr.Textbox(
            label="Folder giấy tờ đầu vào",
            value="input_data/test_input_1",
            placeholder="vd: input_data/test_input_multi",
            scale=3,
        )
        output_path_input = gr.Textbox(
            label="File lưu kết quả JSON",
            value="output/result.json",
            scale=2,
        )

    run_btn = gr.Button("🚀 Bắt đầu xử lý", variant="primary")
    status_md = gr.Markdown()

    with gr.Group(visible=False) as group_section:
        gr.Markdown("## 🧑‍💼 Cần xác nhận: AI đề xuất gom nhóm tài sản")
        proposal_text = gr.Textbox(label="Đề xuất gom nhóm", lines=18, interactive=False)
        edited_groups_json = gr.Code(
            label="Nhóm tài sản (JSON) — chỉ áp dụng khi chọn hành động 'edit'",
            language="json",
            lines=14,
        )
        action_radio = gr.Radio(
            ["confirm", "edit"], value="confirm", label="Hành động",
            info="confirm = giữ nguyên đề xuất AI; edit = dùng JSON đã chỉnh sửa ở trên",
        )
        note_input = gr.Textbox(label="Ghi chú lý do chỉnh sửa (tuỳ chọn)")
        confirm_btn = gr.Button("✅ Gửi xác nhận", variant="primary")

    with gr.Group(visible=False) as report_section:
        gr.Markdown("## 📋 Kết quả thẩm định")
        report_text = gr.Textbox(label="Báo cáo", lines=30, interactive=False)
        result_file = gr.File(label="Tải file kết quả JSON")

    session_state = gr.State({})

    _outputs = [
        status_md,
        group_section,
        proposal_text,
        edited_groups_json,
        report_section,
        report_text,
        result_file,
        session_state,
    ]

    run_btn.click(run_start, inputs=[folder_input, output_path_input, session_state], outputs=_outputs)
    confirm_btn.click(
        run_confirm,
        inputs=[action_radio, edited_groups_json, note_input, output_path_input, session_state],
        outputs=_outputs,
    )


if __name__ == "__main__":
    demo.launch()
