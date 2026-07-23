"""
api.py — Backend FastAPI cho luồng thẩm định TSBĐ (thay thế app.py/Gradio).

Bọc quanh graph.py/main.py y hệt app.py cũ — KHÔNG đổi logic xử lý.
Khác biệt: thay vì render UI bằng Gradio (không kéo-thả được, không xem
file phóng to được), backend này chỉ trả JSON + phục vụ file gốc để một
frontend tách riêng (thư mục static/, HTML+JS thuần) render giao diện đầy
đủ: kéo-thả tài sản, box giấy tờ nhân thân riêng, xem file phóng to, và
báo cáo kết quả theo từng tài sản kèm flag.

Chạy:
    pip install fastapi uvicorn python-multipart
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Sau đó mở: http://localhost:8000/
(FastAPI serve luôn thư mục static/ ở "/", và expose API ở "/api/...")
"""
from __future__ import annotations

import os
import uuid
import mimetypes
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command

from schemas import GraphState, AssetGroupCandidate
from graph import build_graph
from main import _ALLOWED_MSGPACK_MODULES, save_output


# ─────────────────────────────────────────────
# In-memory session store
# ─────────────────────────────────────────────
# Mỗi phiên xử lý (1 lần bấm "Bắt đầu xử lý" trên UI) tương ứng với 1
# thread_id của LangGraph checkpointer. Ta cần giữ graph + config sống giữa
# request "start" và request "confirm" (vì graph đang dừng ở interrupt()),
# nên lưu tạm trong RAM theo session_id. Phù hợp cho 1 tiến trình server
# chạy nội bộ (nội bộ ngân hàng); nếu cần scale multi-worker, thay bằng
# checkpointer dạng SQLite/Postgres + session store ngoài (Redis...).
class Session(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    session_id: str
    folder_path: str
    output_path: str
    graph: Any = None
    config: dict = {}
    documents: list[dict] = []  # snapshot documents cấp hồ sơ (path, filename, doc_type, ...)


SESSIONS: dict[str, Session] = {}


app = FastAPI(title="Thẩm định TSBĐ — API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _dump(obj: Any) -> Any:
    """graph.invoke() có thể trả về các phần tử vẫn là instance Pydantic
    (DocumentItem, AssetGroupCandidate, ...) thay vì dict thuần — tuỳ phiên
    bản LangGraph/cách state được merge lại sau interrupt. Luôn ép về dict
    trước khi lưu vào Session hoặc trả ra JSON để tránh lỗi validate kiểu
    list[dict]."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj


def _dump_list(items: Any) -> "list[dict]":
    return [_dump(x) for x in (items or [])]

# doc_type coi là "giấy tờ nhân thân" -> hiển thị ở box riêng trên UI
IDENTITY_DOC_TYPES = {"CCCD", "CMTND", "CAN_CUOC_CONG_DAN"}


def _doc_to_public(d: dict) -> dict:
    """Loại raw_text (nặng, không cần cho UI) trước khi trả về client."""
    out = {k: v for k, v in d.items() if k != "raw_text"}
    out["is_identity_doc"] = str(out.get("doc_type", "")).upper() in IDENTITY_DOC_TYPES
    return out


def _group_to_public(g: AssetGroupCandidate | dict) -> dict:
    return g.model_dump() if isinstance(g, AssetGroupCandidate) else g


def _interrupt_response(session: Session, result: dict, message: str) -> dict:
    payload = result["__interrupt__"][0].value
    documents = _dump_list(result.get("documents")) or session.documents
    session.documents = documents
    return {
        "status": "awaiting_confirmation",
        "session_id": session.session_id,
        "message": message,
        "documents": [_doc_to_public(d) for d in documents],
        "asset_groups": _dump_list(payload.get("asset_groups", [])),
    }


def _final_response(session: Session, result: dict, message: str) -> dict:
    final_state = GraphState(**result)
    try:
        save_output(final_state, session.output_path)
    except Exception:
        pass  # việc lưu file JSON không được làm hỏng response API

    dump = final_state.model_dump()
    return {
        "status": "done",
        "session_id": session.session_id,
        "message": message,
        "documents": [_doc_to_public(d) for d in dump.get("documents", [])],
        "asset_groups": [_group_to_public(g) for g in dump.get("asset_groups", [])],
        "asset_results": dump.get("asset_results", []),
        "flags": dump.get("flags", []),
        "warnings": dump.get("warnings", []),
        "has_critical_flags": dump.get("has_critical_flags", False),
        "processing_notes": dump.get("processing_notes", []),
        "error": dump.get("error"),
        "output_path": session.output_path,
    }


# ─────────────────────────────────────────────
# Request models
# ─────────────────────────────────────────────

class StartRequest(BaseModel):
    folder_path: str
    output_path: str = "output/result.json"


class ConfirmRequest(BaseModel):
    session_id: str
    action: str  # "confirm" | "edit"
    asset_groups: list[dict] = []
    note: str = ""


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.post("/api/start")
def start(req: StartRequest):
    folder_path = (req.folder_path or "").strip()
    output_path = (req.output_path or "output/result.json").strip()

    if not folder_path:
        raise HTTPException(400, "Vui lòng nhập đường dẫn folder chứa giấy tờ đầu vào.")
    if not os.path.isdir(folder_path):
        raise HTTPException(400, f"Không tìm thấy folder: {folder_path}")
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(500, "GROQ_API_KEY chưa được set (kiểm tra file .env).")

    checkpointer = MemorySaver(
        serde=JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_MSGPACK_MODULES)
    )
    graph = build_graph(checkpointer=checkpointer)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = graph.invoke(GraphState(input_folder=folder_path), config=config)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"Lỗi khi xử lý: {e}")

    session_id = str(uuid.uuid4())
    session = Session(
        session_id=session_id,
        folder_path=folder_path,
        output_path=output_path,
        graph=graph,
        config=config,
        documents=_dump_list(result.get("documents")),
    )
    SESSIONS[session_id] = session

    if "__interrupt__" in result:
        return _interrupt_response(
            session, result, "Đang chờ cán bộ tín dụng xác nhận gom nhóm tài sản."
        )
    return _final_response(session, result, "Hoàn tất xử lý (không có bước cần xác nhận).")


@app.post("/api/confirm")
def confirm(req: ConfirmRequest):
    session = SESSIONS.get(req.session_id)
    if session is None or session.graph is None:
        raise HTTPException(404, "Không tìm thấy phiên xử lý. Vui lòng bắt đầu lại.")

    if req.action == "edit":
        try:
            groups = [AssetGroupCandidate(**g) for g in req.asset_groups]
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"Dữ liệu nhóm tài sản không hợp lệ: {e}")
        human_response = {
            "action": "edit",
            "asset_groups": [g.model_dump() for g in groups],
            "note": req.note or "",
        }
    else:
        human_response = {"action": "confirm"}

    try:
        result = session.graph.invoke(Command(resume=human_response), config=session.config)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"Lỗi khi xử lý xác nhận: {e}")

    if "__interrupt__" in result:
        return _interrupt_response(session, result, "Vẫn đang chờ xác nhận (vòng tiếp theo).")
    return _final_response(session, result, "Hoàn tất xử lý.")


@app.get("/api/session/{session_id}/file")
def get_file(session_id: str, filename: str):
    """Phục vụ file gốc (ảnh/PDF) theo filename để frontend xem/phóng to."""
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, "Không tìm thấy phiên xử lý.")

    match: Optional[dict] = next(
        (d for d in session.documents if d.get("filename") == filename), None
    )
    file_path = match["path"] if match else str(Path(session.folder_path) / filename)

    if not os.path.isfile(file_path):
        raise HTTPException(404, f"Không tìm thấy file: {filename}")

    media_type, _ = mimetypes.guess_type(file_path)
    return FileResponse(file_path, media_type=media_type or "application/octet-stream")


@app.get("/api/session/{session_id}/result-file")
def get_result_file(session_id: str):
    session = SESSIONS.get(session_id)
    if session is None or not os.path.isfile(session.output_path):
        raise HTTPException(404, "Chưa có file kết quả.")
    return FileResponse(
        session.output_path, media_type="application/json", filename=Path(session.output_path).name
    )


@app.get("/api/health")
def health():
    return JSONResponse({"ok": True, "groq_api_key_set": bool(os.getenv("GROQ_API_KEY"))})


# ─────────────────────────────────────────────
# Serve frontend tĩnh (HTML/CSS/JS tách riêng khỏi Gradio)
# ─────────────────────────────────────────────
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")