"""
B1 - Input Node
Nhận đường dẫn 3 file, chạy EasyOCR, lưu raw text vào GraphState.
"""
from __future__ import annotations
from pathlib import Path

from schemas import GraphState, OcrRawResult, FlagItem
from ocr_utils import ocr_file


def node_b1_input(state: GraphState) -> GraphState:
    """
    LangGraph node B1.
    Input : state với cccd_path, gcn_path, hop_dong_path
    Output: state với ocr_raw được điền đầy đủ
    """
    print("\n" + "="*60)
    print("B1 · INPUT NODE — Đang chạy OCR trên 3 file đầu vào")
    print("="*60)

    updates: dict = {}
    flags = list(state.flags)
    notes = list(state.processing_notes)

    # Helper: OCR một file, bắt lỗi gracefully
    def safe_ocr(path_str: str, label: str) -> str:
        path = Path(path_str)
        if not path.exists():
            msg = f"[B1] Không tìm thấy file {label}: {path}"
            print(msg)
            flags.append(FlagItem(
                flag_type="OCR_THIEU_DU_LIEU",
                severity="ERROR",
                description=f"File không tồn tại: {path}",
                affected_field=label,
            ))
            return ""
        try:
            print(f"[B1] OCR {label}: {path.name}")
            text = ocr_file(path)
            char_count = len(text)
            print(f"[B1] {label} — nhận dạng được {char_count} ký tự")
            if char_count < 50:
                flags.append(FlagItem(
                    flag_type="OCR_THIEU_DU_LIEU",
                    severity="WARNING",
                    description=f"OCR {label} trả về ít ký tự ({char_count}). Có thể ảnh mờ hoặc scan kém.",
                    affected_field=label,
                ))
            return text
        except Exception as exc:
            msg = f"[B1] Lỗi OCR {label}: {exc}"
            print(msg)
            flags.append(FlagItem(
                flag_type="OCR_THIEU_DU_LIEU",
                severity="ERROR",
                description=str(exc),
                affected_field=label,
            ))
            return ""

    cccd_text    = safe_ocr(state.cccd_path,     "CCCD")
    gcn_text     = safe_ocr(state.gcn_path,      "GCN")
    hop_dong_text = safe_ocr(state.hop_dong_path, "HopDong")

    ocr_raw = OcrRawResult(
        cccd_text=cccd_text,
        gcn_text=gcn_text,
        hop_dong_text=hop_dong_text,
    )

    notes.append("B1 hoàn thành: OCR 3 file xong.")
    print("[B1] Hoàn thành.\n")

    return state.model_copy(update={
        "ocr_raw": ocr_raw,
        "flags": flags,
        "processing_notes": notes,
    })