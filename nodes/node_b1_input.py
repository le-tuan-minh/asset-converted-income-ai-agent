"""
B1 - Input Node
Nhận folder_path chứa số lượng file bất kỳ (pdf/ảnh) — CÓ THỂ ứng với NHIỀU
tài sản khác nhau (vd 2 GCN của 2 thửa đất riêng biệt). Với từng file:
  1. Hybrid extract text (native text layer nếu có, fallback OCR nếu không).
  2. Phân loại loại giấy tờ (rule-based, fallback LLM).
Lưu kết quả vào state.documents. Việc GOM NHÓM file theo từng tài sản được
thực hiện ở bước B1b (nodes/node_b1b_group_assets.py), không phải ở đây.

Ràng buộc nghiệp vụ về hồ sơ đầu vào (bắt buộc, ở CẤP HỒ SƠ — áp dụng chung
cho toàn bộ folder, bất kể có bao nhiêu tài sản bên trong):
  - Giấy tờ nhân thân (CCCD/CMTND): BẮT BUỘC có ít nhất 1 file, dùng để đối
    chiếu chủ tài sản cho MỌI tài sản trong hồ sơ.
  - Giấy chứng nhận QSDĐ (GCN): BẮT BUỘC có ít nhất 1 file — là căn cứ pháp
    lý gốc xác lập tài sản. Hợp đồng mua bán / văn bản chuyển nhượng / xác
    nhận chuyển nhượng / hợp đồng thế chấp / xác nhận thế chấp CHỈ có giá trị
    BỔ SUNG, KHÔNG được dùng để thay thế GCN.

Nếu thiếu 1 trong 2 nhóm bắt buộc trên → flag ERROR và has_critical_flags=True
ngay tại B1, để graph dừng sớm (route sang human_review), không đưa dữ liệu
thiếu/rỗng vào B1b/B2 (tránh gom nhóm/so khớp trên dữ liệu không đủ căn cứ).
"""
from __future__ import annotations

from schemas import GraphState, DocumentItem, DocumentType, FlagItem, DOCUMENT_CATEGORY_MAP
from ocr_utils import list_input_files, extract_text_hybrid
from nodes.document_classifier import classify_document

MIN_CHARS_WARNING_THRESHOLD = 50


def node_b1_input(state: GraphState) -> GraphState:
    """
    LangGraph node B1.
    Input : state với input_folder
    Output: state với documents được điền đầy đủ (text + doc_type)
    """
    print("\n" + "=" * 60)
    print("B1 · INPUT NODE — OCR (hybrid) + phân loại giấy tờ")
    print("=" * 60)

    flags = list(state.flags)
    notes = list(state.processing_notes)
    documents: list[DocumentItem] = []

    # ── Liệt kê file trong folder ───────────────────────────────
    try:
        file_paths = list_input_files(state.input_folder)
    except FileNotFoundError as exc:
        msg = f"[B1] {exc}"
        print(msg)
        flags.append(FlagItem(
            flag_type="OCR_THIEU_DU_LIEU",
            severity="ERROR",
            description=str(exc),
            affected_field="input_folder",
        ))
        notes.append(msg)
        return state.model_copy(update={
            "flags": flags,
            "processing_notes": notes,
            "error": str(exc),
            "has_critical_flags": True,
        })

    if not file_paths:
        msg = f"[B1] Folder '{state.input_folder}' không có file hợp lệ (pdf/ảnh)."
        print(msg)
        flags.append(FlagItem(
            flag_type="OCR_THIEU_DU_LIEU",
            severity="ERROR",
            description=msg,
            affected_field="input_folder",
        ))
        notes.append(msg)
        return state.model_copy(update={
            "flags": flags,
            "processing_notes": notes,
            "has_critical_flags": True,
        })

    print(f"[B1] Tìm thấy {len(file_paths)} file trong '{state.input_folder}'.")

    # ── Xử lý từng file: hybrid extract + classify ──────────────
    for path in file_paths:
        print(f"\n[B1] --- Xử lý: {path.name} ---")

        try:
            text, source = extract_text_hybrid(path)
        except Exception as exc:
            msg = f"[B1] Lỗi extract text {path.name}: {exc}"
            print(msg)
            flags.append(FlagItem(
                flag_type="OCR_THIEU_DU_LIEU",
                severity="ERROR",
                description=str(exc),
                affected_field=path.name,
            ))
            documents.append(DocumentItem(
                path=str(path), filename=path.name,
                doc_type=DocumentType.KHONG_XAC_DINH,
                extraction_source="", raw_text="", char_count=0,
            ))
            continue

        char_count = len(text)
        print(f"[B1] {path.name}: {char_count} ký tự (source={source})")

        if char_count < MIN_CHARS_WARNING_THRESHOLD:
            flags.append(FlagItem(
                flag_type="OCR_THIEU_DU_LIEU",
                severity="WARNING",
                description=(
                    f"Extract '{path.name}' trả về ít ký tự ({char_count}). "
                    "Có thể ảnh mờ, scan kém, hoặc file trống."
                ),
                affected_field=path.name,
            ))

        doc_type, confidence, method = classify_document(text, filename=path.name)
        print(f"[B1] {path.name}: phân loại = {doc_type.value} "
              f"(confidence={confidence:.2f}, method={method})")

        if doc_type == DocumentType.KHONG_XAC_DINH:
            flags.append(FlagItem(
                flag_type="PHAN_LOAI_GIAY_TO_KHONG_XAC_DINH",
                severity="WARNING",
                description=(
                    f"Không xác định được loại giấy tờ cho file '{path.name}'. "
                    "Cần cán bộ tín dụng phân loại thủ công."
                ),
                affected_field=path.name,
            ))

        documents.append(DocumentItem(
            path=str(path),
            filename=path.name,
            doc_type=doc_type,
            classify_method=method,
            classify_confidence=confidence,
            extraction_source=source,
            raw_text=text,
            char_count=char_count,
        ))

    # ── Kiểm tra đủ 2 nhóm giấy tờ bắt buộc ở CẤP HỒ SƠ ──────────
    has_nhan_than = any(d.doc_type == DocumentType.CCCD for d in documents)
    has_gcn = any(d.doc_type == DocumentType.GCN for d in documents)

    has_critical = False
    if not has_nhan_than or not has_gcn:
        missing = []
        if not has_nhan_than:
            missing.append("giấy tờ nhân thân (CCCD/CMTND)")
        if not has_gcn:
            missing.append("Giấy chứng nhận QSDĐ (GCN)")
        msg = (
            f"[B1] Hồ sơ thiếu bắt buộc: {', '.join(missing)}. "
            "Không đủ căn cứ để gom nhóm tài sản / xác định chủ tài sản."
        )
        print(msg)
        flags.append(FlagItem(
            flag_type="OCR_THIEU_DU_LIEU",
            severity="ERROR",
            description=msg,
            affected_field="documents",
        ))
        notes.append(msg)
        has_critical = True
    else:
        n_gcn = sum(1 for d in documents if d.doc_type == DocumentType.GCN)
        print(f"[B1] Hồ sơ hợp lệ: {n_gcn} GCN được phát hiện → có thể ứng với {n_gcn} tài sản.")
        notes.append(f"B1 hoàn thành: {len(documents)} file, phát hiện {n_gcn} GCN.")

    print("[B1] Hoàn thành.\n")

    return state.model_copy(update={
        "documents": documents,
        "flags": flags,
        "processing_notes": notes,
        "has_critical_flags": has_critical,
    })