"""
B1 - Input Node
Nhận folder_path chứa số lượng file bất kỳ (pdf/ảnh), với từng file:
  1. Hybrid extract text (native text layer nếu có, fallback OCR nếu không).
  2. Phân loại loại giấy tờ (rule-based, fallback LLM).
Lưu kết quả vào state.documents.

Ràng buộc nghiệp vụ về hồ sơ đầu vào (bắt buộc):
  - Giấy tờ nhân thân (CCCD/CMTND): BẮT BUỘC, dùng để đối chiếu chủ tài sản.
  - Giấy chứng nhận QSDĐ (GCN): BẮT BUỘC, là căn cứ pháp lý gốc xác lập tài sản.
    Hợp đồng mua bán / văn bản chuyển nhượng / xác nhận chuyển nhượng / hợp đồng
    thế chấp / xác nhận thế chấp CHỈ có giá trị BỔ SUNG (đối chiếu thêm thông tin
    biến động, mục đích sử dụng...), KHÔNG được dùng để thay thế GCN.

Nếu thiếu 1 trong 2 nhóm bắt buộc trên → flag ERROR và has_critical_flags=True ngay
tại B1, để graph có thể dừng sớm (route sang human_review), không đưa dữ liệu
thiếu/rỗng vào B2 (tránh LLM phải "so khớp" trên dữ liệu không đủ căn cứ, dẫn đến
kết luận owner_matched sai/giả).
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

        # 1) Hybrid text extraction
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

        # 2) Phân loại giấy tờ
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

    # ── Kiểm tra các nhóm giấy tờ bắt buộc đã có đủ chưa ────────
    categories_present = {
        DOCUMENT_CATEGORY_MAP[d.doc_type] for d in documents
    }

    # Ràng buộc nghiệp vụ: tối thiểu phải có
    #   (1) giấy tờ nhân thân (CCCD/CMTND) để đối chiếu chủ tài sản
    #   (2) Giấy chứng nhận QSDĐ (GCN) — căn cứ pháp lý gốc, KHÔNG được thay thế
    #       bằng hợp đồng mua bán/văn bản chuyển nhượng. Các giấy tờ này (nếu có)
    #       chỉ mang giá trị bổ sung/đối chiếu thêm, không đủ tự thân xác lập
    #       quyền sử dụng đất hiện tại.
    if "nhan_than" not in categories_present:
        flags.append(FlagItem(
            flag_type="OCR_THIEU_DU_LIEU",
            severity="ERROR",
            description="Hồ sơ thiếu giấy tờ nhân thân (CCCD/CMTND) để đối chiếu chủ tài sản.",
            affected_field="nhan_than",
        ))
    if "gcn" not in categories_present:
        flags.append(FlagItem(
            flag_type="OCR_THIEU_DU_LIEU",
            severity="ERROR",
            description=(
                "Hồ sơ thiếu Giấy chứng nhận QSDĐ (GCN) — đây là giấy tờ BẮT BUỘC để "
                "xác định tài sản. Hợp đồng mua bán/văn bản chuyển nhượng (nếu có) chỉ "
                "có giá trị bổ sung, không thay thế được GCN."
            ),
            affected_field="gcn",
        ))

    has_critical_input = any(f.severity == "ERROR" for f in flags)

    notes.append(
        f"B1 hoàn thành: {len(documents)} file, "
        f"nhóm giấy tờ có mặt: {sorted(categories_present)}."
    )
    print(f"\n[B1] Hoàn thành. {len(documents)} document(s) đã xử lý.\n")
    if has_critical_input:
        print("[B1] ⛔ Hồ sơ thiếu giấy tờ bắt buộc (nhan_than/gcn) — "
              "chuyển thẳng Human Review, bỏ qua B2/B2c/B3.")

    return state.model_copy(update={
        "documents": documents,
        "flags": flags,
        "processing_notes": notes,
        "has_critical_flags": has_critical_input,
    })