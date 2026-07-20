"""
OCR utilities: EasyOCR cho ảnh + pdf2image để chuyển PDF → ảnh trước khi OCR.
Hỗ trợ tiếng Việt (vi) và tiếng Anh (en).

Hybrid extraction cho PDF:
  1. Thử đọc text layer sẵn có bằng pypdf (nhanh, chính xác 100%, không tốn OCR).
  2. Nếu text layer rỗng/quá ít (PDF scan ảnh) → fallback OCR qua pdf2image + EasyOCR.
"""
from __future__ import annotations
import os
import tempfile
from pathlib import Path

import easyocr
from pdf2image import convert_from_path
from pypdf import PdfReader

# Các định dạng file được hệ thống chấp nhận làm input
SUPPORTED_EXTENSIONS: set[str] = {
    ".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp",
}

# Ngưỡng số ký tự trung bình/trang để coi PDF là "có text layer đủ dùng".
# Dưới ngưỡng này (thường do PDF scan ảnh, hoặc trang chỉ có vài chữ ký/dấu)
# thì coi là không đáng tin và chuyển sang OCR.
MIN_CHARS_PER_PAGE = 40

# Khởi tạo reader một lần duy nhất (tránh reload model nhiều lần)
_reader: easyocr.Reader | None = None


def get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        print("[OCR] Khởi tạo EasyOCR reader (vi + en)...")
        _reader = easyocr.Reader(["vi", "en"], gpu=False)
        print("[OCR] EasyOCR sẵn sàng.")
    return _reader


def list_input_files(folder_path: str | Path) -> list[Path]:
    """
    Liệt kê toàn bộ file hợp lệ (pdf/ảnh) trong một folder, sắp xếp theo tên.
    Không đệ quy vào subfolder — mỗi hồ sơ là 1 folder phẳng chứa các file giấy tờ.
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Không tìm thấy folder input: {folder}")

    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return files


# ─────────────────────────────────────────────
# OCR thuần (fallback)
# ─────────────────────────────────────────────

def ocr_image_file(image_path: str | Path) -> str:
    """
    OCR trực tiếp từ file ảnh (jpg/png/...).
    Trả về chuỗi text đã ghép từ tất cả vùng nhận dạng.
    """
    reader = get_reader()
    results = reader.readtext(str(image_path), detail=0, paragraph=True)
    return "\n".join(results)


def ocr_pdf_file(pdf_path: str | Path, dpi: int = 200) -> str:
    """
    Chuyển PDF → list ảnh (mỗi trang) → OCR từng trang → ghép text.
    dpi=200 là cân bằng tốt giữa tốc độ và chất lượng nhận dạng.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file PDF: {pdf_path}")

    print(f"[OCR] Đang chuyển PDF → ảnh (fallback OCR): {pdf_path.name}")
    pages = convert_from_path(str(pdf_path), dpi=dpi)
    print(f"[OCR] PDF có {len(pages)} trang, bắt đầu OCR...")

    reader = get_reader()
    all_text: list[str] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, page_img in enumerate(pages):
            tmp_img = os.path.join(tmp_dir, f"page_{i+1}.jpg")
            page_img.save(tmp_img, "JPEG")
            results = reader.readtext(tmp_img, detail=0, paragraph=True)
            page_text = "\n".join(results)
            all_text.append(f"--- Trang {i+1} ---\n{page_text}")
            print(f"[OCR]   Trang {i+1}/{len(pages)} xong.")

    return "\n\n".join(all_text)


# ─────────────────────────────────────────────
# Native text extraction (PDF có text layer, không cần OCR)
# ─────────────────────────────────────────────

def extract_pdf_native_text(pdf_path: str | Path) -> tuple[str, int]:
    """
    Đọc text layer sẵn có trong PDF bằng pypdf (không rasterize, không OCR).
    Trả về (text, số_trang). Nếu PDF lỗi/hỏng, trả về ("", 0).
    """
    pdf_path = Path(pdf_path)
    try:
        reader = PdfReader(str(pdf_path))
        num_pages = len(reader.pages)
        page_texts = [(page.extract_text() or "") for page in reader.pages]
        return "\n\n".join(
            f"--- Trang {i+1} ---\n{t}" for i, t in enumerate(page_texts)
        ), num_pages
    except Exception as exc:
        print(f"[OCR] Không đọc được text layer native của {pdf_path.name}: {exc}")
        return "", 0


def _is_native_text_sufficient(text: str, num_pages: int) -> bool:
    """PDF được coi là 'có text layer đủ dùng' nếu mật độ ký tự/trang đạt ngưỡng."""
    if num_pages <= 0:
        return False
    # Bỏ header "--- Trang N ---" khi tính mật độ ký tự thực
    stripped_len = len(text.strip())
    return (stripped_len / num_pages) >= MIN_CHARS_PER_PAGE


# ─────────────────────────────────────────────
# Hybrid entrypoint — dùng cho toàn bộ pipeline
# ─────────────────────────────────────────────

def extract_text_hybrid(file_path: str | Path) -> tuple[str, str]:
    """
    Trích xuất text từ 1 file, tự động chọn chiến lược phù hợp:
      - Ảnh (jpg/png/...)      → luôn OCR (không có khái niệm text layer).
      - PDF có text layer đủ dùng → đọc trực tiếp bằng pypdf (nhanh, chính xác).
      - PDF scan / text layer quá ít → fallback OCR qua pdf2image + EasyOCR.

    Trả về (text, source) với source ∈ {"native_text", "ocr"}.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Định dạng file không hỗ trợ: {suffix}")

    if suffix != ".pdf":
        # Ảnh: luôn OCR
        text = ocr_image_file(file_path)
        return text, "ocr"

    # PDF: thử native text trước
    native_text, num_pages = extract_pdf_native_text(file_path)
    if _is_native_text_sufficient(native_text, num_pages):
        print(
            f"[OCR] {file_path.name}: dùng text layer native "
            f"({len(native_text)} ký tự / {num_pages} trang) — bỏ qua OCR."
        )
        return native_text, "native_text"

    print(
        f"[OCR] {file_path.name}: text layer không đủ dùng "
        f"({len(native_text)} ký tự / {max(num_pages, 1)} trang) — fallback OCR."
    )
    ocr_text = ocr_pdf_file(file_path)
    return ocr_text, "ocr"


def ocr_file(file_path: str | Path) -> str:
    """
    Giữ lại cho tương thích ngược: OCR/extract thuần, chỉ trả về text.
    Nội bộ dùng hybrid extraction.
    """
    text, _source = extract_text_hybrid(file_path)
    return text