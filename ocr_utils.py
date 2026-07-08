"""
OCR utilities: EasyOCR cho ảnh + pdf2image để chuyển PDF → ảnh trước khi OCR.
Hỗ trợ tiếng Việt (vi) và tiếng Anh (en).
"""
from __future__ import annotations
import os
import tempfile
from pathlib import Path

import easyocr
from pdf2image import convert_from_path

# Khởi tạo reader một lần duy nhất (tránh reload model nhiều lần)
_reader: easyocr.Reader | None = None


def get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        print("[OCR] Khởi tạo EasyOCR reader (vi + en)...")
        _reader = easyocr.Reader(["vi", "en"], gpu=False)
        print("[OCR] EasyOCR sẵn sàng.")
    return _reader


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

    print(f"[OCR] Đang chuyển PDF → ảnh: {pdf_path.name}")
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


def ocr_file(file_path: str | Path) -> str:
    """
    Auto-detect ảnh vs PDF và gọi hàm OCR tương ứng.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return ocr_pdf_file(file_path)
    elif suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}:
        return ocr_image_file(file_path)
    else:
        raise ValueError(f"Định dạng file không hỗ trợ: {suffix}")