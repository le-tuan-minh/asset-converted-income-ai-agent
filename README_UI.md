# UI mới — Backend FastAPI + Frontend tách riêng (thay Gradio)

Gradio không hỗ trợ tốt kéo-thả tự do giữa các box và xem file phóng to
cuộn được, nên phần UI này tách hẳn:

- **Backend**: `api.py` — FastAPI, bọc quanh `graph.py`/`schemas.py`/`main.py`
  y hệt logic cũ (không đổi node nào), chỉ đổi lớp giao diện.
- **Frontend**: `static/index.html` + `static/app.js` + `static/style.css`
  — HTML/CSS/JS thuần (không cần Node/npm/build step).

FastAPI serve luôn thư mục `static/` ở `/`, nên chỉ cần **1 link duy nhất**
để truy cập cả UI lẫn API.

## Cài đặt

```bash
pip install fastapi uvicorn python-multipart
# (các package cũ: langgraph, langchain-groq, easyocr, pdf2image, pypdf,
#  pydantic, python-dotenv... giữ nguyên theo requirements.txt hiện có)
```

## Cách đặt file

Copy `api.py` vào **thư mục gốc repo** (ngang hàng với `graph.py`,
`schemas.py`, `main.py`, thư mục `nodes/`, `cores/`), và copy thư mục
`static/` vào cùng vị trí đó.

```
repo/
├── api.py                 ← file mới
├── static/                ← thư mục mới
│   ├── index.html
│   ├── style.css
│   └── app.js
├── graph.py
├── schemas.py
├── main.py
├── nodes/
├── cores/
└── .env
```

## Chạy

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Mở trình duyệt: **http://localhost:8000/**

## Luồng sử dụng trên UI

1. Nhập đường dẫn folder chứa giấy tờ đầu vào → bấm **"Bắt đầu xử lý"**.
   Backend chạy B1a (OCR) → B1b (AI gom nhóm) → dừng ở B1c (interrupt).
2. Giao diện hiển thị:
   - **Box "🪪 Giấy tờ nhân thân"** — CCCD/CMTND dùng chung, tách riêng.
   - **Box "📎 File chưa gán tài sản"** — nếu AI còn sót file nào.
   - **Các thẻ "🏠 Nhóm tài sản"** — mỗi thẻ là 1 tài sản do AI đề xuất,
     kèm độ tin cậy, số GCN/địa chỉ gợi ý, lý do AI gom.
   - Kéo-thả file (chip) giữa các box để điều chỉnh; đổi tên `asset_id`
     trực tiếp trên ô input đầu thẻ; bấm **"+ Thêm nhóm tài sản"** để tách
     thêm nhóm mới, hoặc **"🗑 Xóa nhóm này"** để gộp file về pool chưa gán.
   - Bấm biểu tượng 👁 trên 1 file để **xem phóng to** (ảnh: zoom +/−/reset,
     cuộn tự do trong khung; PDF: trình xem PDF gốc của trình duyệt, có
     cuộn/zoom sẵn).
3. Bấm **"✅ Giữ nguyên đề xuất AI"** (nếu không sửa gì) hoặc
   **"💾 Lưu thay đổi & xác nhận"** (nếu có kéo-thả/đổi tên/thêm/xóa nhóm).
   Backend resume graph, chạy song song B2a → B2b → B3 cho từng tài sản.
4. Kết quả hiển thị theo **từng tài sản**: bảng thông tin chủ tài sản/tài
   sản/kiểm tra nhân thân/mục đích sử dụng đất, và danh sách **flag** màu
   đỏ (ERROR) / vàng (WARNING) kèm mô tả. Có nút tải file JSON kết quả.

## Ghi chú kỹ thuật

- Session (graph + config đang interrupt) được giữ trong RAM
  (`SESSIONS` dict trong `api.py`) theo `session_id` — phù hợp chạy 1
  tiến trình nội bộ. Nếu cần nhiều worker/scale ngang, cần chuyển
  checkpointer sang SQLite/Postgres và session store ra Redis.
- Endpoint `GET /api/session/{id}/file?filename=...` phục vụ đúng file gốc
  (ảnh/PDF) từ đường dẫn đã OCR ở B1a để frontend hiển thị trong modal.
- Không đổi bất kỳ logic nghiệp vụ nào trong `nodes/`, `graph.py`,
  `schemas.py` — chỉ thêm lớp API/giao diện mới, có thể chạy song song với
  `app.py` (Gradio) cũ nếu muốn giữ lại.