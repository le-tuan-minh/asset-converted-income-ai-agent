# Asset Credit AI Agent — B1 đến B3

Hệ thống AI thẩm định tài sản bảo đảm cho khách hàng cá nhân tại ngân hàng Việt Nam.

## Cấu trúc project

```
asset-credit-agent/
├── main.py                  # Entrypoint
├── graph.py                 # LangGraph StateGraph
├── schemas.py               # GraphState + domain models (Pydantic v2)
├── ocr_utils.py             # EasyOCR + pdf2image utilities
├── nodes/
│   ├── node_b1_input.py     # B1: OCR 3 file đầu vào
│   ├── node_b2_verify.py    # B2: Groq LLM extract & verify
│   └── node_b3_flag.py      # B3: Rule-based flag engine
├── input_data/
│   └── test_input_1/
│       ├── cccd_kh.jpg
│       ├── giay_chung_nhan.pdf
│       └── hop_dong_mua_ban.pdf
├── output/
│   └── result.json          # Kết quả sau khi chạy
├── requirements.txt
└── .env.example
```

## Luồng xử lý

```
START
  │
  ▼
B1 · Input Node
  EasyOCR đọc:
  - cccd_kh.jpg       → raw text CCCD
  - giay_chung_nhan.pdf → raw text GCN
  - hop_dong_mua_ban.pdf → raw text HĐ mua bán
  │
  ▼
B2 · Verify Node  (Groq LLM - llama-3.3-70b)
  - Extract: owner_info, asset_info
  - Kiểm tra chủ tài sản khớp CCCD
  - Phát hiện tặng cho / thừa kế
  - Xác định ngày hình thành tài sản
  - Phân loại mục đích sử dụng đất + diện tích
  │
  ▼
B3 · Flag Engine  (Rule-based)
  - Flag CHU_TAI_SAN_LECH nếu không khớp
  - Flag TANG_CHO_THUA_KE
  - Cảnh báo TAI_SAN_MOI_HINH_THANH (< 24 tháng)
  - Flag TMDV_NGOAI_DU_AN
  │
  ├── has_critical_flags → Human Review Queue
  └── clean → END (tiếp tục B4)
```

## Cài đặt

```bash
# 1. Tạo virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux/Mac
.venv\Scripts\activate             # Windows

# 2. Cài dependencies
pip install -r requirements.txt

# Trên Ubuntu cần thêm:
sudo apt-get install -y poppler-utils

# 3. Tạo file .env
cp .env.example .env
# Điền GROQ_API_KEY vào .env

# 4. Đặt file input vào đúng vị trí
# input_data/test_input_1/cccd_kh.jpg
# input_data/test_input_1/giay_chung_nhan.pdf
# input_data/test_input_1/hop_dong_mua_ban.pdf
```

## Chạy

```bash
# Chạy với file mặc định
python main.py

# Chạy với file tùy chỉnh
python main.py \
  --cccd input_data/test_input_1/cccd_kh.jpg \
  --gcn input_data/test_input_1/giay_chung_nhan.pdf \
  --hop-dong input_data/test_input_1/hop_dong_mua_ban.pdf \
  --output output/result.json
```

## Output

File `output/result.json` chứa toàn bộ GraphState sau khi xử lý:
- `owner_info`: Thông tin chủ tài sản
- `asset_info`: Thông tin tài sản
- `identity_check`: Kết quả kiểm tra nhân thân
- `land_purpose`: Phân loại mục đích sử dụng đất
- `flags`: Danh sách cờ cảnh báo
- `warnings`: Danh sách cảnh báo dạng text
- `has_critical_flags`: Có flag ERROR không

## Dependencies chính

| Package | Vai trò |
|---------|---------|
| `langgraph` | Orchestration framework |
| `langchain-groq` | Groq LLM integration |
| `easyocr` | OCR tiếng Việt từ ảnh |
| `pdf2image` | Chuyển PDF → ảnh cho OCR |
| `pydantic v2` | Schema validation |