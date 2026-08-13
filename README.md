# Asset-converted Income AI Agent — B1 đến B3

Hệ thống AI thẩm định tài sản bảo đảm là bất động sản cho khách hàng cá nhân tại ngân hàng Việt Nam.

(Quy trình thẩm định TSBĐ đầy đủ gồm nhiều bước, từ kiểm tra hồ sơ khách hàng và tài sản, xác định diện tích đủ điều kiện, kiểm tra CIC, tìm kiếm các tài sản tương đồng để đánh giá tính hợp lý của giá trị tài 
sản.

Hệ thống hiện tại trong project này thực hiện 3 bước là nhận hồ sơ, trích xuất, kiểm tra thông tin và flag cảnh báo)

## Cấu trúc project

```text
asset-converted-income-ai-agent/
├── main.py                         # Entrypoint
├── api.py                          # FastAPI backend cho Web UI
├── graph.py                        # LangGraph StateGraph
├── schemas.py                      # GraphState + domain models (Pydantic v2)
├── requirements.txt
├── README.md
├── README_UI.md
│
├── cores/
│   ├── area_rules.py               # Rule xử lý diện tích
│   ├── document_classifier.py      # Rule phân loại giấy tờ
│   ├── identity_rules.py           # Rule đối chiếu nhân thân
│   └── land_rules.py               # Rule mục đích sử dụng đất
│
├── nodes/
│   ├── node_b1a_input.py           # B1a: OCR + phân loại giấy tờ + kiểm tra hồ sơ
│   ├── node_b1b_group_assets.py    # B1b: AI gom nhóm tài liệu theo tài sản
│   ├── node_b1c_confirm_grouping.py# B1c: Human-in-the-loop xác nhận grouping
│   ├── node_b2_process_assets.py   # B2: Xử lý song song từng tài sản
│   ├── node_b2a_extract_verify.py  # B2a: LLM extract & verify
│   ├── node_b2b_websearch_tmdv.py  # B2b: Web search bổ sung cho đất TMDV
│   ├── node_b3_flag.py             # B3: Rule-based flag engine
│   └── node_human_review.py        # Human Review
│
├── utils/
│   ├── ocr_utils.py                # EasyOCR + pdf2image utilities
│   ├── parsing_utils.py            # Parsing / data utilities
│   └── llm_config.py               # LLM configuration
│
├── static/
│   ├── index.html                  # Web UI
│   ├── app.js                      # Frontend logic
│   └── style.css                   # Frontend styling
│
├── input_data/
│   ├── test_input_1/
│   ├── test_diff_name/
│   ├── test_tmdv_da/
│   ├── test_input_missing_cccd/
│   ├── test_input_missing_gcn/
│   ├── test_input_multi_asset/
│   └── test_tmdv_websrch/
│
└── output/
    └── result.json                 # Kết quả sau khi chạy
```

## Điều kiện hồ sơ đầu vào

Theo nghiệp vụ thẩm định thực tế, hồ sơ đầu vào bắt buộc phải có tối thiểu:

| Nhóm giấy tờ                                                                                              | Bắt buộc?  | Vai trò                                                     |
| --------------------------------------------------------------------------------------------------------- | ---------- | ----------------------------------------------------------- |
| Giấy tờ nhân thân (CCCD/CMTND)                                                                            | ✅ Bắt buộc | Đối chiếu chủ tài sản                                       |
| Giấy chứng nhận QSDĐ (GCN)                                                                                | ✅ Bắt buộc | Căn cứ pháp lý gốc xác lập tài sản                          |
| Hợp đồng mua bán / Văn bản chuyển nhượng / Xác nhận chuyển nhượng / Hợp đồng thế chấp / Xác nhận thế chấp | ⭕ Bổ sung  | Đối chiếu/bổ sung thông tin biến động, mục đích sử dụng đất |

Lưu ý quan trọng: Hợp đồng mua bán/văn bản chuyển nhượng không được dùng để thay thế GCN. GCN là căn cứ pháp lý gốc, bắt buộc phải có để các bước xử lý tài sản xác định chính xác chủ sử dụng đất và mục đích sử dụng đất.

Nếu hồ sơ thiếu CCCD hoặc thiếu GCN, hệ thống sẽ:

* Sinh flag `OCR_THIEU_DU_LIEU` mức `ERROR` ngay tại B1a.
* Dừng luồng xử lý và chuyển sang Human Review.
* Không chạy các bước grouping và xử lý tài sản phía sau.

## Luồng xử lý

```text
START
  │
  ▼
B1a · Input / OCR
  Hybrid extract text (native text layer / OCR fallback)
  từng file trong folder:
  - CCCD/CMTND       → raw text nhân thân
  - GCN              → raw text GCN
  - HĐ mua bán / VB chuyển nhượng / HĐ thế chấp
                       → raw text bổ sung
  │
  ├── THIẾU CCCD/GCN
  │       │
  │       ▼
  │   Human Review
  │
  ▼
B1b · Group Assets
  - AI gom nhóm tài liệu theo từng tài sản
  - Xác định các tài liệu chưa được gán
  │
  ▼
B1c · Confirm Grouping
  - Human-in-the-loop
  - interrupt / resume
  │
  ▼
B2 · Process Assets
  Xử lý song song từng tài sản
  │
  ├── B2a · Extract & Verify
  │       - Extract owner_info, asset_info
  │       - Kiểm tra chủ tài sản khớp CCCD
  │       - Phát hiện tặng cho / thừa kế
  │       - Xác định ngày hình thành tài sản
  │       - Phân loại mục đích sử dụng đất + diện tích
  │
  └── B2b · TMDV Web Search (nếu cần)
          - Tra cứu bổ sung thông tin đất TMDV
          - Chỉ chạy khi cần xác minh thông tin dự án
  │
  ▼
B3 · Flag Engine (Rule-based)
  - Flag CHU_TAI_SAN_LECH nếu không khớp
  - Flag TANG_CHO_THUA_KE
  - Cảnh báo TAI_SAN_MOI_HINH_THANH (< 24 tháng)
  - Cảnh báo NGAY_HINH_THANH_KHONG_XAC_DINH
  - Flag TMDV_NGOAI_DU_AN
  │
  ├── has_critical_flags → Human Review Queue
  └── clean → END (tiếp tục B4)
```

## Cài đặt

```bash
# 1. Tạo virtual environment
python -m venv .venv

# Linux/Mac
source .venv/bin/activate

# Windows
.venv\Scripts\activate

# 2. Cài dependencies
pip install -r requirements.txt

# Trên Ubuntu cần thêm:
sudo apt-get install -y poppler-utils

# 3. Tạo file .env và điền API keys cần thiết
# GROQ_API_KEY=...
# TAVILY_API_KEY=...
```

## Chạy

```bash
# Chạy với file mặc định
python main.py

# Chạy với folder tùy chỉnh
python main.py --folder input_data/test_input_1 --output output/result.json
```

### Chạy Web UI (Không cần phải chạy Python trước)

UI sử dụng **FastAPI + HTML/CSS/JS**, không cần Node/npm/build step.

```bash
pip install fastapi uvicorn python-multipart

uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Mở trình duyệt:

```text
http://localhost:8000/
```

Luồng UI:

```text
Input folder
    ↓
B1a OCR
    ↓
B1b AI Grouping
    ↓
B1c Human Confirmation
    ↓
B2a → B2b → B3
    ↓
Kết quả theo từng tài sản
```

Tại B1c, người dùng có thể kéo-thả file giữa các nhóm, đổi `asset_id`, thêm/xóa nhóm tài sản và xem trước ảnh/PDF trước khi xác nhận. Sau khi xác nhận, graph được resume và tiếp tục xử lý song song các tài sản.

## Test cases

Các bộ test hiện có trong `input_data/`:

| Test case                 | Mục đích                                            |
| ------------------------- | ----------------------------------------------------|
| `test_input_1`            | Happy path                                          |
| `test_diff_name`          | Hồ sơ có tên chủ sở hữu trong giấy tờ mua bán khác  |
| `test_tmdv_da`            | Đất TMDV nhưng có thuộc dự án (không cần Web Search)|
| `test_input_missing_cccd` | Hồ sơ thiếu CCCD → Human Review                     |
| `test_input_missing_gcn`  | Hồ sơ thiếu GCN → Human Review                      |
| `test_input_multi_asset`  | Xử lý nhiều tài sản trong cùng hồ sơ                |
| `test_tmdv_websrch`       | Kiểm tra nhánh TMDV Web Search                      |

Ví dụ:

```bash
python main.py --folder input_data/test_input_multi_asset --output output/result.json
```

## Output

File `output/result.json` chứa GraphState sau khi xử lý:

* `owner_info`: Thông tin chủ tài sản
* `asset_info`: Thông tin tài sản
* `identity_check`: Kết quả kiểm tra nhân thân
* `land_purpose`: Phân loại mục đích sử dụng đất
* `flags`: Danh sách cờ cảnh báo
* `warnings`: Danh sách cảnh báo dạng text
* `has_critical_flags`: Có flag ERROR không

### Danh sách flag_type

| flag_type                          | Severity      | Ý nghĩa                                           |
| ---------------------------------- | ------------- | ------------------------------------------------- |
| `OCR_THIEU_DU_LIEU`                | ERROR/WARNING | Thiếu giấy tờ bắt buộc, hoặc extract quá ít ký tự |
| `PHAN_LOAI_GIAY_TO_KHONG_XAC_DINH` | WARNING       | Không phân loại được loại giấy tờ                 |
| `CHU_TAI_SAN_LECH`                 | ERROR         | Chủ tài sản trên GCN/HĐ không khớp CCCD           |
| `TANG_CHO_THUA_KE`                 | WARNING       | Tài sản có nguồn gốc tặng cho/thừa kế             |
| `TAI_SAN_MOI_HINH_THANH`           | WARNING       | Tài sản hình thành < 24 tháng                     |
| `NGAY_HINH_THANH_KHONG_XAC_DINH`   | WARNING       | Không xác định được ngày hình thành tài sản       |
| `TMDV_NGOAI_DU_AN`                 | ERROR         | Đất TMDV không thuộc dự án được phê duyệt         |
| `TMDV_KHONG_KHOP_RULE_BASED`       | WARNING       | Rule-based phát hiện tín hiệu TMDV mà LLM bỏ sót  |
| `TMDV_CAN_XAC_MINH_THU_CONG`       | WARNING       | Cần cán bộ tín dụng xác minh thủ công đất TMDV    |
| `TMDV_DU_AN_XAC_MINH_WEB`          | WARNING       | Đã tra cứu web bổ sung, chỉ mang tính tham khảo   |

## Dependencies chính

| Package            | Vai trò                                |
| ------------------ | -------------------------------------- |
| `langgraph`        | Orchestration framework                |
| `langchain-groq`   | Groq LLM integration                   |
| `easyocr`          | OCR tiếng Việt từ ảnh                  |
| `pdf2image`        | Chuyển PDF → ảnh cho OCR               |
| `pypdf`            | Đọc text layer native của PDF          |
| `pydantic v2`      | Schema validation                      |
| `tavily-python`    | Web search cho B2b (xác minh đất TMDV) |
| `fastapi`          | Backend Web UI                         |
| `uvicorn`          | ASGI server                            |
| `python-multipart` | Upload/form-data cho API               |
