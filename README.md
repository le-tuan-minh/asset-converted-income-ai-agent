# Asset Credit AI Agent — B1 đến B3

Hệ thống AI thẩm định tài sản bảo đảm cho khách hàng cá nhân tại ngân hàng Việt Nam.

## Cấu trúc project

```
asset-credit-agent/
├── main.py                  # Entrypoint
├── graph.py                 # LangGraph StateGraph
├── schemas.py                # GraphState + domain models (Pydantic v2)
├── ocr_utils.py              # EasyOCR + pdf2image utilities
├── nodes/
│   ├── node_b1_input.py      # B1: OCR + phân loại giấy tờ, kiểm tra giấy tờ bắt buộc
│   ├── node_b2_verify.py     # B2: Groq LLM extract & verify
│   ├── node_b2c_tmdv_websearch.py  # B2c: tra cứu web bổ sung cho đất TMDV
│   └── node_b3_flag.py       # B3: Rule-based flag engine
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

## Điều kiện hồ sơ đầu vào

Theo nghiệp vụ thẩm định thực tế, hồ sơ đầu vào **bắt buộc** phải có tối thiểu:

| Nhóm giấy tờ | Bắt buộc? | Vai trò |
|---|---|---|
| Giấy tờ nhân thân (CCCD/CMTND) | ✅ Bắt buộc | Đối chiếu chủ tài sản |
| Giấy chứng nhận QSDĐ (GCN) | ✅ Bắt buộc | Căn cứ pháp lý gốc xác lập tài sản |
| Hợp đồng mua bán / Văn bản chuyển nhượng / Xác nhận chuyển nhượng / Hợp đồng thế chấp / Xác nhận thế chấp | ⭕ Bổ sung (không bắt buộc) | Đối chiếu/bổ sung thông tin biến động, mục đích sử dụng đất |

**Lưu ý quan trọng:** Hợp đồng mua bán/văn bản chuyển nhượng **không được dùng để thay thế GCN**. GCN là căn cứ pháp lý gốc, bắt buộc phải có để B2/B3 xác định chính xác chủ sử dụng đất và mục đích sử dụng đất.

Nếu hồ sơ **thiếu CCCD hoặc thiếu GCN**, hệ thống sẽ:
- Sinh flag `OCR_THIEU_DU_LIEU` mức `ERROR` ngay tại B1.
- **Dừng luồng xử lý ngay sau B1**, chuyển thẳng sang Human Review — **không chạy B2/B2c/B3**.
- Lý do: nếu thiếu 1 trong 2 nhóm bắt buộc, dữ liệu chủ tài sản/tài sản sẽ rỗng hoặc không đầy đủ; nếu vẫn đưa vào B2, LLM buộc phải "so khớp" trên dữ liệu không đủ căn cứ, dễ dẫn đến kết luận sai lệch giả (ví dụ flag `CHU_TAI_SAN_LECH` dù bản chất chỉ là thiếu hồ sơ, không phải phát hiện sai khác thực sự).

## Luồng xử lý

```
START
  │
  ▼
B1 · Input Node
  Hybrid extract text (native text layer / OCR fallback) từng file trong folder:
  - CCCD/CMTND       → raw text nhân thân (BẮT BUỘC)
  - GCN              → raw text GCN (BẮT BUỘC)
  - HĐ mua bán / VB chuyển nhượng / HĐ thế chấp (nếu có) → raw text bổ sung
  Kiểm tra đủ 2 nhóm giấy tờ bắt buộc (nhan_than + gcn) chưa.
  │
  ├── THIẾU giấy tờ bắt buộc → has_critical_flags=True → Human Review (dừng ở đây)
  │
  ▼ (đủ điều kiện)
B2 · Verify Node  (Groq LLM - llama-3.3-70b)
  - Extract: owner_info, asset_info
  - Kiểm tra chủ tài sản khớp CCCD
  - Phát hiện tặng cho / thừa kế
  - Xác định ngày hình thành tài sản
  - Phân loại mục đích sử dụng đất + diện tích
  │
  ▼
B2c · TMDV Web Verify (nếu cần)
  Chỉ chạy khi is_tmdv=True và thuoc_du_an chưa xác định từ hồ sơ nội bộ.
  │
  ▼
B3 · Flag Engine  (Rule-based)
  - Flag CHU_TAI_SAN_LECH nếu không khớp
  - Flag TANG_CHO_THUA_KE
  - Cảnh báo TAI_SAN_MOI_HINH_THANH (< 24 tháng)
  - Cảnh báo NGAY_HINH_THANH_KHONG_XAC_DINH nếu không xác định được ngày hình thành
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

# 4. Đặt file input vào đúng vị trí (tối thiểu: 1 CCCD/CMTND + 1 GCN)
# input_data/test_input_1/cccd_kh.jpg
# input_data/test_input_1/giay_chung_nhan.pdf
# input_data/test_input_1/hop_dong_mua_ban.pdf   (bổ sung, không bắt buộc)
```

## Chạy

```bash
# Chạy với file mặc định
python main.py

# Chạy với folder tùy chỉnh
python main.py --folder input_data/test_input_1 --output output/result.json
```

## Output

File `output/result.json` chứa toàn bộ GraphState sau khi xử lý:
- `owner_info`: Thông tin chủ tài sản
- `asset_info`: Thông tin tài sản
- `identity_check`: Kết quả kiểm tra nhân thân
- `land_purpose`: Phân loại mục đích sử dụng đất
- `flags`: Danh sách cờ cảnh báo
- `warnings`: Danh sách cảnh báo dạng text
- `has_critical_flags`: Có flag ERROR không (bao gồm cả trường hợp thiếu giấy tờ bắt buộc ngay từ B1)

### Danh sách flag_type

| flag_type | Severity | Ý nghĩa |
|---|---|---|
| `OCR_THIEU_DU_LIEU` | ERROR/WARNING | Thiếu giấy tờ bắt buộc, hoặc extract quá ít ký tự |
| `PHAN_LOAI_GIAY_TO_KHONG_XAC_DINH` | WARNING | Không phân loại được loại giấy tờ |
| `CHU_TAI_SAN_LECH` | ERROR | Chủ tài sản trên GCN/HĐ không khớp CCCD |
| `TANG_CHO_THUA_KE` | WARNING | Tài sản có nguồn gốc tặng cho/thừa kế |
| `TAI_SAN_MOI_HINH_THANH` | WARNING | Tài sản hình thành < 24 tháng |
| `NGAY_HINH_THANH_KHONG_XAC_DINH` | WARNING | Không xác định được ngày hình thành tài sản — cần xác minh thủ công |
| `TMDV_NGOAI_DU_AN` | ERROR | Đất TMDV không thuộc dự án được phê duyệt |
| `TMDV_KHONG_KHOP_RULE_BASED` | WARNING | Rule-based phát hiện tín hiệu TMD mà LLM bỏ sót |
| `TMDV_CAN_XAC_MINH_THU_CONG` | WARNING | Cần cán bộ tín dụng xác minh thủ công đất TMDV |
| `TMDV_DU_AN_XAC_MINH_WEB` | WARNING | Đã tra cứu web bổ sung, chỉ mang tính tham khảo |

## Dependencies chính

| Package | Vai trò |
|---------|---------|
| `langgraph` | Orchestration framework |
| `langchain-groq` | Groq LLM integration |
| `easyocr` | OCR tiếng Việt từ ảnh |
| `pdf2image` | Chuyển PDF → ảnh cho OCR |
| `pypdf` | Đọc text layer native của PDF |
| `pydantic v2` | Schema validation |
| `tavily-python` | Web search cho B2c (xác minh đất TMDV thuộc dự án) |