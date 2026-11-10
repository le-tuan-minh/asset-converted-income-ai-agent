# Asset-converted Income AI Agent — B1 to B3

## User Guide

### 1. Enter the document folder path
<img width="941" height="504" alt="image" src="https://github.com/user-attachments/assets/e0b3ca7b-123f-4eef-a234-633c1cb138bd" />

Enter the path to the folder containing the documents to be processed.

### 2. Review OCR results and document grouping
<img width="941" height="499" alt="image" src="https://github.com/user-attachments/assets/74783a12-30f4-41b3-b25e-a25ac6734d0e" />
<img width="975" height="520" alt="image" src="https://github.com/user-attachments/assets/eb87b7f1-eacd-4b78-94b1-c8d411bd2124" />

The agent performs OCR and document classification, then groups documents belonging to the same asset.

- Click a file to view its contents.
- If a file has been placed in the wrong group, drag and drop it into the correct group.
- Review the proposed grouping and make any necessary changes before continuing.

### 3. Review asset information and warnings
<img width="975" height="520" alt="image" src="https://github.com/user-attachments/assets/d8133041-4ad3-47e2-a512-50fb9065b9ff" />
<img width="975" height="519" alt="image" src="https://github.com/user-attachments/assets/56e17eb7-a65c-4c5d-b12b-6e1d4d7ec753" />
<img width="941" height="500" alt="image" src="https://github.com/user-attachments/assets/bc323751-6bc8-43bc-8b44-fba579582a7d" />
<img width="941" height="502" alt="image" src="https://github.com/user-attachments/assets/a3c7301b-a67e-456f-8a14-25479eadf1ab" />

After the grouping is confirmed, the agent checks the information and generates warnings for each asset. Review the extracted information and warnings displayed for each asset.

---

## Project Overview

An AI system for assessing real-estate collateral for individual customers at banks in Vietnam.

(A complete collateral assessment process includes multiple steps, from checking customer and asset documents, determining eligible areas, and checking CIC, to searching for comparable properties to assess whether the asset value is reasonable.

The system in this project currently implements three stages: document intake, information extraction and verification, and warning flagging.)

## Project Structure

```text
asset-converted-income-ai-agent/
├── main.py                         # Entrypoint
├── api.py                          # FastAPI backend for the Web UI
├── graph.py                        # LangGraph StateGraph
├── schemas.py                      # GraphState + domain models (Pydantic v2)
├── requirements.txt
├── README.md
├── README_UI.md
│
├── cores/
│   ├── area_rules.py               # Area processing rules
│   ├── document_classifier.py      # Document classification rules
│   ├── identity_rules.py           # Identity verification rules
│   └── land_rules.py               # Land-use purpose rules
│
├── nodes/
│   ├── node_b1a_input.py           # B1a: OCR + document classification + document checks
│   ├── node_b1b_group_assets.py    # B1b: AI groups documents by asset
│   ├── node_b1c_confirm_grouping.py# B1c: Human-in-the-loop grouping confirmation
│   ├── node_b2_process_assets.py   # B2: Parallel processing of each asset
│   ├── node_b2a_extract_verify.py  # B2a: LLM extraction and verification
│   ├── node_b2b_websearch_tmdv.py  # B2b: Web search for commercial-service land
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
    └── result.json                 # Processing result
```

## Input Document Requirements

Based on actual collateral assessment procedures, the input application must contain at least:

| Document group | Required? | Purpose |
| --- | --- | --- |
| Identity documents (CCCD/CMTND) | Required | Verify the asset owner |
| Land Use Rights Certificate (GCN) | Required | The original legal basis establishing the asset |
| Sale and purchase contract / Transfer document / Transfer confirmation / Mortgage contract / Mortgage confirmation | Additional | Verify or supplement information about changes and land-use purpose |

Important: A sale and purchase contract or transfer document cannot replace the GCN. The GCN is the original legal document and is mandatory for the asset processing stages to accurately identify the land user and land-use purpose.

If the application is missing either the CCCD or the GCN, the system will:

* Generate an `OCR_THIEU_DU_LIEU` flag with `ERROR` severity at B1a.
* Stop the workflow and route the case to Human Review.
* Skip the subsequent grouping and asset-processing stages.

## Processing Workflow

```text
START
  │
  ▼
B1a · Input / OCR
  Hybrid text extraction (native text layer / OCR fallback)
  for each file in the folder:
  - CCCD/CMTND       → raw identity text
  - GCN              → raw GCN text
  - Sale and purchase contract / Transfer document / Mortgage contract
                       → supplementary raw text
  │
  ├── MISSING CCCD/GCN
  │       │
  │       ▼
  │   Human Review
  │
  ▼
B1b · Group Assets
  - AI groups documents by asset
  - Identifies documents that have not been assigned
  │
  ▼
B1c · Confirm Grouping
  - Human-in-the-loop
  - interrupt / resume
  │
  ▼
B2 · Process Assets
  Process each asset in parallel
  │
  ├── B2a · Extract & Verify
  │       - Extract owner_info, asset_info
  │       - Check whether the asset owner matches the CCCD
  │       - Detect gifts / inheritance
  │       - Determine the asset formation date
  │       - Classify land-use purpose and area
  │
  └── B2b · TMDV Web Search (if needed)
          - Search for supplementary information about TMDV land
          - Runs only when project information needs to be verified
  │
  ▼
B3 · Flag Engine (Rule-based)
  - Flag CHU_TAI_SAN_LECH if the information does not match
  - Flag TANG_CHO_THUA_KE
  - Warning TAI_SAN_MOI_HINH_THANH (< 24 months)
  - Warning NGAY_HINH_THANH_KHONG_XAC_DINH
  - Flag TMDV_NGOAI_DU_AN
  │
  ├── has_critical_flags → Human Review Queue
  └── clean → END (proceed to B4)
```

## Installation

```bash
# 1. Create a virtual environment
python -m venv .venv

# Linux/Mac
source .venv/bin/activate

# Windows
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# On Ubuntu, also install:
sudo apt-get install -y poppler-utils

# 3. Create a .env file and enter the required API keys
# GROQ_API_KEY=...
# TAVILY_API_KEY=...
```

## Running the Application

```bash
# Run with the default folder
python main.py

# Run with a custom folder
python main.py --folder input_data/test_input_1 --output output/result.json
```

### Run the Web UI (No need to run Python beforehand)

The UI uses **FastAPI + HTML/CSS/JS** and does not require a Node/npm/build step.

```bash
pip install fastapi uvicorn python-multipart

uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Open the browser:

```text
http://localhost:8000/
```

UI workflow:

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
Results for each asset
```

At B1c, users can drag and drop files between groups, change `asset_id`, add or delete asset groups, and preview images/PDFs before confirming. After confirmation, the graph resumes and continues processing the assets in parallel.

## Test Cases

The existing test sets in `input_data/`:

| Test case | Purpose |
| --- | --- |
| `test_input_1` | Happy path |
| `test_diff_name` | The asset owner's name differs in the sale and purchase documents |
| `test_tmdv_da` | TMDV land that belongs to a project (no Web Search required) |
| `test_input_missing_cccd` | Missing CCCD → Human Review |
| `test_input_missing_gcn` | Missing GCN → Human Review |
| `test_input_multi_asset` | Multiple assets in the same application |
| `test_tmdv_websrch` | Test the TMDV Web Search branch |

Example:

```bash
python main.py --folder input_data/test_input_multi_asset --output output/result.json
```

## Output

The `output/result.json` file contains the GraphState after processing:

* `owner_info`: Asset owner information
* `asset_info`: Asset information
* `identity_check`: Identity verification result
* `land_purpose`: Land-use purpose classification
* `flags`: List of flags
* `warnings`: List of text warnings
* `has_critical_flags`: Whether an `ERROR` flag exists

### List of flag_type Values

| flag_type | Severity | Meaning |
| --- | --- | --- |
| `OCR_THIEU_DU_LIEU` | ERROR/WARNING | Missing required documents or too few extracted characters |
| `PHAN_LOAI_GIAY_TO_KHONG_XAC_DINH` | WARNING | Unable to classify the document type |
| `CHU_TAI_SAN_LECH` | ERROR | The asset owner on the GCN/contract does not match the CCCD |
| `TANG_CHO_THUA_KE` | WARNING | The asset originated from a gift or inheritance |
| `TAI_SAN_MOI_HINH_THANH` | WARNING | Asset formed less than 24 months ago |
| `NGAY_HINH_THANH_KHONG_XAC_DINH` | WARNING | Unable to determine the asset formation date |
| `TMDV_NGOAI_DU_AN` | ERROR | TMDV land does not belong to an approved project |
| `TMDV_KHONG_KHOP_RULE_BASED` | WARNING | Rule-based checks detected TMDV signals that the LLM missed |
| `TMDV_CAN_XAC_MINH_THU_CONG` | WARNING | Credit officers need to manually verify the TMDV land |
| `TMDV_DU_AN_XAC_MINH_WEB` | WARNING | Additional web search was performed; results are for reference only |

## Main Dependencies

| Package | Role |
| --- | --- |
| `langgraph` | Orchestration framework |
| `langchain-groq` | Groq LLM integration |
| `easyocr` | Vietnamese OCR for images |
| `pdf2image` | Convert PDF to images for OCR |
| `pypdf` | Read the native text layer of PDFs |
| `pydantic v2` | Schema validation |
| `tavily-python` | Web search for B2b (TMDV land verification) |
| `fastapi` | Web UI backend |
| `uvicorn` | ASGI server |
| `python-multipart` | Upload/form-data for the API |


---

# Bản tiếng Việt

# Asset-converted Income AI Agent — B1 đến B3

## Hướng dẫn sử dụng

### 1. Nhập đường dẫn thư mục tài liệu

Nhập đường dẫn tới thư mục chứa các tài liệu cần xử lý.

### 2. Kiểm tra kết quả OCR và phân nhóm giấy tờ

Agent thực hiện OCR và phân loại giấy tờ, sau đó gom các giấy tờ thuộc cùng một tài sản vào chung một nhóm.

- Người dùng có thể click vào file để xem nội dung.
- Nếu file bị xếp nhầm nhóm, người dùng có thể kéo thả file sang nhóm khác.
- Kiểm tra lại kết quả phân nhóm và điều chỉnh nếu cần trước khi tiếp tục.

### 3. Kiểm tra thông tin tài sản và cảnh báo

Sau khi xác nhận phân nhóm, agent kiểm tra thông tin và sinh cảnh báo cho từng tài sản. Người dùng kiểm tra thông tin được trích xuất và các cảnh báo hiển thị tương ứng với từng tài sản.

---

## Giới thiệu

Hệ thống AI thẩm định tài sản bảo đảm là bất động sản cho khách hàng cá nhân tại ngân hàng Việt Nam.

(Quy trình thẩm định TSBĐ đầy đủ gồm nhiều bước, từ kiểm tra hồ sơ khách hàng và tài sản, xác định diện tích đủ điều kiện, kiểm tra CIC, tìm kiếm các tài sản tương đồng để đánh giá tính hợp lý của giá trị tài sản.

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

| Nhóm giấy tờ | Bắt buộc? | Vai trò |
| --- | --- | --- |
| Giấy tờ nhân thân (CCCD/CMTND) | ✅ Bắt buộc | Đối chiếu chủ tài sản |
| Giấy chứng nhận QSDĐ (GCN) | ✅ Bắt buộc | Căn cứ pháp lý gốc xác lập tài sản |
| Hợp đồng mua bán / Văn bản chuyển nhượng / Xác nhận chuyển nhượng / Hợp đồng thế chấp / Xác nhận thế chấp | ⭕ Bổ sung | Đối chiếu/bổ sung thông tin biến động, mục đích sử dụng đất |

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

| Test case | Mục đích |
| --- | --- |
| `test_input_1` | Happy path |
| `test_diff_name` | Hồ sơ có tên chủ sở hữu trong giấy tờ mua bán khác |
| `test_tmdv_da` | Đất TMDV nhưng có thuộc dự án (không cần Web Search) |
| `test_input_missing_cccd` | Hồ sơ thiếu CCCD → Human Review |
| `test_input_missing_gcn` | Hồ sơ thiếu GCN → Human Review |
| `test_input_multi_asset` | Xử lý nhiều tài sản trong cùng hồ sơ |
| `test_tmdv_websrch` | Kiểm tra nhánh TMDV Web Search |

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

| flag_type | Severity | Ý nghĩa |
| --- | --- | --- |
| `OCR_THIEU_DU_LIEU` | ERROR/WARNING | Thiếu giấy tờ bắt buộc, hoặc extract quá ít ký tự |
| `PHAN_LOAI_GIAY_TO_KHONG_XAC_DINH` | WARNING | Không phân loại được loại giấy tờ |
| `CHU_TAI_SAN_LECH` | ERROR | Chủ tài sản trên GCN/HĐ không khớp CCCD |
| `TANG_CHO_THUA_KE` | WARNING | Tài sản có nguồn gốc tặng cho/thừa kế |
| `TAI_SAN_MOI_HINH_THANH` | WARNING | Tài sản hình thành < 24 tháng |
| `NGAY_HINH_THANH_KHONG_XAC_DINH` | WARNING | Không xác định được ngày hình thành tài sản |
| `TMDV_NGOAI_DU_AN` | ERROR | Đất TMDV không thuộc dự án được phê duyệt |
| `TMDV_KHONG_KHOP_RULE_BASED` | WARNING | Rule-based phát hiện tín hiệu TMDV mà LLM bỏ sót |
| `TMDV_CAN_XAC_MINH_THU_CONG` | WARNING | Cần cán bộ tín dụng xác minh thủ công đất TMDV |
| `TMDV_DU_AN_XAC_MINH_WEB` | WARNING | Đã tra cứu web bổ sung, chỉ mang tính tham khảo |

## Dependencies chính

| Package | Vai trò |
| --- | --- |
| `langgraph` | Orchestration framework |
| `langchain-groq` | Groq LLM integration |
| `easyocr` | OCR tiếng Việt từ ảnh |
| `pdf2image` | Chuyển PDF → ảnh cho OCR |
| `pypdf` | Đọc text layer native của PDF |
| `pydantic v2` | Schema validation |
| `tavily-python` | Web search cho B2b (xác minh đất TMDV) |
| `fastapi` | Backend Web UI |
| `uvicorn` | ASGI server |
| `python-multipart` | Upload/form-data cho API |
