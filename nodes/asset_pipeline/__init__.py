"""
Pipeline xử lý cấp TÀI SẢN: B2 (Groq LLM extract & verify) → B2c (web search
TMDV nếu cần) → B3 (rule-based flag engine).

Được gọi LẶP LẠI cho từng AssetGroupCandidate đã được con người xác nhận ở
B1c — mỗi lần gọi CHỈ thấy raw_text của các file thuộc về đúng 1 tài sản
(GCN + hợp đồng/thế chấp liên quan) + các file dùng chung (CCCD), hoàn toàn
tách biệt dữ liệu giữa các tài sản để tránh LLM nhầm lẫn/pha trộn thông tin
giữa 2 thửa đất khác nhau.

  - extract.py:    B2, gọi LLM trích xuất + đối chiếu rule-based
  - websearch.py:  B2c, tra cứu web bổ sung cho đất TMDV
  - flags.py:      B3, flag/alert engine
  - pipeline.py:   điều phối B2 → B2c → B3 cho 1 tài sản
"""
from __future__ import annotations

from .pipeline import process_single_asset

__all__ = ["process_single_asset"]
