"""
Schemas: GraphState và các domain model cho luồng thẩm định tín dụng B1-B3.
"""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Domain models
# ─────────────────────────────────────────────

class OcrRawResult(BaseModel):
    """Raw OCR text từ từng file đầu vào."""
    cccd_text: str = ""
    gcn_text: str = ""          # Giấy chứng nhận quyền sử dụng đất
    hop_dong_text: str = ""     # Hợp đồng mua bán


class OwnerInfo(BaseModel):
    """Thông tin chủ tài sản được extract từ OCR."""
    ho_ten: str = ""
    so_cccd: str = ""           # Số CCCD / CMTND
    so_cmtnd_cu: str = ""
    ngay_sinh: str = ""
    dia_chi_thuong_tru: str = ""


class AssetInfo(BaseModel):
    """Thông tin tài sản từ GCN và hợp đồng."""
    so_gcn: str = ""
    chu_su_dung: str = ""       # Chủ sử dụng ghi trên GCN
    ngay_cap_gcn: str = ""
    ngay_chuyen_nhuong: str = ""
    muc_dich_su_dung: str = ""  # Đất ở / Nhà ở / NN / TMDV
    dien_tich_tong: str = ""
    dien_tich_dat_o: str = ""
    dien_tich_nha_o: str = ""
    dien_tich_nn: str = ""
    dien_tich_tmdv: str = ""
    co_thong_tin_tang_cho: bool = False
    thuoc_du_an: Optional[bool] = None  # Chỉ set khi là TMDV
    nguon_goc_tai_san: str = ""         # Ngày hình thành (mua / cấp / tặng cho)


class IdentityCheckResult(BaseModel):
    """Kết quả kiểm tra trùng khớp chủ tài sản."""
    owner_matched: bool = False
    mismatch_fields: list[str] = Field(default_factory=list)
    is_tang_cho: bool = False
    is_thua_ke: bool = False
    asset_formation_date: str = ""
    asset_formation_note: str = ""


class LandPurposeResult(BaseModel):
    """Kết quả phân loại mục đích sử dụng đất."""
    muc_dich: str = ""
    dien_tich_du_dieu_kien: str = ""    # Diện tích được dùng để tính giá trị BĐS
    is_tmdv: bool = False
    thuoc_du_an: Optional[bool] = None
    warning_tmdv: str = ""


class FlagItem(BaseModel):
    """Một cờ cảnh báo trong hệ thống."""
    flag_type: Literal[
        "CHU_TAI_SAN_LECH",
        "TANG_CHO_THUA_KE",
        "TAI_SAN_MOI_HINH_THANH",
        "TMDV_NGOAI_DU_AN",
        "OCR_THIEU_DU_LIEU",
    ]
    severity: Literal["WARNING", "ERROR"] = "WARNING"
    description: str = ""
    affected_field: str = ""


# ─────────────────────────────────────────────
# LangGraph State
# ─────────────────────────────────────────────

class GraphState(BaseModel):
    """
    State xuyên suốt toàn bộ LangGraph.
    Mỗi node nhận state, xử lý, trả về state đã cập nhật.
    """
    model_config = {"arbitrary_types_allowed": True}

    # Input paths (relative)
    cccd_path: str = "input_data/test_input_1/cccd_kh.jpg"
    gcn_path: str = "input_data/test_input_1/giay_chung_nhan.pdf"
    hop_dong_path: str = "input_data/test_input_1/hop_dong_mua_ban.pdf"

    # B1: OCR raw
    ocr_raw: OcrRawResult = Field(default_factory=OcrRawResult)

    # B2: Parsed entities
    owner_info: OwnerInfo = Field(default_factory=OwnerInfo)
    asset_info: AssetInfo = Field(default_factory=AssetInfo)
    identity_check: IdentityCheckResult = Field(default_factory=IdentityCheckResult)
    land_purpose: LandPurposeResult = Field(default_factory=LandPurposeResult)

    # B3: Flags & warnings
    flags: list[FlagItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    # Routing
    has_critical_flags: bool = False
    processing_notes: list[str] = Field(default_factory=list)
    error: Optional[str] = None