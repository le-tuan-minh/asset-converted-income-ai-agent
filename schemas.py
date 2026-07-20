"""
Schemas: GraphState và các domain model cho luồng thẩm định tín dụng B1-B3.
"""
from __future__ import annotations
from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Document classification
# ─────────────────────────────────────────────

class DocumentType(str, Enum):
    """Các loại giấy tờ đã định nghĩa trước cho hồ sơ TSBĐ."""
    CCCD = "CCCD"                                   # CCCD / CMTND cũ
    GCN = "GCN"                                      # Giấy chứng nhận QSDĐ
    HOP_DONG_MUA_BAN = "HOP_DONG_MUA_BAN"
    VAN_BAN_CHUYEN_NHUONG = "VAN_BAN_CHUYEN_NHUONG"
    XAC_NHAN_CHUYEN_NHUONG = "XAC_NHAN_CHUYEN_NHUONG"
    HOP_DONG_THE_CHAP = "HOP_DONG_THE_CHAP"
    XAC_NHAN_THE_CHAP = "XAC_NHAN_THE_CHAP"
    KHONG_XAC_DINH = "KHONG_XAC_DINH"               # Không phân loại được


# Nhóm nghiệp vụ dùng để build prompt cho B2 (nhiều doc_type có thể cùng nhóm)
DOCUMENT_CATEGORY_MAP: dict[DocumentType, str] = {
    DocumentType.CCCD: "nhan_than",
    DocumentType.GCN: "gcn",
    DocumentType.HOP_DONG_MUA_BAN: "chuyen_nhuong",
    DocumentType.VAN_BAN_CHUYEN_NHUONG: "chuyen_nhuong",
    DocumentType.XAC_NHAN_CHUYEN_NHUONG: "chuyen_nhuong",
    DocumentType.HOP_DONG_THE_CHAP: "the_chap",
    DocumentType.XAC_NHAN_THE_CHAP: "the_chap",
    DocumentType.KHONG_XAC_DINH: "khac",
}


class DocumentItem(BaseModel):
    """Một file input sau khi đã OCR (hybrid) và phân loại."""
    path: str
    filename: str
    doc_type: DocumentType = DocumentType.KHONG_XAC_DINH
    classify_method: Literal["rule", "llm", "none"] = "none"
    classify_confidence: float = 0.0
    extraction_source: Literal["native_text", "ocr", ""] = ""
    raw_text: str = ""
    char_count: int = 0


# ─────────────────────────────────────────────
# Domain models (kết quả B2)
# ─────────────────────────────────────────────

class OwnerInfo(BaseModel):
    """Thông tin chủ tài sản được extract từ OCR."""
    ho_ten: str = ""
    so_cccd: str = ""           # Số CCCD / CMTND
    so_cmtnd_cu: str = ""
    ngay_sinh: str = ""
    dia_chi_thuong_tru: str = ""


class BienDongItem(BaseModel):
    """Một mục biến động ghi trong GCN (chuyển nhượng, tặng cho, thừa kế, thế chấp...)."""
    ngay: str = ""            # Ngày ghi nhận biến động (DD/MM/YYYY)
    noi_dung: str = ""        # Tóm tắt nội dung biến động (vd: "Chuyển nhượng cho ông...")
    chu_moi: str = ""         # Họ tên chủ mới sau biến động này (nếu có)


class AssetInfo(BaseModel):
    """Thông tin tài sản từ GCN và các văn bản chuyển nhượng/thế chấp."""
    so_gcn: str = ""
    chu_su_dung_goc: str = ""            # Người sử dụng đất/sở hữu GHI NHẬN BAN ĐẦU khi cấp GCN
    chu_su_dung_hien_tai: str = ""       # Chủ sử dụng/sở hữu HIỆN TẠI (sau biến động gần nhất,
                                          # bằng chu_su_dung_goc nếu GCN chưa từng biến động)
    bien_dong_lich_su: list[BienDongItem] = Field(default_factory=list)
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
    matched_against: Literal["chu_hien_tai", "chu_goc", "khong_ro"] = "khong_ro"
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
        "PHAN_LOAI_GIAY_TO_KHONG_XAC_DINH",
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

    # Input: một folder chứa số lượng file bất kỳ (pdf/ảnh)
    input_folder: str = "input_data/test_input_1"

    # B1: danh sách file sau khi OCR (hybrid) + phân loại
    documents: list[DocumentItem] = Field(default_factory=list)

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