"""
Schemas: GraphState và các domain model cho luồng thẩm định tín dụng.

CẬP NHẬT (multi-asset): hồ sơ khách hàng có thể chứa NHIỀU tài sản bảo đảm
(ví dụ 2 Giấy chứng nhận QSDĐ tương ứng 2 thửa đất khác nhau trong cùng 1
folder input). Toàn bộ pipeline được tổ chức lại theo 2 cấp:

  Cấp HỒ SƠ (folder)  : GraphState — chứa danh sách file đã OCR (documents),
                         danh sách nhóm tài sản đề xuất (asset_groups) và kết
                         quả xử lý từng tài sản (asset_results).
  Cấp TÀI SẢN (asset) : AssetResult — tương đương với "state cũ" (owner_info,
                         asset_info, identity_check, land_purpose, flags,
                         warnings, has_critical_flags) nhưng scope theo 1
                         tài sản/1 GCN duy nhất.

Việc "tài sản nào gồm những file nào" được xác định qua bước B1b (AI gom
nhóm tài sản — Reasoning AI) và LUÔN được xác nhận lại bởi con người (human-
in-the-loop) trước khi hệ thống chạy tiếp B2/B2c/B3 cho từng tài sản.
"""
from __future__ import annotations
from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────
# Helper: coerce diện tích dạng number → string
# ─────────────────────────────────────────────
# Groq LLM đôi khi trả field diện tích dạng number (vd 154.1, hoặc 0 khi
# trống) thay vì string, dù prompt đã yêu cầu string. Vì Pydantic v2 reject
# NGUYÊN CẢ object khi 1 field sai kiểu (không chỉ field đó), lỗi này có thể
# làm mất luôn các field khác của cùng object đã được LLM trích xuất ĐÚNG.
# Coerce tại đây để mọi field khác trong object không bị ảnh hưởng.
def _coerce_numeric_to_str(v):
    if isinstance(v, bool):
        # bool là subclass của int trong Python, tránh coerce nhầm True/False
        return v
    if isinstance(v, (int, float)):
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)
    return v


# ─────────────────────────────────────────────
# Helper: coerce giá trị Literal "dễ vỡ" do LLM trả tự do
# ─────────────────────────────────────────────
# Cùng vấn đề như _coerce_numeric_to_str ở trên: Pydantic v2 reject NGUYÊN CẢ
# object khi CHỈ 1 field Literal sai giá trị (vd LLM trả nhầm câu trả lời khác
# vào field enum). Coerce về giá trị mặc định an toàn TRƯỚC khi Pydantic
# validate, để các field khác trong cùng object không bị mất theo.

_NGUON_XAC_DINH_DU_AN_ALLOWED = {"ho_so_noi_bo", "rule_based_signal", "web_search", "chua_xac_dinh"}


def _coerce_nguon_xac_dinh_du_an(v):
    if v in _NGUON_XAC_DINH_DU_AN_ALLOWED:
        return v
    return "chua_xac_dinh"


_MATCHED_AGAINST_ALLOWED = {"chu_hien_tai", "chu_goc", "khong_ro"}


def _coerce_matched_against(v):
    s = str(v or "").strip().lower()
    if s in _MATCHED_AGAINST_ALLOWED:
        return s
    if "hien" in s:
        return "chu_hien_tai"
    if "goc" in s:
        return "chu_goc"
    return "khong_ro"


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

# Loại giấy tờ được coi là "định danh 1 tài sản" (mỗi file thuộc nhóm này,
# về nguyên tắc, là căn cứ pháp lý gốc của 1 thửa đất/tài sản riêng biệt).
ASSET_DEFINING_DOC_TYPES: set[DocumentType] = {DocumentType.GCN}

# Loại giấy tờ "dùng chung" cho toàn bộ hồ sơ khách hàng (không gắn với 1
# tài sản cụ thể) — CCCD/CMTND của chủ tài sản.
SHARED_DOC_TYPES: set[DocumentType] = {DocumentType.CCCD}


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
# Domain models (kết quả B2, theo TỪNG tài sản)
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
    ma_ky_hieu_dat: str = ""    # Mã ký hiệu loại đất theo Thông tư 08/2024/TT-BTNMT (vd: TMD, ODT, ONT, SKC...)
    dia_chi_tai_san: str = ""   # Địa chỉ/vị trí thửa đất (thửa số, tờ BĐ, xã/phường, quận/huyện, tỉnh/TP) ghi trên GCN
    dien_tich_tong: str = ""
    dien_tich_dat_o: str = ""
    dien_tich_nha_o: str = ""
    dien_tich_nn: str = ""      # CHỈ đất trồng cây lâu năm/lúa/NN khác (CLN/LUC/LUK/NKH) — KHÔNG gồm NTS
    dien_tich_nts: str = ""     # Diện tích đất nuôi trồng thủy sản (NTS), tách riêng khỏi dien_tich_nn
    dien_tich_tmdv: str = ""
    co_thong_tin_tang_cho: bool = False
    thuoc_du_an: Optional[bool] = None  # Chỉ set khi là TMDV
    ten_du_an: str = ""                 # Tên dự án đầu tư nếu hồ sơ có nêu rõ
    can_cu_phap_ly_du_an: str = ""      # Số + ngày QĐ phê duyệt dự án/chủ trương đầu tư nếu có
    nguon_goc_tai_san: str = ""         # Ngày hình thành (mua / cấp / tặng cho)

    # ── Trích xuất ĐỘC LẬP từ Nhóm 3 (Hợp đồng mua bán / văn bản chuyển nhượng) ──
    # Các field này PHẢI được LLM đọc trực tiếp từ văn bản hợp đồng/chuyển nhượng,
    # KHÔNG được tự động gán/đồng bộ theo chu_su_dung_hien_tai (vốn lấy từ GCN).
    # Mục đích: cho phép rule-based cross-check (nodes/identity_rules.py, lớp 2
    # trong node_b2_verify.py) phát hiện trường hợp GCN và Hợp đồng ghi 2 tên
    # khác nhau (mà nếu chỉ có 1 field duy nhất thì LLM có thể đã "hoà giải" 2
    # nguồn với nhau trước khi trả kết quả).
    ben_mua_hop_dong: str = ""            # Tên bên mua/bên nhận chuyển nhượng ghi
                                           # NGUYÊN VĂN trên hợp đồng/văn bản (Nhóm 3)
    ben_mua_so_cccd_hop_dong: str = ""    # Số CCCD/CMTND bên mua ghi trên hợp đồng, nếu có
    ben_ban_hop_dong: str = ""            # Tên bên bán/bên chuyển nhượng ghi trên hợp đồng (Nhóm 3)

    # ── Coerce kiểu dữ liệu diện tích (xem _coerce_numeric_to_str ở đầu file) ──
    @field_validator(
        "dien_tich_tong", "dien_tich_dat_o", "dien_tich_nha_o",
        "dien_tich_nn", "dien_tich_nts", "dien_tich_tmdv",
        mode="before",
    )
    @classmethod
    def _validate_area_fields(cls, v):
        return _coerce_numeric_to_str(v)


class IdentityCheckResult(BaseModel):
    """Kết quả kiểm tra trùng khớp chủ tài sản."""
    owner_matched: bool = False
    matched_against: Literal["chu_hien_tai", "chu_goc", "khong_ro"] = "khong_ro"
    mismatch_fields: list[str] = Field(default_factory=list)
    is_tang_cho: bool = False
    is_thua_ke: bool = False
    asset_formation_date: str = ""
    asset_formation_note: str = ""

    # ── Coerce giá trị Literal (xem _coerce_matched_against ở đầu file) ──
    @field_validator("matched_against", mode="before")
    @classmethod
    def _validate_matched_against(cls, v):
        return _coerce_matched_against(v)


class LandPurposeResult(BaseModel):
    """Kết quả phân loại mục đích sử dụng đất."""
    muc_dich: str = ""
    ma_ky_hieu_dat: str = ""            # Mã ký hiệu loại đất (TMD/ODT/ONT/SKC/...)
    dien_tich_du_dieu_kien: str = ""    # Diện tích được dùng để tính giá trị BĐS — LUÔN do
                                         # code tính tất định (xem nodes/area_rules.py), không
                                         # lấy trực tiếp từ số LLM tự cộng.
    is_tmdv: bool = False
    thuoc_du_an: Optional[bool] = None
    ten_du_an: str = ""                 # Tên dự án nếu is_tmdv=True và thuoc_du_an=True
    can_cu_phap_ly_du_an: str = ""      # Số/ngày QĐ phê duyệt dự án (nếu có trong hồ sơ)
    nguon_xac_dinh_du_an: Literal[
        "ho_so_noi_bo",       # LLM xác định trực tiếp từ text OCR trong hồ sơ
        "rule_based_signal",  # Rule-based tìm thấy tín hiệu nhưng chưa đủ khẳng định
        "web_search",         # Xác định qua bước tra cứu web bổ sung (Tavily)
        "chua_xac_dinh",      # Chưa có căn cứ nào, kể cả sau khi đã thử web search
    ] = "chua_xac_dinh"
    web_verification_sources: list[str] = Field(default_factory=list)  # URL nguồn đã tra cứu (audit trail)
    web_verification_summary: str = ""  # Tóm tắt lý do LLM kết luận, kèm trích dẫn nguồn
    warning_tmdv: str = ""

    # ── Coerce kiểu dữ liệu (xem _coerce_numeric_to_str ở đầu file) ──
    @field_validator("dien_tich_du_dieu_kien", mode="before")
    @classmethod
    def _validate_dien_tich_du_dieu_kien(cls, v):
        return _coerce_numeric_to_str(v)

    # ── Coerce giá trị Literal (xem _coerce_nguon_xac_dinh_du_an ở đầu file) ──
    # Đây chính là bug thực tế đã gặp: LLM trả nhầm "Giấy chứng nhận quyền sử
    # dụng đất" vào field này thay vì 1 trong 4 giá trị enum, khiến TOÀN BỘ
    # LandPurposeResult (kể cả các field đã trích xuất đúng khác) bị Pydantic
    # reject và rơi về default rỗng. Coerce tại đây để lỗi chỉ giới hạn ở
    # đúng field này.
    @field_validator("nguon_xac_dinh_du_an", mode="before")
    @classmethod
    def _validate_nguon_xac_dinh_du_an(cls, v):
        return _coerce_nguon_xac_dinh_du_an(v)


class FlagItem(BaseModel):
    """Một cờ cảnh báo trong hệ thống."""
    flag_type: Literal[
        "CHU_TAI_SAN_LECH",
        "CHU_TAI_SAN_LECH_RULE_BASED",
        "CHU_TAI_SAN_KHONG_DONG_NHAT_GIUA_HO_SO",
        "TANG_CHO_THUA_KE",
        "TAI_SAN_MOI_HINH_THANH",
        "NGAY_HINH_THANH_KHONG_XAC_DINH",
        "TMDV_NGOAI_DU_AN",
        "TMDV_KHONG_KHOP_RULE_BASED",
        "TMDV_CAN_XAC_MINH_THU_CONG",
        "TMDV_DU_AN_XAC_MINH_WEB",
        "DIEN_TICH_KHONG_KHOP",
        "OCR_THIEU_DU_LIEU",
        "PHAN_LOAI_GIAY_TO_KHONG_XAC_DINH",
        # ── Flags mới liên quan tới gom nhóm nhiều tài sản (multi-asset) ──
        "GOM_NHOM_TAI_SAN_DO_TIN_CAY_THAP",   # AI gom nhóm với confidence thấp, cần người xác nhận kỹ
        "GOM_NHOM_TAI_SAN_CON_FILE_LE",        # Có file không gán được vào tài sản nào
        "GOM_NHOM_TAI_SAN_DA_CHINH_SUA",       # Cán bộ đã chỉnh sửa nhóm do AI đề xuất
    ]
    severity: Literal["WARNING", "ERROR"] = "WARNING"
    description: str = ""
    affected_field: str = ""


# ─────────────────────────────────────────────
# Multi-asset grouping (B1b) — AI đề xuất, con người xác nhận
# ─────────────────────────────────────────────

class AssetGroupCandidate(BaseModel):
    """
    Một nhóm tài sản được đề xuất (trước hoặc sau khi con người xác nhận).
    Mỗi nhóm tương ứng với 1 tài sản bảo đảm độc lập (thường gắn với 1 GCN).
    """
    asset_id: str = ""                     # "asset_1", "asset_2"... — do hệ thống tự sinh
    so_gcn_goi_y: str = ""                 # Số GCN suy đoán được (nếu đọc được từ raw_text)
    dia_chi_goi_y: str = ""                # Địa chỉ/thửa đất suy đoán được, giúp người xác nhận dễ phân biệt
    filenames: list[str] = Field(default_factory=list)   # Danh sách file (GCN + hợp đồng/thế chấp liên quan)
    shared_filenames: list[str] = Field(default_factory=list)  # File dùng chung (CCCD) — không thuộc riêng nhóm này
    grouping_method: Literal["rule_based", "llm", "human_edited", "fallback_single"] = "rule_based"
    grouping_confidence: float = 0.0
    grouping_reason: str = ""              # Giải thích ngắn gọn vì sao gom các file này lại 1 nhóm


class AssetResult(BaseModel):
    """
    Kết quả xử lý B2 → B2c → B3 cho MỘT tài sản (tương đương "state cũ" khi
    hệ thống chỉ xử lý 1 tài sản/hồ sơ).
    """
    asset_id: str = ""
    document_filenames: list[str] = Field(default_factory=list)  # File thuộc tài sản này (đã gồm cả file dùng chung)

    owner_info: OwnerInfo = Field(default_factory=OwnerInfo)
    asset_info: AssetInfo = Field(default_factory=AssetInfo)
    identity_check: IdentityCheckResult = Field(default_factory=IdentityCheckResult)
    land_purpose: LandPurposeResult = Field(default_factory=LandPurposeResult)

    flags: list[FlagItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    has_critical_flags: bool = False
    processing_notes: list[str] = Field(default_factory=list)
    error: Optional[str] = None


# ─────────────────────────────────────────────
# LangGraph State (cấp HỒ SƠ — có thể chứa nhiều tài sản)
# ─────────────────────────────────────────────

class GraphState(BaseModel):
    """
    State xuyên suốt toàn bộ LangGraph.
    Mỗi node nhận state, xử lý, trả về state đã cập nhật.
    """
    model_config = {"arbitrary_types_allowed": True}

    # Input: một folder chứa số lượng file bất kỳ (pdf/ảnh), có thể ứng với NHIỀU tài sản
    input_folder: str = "input_data/test_input_1"

    # B1: danh sách file sau khi OCR (hybrid) + phân loại — cấp HỒ SƠ, dùng chung
    documents: list[DocumentItem] = Field(default_factory=list)

    # B1b: AI đề xuất gom nhóm tài sản (Reasoning AI, có rule-based hỗ trợ)
    asset_groups: list[AssetGroupCandidate] = Field(default_factory=list)

    # Human-in-the-loop: xác nhận/chỉnh sửa gom nhóm trước khi chạy B2-B3
    grouping_confirmed: bool = False
    grouping_human_notes: str = ""

    # B2 → B2c → B3: kết quả xử lý cho TỪNG tài sản sau khi nhóm đã được xác nhận
    asset_results: list[AssetResult] = Field(default_factory=list)

    # Flags/warnings CẤP HỒ SƠ (không gắn riêng 1 tài sản) — vd lỗi B1, lỗi gom nhóm.
    # Flags cấp tài sản nằm trong từng AssetResult.flags ở trên.
    flags: list[FlagItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    # Routing — True nếu B1 lỗi HOẶC bất kỳ tài sản nào có flag ERROR
    has_critical_flags: bool = False
    processing_notes: list[str] = Field(default_factory=list)
    error: Optional[str] = None