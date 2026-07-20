"""
B2 - Verify Node (Reasoning AI)
Gom text theo nhóm nghiệp vụ từ state.documents (đã OCR + phân loại ở B1),
gửi lên Groq LLM để:
  - Extract thông tin chủ tài sản từ CCCD, GCN, các văn bản chuyển nhượng/thế chấp
  - So khớp chủ sở hữu
  - Xác định tặng cho / thừa kế
  - Xác định ngày hình thành tài sản
  - Phân loại mục đích sử dụng đất (theo Điều 9 Luật Đất đai 2024 + mã ký hiệu
    Thông tư 08/2024/TT-BTNMT) và diện tích, đặc biệt xác định đất TMDV có
    thuộc dự án được phê duyệt hay không (nếu căn cứ có sẵn ngay trong hồ sơ)

Sau khi LLM trả JSON, có thêm bước rule-based cross-check (nodes/land_rules.py)
để giảm rủi ro LLM bỏ sót/nhận định sai loại đất TMDV — nguyên tắc: rule-based
chỉ được phép làm hệ thống THẬN TRỌNG HƠN, không tự ý hạ thấp rủi ro.
"""
from __future__ import annotations
import json
import os
import re

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from schemas import (
    GraphState, OwnerInfo, AssetInfo,
    IdentityCheckResult, LandPurposeResult, DOCUMENT_CATEGORY_MAP, FlagItem,
)
from nodes.land_rules import detect_tmdv_rule_based, LAND_USE_CODE_REFERENCE

# ─── LLM setup ───────────────────────────────
def _get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY chưa được set trong .env")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=api_key,
    )


SYSTEM_PROMPT = """Bạn là chuyên gia thẩm định tín dụng ngân hàng Việt Nam, nắm vững
Luật Đất đai 2024 và các văn bản hướng dẫn thi hành.
Nhiệm vụ: phân tích văn bản OCR từ hồ sơ vay vốn và trích xuất thông tin chính xác.
Luôn trả về JSON hợp lệ, không có markdown, không có giải thích thêm.
Với các trường không tìm thấy trong văn bản, để chuỗi rỗng "".
TUYỆT ĐỐI KHÔNG suy diễn/phỏng đoán khi không có căn cứ rõ ràng trong văn bản —
với các trường quan trọng về pháp lý (is_tmdv, thuoc_du_an), thà để trống/null
còn hơn đoán sai, vì sai sót ở đây ảnh hưởng trực tiếp đến quyết định tín dụng.
"""

EXTRACT_PROMPT_TEMPLATE = """Dưới đây là nội dung OCR từ các tài liệu trong hồ sơ, đã được
gom theo nhóm nghiệp vụ (mỗi nhóm có thể gồm nhiều file):

=== NHÓM 1: Giấy tờ nhân thân (CCCD/CMTND) ===
{nhan_than_text}

=== NHÓM 2: Giấy chứng nhận quyền sử dụng đất (GCN) ===
{gcn_text}

=== NHÓM 3: Hợp đồng / văn bản chuyển nhượng / thế chấp ===
{hop_dong_text}

=== CĂN CỨ PHÁP LÝ PHÂN LOẠI MỤC ĐÍCH SỬ DỤNG ĐẤT ===
Theo Điều 9 Luật Đất đai 2024, đất đai được phân thành 3 nhóm: đất nông nghiệp, đất phi
nông nghiệp, đất chưa sử dụng. Mã ký hiệu loại đất ghi trên GCN/bản đồ địa chính theo
Phụ lục II - Mục A, Thông tư 08/2024/TT-BTNMT (hiệu lực từ 01/08/2024), một số mã liên
quan trực tiếp tới nghiệp vụ này:
  - ODT = Đất ở tại đô thị            - ONT = Đất ở tại nông thôn
  - TMD = Đất thương mại, dịch vụ (TMDV)   ← LOẠI ĐẤT CẦN ĐẶC BIỆT LƯU Ý
  - SKC = Đất cơ sở sản xuất phi nông nghiệp
  - SKK/SCC = Đất khu công nghiệp, cụm công nghiệp
  - LUC/LUK = Đất trồng lúa           - CLN = Đất trồng cây lâu năm
  - NTS = Đất nuôi trồng thủy sản     - NKH = Đất nông nghiệp khác

Hãy trích xuất và trả về JSON với cấu trúc sau:

{{
  "owner_info": {{
    "ho_ten": "Họ tên đầy đủ trên CCCD",
    "so_cccd": "Số CCCD 12 chữ số",
    "so_cmtnd_cu": "Số CMTND 9 chữ số nếu có",
    "ngay_sinh": "DD/MM/YYYY",
    "dia_chi_thuong_tru": "Địa chỉ thường trú"
  }},
  "asset_info": {{
    "so_gcn": "Số seri/mã GCN",
    "chu_su_dung_goc": "Họ tên người sử dụng đất/sở hữu GHI NHẬN BAN ĐẦU khi cấp GCN (mục 'Người sử dụng đất' ở trang đầu, KHÔNG lấy theo mục biến động)",
    "chu_su_dung_hien_tai": "Họ tên chủ sử dụng/sở hữu HIỆN TẠI — lấy theo biến động GẦN NHẤT trong mục 'Những thay đổi sau khi cấp GCN' nếu có; nếu GCN chưa từng có biến động thì bằng chu_su_dung_goc",
    "bien_dong_lich_su": [
      {{
        "ngay": "Ngày ghi nhận biến động (DD/MM/YYYY)",
        "noi_dung": "Tóm tắt nội dung biến động (vd: Chuyển nhượng cho ông Nguyễn Văn X theo hồ sơ số...)",
        "chu_moi": "Họ tên chủ mới sau biến động này"
      }}
    ],
    "ngay_cap_gcn": "Ngày cấp GCN lần đầu (DD/MM/YYYY)",
    "ngay_chuyen_nhuong": "Ngày chuyển nhượng/biến động gần nhất (DD/MM/YYYY)",
    "muc_dich_su_dung": "Ghi ĐÚNG NGUYÊN VĂN mục đích sử dụng đất như trên GCN (vd: Đất ở tại đô thị / Đất thương mại, dịch vụ / Đất trồng cây lâu năm...)",
    "ma_ky_hieu_dat": "Mã ký hiệu loại đất ghi trên GCN nếu có, đối chiếu bảng mã ở trên (vd: TMD, ODT, ONT, SKC...). Để trống nếu GCN không ghi mã.",
    "dia_chi_tai_san": "Địa chỉ/vị trí thửa đất ghi trên GCN: thửa số, tờ bản đồ số, xã/phường, quận/huyện, tỉnh/thành phố. KHÔNG phải địa chỉ thường trú của chủ sở hữu.",
    "dien_tich_tong": "Tổng diện tích m2",
    "dien_tich_dat_o": "Diện tích đất ở (ODT/ONT) m2",
    "dien_tich_nha_o": "Diện tích nhà ở m2",
    "dien_tich_nn": "Diện tích nông nghiệp m2",
    "dien_tich_tmdv": "Diện tích đất thương mại, dịch vụ (TMD) m2",
    "co_thong_tin_tang_cho": false,
    "thuoc_du_an": null,
    "ten_du_an": "Tên dự án đầu tư nếu hồ sơ có nêu rõ (vd: Khu đô thị..., Khu công nghiệp...). Để trống nếu không có.",
    "can_cu_phap_ly_du_an": "Số + ngày quyết định phê duyệt dự án/chủ trương đầu tư/giấy chứng nhận đầu tư NẾU hồ sơ có ghi rõ. Để trống nếu không có.",
    "nguon_goc_tai_san": "Mô tả nguồn gốc: mua bán / được cấp / tặng cho / thừa kế"
  }},
  "identity_check": {{
    "owner_matched": true,
    "matched_against": "Chỉ được trả đúng MỘT trong 3 giá trị sau (không thêm chữ khác): chu_hien_tai | chu_goc | khong_ro",
    "mismatch_fields": ["danh sách trường không khớp nếu có"],
    "is_tang_cho": false,
    "is_thua_ke": false,
    "asset_formation_date": "Ngày hình thành tài sản của CHỦ HIỆN TẠI (ưu tiên ngày biến động/chuyển nhượng gần nhất; nếu chưa từng biến động thì dùng ngày cấp GCN)",
    "asset_formation_note": "Ghi chú về thời điểm hình thành"
  }},
  "land_purpose": {{
    "muc_dich": "Đất ở / Đất thương mại, dịch vụ / Đất nông nghiệp / ... (đồng bộ với asset_info.muc_dich_su_dung)",
    "ma_ky_hieu_dat": "Mã ký hiệu loại đất tương ứng (đồng bộ với asset_info.ma_ky_hieu_dat)",
    "dien_tich_du_dieu_kien": "Diện tích đủ điều kiện quy đổi (chỉ tính đất ở ODT/ONT + nhà ở)",
    "is_tmdv": false,
    "thuoc_du_an": null,
    "ten_du_an": "Sao chép lại asset_info.ten_du_an nếu is_tmdv=true",
    "can_cu_phap_ly_du_an": "Sao chép lại asset_info.can_cu_phap_ly_du_an nếu có",
    "warning_tmdv": "Cảnh báo nếu là TMDV không thuộc dự án, hoặc TMDV nhưng chưa xác định được có thuộc dự án không"
  }}
}}

Lưu ý quan trọng:
- GCN thường có 2 phần: (1) "Người sử dụng đất/sở hữu" ghi ở trang đầu khi cấp GCN lần đầu
  → đây là chu_su_dung_goc; (2) mục "Những thay đổi sau khi cấp Giấy chứng nhận" (biến động)
  ghi các lần chuyển nhượng/tặng cho/thừa kế về sau → lấy biến động GẦN NHẤT làm
  chu_su_dung_hien_tai. Nếu KHÔNG có mục biến động nào, chu_su_dung_hien_tai = chu_su_dung_goc.
- Liệt kê ĐẦY ĐỦ mọi mục biến động tìm thấy vào bien_dong_lich_su (không chỉ mục gần nhất),
  sắp xếp theo thời gian tăng dần.
- owner_matched = true nếu họ tên / số CCCD trên hồ sơ khách hàng TRÙNG với chu_su_dung_hien_tai
  (ưu tiên) hoặc chu_su_dung_goc (nếu GCN chưa từng biến động). Ghi rõ vào matched_against
  đã so khớp với chủ nào.
- is_tang_cho = true nếu chu_su_dung_goc, hợp đồng, HOẶC bất kỳ mục nào trong bien_dong_lich_su
  có chữ "tặng cho", "cho tặng", "thừa kế", "di chúc"
- asset_info.dia_chi_tai_san là địa chỉ CỦA THỬA ĐẤT (trong mục "Thửa đất" trên GCN), khác hoàn
  toàn với owner_info.dia_chi_thuong_tru (địa chỉ thường trú của chủ sở hữu) — không được lấy
  nhầm 2 trường này của nhau.

QUY TẮC XÁC ĐỊNH is_tmdv (BẮT BUỘC TUÂN THỦ NGHIÊM NGẶT, KHÔNG SUY DIỄN):
- is_tmdv = true CHỈ KHI văn bản ghi rõ mục đích sử dụng đất là "đất thương mại, dịch vụ"
  / "đất thương mại dịch vụ", HOẶC ghi mã ký hiệu "TMD" cạnh diện tích/thửa đất trên GCN.
- KHÔNG được suy ra is_tmdv=true chỉ vì hợp đồng có nội dung kinh doanh, cho thuê mặt bằng,
  ngành nghề kinh doanh của bên mua/bán, hay tên gọi thương mại — phải có căn cứ RÕ RÀNG
  bằng mục đích sử dụng đất ghi trên GCN.
- Nếu 1 thửa đất có NHIỀU mục đích sử dụng trên cùng 1 GCN (vd vừa đất ở vừa TMDV), ghi nhận
  riêng diện tích từng loại vào dien_tich_dat_o / dien_tich_tmdv tương ứng, và is_tmdv=true
  nếu có ít nhất một phần diện tích là đất TMD.

QUY TẮC XÁC ĐỊNH thuoc_du_an (CHỈ XEM XÉT KHI is_tmdv=true):
- thuoc_du_an = true CHỈ KHI hồ sơ nêu RÕ RÀNG: tên dự án đầu tư cụ thể VÀ/HOẶC số + ngày
  quyết định phê duyệt dự án / chủ trương đầu tư / giấy chứng nhận đầu tư liên quan trực
  tiếp đến chính thửa đất này. Khi đó PHẢI điền đầy đủ ten_du_an và can_cu_phap_ly_du_an.
- thuoc_du_an = false CHỈ KHI hồ sơ nêu rõ đây là đất TMDV KHÔNG thuộc dự án (vd: "đất
  thương mại dịch vụ xen kẹt", "không thuộc quy hoạch dự án", "tự chuyển mục đích sử dụng
  đất", nguồn gốc là đất ở/đất vườn tự chuyển đổi mục đích chứ không phải đất được giao/cho
  thuê theo dự án).
- Nếu KHÔNG tìm thấy căn cứ rõ ràng theo 1 trong 2 trường hợp trên → PHẢI để thuoc_du_an =
  null (KHÔNG được đoán). Việc suy diễn sai ở trường này có rủi ro pháp lý cao vì ảnh hưởng
  trực tiếp đến việc tài sản có đủ điều kiện làm tài sản bảo đảm (TSBĐ) hay không — trường
  hợp null sẽ được hệ thống xử lý ở bước xác minh bổ sung, không phải do LLM tự quyết.
- dien_tich_du_dieu_kien chỉ bao gồm đất ở (ODT/ONT) + nhà ở, KHÔNG tính nông nghiệp và TMDV.
- Nhóm 3 có thể chứa nhiều loại văn bản (hợp đồng mua bán, văn bản/xác nhận chuyển nhượng,
  hợp đồng/xác nhận thế chấp) — hãy tổng hợp thông tin từ tất cả, ưu tiên văn bản có ngày
  gần nhất khi có xung đột.
"""


def _parse_json_safe(raw: str) -> dict:
    """Extract JSON từ response LLM, bỏ qua markdown fences."""
    raw = raw.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
    if match:
        raw = match.group(1)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        brace_match = re.search(r"\{[\s\S]+\}", raw)
        if brace_match:
            return json.loads(brace_match.group(0))
        raise


def _build_grouped_text(state: GraphState) -> dict[str, str]:
    """
    Gom raw_text của các DocumentItem theo nhóm nghiệp vụ
    (nhan_than / gcn / chuyen_nhuong+the_chap), mỗi file có header tên riêng.
    """
    groups: dict[str, list[str]] = {"nhan_than": [], "gcn": [], "hop_dong": []}

    for doc in state.documents:
        category = DOCUMENT_CATEGORY_MAP[doc.doc_type]
        if category == "nhan_than":
            groups["nhan_than"].append(f"[{doc.filename}]\n{doc.raw_text}")
        elif category == "gcn":
            groups["gcn"].append(f"[{doc.filename}]\n{doc.raw_text}")
        elif category in ("chuyen_nhuong", "the_chap"):
            groups["hop_dong"].append(f"[{doc.filename}]\n{doc.raw_text}")
        # category == "khac" (KHONG_XAC_DINH) → không đưa vào prompt B2,
        # đã được flag ở B1 để cán bộ xử lý thủ công.

    return {
        "nhan_than_text": "\n\n".join(groups["nhan_than"]),
        "gcn_text": "\n\n".join(groups["gcn"]),
        "hop_dong_text": "\n\n".join(groups["hop_dong"]),
    }


def _normalize_matched_against(raw_value) -> str:
    """
    Chuẩn hoá giá trị matched_against từ LLM về đúng 1 trong 3 literal hợp lệ.
    LLM đôi khi trả nhầm tên field (vd "chu_su_dung_hien_tai") thay vì đúng giá
    trị enum ("chu_hien_tai") do lặp lại chữ trong hướng dẫn prompt.
    """
    v = str(raw_value or "").strip().lower()
    if "hien" in v:
        return "chu_hien_tai"
    if "goc" in v:
        return "chu_goc"
    return "khong_ro"


def _safe_build(model_cls, raw_data, label):
    """
    Khởi tạo 1 domain model từ dict do LLM trả về; nếu validate lỗi (LLM trả sai
    kiểu/giá trị cho 1 field nào đó), fallback về giá trị mặc định thay vì crash
    toàn bộ node, đồng thời trả về thông báo cảnh báo để ghi vào notes/flags.
    """
    try:
        return model_cls(**(raw_data or {})), None
    except Exception as exc:
        warn_msg = (
            f"[B2] Dữ liệu '{label}' từ LLM không hợp lệ ({exc}). "
            f"Dùng giá trị mặc định cho '{label}', cần kiểm tra thủ công."
        )
        print(warn_msg)
        return model_cls(), warn_msg


def _cross_check_tmdv_rule_based(
    land_purpose: LandPurposeResult,
    asset_info: AssetInfo,
    grouped: dict[str, str],
    flags: list[FlagItem],
    notes: list[str],
) -> tuple[LandPurposeResult, AssetInfo]:
    """
    Rule-based cross-check sau khi LLM đã extract xong land_purpose.
    Nguyên tắc: CHỈ được làm hệ thống thận trọng hơn (tăng cảnh báo/giữ null),
    KHÔNG được tự động hạ is_tmdv/thuoc_du_an xuống mức "an toàn hơn" so với LLM.
    """
    source_text = f"{grouped.get('gcn_text', '')}\n{grouped.get('hop_dong_text', '')}"
    signal = detect_tmdv_rule_based(source_text)

    # 1) Rule-based phát hiện dấu hiệu TMD rõ ràng (mã "TMD" hoặc cụm từ chuẩn)
    #    nhưng LLM lại không gắn cờ is_tmdv=True → ưu tiên an toàn, override True.
    if signal["is_tmdv_signal"] and not land_purpose.is_tmdv:
        land_purpose = land_purpose.model_copy(update={"is_tmdv": True})
        flags.append(FlagItem(
            flag_type="TMDV_KHONG_KHOP_RULE_BASED",
            severity="WARNING",
            description=(
                "Rule-based phát hiện mã ký hiệu 'TMD' hoặc cụm từ 'đất thương mại, dịch vụ' "
                "trong văn bản gốc (GCN/hợp đồng), nhưng LLM không gắn cờ is_tmdv=True. "
                "Đã tự động điều chỉnh is_tmdv=True theo hướng thận trọng — "
                "cần cán bộ tín dụng đối chiếu lại bản gốc để xác nhận."
            ),
            affected_field="land_purpose.is_tmdv",
        ))
        notes.append(
            "[B2] Rule-based override: land_purpose.is_tmdv = True "
            "(phát hiện tín hiệu TMD trong text gốc mà LLM bỏ sót)."
        )

    # 2) Nếu là TMDV nhưng LLM chưa xác định được thuoc_du_an (None), và rule-based
    #    tìm thấy số quyết định phê duyệt trong văn bản → lưu làm căn cứ tham khảo,
    #    KHÔNG tự ý set thuoc_du_an=True chỉ từ 1 regex match (có thể là quyết định
    #    khác không liên quan trực tiếp tới thửa đất này).
    if land_purpose.is_tmdv and land_purpose.thuoc_du_an is None:
        if signal["decision_numbers_found"]:
            found = "; ".join(signal["decision_numbers_found"])
            land_purpose = land_purpose.model_copy(update={
                "can_cu_phap_ly_du_an": found,
                "nguon_xac_dinh_du_an": "rule_based_signal",
            })
            notes.append(
                f"[B2] Rule-based tìm thấy số quyết định liên quan trong hồ sơ ({found}) "
                "nhưng chưa đủ căn cứ để LLM khẳng định thuộc dự án. Giữ thuoc_du_an=null, "
                "cần xác minh thêm (vd bước tra cứu bổ sung ở B3)."
            )
        elif signal["negative_project_signal"]:
            notes.append(
                "[B2] Rule-based phát hiện cụm từ phủ định dự án (vd 'không thuộc dự án', "
                "'đất xen kẹt') trong văn bản, nhưng LLM chưa gắn cờ thuoc_du_an=False. "
                "Cần cán bộ tín dụng đối chiếu lại — hệ thống KHÔNG tự động set False."
            )

    # 3) Ghi lại nguồn xác định nếu LLM đã tự tin xác định trực tiếp từ hồ sơ
    if land_purpose.thuoc_du_an is not None and land_purpose.nguon_xac_dinh_du_an == "chua_xac_dinh":
        land_purpose = land_purpose.model_copy(update={"nguon_xac_dinh_du_an": "ho_so_noi_bo"})

    # 4) Đồng bộ ma_ky_hieu_dat giữa asset_info và land_purpose nếu một bên có mà bên kia thiếu
    if land_purpose.ma_ky_hieu_dat and not asset_info.ma_ky_hieu_dat:
        asset_info = asset_info.model_copy(update={"ma_ky_hieu_dat": land_purpose.ma_ky_hieu_dat})
    elif asset_info.ma_ky_hieu_dat and not land_purpose.ma_ky_hieu_dat:
        land_purpose = land_purpose.model_copy(update={"ma_ky_hieu_dat": asset_info.ma_ky_hieu_dat})

    # 5) Nếu is_tmdv=True nhưng vẫn chưa xác định thuoc_du_an → đảm bảo có cảnh báo
    #    tường minh trong warning_tmdv (không phụ thuộc hoàn toàn vào B3 mới sinh ra).
    if land_purpose.is_tmdv and land_purpose.thuoc_du_an is None and not land_purpose.warning_tmdv:
        land_purpose = land_purpose.model_copy(update={
            "warning_tmdv": (
                "Đất thương mại, dịch vụ (mã TMD) nhưng chưa xác định được có thuộc dự án "
                "được phê duyệt hay không từ hồ sơ hiện có. Cần xác minh bổ sung."
            )
        })

    return land_purpose, asset_info


def node_b2_verify(state: GraphState) -> GraphState:
    """
    LangGraph node B2.
    Gom OCR text theo nhóm nghiệp vụ, gửi lên Groq LLM, nhận lại structured JSON,
    parse thành domain models, chạy rule-based cross-check cho đất TMDV, và lưu
    vào state.
    """
    print("\n" + "=" * 60)
    print("B2 · VERIFY NODE — Gửi OCR text (đã gom nhóm) lên Groq LLM")
    print("=" * 60)

    notes = list(state.processing_notes)
    flags = list(state.flags)

    grouped = _build_grouped_text(state)

    if not grouped["nhan_than_text"] and not grouped["gcn_text"]:
        notes.append("B2 bỏ qua: không có text nhóm nhân thân/GCN từ B1.")
        print("[B2] Không có OCR text phù hợp, bỏ qua.")
        return state.model_copy(update={"processing_notes": notes})

    # Cắt ngắn text nếu quá dài (Groq có context limit)
    def truncate(text: str, max_chars: int = 4000) -> str:
        return text[:max_chars] + "\n...[đã cắt ngắn]" if len(text) > max_chars else text

    prompt_text = EXTRACT_PROMPT_TEMPLATE.format(
        nhan_than_text=truncate(grouped["nhan_than_text"]),
        gcn_text=truncate(grouped["gcn_text"]),
        hop_dong_text=truncate(grouped["hop_dong_text"], max_chars=6000),
    )

    llm = _get_llm()
    print("[B2] Đang gọi Groq LLM...")
    try:
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt_text),
        ])
        raw_output = response.content
        print(f"[B2] Nhận response từ Groq ({len(raw_output)} ký tự)")
    except Exception as exc:
        error_msg = f"[B2] Lỗi gọi Groq API: {exc}"
        print(error_msg)
        notes.append(error_msg)
        return state.model_copy(update={
            "processing_notes": notes,
            "error": str(exc),
        })

    # Parse JSON response
    try:
        data = _parse_json_safe(raw_output)
    except Exception as exc:
        error_msg = f"[B2] Không parse được JSON từ LLM: {exc}\nRaw: {raw_output[:500]}"
        print(error_msg)
        notes.append(error_msg)
        return state.model_copy(update={
            "processing_notes": notes,
            "error": str(exc),
        })

    # ── Chuẩn hoá các trường Literal dễ bị LLM trả sai định dạng ─
    ic_data = data.get("identity_check", {}) or {}
    if "matched_against" in ic_data:
        ic_data["matched_against"] = _normalize_matched_against(ic_data.get("matched_against"))

    # ── Map vào domain models — bọc từng model riêng để 1 field lỗi
    # (LLM trả sai kiểu/giá trị) không làm crash toàn bộ pipeline ─────
    owner_info, w1 = _safe_build(OwnerInfo, data.get("owner_info"), "owner_info")
    asset_info, w2 = _safe_build(AssetInfo, data.get("asset_info"), "asset_info")
    identity_check, w3 = _safe_build(IdentityCheckResult, ic_data, "identity_check")
    land_purpose, w4 = _safe_build(LandPurposeResult, data.get("land_purpose"), "land_purpose")

    for w in (w1, w2, w3, w4):
        if w:
            notes.append(w)
            flags.append(FlagItem(
                flag_type="OCR_THIEU_DU_LIEU",
                severity="WARNING",
                description=w,
                affected_field="B2_llm_output",
            ))

    # ── Rule-based cross-check cho đất TMDV (an toàn hơn, không thay LLM quyết định) ──
    land_purpose, asset_info = _cross_check_tmdv_rule_based(
        land_purpose, asset_info, grouped, flags, notes,
    )

    notes.append("B2 hoàn thành: LLM extract thành công.")
    print(
        f"[B2] owner_matched={identity_check.owner_matched} | "
        f"muc_dich={land_purpose.muc_dich} ({land_purpose.ma_ky_hieu_dat or 'N/A'}) | "
        f"is_tmdv={land_purpose.is_tmdv} | thuoc_du_an={land_purpose.thuoc_du_an} "
        f"(nguồn={land_purpose.nguon_xac_dinh_du_an})"
    )
    print("[B2] Hoàn thành.\n")

    return state.model_copy(update={
        "owner_info": owner_info,
        "asset_info": asset_info,
        "identity_check": identity_check,
        "land_purpose": land_purpose,
        "flags": flags,
        "processing_notes": notes,
    })