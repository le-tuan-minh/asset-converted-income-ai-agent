"""
B2c - TMDV Web Verification Node (bổ sung, chạy giữa B2 và B3)

Chỉ kích hoạt khi:
  - land_purpose.is_tmdv == True, VÀ
  - land_purpose.thuoc_du_an is None (B2 + rule-based cross-check đã KHÔNG đủ căn cứ
    xác định trực tiếp từ hồ sơ nội bộ)

Cơ chế: Groq LLM tool-calling với 1 tool duy nhất là tavily_search — LLM tự quyết định
truy vấn gì, đọc kết quả, và kết luận có trích dẫn nguồn cụ thể (URL). Đây là bước
"tra cứu bổ sung", KHÔNG phải nguồn xác nhận pháp lý cuối cùng — do đó:
  - Chỉ chấp nhận kết luận true/false khi LLM tự báo độ tin cậy CAO và có ít nhất 1 nguồn
    trích dẫn cụ thể; mọi trường hợp khác giữ nguyên None (chưa xác định).
  - Dù kết luận true hay false, LUÔN sinh flag WARNING nhắc cán bộ tín dụng xác minh lại
    qua kênh chính thức (cổng quy hoạch tỉnh / Sở TN&MT) trước khi ra quyết định cuối —
    không để hệ thống "âm thầm" tự tin dựa hoàn toàn vào search web.
  - KHÔNG đưa PII của khách hàng (họ tên, số CCCD) vào truy vấn — chỉ dùng tên dự án,
    địa chỉ thửa đất, số GCN.
"""
from __future__ import annotations
import json
import os
import re

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from schemas import GraphState, FlagItem, LandPurposeResult

MAX_TOOL_CALLS = 3
MAX_SNIPPET_CHARS = 500


# ─────────────────────────────────────────────
# Tavily tool (chỉ tạo khi thực sự cần gọi, tránh khởi tạo client thừa)
# ─────────────────────────────────────────────

def _build_tavily_tool():
    from tavily import TavilyClient  # import trễ để không bắt buộc cài đặt nếu không dùng node này

    api_key = os.getenv("TAVILY_API_KEY")
    client = TavilyClient(api_key=api_key)

    @tool
    def tavily_search(query: str) -> str:
        """Tìm kiếm trên web (qua Tavily) để tra cứu quyết định phê duyệt dự án bất động sản,
        quy hoạch đất đai, hoặc thông tin dự án đầu tư tại Việt Nam. Trả về JSON danh sách
        kết quả gồm title, url, content (đoạn trích ngắn)."""
        try:
            resp = client.search(query=query, max_results=5, search_depth="advanced")
            results = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": (r.get("content") or "")[:MAX_SNIPPET_CHARS],
                }
                for r in resp.get("results", [])
            ]
            return json.dumps(results, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

    return tavily_search


# ─────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """Bạn là trợ lý tra cứu pháp lý bất động sản Việt Nam cho ngân hàng.
Nhiệm vụ: xác định 1 thửa đất thương mại, dịch vụ (TMDV) có THUỘC dự án đầu tư đã được cơ
quan nhà nước có thẩm quyền phê duyệt hay không, bằng cách tìm kiếm trên web.

QUY TẮC BẮT BUỘC:
- Chỉ dùng thông tin định danh dự án/thửa đất (tên dự án, địa chỉ, số GCN) để tìm kiếm.
  TUYỆT ĐỐI KHÔNG tìm kiếm bằng tên hoặc số CCCD của cá nhân khách hàng.
- Ưu tiên nguồn chính thống: cổng thông tin điện tử UBND tỉnh/thành, Sở Tài nguyên & Môi
  trường, Sở Xây dựng, Cổng thông tin điện tử Chính phủ, báo chí uy tín (không dùng diễn đàn,
  mạng xã hội, web rao vặt bất động sản làm căn cứ kết luận).
- Có thể gọi tool tavily_search tối đa vài lần với các từ khóa khác nhau để tìm đủ căn cứ.
- CHỈ kết luận thuộc dự án (true) hoặc không thuộc dự án (false) khi có bằng chứng CỤ THỂ,
  RÕ RÀNG (tên dự án khớp, hoặc số quyết định phê duyệt khớp) — nếu chỉ tìm thấy thông tin
  chung chung, không khớp rõ ràng với thửa đất đang xét, PHẢI kết luận "khong_xac_dinh".
  Thà báo không xác định còn hơn kết luận sai — sai ở đây ảnh hưởng trực tiếp đến quyết định
  tín dụng của khách hàng.
"""

FINAL_VERDICT_INSTRUCTION = """Dựa trên toàn bộ kết quả tìm kiếm ở trên, hãy trả lời CHỈ bằng
JSON theo đúng cấu trúc sau (không markdown, không giải thích thêm):

{
  "ket_luan": "thuoc_du_an" | "khong_thuoc_du_an" | "khong_xac_dinh",
  "do_tin_cay": "cao" | "trung_binh" | "thap",
  "ten_du_an_xac_nhan": "Tên dự án tìm được nếu có, để trống nếu không",
  "can_cu": "Mô tả ngắn gọn căn cứ (vd: số quyết định phê duyệt, tên văn bản) - để trống nếu không có",
  "nguon_trich_dan": ["url1", "url2"],
  "ghi_chu": "Giải thích ngắn gọn vì sao kết luận như vậy, hoặc vì sao không xác định được"
}

Nếu không tìm được bằng chứng đủ rõ ràng, PHẢI trả ket_luan = "khong_xac_dinh" và
do_tin_cay = "thap", KHÔNG được đoán.
"""


def _build_task_message(asset_info, land_purpose) -> str:
    parts = [
        f"- Loại đất: {land_purpose.muc_dich or 'Đất thương mại, dịch vụ'} "
        f"(mã {land_purpose.ma_ky_hieu_dat or 'TMD'})",
        f"- Diện tích TMDV: {asset_info.dien_tich_tmdv or 'N/A'} m2",
    ]
    if asset_info.ten_du_an:
        parts.append(f"- Tên dự án ghi trong hồ sơ (nếu có): {asset_info.ten_du_an}")
    if asset_info.dia_chi_tai_san:
        parts.append(f"- Địa chỉ thửa đất: {asset_info.dia_chi_tai_san}")
    if asset_info.so_gcn:
        parts.append(f"- Số GCN: {asset_info.so_gcn}")
    if land_purpose.can_cu_phap_ly_du_an:
        parts.append(f"- Manh mối số quyết định tìm được trong hồ sơ: {land_purpose.can_cu_phap_ly_du_an}")

    return (
        "Hãy tra cứu web để xác định thửa đất TMDV sau đây có thuộc dự án đầu tư đã được "
        "phê duyệt hay không:\n" + "\n".join(parts)
    )


def _parse_json_safe(raw: str) -> dict:
    raw = (raw or "").strip()
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


def _run_agent(asset_info, land_purpose) -> tuple[dict, list[str]]:
    """
    Vòng lặp tool-calling đơn giản: LLM tự gọi tavily_search tối đa MAX_TOOL_CALLS lần,
    sau đó bị ép trả JSON kết luận cuối cùng.
    Trả về (verdict_dict, danh_sach_url_da_tra_cuu).
    """
    tavily_tool = _build_tavily_tool()
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    llm_with_tools = llm.bind_tools([tavily_tool])

    messages = [
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        HumanMessage(content=_build_task_message(asset_info, land_purpose)),
    ]

    urls_seen: list[str] = []
    tool_calls_used = 0

    while tool_calls_used < MAX_TOOL_CALLS:
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        if not tool_calls:
            break

        for tc in tool_calls:
            if tool_calls_used >= MAX_TOOL_CALLS:
                break
            tool_calls_used += 1
            print(f"[B2c] Tool call #{tool_calls_used}: tavily_search({tc['args']})")
            result_str = tavily_tool.invoke(tc["args"])
            try:
                parsed = json.loads(result_str)
                if isinstance(parsed, list):
                    urls_seen.extend(r.get("url", "") for r in parsed if r.get("url"))
            except Exception:
                pass
            messages.append(ToolMessage(content=result_str, tool_call_id=tc["id"]))

    # Ép LLM trả kết luận cuối cùng dạng JSON (không cho gọi tool nữa ở bước này)
    messages.append(HumanMessage(content=FINAL_VERDICT_INSTRUCTION))
    final_msg = llm.invoke(messages)
    verdict = _parse_json_safe(final_msg.content)

    # unique, giữ thứ tự
    urls_unique = list(dict.fromkeys(u for u in urls_seen if u))
    return verdict, urls_unique


def node_b2c_tmdv_websearch(state: GraphState) -> GraphState:
    """LangGraph node B2c — tra cứu web bổ sung cho đất TMDV chưa xác định thuộc dự án."""
    print("\n" + "=" * 60)
    print("B2c · TMDV WEB VERIFY — Tra cứu bổ sung qua Tavily (nếu cần)")
    print("=" * 60)

    notes = list(state.processing_notes)
    flags = list(state.flags)
    lp = state.land_purpose
    ai = state.asset_info

    # ── Guard 1: chỉ chạy khi cần ───────────────────────────────
    if not (lp.is_tmdv and lp.thuoc_du_an is None):
        print("[B2c] Bỏ qua: không phải TMDV chưa xác định, không cần tra cứu web.")
        return state

    # ── Guard 2: thiếu API key → không chặn pipeline, chỉ ghi nhận ─
    if not os.getenv("TAVILY_API_KEY"):
        msg = "[B2c] Bỏ qua tra cứu web: chưa cấu hình TAVILY_API_KEY trong .env."
        print(msg)
        notes.append(msg)
        return state.model_copy(update={"processing_notes": notes})

    # ── Guard 3: không đủ thông tin định danh để search có ý nghĩa ─
    if not ai.ten_du_an and not ai.dia_chi_tai_san and not lp.can_cu_phap_ly_du_an:
        msg = (
            "[B2c] Bỏ qua tra cứu web: hồ sơ không có tên dự án/địa chỉ thửa đất để xây "
            "truy vấn có ý nghĩa. Cần cán bộ tín dụng xác minh thủ công."
        )
        print(msg)
        notes.append(msg)
        flags.append(FlagItem(
            flag_type="TMDV_CAN_XAC_MINH_THU_CONG",
            severity="WARNING",
            description=(
                "Đất TMDV chưa xác định thuộc dự án, và hồ sơ không đủ thông tin định danh "
                "(tên dự án/địa chỉ) để tra cứu bổ sung qua web. Cần cán bộ tín dụng xác minh "
                "thủ công qua cổng thông tin quy hoạch địa phương hoặc Sở TN&MT."
            ),
            affected_field="land_purpose.thuoc_du_an",
        ))
        return state.model_copy(update={"flags": flags, "processing_notes": notes})

    # ── Chạy agent tra cứu ───────────────────────────────────────
    try:
        verdict, urls = _run_agent(ai, lp)
    except Exception as exc:
        msg = f"[B2c] Lỗi khi chạy web verification: {exc}"
        print(msg)
        notes.append(msg)
        flags.append(FlagItem(
            flag_type="TMDV_CAN_XAC_MINH_THU_CONG",
            severity="WARNING",
            description=(
                f"Tra cứu web bị lỗi ({exc}). Đất TMDV vẫn chưa xác định thuộc dự án, "
                "cần cán bộ tín dụng xác minh thủ công."
            ),
            affected_field="land_purpose.thuoc_du_an",
        ))
        return state.model_copy(update={"flags": flags, "processing_notes": notes})

    ket_luan = verdict.get("ket_luan", "khong_xac_dinh")
    do_tin_cay = verdict.get("do_tin_cay", "thap")
    nguon_trich_dan = verdict.get("nguon_trich_dan") or urls
    ghi_chu = verdict.get("ghi_chu", "")

    print(f"[B2c] Kết luận LLM: {ket_luan} (độ tin cậy: {do_tin_cay}), "
          f"{len(nguon_trich_dan)} nguồn trích dẫn.")

    # ── Nguyên tắc an toàn: chỉ chấp nhận true/false khi tin cậy CAO
    # và có ít nhất 1 nguồn trích dẫn cụ thể. Mọi trường hợp khác → giữ None. ─
    accept_verdict = do_tin_cay == "cao" and len(nguon_trich_dan) > 0

    if accept_verdict and ket_luan == "thuoc_du_an":
        new_thuoc_du_an: bool | None = True
    elif accept_verdict and ket_luan == "khong_thuoc_du_an":
        new_thuoc_du_an = False
    else:
        new_thuoc_du_an = None

    lp = lp.model_copy(update={
        "thuoc_du_an": new_thuoc_du_an,
        "ten_du_an": verdict.get("ten_du_an_xac_nhan") or lp.ten_du_an,
        "can_cu_phap_ly_du_an": verdict.get("can_cu") or lp.can_cu_phap_ly_du_an,
        "nguon_xac_dinh_du_an": "web_search",
        "web_verification_sources": nguon_trich_dan,
        "web_verification_summary": ghi_chu,
    })

    # ── Luôn sinh flag nhắc xác minh lại qua kênh chính thức ───────
    flags.append(FlagItem(
        flag_type="TMDV_DU_AN_XAC_MINH_WEB",
        severity="WARNING",
        description=(
            f"Đã tra cứu web bổ sung để xác định đất TMDV có thuộc dự án hay không. "
            f"Kết quả sơ bộ: {ket_luan} (độ tin cậy: {do_tin_cay}). {ghi_chu} "
            f"Đây CHỈ là kết quả tham khảo từ tìm kiếm web, KHÔNG thay thế xác nhận chính thức "
            f"— cán bộ tín dụng cần đối chiếu lại qua cổng thông tin quy hoạch địa phương/"
            f"Sở TN&MT trước khi ra quyết định."
        ),
        affected_field="land_purpose.thuoc_du_an",
    ))

    if new_thuoc_du_an is None:
        flags.append(FlagItem(
            flag_type="TMDV_CAN_XAC_MINH_THU_CONG",
            severity="WARNING",
            description=(
                "Tra cứu web không đủ bằng chứng rõ ràng/độ tin cậy cao để kết luận thửa đất "
                "TMDV có thuộc dự án hay không. Cần cán bộ tín dụng xác minh thủ công."
            ),
            affected_field="land_purpose.thuoc_du_an",
        ))

    notes.append(
        f"B2c hoàn thành: nguon_xac_dinh_du_an=web_search, thuoc_du_an={new_thuoc_du_an}, "
        f"{len(nguon_trich_dan)} nguồn tham khảo."
    )
    print("[B2c] Hoàn thành.\n")

    return state.model_copy(update={
        "land_purpose": lp,
        "flags": flags,
        "processing_notes": notes,
    })