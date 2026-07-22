"""B2c — Web search bổ sung cho đất TMDV (Tavily, nếu có TAVILY_API_KEY)."""
from __future__ import annotations
import json
import os

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from schemas import AssetInfo, LandPurposeResult
from utils.llm_config import get_llm
from utils.parsing_utils import parse_json_safe

MAX_TOOL_CALLS = 4

AGENT_SYSTEM_PROMPT = """Bạn là trợ lý tra cứu thông tin quy hoạch/dự án bất động sản
tại Việt Nam. Bạn có công cụ tavily_search để tìm kiếm trên web. Nhiệm vụ: xác
định xem thửa đất/dự án được mô tả có thuộc một dự án đầu tư đã được phê duyệt
hay không. Sau khi tra cứu, trả lời CHỈ bằng JSON:
{"thuoc_du_an": true/false/null, "ten_du_an": "", "can_cu_phap_ly_du_an": "", "tom_tat": ""}
Nếu không tìm thấy căn cứ đủ tin cậy, để thuoc_du_an=null."""

FINAL_VERDICT_INSTRUCTION = (
    "Dựa trên các kết quả tra cứu ở trên, hãy trả lời CHỈ bằng JSON đúng định dạng đã nêu, "
    "không thêm chữ nào khác, không dùng markdown."
)


def _build_tavily_tool():
    from langchain_core.tools import tool

    api_key = os.getenv("TAVILY_API_KEY")

    @tool
    def tavily_search(query: str) -> str:
        """Tìm kiếm thông tin dự án bất động sản/quy hoạch trên web qua Tavily."""
        if not api_key:
            return json.dumps([])
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=api_key)
            res = client.search(query, max_results=5)
            results = [{"url": r.get("url", ""), "content": r.get("content", "")[:500]} for r in res.get("results", [])]
            return json.dumps(results, ensure_ascii=False)
        except Exception as exc:
            return json.dumps([{"error": str(exc)}])

    return tavily_search


def _build_task_message(asset_info: AssetInfo, land_purpose: LandPurposeResult) -> str:
    return (
        f"Thửa đất địa chỉ: {asset_info.dia_chi_tai_san or '(không rõ)'}\n"
        f"Mục đích sử dụng: {land_purpose.muc_dich or asset_info.muc_dich_su_dung} "
        f"(mã {land_purpose.ma_ky_hieu_dat or asset_info.ma_ky_hieu_dat})\n"
        f"Tên dự án (nếu hồ sơ có nêu): {asset_info.ten_du_an or '(không có)'}\n"
        "Hãy tra cứu xem thửa đất/khu vực này có thuộc 1 dự án đầu tư đã được phê duyệt hay không."
    )


def _run_web_agent(asset_info: AssetInfo, land_purpose: LandPurposeResult) -> tuple[dict, list[str]]:
    tavily_tool = _build_tavily_tool()
    llm = get_llm()
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

    messages.append(HumanMessage(content=FINAL_VERDICT_INSTRUCTION))
    final_msg = llm.invoke(messages)
    verdict = parse_json_safe(final_msg.content)
    urls_unique = list(dict.fromkeys(u for u in urls_seen if u))
    return verdict, urls_unique


def tmdv_websearch_asset(asset_info: AssetInfo, land_purpose: LandPurposeResult, notes: list[str]) -> LandPurposeResult:
    """B2c cho 1 tài sản — chỉ chạy nếu is_tmdv=True và thuoc_du_an chưa xác định."""
    if not (land_purpose.is_tmdv and land_purpose.thuoc_du_an is None):
        return land_purpose

    if not os.getenv("TAVILY_API_KEY"):
        notes.append("[B2c] Bỏ qua tra cứu web: chưa cấu hình TAVILY_API_KEY.")
        return land_purpose

    print("[B2c] Đất TMDV chưa xác định thuộc dự án — tra cứu web bổ sung qua Tavily...")
    try:
        verdict, urls = _run_web_agent(asset_info, land_purpose)
    except Exception as exc:
        notes.append(f"[B2c] Lỗi tra cứu web: {exc}")
        return land_purpose

    thuoc_du_an = verdict.get("thuoc_du_an", None)
    if thuoc_du_an is not None:
        land_purpose = land_purpose.model_copy(update={
            "thuoc_du_an": bool(thuoc_du_an),
            "ten_du_an": verdict.get("ten_du_an", "") or land_purpose.ten_du_an,
            "can_cu_phap_ly_du_an": verdict.get("can_cu_phap_ly_du_an", "") or land_purpose.can_cu_phap_ly_du_an,
            "nguon_xac_dinh_du_an": "web_search",
            "web_verification_sources": urls,
            "web_verification_summary": verdict.get("tom_tat", ""),
        })
        notes.append(f"[B2c] Web search kết luận thuoc_du_an={thuoc_du_an}.")
    else:
        notes.append("[B2c] Web search không tìm được căn cứ đủ tin cậy — giữ thuoc_du_an=null.")
    return land_purpose
