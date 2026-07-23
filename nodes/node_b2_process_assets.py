"""
B2 (cấp hồ sơ) — chạy B2a → B2b → B3 cho TỪNG tài sản đã được con người xác
nhận ở B1c (state.asset_groups), tổng hợp kết quả vào state.asset_results.

Mỗi tài sản gọi Groq LLM (B2a, và B2b nếu cần tra cứu web TMDV) độc lập với
các tài sản khác — không có phụ thuộc dữ liệu giữa chúng. Vì đây là các lệnh
gọi I/O-bound (network) tới Groq/Tavily, xử lý TỪNG tài sản tuần tự sẽ lãng
phí thời gian chờ round-trip. Node này dùng asyncio.gather để chạy pipeline
của tất cả tài sản SONG SONG, sau đó gom kết quả lại đúng theo thứ tự
asset_groups ban đầu (kết quả/flags/quyết định routing giữ nguyên như xử lý
tuần tự — chỉ khác về thời gian chạy).

has_critical_flags cấp HỒ SƠ = True nếu B1 đã lỗi, HOẶC có bất kỳ tài sản nào
có has_critical_flags=True → toàn bộ hồ sơ sẽ được route sang human_review để
cán bộ tín dụng rà soát (dù có thể vẫn còn tài sản khác trong hồ sơ không có
vấn đề gì).
"""
from __future__ import annotations

import asyncio

from schemas import GraphState, DocumentItem, AssetGroupCandidate, AssetResult, FlagItem
from nodes.node_b2a_extract_verify import verify_asset_async
from nodes.node_b2b_websearch_tmdv import tmdv_websearch_asset_async
from nodes.node_b3_flag import flag_asset


async def _process_single_asset(group: AssetGroupCandidate, all_documents: list[DocumentItem]) -> AssetResult:
    """
    Xử lý B2a → B2b → B3 cho MỘT tài sản. Documents đưa vào LLM CHỈ gồm các
    file thuộc group.filenames + group.shared_filenames (CCCD dùng chung) —
    KHÔNG bao giờ trộn text của tài sản khác vào.
    """
    print("\n" + "#" * 60)
    print(f"# XỬ LÝ TÀI SẢN: {group.asset_id}  (GCN gợi ý: {group.so_gcn_goi_y or 'N/A'})")
    print("#" * 60)

    wanted_filenames = set(group.filenames) | set(group.shared_filenames)
    documents = [d for d in all_documents if d.filename in wanted_filenames]

    if not documents:
        return AssetResult(
            asset_id=group.asset_id,
            document_filenames=[],
            has_critical_flags=True,
            error="Không có file nào được gán cho tài sản này.",
            flags=[FlagItem(
                flag_type="OCR_THIEU_DU_LIEU", severity="ERROR",
                description=f"Nhóm tài sản '{group.asset_id}' không có file nào — không thể xử lý B2/B3.",
                affected_field="asset_groups",
            )],
        )

    owner_info, asset_info, identity_check, land_purpose, flags, warnings, notes, error = await verify_asset_async(documents)

    if error:
        return AssetResult(
            asset_id=group.asset_id,
            document_filenames=[d.filename for d in documents],
            owner_info=owner_info, asset_info=asset_info,
            identity_check=identity_check, land_purpose=land_purpose,
            flags=flags, warnings=warnings, processing_notes=notes,
            has_critical_flags=True, error=error,
        )

    land_purpose = await tmdv_websearch_asset_async(asset_info, land_purpose, notes)
    flags, warnings, notes = flag_asset(owner_info, asset_info, identity_check, land_purpose, flags, warnings, notes)

    has_critical = any(f.severity == "ERROR" for f in flags)

    return AssetResult(
        asset_id=group.asset_id,
        document_filenames=[d.filename for d in documents],
        owner_info=owner_info,
        asset_info=asset_info,
        identity_check=identity_check,
        land_purpose=land_purpose,
        flags=flags,
        warnings=warnings,
        processing_notes=notes,
        has_critical_flags=has_critical,
        error=None,
    )


async def _process_all_assets(
    groups: list[AssetGroupCandidate], documents: list[DocumentItem]
) -> list[AssetResult]:
    """Chạy pipeline của tất cả tài sản song song, giữ nguyên thứ tự groups."""
    return list(await asyncio.gather(*(_process_single_asset(g, documents) for g in groups)))


def node_b2_process_assets(state: GraphState) -> GraphState:
    print("\n" + "=" * 60)
    print(f"B2→B2b→B3 · XỬ LÝ SONG SONG {len(state.asset_groups)} TÀI SẢN ĐÃ XÁC NHẬN")
    print("=" * 60)

    notes = list(state.processing_notes)

    asset_results = asyncio.run(_process_all_assets(state.asset_groups, state.documents))

    for group, result in zip(state.asset_groups, asset_results):
        notes.append(
            f"[B2-B3] Tài sản '{group.asset_id}' xử lý xong — "
            f"{len(result.flags)} flag(s), has_critical_flags={result.has_critical_flags}."
        )

    has_critical = state.has_critical_flags or any(r.has_critical_flags for r in asset_results)

    print(f"\n[B2-B3] Hoàn thành xử lý {len(asset_results)} tài sản.")
    n_critical = sum(1 for r in asset_results if r.has_critical_flags)
    if n_critical:
        print(f"[B2-B3] ⛔ {n_critical}/{len(asset_results)} tài sản có flag ERROR — cần Human Review.")
    else:
        print("[B2-B3] ✅ Không có tài sản nào có flag ERROR.")

    return state.model_copy(update={
        "asset_results": asset_results,
        "has_critical_flags": has_critical,
        "processing_notes": notes,
    })
