"""Entry point: xử lý 1 tài sản (B2 → B2c → B3) → AssetResult."""
from __future__ import annotations

from schemas import DocumentItem, AssetGroupCandidate, AssetResult, FlagItem
from .extract import verify_asset
from .websearch import tmdv_websearch_asset
from .flags import flag_asset


def process_single_asset(group: AssetGroupCandidate, all_documents: list[DocumentItem]) -> AssetResult:
    """
    Xử lý B2 → B2c → B3 cho MỘT tài sản. Documents đưa vào LLM CHỈ gồm các
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

    owner_info, asset_info, identity_check, land_purpose, flags, warnings, notes, error = verify_asset(documents)

    if error:
        return AssetResult(
            asset_id=group.asset_id,
            document_filenames=[d.filename for d in documents],
            owner_info=owner_info, asset_info=asset_info,
            identity_check=identity_check, land_purpose=land_purpose,
            flags=flags, warnings=warnings, processing_notes=notes,
            has_critical_flags=True, error=error,
        )

    land_purpose = tmdv_websearch_asset(asset_info, land_purpose, notes)
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
