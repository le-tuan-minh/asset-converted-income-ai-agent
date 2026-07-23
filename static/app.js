// ============================================================
// app.js — Frontend thuần JS (không cần build) cho luồng thẩm định TSBĐ.
// Gọi API FastAPI (api.py) qua fetch(). Toàn bộ state gom nhóm tài sản
// được giữ ở client (state.groups / state.identity / state.unassigned)
// và chỉ gửi lên server khi bấm "Xác nhận".
// ============================================================

const API_BASE = ""; // cùng origin với FastAPI (server serve luôn static/)

const state = {
  sessionId: null,
  documents: {},     // filename -> document info {filename, doc_type, ...}
  identity: [],       // danh sách filename (giấy tờ nhân thân, dùng chung)
  unassigned: [],      // danh sách filename chưa gán tài sản nào
  groups: [],          // [{asset_id, so_gcn_goi_y, dia_chi_goi_y, filenames:[], shared_filenames:[], grouping_method, grouping_confidence, grouping_reason}]
  dirty: false,        // người dùng đã chỉnh sửa gì chưa
  groupCounter: 0,
};

// ---------------- DOM refs ----------------
const $ = (id) => document.getElementById(id);
const startSection = $("startSection");
const groupingSection = $("groupingSection");
const resultSection = $("resultSection");

// ============================================================
// BR-09: nhãn nghiệp vụ cho field key kỹ thuật
// ============================================================
const FIELD_LABELS = {
  // owner_info
  ho_ten: "Họ tên",
  so_cccd: "Số CCCD",
  so_cmtnd_cu: "Số CMTND cũ",
  ngay_sinh: "Ngày sinh",
  dia_chi_thuong_tru: "Địa chỉ thường trú",
  // asset_info
  so_gcn: "Số giấy chứng nhận (GCN)",
  chu_su_dung_goc: "Chủ sử dụng gốc",
  chu_su_dung_hien_tai: "Chủ sử dụng hiện tại",
  bien_dong_lich_su: "Lịch sử biến động",
  ngay_cap_gcn: "Ngày cấp GCN",
  ngay_chuyen_nhuong: "Ngày chuyển nhượng",
  muc_dich_su_dung: "Mục đích sử dụng đất",
  ma_ky_hieu_dat: "Mã ký hiệu loại đất",
  dia_chi_tai_san: "Địa chỉ tài sản",
  dien_tich_tong: "Diện tích tổng",
  dien_tich_dat_o: "Diện tích đất ở",
  dien_tich_nha_o: "Diện tích nhà ở",
  dien_tich_nn: "Diện tích đất nông nghiệp",
  dien_tich_nts: "Diện tích đất nuôi trồng thủy sản",
  dien_tich_tmdv: "Diện tích đất TMDV",
  co_thong_tin_tang_cho: "Có thông tin tặng cho",
  thuoc_du_an: "Thuộc dự án",
  ten_du_an: "Tên dự án",
  can_cu_phap_ly_du_an: "Căn cứ pháp lý dự án",
  nguon_goc_tai_san: "Nguồn gốc tài sản",
  ben_mua_hop_dong: "Bên mua (trên hợp đồng)",
  ben_mua_so_cccd_hop_dong: "Số CCCD bên mua (trên hợp đồng)",
  ben_ban_hop_dong: "Bên bán (trên hợp đồng)",
  // identity_check
  owner_matched: "Chủ tài sản khớp",
  matched_against: "Đối chiếu với",
  mismatch_fields: "Trường không khớp",
  is_tang_cho: "Là tài sản tặng cho",
  is_thua_ke: "Là tài sản thừa kế",
  asset_formation_date: "Ngày hình thành tài sản",
  asset_formation_note: "Ghi chú hình thành tài sản",
  owner_name_similarity: "Độ tương đồng tên chủ sở hữu",
  // land_purpose
  muc_dich: "Mục đích sử dụng",
  dien_tich_dat_o_du_dieu_kien: "Diện tích đất ở đủ điều kiện",
  dien_tich_nha_o_du_dieu_kien: "Diện tích nhà ở đủ điều kiện",
  is_tmdv: "Là đất thương mại dịch vụ (TMDV)",
  nguon_xac_dinh_du_an: "Nguồn xác định dự án",
  web_verification_sources: "Nguồn xác minh web",
  web_verification_summary: "Tóm tắt xác minh web",
  warning_tmdv: "Cảnh báo TMDV",
};

// Dịch giá trị Literal sang tiếng Việt dễ hiểu
const VALUE_LABELS = {
  matched_against: { chu_hien_tai: "Chủ hiện tại", chu_goc: "Chủ gốc", khong_ro: "Không rõ" },
  nguon_xac_dinh_du_an: {
    ho_so_noi_bo: "Hồ sơ nội bộ",
    rule_based_signal: "Tín hiệu rule-based",
    web_search: "Tra cứu web",
    chua_xac_dinh: "Chưa xác định",
  },
};

// ============================================================
// BR-08: chú giải mã ký hiệu đất (Thông tư 08/2024/TT-BTNMT) + đơn vị m²
// ============================================================
const LAND_CODE_LABELS = {
  ONT: "Đất ở tại nông thôn",
  ODT: "Đất ở tại đô thị",
  CLN: "Đất trồng cây lâu năm",
  LUC: "Đất chuyên trồng lúa",
  LUK: "Đất trồng lúa còn lại",
  NKH: "Đất nông nghiệp khác",
  NTS: "Đất nuôi trồng thủy sản",
  TMD: "Đất thương mại, dịch vụ",
  SKC: "Đất cơ sở sản xuất, kinh doanh phi nông nghiệp",
};

const AREA_FIELDS = new Set([
  "dien_tich_tong", "dien_tich_dat_o", "dien_tich_nha_o",
  "dien_tich_nn", "dien_tich_nts", "dien_tich_tmdv",
  "dien_tich_dat_o_du_dieu_kien", "dien_tich_nha_o_du_dieu_kien",
]);

function fmtArea(v) {
  if (v === "" || v === null || v === undefined) return "—";
  return `${v} m²`;
}

function landCodeTooltip(code) {
  const name = LAND_CODE_LABELS[String(code || "").trim().toUpperCase()];
  if (!code) return "—";
  if (!name) return escapeHtml(code);
  return `<span class="info-tooltip" title="${escapeHtml(name)}">${escapeHtml(code)} — ${escapeHtml(name)}</span>`;
}

// ============================================================
// BR-06: hiển thị đầy đủ nội dung (không rút gọn/ẩn), chống tràn chữ
// đã xử lý bằng CSS word-break ở index.html.
// ============================================================
function renderLongText(text) {
  return escapeHtml(text);
}

// ============================================================
// BR-07: bien_dong_lich_su -> timeline dễ đọc
// ============================================================
function parseVNDate(s) {
  if (!s || typeof s !== "string") return null;
  const m = s.match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})$/);
  if (!m) return null;
  return new Date(Number(m[3]), Number(m[2]) - 1, Number(m[1])).getTime();
}

function renderTimeline(items) {
  if (!Array.isArray(items) || items.length === 0) return "—";
  const lis = items
    .slice()
    .sort((a, b) => {
      const da = parseVNDate(a.ngay), db = parseVNDate(b.ngay);
      if (da && db) return da - db;
      return 0;
    })
    .map((it) => `
      <li>
        <b>${escapeHtml(it.ngay || "—")}</b>: ${escapeHtml(it.noi_dung || "")}
        ${it.chu_moi ? ` <span class="text-dim">(chủ mới: ${escapeHtml(it.chu_moi)})</span>` : ""}
      </li>
    `)
    .join("");
  return `<ul class="timeline-list">${lis}</ul>`;
}

// ============================================================
// BR-14: web_verification_sources -> link bấm được
// ============================================================
function shortDomain(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch (e) {
    return url;
  }
}

function renderWebSources(urls) {
  if (!Array.isArray(urls) || urls.length === 0) return "—";
  return urls
    .map((u) => `
      <a class="web-source-link" href="${escapeHtml(u)}" target="_blank" rel="noopener noreferrer">
        🔗 ${escapeHtml(u)} <span class="web-source-domain">(${escapeHtml(shortDomain(u))})</span>
      </a>
    `)
    .join("");
}

// ---------------- Health check ----------------
(async function checkHealth() {
  const badge = $("apiStatus");
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    const data = await res.json();
    if (data.ok && data.groq_api_key_set) {
      badge.textContent = "✅ Backend sẵn sàng";
      badge.className = "badge badge-ok";
    } else if (data.ok) {
      badge.textContent = "⚠️ Thiếu GROQ_API_KEY";
      badge.className = "badge badge-error";
    }
  } catch (e) {
    badge.textContent = "❌ Không kết nối được backend";
    badge.className = "badge badge-error";
  }
})();

// ============================================================
// BR-01: thanh tiến trình (progress stepper) — CHỈ hiển thị các bước
// hệ thống THỰC SỰ đi qua trong luồng hiện tại (B1a → B1b → B1c → B2-B3),
// không hiển thị các bước B4+ (kiểm tra CIC, định giá...) vì backend
// hiện tại (graph.py) chưa triển khai các bước đó.
// ============================================================
function setStep(step) {
  const stepper = $("progressStepper");
  if (!stepper) return;
  stepper.querySelectorAll(".step-item").forEach((el) => {
    const n = Number(el.dataset.step);
    el.classList.toggle("done", n < step);
    el.classList.toggle("active", n === step);
  });
}
setStep(1);

// ============================================================
// BƯỚC 1: bắt đầu xử lý
// ============================================================
$("startBtn").addEventListener("click", async () => {
  const folder_path = $("folderPath").value.trim();
  const output_path = $("outputPath").value.trim() || "output/result.json";
  const msg = $("startMsg");
  msg.textContent = "";
  msg.className = "msg";

  if (!folder_path) {
    msg.textContent = "Vui lòng nhập đường dẫn folder.";
    msg.className = "msg error";
    return;
  }

  $("startBtn").disabled = true;
  msg.textContent = "⏳ Đang xử lý (OCR + phân loại + AI gom nhóm)...";
  setStep(1);

  try {
    const res = await fetch(`${API_BASE}/api/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folder_path, output_path }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Lỗi không xác định.");

    handleServerResponse(data);
  } catch (e) {
    msg.textContent = `❌ ${e.message}`;
    msg.className = "msg error";
  } finally {
    $("startBtn").disabled = false;
  }
});

function handleServerResponse(data) {
  state.sessionId = data.session_id;

  if (data.status === "awaiting_confirmation") {
    $("startMsg").textContent = `⏸️ ${data.message}`;
    $("startMsg").className = "msg";
    loadGroupingUI(data);
    groupingSection.classList.remove("hidden");
    resultSection.classList.add("hidden");
    setStep(3); // OCR + gom nhóm AI xong, đang chờ CBTD xác nhận
  } else if (data.status === "done") {
    $("startMsg").textContent = `✅ ${data.message}`;
    $("startMsg").className = "msg ok";
    groupingSection.classList.add("hidden");
    renderResult(data);
    resultSection.classList.remove("hidden");
    setStep(5); // Đã có kết quả thẩm định — bước cuối trong luồng hiện tại
  }
}

// ============================================================
// BƯỚC 2: gom nhóm tài sản — dựng state kéo-thả từ đề xuất AI
// ============================================================
function loadGroupingUI(data) {
  state.documents = {};
  (data.documents || []).forEach((d) => { state.documents[d.filename] = d; });

  const sharedSet = new Set();
  (data.asset_groups || []).forEach((g) => (g.shared_filenames || []).forEach((f) => sharedSet.add(f)));
  // fallback: nếu AI không set shared_filenames, dùng doc_type CCCD
  Object.values(state.documents).forEach((d) => { if (d.is_identity_doc) sharedSet.add(d.filename); });
  state.identity = Array.from(sharedSet);

  state.groups = (data.asset_groups || []).map((g) => ({
    asset_id: g.asset_id,
    so_gcn_goi_y: g.so_gcn_goi_y || "",
    dia_chi_goi_y: g.dia_chi_goi_y || "",
    filenames: (g.filenames || []).filter((f) => !state.identity.includes(f)),
    shared_filenames: state.identity.slice(),
    grouping_method: g.grouping_method || "llm",
    grouping_confidence: g.grouping_confidence ?? 0.5,
    grouping_reason: g.grouping_reason || "",
  }));

  const assignedElsewhere = new Set(state.identity);
  state.groups.forEach((g) => g.filenames.forEach((f) => assignedElsewhere.add(f)));
  state.unassigned = Object.keys(state.documents).filter((f) => !assignedElsewhere.has(f));

  state.groupCounter = state.groups.length;
  state.dirty = false;

  renderAll();
}

function renderAll() {
  renderChipList("identityChips", state.identity, "identity");
  renderChipList("unassignedChips", state.unassigned, "unassigned");
  renderGroups();
}

function docBadge(filename) {
  const d = state.documents[filename];
  return d ? (d.doc_type || "?") : "?";
}

function makeChip(filename, zone) {
  const chip = document.createElement("div");
  chip.className = "file-chip" + (zone === "identity" ? " identity-chip" : "");
  chip.draggable = true;
  chip.dataset.filename = filename;
  chip.innerHTML = `
    <span class="view-eye" title="Xem file">👁</span>
    <span class="chip-name">${escapeHtml(filename)}</span>
    <span class="doc-type-tag">${escapeHtml(docBadge(filename))}</span>
  `;
  chip.addEventListener("dragstart", (e) => {
    e.dataTransfer.setData("text/plain", filename);
    e.dataTransfer.effectAllowed = "move";
  });
  chip.querySelector(".view-eye").addEventListener("click", (e) => {
    e.stopPropagation();
    openFileModal(filename);
  });
  return chip;
}

function renderChipList(containerId, filenames, zone) {
  const el = $(containerId);
  el.innerHTML = "";
  filenames.forEach((f) => el.appendChild(makeChip(f, zone)));
}

function confidenceClass(conf) {
  if (conf >= 0.75) return "conf-high";
  if (conf >= 0.4) return "conf-mid";
  return "conf-low";
}

// BR-04: gợi ý tên nhóm tài sản theo địa chỉ/dự án thực tế
function suggestAssetName(group, idx) {
  if (!group.dia_chi_goi_y) return group.asset_id;
  const shortAddr = group.dia_chi_goi_y.split(/[-,]/)[0].trim();
  return `TS${idx + 1} – ${shortAddr}`;
}

// BR-05: hộp thoại xác nhận dùng chung (xoá nhóm...)
function showConfirmDialog(message, onConfirm) {
  const overlay = document.createElement("div");
  overlay.className = "confirm-overlay";
  overlay.innerHTML = `
    <div class="confirm-box">
      <p>${escapeHtml(message)}</p>
      <div class="confirm-box-actions">
        <button type="button" class="btn-cancel">Huỷ</button>
        <button type="button" class="btn-danger">Xoá</button>
      </div>
    </div>
  `;
  document.body.appendChild(overlay);
  overlay.querySelector(".btn-cancel").addEventListener("click", () => overlay.remove());
  overlay.querySelector(".btn-danger").addEventListener("click", () => {
    overlay.remove();
    onConfirm();
  });
}

function renderGroups() {
  const container = $("assetGroups");
  container.innerHTML = "";

  state.groups.forEach((group, idx) => {
    const card = document.createElement("div");
    card.className = "asset-card dropzone";
    card.dataset.zone = `group:${group.asset_id}`;

    const confPct = Math.round((group.grouping_confidence || 0) * 100);
    // BR-03: liệt kê rõ căn cứ đối chiếu thay vì chỉ 1 câu mô tả chung
    const criteriaChips = [
      group.so_gcn_goi_y ? "Số GCN" : null,
      group.dia_chi_goi_y ? "Địa chỉ" : null,
    ].filter(Boolean);
    // BR-04: gợi ý tên nhóm theo địa chỉ thực tế thay vì asset_1/asset_2
    const suggestedName = suggestAssetName(group, idx);

    card.innerHTML = `
      <div class="asset-card-header">
        <input type="text" class="asset-name-input" value="${escapeHtml(group.asset_id)}" />
        <span class="confidence-chip ${confidenceClass(group.grouping_confidence || 0)}">${confPct}%</span>
      </div>
      <div class="asset-meta">
        <div><b>Số GCN gợi ý:</b> ${escapeHtml(group.so_gcn_goi_y || "N/A")}</div>
        <div><b>Địa chỉ gợi ý:</b> ${escapeHtml(group.dia_chi_goi_y || "N/A")}</div>
        <div><b>Lý do AI gom:</b> ${escapeHtml(group.grouping_reason || "N/A")}
          ${criteriaChips.length ? `<span class="hint"> (đối chiếu theo: ${criteriaChips.join(", ")})</span>` : ""}
        </div>
      </div>
      ${suggestedName !== group.asset_id ? `
        <button type="button" class="suggest-name-btn" data-suggest="${escapeHtml(suggestedName)}">
          ✏️ Dùng tên gợi ý: "${escapeHtml(suggestedName)}"
        </button>
      ` : ""}
      <div class="chip-list" data-filelist></div>
      <button class="remove-group-btn" data-remove>🗑 Xóa nhóm này</button>
    `;

    const chipList = card.querySelector("[data-filelist]");
    group.filenames.forEach((f) => chipList.appendChild(makeChip(f, "group")));

    const nameInput = card.querySelector(".asset-name-input");
    nameInput.addEventListener("change", (e) => {
      group.asset_id = e.target.value.trim() || group.asset_id;
      state.dirty = true;
    });

    const suggestBtn = card.querySelector(".suggest-name-btn");
    if (suggestBtn) {
      suggestBtn.addEventListener("click", () => {
        nameInput.value = suggestBtn.dataset.suggest;
        group.asset_id = suggestBtn.dataset.suggest;
        state.dirty = true;
        renderGroups();
      });
    }

    // BR-05: xác nhận trước khi xoá nhóm tài sản (thay vì xoá ngay)
    card.querySelector("[data-remove]").addEventListener("click", () => {
      showConfirmDialog(
        `Bạn có chắc muốn xoá nhóm "${group.asset_id}"? ${group.filenames.length} file sẽ được đưa về danh sách "chưa gán tài sản".`,
        () => {
          state.unassigned.push(...group.filenames);
          state.groups.splice(idx, 1);
          state.dirty = true;
          renderAll();
        }
      );
    });

    attachDropzone(card);
    container.appendChild(card);
  });

  attachDropzone($("identityBox"));
  attachDropzone($("unassignedBox"));
}

// ---------------- Drag & drop wiring ----------------
function attachDropzone(el) {
  el.addEventListener("dragover", (e) => {
    e.preventDefault();
    el.classList.add("drag-over");
  });
  el.addEventListener("dragleave", () => el.classList.remove("drag-over"));
  el.addEventListener("drop", (e) => {
    e.preventDefault();
    el.classList.remove("drag-over");
    const filename = e.dataTransfer.getData("text/plain");
    if (!filename) return;
    moveFileToZone(filename, el.dataset.zone);
  });
}

function removeFromAllZones(filename) {
  state.identity = state.identity.filter((f) => f !== filename);
  state.unassigned = state.unassigned.filter((f) => f !== filename);
  state.groups.forEach((g) => { g.filenames = g.filenames.filter((f) => f !== filename); });
}

function moveFileToZone(filename, zoneKey) {
  removeFromAllZones(filename);

  if (zoneKey === "identity") {
    state.identity.push(filename);
    state.groups.forEach((g) => { g.shared_filenames = state.identity.slice(); });
  } else if (zoneKey === "unassigned") {
    state.unassigned.push(filename);
  } else if (zoneKey.startsWith("group:")) {
    const assetId = zoneKey.slice("group:".length);
    const group = state.groups.find((g) => g.asset_id === assetId);
    if (group) group.filenames.push(filename);
    else state.unassigned.push(filename);
  }
  state.dirty = true;
  renderAll();
}

// ---------------- Thêm nhóm tài sản mới ----------------
$("addGroupBtn").addEventListener("click", () => {
  state.groupCounter += 1;
  state.groups.push({
    asset_id: `asset_${state.groupCounter}_moi`,
    so_gcn_goi_y: "",
    dia_chi_goi_y: "",
    filenames: [],
    shared_filenames: state.identity.slice(),
    grouping_method: "human_edited",
    grouping_confidence: 1.0,
    grouping_reason: "Nhóm tạo thủ công trên UI.",
  });
  state.dirty = true;
  renderAll();
});

// ============================================================
// Xác nhận / gửi chỉnh sửa
// ============================================================
async function submitConfirmation(action) {
  const msg = $("confirmMsg");
  msg.textContent = "⏳ Đang gửi xác nhận và xử lý B2-B3 cho từng tài sản...";
  msg.className = "msg";
  setStep(4);

  const payload = {
    session_id: state.sessionId,
    action,
    note: $("editNote").value.trim(),
    asset_groups: state.groups.map((g) => ({
      asset_id: g.asset_id,
      so_gcn_goi_y: g.so_gcn_goi_y,
      dia_chi_goi_y: g.dia_chi_goi_y,
      filenames: g.filenames,
      shared_filenames: state.identity,
      grouping_method: action === "edit" ? "human_edited" : g.grouping_method,
      grouping_confidence: g.grouping_confidence,
      grouping_reason: g.grouping_reason,
    })),
  };

  try {
    const res = await fetch(`${API_BASE}/api/confirm`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Lỗi không xác định.");
    msg.textContent = "";
    handleServerResponse(data);
  } catch (e) {
    msg.textContent = `❌ ${e.message}`;
    msg.className = "msg error";
  }
}

// BR-02: cảnh báo + chặn nếu còn file chưa gán tài sản; cho phép CBTD
// xác nhận bỏ qua có ghi log lý do (yêu cầu nhập ghi chú bắt buộc).
function confirmProceedDespiteUnassigned() {
  if (state.unassigned.length === 0) return true;

  const noteEl = $("editNote");
  const reason = window.prompt(
    `⚠️ Còn ${state.unassigned.length} file chưa được gán vào nhóm tài sản nào:\n` +
    state.unassigned.join(", ") +
    `\n\nNếu vẫn muốn tiếp tục xử lý mà bỏ qua các file này, vui lòng nhập lý do ` +
    `(sẽ được ghi lại trong ghi chú xử lý). Để trống hoặc bấm Huỷ để quay lại gán file.`
  );
  if (reason === null || reason.trim() === "") return false;

  console.warn("[BR-02] CBTD xác nhận bỏ qua file chưa gán:", state.unassigned, "Lý do:", reason);
  if (noteEl) {
    noteEl.value = (noteEl.value ? noteEl.value + " | " : "") +
      `[Bỏ qua ${state.unassigned.length} file chưa gán] ${reason.trim()}`;
  }
  return true;
}

$("confirmAiBtn").addEventListener("click", () => {
  if (!confirmProceedDespiteUnassigned()) return;
  submitConfirmation(state.dirty ? "edit" : "confirm");
});
$("confirmEditBtn").addEventListener("click", () => {
  if (!confirmProceedDespiteUnassigned()) return;
  submitConfirmation("edit");
});

// ============================================================
// Modal xem file (phóng to, scroll, zoom)
// ============================================================
let zoomLevel = 1;
const modal = $("fileModal");
const modalViewer = $("modalViewer");

function openFileModal(filename) {
  if (!state.sessionId) return;
  zoomLevel = 1;
  $("zoomLevel").textContent = "100%";
  $("modalTitle").textContent = filename;

  const url = `${API_BASE}/api/session/${state.sessionId}/file?filename=${encodeURIComponent(filename)}`;
  const lower = filename.toLowerCase();
  modalViewer.innerHTML = "";

  if (lower.endsWith(".pdf")) {
    const iframe = document.createElement("iframe");
    iframe.src = url;
    modalViewer.appendChild(iframe);
  } else {
    const img = document.createElement("img");
    img.src = url;
    img.style.transform = "scale(1)";
    modalViewer.appendChild(img);
  }
  modal.classList.remove("hidden");
}

function closeModal() {
  modal.classList.add("hidden");
  modalViewer.innerHTML = "";
}

$("modalCloseBtn").addEventListener("click", closeModal);
$("modalBackdrop").addEventListener("click", closeModal);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });

function applyZoom() {
  const img = modalViewer.querySelector("img");
  if (img) img.style.transform = `scale(${zoomLevel})`;
  $("zoomLevel").textContent = `${Math.round(zoomLevel * 100)}%`;
}
$("zoomInBtn").addEventListener("click", () => { zoomLevel = Math.min(zoomLevel + 0.25, 4); applyZoom(); });
$("zoomOutBtn").addEventListener("click", () => { zoomLevel = Math.max(zoomLevel - 0.25, 0.25); applyZoom(); });
$("zoomResetBtn").addEventListener("click", () => { zoomLevel = 1; applyZoom(); });

// ============================================================
// BƯỚC 3: render kết quả thẩm định
// ============================================================
// BR-13: xác định trạng thái Xanh/Vàng/Đỏ từ severity thực tế của flags,
// không chỉ dựa vào has_critical_flags (vốn dễ hiểu nhầm khi vẫn còn WARNING).
function computeAssetStatus(r) {
  const flags = r.flags || [];
  const nError = flags.filter((f) => f.severity === "ERROR").length;
  const nWarning = flags.filter((f) => f.severity === "WARNING").length;
  let status = "ok";
  if (nError > 0 || r.error) status = "error";
  else if (nWarning > 0) status = "warning";
  return { status, nError, nWarning };
}

function statusDotHtml(status) {
  const cls = status === "error" ? "status-error" : status === "warning" ? "status-warning" : "status-ok";
  const label = status === "error" ? "Có lỗi nghiêm trọng" : status === "warning" ? "Có cảnh báo cần xem xét" : "Không cảnh báo";
  return `<span class="status-dot ${cls}" title="${escapeHtml(label)}"></span>`;
}

function renderResult(data) {
  const summary = $("resultSummary");
  const results = data.asset_results || [];
  const nAssets = results.length;
  const nErrorAssets = results.filter((r) => computeAssetStatus(r).status === "error").length;
  const nWarningAssets = results.filter((r) => computeAssetStatus(r).status === "warning").length;

  summary.innerHTML = `
    <span class="stat-pill">📁 ${escapeHtml(data.session_id)}</span>
    <span class="stat-pill">🏠 ${nAssets} tài sản</span>
    <span class="stat-pill">${nErrorAssets ? "🔴" : "🟢"} ${nErrorAssets} tài sản có lỗi nghiêm trọng</span>
    <span class="stat-pill">🟡 ${nWarningAssets} tài sản có cảnh báo</span>
  `;

  const assetsEl = $("resultAssets");
  assetsEl.innerHTML = "";

  results.forEach((r) => {
    const { status, nError, nWarning } = computeAssetStatus(r);

    const card = document.createElement("div");
    card.className = "asset-result-card" + (r.has_critical_flags ? " has-error" : "");
    // BR-20: gắn sẵn dữ liệu để lọc/tìm kiếm
    card.dataset.status = status;
    const ownerName = (r.owner_info && r.owner_info.ho_ten) || "";
    card.dataset.searchText = `${r.asset_id || ""} ${ownerName}`.toLowerCase();

    const infoBlocks = ["owner_info", "asset_info", "identity_check", "land_purpose"]
      .filter((k) => r[k] && typeof r[k] === "object")
      .map((k) => renderInfoBlock(k, r[k]))
      .join("");

    // Badge số lượng theo mức ngay trên thẻ tài sản
    const severityChips = `
      ${nError ? `<span class="severity-chip sev-error">${nError} nghiêm trọng</span>` : ""}
      ${nWarning ? `<span class="severity-chip sev-warning">${nWarning} cảnh báo</span>` : ""}
    `;

    const flagsHtml = (r.flags || [])
      .map((f) => `
        <div class="flag-item ${f.severity}">
          <span class="flag-type">${escapeHtml(f.flag_type)}</span>
          <span class="flag-sev">[${f.severity}]</span>
          <div>${escapeHtml(f.description || "")}</div>
        </div>
      `)
      .join("") || "<div class='hint'>Không có flag nào.</div>";

    // BR-11: chỉ hiện phần "Cảnh báo khác" KHÔNG trùng nội dung với flags ở trên
    // (tránh lặp lại cùng 1 cảnh báo ở 2 chỗ gây hiểu nhầm là 2 vấn đề khác nhau)
    const flagDescriptions = new Set((r.flags || []).map((f) => (f.description || "").trim()));
    const uniqueWarnings = (r.warnings || []).filter((w) => !flagDescriptions.has((w || "").trim()));
    const warningsHtml = uniqueWarnings.length
      ? `<ul>${uniqueWarnings.map((w) => `<li>${escapeHtml(w)}</li>`).join("")}</ul>`
      : "";

    card.innerHTML = `
      <h3>${statusDotHtml(status)} ${escapeHtml(r.asset_id)} ${severityChips}
        ${r.error ? `<span class="badge badge-error">${escapeHtml(r.error)}</span>` : ""}
      </h3>
      <div class="info-grid">${infoBlocks}</div>
      <h4>🚩 Flags</h4>
      <div class="flag-list">${flagsHtml}</div>
      ${warningsHtml ? `<h4>⚠️ Cảnh báo khác</h4>${warningsHtml}` : ""}
    `;
    assetsEl.appendChild(card);
  });

  const dlLink = $("downloadResultLink");
  dlLink.href = `${API_BASE}/api/session/${state.sessionId}/result-file`;

  // BR-20: áp dụng lại bộ lọc hiện có (nếu người dùng đã gõ trước đó)
  applyResultFilter();
}

// ============================================================
// BR-20: tìm kiếm / lọc tài sản theo trạng thái, tên chủ, mã tài sản
// ============================================================
function applyResultFilter() {
  const searchEl = $("resultSearchInput");
  const statusEl = $("resultStatusFilter");
  const countEl = $("resultFilterCount");
  if (!searchEl || !statusEl) return;

  const q = searchEl.value.trim().toLowerCase();
  const statusFilter = statusEl.value;
  const cards = document.querySelectorAll("#resultAssets .asset-result-card");
  let visible = 0;

  cards.forEach((card) => {
    const text = card.dataset.searchText || "";
    const status = card.dataset.status || "ok";
    const matchesText = !q || text.includes(q);
    const matchesStatus = statusFilter === "all" || status === statusFilter;
    const show = matchesText && matchesStatus;
    card.classList.toggle("filtered-out", !show);
    if (show) visible++;
  });

  if (countEl) countEl.textContent = `Hiển thị ${visible}/${cards.length} tài sản`;
}

if ($("resultSearchInput")) $("resultSearchInput").addEventListener("input", applyResultFilter);
if ($("resultStatusFilter")) $("resultStatusFilter").addEventListener("change", applyResultFilter);

function renderInfoBlock(key, obj) {
  const titles = {
    owner_info: "👤 Chủ tài sản",
    asset_info: "🏠 Thông tin tài sản",
    identity_check: "🪪 Kiểm tra nhân thân",
    land_purpose: "📄 Mục đích sử dụng đất",
  };

  const rows = Object.entries(obj)
    .filter(([k]) => k !== "raw_text")
    .map(([k, v]) => {
      // BR-09: nhãn nghiệp vụ tiếng Việt thay cho field key kỹ thuật
      const label = FIELD_LABELS[k] || k;
      let valueHtml;

      if (k === "bien_dong_lich_su") {
        // BR-07: JSON biến động -> timeline dễ đọc
        valueHtml = renderTimeline(v);
      } else if (k === "web_verification_sources") {
        // BR-14: mảng URL -> link bấm được
        valueHtml = renderWebSources(v);
      } else if (k === "ma_ky_hieu_dat") {
        // BR-08: chú giải mã ký hiệu đất
        valueHtml = landCodeTooltip(v);
      } else if (AREA_FIELDS.has(k)) {
        // BR-08: kèm đơn vị m²
        valueHtml = escapeHtml(fmtArea(v));
      } else if (VALUE_LABELS[k] && VALUE_LABELS[k][v] !== undefined) {
        valueHtml = escapeHtml(VALUE_LABELS[k][v]);
      } else if (typeof v === "boolean") {
        valueHtml = escapeHtml(v ? "Có" : "Không");
      } else {
        // BR-06: rút gọn chữ dài + nút "xem đầy đủ"
        const str = formatValue(v);
        valueHtml = typeof str === "string" ? renderLongText(str) : escapeHtml(String(str));
      }

      return `<tr><td>${escapeHtml(label)}</td><td>${valueHtml}</td></tr>`;
    })
    .join("");

  return `
    <div class="info-block">
      <h4>${titles[key] || key}</h4>
      <table>${rows}</table>
    </div>
  `;
}

function formatValue(v) {
  if (v === null || v === undefined || v === "") return "—";
  if (typeof v === "boolean") return v ? "Có" : "Không";
  if (Array.isArray(v)) return v.length ? v.join(", ") : "—";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

function escapeHtml(str) {
  return String(str ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}