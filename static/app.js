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
  } else if (data.status === "done") {
    $("startMsg").textContent = `✅ ${data.message}`;
    $("startMsg").className = "msg ok";
    groupingSection.classList.add("hidden");
    renderResult(data);
    resultSection.classList.remove("hidden");
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

function renderGroups() {
  const container = $("assetGroups");
  container.innerHTML = "";

  state.groups.forEach((group, idx) => {
    const card = document.createElement("div");
    card.className = "asset-card dropzone";
    card.dataset.zone = `group:${group.asset_id}`;

    const confPct = Math.round((group.grouping_confidence || 0) * 100);
    card.innerHTML = `
      <div class="asset-card-header">
        <input type="text" class="asset-name-input" value="${escapeHtml(group.asset_id)}" />
        <span class="confidence-chip ${confidenceClass(group.grouping_confidence || 0)}">${confPct}%</span>
      </div>
      <div class="asset-meta">
        <div><b>Số GCN gợi ý:</b> ${escapeHtml(group.so_gcn_goi_y || "N/A")}</div>
        <div><b>Địa chỉ gợi ý:</b> ${escapeHtml(group.dia_chi_goi_y || "N/A")}</div>
        <div><b>Lý do AI gom:</b> ${escapeHtml(group.grouping_reason || "N/A")}</div>
      </div>
      <div class="chip-list" data-filelist></div>
      <button class="remove-group-btn" data-remove>🗑 Xóa nhóm này</button>
    `;

    const chipList = card.querySelector("[data-filelist]");
    group.filenames.forEach((f) => chipList.appendChild(makeChip(f, "group")));

    card.querySelector(".asset-name-input").addEventListener("change", (e) => {
      group.asset_id = e.target.value.trim() || group.asset_id;
      state.dirty = true;
    });

    card.querySelector("[data-remove]").addEventListener("click", () => {
      state.unassigned.push(...group.filenames);
      state.groups.splice(idx, 1);
      state.dirty = true;
      renderAll();
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

$("confirmAiBtn").addEventListener("click", () => {
  submitConfirmation(state.dirty ? "edit" : "confirm");
});
$("confirmEditBtn").addEventListener("click", () => submitConfirmation("edit"));

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
function renderResult(data) {
  const summary = $("resultSummary");
  const nAssets = (data.asset_results || []).length;
  const nErrors = (data.asset_results || []).filter((r) => r.has_critical_flags).length;
  summary.innerHTML = `
    <span class="stat-pill">📁 ${escapeHtml(data.session_id)}</span>
    <span class="stat-pill">🏠 ${nAssets} tài sản</span>
    <span class="stat-pill ${nErrors ? "" : ""}">${nErrors ? "🔴" : "🟢"} ${nErrors} tài sản có lỗi nghiêm trọng</span>
  `;

  const assetsEl = $("resultAssets");
  assetsEl.innerHTML = "";

  (data.asset_results || []).forEach((r) => {
    const card = document.createElement("div");
    card.className = "asset-result-card" + (r.has_critical_flags ? " has-error" : "");

    const infoBlocks = ["owner_info", "asset_info", "identity_check", "land_purpose"]
      .filter((k) => r[k] && typeof r[k] === "object")
      .map((k) => renderInfoBlock(k, r[k]))
      .join("");

    const flagsHtml = (r.flags || [])
      .map((f) => `
        <div class="flag-item ${f.severity}">
          <span class="flag-type">${escapeHtml(f.flag_type)}</span>
          <span class="flag-sev">[${f.severity}]</span>
          <div>${escapeHtml(f.description || "")}</div>
        </div>
      `)
      .join("") || "<div class='hint'>Không có flag nào.</div>";

    const warningsHtml = (r.warnings || []).length
      ? `<ul>${r.warnings.map((w) => `<li>${escapeHtml(w)}</li>`).join("")}</ul>`
      : "";

    card.innerHTML = `
      <h3>${r.has_critical_flags ? "🔴" : "🟢"} ${escapeHtml(r.asset_id)}
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
}

function renderInfoBlock(key, obj) {
  const titles = {
    owner_info: "👤 Chủ tài sản",
    asset_info: "🏠 Thông tin tài sản",
    identity_check: "🪪 Kiểm tra nhân thân",
    land_purpose: "📄 Mục đích sử dụng đất",
  };
  const rows = Object.entries(obj)
    .filter(([k]) => k !== "raw_text")
    .map(([k, v]) => `<tr><td>${escapeHtml(k)}</td><td>${escapeHtml(formatValue(v))}</td></tr>`)
    .join("");
  return `
    <div class="info-block">
      <h4>${titles[key] || key}</h4>
      <table>${rows}</table>
    </div>
  `;
}

function formatValue(v) {
  if (v === null || v === undefined) return "—";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

function escapeHtml(str) {
  return String(str ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}