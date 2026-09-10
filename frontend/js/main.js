/**
 * KlaimGuard AI — Frontend Dashboard Logic
 *
 * Handles:
 * - "Simulasikan 1 Klaim" → POST /api/claims/v1/simulate-batch {n: 1}
 * - "Simulasi Massal (1000 Klaim)" → POST /api/claims/v1/simulate-batch {n: 1000}
 * - Live Dashboard Table population with XAI reasons
 * - Stats dashboard (total, flagged, safe, processing time)
 * - Filter toggles (All / Flagged / Safe)
 */

// ═══ Configuration ══════════════════════════════════════════
const API_BASE = window.location.origin;
const API = {
    health:        `${API_BASE}/api/health`,
    analyzeClaim:  `${API_BASE}/api/claims/v1/analyze`,
    simulateBatch: `${API_BASE}/api/claims/v1/simulate-batch`,
};

// ═══ State ══════════════════════════════════════════════════
const state = {
    results: [],
    totalAnalyzed: 0,
    totalFlagged: 0,
    totalSafe: 0,
    lastProcessingTime: null,
    modelLoaded: false,
    activeFilter: 'all',
};

// ═══ DOM References ═════════════════════════════════════════
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ═══ Initialization ═════════════════════════════════════════
document.addEventListener("DOMContentLoaded", () => {
    checkConnection();
    setupEventHandlers();
    setInterval(checkConnection, 30000);
});

// ═══ Connection Health Check ════════════════════════════════
async function checkConnection() {
    const badge = $("#connectionStatus");
    try {
        const res = await fetch(API.health, { signal: AbortSignal.timeout(5000) });
        const data = await res.json();

        badge.className = "status-badge status-connected";
        badge.querySelector(".status-text").textContent = "Connected";
        state.modelLoaded = data.ml_model_loaded || false;
        updateModelBadge();
    } catch (err) {
        badge.className = "status-badge status-error";
        badge.querySelector(".status-text").textContent = "Disconnected";
        state.modelLoaded = false;
        updateModelBadge();
    }
}

function updateModelBadge() {
    const badge = $("#modelBadge");
    const text = $("#modelStatusText");
    const dot = badge.querySelector(".hero-badge-dot");

    if (state.modelLoaded) {
        text.textContent = "XGBoost ML Ready";
        badge.style.background = "#E8F5E9";
        badge.style.color = "#2E7D32";
        badge.style.borderColor = "#A5D6A7";
        dot.style.background = "#2E7D32";
    } else {
        text.textContent = "Rule-Based Only";
        badge.style.background = "#FFF8E1";
        badge.style.color = "#F57F17";
        badge.style.borderColor = "#FFE082";
        dot.style.background = "#F57F17";
    }
}

// ═══ Event Handlers ═════════════════════════════════════════
function setupEventHandlers() {
    // Simulasikan 1 Klaim
    $("#simulateOneBtn").addEventListener("click", () => simulateClaims(1));

    // Simulasi Massal (1000 Klaim)
    $("#simulateBatchBtn").addEventListener("click", () => simulateClaims(1000));

    $("#claimForm").addEventListener("submit", analyzeManualClaim);

    // Filter badges
    $$(".filter-badge").forEach(badge => {
        badge.addEventListener("click", () => {
            $$(".filter-badge").forEach(b => b.classList.remove("active"));
            badge.classList.add("active");
            state.activeFilter = badge.dataset.filter;
            renderTable();
        });
    });
}

async function analyzeManualClaim(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const button = $("#analyzeClaimBtn");
    const formData = new FormData(form);
    const claim = {
        claim_id: formData.get("claim_id").trim(),
        doctor_notes: formData.get("doctor_notes").trim(),
        admin_billing_code: formData.get("admin_billing_code").split(",").map(code => code.trim()).filter(Boolean),
        length_of_stay: Number(formData.get("length_of_stay")),
        total_charge: Number(formData.get("total_charge")),
    };

    button.disabled = true;
    button.classList.add("is-loading");
    button.textContent = "Menganalisis...";

    try {
        const startTime = performance.now();
        const res = await fetch(API.analyzeClaim, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(claim),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: "Gagal memproses klaim" }));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }

        const result = await res.json();
        const elapsedMs = performance.now() - startTime;
        state.results = [result];
        state.totalAnalyzed = 1;
        state.totalFlagged = result.status === "FLAGGED" ? 1 : 0;
        state.totalSafe = result.status === "SAFE" ? 1 : 0;
        state.lastProcessingTime = elapsedMs;
        updateStats({
            total: 1,
            flagged: state.totalFlagged,
            safe: state.totalSafe,
            processing_time_ms: elapsedMs,
        });
        renderTable();
        showToast(result.status === "FLAGGED" ? "Klaim memerlukan tinjauan verifikator" : "Klaim terverifikasi aman", result.status === "FLAGGED" ? "error" : "success");
        $("#resultsSection").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
        showToast(`Error: ${err.message}`, "error");
        console.error("Manual analysis failed:", err);
    } finally {
        button.disabled = false;
        button.classList.remove("is-loading");
        button.textContent = "Analisis Klaim";
    }
}

// ═══ Simulate Claims ════════════════════════════════════════
async function simulateClaims(n) {
    const btn = n === 1 ? $("#simulateOneBtn") : $("#simulateBatchBtn");
    const overlay = $("#loadingOverlay");
    const loadingText = $("#loadingText");
    const loadingSubtext = $("#loadingSubtext");

    // Show loading
    btn.disabled = true;
    overlay.classList.remove("hidden");
    loadingText.textContent = n === 1
        ? "Menganalisis 1 klaim..."
        : `Memproses ${n.toLocaleString('id-ID')} klaim...`;
    loadingSubtext.textContent = "Menjalankan Hybrid AI Engine (XGBoost + Rule Engine)";

    try {
        const startTime = performance.now();
        const res = await fetch(API.simulateBatch, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ n }),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: "Unknown error" }));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }

        const data = await res.json();
        const clientTime = performance.now() - startTime;

        // Update state
        state.results = data.results;
        state.totalAnalyzed = data.total;
        state.totalFlagged = data.flagged;
        state.totalSafe = data.safe;
        state.lastProcessingTime = data.processing_time_ms;

        // Update UI
        updateStats(data);
        renderTable();

        const msg = n === 1
            ? `1 klaim dianalisis dalam ${(data.processing_time_ms).toFixed(0)}ms`
            : `Berhasil memproses ${data.total.toLocaleString('id-ID')} klaim dalam ${(data.processing_time_ms / 1000).toFixed(1)} detik. Ditemukan ${data.flagged} indikasi fraud, dan ${data.safe} klaim valid.`;
        showToast(msg, "success");

    } catch (err) {
        showToast(`Error: ${err.message}`, "error");
        console.error("Simulation failed:", err);
    } finally {
        btn.disabled = false;
        overlay.classList.add("hidden");
    }
}

// ═══ Update Stats Panel ═════════════════════════════════════
function updateStats(data) {
    // Total analyzed
    animateNumber($("#totalClaimsValue"), data.total);

    // Flagged
    animateNumber($("#flaggedClaimsValue"), data.flagged);
    const flaggedPct = data.total > 0
        ? ((data.flagged / data.total) * 100).toFixed(0)
        : 0;
    $("#flaggedPercent").textContent = `(${flaggedPct}%)`;

    // Safe
    animateNumber($("#safeClaimsValue"), data.safe);

    // Processing time
    if (data.processing_time_ms < 1000) {
        $("#processingTimeValue").textContent = `${data.processing_time_ms.toFixed(0)}ms`;
    } else {
        $("#processingTimeValue").textContent = `${(data.processing_time_ms / 1000).toFixed(1)}s`;
    }

    // Summary text
    const summaryText = data.total === 1 
        ? `1 klaim dianalisis dalam ${(data.processing_time_ms).toFixed(0)}ms`
        : `Berhasil memproses ${data.total.toLocaleString('id-ID')} klaim dalam ${(data.processing_time_ms / 1000).toFixed(1)} detik. Ditemukan ${data.flagged} indikasi fraud, dan ${data.safe} klaim valid.`;
    $("#resultsSummary").textContent = summaryText;
}

function animateNumber(el, target) {
    const current = parseInt(el.textContent) || 0;
    if (current === target) {
        el.textContent = target;
        return;
    }

    const duration = 600;
    const start = performance.now();

    function step(timestamp) {
        const progress = Math.min((timestamp - start) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
        el.textContent = Math.round(current + (target - current) * eased);
        if (progress < 1) requestAnimationFrame(step);
    }

    requestAnimationFrame(step);
}

// ═══ Render Dashboard Table ═════════════════════════════════
function renderTable() {
    const tbody = $("#resultsTableBody");
    let filtered = state.results;

    // Apply filter
    if (state.activeFilter === 'flagged') {
        filtered = state.results.filter(r => r.status === 'FLAGGED');
    } else if (state.activeFilter === 'safe') {
        filtered = state.results.filter(r => r.status === 'SAFE');
    }

    if (filtered.length === 0) {
        tbody.innerHTML = `
            <tr class="empty-row">
                <td colspan="6">
                    <div class="empty-state">
                        <p>Tidak ada data${state.activeFilter !== 'all' ? ' untuk filter ini' : ''}</p>
                    </div>
                </td>
            </tr>
        `;
        return;
    }

    // Render rows — show flagged first for impact
    const sorted = [...filtered].sort((a, b) => {
        if (a.status === 'FLAGGED' && b.status !== 'FLAGGED') return -1;
        if (a.status !== 'FLAGGED' && b.status === 'FLAGGED') return 1;
        return b.probability_score - a.probability_score;
    });

    tbody.innerHTML = sorted.map((r, i) => {
        const isFlagged = r.status === 'FLAGGED';
        const rowClass = isFlagged ? 'flagged-row' : 'safe-row';

        // Format billing codes with translations
        const billingHtml = r.admin_billing_translated
            .map(t => `<span class="billing-tag" title="${escapeHtml(t.code)}">${escapeHtml(t.description)}</span>`)
            .join(' ');

        // Format cost as IDR
        const costFormatted = `Rp ${r.total_charge.toLocaleString('id-ID', { minimumFractionDigits: 0 })}`;

        // Status pill
        const statusHtml = isFlagged
            ? '<span class="status-pill pill-flagged">⚠ FLAGGED</span>'
            : '<span class="status-pill pill-safe">✓ SAFE</span>';

        // XAI reason (parse markdown bold)
        let reasonHtml = escapeHtml(r.reason_narrative);
        reasonHtml = reasonHtml.replace(/\*\*(.*?)\*\*/g, '<b>$1</b>');

        const reasonShort = reasonHtml.length > 200
            ? reasonHtml.substring(0, 197) + '...'
            : reasonHtml;

        return `
            <tr class="${rowClass} row-animate" style="animation-delay: ${Math.min(i * 20, 500)}ms">
                <td class="claim-id-cell">${escapeHtml(r.claim_id)}</td>
                <td class="doctor-notes-cell">${escapeHtml(r.doctor_notes)}</td>
                <td class="billing-cell">${billingHtml}</td>
                <td class="cost-cell">${costFormatted}</td>
                <td>${statusHtml}</td>
                <td class="xai-cell" title="${escapeHtml(r.reason_narrative)}">${reasonShort}</td>
            </tr>
        `;
    }).join('');
}

// ═══ Toast Notifications ════════════════════════════════════
function showToast(message, type = "info") {
    const container = $("#toastContainer");
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        if (toast.parentNode) toast.remove();
    }, 4000);
}

// ═══ Utility ════════════════════════════════════════════════
function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}
