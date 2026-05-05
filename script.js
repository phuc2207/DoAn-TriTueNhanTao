/* ════════════════════════════════════════════════════
   BĐS TP.HCM – SCRIPT.JS v2
   - OpenRouteService API for real road routing
   - Dynamic location dropdowns (all 20 districts + POIs)
   - NLP section removed from predict UI
════════════════════════════════════════════════════ */

// ─────────────────────────────────────────────────────
// CONFIG
// ─────────────────────────────────────────────────────
const API_BASE = 'http://localhost:8001';

const ORS_KEY = 'eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjQ1OWJmM2MxNGU2ZDRmOTJhYWFhYTNkZWNmYmMzZmJjIiwiaCI6Im11cm11cjY0In0=';
const ORS_URL = 'https://api.openrouteservice.org/v2/directions/driving-car';

const DISTRICTS = [
  'Quan_1','Quan_2','Quan_3','Quan_4','Quan_5','Quan_6',
  'Quan_7','Quan_8','Quan_10','Quan_11','Quan_12',
  'Quan_Binh_Thanh','Quan_Tan_Binh','Quan_Phu_Nhuan',
  'Quan_Go_Vap','Quan_Binh_Tan','Quan_Thu_Duc',
  'Huyen_Binh_Chanh','Huyen_Nha_Be','Huyen_Hoc_Mon'
];

const DISTRICT_LABEL = (d) => d.replace(/_/g, ' ');

const NLP_KEYWORDS = [
  'sổ hồng','sổ đỏ','hẻm xe hơi','ô tô vào','nở hậu',
  'mặt tiền','nội thất cao cấp','hồ bơi','hầm xe',
  'biệt thự','chính chủ','nội thất đầy đủ'
];

// ─────────────────────────────────────────────────────
// ROUTE LOCATIONS – all 20 districts + key HCMC POIs
// Each entry has real coordinates for ORS API
// ─────────────────────────────────────────────────────
const ROUTE_LOCATIONS = {
  // ── Quận nội thành ──
  q1:          { name: 'Quận 1 – Bến Nghé',           lat: 10.7769, lng: 106.7009, group: 'Quận / Huyện' },
  q2:          { name: 'Quận 2 – Thảo Điền',          lat: 10.7871, lng: 106.7500, group: 'Quận / Huyện' },
  q3:          { name: 'Quận 3',                       lat: 10.7810, lng: 106.6880, group: 'Quận / Huyện' },
  q4:          { name: 'Quận 4',                       lat: 10.7579, lng: 106.7040, group: 'Quận / Huyện' },
  q5:          { name: 'Quận 5 – Chợ Lớn',            lat: 10.7553, lng: 106.6630, group: 'Quận / Huyện' },
  q6:          { name: 'Quận 6',                       lat: 10.7490, lng: 106.6360, group: 'Quận / Huyện' },
  q7:          { name: 'Quận 7 – Phú Mỹ Hưng',        lat: 10.7320, lng: 106.7200, group: 'Quận / Huyện' },
  q8:          { name: 'Quận 8',                       lat: 10.7244, lng: 106.6640, group: 'Quận / Huyện' },
  q10:         { name: 'Quận 10',                      lat: 10.7743, lng: 106.6680, group: 'Quận / Huyện' },
  q11:         { name: 'Quận 11',                      lat: 10.7640, lng: 106.6530, group: 'Quận / Huyện' },
  q12:         { name: 'Quận 12',                      lat: 10.8600, lng: 106.6640, group: 'Quận / Huyện' },
  binh_thanh:  { name: 'Bình Thạnh',                   lat: 10.8120, lng: 106.7130, group: 'Quận / Huyện' },
  tan_binh:    { name: 'Tân Bình',                     lat: 10.8022, lng: 106.6520, group: 'Quận / Huyện' },
  phu_nhuan:   { name: 'Phú Nhuận',                   lat: 10.7990, lng: 106.6820, group: 'Quận / Huyện' },
  go_vap:      { name: 'Gò Vấp',                      lat: 10.8380, lng: 106.6650, group: 'Quận / Huyện' },
  binh_tan:    { name: 'Bình Tân',                     lat: 10.7647, lng: 106.6046, group: 'Quận / Huyện' },
  thu_duc:     { name: 'TP. Thủ Đức',                  lat: 10.8590, lng: 106.7550, group: 'Quận / Huyện' },
  binh_chanh:  { name: 'Huyện Bình Chánh',             lat: 10.6870, lng: 106.6240, group: 'Quận / Huyện' },
  nha_be:      { name: 'Huyện Nhà Bè',                lat: 10.6910, lng: 106.7520, group: 'Quận / Huyện' },
  hoc_mon:     { name: 'Huyện Hóc Môn',               lat: 10.8880, lng: 106.5920, group: 'Quận / Huyện' },
  // ── Địa điểm nổi bật ──
  ben_thanh:   { name: '🏛 Chợ Bến Thành',             lat: 10.7726, lng: 106.6980, group: 'Địa điểm' },
  tan_son_nhat:{ name: '✈️ Sân bay Tân Sơn Nhất',     lat: 10.8184, lng: 106.6520, group: 'Địa điểm' },
  cho_ray:     { name: '🏥 Bệnh viện Chợ Rẫy',         lat: 10.7553, lng: 106.6611, group: 'Địa điểm' },
  tu_du:       { name: '🏥 Bệnh viện Từ Dũ',            lat: 10.7634, lng: 106.6841, group: 'Địa điểm' },
  bach_khoa:   { name: '🎓 ĐH Bách Khoa TP.HCM',       lat: 10.7720, lng: 106.6584, group: 'Địa điểm' },
  su_pham:     { name: '🎓 ĐH Sư Phạm TP.HCM',         lat: 10.7615, lng: 106.6822, group: 'Địa điểm' },
  rmit:        { name: '🎓 ĐH RMIT Vietnam',            lat: 10.7292, lng: 106.7222, group: 'Địa điểm' },
  bitexco:     { name: '🏙 Tháp Bitexco',               lat: 10.7717, lng: 106.7037, group: 'Địa điểm' },
  vincom_q1:   { name: '🛍 Vincom Center Q.1',          lat: 10.7757, lng: 106.7017, group: 'Địa điểm' },
  sc_vivocity: { name: '🛍 SC VivoCity Q.7',            lat: 10.7295, lng: 106.7210, group: 'Địa điểm' },
  aeon_bd:     { name: '🛍 AEON Mall Bình Dương (Q.12)',lat: 10.8580, lng: 106.6690, group: 'Địa điểm' },
  cong_vien_23:{ name: '🌿 Công viên 23/9',             lat: 10.7683, lng: 106.6944, group: 'Địa điểm' },
  ga_sg:       { name: '🚂 Ga Sài Gòn',                lat: 10.7815, lng: 106.6822, group: 'Địa điểm' },
};

// District center coords (for predict tab dijkstra display)
const DISTRICT_COORDS = {
  Quan_1: [10.7769, 106.7009], Quan_2: [10.7871, 106.7500],
  Quan_3: [10.7810, 106.6880], Quan_4: [10.7579, 106.7040],
  Quan_5: [10.7553, 106.6630], Quan_6: [10.7490, 106.6360],
  Quan_7: [10.7320, 106.7200], Quan_8: [10.7244, 106.6640],
  Quan_10:[10.7743, 106.6680], Quan_11:[10.7640, 106.6530],
  Quan_12:[10.8600, 106.6640], Quan_Binh_Thanh:[10.8120, 106.7130],
  Quan_Tan_Binh:[10.8022, 106.6520], Quan_Phu_Nhuan:[10.7990, 106.6820],
  Quan_Go_Vap:[10.8380, 106.6650], Quan_Binh_Tan:[10.7647, 106.6046],
  Quan_Thu_Duc:[10.8590, 106.7550], Huyen_Binh_Chanh:[10.6870, 106.6240],
  Huyen_Nha_Be:[10.6910, 106.7520], Huyen_Hoc_Mon:[10.8880, 106.5920]
};

// ─────────────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  populateDistricts();
  populateRouteLocations();
  renderKeywordChips();
  checkHealth();
  setupDescListener();
});

function populateDistricts() {
  ['#p-district', '#r-district', '#m-district'].forEach(sel => {
    const el = document.querySelector(sel);
    if (!el) return;
    if (sel === '#m-district') {
      el.innerHTML = '<option value="">– Tất cả quận –</option>';
    } else {
      el.innerHTML = '';
    }
    DISTRICTS.forEach(d => {
      const opt = document.createElement('option');
      opt.value = d; opt.textContent = DISTRICT_LABEL(d);
      el.appendChild(opt);
    });
  });
}

function renderKeywordChips() {
  const wrap = document.getElementById('keyword-chips');
  if (!wrap) return;
  NLP_KEYWORDS.forEach(kw => {
    const chip = document.createElement('span');
    chip.className = 'kw-chip';
    chip.textContent = kw;
    chip.onclick = () => {
      const ta = document.getElementById('p-desc');
      if (!ta.value.includes(kw)) ta.value += (ta.value ? ', ' : '') + kw;
      highlightKeywords();
    };
    wrap.appendChild(chip);
  });
}

function setupDescListener() {
  const ta = document.getElementById('p-desc');
  if (ta) ta.addEventListener('input', highlightKeywords);
}

function highlightKeywords() {
  const ta = document.getElementById('p-desc');
  const val = ta ? ta.value.toLowerCase() : '';
  document.querySelectorAll('.kw-chip').forEach(chip => {
    const active = val.includes(chip.textContent);
    chip.classList.toggle('active', active);
  });
}

// ─────────────────────────────────────────────────────
// HEALTH CHECK
// ─────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(4000) });
    const data = await res.json();
    const dot = document.getElementById('status-dot');
    const txt = document.getElementById('status-text');
    if (data.status === 'ok') {
      dot.className = 'dot dot-ok';
      txt.textContent = data.model_loaded ? 'ML Model Ready' : 'Server Online';
    } else throw new Error();
  } catch {
    document.getElementById('status-dot').className = 'dot dot-err';
    document.getElementById('status-text').textContent = 'Server Offline';
  }
}

// ─────────────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────────────
function setLoading(selector, loading) {
  const btn = document.querySelector(selector);
  if (!btn) return;
  const lbl = btn.querySelector('.btn-label');
  const sp  = btn.querySelector('.btn-spinner');
  btn.disabled = loading;
  lbl?.classList.toggle('d-none', loading);
  sp?.classList.toggle('d-none', !loading);
}

function showToast(msg, type = 'info') {
  const el   = document.getElementById('toast-msg');
  const body = document.getElementById('toast-body');
  body.textContent = msg;
  el.className = `toast align-items-center border-0 text-bg-${type === 'error' ? 'danger' : type === 'success' ? 'success' : 'secondary'}`;
  bootstrap.Toast.getOrCreateInstance(el, { delay: 3500 }).show();
}

async function apiPost(endpoint, body) {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(15000)
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Lỗi không xác định' }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

function liquidityClass(score) {
  if (score >= 7) return 'liq-high';
  if (score >= 5) return 'liq-medium';
  return 'liq-low';
}

// ─────────────────────────────────────────────────────
// ORS API – Real Road Routing
// ─────────────────────────────────────────────────────

/**
 * Call OpenRouteService Directions API
 * Coordinates: [longitude, latitude] (ORS uses lng,lat order)
 */
/**
 * Gọi ORS qua proxy FastAPI /ors-route → tránh CORS hoàn toàn
 */
async function orsRoute(fromLat, fromLng, toLat, toLng) {
  return apiPost('/ors-route', {
    from_lat: fromLat,
    from_lng: fromLng,
    to_lat:   toLat,
    to_lng:   toLng,
  });
}

/**
 * Decode Google/ORS Polyline encoding (precision 5)
 * Returns array of [lat, lng]
 */
function decodePolyline(encoded, precision = 5) {
  const factor = Math.pow(10, precision);
  const coords = [];
  let index = 0, lat = 0, lng = 0;

  while (index < encoded.length) {
    let b, shift = 0, result = 0;
    do {
      b = encoded.charCodeAt(index++) - 63;
      result |= (b & 0x1f) << shift;
      shift += 5;
    } while (b >= 0x20);
    lat += (result & 1) ? ~(result >> 1) : (result >> 1);

    shift = 0; result = 0;
    do {
      b = encoded.charCodeAt(index++) - 63;
      result |= (b & 0x1f) << shift;
      shift += 5;
    } while (b >= 0x20);
    lng += (result & 1) ? ~(result >> 1) : (result >> 1);

    coords.push([lat / factor, lng / factor]);
  }
  return coords;
}

// ─────────────────────────────────────────────────────
// TAB 1 – PREDICT
// ─────────────────────────────────────────────────────
async function predictPrice() {
  const area     = parseFloat(document.getElementById('p-area').value);
  const rooms    = parseInt(document.getElementById('p-rooms').value);
  const tang     = parseInt(document.getElementById('p-tang')?.value || '2');
  const mattien  = parseFloat(document.getElementById('p-mattien')?.value || '4');
  const huong    = document.getElementById('p-huong')?.value || 'Nam';
  const district = document.getElementById('p-district').value;
  const desc     = document.getElementById('p-desc').value.trim();

  if (!area || area <= 0)   return showToast('Vui lòng nhập diện tích hợp lệ.', 'error');
  if (!rooms || rooms <= 0) return showToast('Vui lòng nhập số phòng hợp lệ.', 'error');
  if (!desc)                return showToast('Vui lòng nhập mô tả nhà.', 'error');

  setLoading('#btn-predict', true);
  try {
    const data = await apiPost('/predict', {
      Dien_tich:      area,
      So_phong:       rooms,
      So_tang:        tang,
      Mat_tien_m:     mattien,
      Huong_nha:      huong,
      Vi_tri:         district,
      Noi_dung_mo_ta: desc
    });
    renderPredictResult(data);
    showToast('Dự báo thành công!', 'success');
  } catch (e) {
    showToast('Lỗi: ' + e.message, 'error');
  } finally {
    setLoading('#btn-predict', false);
  }
}

function renderPredictResult(d) {
  document.getElementById('predict-empty').classList.add('d-none');
  const resultEl = document.getElementById('predict-result');
  resultEl.classList.remove('d-none');

  // Price
  const price = d.du_bao?.gia_ty_vnd ?? 0;
  document.getElementById('res-price').textContent = price.toFixed(2) + ' tỷ';
  document.getElementById('res-method').textContent = d.du_bao?.phuong_phap ?? '';

  // Stats
  document.getElementById('res-dist').textContent = d.vi_tri?.khoang_cach_quan1_km ?? '–';
  document.getElementById('res-quan').textContent  = d.vi_tri?.quan ?? '–';

  // Liquidity
  const liqScore = d.thanh_khoan?.diem ?? 0;
  const liqLabel = d.thanh_khoan?.nhan ?? '–';
  document.getElementById('res-liq-score').textContent  = liqScore;
  document.getElementById('res-liq-score2').textContent = liqScore;

  const badgeEl = document.getElementById('res-liq-label');
  badgeEl.textContent = liqLabel;
  badgeEl.className = 'liq-badge mt-1 ' + liquidityClass(liqScore);

  const pct = Math.min((liqScore / 10) * 100, 100);
  document.getElementById('res-liq-bar').style.width = pct + '%';

  // Route
  const routeEl  = document.getElementById('res-route');
  const routeStr = d.vi_tri?.duong_di ?? '';
  routeEl.innerHTML = '';
  if (routeStr) {
    routeStr.split('➔').map(s => s.trim()).filter(Boolean).forEach((node, i, arr) => {
      const span = document.createElement('span');
      span.className = 'route-node';
      span.textContent = node;
      routeEl.appendChild(span);
      if (i < arr.length - 1) {
        const arrow = document.createElement('span');
        arrow.className = 'route-arrow';
        arrow.innerHTML = '<i class="fa-solid fa-chevron-right"></i>';
        routeEl.appendChild(arrow);
      }
    });
  } else {
    routeEl.textContent = 'Không tìm thấy đường đi.';
  }

  // Animate
  resultEl.querySelectorAll('.price-hero, .stat-card, .glass-card').forEach((el, i) => {
    el.classList.add('animate-in', `stagger-${Math.min(i + 1, 5)}`);
  });
}

// ─────────────────────────────────────────────────────
// TAB 2 – RECOMMEND
// ─────────────────────────────────────────────────────
async function recommendProps() {
  const area     = parseFloat(document.getElementById('r-area').value);
  const rooms    = parseInt(document.getElementById('r-rooms').value);
  const district = document.getElementById('r-district').value;
  const price    = parseFloat(document.getElementById('r-price').value);

  if (!area || !rooms || !price) return showToast('Vui lòng điền đầy đủ thông tin.', 'error');

  const btn = document.querySelector('#tab-recommend .btn-primary-custom');
  const lbl = btn?.querySelector('.btn-label');
  const sp  = btn?.querySelector('.btn-spinner');
  if (btn) { btn.disabled = true; lbl?.classList.add('d-none'); sp?.classList.remove('d-none'); }

  try {
    const data = await apiPost('/recommend', {
      Dien_tich: area, So_phong: rooms, Vi_tri: district, Gia: price
    });
    renderRecommendations(data);
    showToast(`Tìm thấy ${data.total_found} BĐS tương tự.`, 'success');
  } catch (e) {
    showToast('Lỗi: ' + e.message, 'error');
  } finally {
    if (btn) { btn.disabled = false; lbl?.classList.remove('d-none'); sp?.classList.add('d-none'); }
  }
}

function renderRecommendations(data) {
  document.getElementById('rec-empty').classList.add('d-none');
  const wrap = document.getElementById('rec-results');
  wrap.classList.remove('d-none');
  wrap.innerHTML = '';

  if (!data.recommendations?.length) {
    wrap.innerHTML = '<div class="col-12"><div class="glass-card empty-state"><div class="empty-icon-wrap mb-3"><i class="fa-solid fa-face-frown"></i></div><p class="text-muted">Không tìm thấy BĐS phù hợp.</p></div></div>';
    return;
  }

  data.recommendations.forEach((prop, i) => {
    const col = document.createElement('div');
    col.className = 'col-md-6 animate-in stagger-' + Math.min(i + 1, 5);
    col.innerHTML = `
      <div class="rec-card">
        <span class="rec-similarity">${prop.do_tuong_dong_pct}% phù hợp</span>
        <div class="rec-price">${prop.gia_ban_ty?.toFixed(2)} tỷ</div>
        <div class="rec-meta mt-1">
          <i class="fa-solid fa-location-dot me-1"></i>${prop.vi_tri}
          &nbsp;·&nbsp;<i class="fa-solid fa-ruler-combined me-1"></i>${prop.dien_tich_m2} m²
          &nbsp;·&nbsp;<i class="fa-solid fa-bed me-1"></i>${prop.so_phong} phòng
        </div>
        ${prop.mo_ta ? `<div class="rec-desc">${prop.mo_ta}</div>` : ''}
      </div>`;
    wrap.appendChild(col);
  });
}

// ─────────────────────────────────────────────────────
// TAB 3 – MARKET ANALYSIS
// ─────────────────────────────────────────────────────
let chartBar = null, chartPie = null;

async function analyzeMarket() {
  const district = document.getElementById('m-district').value;
  const minPrice = parseFloat(document.getElementById('m-min').value) || null;
  const maxPrice = parseFloat(document.getElementById('m-max').value) || null;

  const btn = document.querySelector('#tab-market .btn-primary-custom');
  if (btn) btn.disabled = true;

  try {
    const body = {};
    if (district)            body.Vi_tri     = district;
    if (minPrice !== null)   body.min_price  = minPrice;
    if (maxPrice !== null)   body.max_price  = maxPrice;

    const data = await apiPost('/market-analysis', body);
    renderMarketResult(data);
    showToast('Phân tích hoàn tất.', 'success');
  } catch (e) {
    showToast('Lỗi: ' + e.message, 'error');
  } finally {
    if (btn) btn.disabled = false;
  }
}

function renderMarketResult(d) {
  document.getElementById('market-empty').classList.add('d-none');
  document.getElementById('market-result').classList.remove('d-none');

  document.getElementById('m-total').textContent = d.tong_tin?.toLocaleString() ?? '–';
  document.getElementById('m-avg').textContent   = d.gia_trung_binh?.toFixed(2) ?? '–';
  document.getElementById('m-min-v').textContent = d.gia_thap_nhat?.toFixed(2) ?? '–';
  document.getElementById('m-max-v').textContent = d.gia_cao_nhat?.toFixed(2) ?? '–';

  const districts = d.phan_tich_theo_quan ?? [];

  const tbody = document.getElementById('market-table');
  tbody.innerHTML = '';
  districts.forEach(row => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><span class="route-node">${DISTRICT_LABEL(row.Vi_tri)}</span></td>
      <td>${row.so_tin}</td>
      <td><strong style="color:var(--accent)">${row.gia_tb?.toFixed(2)}</strong></td>
      <td>${row.gia_min?.toFixed(2)}</td>
      <td>${row.gia_max?.toFixed(2)}</td>`;
    tbody.appendChild(tr);
  });

  const labels    = districts.map(r => DISTRICT_LABEL(r.Vi_tri));
  const avgPrices = districts.map(r => r.gia_tb);
  const counts    = districts.map(r => r.so_tin);
  const tickStyle = { color: '#5e7090', font: { family: 'Outfit', size: 11 } };
  const gridStyle = { color: 'rgba(255,255,255,.05)' };

  if (chartBar) chartBar.destroy();
  chartBar = new Chart(document.getElementById('chart-district').getContext('2d'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Giá TB (tỷ)', data: avgPrices,
        backgroundColor: 'rgba(212,168,83,0.5)',
        borderColor: 'rgba(212,168,83,0.9)',
        borderWidth: 1, borderRadius: 6, borderSkipped: false
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: '#a0aec0', font: { family: 'Outfit', size: 12 } } } },
      scales: { x: { ticks: tickStyle, grid: gridStyle }, y: { ticks: tickStyle, grid: gridStyle } }
    }
  });

  if (chartPie) chartPie.destroy();
  const PIE_COLORS = ['#d4a853','#60a5fa','#4ade80','#f87171','#a78bfa','#fb923c','#22d3ee','#e879f9','#facc15','#94a3b8','#34d399','#f472b6','#c084fc','#38bdf8','#fbbf24','#e11d48','#06b6d4','#8b5cf6','#ec4899','#14b8a6'];
  chartPie = new Chart(document.getElementById('chart-pie').getContext('2d'), {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data: counts,
        backgroundColor: PIE_COLORS.slice(0, labels.length),
        borderColor: '#111827', borderWidth: 2
      }]
    },
    options: {
      responsive: true, cutout: '62%',
      plugins: {
        legend: {
          position: 'right',
          labels: { color: '#a0aec0', font: { family: 'Outfit', size: 11 }, boxWidth: 12, padding: 8 }
        }
      }
    }
  });
}

// ─────────────────────────────────────────────────────
// TAB 4 – ROUTE / MAP  (OpenRouteService)
// ─────────────────────────────────────────────────────
let leafletMap  = null;
let routeLayer  = [];

/**
 * Populate both route dropdowns with grouped options
 * (Quận/Huyện  +  Địa điểm nổi bật)
 */
function populateRouteLocations() {
  const selectors = ['#rt-start', '#rt-end'];
  const groups = {};

  Object.entries(ROUTE_LOCATIONS).forEach(([key, loc]) => {
    if (!groups[loc.group]) groups[loc.group] = [];
    groups[loc.group].push({ key, ...loc });
  });

  selectors.forEach((sel, idx) => {
    const el = document.querySelector(sel);
    if (!el) return;
    el.innerHTML = '';
    Object.entries(groups).forEach(([groupName, items]) => {
      const optGroup = document.createElement('optgroup');
      optGroup.label = groupName;
      items.forEach(item => {
        const opt = document.createElement('option');
        opt.value = item.key;
        opt.textContent = item.name;
        optGroup.appendChild(opt);
      });
      el.appendChild(optGroup);
    });
  });

  // Default: different start/end
  const endEl = document.querySelector('#rt-end');
  if (endEl) endEl.value = 'q1';
  const startEl = document.querySelector('#rt-start');
  if (startEl) startEl.value = 'tan_son_nhat';
}

async function findRoute() {
  const startKey = document.querySelector('#rt-start')?.value;
  const endKey   = document.querySelector('#rt-end')?.value;

  if (!startKey || !endKey) return showToast('Chọn điểm xuất phát và đích đến.', 'error');
  if (startKey === endKey)  return showToast('Điểm xuất phát và đích không được trùng nhau.', 'error');

  const startLoc = ROUTE_LOCATIONS[startKey];
  const endLoc   = ROUTE_LOCATIONS[endKey];

  if (!startLoc || !endLoc) return showToast('Không tìm thấy tọa độ điểm đã chọn.', 'error');

  const btn = document.querySelector('#tab-route .btn-primary-custom');
  const lbl = btn?.querySelector('.btn-label');
  const sp  = btn?.querySelector('.btn-spinner');
  if (btn) { btn.disabled = true; lbl?.classList.add('d-none'); sp?.classList.remove('d-none'); }

  try {
    const data = await orsRoute(startLoc.lat, startLoc.lng, endLoc.lat, endLoc.lng);
    renderORSRouteResult(data, startLoc, endLoc);
    showToast('Tìm đường thành công!', 'success');
  } catch (e) {
    showToast('ORS lỗi: ' + e.message, 'error');
  } finally {
    if (btn) { btn.disabled = false; lbl?.classList.remove('d-none'); sp?.classList.add('d-none'); }
  }
}

function renderORSRouteResult(data, startLoc, endLoc) {
  const route = data.routes?.[0];
  if (!route) { showToast('Không có kết quả từ ORS.', 'error'); return; }

  const summary = route.summary;
  const distKm  = (summary.distance / 1000).toFixed(2);
  const timeMins = Math.round(summary.duration / 60);

  // Steps count
  const steps = route.segments?.[0]?.steps ?? [];
  const stepCount = steps.length;

  document.getElementById('route-info').classList.remove('d-none');
  document.getElementById('rt-dist').textContent  = distKm;
  document.getElementById('rt-time').textContent  = timeMins;
  document.getElementById('rt-turns').textContent = stepCount;

  // Path nodes: start → up to 3 via points → end
  const pathEl = document.getElementById('rt-path');
  pathEl.innerHTML = '';

  const pathNodes = [startLoc.name];

  // Add significant turn instructions (not depart/arrive)
  steps.filter(s => s.type !== 1 && s.type !== 11 && s.name && s.name !== '-')
       .slice(0, 3)
       .forEach(s => pathNodes.push(s.name));

  pathNodes.push(endLoc.name);

  pathNodes.forEach((name, i, arr) => {
    const node = document.createElement('span');
    node.className = 'route-node';
    node.textContent = name;
    pathEl.appendChild(node);
    if (i < arr.length - 1) {
      const arrow = document.createElement('span');
      arrow.className = 'route-arrow';
      arrow.innerHTML = '<i class="fa-solid fa-chevron-right"></i>';
      pathEl.appendChild(arrow);
    }
  });

  // Draw on map
  const decodedCoords = decodePolyline(route.geometry);
  drawORSRoute(decodedCoords, startLoc, endLoc, distKm, timeMins);
}

function drawORSRoute(coords, startLoc, endLoc, distKm, timeMins) {
  const mapDiv     = document.getElementById('map');
  const placeholder = document.getElementById('map-placeholder');

  mapDiv.style.display = 'block';
  placeholder.style.display = 'none';

  if (!leafletMap) {
    leafletMap = L.map('map', { zoomControl: true }).setView([startLoc.lat, startLoc.lng], 13);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap · Route by OpenRouteService',
      maxZoom: 19
    }).addTo(leafletMap);
  }

  // Clear old layers
  routeLayer.forEach(l => leafletMap.removeLayer(l));
  routeLayer = [];

  // Draw route polyline (actual road path)
  const polyline = L.polyline(coords, {
    color: '#d4a853',
    weight: 5,
    opacity: 0.9,
    lineJoin: 'round',
    lineCap: 'round'
  }).addTo(leafletMap);
  routeLayer.push(polyline);

  // Glow layer
  const glowLine = L.polyline(coords, {
    color: '#e8c06a',
    weight: 10,
    opacity: 0.18,
    lineJoin: 'round',
  }).addTo(leafletMap);
  routeLayer.push(glowLine);

  // Marker icon factory
  const makeMarker = (label, color, isPin = false) => L.divIcon({
    className: '',
    html: `<div style="
      background:${color};width:38px;height:38px;
      border-radius:50% 50% 50% 0;transform:rotate(-45deg);
      border:3px solid rgba(255,255,255,0.85);
      box-shadow:0 3px 12px rgba(0,0,0,0.55);
      display:flex;align-items:center;justify-content:center;">
      <span style="transform:rotate(45deg);font-size:13px;font-weight:700;color:#fff;">${label}</span>
    </div>`,
    iconSize: [38, 38], iconAnchor: [19, 38], popupAnchor: [0, -38]
  });

  // Start marker
  const startMarker = L.marker([startLoc.lat, startLoc.lng], { icon: makeMarker('A', '#4ade80') })
    .addTo(leafletMap)
    .bindPopup(`<strong>${startLoc.name}</strong><br><small style="color:#5e7090">Điểm xuất phát</small>`);
  routeLayer.push(startMarker);

  // End marker
  const endMarker = L.marker([endLoc.lat, endLoc.lng], { icon: makeMarker('B', '#f87171') })
    .addTo(leafletMap)
    .bindPopup(`<strong>${endLoc.name}</strong><br><small style="color:#5e7090">Điểm đến · ${distKm} km · ${timeMins} phút</small>`);
  routeLayer.push(endMarker);

  leafletMap.fitBounds(polyline.getBounds(), { padding: [50, 50] });
  setTimeout(() => leafletMap.invalidateSize(), 200);
}

// Fix map size when tab becomes visible
document.querySelectorAll('[data-bs-toggle="tab"]').forEach(tab => {
  tab.addEventListener('shown.bs.tab', e => {
    if (e.target.getAttribute('href') === '#tab-route' && leafletMap) {
      setTimeout(() => leafletMap.invalidateSize(), 100);
    }
  });
});