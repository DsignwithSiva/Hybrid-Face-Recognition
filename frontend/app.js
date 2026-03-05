/* ═══════════════════════════════════════════════════════════
   HYBRID FACE RECOGNITION — Frontend App Logic
   v2.0 — Real FastAPI Backend
═══════════════════════════════════════════════════════════ */

// ─── State ─────────────────────────────────────────────────
const state = {
  currentPage: 'dashboard',
  jobsRun: 0,
  storedFaces: 0,
  searchCount: 0,
  results: [],
  backendOnline: false,
  namespaces: {},
};

// ─── Navigation ────────────────────────────────────────────
function navigateTo(page) {
  const pages = document.querySelectorAll('.page');
  const navItems = document.querySelectorAll('.nav-item');

  pages.forEach(p => p.classList.remove('active'));
  navItems.forEach(n => n.classList.remove('active'));

  const target = document.getElementById(`page-${page}`);
  const nav = document.querySelector(`[data-page="${page}"]`);

  if (target) target.classList.add('active');
  if (nav) nav.classList.add('active');

  const crumb = document.getElementById('breadcrumbCurrent');
  const labels = {
    dashboard: 'Dashboard',
    store: 'Store Faces',
    search: 'Search Person',
    batch: 'Batch Search',
    multivideo: 'Multi-Video Search',
    ultimate: 'Ultimate Search',
    bulkstore: 'Bulk Store',
    results: 'Results',
  };
  if (crumb) crumb.textContent = labels[page] || page;
  state.currentPage = page;

  // Close sidebar on mobile
  if (window.innerWidth <= 700) {
    document.getElementById('sidebar').classList.remove('open');
  }
}

// Click all nav items
document.querySelectorAll('.nav-item').forEach(item => {
  item.addEventListener('click', e => {
    e.preventDefault();
    navigateTo(item.dataset.page);
  });
});

// Sidebar toggle (mobile)
document.getElementById('sidebarToggle').addEventListener('click', () => {
  document.getElementById('sidebar').classList.toggle('open');
});

// ─── Particles ─────────────────────────────────────────────
function spawnParticles() {
  const container = document.getElementById('particles');
  if (!container) return;
  const colors = ['#00d4ff', '#a855f7', '#22c55e', '#f59e0b', '#ec4899'];
  for (let i = 0; i < 35; i++) {
    const p = document.createElement('div');
    p.className = 'particle';
    const size = Math.random() * 3 + 1;
    p.style.cssText = `
      width:${size}px; height:${size}px;
      left:${Math.random() * 100}%;
      top:${Math.random() * 100}%;
      background:${colors[Math.floor(Math.random() * colors.length)]};
      animation-duration:${Math.random() * 12 + 8}s;
      animation-delay:${Math.random() * 8}s;
    `;
    container.appendChild(p);
  }
}
spawnParticles();

// ─── Stat Counter Animation ────────────────────────────────
function animateCounter(el, end, duration = 1200, suffix = '') {
  const start = 0;
  const step = (end - start) / (duration / 16);
  let current = start;
  const tick = setInterval(() => {
    current += step;
    if (current >= end) { current = end; clearInterval(tick); }
    el.textContent = Math.floor(current).toLocaleString() + suffix;
  }, 16);
}

// ─── Backend Status & Real Stats ───────────────────────────
async function loadStatus() {
  try {
    const r = await fetch('/api/status');
    if (!r.ok) throw new Error('offline');
    const data = await r.json();
    if (!data.ok) throw new Error(data.error || 'offline');

    state.backendOnline = true;
    state.namespaces    = data.namespaces || {};

    // Device badge
    const label = document.getElementById('deviceLabel');
    const badge = document.getElementById('deviceBadge');
    const icon  = badge && badge.querySelector('i');
    if (label) label.textContent = data.device === 'cuda' ? 'CUDA · GPU Ready' : 'CPU Mode';
    if (badge && data.device === 'cuda') badge.style.borderColor = 'rgba(34,197,94,0.35)';
    if (icon)  icon.style.color = data.device === 'cuda' ? 'var(--green)' : 'var(--orange)';

    // Live face counts
    const totalVecs = data.total_vectors || 0;
    animateCounter(document.getElementById('stat-stored'), totalVecs);

    // Status pill
    const pillTx = document.getElementById('statusText');
    if (pillTx) pillTx.textContent = 'Backend Online';
    const pill = document.getElementById('statusPill');
    if (pill)  pill.style.borderColor = 'rgba(34,197,94,0.35)';

    // Namespace autocomplete for dropdowns
    const nsKeys = Object.keys(state.namespaces);
    ['searchNamespace','batchNamespace'].forEach(id => {
      const el = document.getElementById(id);
      if (!el) return;
      let dl = document.getElementById(`${id}-dl`);
      if (!dl) {
        dl = document.createElement('datalist');
        dl.id = `${id}-dl`;
        document.body.appendChild(dl);
        el.setAttribute('list', `${id}-dl`);
      }
      dl.innerHTML = nsKeys.map(k =>
        `<option value="${k}">${k} (${state.namespaces[k]} faces)</option>`
      ).join('');
    });

    return data;
  } catch {
    const pillTx = document.getElementById('statusText');
    if (pillTx) pillTx.textContent = 'Connecting…';
    return null;
  }
}

// ─── Toast ─────────────────────────────────────────────────
function showToast(msg, type = 'success') {
  const toast = document.getElementById('toast');
  const toastMsg = document.getElementById('toastMsg');
  const icon = toast.querySelector('.toast-icon');

  toastMsg.textContent = msg;
  const colors = { success: 'var(--green)', error: 'var(--red)', info: 'var(--cyan)', warning: 'var(--orange)' };
  icon.style.color = colors[type] || colors.success;
  toast.style.borderColor = (colors[type] || colors.success).replace('var(', 'rgba(').replace(')', ', 0.3)');

  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 3000);
}

// ─── Temp File Store ──────────────────────────────────────
// Maps a logical key → { file, name, tempPath, objectURL }
const tempFiles = {};

function storeTempFile(key, file) {
  // Revoke old URL if present
  if (tempFiles[key]) URL.revokeObjectURL(tempFiles[key].objectURL);
  const objectURL = URL.createObjectURL(file);
  const tempPath  = `temp://${key}_${file.name}`;
  tempFiles[key] = { file, name: file.name, tempPath, objectURL };
  return tempFiles[key];
}

// ─── File Upload Handlers ───────────────────────────────────
function handleVideoUpload(input, zoneId, nameId) {
  const file = input.files[0];
  if (!file) return;
  const nameEl = document.getElementById(nameId);
  const zone = document.getElementById(zoneId);
  if (nameEl) nameEl.textContent = `✔ ${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB)`;
  zone.style.borderColor = 'var(--cyan)';
  zone.style.background = 'rgba(0,212,255,0.04)';

  const tf = storeTempFile('video_store', file);
  const ns = document.getElementById('storeNamespace');
  if (ns) ns.value = `video_${file.name.replace(/\.[^.]+$/, '')}`;

  // Show temp path badge
  const badge = document.getElementById('storeTempBadge');
  if (badge) {
    badge.classList.remove('hidden');
    const pathEl = badge.querySelector('.tpb-path');
    if (pathEl) pathEl.textContent = tf.tempPath;
  }

  showToast(`Video loaded: ${file.name}`, 'success');
}

// Compact video-item row upload chip (Bulk Store / Ultimate Search)
function handleVideoItemUpload(input, iconId, chipSpanId) {
  const file = input.files[0];
  if (!file) return;
  storeTempFile(input.id, file);

  const icon = document.getElementById(iconId);
  if (icon) {
    icon.style.borderColor = 'var(--cyan)';
    icon.innerHTML = '<i class="fa-solid fa-circle-check" style="color:var(--cyan)"></i>';
  }
  const chip = document.getElementById(chipSpanId);
  if (chip) chip.innerHTML = `<i class="fa-solid fa-circle-check" style="color:var(--cyan)"></i> ${file.name}`;

  showToast(`Video saved → temp://${input.id}_${file.name}`, 'success');
}

// Main image upload zone (Search Person / Multi-Video)
// Signature: (input, zoneId, previewId, contentId, badgeId)
function handleImageUpload(input, zoneId, previewId, contentId, badgeId) {
  const file = input.files[0];
  if (!file) return;

  // Store in temp
  const key = zoneId;
  const tf  = storeTempFile(key, file);

  // Preview image
  const preview = document.getElementById(previewId);
  const content = document.getElementById(contentId);
  const previewImg = preview?.querySelector('img');
  if (previewImg) previewImg.src = tf.objectURL;
  if (preview) preview.style.display = 'block';
  if (content) content.style.display = 'none';

  // Show temp path badge
  const badge = document.getElementById(badgeId);
  if (badge) {
    badge.classList.remove('hidden');
    const pathEl = badge.querySelector('.tpb-path');
    if (pathEl) pathEl.textContent = tf.tempPath;
  }

  document.getElementById(zoneId).style.borderColor = 'var(--purple)';
  showToast(`Photo saved → ${tf.tempPath}`, 'success');
}

// Compact person-row upload chip
function handlePersonUpload(input, avatarId, chipSpanId) {
  const file = input.files[0];
  if (!file) return;

  const key = input.id;
  const tf  = storeTempFile(key, file);

  // Swap avatar icon → thumbnail
  const avatar = document.getElementById(avatarId);
  if (avatar) {
    avatar.innerHTML = `<img src="${tf.objectURL}" style="width:100%;height:100%;object-fit:cover;border-radius:50%" />`;
    avatar.style.padding = '0';
    avatar.style.border = '2px solid var(--green)';
  }

  // Update chip label to show filename + temp indicator
  const chip = document.getElementById(chipSpanId);
  if (chip) {
    chip.innerHTML = `<i class="fa-solid fa-circle-check" style="color:var(--green)"></i> ${file.name}`;
  }

  showToast(`Saved → temp://${key}_${file.name}`, 'success');
}

// ─── Person / Video List Helpers ───────────────────────────
let personCount = 3;
function addPerson() {
  const list = document.getElementById('batchPeopleList');
  if (!list) return;
  const num = list.querySelectorAll('.person-item').length + 1;
  personCount++;
  const uid = `bd${personCount}`;
  const item = document.createElement('div');
  item.className = 'person-item';
  item.innerHTML = `
    <div class="person-avatar" id="pav-${uid}"><i class="fa-solid fa-user"></i></div>
    <div class="person-fields">
      <input class="field-input" type="text" placeholder="Person ${num}" />
      <label class="upload-chip">
        <input type="file" id="pup-${uid}" accept="image/*" hidden onchange="handlePersonUpload(this,'pav-${uid}','pchip-${uid}')" />
        <i class="fa-solid fa-camera"></i>
        <span id="pchip-${uid}">Upload photo</span>
      </label>
    </div>
    <button class="person-remove" onclick="removePerson(this)"><i class="fa-solid fa-xmark"></i></button>
  `;
  list.appendChild(item);
  updateUltimateCombos();
}

function addPersonTo(listId) {
  const list = document.getElementById(listId);
  if (!list) return;
  const num = list.querySelectorAll('.person-item').length + 1;
  personCount++;
  const uid = `ul${personCount}`;
  const item = document.createElement('div');
  item.className = 'person-item';
  item.innerHTML = `
    <div class="person-avatar" id="pav-${uid}"><i class="fa-solid fa-user"></i></div>
    <div class="person-fields">
      <input class="field-input" type="text" placeholder="Person ${num}" />
      <label class="upload-chip">
        <input type="file" id="pup-${uid}" accept="image/*" hidden onchange="handlePersonUpload(this,'pav-${uid}','pchip-${uid}'); updateUltimateCombos()" />
        <i class="fa-solid fa-camera"></i>
        <span id="pchip-${uid}">Upload photo</span>
      </label>
    </div>
    <button class="person-remove" onclick="removePerson(this); updateUltimateCombos()"><i class="fa-solid fa-xmark"></i></button>
  `;
  list.appendChild(item);
  updateUltimateCombos();
}

function removePerson(btn) {
  btn.closest('.person-item').remove();
  updateUltimateCombos();
}

function addVideo() {
  const list = document.getElementById('multiVideoList');
  if (!list) return;
  const item = document.createElement('div');
  item.className = 'video-item';
  item.innerHTML = `
    <div class="video-icon"><i class="fa-solid fa-film"></i></div>
    <input class="field-input" type="text" placeholder="video.mp4" />
    <button class="person-remove" onclick="this.parentElement.remove()"><i class="fa-solid fa-xmark"></i></button>
  `;
  list.appendChild(item);
}

function addVideoTo(listId) {
  const list = document.getElementById(listId);
  if (!list) return;
  const colorMap  = { ultimateVideoList: '', bulkVideoList: 'icon-pink' };
  const cls  = colorMap[listId] || '';
  const idx  = list.querySelectorAll('.video-item').length + 1;
  const uid  = `vup-${listId.slice(0, 4)}-${Date.now()}`;
  const avid = `vav-${listId.slice(0, 4)}-${Date.now()}`;
  const cid  = `vchip-${listId.slice(0, 4)}-${Date.now()}`;
  const onchg = listId === 'ultimateVideoList'
    ? `handleVideoItemUpload(this,'${avid}','${cid}'); updateUltimateCombos()`
    : `handleVideoItemUpload(this,'${avid}','${cid}')`;
  const onrm = listId === 'ultimateVideoList'
    ? `this.parentElement.remove(); updateUltimateCombos()`
    : `this.parentElement.remove()`;
  const item = document.createElement('div');
  item.className = 'video-item';
  item.innerHTML = `
    <div class="video-icon ${cls}" id="${avid}"><i class="fa-solid fa-film"></i></div>
    <label class="upload-chip" for="${uid}"><i class="fa-solid fa-film"></i><span id="${cid}">Upload Video ${idx}</span></label>
    <input type="file" id="${uid}" accept="video/*" hidden onchange="${onchg}" />
    <button class="person-remove" onclick="${onrm}"><i class="fa-solid fa-xmark"></i></button>
  `;
  list.appendChild(item);
  updateUltimateCombos();
}

function updateUltimateCombos() {
  const peopleList = document.getElementById('ultimatePeopleList');
  const videoList = document.getElementById('ultimateVideoList');
  if (!peopleList || !videoList) return;
  const pCount = peopleList.querySelectorAll('.person-item').length;
  const vCount = videoList.querySelectorAll('.video-item').length;
  const el = document.getElementById('uhPeopleCount');
  const ev = document.getElementById('uhVideosCount');
  const ec = document.getElementById('uhCombos');
  if (el) el.textContent = pCount;
  if (ev) ev.textContent = vCount;
  if (ec) ec.textContent = pCount * vCount;
}

// ─── Modal ──────────────────────────────────────────────────
function openModal(title) {
  document.getElementById('modalTitle').textContent = title;
  document.getElementById('terminalOutput').innerHTML =
    '<div class="term-line term-system">&#9658; Initializing...</div>';
  document.getElementById('modalProgressFill').style.width = '0%';
  const pctEl = document.getElementById('modalProgressPct');
  if (pctEl) pctEl.textContent = '0%';
  document.getElementById('runModal').classList.remove('hidden');
}
function closeModal() {
  document.getElementById('runModal').classList.add('hidden');
}
document.getElementById('runModal').addEventListener('click', function (e) {
  if (e.target === this) closeModal();
});

// ─── Terminal Logger (for modal) ────────────────────────────
function termLog(msg, type = 'info', delay = 0) {
  return new Promise(resolve => {
    setTimeout(() => {
      const out = document.getElementById('terminalOutput');
      const line = document.createElement('div');
      line.className = `term-line term-${type}`;
      line.textContent = msg;
      out.appendChild(line);
      out.scrollTop = out.scrollHeight;
      resolve();
    }, delay);
  });
}

async function termLogSequence(lines) {
  let delay = 0;
  for (const [msg, type, d] of lines) {
    await termLog(msg, type, delay);
    delay += (d || 300);
  }
}

function setProgress(pct) {
  document.getElementById('modalProgressFill').style.width = `${pct}%`;
  const pctEl = document.getElementById('modalProgressPct');
  if (pctEl) pctEl.textContent = `${Math.round(pct)}%`;
}

function captureTerminalOutput() {
  const out = document.getElementById('terminalOutput');
  return out ? out.innerText : '';
}

// ─── SSE Stream Consumer ────────────────────────────────────
/**
 * Open a Server-Sent Events stream for the given job_id.
 * Pipes backend log lines into the modal terminal.
 * Resolves with the result object on job completion.
 */
function streamJob(jobId) {
  return new Promise((resolve, reject) => {
    const source = new EventSource(`/api/stream/${jobId}`);

    source.onmessage = (event) => {
      let msg;
      try { msg = JSON.parse(event.data); } catch { return; }

      if (msg.type === 'ping') return;

      if (msg.type === 'log') {
        const text = msg.text || '';
        let lineType = 'info';
        if (/✅|FOUND|complete|stored|uploaded|done/i.test(text)) lineType = 'success';
        else if (/❌|ERROR|error|Exception/i.test(text))             lineType = 'error';
        else if (/⚠️|Warning|WARNING/i.test(text))                   lineType = 'warning';
        else if (/📊|📌|📍|🎯|⭐|└─|Segment|Time:|Frame|Faces:/i.test(text)) lineType = 'data';
        else if (/^[=\-]{3}|MODE:|🚀|📦|🔍|🎬|👥|👤/i.test(text)) lineType = 'system';

        const out = document.getElementById('terminalOutput');
        const line = document.createElement('div');
        line.className  = `term-line term-${lineType}`;
        line.textContent = text;
        out.appendChild(line);
        out.scrollTop = out.scrollHeight;

      } else if (msg.type === 'progress') {
        setProgress(Math.min(98, msg.value));

      } else if (msg.type === 'done') {
        setProgress(100);
        source.close();
        resolve(msg.result || {});

      } else if (msg.type === 'error') {
        const out = document.getElementById('terminalOutput');
        const line = document.createElement('div');
        line.className  = 'term-line term-error';
        line.textContent = `❌ Backend error: ${msg.message}`;
        out.appendChild(line);
        out.scrollTop = out.scrollHeight;
        setProgress(100);
        source.close();
        reject(new Error(msg.message));
      }
    };

    source.onerror = () => {
      const out = document.getElementById('terminalOutput');
      const line = document.createElement('div');
      line.className  = 'term-line term-error';
      line.textContent = '❌ Connection lost. Is the backend running? Start with: python run.py';
      out.appendChild(line);
      out.scrollTop = out.scrollHeight;
      source.close();
      reject(new Error('SSE connection failed'));
    };
  });
}

/** Auto-fills a namespace text input from an uploaded file's name */
function fillNsFromFile(input, nsInputId) {
  const file = input.files[0];
  if (!file) return;
  const nsInput = document.getElementById(nsInputId);
  if (nsInput && !nsInput.value) nsInput.value = file.name;
}

/** POST to endpoint, stream SSE progress, return result. */
async function _startJob(url, formData, label) {
  try {
    const resp = await fetch(url, { method: 'POST', body: formData });
    if (!resp.ok) {
      let detail = `HTTP ${resp.status}`;
      try { const j = await resp.json(); detail = j.detail || detail; } catch {}
      throw new Error(detail);
    }
    const { job_id } = await resp.json();
    const result = await streamJob(job_id);
    addLogEntry(`${label} — completed`, 'success');
    loadStatus();    // refresh namespace counts
    return result;
  } catch (err) {
    await termLog(`❌ ${err.message}`, 'error');
    showToast(`${label} failed — see terminal`, 'error');
    addLogEntry(`${label} failed: ${err.message}`, 'error');
    return null;
  }
}

// ─── Run Mode Dispatcher ────────────────────────────────────
async function runMode(mode) {
  const modeLabels = {
    store:              'Store Faces',
    search:             'Search Person',
    batch_search:       'Batch Search',
    multi_video_search: 'Multi-Video Search',
    ultimate_search:    'Ultimate Search',
    bulk_store:         'Bulk Store',
  };
  openModal(`Running: ${modeLabels[mode] || mode}`);
  addLogEntry(`Launched mode: ${mode}`, 'info');
  state.jobsRun++;

  if      (mode === 'store')              await runStoreReal();
  else if (mode === 'search')             await runSearchReal();
  else if (mode === 'batch_search')       await runBatchReal();
  else if (mode === 'multi_video_search') await runMultiVideoReal();
  else if (mode === 'ultimate_search')    await runUltimateReal();
  else if (mode === 'bulk_store')         await runBulkStoreReal();
}

// ══ Store ═══════════════════════════════════════════════════
async function runStoreReal() {
  const tf        = tempFiles['video_store'];
  const ns        = document.getElementById('storeNamespace')?.value   || 'video_default';
  const frameSkip = document.getElementById('frameSkip')?.value        || 30;
  const minFace   = document.getElementById('minFaceSize')?.value      || 80;
  const maxFaces  = document.getElementById('maxFaces')?.value         || 500;
  const gpuBatch  = document.getElementById('gpuBatch')?.value         || 32;
  const confEl    = document.querySelector('input[name="conflict"]:checked');
  const conflict  = confEl ? confEl.value : 'skip';

  if (!tf) {
    await termLog('⚠️  No video uploaded. Please upload a video file first.', 'warning');
    setProgress(100); showToast('Upload a video first', 'warning'); return;
  }

  await termLog(`⚙️  MODE: STORE ALL FACES`, 'system');

  const fd = new FormData();
  fd.append('video', tf.file, tf.name);
  fd.append('namespace',     ns);
  fd.append('frame_skip',    frameSkip);
  fd.append('min_face_size', minFace);
  fd.append('max_faces',     maxFaces);
  fd.append('gpu_batch',     gpuBatch);
  fd.append('conflict',      conflict);

  const result = await _startJob('/api/store', fd, `Store → ${ns}`);
  if (result) {
    showToast(`✅ Faces stored in namespace '${ns}'`, 'success');
    storeResult({ type: 'store', ns, video: tf.name, timestamp: Date.now(), terminal: captureTerminalOutput() });
  }
}

// ══ Search ═══════════════════════════════════════════════════
async function runSearchReal() {
  const tf        = tempFiles['searchUploadZone'];
  const ns        = document.getElementById('searchNamespace')?.value    || 'video_peop';
  const threshold = document.getElementById('distThreshold')?.value      || '0.50';
  const topK      = document.getElementById('topK')?.value               || '100';
  const cluster   = document.getElementById('clusterThreshold')?.value   || '30';

  if (!tf) {
    await termLog('⚠️  No image uploaded. Please upload a reference photo first.', 'warning');
    setProgress(100); showToast('Upload a reference image first', 'warning'); return;
  }

  const fd = new FormData();
  fd.append('image',     tf.file, tf.name);
  fd.append('namespace', ns);
  fd.append('threshold', threshold);
  fd.append('top_k',     topK);
  fd.append('cluster',   cluster);

  const result = await _startJob('/api/search', fd, `Search in ${ns}`);
  if (result) {
    state.searchCount++;
    document.getElementById('stat-searches').textContent = state.searchCount;
    showToast('Search complete', 'success');
    storeResult({ type: 'search', img: tf.name, ns, timestamp: Date.now(), terminal: captureTerminalOutput() });
  }
}

// ══ Batch Search ══════════════════════════════════════════════
async function runBatchReal() {
  const people    = document.querySelectorAll('#batchPeopleList .person-item');
  const ns        = document.getElementById('batchNamespace')?.value   || 'video_peop';
  const threshold = document.getElementById('distThreshold')?.value    || '0.50';
  const topK      = document.getElementById('topK')?.value             || '100';
  const cluster   = document.getElementById('clusterThreshold')?.value || '30';

  const uploads = [];
  people.forEach(p => {
    const fi = p.querySelector('input[type="file"]');
    if (fi && tempFiles[fi.id]) uploads.push(tempFiles[fi.id]);
  });

  if (!uploads.length) {
    await termLog('⚠️  No photos uploaded. Upload a photo for each person first.', 'warning');
    setProgress(100); showToast('Upload photos for each person', 'warning'); return;
  }

  await termLog(`👥 MODE: BATCH SEARCH`, 'system');
  await termLog(`📊 ${uploads.length} people  →  Namespace: ${ns}`, 'info');

  const fd = new FormData();
  uploads.forEach(tf => fd.append('images', tf.file, tf.name));
  fd.append('namespace', ns);
  fd.append('threshold', threshold);
  fd.append('top_k',     topK);
  fd.append('cluster',   cluster);

  const result = await _startJob('/api/batch-search', fd, `Batch (${uploads.length} people)`);
  if (result) {
    showToast(`Batch search done — ${uploads.length} people`, 'success');
    state.searchCount++;
    document.getElementById('stat-searches').textContent = state.searchCount;
    storeResult({ type: 'batch_search', people: uploads.length, ns, timestamp: Date.now(), terminal: captureTerminalOutput() });
  }
}

// ══ Multi-Video Search ════════════════════════════════════════
async function runMultiVideoReal() {
  const tf        = tempFiles['mvUploadZone'];
  const threshold = document.getElementById('distThreshold')?.value    || '0.50';
  const topK      = document.getElementById('topK')?.value             || '100';
  const cluster   = document.getElementById('clusterThreshold')?.value || '30';

  if (!tf) {
    await termLog('⚠️  No reference image uploaded.', 'warning');
    setProgress(100); showToast('Upload a reference image first', 'warning'); return;
  }

  const videoNames = [];
  document.querySelectorAll('#multiVideoList .video-item').forEach(item => {
    const v = item.querySelector('input[type="text"]')?.value?.trim();
    if (v) videoNames.push(v);
  });

  if (!videoNames.length) {
    await termLog('⚠️  No video names specified (e.g. bahu_480.mp4).', 'warning');
    setProgress(100); showToast('Add at least one video name', 'warning'); return;
  }

  await termLog(`🎬 MODE: MULTI-VIDEO SEARCH`, 'system');

  const fd = new FormData();
  fd.append('image',       tf.file, tf.name);
  fd.append('video_names', JSON.stringify(videoNames));
  fd.append('threshold',   threshold);
  fd.append('top_k',       topK);
  fd.append('cluster',     cluster);

  const result = await _startJob('/api/multi-video-search', fd, `Multi-video (${videoNames.length})`);
  if (result) {
    showToast(`Searched ${videoNames.length} video(s)`, 'success');
    state.searchCount++;
    document.getElementById('stat-searches').textContent = state.searchCount;
    storeResult({ type: 'multi_video', img: tf.name, videos: videoNames.length, timestamp: Date.now(), terminal: captureTerminalOutput() });
  }
}

// ══ Ultimate Search ═══════════════════════════════════════════
async function runUltimateReal() {
  const threshold = document.getElementById('distThreshold')?.value    || '0.50';
  const topK      = document.getElementById('topK')?.value             || '100';
  const cluster   = document.getElementById('clusterThreshold')?.value || '30';

  // Collect person images
  const uploads = [];
  document.querySelectorAll('#ultimatePeopleList .person-item').forEach(p => {
    const fi = p.querySelector('input[type="file"]');
    if (fi && tempFiles[fi.id]) uploads.push(tempFiles[fi.id]);
  });

  // Collect video names/namespaces
  const videoNames = [];
  document.querySelectorAll('#ultimateVideoList .video-item').forEach(item => {
    const nsInput = item.querySelector('input[type="text"]');
    const val = nsInput?.value?.trim();
    if (val) {
      videoNames.push(val);
    } else {
      const fi = item.querySelector('input[type="file"]');
      if (fi && tempFiles[fi.id]) videoNames.push(tempFiles[fi.id].name);
    }
  });

  if (!uploads.length) {
    await termLog('⚠️  No person photos uploaded.', 'warning');
    setProgress(100); showToast('Upload photos for each person', 'warning'); return;
  }
  if (!videoNames.length) {
    await termLog('⚠️  No video namespaces specified.', 'warning');
    setProgress(100); showToast('Enter namespace names or upload videos', 'warning'); return;
  }

  const combos = uploads.length * videoNames.length;
  await termLog(`🚀 MODE: ULTIMATE SEARCH`, 'system');
  await termLog(`📊 ${uploads.length} people × ${videoNames.length} videos = ${combos} combos`, 'info');

  const fd = new FormData();
  uploads.forEach(tf => fd.append('images', tf.file, tf.name));
  fd.append('video_names', JSON.stringify(videoNames));
  fd.append('threshold',   threshold);
  fd.append('top_k',       topK);
  fd.append('cluster',     cluster);

  const result = await _startJob('/api/ultimate-search', fd, `Ultimate (${combos} combos)`);
  if (result) {
    showToast(`Ultimate done — ${combos} combinations`, 'success');
    state.searchCount++;
    document.getElementById('stat-searches').textContent = state.searchCount;
    storeResult({ type: 'ultimate_search', people: uploads.length, videos: videoNames.length, timestamp: Date.now(), terminal: captureTerminalOutput() });
  }
}

// ══ Bulk Store ════════════════════════════════════════════════
async function runBulkStoreReal() {
  const validFiles = [];
  document.querySelectorAll('#bulkVideoList .video-item').forEach(item => {
    const fi = item.querySelector('input[type="file"]');
    if (fi && tempFiles[fi.id]) validFiles.push(tempFiles[fi.id]);
  });

  if (!validFiles.length) {
    await termLog('⚠️  No videos uploaded. Please upload video files first.', 'warning');
    setProgress(100); showToast('Upload videos first', 'warning'); return;
  }

  await termLog(`📦 MODE: BULK STORE`, 'system');
  await termLog(`📊 Processing ${validFiles.length} video(s)…`, 'info');

  const fd = new FormData();
  validFiles.forEach(tf => fd.append('videos', tf.file, tf.name));

  const result = await _startJob('/api/bulk-store', fd, `Bulk store (${validFiles.length} videos)`);
  if (result) showToast(`Bulk stored ${validFiles.length} video(s)`, 'success');
}



// ─── (legacy simulation code – disabled, not called) ───────────
/* removed auto-executing IIFEs that were firing toasts on page load
(async () => {
  const tf = tempFiles['video_store'];
  const ns = document.getElementById('storeNamespace')?.value || 'video_peop';
  const frameSkip = document.getElementById('frameSkip')?.value || 30;

  if (!tf) {
    await termLog('⚠️  No video uploaded. Please upload a video file first.', 'warning', 0);
    setProgress(100);
    showToast('Upload a video first', 'warning');
    return;
  }

  const videoName = tf.name;

  await termLogSequence([
    [`🚀 Starting face detection with RetinaFace...`, 'info', 400],
  ]);
  setProgress(10);

  // Simulate frame-by-frame processing
  const totalFrames = 120;
  let storedFaces = 0;
  const batchSize = 10;
  const wait = ms => new Promise(r => setTimeout(r, ms));

  for (let i = 0; i < totalFrames; i += batchSize) {
    const batchEnd = Math.min(i + batchSize, totalFrames);
    const facesInBatch = Math.floor(Math.random() * 6) + 2; // 2–7 faces per batch
    storedFaces += facesInBatch;
    const frameNum = batchEnd * parseInt(frameSkip);
    const pct = 10 + ((batchEnd / totalFrames) * 80);

    await wait(350); // wait before each batch update
    const out = document.getElementById('terminalOutput');
    const line = document.createElement('div');
    line.className = 'term-line term-data';
    line.textContent = `   🖼️  Frames ${i * frameSkip}–${frameNum} → ${facesInBatch} face(s) detected & encoded`;
    out.appendChild(line);
    out.scrollTop = out.scrollHeight;

    setProgress(pct);
    document.getElementById('stat-stored').textContent = (state.storedFaces + storedFaces).toLocaleString();
  }

  await termLogSequence([
    [``, 'system', 100],
    [`✅ All ${storedFaces} vectors stored successfully!`, 'success', 600],
  ]);
  setProgress(100);
  state.storedFaces += storedFaces;
  document.getElementById('stat-stored').textContent = state.storedFaces.toLocaleString();
  showToast(`✅ Stored ${storedFaces} faces from ${videoName}`, 'success');
  addLogEntry(`Stored ${storedFaces} faces from ${videoName} → ${ns}`, 'success');
}); // end disabled store simulation


(async () => { // legacy simulation (not called, kept for reference)
  // Get temp file name or fallback
  const tf  = tempFiles['searchUploadZone'];
  const imgPath = tf ? tf.tempPath : '(no image uploaded)';
  const imgName = tf ? tf.name : 'reference.jpg';
  const ns  = document.getElementById('searchNamespace')?.value || 'video_peop';
  const thresh = document.getElementById('distThreshold')?.value || '0.50';
  const topK   = document.getElementById('topK')?.value || '100';

  if (!tf) {
    await termLog('⚠️  No image uploaded. Please upload a reference photo first.', 'warning', 0);
    setProgress(100);
    showToast('Upload a reference image first', 'warning');
    return;
  }

  await termLogSequence([
    [`📊 Namespace '${ns}' has 4820 stored faces`, 'data', 400],
    [`✅ Reference face encoded → 512-dim vector`, 'success', 600],
  ]);
  setProgress(40);

  await new Promise(r => setTimeout(r, 800));
  const searchMs = Math.floor(Math.random() * 80) + 60;
  await termLog(`✅ Search completed in ${searchMs}ms`, 'success', 0);
  setProgress(65);

  // Generate fake clusters
  const numClusters = Math.floor(Math.random() * 3) + 1;
  const clusters = [];
  for (let i = 0; i < numClusters; i++) {
    const startSec = parseFloat((Math.random() * 100 + 5).toFixed(2));
    const endSec = parseFloat((startSec + Math.random() * 12 + 2).toFixed(2));
    const dist = parseFloat((Math.random() * 0.2 + 0.15).toFixed(4));
    const conf = parseFloat((Math.random() * 0.3 + 0.7).toFixed(3));
    clusters.push({ startSec, endSec, dist, conf });
  }

  const matchLines = [
    [``, 'system', 200],
    [`📊 Search Results:`, 'info', 0],
    [`   - Matching frames: ${numClusters * 15 + Math.floor(Math.random() * 10)}`, 'data', 200],
    [`   - Appearance segments: ${numClusters}`, 'data', 200],
    [``, 'system', 200],
    [`✅ RESULT: Person IS PRESENT in the video`, 'success', 300],
  ];
  await termLogSequence(matchLines);

  for (let i = 0; i < clusters.length; i++) {
    const c = clusters[i];
    const quality = c.dist < 0.30 ? 'STRONG' : 'GOOD';
    await termLogSequence([
      [`\n   Segment ${i + 1}:`, 'data', 200],
      [`      📍 Time: ${c.startSec}s – ${c.endSec}s  (dur: ${(c.endSec - c.startSec).toFixed(2)}s)`, 'data', 150],
      [`      🎯 Best distance: ${c.dist}  |  Quality: ${quality}`, 'data', 150],
      [`      ⭐ Confidence: ${(c.conf * 100).toFixed(1)}%`, 'data', 150],
    ]);
  }
  setProgress(100);

  // Display results on page
  showSearchResults(clusters, imgPath, ns, searchMs);
  state.searchCount++;
  document.getElementById('stat-searches').textContent = state.searchCount;
  showToast(`Found ${numClusters} appearance segment(s)`, 'success');
  addLogEntry(`Search: ${imgPath} in '${ns}' → ${numClusters} segment(s)`, 'success');
}); // end disabled search simulation
*/

function showSearchResults(clusters, imgPath, ns, ms) {
  const section = document.getElementById('searchResults');
  const content = document.getElementById('searchResultsContent');
  section.classList.remove('hidden');

  const segmentsHtml = clusters.map((c, i) => `
    <div class="segment-item">
      <div class="seg-field"><label>Segment</label><span>#${i + 1}</span></div>
      <div class="seg-field"><label>Start</label><span>${c.startSec}s</span></div>
      <div class="seg-field"><label>End</label><span>${c.endSec}s</span></div>
      <div class="seg-field"><label>Duration</label><span>${(c.endSec - c.startSec).toFixed(2)}s</span></div>
      <div class="seg-field"><label>Distance</label><span>${c.dist}</span></div>
      <div class="seg-field"><label>Confidence</label><span>${(c.conf * 100).toFixed(1)}%</span></div>
    </div>
  `).join('');

  content.innerHTML = `
    <div class="result-card">
      <div class="result-card-header">
        <div>
          <strong style="color:var(--text-primary)">${imgPath}</strong>
          <span style="color:var(--text-muted);font-size:12px;margin-left:8px">in namespace: ${ns}</span>
        </div>
        <span class="result-badge-found"><i class="fa-solid fa-check"></i> FOUND · ${ms}ms</span>
      </div>
      <div class="segment-list">${segmentsHtml}</div>
    </div>
  `;

  storeResult({ type: 'search', img: imgPath, ns, clusters, ms });
}

// ── Batch Simulation ─────────────────────────────────────────
async function runBatchSimulation() {
  const people = document.querySelectorAll('#batchPeopleList .person-item');
  await termLogSequence([
    [`👥 MODE: BATCH SEARCH`, 'system', 0],
    [`📊 Searching for ${people.length} people...`, 'info', 400],
  ]);
  setProgress(10);

  let done = 0;
  for (let i = 0; i < people.length; i++) {
    const nameInput = people[i].querySelector('input[type="text"]');
    const fileInput = people[i].querySelector('input[type="file"]');
    const name = nameInput?.value || `Person ${i + 1}`;
    const tf   = fileInput ? tempFiles[fileInput.id] : null;

    if (!tf) {
      await termLog(`⚠️  ${name}: no photo uploaded — skipping`, 'warning', 300);
      continue;
    }

    await termLogSequence([
      [`\n🔍 Searching: ${name} (${tf.name})`, 'info', 300],
      [`   Temp path: ${tf.tempPath}`, 'data', 200],
      [`   Encoding reference face...`, 'data', 400],
      [`   Querying Pinecone...`, 'data', 500],
      [`   ✅ ${name}: ${Math.floor(Math.random() * 3) + 1} segment(s) found`, 'success', 400],
    ]);
    done++;
    setProgress(10 + ((i + 1) / people.length) * 85);
  }

  if (done === 0) {
    await termLog('⚠️  No images uploaded. Add photos to each person row.', 'warning', 0);
    showToast('Upload photos for each person first', 'warning');
  } else {
    await termLog(`🎉 Batch search complete for ${done} people!`, 'success', 300);
    showToast(`Batch search done for ${done} people`, 'success');
    addLogEntry(`Batch search: ${done} people searched`, 'success');
  }
  setProgress(100);
}

// ── Multi-Video Simulation ────────────────────────────────────
async function runMultiVideoSimulation() {
  const videos = document.querySelectorAll('#multiVideoList .video-item');
  const tf = tempFiles['mvUploadZone'];

  await termLogSequence([
    [`📊 Searching across ${videos.length} videos...`, 'info', 400],
  ]);

  if (!tf) {
    await termLog('⚠️  No reference image uploaded. Please upload a photo first.', 'warning', 300);
    setProgress(100);
    showToast('Upload a reference image first', 'warning');
    return;
  }

  await termLogSequence([
    [`📸 Reference person: ${tf.name}`, 'info', 300],
    [`   Temp path: ${tf.tempPath}`, 'data', 200],
    [`✅ Reference encoded (512-dim)`, 'success', 500],
  ]);
  setProgress(15);

  for (let i = 0; i < videos.length; i++) {
    const vidName = videos[i].querySelector('input')?.value || `video${i + 1}.mp4`;
    const ns = `video_${vidName.replace(/\.[^.]+$/, '')}`;
    const segs = Math.floor(Math.random() * 3);
    await termLogSequence([
      [`\n🎥 Searching in: ${vidName}`, 'info', 300],
      [`   Querying...`, 'data', 500],
      segs > 0
        ? [`   ✅ ${segs} segment(s) found`, 'success', 300]
        : [`   ⚠️  Not found in this video`, 'warning', 300],
    ]);
    setProgress(15 + ((i + 1) / videos.length) * 80);
  }

  await termLog(`🎉 Multi-video search complete!`, 'success', 400);
  setProgress(100);
  showToast(`Searched ${videos.length} videos`, 'success');
  addLogEntry(`Multi-video search: ${videos.length} videos`, 'success');
}

// ── Ultimate Simulation ───────────────────────────────────────
async function runUltimateSimulation() {
  const pItems = document.querySelectorAll('#ultimatePeopleList .person-item');
  const videos = document.querySelectorAll('#ultimateVideoList .video-item');
  const vCount = videos.length;

  // Collect only people with uploaded photos
  const validPeople = [];
  pItems.forEach(p => {
    const nameInput = p.querySelector('input[type="text"]');
    const fileInput = p.querySelector('input[type="file"]');
    const tf = fileInput ? tempFiles[fileInput.id] : null;
    if (tf) validPeople.push({ name: nameInput?.value || 'Person', tf });
  });

  const combos = validPeople.length * vCount;

  await termLogSequence([
    [`🚀 MODE: ULTIMATE SEARCH`, 'system', 0],
    [`📊 ${validPeople.length} people × ${vCount} videos = ${combos} combinations`, 'info', 400],
  ]);

  if (validPeople.length === 0) {
    await termLog('⚠️  No photos uploaded. Upload a photo for each person first.', 'warning', 200);
    setProgress(100);
    showToast('Upload photos for each person first', 'warning');
    return;
  }

  await termLog(`✅ ${validPeople.length} reference face(s) encoded`, 'success', 500);
  setProgress(5);

  let done = 0;
  for (const person of validPeople) {
    await termLog(`\n👤 Processing: ${person.name} (${person.tf.name})`, 'info', 200);
    for (const video of videos) {
      const vn = video.querySelector('input')?.value || 'video.mp4';
      const segs = Math.floor(Math.random() * 3);
      await termLog(
        `   ${vn} → ${segs > 0 ? '✅ ' + segs + ' segment(s)' : '⚠️  not found'}`,
        segs > 0 ? 'success' : 'warning',
        350
      );
      done++;
      setProgress(5 + (done / Math.max(combos, 1)) * 90);
    }
  }

  await termLog(`🏆 Ultimate search complete! ${combos} combinations checked.`, 'success', 400);
  setProgress(100);
  showToast(`Ultimate search done — ${combos} combinations scanned`, 'success');
  addLogEntry(`Ultimate search: ${validPeople.length}p × ${vCount}v`, 'success');
}

// ─── Demo Modal ────────────────────────────────────────────
function closeDemoModal() {
  document.getElementById('demoModal').classList.add('hidden');
}
document.getElementById('demoModal')?.addEventListener('click', function(e) {
  if (e.target === this) closeDemoModal();
});

function setDemoStep(idx, state) {
  // state: 'loading' | 'done' | 'error'
  const step = document.getElementById(`dstep-${idx}`);
  if (!step) return;
  const statusEl = step.querySelector('.ds-status');
  step.className = `demo-step ds-${state}`;
  if (state === 'loading') {
    statusEl.innerHTML = '<span class="ds-loading"><span class="ds-dot"></span><span class="ds-dot"></span><span class="ds-dot"></span></span>';
  } else if (state === 'done') {
    statusEl.innerHTML = '<i class="fa-solid fa-circle-check ds-check"></i>';
  } else if (state === 'error') {
    statusEl.innerHTML = '<i class="fa-solid fa-circle-xmark ds-error"></i>';
  }
}

function setDemoProgress(pct) {
  document.getElementById('demoProgressFill').style.width = `${pct}%`;
}

function setDemoLabels(headline, sub) {
  const h = document.getElementById('demoHeadline');
  const s = document.getElementById('demoSubline');
  if (h) { h.style.animation = 'none'; h.offsetWidth; h.style.animation = ''; h.textContent = headline; }
  if (s) { s.textContent = sub; }
}

function animateBars() {
  document.querySelectorAll('.dr-bar-fill').forEach(bar => {
    const target = bar.dataset.w;
    bar.style.transition = 'width 1s cubic-bezier(0.4,0,0.2,1)';
    setTimeout(() => { bar.style.width = target + '%'; }, 80);
  });
}

async function simulateProcessing() {
  // Reset
  const modal = document.getElementById('demoModal');
  const result = document.getElementById('demoResult');
  result.classList.add('hidden');
  document.querySelectorAll('.dr-bar-fill').forEach(b => b.style.width = '0%');
  for (let i = 0; i < 8; i++) {
    const step = document.getElementById(`dstep-${i}`);
    if (step) {
      step.className = 'demo-step';
      step.querySelector('.ds-status').innerHTML = '<span class="ds-idle">—</span>';
    }
  }
  setDemoProgress(0);
  setDemoLabels('Hybrid Face Recognition', 'Initializing pipeline...');
  document.getElementById('demoEyeWrap').className = 'demo-eye-wrap';
  modal.classList.remove('hidden');

  const wait = ms => new Promise(r => setTimeout(r, ms));

  const steps = [
    { label: 'Loading FaceNet model', sub: 'InceptionResnetV1 · VECTOR_DIM=512', dur: 900 },
    { label: 'Connecting to Pinecone', sub: 'Vector index — namespace ready', dur: 700 },
    { label: 'Extracting video frames', sub: 'peop.mp4 · 120s · 30fps · skip=30', dur: 800 },
    { label: 'Detecting faces', sub: 'RetinaFace · 10 faces found', dur: 900 },
    { label: 'Encoding embeddings', sub: 'GPU batch · 10 × 512-dim vectors', dur: 800 },
    { label: 'Uploading to Pinecone', sub: '10 vectors → namespace video_demo', dur: 700 },
    { label: 'Clustering matches', sub: '2 appearance segments detected', dur: 500 },
  ];

  for (let i = 0; i < steps.length; i++) {
    const s = steps[i];
    setDemoLabels(s.label, s.sub);
    setDemoStep(i, 'loading');
    setDemoProgress((i / steps.length) * 90);
    await wait(s.dur);
    setDemoStep(i, 'done');
    await wait(120);
  }

  setDemoProgress(100);
  setDemoLabels('Pipeline Complete', 'Person found in 2 segments · 87ms');
  document.getElementById('demoEyeWrap').className = 'demo-eye-wrap eye-done';

  await wait(300);
  result.classList.remove('hidden');
  requestAnimationFrame(() => requestAnimationFrame(animateBars));

  showToast('Demo complete — person found!', 'success');
  addLogEntry('Demo run — 2 appearance segments found', 'success');
}



// ─── Results Page ────────────────────────────────────────────
function storeResult(result) {
  state.results.unshift(result);
  renderResultsPage();
}

const _TYPE_META = {
  search:         { label: '🔍 Search',        color: 'var(--cyan)'   },
  batch_search:   { label: '👥 Batch Search',   color: 'var(--purple)' },
  multi_video:    { label: '🎬 Multi-Video',    color: 'var(--green)'  },
  ultimate_search:{ label: '🚀 Ultimate',       color: 'var(--orange)' },
  store:          { label: '📦 Store',          color: '#60a5fa'       },
};

function _resultSummary(r) {
  if (r.type === 'search')          return `${r.img || ''}  →  namespace: ${r.ns || ''}`;
  if (r.type === 'batch_search')    return `${r.people || '?'} people  →  namespace: ${r.ns || ''}`;
  if (r.type === 'multi_video')     return `${r.img || ''}  →  ${r.videos || '?'} video(s)`;
  if (r.type === 'ultimate_search') return `${r.people || '?'} people × ${r.videos || '?'} videos`;
  if (r.type === 'store')           return `${r.video || ''}  →  namespace: ${r.ns || ''}`;
  return '';
}

function _escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function renderResultsPage() {
  const empty = document.getElementById('resultsEmpty');
  const container = document.getElementById('resultsContainer');
  if (!container) return;

  if (state.results.length === 0) {
    if (empty) empty.style.display = 'block';
    container.innerHTML = '';
    return;
  }
  if (empty) empty.style.display = 'none';

  container.innerHTML = state.results.map((r, idx) => {
    const meta    = _TYPE_META[r.type] || { label: r.type, color: 'var(--cyan)' };
    const summary = _resultSummary(r);
    const ts      = r.timestamp ? new Date(r.timestamp).toLocaleTimeString() : '';
    const termTxt = _escHtml(r.terminal || '(no output captured)');
    return `
      <div class="result-card" id="rc-${idx}">
        <div class="result-card-header">
          <div style="display:flex;flex-direction:column;gap:3px">
            <div>
              <strong style="color:${meta.color}">${meta.label}</strong>
              ${ts ? `<span style="color:var(--text-muted);font-size:11px;margin-left:10px">${ts}</span>` : ''}
            </div>
            ${summary ? `<span style="color:var(--text-secondary);font-size:12px">${_escHtml(summary)}</span>` : ''}
          </div>
          <div style="display:flex;gap:6px;align-items:center;flex-shrink:0">
            <button onclick="downloadResult(${idx})" title="Download as .txt"
              style="background:rgba(0,212,255,0.1);border:1px solid rgba(0,212,255,0.3);color:var(--cyan);padding:5px 11px;border-radius:7px;cursor:pointer;font-size:12px">
              <i class="fa-solid fa-download"></i> Download
            </button>
            <button onclick="removeResult(${idx})" title="Remove this result"
              style="background:rgba(255,60,60,0.1);border:1px solid rgba(255,60,60,0.3);color:#f87171;padding:5px 9px;border-radius:7px;cursor:pointer;font-size:12px">
              <i class="fa-solid fa-xmark"></i>
            </button>
          </div>
        </div>
        <pre style="background:rgba(0,0,0,0.35);border:1px solid rgba(255,255,255,0.07);border-radius:8px;padding:12px;font-size:11.5px;line-height:1.6;color:var(--text-secondary);max-height:240px;overflow-y:auto;white-space:pre-wrap;word-break:break-word;margin-top:10px;font-family:'Courier New',monospace">${termTxt}</pre>
      </div>`;
  }).join('');
}

function downloadResult(idx) {
  const r = state.results[idx];
  if (!r) return;
  const meta  = _TYPE_META[r.type] || { label: r.type };
  const label = meta.label.replace(/[^a-z0-9]/gi, '_').toLowerCase();
  const ts    = r.timestamp ? new Date(r.timestamp).toISOString().replace(/[:.]/g, '-').slice(0, 19) : Date.now();
  const header = [
    'HYBRID FACE RECOGNITION — ' + meta.label.toUpperCase() + ' RESULT',
    'Time    : ' + (r.timestamp ? new Date(r.timestamp).toLocaleString() : 'unknown'),
    'Summary : ' + _resultSummary(r),
    '='.repeat(60),
    '',
  ].join('\n');
  const blob = new Blob([header + (r.terminal || '(no output)')], { type: 'text/plain' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `hfr_${label}_${ts}.txt`;
  a.click();
  showToast('Downloaded as .txt', 'success');
}

function removeResult(idx) {
  state.results.splice(idx, 1);
  renderResultsPage();
}

function clearResults() {
  state.results = [];
  renderResultsPage();
  showToast('Results cleared', 'info');
}

function exportResults() {
  // Build a single combined text file of all results
  const allText = state.results.map((r, i) => {
    const meta = _TYPE_META[r.type] || { label: r.type };
    return [
      `${'='.repeat(60)}`,
      `Result #${i + 1} — ${meta.label}`,
      `Time    : ${r.timestamp ? new Date(r.timestamp).toLocaleString() : 'unknown'}`,
      `Summary : ${_resultSummary(r)}`,
      `${'='.repeat(60)}`,
      r.terminal || '(no output)',
      '',
    ].join('\n');
  }).join('\n');
  const blob = new Blob([allText], { type: 'text/plain' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `hfr_all_results_${Date.now()}.txt`;
  a.click();
  showToast(`Exported ${state.results.length} result(s) as .txt`, 'success');
}

// ─── Password Toggle ─────────────────────────────────────────
function togglePassword(inputId, btn) {
  const input = document.getElementById(inputId);
  if (!input) return;
  const isPass = input.type === 'password';
  input.type = isPass ? 'text' : 'password';
  btn.innerHTML = isPass ? '<i class="fa-solid fa-eye-slash"></i>' : '<i class="fa-solid fa-eye"></i>';
}

// ─── Activity Log ──────────────────────────────────────────────
function addLogEntry(msg, type = 'info') {
  console.log(`[${type.toUpperCase()}] ${msg}`);
}

// ─── Init ────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  navigateTo('dashboard');
  renderResultsPage();
  updateUltimateCombos();

  // Update ultimate combos on input change
  document.getElementById('ultimatePeopleList')?.addEventListener('input', updateUltimateCombos);
  document.getElementById('ultimateVideoList')?.addEventListener('input', updateUltimateCombos);

  // Connect to backend
  addLogEntry('Connecting to backend…', 'info');
  const status = await loadStatus();
  if (status && status.ok) {
    const nsCount = Object.keys(status.namespaces || {}).length;
    addLogEntry(`Backend online · ${status.device.toUpperCase()} · ${nsCount} namespace(s) found`, 'success');
  } else {
    addLogEntry('Backend unreachable — run: python run.py', 'warning');
  }
});

// ─── Keyboard shortcut: Escape closes modal ─────────────────
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeModal();
});
