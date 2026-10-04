const $ = (id) => document.getElementById(id);

// ---- Settings ----
const apiKeyInput = $('apiKey');
const apiUrlInput = $('apiUrl');
const modeSelect = $('retrievalMode');
apiKeyInput.value = localStorage.getItem('documind_api_key') || '';
apiUrlInput.value = localStorage.getItem('documind_api_url') || 'http://localhost:4500';
apiKeyInput.addEventListener('change', () => localStorage.setItem('documind_api_key', apiKeyInput.value));
apiUrlInput.addEventListener('change', () => localStorage.setItem('documind_api_url', apiUrlInput.value));
modeSelect.value = localStorage.getItem('documind_mode') || 'hybrid';
modeSelect.addEventListener('change', () => localStorage.setItem('documind_mode', modeSelect.value));

function apiBase() { return apiUrlInput.value.replace(/\/$/, ''); }
function apiHeaders(extra = {}) {
  return apiKeyInput.value ? { 'x-gemini-key': apiKeyInput.value, ...extra } : extra;
}

// ---- Documents ----
const fileInput = $('fileInput');
const uploadBtn = $('uploadBtn');
const uploadStatus = $('uploadStatus');
const docsList = $('docsList');
const docsEmpty = $('docsEmpty');
const docCount = $('docCount');

let documents = [];

async function loadDocuments() {
  try {
    const res = await fetch(`${apiBase()}/api/documents`, { headers: apiHeaders() });
    const data = await res.json();
    documents = data.documents || [];
    renderDocuments();
  } catch {
    docsEmpty.textContent = 'Could not reach documind-api. Is it running?';
    docsEmpty.hidden = false;
  }
}

function renderDocuments() {
  docCount.textContent = documents.length ? `(${documents.length})` : '';
  docsEmpty.hidden = documents.length > 0;
  docsList.innerHTML = documents.map((d) => `
    <div class="doc-item">
      <input type="checkbox" class="doc-check" data-id="${d.id}" checked />
      <label>${escapeHtml(d.name)}<div class="doc-meta">${d.chunkCount} chunks</div></label>
      <button class="del-btn" data-id="${d.id}" title="Delete">&times;</button>
    </div>
  `).join('');

  docsList.querySelectorAll('.del-btn').forEach((btn) => {
    btn.addEventListener('click', async () => {
      await fetch(`${apiBase()}/api/documents/${btn.dataset.id}`, { method: 'DELETE', headers: apiHeaders() });
      await loadDocuments();
    });
  });
}

function selectedDocIds() {
  return [...docsList.querySelectorAll('.doc-check:checked')].map((el) => el.dataset.id);
}

uploadBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', async () => {
  const file = fileInput.files?.[0];
  if (!file) return;

  uploadStatus.hidden = false;
  uploadStatus.className = 'upload-status';
  uploadStatus.innerHTML = '<span class="spinner"></span>Chunking & embedding...';

  try {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${apiBase()}/api/documents`, { method: 'POST', headers: apiHeaders(), body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Upload failed.');

    uploadStatus.className = 'upload-status ok';
    uploadStatus.textContent = `Indexed "${data.name}" (${data.chunkCount} chunks).`;
    await loadDocuments();
  } catch (err) {
    uploadStatus.className = 'upload-status error';
    uploadStatus.textContent = err.message || 'Upload failed.';
  } finally {
    fileInput.value = '';
  }
});

loadDocuments();

// ---- Q&A ----
const chatLog = $('chatLog');
const questionInput = $('questionInput');
const askBtn = $('askBtn');
const askError = $('askError');

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function linkifyCitations(text) {
  return escapeHtml(text).replace(/\[(\d+)\]/g, '<span class="cite" data-n="$1">[$1]</span>');
}

function evalBadgeClass(score) {
  if (score === null || score === undefined) return 'mid';
  if (score >= 80) return 'high';
  if (score >= 50) return 'mid';
  return 'low';
}

let qaCounter = 0;
askBtn.addEventListener('click', handleAsk);
questionInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleAsk();
});

async function handleAsk() {
  const question = questionInput.value.trim();
  if (!question) return;

  askError.hidden = true;
  askBtn.disabled = true;
  questionInput.value = '';

  const block = document.createElement('div');
  block.className = 'qa-block';
  const blockId = `qa${++qaCounter}`;
  block.innerHTML = `
    <div class="q-bubble">${escapeHtml(question)}</div>
    <div class="a-card">
      <div class="a-text"><span class="spinner"></span>Thinking...</div>
      <div class="sources-wrap" hidden></div>
    </div>
  `;
  chatLog.appendChild(block);
  chatLog.scrollTop = chatLog.scrollHeight;

  const aText = block.querySelector('.a-text');
  const sourcesWrap = block.querySelector('.sources-wrap');
  let answerSoFar = '';
  let sourcesData = [];

  try {
    const res = await fetch(`${apiBase()}/api/query`, {
      method: 'POST',
      headers: apiHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ question, documentIds: selectedDocIds(), mode: modeSelect.value }),
    });

    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || 'Request failed.');
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let firstDelta = true;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const chunks = buffer.split('\n\n');
      buffer = chunks.pop();

      for (const raw of chunks) {
        const eventMatch = raw.match(/^event: (.+)$/m);
        const dataMatch = raw.match(/^data: (.+)$/m);
        if (!eventMatch || !dataMatch) continue;
        const eventName = eventMatch[1];
        const payload = JSON.parse(dataMatch[1]);

        if (eventName === 'sources') {
          sourcesData = payload;
          if (payload.length) {
            sourcesWrap.hidden = false;
            sourcesWrap.innerHTML = `<p class="sources-title">Sources</p>` + payload.map((s) => `
              <div class="source-item" id="${blockId}-src-${s.n}">
                <div class="source-head"><span>[${s.n}] ${escapeHtml(s.docName)}</span><span class="source-score">${s.similarity !== null && s.similarity !== undefined ? `similarity ${s.similarity}` : `relevance ${s.score}`}</span></div>
                <p>${escapeHtml(s.text.slice(0, 220))}${s.text.length > 220 ? '…' : ''}</p>
              </div>
            `).join('');
          }
        } else if (eventName === 'delta') {
          if (firstDelta) { aText.innerHTML = ''; firstDelta = false; }
          answerSoFar += payload.text;
          aText.innerHTML = linkifyCitations(answerSoFar);
          chatLog.scrollTop = chatLog.scrollHeight;
        } else if (eventName === 'eval') {
          const cls = evalBadgeClass(payload.faithfulness);
          const badgeWrap = document.createElement('div');
          badgeWrap.innerHTML = `
            <span class="eval-badge ${cls}">Faithfulness: ${payload.faithfulness === null || payload.faithfulness === undefined ? 'n/a' : payload.faithfulness + '%'}</span>
            ${payload.fromCache ? '<span class="eval-badge cache">from cache</span>' : ''}
            ${payload.notes ? `<div class="eval-notes">${escapeHtml(payload.notes)}</div>` : ''}
            ${Array.isArray(payload.unsupported_claims) && payload.unsupported_claims.length ? `<div class="eval-notes">Unsupported: ${payload.unsupported_claims.map(escapeHtml).join('; ')}</div>` : ''}
          `;
          block.querySelector('.a-card').appendChild(badgeWrap);
        } else if (eventName === 'error') {
          throw new Error(payload.error || 'Something went wrong mid-stream.');
        }
      }
    }

    // wire up citation clicks now that sources are in the DOM
    aText.querySelectorAll('.cite').forEach((el) => {
      el.addEventListener('click', () => {
        const target = document.getElementById(`${blockId}-src-${el.dataset.n}`);
        if (target) target.scrollIntoView({ behavior: 'smooth', block: 'center' });
      });
    });
  } catch (err) {
    aText.textContent = '';
    askError.hidden = false;
    askError.textContent = err.message || 'Something went wrong.';
    block.remove();
  } finally {
    askBtn.disabled = false;
  }
}
