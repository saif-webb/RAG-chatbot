/* RAG Chatbot frontend — no framework, no build step.
 *
 * Two things here are load-bearing rather than cosmetic:
 *
 * 1. Document polling. Ingestion runs as a background task on the server, and
 *    free hosts spin down when there is no *inbound* traffic — a background job
 *    does not count. Polling for status keeps the instance awake long enough to
 *    finish embedding, as well as driving the UI.
 * 2. SSE over fetch. EventSource cannot issue a POST, so the stream is read
 *    from the response body and parsed by hand.
 */

const api = {
  health:    '/healthz',
  documents: '/api/documents',
  chat:      '/api/chat',
};

const POLL_ACTIVE_MS = 2000;   // while something is ingesting
const POLL_IDLE_MS   = 20000;  // otherwise, just to catch outside changes
const COLD_START_MS  = 3500;   // when to admit the server is waking up

const el = {
  uploadForm:  document.getElementById('upload-form'),
  fileInput:   document.getElementById('file-input'),
  uploadError: document.getElementById('upload-error'),
  docList:     document.getElementById('doc-list'),
  docCount:    document.getElementById('doc-count'),
  docEmpty:    document.getElementById('doc-empty'),
  health:      document.getElementById('health'),
  transcript:  document.getElementById('transcript'),
  chatForm:    document.getElementById('chat-form'),
  chatInput:   document.getElementById('chat-input'),
  sendBtn:     document.getElementById('send-btn'),
};

let conversationId = null;
let pollTimer = null;
let sending = false;

/* ── Documents ─────────────────────────────────────────────────────────── */

async function refreshDocuments() {
  clearTimeout(pollTimer);
  let documents = [];

  try {
    const response = await fetch(api.documents);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    ({ documents } = await response.json());
    renderDocuments(documents);
  } catch (error) {
    console.warn('Could not load documents:', error);
  }

  const busy = documents.some(d => d.status === 'pending' || d.status === 'processing');
  pollTimer = setTimeout(refreshDocuments, busy ? POLL_ACTIVE_MS : POLL_IDLE_MS);
}

function renderDocuments(documents) {
  el.docCount.textContent = documents.length;
  el.docEmpty.hidden = documents.length > 0;
  el.docList.replaceChildren(...documents.map(documentRow));
}

function documentRow(doc) {
  const li = document.createElement('li');
  li.className = 'doc';

  const name = document.createElement('span');
  name.className = 'doc-name';
  name.textContent = doc.filename;
  name.title = doc.filename;

  const status = document.createElement('span');
  status.className = `doc-status ${doc.status}`;
  if (doc.status === 'pending' || doc.status === 'processing') {
    status.innerHTML = '<span class="spinner"></span>';
  }
  status.append(doc.status);

  const meta = document.createElement('span');
  meta.className = 'doc-meta';
  if (doc.status === 'failed') {
    meta.classList.add('failed');
    meta.textContent = doc.error || 'Ingestion failed.';
  } else if (doc.status === 'ready') {
    meta.textContent = `${doc.chunk_count} chunks · ${formatSize(doc.size_bytes)}`;
  } else {
    meta.textContent = `${formatSize(doc.size_bytes)} · reading…`;
  }

  const remove = document.createElement('button');
  remove.className = 'doc-delete';
  remove.type = 'button';
  remove.textContent = 'Delete';
  remove.addEventListener('click', () => deleteDocument(doc));

  li.append(name, status, meta, remove);
  return li;
}

async function deleteDocument(doc) {
  if (!confirm(`Delete "${doc.filename}" and everything indexed from it?`)) return;
  try {
    const response = await fetch(`${api.documents}/${doc.id}`, { method: 'DELETE' });
    if (!response.ok) throw new Error(await errorMessage(response));
    refreshDocuments();
  } catch (error) {
    showUploadError(`Could not delete "${doc.filename}": ${error.message}`);
  }
}

async function uploadFiles(files) {
  hideUploadError();
  const problems = [];

  for (const file of files) {
    const body = new FormData();
    body.append('file', file);
    try {
      const response = await fetch(api.documents, { method: 'POST', body });
      if (!response.ok) throw new Error(await errorMessage(response));
      const doc = await response.json();
      if (doc.duplicate) {
        problems.push(`"${file.name}" was already uploaded — reusing the existing index.`);
      }
    } catch (error) {
      problems.push(`"${file.name}": ${error.message}`);
    }
  }

  if (problems.length) showUploadError(problems.join('\n'));
  refreshDocuments();
}

function showUploadError(message) {
  el.uploadError.textContent = message;
  el.uploadError.hidden = false;
}

function hideUploadError() {
  el.uploadError.hidden = true;
}

/* ── Chat ──────────────────────────────────────────────────────────────── */

async function sendMessage(text) {
  if (sending) return;
  sending = true;
  el.sendBtn.disabled = true;

  document.querySelector('.welcome')?.remove();
  addMessage('user', text);

  const bubble = addMessage('assistant', '');
  bubble.innerHTML = '<span class="typing"><i></i><i></i><i></i></span>';

  const coldStart = setTimeout(() => {
    if (!bubble.dataset.started) {
      bubble.innerHTML =
        '<span class="typing"><i></i><i></i><i></i></span> ' +
        '<span class="muted small">waking the server, this can take a minute…</span>';
    }
  }, COLD_START_MS);

  let answer = '';
  let citations = [];
  let grounded = true;

  try {
    const response = await fetch(api.chat, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, conversation_id: conversationId }),
    });
    if (!response.ok) throw new Error(await errorMessage(response));

    for await (const event of readEvents(response)) {
      clearTimeout(coldStart);
      switch (event.type) {
        case 'meta':
          if (event.data.conversation_id) conversationId = event.data.conversation_id;
          break;
        case 'citations':
          citations = event.data;
          break;
        case 'token':
          bubble.dataset.started = '1';
          answer += event.data.text;
          bubble.innerHTML = renderAnswer(answer);
          scrollToBottom();
          break;
        case 'done':
          grounded = event.data.grounded !== false;
          break;
        case 'error':
          throw new Error(event.data.message || 'The server could not answer.');
      }
    }

    if (!answer.trim()) throw new Error('The model returned an empty answer.');

    bubble.innerHTML = renderAnswer(answer);
    if (!grounded) bubble.classList.add('refusal');
    if (citations.length) bubble.closest('.msg').append(renderSources(citations));
  } catch (error) {
    bubble.classList.add('error');
    bubble.textContent = error.message;
  } finally {
    clearTimeout(coldStart);
    sending = false;
    el.sendBtn.disabled = false;
    scrollToBottom();
    el.chatInput.focus();
  }
}

/** Parse an SSE body from a fetch response into {type, data} objects. */
async function* readEvents(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // Events are separated by a blank line; the trailing fragment stays in the
    // buffer until its terminator arrives.
    const blocks = buffer.split('\n\n');
    buffer = blocks.pop();

    for (const block of blocks) {
      let type = 'message';
      const data = [];
      for (const line of block.split('\n')) {
        if (line.startsWith('event:')) type = line.slice(6).trim();
        else if (line.startsWith('data:')) data.push(line.slice(5).trim());
      }
      if (!data.length) continue;
      try {
        yield { type, data: JSON.parse(data.join('\n')) };
      } catch {
        console.warn('Skipping unparseable SSE block:', block);
      }
    }
  }
}

function addMessage(role, text) {
  const wrapper = document.createElement('div');
  wrapper.className = `msg ${role}`;

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;

  wrapper.append(bubble);
  el.transcript.append(wrapper);
  scrollToBottom();
  return bubble;
}

function renderSources(citations) {
  const box = document.createElement('div');
  box.className = 'sources';

  const heading = document.createElement('div');
  heading.className = 'sources-head';
  heading.textContent = `${citations.length} source${citations.length === 1 ? '' : 's'}`;
  box.append(heading);

  for (const citation of citations) {
    const details = document.createElement('details');
    details.className = 'source';
    details.dataset.index = citation.index;

    const summary = document.createElement('summary');
    const number = document.createElement('span');
    number.className = 'source-num';
    number.textContent = citation.index;

    const file = document.createElement('span');
    file.className = 'source-file';
    file.textContent = citation.page != null
      ? `${citation.filename} · page ${citation.page}`
      : citation.filename;

    const score = document.createElement('span');
    score.className = 'source-score';
    score.textContent = citation.score.toFixed(2);
    score.title = 'Cosine similarity to your question';

    summary.append(number, file, score);

    const body = document.createElement('p');
    body.className = 'source-body';
    body.textContent = citation.snippet;

    details.append(summary, body);
    box.append(details);
  }
  return box;
}

/** Escape, then apply the small subset of formatting the model actually emits. */
function renderAnswer(text) {
  const escaped = text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  return escaped
    .replace(/`([^`\n]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\[(\d+)\]/g, '<span class="cite-ref" data-ref="$1">$1</span>');
}

/* ── Wiring ────────────────────────────────────────────────────────────── */

el.uploadForm.addEventListener('click', () => el.fileInput.click());
el.uploadForm.addEventListener('keydown', event => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    el.fileInput.click();
  }
});
el.uploadForm.addEventListener('submit', event => event.preventDefault());

el.fileInput.addEventListener('change', () => {
  if (el.fileInput.files.length) uploadFiles([...el.fileInput.files]);
  el.fileInput.value = '';
});

for (const type of ['dragenter', 'dragover']) {
  el.uploadForm.addEventListener(type, event => {
    event.preventDefault();
    el.uploadForm.classList.add('dragover');
  });
}
for (const type of ['dragleave', 'drop']) {
  el.uploadForm.addEventListener(type, event => {
    event.preventDefault();
    el.uploadForm.classList.remove('dragover');
  });
}
el.uploadForm.addEventListener('drop', event => {
  const files = [...(event.dataTransfer?.files ?? [])];
  if (files.length) uploadFiles(files);
});

el.chatForm.addEventListener('submit', event => {
  event.preventDefault();
  const text = el.chatInput.value.trim();
  if (!text) return;
  el.chatInput.value = '';
  autosize();
  sendMessage(text);
});

el.chatInput.addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    el.chatForm.requestSubmit();
  }
});
el.chatInput.addEventListener('input', autosize);

// Clicking a [n] marker in an answer reveals the passage it refers to.
el.transcript.addEventListener('click', event => {
  const ref = event.target.closest('.cite-ref');
  if (!ref) return;
  const sources = ref.closest('.msg')?.querySelector('.sources');
  const target = sources?.querySelector(`.source[data-index="${ref.dataset.ref}"]`);
  if (!target) return;
  target.open = true;
  target.classList.remove('flash');
  void target.offsetWidth; // restart the animation
  target.classList.add('flash');
  target.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
});

/* ── Helpers ───────────────────────────────────────────────────────────── */

function autosize() {
  el.chatInput.style.height = 'auto';
  el.chatInput.style.height = `${Math.min(el.chatInput.scrollHeight, 180)}px`;
}

function scrollToBottom() {
  el.transcript.scrollTop = el.transcript.scrollHeight;
}

function formatSize(bytes) {
  if (!bytes) return '0 KB';
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

async function errorMessage(response) {
  try {
    const body = await response.json();
    return body.detail || `HTTP ${response.status}`;
  } catch {
    return `HTTP ${response.status}`;
  }
}

async function checkHealth() {
  try {
    const response = await fetch(api.health);
    const body = await response.json();
    // Model ids are namespaced ("sentence-transformers/all-MiniLM-L6-v2"); only
    // the last segment is worth the space in a footer line.
    const model = String(body.embeddings ?? '').split('/').pop();
    el.health.textContent = response.ok
      ? `${body.store} · ${body.llm} · ${model} (${body.embed_dim}d)`
      : `Backend degraded: ${body.detail}`;
  } catch {
    el.health.textContent = 'Backend unreachable.';
  }
}

checkHealth();
refreshDocuments();
autosize();
el.chatInput.focus();
