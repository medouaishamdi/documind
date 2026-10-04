// Test doubles: a fake model client (deterministic, offline) and a server on a random port.
import { createApp } from '../src/app.js';
import { createStore } from '../src/store.js';
import { tokenize } from '../src/bm25.js';

// Bag-of-words hashed into 64 dimensions: texts sharing words get similar vectors.
export function fakeEmbedding(text, dim = 64) {
  const v = new Array(dim).fill(0);
  for (const t of tokenize(text)) {
    let h = 0;
    for (const ch of t) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
    v[h % dim] += 1;
  }
  return v;
}

export function fakeLlm({ answer = 'The refund window is 30 days [1].', judge = { faithfulness: 95, unsupported_claims: [], notes: 'ok' }, failEmbed = null } = {}) {
  const calls = { embed: 0, generate: 0, judge: 0 };
  return {
    calls,
    async batchEmbed({ texts }) {
      calls.embed += 1;
      if (failEmbed) throw failEmbed;
      return texts.map((t) => fakeEmbedding(t));
    },
    async streamGenerate({ onDelta }) {
      calls.generate += 1;
      for (const piece of answer.match(/.{1,8}/gs)) onDelta(piece);
      return answer;
    },
    async generateJson() {
      calls.judge += 1;
      if (judge instanceof Error) throw judge;
      return judge;
    },
  };
}

export async function startServer(options = {}) {
  const llm = options.llm || fakeLlm();
  const store = options.store || createStore({ storePath: null });
  const app = createApp({ llm, store, env: { GEMINI_API_KEY: 'test-key', ...(options.env || {}) }, embedBatchDelayMs: 0, ...options.app });
  const server = await new Promise((resolve) => { const s = app.listen(0, () => resolve(s)); });
  const base = `http://127.0.0.1:${server.address().port}`;
  return { base, llm, store, close: () => new Promise((r) => server.close(r)) };
}

export async function uploadText(base, name, text, headers = {}) {
  const form = new FormData();
  form.append('file', new Blob([text], { type: 'text/plain' }), name);
  return fetch(`${base}/api/documents`, { method: 'POST', body: form, headers });
}

// Reads a Server-Sent Events response into [{ event, data }].
export async function readSse(res) {
  const raw = await res.text();
  return raw.split('\n\n').filter(Boolean).map((block) => ({
    event: /^event: (.+)$/m.exec(block)[1],
    data: JSON.parse(/^data: (.+)$/m.exec(block)[1]),
  }));
}

export const REFUND_DOC = [
  'Refund policy. Customers can request a full refund within 30 days of delivery. Refunds are paid back to the original payment method within 5 business days.',
  '',
  'Shipping. Orders above 50 euros ship for free inside France. Express delivery costs 9 euros and arrives the next business day.',
  '',
  'Support. Our support team answers emails within 24 hours, Monday to Friday.',
].join('\n');
