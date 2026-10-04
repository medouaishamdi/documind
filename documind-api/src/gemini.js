const BASE = 'https://generativelanguage.googleapis.com/v1beta/models';

function apiError(res, data) {
  const message = data?.error?.message || `Gemini API request failed (${res.status}).`;
  const err = new Error(message);
  // Gemini answers 400 (not 401) for an invalid key; only that case is an auth error.
  const badKey = res.status === 400 && /api key/i.test(message);
  err.httpStatus = badKey ? 401 : res.status >= 500 ? 502 : res.status;
  return err;
}

// Embeds many chunks in one call. taskType is RETRIEVAL_DOCUMENT for indexing,
// RETRIEVAL_QUERY for search — Gemini's embeddings are trained asymmetrically,
// so using the right task type for each side meaningfully improves retrieval quality.
async function batchEmbedOnce({ apiKey, model, texts, taskType }) {
  const res = await fetch(`${BASE}/${model}:batchEmbedContents?key=${encodeURIComponent(apiKey)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      requests: texts.map((text) => ({
        model: `models/${model}`,
        content: { parts: [{ text }] },
        taskType,
      })),
    }),
  });
  const data = await res.json();
  if (!res.ok) throw apiError(res, data);
  return (data.embeddings || []).map((e) => e.values);
}

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

// The free tier caps embedding requests per minute — each text inside a batch call
// counts individually against that cap, so uploading a document with many chunks can
// burst past it even though it's technically one HTTP call. Retry with backoff instead
// of failing the whole upload: Gemini's 429 message usually includes a "retry in Xs"
// hint, so honor that when present, otherwise fall back to a fixed delay.
export async function batchEmbed({ apiKey, model, texts, taskType, maxRetries = 4 }) {
  let attempt = 0;
  while (true) {
    try {
      return await batchEmbedOnce({ apiKey, model, texts, taskType });
    } catch (err) {
      const isRateLimit = err.httpStatus === 429;
      if (!isRateLimit || attempt >= maxRetries) throw err;
      const hinted = /retry in ([\d.]+)s/i.exec(err.message);
      const waitMs = hinted ? Math.ceil(parseFloat(hinted[1]) * 1000) + 500 : 4000 * (attempt + 1);
      attempt += 1;
      await sleep(waitMs);
    }
  }
}

export async function embedOne({ apiKey, model, text, taskType }) {
  const [vec] = await batchEmbed({ apiKey, model, texts: [text], taskType });
  return vec;
}

export { cosineSimilarity } from './retrieve.js';

// Streams a plain-text answer via Gemini's SSE endpoint, calling onDelta(text) for each piece.
export async function streamGenerate({ apiKey, model, systemPrompt, userPrompt, onDelta }) {
  const res = await fetch(`${BASE}/${model}:streamGenerateContent?alt=sse&key=${encodeURIComponent(apiKey)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      system_instruction: { parts: [{ text: systemPrompt }] },
      contents: [{ role: 'user', parts: [{ text: userPrompt }] }],
      generationConfig: { temperature: 0.3 },
    }),
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw apiError(res, data);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let fullText = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split('\n');
    buffer = lines.pop(); // keep the last (possibly incomplete) line for next round

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const jsonStr = line.slice(6).trim();
      if (!jsonStr) continue;
      try {
        const parsed = JSON.parse(jsonStr);
        const delta = parsed?.candidates?.[0]?.content?.parts?.[0]?.text;
        if (delta) { fullText += delta; onDelta(delta); }
      } catch { /* ignore malformed keep-alive lines */ }
    }
  }

  return fullText;
}

// Non-streamed JSON call, used for the faithfulness/eval check.
export async function generateJson({ apiKey, model, systemPrompt, userPrompt }) {
  const res = await fetch(`${BASE}/${model}:generateContent?key=${encodeURIComponent(apiKey)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      system_instruction: { parts: [{ text: systemPrompt }] },
      contents: [{ role: 'user', parts: [{ text: userPrompt }] }],
      generationConfig: { responseMimeType: 'application/json', temperature: 0 },
    }),
  });
  const data = await res.json();
  if (!res.ok) throw apiError(res, data);
  const raw = data?.candidates?.[0]?.content?.parts?.[0]?.text;
  if (!raw) throw Object.assign(new Error('Model returned an empty eval response.'), { httpStatus: 502 });
  try {
    return JSON.parse(raw);
  } catch {
    throw Object.assign(new Error('Model returned an unparsable eval response.'), { httpStatus: 502 });
  }
}
