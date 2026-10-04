import { test } from 'node:test';
import assert from 'node:assert/strict';
import { startServer, uploadText, readSse, fakeLlm, REFUND_DOC } from './helpers.js';

const ask = (base, body, headers = {}) => fetch(`${base}/api/query`, {
  method: 'POST', headers: { 'Content-Type': 'application/json', ...headers }, body: JSON.stringify(body),
});

test('upload a document, then ask: SSE events arrive in order with cited sources and an eval', async () => {
  const srv = await startServer();
  try {
    const up = await uploadText(srv.base, 'policy.txt', REFUND_DOC);
    assert.equal(up.status, 200);
    const doc = await up.json();
    assert.equal(doc.chunkCount, 1);

    const docs = await (await fetch(`${srv.base}/api/documents`)).json();
    assert.equal(docs.documents[0].name, 'policy.txt');

    const res = await ask(srv.base, { question: 'How long do I have to ask for a refund?' });
    assert.equal(res.status, 200);
    assert.match(res.headers.get('content-type'), /text\/event-stream/);
    const events = await readSse(res);
    const names = events.map((e) => e.event);
    assert.equal(names[0], 'sources');
    assert.ok(names.slice(1, -2).every((n) => n === 'delta'));
    assert.deepEqual(names.slice(-2), ['eval', 'done']);

    const sources = events[0].data;
    assert.equal(sources[0].n, 1);
    assert.equal(sources[0].docName, 'policy.txt');
    assert.ok(sources[0].text.includes('30 days'));
    const answer = events.filter((e) => e.event === 'delta').map((e) => e.data.text).join('');
    assert.equal(answer, 'The refund window is 30 days [1].');
    assert.equal(events.at(-2).data.faithfulness, 95);
  } finally { await srv.close(); }
});

test('semantic cache: asking the same question again skips generation', async () => {
  const srv = await startServer();
  try {
    await uploadText(srv.base, 'policy.txt', REFUND_DOC);
    await readSse(await ask(srv.base, { question: 'What is the refund window?' }));
    const events = await readSse(await ask(srv.base, { question: 'What is the refund window?' }));
    assert.equal(events.find((e) => e.event === 'eval').data.fromCache, true);
    assert.equal(srv.llm.calls.generate, 1);
    // a different retrieval mode is a different cache entry
    await readSse(await ask(srv.base, { question: 'What is the refund window?', mode: 'bm25' }));
    assert.equal(srv.llm.calls.generate, 2);
  } finally { await srv.close(); }
});

test('a failing faithfulness check does not break the answer', async () => {
  const srv = await startServer({ llm: fakeLlm({ judge: new Error('judge down') }) });
  try {
    await uploadText(srv.base, 'policy.txt', REFUND_DOC);
    const events = await readSse(await ask(srv.base, { question: 'refund?' }));
    const ev = events.find((e) => e.event === 'eval').data;
    assert.equal(ev.faithfulness, null);
    assert.equal(events.at(-1).event, 'done');
  } finally { await srv.close(); }
});

test('input validation', async () => {
  const srv = await startServer();
  try {
    assert.equal((await ask(srv.base, { question: 'anything' })).status, 400, 'no documents yet');
    await uploadText(srv.base, 'policy.txt', REFUND_DOC);
    assert.equal((await ask(srv.base, { question: '   ' })).status, 400);
    assert.equal((await ask(srv.base, { question: 'x'.repeat(2001) })).status, 400);
    assert.equal((await ask(srv.base, { question: 'ok', documentIds: 'abc' })).status, 400);
    assert.equal((await ask(srv.base, { question: 'ok', mode: 'magic' })).status, 400);
    assert.equal((await ask(srv.base, { question: 'ok', documentIds: ['deleted-id'] })).status, 400);
    assert.equal((await uploadText(srv.base, 'empty.txt', '  \n ')).status, 400);
    const exe = new FormData();
    exe.append('file', new Blob(['MZ'], { type: 'application/octet-stream' }), 'virus.exe');
    assert.equal((await fetch(`${srv.base}/api/documents`, { method: 'POST', body: exe })).status, 415);
    const pdf = new FormData();
    pdf.append('file', new Blob(['not really a pdf'], { type: 'application/pdf' }), 'broken.pdf');
    assert.equal((await fetch(`${srv.base}/api/documents`, { method: 'POST', body: pdf })).status, 422);
    assert.equal((await fetch(`${srv.base}/api/documents/nope`, { method: 'DELETE' })).status, 404);
  } finally { await srv.close(); }
});

test('no API key anywhere gives a clear 401; a key in the header works', async () => {
  const srv = await startServer({ env: { GEMINI_API_KEY: '' } });
  try {
    const res = await uploadText(srv.base, 'policy.txt', REFUND_DOC);
    assert.equal(res.status, 401);
    assert.match((await res.json()).error, /aistudio\.google\.com/);
    assert.equal((await uploadText(srv.base, 'policy.txt', REFUND_DOC, { 'x-gemini-key': 'mine' })).status, 200);
  } finally { await srv.close(); }
});

test('Gemini errors are passed through with a useful status', async () => {
  const err = Object.assign(new Error('API key not valid'), { httpStatus: 401 });
  const srv = await startServer({ llm: fakeLlm({ failEmbed: err }) });
  try {
    const res = await uploadText(srv.base, 'policy.txt', REFUND_DOC);
    assert.equal(res.status, 401);
    assert.equal((await res.json()).error, 'API key not valid');
  } finally { await srv.close(); }
});

test('per-IP rate limits on queries and uploads', async () => {
  const srv = await startServer({ app: { limits: { queriesPerMinute: 2, uploadsPerMinute: 1 } } });
  try {
    assert.equal((await uploadText(srv.base, 'a.txt', REFUND_DOC)).status, 200);
    const blocked = await uploadText(srv.base, 'b.txt', REFUND_DOC);
    assert.equal(blocked.status, 429);
    assert.ok(Number(blocked.headers.get('retry-after')) >= 1);
    await readSse(await ask(srv.base, { question: 'refund?' }));
    await readSse(await ask(srv.base, { question: 'shipping?' }));
    assert.equal((await ask(srv.base, { question: 'support?' })).status, 429);
  } finally { await srv.close(); }
});

test('daily budget protects the server key but not visitors who bring their own', async () => {
  const srv = await startServer({ env: { SERVER_KEY_DAILY_QUERIES: '1' } });
  try {
    await uploadText(srv.base, 'policy.txt', REFUND_DOC);
    assert.equal((await ask(srv.base, { question: 'refund?' })).status, 200);
    const res = await ask(srv.base, { question: 'shipping?' });
    assert.equal(res.status, 429);
    assert.match((await res.json()).error, /own free Gemini key/);
    assert.equal((await ask(srv.base, { question: 'shipping?' }, { 'x-gemini-key': 'mine' })).status, 200);
    const health = await (await fetch(`${srv.base}/api/health`)).json();
    assert.equal(health.serverKeyQueriesToday, 1);
  } finally { await srv.close(); }
});

test('deleting a document removes it from search', async () => {
  const srv = await startServer();
  try {
    const { id } = await (await uploadText(srv.base, 'policy.txt', REFUND_DOC)).json();
    await uploadText(srv.base, 'other.txt', 'Our office is in Lyon. The cafeteria opens at noon.');
    assert.equal((await fetch(`${srv.base}/api/documents/${id}`, { method: 'DELETE' })).status, 200);
    const events = await readSse(await ask(srv.base, { question: 'refund within 30 days' }));
    assert.ok(events[0].data.every((s) => s.docName === 'other.txt'));
  } finally { await srv.close(); }
});
