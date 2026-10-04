import { test } from 'node:test';
import assert from 'node:assert/strict';
import { chunkText } from '../src/chunk.js';
import { BM25Index, fuse, tokenize } from '../src/bm25.js';
import { rankChunks, cosineSimilarity } from '../src/retrieve.js';
import { rateLimit, dailyBudget } from '../src/ratelimit.js';
import { createStore } from '../src/store.js';
import { fakeEmbedding } from './helpers.js';

// ------------------------------------------------------------------ chunking
test('chunking: empty text gives no chunks, short text gives one', () => {
  assert.deepEqual(chunkText('   \n\n '), []);
  assert.deepEqual(chunkText('Hello world.'), ['Hello world.']);
});

test('chunking: long text is split near the target size, covers everything and overlaps', () => {
  const paragraphs = Array.from({ length: 40 }, (_, i) => `Paragraph ${i} talks about topic number ${i}. It has a second sentence.`);
  const text = paragraphs.join('\n\n');
  const chunks = chunkText(text, { targetSize: 400, overlap: 60 });
  assert.ok(chunks.length > 5);
  assert.ok(chunks.every((c) => c.length <= 400 + 200), 'no chunk is far above the target');
  for (const p of paragraphs) assert.ok(chunks.some((c) => c.includes(p)), `paragraph lost: ${p}`);
  for (let i = 1; i < chunks.length; i++) {
    const tail = chunks[i - 1].slice(-30);
    assert.ok(chunks[i].includes(tail.trim().slice(-15)), 'consecutive chunks overlap');
  }
});

test('chunking: prefers paragraph boundaries over cutting mid-sentence', () => {
  const a = 'A'.repeat(300) + '.';
  const text = `${a}\n\n${'B'.repeat(300)}.`;
  const chunks = chunkText(text, { targetSize: 400, overlap: 0 });
  assert.equal(chunks[0], a);
});

// ---------------------------------------------------------------------- BM25
test('tokenizer: lowercases, strips accents and stopwords, light plural stemming', () => {
  assert.deepEqual(tokenize('The Refunds for Policies, café!'), ['refund', 'policy', 'cafe']);
  assert.deepEqual(tokenize('Error E-07 on the battery'), ['error', 'e', '07', 'battery']);
});

test('BM25: exact rare terms win, and frequent words weigh less', () => {
  const idx = new BM25Index(['the battery shows error E07', 'the battery charges in 4 hours', 'warranty lasts two years']);
  const s = idx.scores('E07 battery');
  assert.ok(s[0] > s[1] && s[1] > s[2]);
  assert.equal(s[2], 0);
  assert.ok(idx.idf('e07') > idx.idf('battery'));
});

test('RRF fusion rewards items ranked well by both lists', () => {
  const fused = fuse([['a', 'b', 'c'], ['b', 'c', 'a']]).map((x) => x.id);
  assert.equal(fused[0], 'b');
  assert.deepEqual(new Set(fused), new Set(['a', 'b', 'c']));
});

test('retrieval modes: dense, bm25 and hybrid all find the obvious chunk', () => {
  const texts = ['refund within 30 days of delivery', 'free shipping above 50 euros', 'support answers in 24 hours'];
  const chunks = texts.map((text) => ({ text, embedding: fakeEmbedding(text) }));
  const q = 'how many days for a refund';
  for (const mode of ['dense', 'bm25', 'hybrid']) {
    const r = rankChunks(chunks, { queryText: q, queryEmbedding: fakeEmbedding(q), mode, topK: 2 });
    assert.equal(r[0].chunk.text, texts[0], mode);
  }
  assert.throws(() => rankChunks(chunks, { queryText: q, mode: 'nope' }));
  assert.equal(cosineSimilarity([0, 0], [1, 1]), 0);
});

test('hybrid retrieval still returns semantic matches that share no keyword', () => {
  const chunks = [
    { text: 'alpha beta', embedding: [1, 0] },
    { text: 'gamma delta', embedding: [0, 1] },
  ];
  const r = rankChunks(chunks, { queryText: 'zzz', queryEmbedding: [0, 1], mode: 'hybrid', topK: 1 });
  assert.equal(r[0].chunk.text, 'gamma delta');
});

// --------------------------------------------------------------- rate limits
function mockRes() {
  return { statusCode: 200, headers: {}, body: null,
    setHeader(k, v) { this.headers[k] = v; }, status(c) { this.statusCode = c; return this; }, json(b) { this.body = b; return this; } };
}

test('rate limiter: blocks after max requests, sets Retry-After, and recovers after the window', () => {
  let t = 0;
  const limiter = rateLimit({ windowMs: 1000, max: 2, now: () => t });
  const run = (ip = '1.1.1.1') => { const res = mockRes(); let passed = false; limiter({ ip }, res, () => { passed = true; }); return { res, passed }; };
  assert.ok(run().passed && run().passed);
  const blocked = run();
  assert.equal(blocked.passed, false);
  assert.equal(blocked.res.statusCode, 429);
  assert.equal(blocked.res.headers['Retry-After'], 1);
  assert.ok(run('2.2.2.2').passed, 'other clients are not affected');
  t = 1500;
  assert.ok(run().passed, 'allowed again after the window');
});

test('rate limiter: forgets idle clients so memory does not grow forever', () => {
  let t = 0;
  const limiter = rateLimit({ windowMs: 1000, max: 5, now: () => t });
  for (let i = 0; i < 100; i++) limiter({ ip: `10.0.0.${i}` }, mockRes(), () => {});
  assert.equal(limiter.trackedClients(), 100);
  t = 5000;
  limiter({ ip: 'new' }, mockRes(), () => {});
  assert.equal(limiter.trackedClients(), 1);
});

test('daily budget: only server-key requests count, and it resets the next day', () => {
  let t = Date.parse('2026-10-04T10:00:00Z');
  const budget = dailyBudget({ max: 2, usesServerKey: (req) => !req.ownKey, now: () => t });
  const run = (req = {}) => { const res = mockRes(); let passed = false; budget(req, res, () => { passed = true; }); return passed; };
  assert.ok(run() && run());
  assert.equal(run(), false);
  assert.ok(run({ ownKey: true }), 'own key bypasses the budget');
  t += 24 * 3600 * 1000;
  assert.ok(run());
});

// --------------------------------------------------------------------- store
test('store: add, search with doc filter, delete, and cache invalidation on delete', () => {
  const store = createStore({ storePath: null });
  const add = (name, texts) => store.addDocument({ name, chunkTexts: texts, embeddings: texts.map((x) => fakeEmbedding(x)) });
  const a = add('a.txt', ['refund within 30 days', 'shipping is free']);
  const b = add('b.txt', ['refund of the train ticket']);
  const q = 'refund';
  const both = store.search({ queryEmbedding: fakeEmbedding(q), queryText: q, topK: 5 });
  assert.equal(new Set(both.map((r) => r.chunk.docId)).size, 2);
  const onlyB = store.search({ queryEmbedding: fakeEmbedding(q), queryText: q, docIds: [b] });
  assert.ok(onlyB.every((r) => r.chunk.docId === b));

  store.cacheAnswer({ queryEmbedding: fakeEmbedding(q), docIdsKey: [a, b].sort().join(','), question: q, answer: 'x', sources: [], evalResult: {} });
  assert.equal(store.cacheStats().entries, 1);
  assert.ok(store.deleteDocument(a));
  assert.equal(store.cacheStats().entries, 0, 'cached answers that used the deleted doc are dropped');
  assert.equal(store.deleteDocument(a), false);
  assert.deepEqual(store.listDocuments().map((d) => d.name), ['b.txt']);
});
