// File-backed document/chunk store (deliberately simple JSON-on-disk instead of
// a real vector DB — this is a personal-scale RAG app; the retrieval math below
// is exactly what pgvector/Pinecone would do, just computed in-process. Swapping
// in pgvector later means replacing this file's internals, not the API surface.)
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { cosineSimilarity } from './gemini.js';

const DATA_DIR = path.join(process.cwd(), 'src', 'data');
const STORE_PATH = path.join(DATA_DIR, 'store.json');

function emptyStore() { return { documents: [], chunks: [] }; }

function load() {
  try {
    if (!fs.existsSync(STORE_PATH)) return emptyStore();
    return { ...emptyStore(), ...JSON.parse(fs.readFileSync(STORE_PATH, 'utf-8')) };
  } catch {
    return emptyStore();
  }
}

function save() {
  fs.mkdirSync(DATA_DIR, { recursive: true });
  const tmp = `${STORE_PATH}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(store, null, 2));
  fs.renameSync(tmp, STORE_PATH);
}

let store = load();

export function addDocument({ name, chunkTexts, embeddings }) {
  const docId = crypto.randomUUID();
  store.documents.push({ id: docId, name, uploadedAt: new Date().toISOString(), chunkCount: chunkTexts.length });
  chunkTexts.forEach((text, i) => {
    store.chunks.push({ id: crypto.randomUUID(), docId, docName: name, index: i, text, embedding: embeddings[i] });
  });
  save();
  return docId;
}

export function listDocuments() {
  return store.documents.map(({ id, name, uploadedAt, chunkCount }) => ({ id, name, uploadedAt, chunkCount }));
}

export function deleteDocument(docId) {
  const before = store.documents.length;
  store.documents = store.documents.filter((d) => d.id !== docId);
  store.chunks = store.chunks.filter((c) => c.docId !== docId);
  save();
  return store.documents.length < before;
}

// Top-K most similar chunks to a query embedding, optionally restricted to a set of doc IDs.
export function search(queryEmbedding, { docIds = null, topK = 5 } = {}) {
  const pool = docIds ? store.chunks.filter((c) => docIds.includes(c.docId)) : store.chunks;
  return pool
    .map((c) => ({ chunk: c, score: cosineSimilarity(queryEmbedding, c.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

// ---- Semantic cache (in-memory only — intentionally resets on restart) ----
// Instead of exact-string caching, we compare the new query's embedding against
// cached query embeddings and reuse the answer on a close semantic match, which
// catches paraphrases ("what's the refund window?" ~= "how long to get a refund?").
const cache = [];
const CACHE_SIMILARITY_THRESHOLD = 0.96;
const CACHE_MAX_ENTRIES = 200;

export function findCachedAnswer(queryEmbedding, docIdsKey) {
  let best = null;
  for (const entry of cache) {
    if (entry.docIdsKey !== docIdsKey) continue;
    const score = cosineSimilarity(queryEmbedding, entry.queryEmbedding);
    if (score >= CACHE_SIMILARITY_THRESHOLD && (!best || score > best.score)) {
      best = { ...entry, score };
    }
  }
  return best;
}

export function cacheAnswer({ queryEmbedding, docIdsKey, question, answer, sources, evalResult }) {
  cache.push({ queryEmbedding, docIdsKey, question, answer, sources, evalResult, cachedAt: new Date().toISOString() });
  if (cache.length > CACHE_MAX_ENTRIES) cache.shift();
}

export function cacheStats() {
  return { entries: cache.length };
}
