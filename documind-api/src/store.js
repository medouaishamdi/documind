// File-backed document/chunk store (deliberately simple JSON-on-disk instead of
// a real vector DB — this is a personal-scale RAG app; the retrieval math is exactly
// what pgvector/Pinecone would do, just computed in-process. Swapping in pgvector
// later means replacing this file's internals, not the API surface.)
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { BM25Index } from './bm25.js';
import { cosineSimilarity, rankChunks } from './retrieve.js';

// Resolved from this file, not from process.cwd(), so the API works whichever folder it is started from.
export const DEFAULT_STORE_PATH = path.join(path.dirname(fileURLToPath(import.meta.url)), 'data', 'store.json');

export function createStore({ storePath = DEFAULT_STORE_PATH, cacheThreshold = 0.96, cacheMaxEntries = 200 } = {}) {
  const emptyStore = () => ({ documents: [], chunks: [] });

  function load() {
    try {
      if (!storePath || !fs.existsSync(storePath)) return emptyStore();
      return { ...emptyStore(), ...JSON.parse(fs.readFileSync(storePath, 'utf-8')) };
    } catch {
      return emptyStore();
    }
  }

  let data = load();
  let bm25Cache = null; // { key, index }: rebuilt only when the searched set of chunks changes

  function save() {
    bm25Cache = null;
    if (!storePath) return; // in-memory store (tests)
    fs.mkdirSync(path.dirname(storePath), { recursive: true });
    const tmp = `${storePath}.tmp`;
    fs.writeFileSync(tmp, JSON.stringify(data));
    fs.renameSync(tmp, storePath);
  }

  // ---- Semantic cache (in-memory only — intentionally resets on restart) ----
  // Instead of exact-string caching, compare the new query's embedding against cached
  // query embeddings and reuse the answer on a close semantic match, which catches
  // paraphrases ("what's the refund window?" ~= "how long to get a refund?").
  let cache = [];

  return {
    addDocument({ name, chunkTexts, embeddings }) {
      const docId = crypto.randomUUID();
      data.documents.push({ id: docId, name, uploadedAt: new Date().toISOString(), chunkCount: chunkTexts.length });
      chunkTexts.forEach((text, i) => {
        data.chunks.push({ id: crypto.randomUUID(), docId, docName: name, index: i, text, embedding: embeddings[i] });
      });
      save();
      return docId;
    },

    listDocuments() {
      return data.documents.map(({ id, name, uploadedAt, chunkCount }) => ({ id, name, uploadedAt, chunkCount }));
    },

    deleteDocument(docId) {
      if (!data.documents.some((d) => d.id === docId)) return false;
      data.documents = data.documents.filter((d) => d.id !== docId);
      data.chunks = data.chunks.filter((c) => c.docId !== docId);
      cache = cache.filter((e) => !e.docIdsKey.split(',').includes(docId)); // never serve answers built on a deleted doc
      save();
      return true;
    },

    // Top-K chunks for a query, optionally restricted to a set of doc IDs. See retrieve.js for the modes.
    search({ queryEmbedding, queryText, docIds = null, topK = 5, mode = 'hybrid' }) {
      const pool = docIds ? data.chunks.filter((c) => docIds.includes(c.docId)) : data.chunks;
      const key = pool.length + ':' + (docIds ? [...docIds].sort().join(',') : '*');
      if (!bm25Cache || bm25Cache.key !== key) bm25Cache = { key, index: new BM25Index(pool.map((c) => c.text)) };
      return rankChunks(pool, { queryEmbedding, queryText, mode, topK, bm25: bm25Cache.index });
    },

    findCachedAnswer(queryEmbedding, docIdsKey) {
      let best = null;
      for (const entry of cache) {
        if (entry.docIdsKey !== docIdsKey) continue;
        const score = cosineSimilarity(queryEmbedding, entry.queryEmbedding);
        if (score >= cacheThreshold && (!best || score > best.score)) best = { ...entry, score };
      }
      return best;
    },

    cacheAnswer({ queryEmbedding, docIdsKey, question, answer, sources, evalResult }) {
      cache.push({ queryEmbedding, docIdsKey, question, answer, sources, evalResult, cachedAt: new Date().toISOString() });
      if (cache.length > cacheMaxEntries) cache.shift();
    },

    cacheStats() {
      return { entries: cache.length };
    },
  };
}
