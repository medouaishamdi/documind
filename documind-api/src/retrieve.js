// Retrieval over a list of chunks: dense (embeddings), sparse (BM25) or hybrid (both, fused).
// Pure functions with no I/O, so the API, the tests and the evaluation script share the same code.
import { BM25Index, fuse } from './bm25.js';

export const MODES = ['hybrid', 'dense', 'bm25'];

export function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function rankBy(scores) {
  return scores.map((s, i) => [i, s]).sort((a, b) => b[1] - a[1]);
}

/**
 * Rank chunks for a query.
 * @param chunks   [{ text, embedding? }]
 * @param options  { queryText, queryEmbedding, mode, topK, candidates, bm25 }
 *                 `bm25` is an optional pre-built BM25Index over the same chunks (saves rebuilding it).
 * @returns        [{ chunk, score, dense, bm25 }] best first; `dense` is the cosine similarity when available.
 */
export function rankChunks(chunks, { queryText, queryEmbedding, mode = 'hybrid', topK = 5, candidates = 20, bm25 } = {}) {
  if (!chunks.length) return [];
  if (!MODES.includes(mode)) throw new Error(`Unknown retrieval mode "${mode}"`);

  const dense = queryEmbedding ? chunks.map((c) => cosineSimilarity(queryEmbedding, c.embedding)) : null;
  const sparse = queryText && mode !== 'dense' ? (bm25 || new BM25Index(chunks.map((c) => c.text))).scores(queryText) : null;
  const pack = (i, score) => ({ chunk: chunks[i], score, dense: dense ? dense[i] : null, bm25: sparse ? sparse[i] : null });

  if (mode === 'dense' || (mode === 'hybrid' && !sparse)) {
    if (!dense) throw new Error('Dense retrieval needs a query embedding');
    return rankBy(dense).slice(0, topK).map(([i, s]) => pack(i, s));
  }
  if (mode === 'bm25' || !dense) {
    return rankBy(sparse).filter(([, s]) => s > 0).slice(0, topK).map(([i, s]) => pack(i, s));
  }

  // Hybrid: fuse the top candidates of each list. Chunks with no keyword overlap at all
  // get no BM25 rank, so a strong semantic match still gets in through the dense list.
  const denseTop = rankBy(dense).slice(0, candidates).map(([i]) => i);
  const sparseTop = rankBy(sparse).filter(([, s]) => s > 0).slice(0, candidates).map(([i]) => i);
  return fuse([denseTop, sparseTop]).slice(0, topK).map(({ id, score }) => pack(id, score));
}
