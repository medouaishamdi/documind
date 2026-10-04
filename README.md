# DocuMind

![tests](https://github.com/medouaishamdi/documind/actions/workflows/tests.yml/badge.svg)

A production-shaped RAG (retrieval-augmented generation) pipeline: upload documents, ask questions, get answers that are streamed live, cited back to their exact source passages, and automatically fact-checked against those sources. **Free to run — Google's Gemini API, no credit card.**

## What it demonstrates

This isn't a ChatGPT wrapper. It's built around the parts of a real RAG system that actually matter in production:

- **Chunking** with paragraph/sentence-aware boundaries (not blind character slicing) and overlap, so each chunk reads coherently as a standalone citation unit.
- **Asymmetric embeddings** — documents are embedded with `RETRIEVAL_DOCUMENT` task type, queries with `RETRIEVAL_QUERY`, which is how Gemini's embedding model is actually trained to be used.
- **Semantic caching** — repeated or paraphrased questions ("what's the refund window?" vs "how long do I have to get a refund?") are detected by comparing query embeddings, not exact string matches, and served instantly without a second model call.
- **Streaming responses** over Server-Sent Events, so answers appear token-by-token instead of after a long wait.
- **Inline citations** — the model is only allowed to answer from the numbered source excerpts it's given, and cites them like `[1]`, `[2]` inline; clicking a citation in the UI scrolls to that exact passage.
- **A faithfulness/hallucination check** — after every answer, a second model call re-reads the answer against only the retrieved sources and scores how well-supported it actually is (0-100), flagging any unsupported claims. This is a lightweight, purpose-built version of what tools like Ragas measure.
- **Hybrid retrieval** — every question is searched two ways: by meaning (Gemini embeddings) and by keywords (BM25, implemented from scratch in `src/bm25.js`). The two rankings are merged with Reciprocal Rank Fusion. Embeddings understand paraphrases but miss exact tokens like error codes, product names and numbers; BM25 is the opposite. The mode can be switched per question (`hybrid`, `dense`, `bm25`) to compare them.
- **A measured retrieval benchmark** — `npm run eval` scores retrieval on a labelled question set (see [Evaluation](#evaluation)) instead of claiming it "works well".
- **Abuse protection for a public demo** — per-IP sliding-window limits on questions (20/min) and uploads (5/min, the expensive embedding calls), Retry-After headers, a memory sweep so the limiter can't grow forever, and an optional **daily budget for the server's own API key**, so visitors can't drain the free quota (people who paste their own key aren't counted). `TRUST_PROXY=1` makes the limits work per visitor behind nginx/Caddy.
- **Tests** — 22 tests (`npm test`, Node's built-in runner, no extra dependencies) covering chunking, BM25, fusion, rate limits, the store and the full HTTP API with a fake model client, so they run offline in under a second.

## Tech stack

- **`documind-api`** — Node.js + Express (plain JavaScript, no SDK dependency — talks to Gemini's REST API directly for embeddings, streaming generation, and the eval check). PDF text extraction via `pdf-parse`. The app is built by a factory (`createApp`) so the model client and the store can be swapped for fakes in tests.
- **`documind-web`** — vanilla HTML/CSS/JS, no build step. Parses the SSE stream by hand (no library) to render live tokens, sources, and the eval badge.

## An honest note on the vector store

This uses a JSON-file-backed store with cosine similarity computed in Node, not Pinecone/Milvus/pgvector. For a personal-scale app (hundreds to low thousands of chunks) this is genuinely fine and is mathematically the same operation those tools do — it just doesn't scale past what fits comfortably in memory. `documind-api/src/store.js` is where you'd swap in pgvector or a real vector DB; the rest of the API wouldn't need to change.

## Setup (free, no credit card)

1. Go to **https://aistudio.google.com/apikey**, sign in with any Google account, click "Create API key." Copy the key shown.
2. The free tier's daily limits are generous enough for personal use and demos.

### 1. Start the API

```bash
cd documind-api
npm install
npm run dev
```

Runs at `http://localhost:4500`.

Provide your API key either by pasting it into the "Gemini API key" field in the web app (stored only in your browser), or by copying `.env.example` to `.env` and setting `GEMINI_API_KEY=...`.

### 2. Open the web app

Open `documind-web/index.html` directly (double-click it), or serve it:
```bash
cd documind-web
npx serve .
```

## How it works, end to end

1. **Upload** a `.txt`, `.md`, or `.pdf` file. The server extracts text, splits it into overlapping chunks, embeds each chunk (in batches), and stores them.
2. **Ask a question.** The question is embedded and checked against the semantic cache. On a miss, it is ranked against every chunk of the selected documents twice — cosine similarity of the embeddings and BM25 keyword scoring — and the two top-20 lists are fused (Reciprocal Rank Fusion) to keep the best 5.
3. Those 5 chunks are numbered and sent to the model with strict instructions: answer only from these sources, cite inline.
4. The answer **streams** back over SSE as it's generated.
5. Once complete, a second model call **fact-checks** the answer against only those same sources and returns a faithfulness score plus any unsupported claims.
6. The question, answer, sources, and eval result are cached (in memory) so a semantically similar future question skips regeneration entirely.

## Evaluation

`documind-api/eval/` holds a small benchmark: three original documents (an employee handbook, an e-bike manual and a privacy policy, all for fictional organisations, 16 chunks) and **40 labelled questions**, each with the exact passage that answers it. Half are **keyword** questions that reuse the document's wording ("What does error E-07 mean?"). The other half are **paraphrases** that deliberately share no key terms with the answer ("How long until my bike is fully juiced up?" for "A full charge from empty takes about 4 hours 30 minutes"). Five extra questions have no answer in the documents, to check that the model refuses instead of inventing.

A question counts as a hit@k when one of the top k retrieved chunks contains its evidence passage. The script uses the app's own chunker and retrieval code.

BM25 alone (measured, no API key needed):

| Retrieval | Questions | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---:|---:|---:|---:|
| BM25 | keyword (20) | 95% | 100% | 100% | 0.975 |
| BM25 | paraphrase (20) | 20% | 50% | 55% | 0.363 |
| BM25 | all (40) | 57% | 75% | 78% | 0.669 |

Keyword search is almost perfect when the user speaks the document's language and fails when they don't: that gap is why the app fuses it with embeddings. To measure dense and hybrid retrieval, and end-to-end answers, with your own key:

```bash
cd documind-api
npm run eval                 # retrieval: BM25 vs embeddings vs hybrid (embeddings are cached in eval/.cache)
npm run eval -- --answers    # + citation accuracy, judged faithfulness, refusals on unanswerable questions
```

Results are printed as tables and saved to `eval/results.json`, with per-question ranks.

**Limitations of this benchmark:**
- It is small: 16 chunks, so hit@5 is easy. Hit@1 and MRR are the informative numbers.
- I wrote the questions myself, knowing the documents.
- The faithfulness score comes from an LLM judge, so it is an indicator, not ground truth.

## Tests

```bash
cd documind-api
npm test
```

## API reference

`GET /api/health` — health check, includes semantic cache size.

`GET /api/documents` — list uploaded documents.

`POST /api/documents` — multipart form, field `file`. Chunks, embeds, and indexes it.

`DELETE /api/documents/:id` — removes a document and its chunks.

`POST /api/query` — JSON body `{ "question": "...", "documentIds": ["..."], "mode": "hybrid" }` (`documentIds` optional — omit to search everything; `mode` is `hybrid` (default), `dense` or `bm25`). Returns a `text/event-stream` with events `sources`, `delta` (repeated), `eval`, `done`.

## Running it publicly

The API reads its settings from `documind-api/.env` (see `.env.example`). For a public demo:
- set `GEMINI_API_KEY` so visitors can try it without a key;
- set `SERVER_KEY_DAILY_QUERIES` (e.g. 200) so the free quota can't be drained;
- set `TRUST_PROXY=1` behind a reverse proxy;
- serve `documind-web/` as static files.

## Notes / possible upgrades

- Swap the JSON store for pgvector or Pinecone once you outgrow in-memory search — the API surface stays the same.
- The semantic cache is in-memory only and resets on restart; persist it the same way the document store is persisted if you want it to survive.
- Add a cross-encoder re-ranker over the fused top-20 and measure it with `npm run eval` before keeping it.
