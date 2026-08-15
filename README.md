# DocuMind

A production-shaped RAG (retrieval-augmented generation) pipeline: upload documents, ask questions, get answers that are streamed live, cited back to their exact source passages, and automatically fact-checked against those sources. **Free to run — Google's Gemini API, no credit card.**

## What it demonstrates

This isn't a ChatGPT wrapper. It's built around the parts of a real RAG system that actually matter in production:

- **Chunking** with paragraph/sentence-aware boundaries (not blind character slicing) and overlap, so each chunk reads coherently as a standalone citation unit.
- **Asymmetric embeddings** — documents are embedded with `RETRIEVAL_DOCUMENT` task type, queries with `RETRIEVAL_QUERY`, which is how Gemini's embedding model is actually trained to be used.
- **Semantic caching** — repeated or paraphrased questions ("what's the refund window?" vs "how long do I have to get a refund?") are detected by comparing query embeddings, not exact string matches, and served instantly without a second model call.
- **Streaming responses** over Server-Sent Events, so answers appear token-by-token instead of after a long wait.
- **Inline citations** — the model is only allowed to answer from the numbered source excerpts it's given, and cites them like `[1]`, `[2]` inline; clicking a citation in the UI scrolls to that exact passage.
- **A faithfulness/hallucination check** — after every answer, a second model call re-reads the answer against only the retrieved sources and scores how well-supported it actually is (0-100), flagging any unsupported claims. This is a lightweight, purpose-built version of what tools like Ragas measure.
- **Rate limiting** — a sliding-window limiter (20 requests/minute/IP) protects the API from being hammered, the same pattern you'd extend with Redis for a multi-instance deployment.

## Tech stack

- **`documind-api`** — Node.js + Express (plain JavaScript, no SDK dependency — talks to Gemini's REST API directly for embeddings, streaming generation, and the eval check). PDF text extraction via `pdf-parse`.
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
2. **Ask a question.** The question is embedded, checked against the semantic cache, and — on a miss — compared via cosine similarity against every chunk from the documents you have selected, keeping the top 5.
3. Those 5 chunks are numbered and sent to the model with strict instructions: answer only from these sources, cite inline.
4. The answer **streams** back over SSE as it's generated.
5. Once complete, a second model call **fact-checks** the answer against only those same sources and returns a faithfulness score plus any unsupported claims.
6. The question, answer, sources, and eval result are cached (in memory) so a semantically similar future question skips regeneration entirely.

## API reference

`GET /api/health` — health check, includes semantic cache size.

`GET /api/documents` — list uploaded documents.

`POST /api/documents` — multipart form, field `file`. Chunks, embeds, and indexes it.

`DELETE /api/documents/:id` — removes a document and its chunks.

`POST /api/query` — JSON body `{ "question": "...", "documentIds": ["..."] }` (`documentIds` optional — omit to search everything). Returns a `text/event-stream` with events `sources`, `delta` (repeated), `eval`, `done`.

## Notes / possible upgrades

- Swap the JSON store for pgvector or Pinecone once you outgrow in-memory search — the API surface stays the same.
- The semantic cache is in-memory only and resets on restart; persist it the same way the document store is persisted if you want it to survive.
- Add re-ranking (a cheap second pass over the top-20 candidates before picking the final top-5) for noticeably better retrieval quality on larger document sets.
