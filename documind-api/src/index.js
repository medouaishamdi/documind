import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import multer from 'multer';
import pdfParse from 'pdf-parse';
import { chunkText } from './chunk.js';
import { batchEmbed, embedOne, streamGenerate, generateJson } from './gemini.js';
import * as store from './store.js';
import { rateLimit } from './ratelimit.js';

const app = express();
const PORT = process.env.PORT || 4500;
const CHAT_MODEL = process.env.CHAT_MODEL || 'gemini-3.6-flash';
const EMBED_MODEL = process.env.EMBED_MODEL || 'gemini-embedding-001';
const EMBED_BATCH_SIZE = 50; // stay well under Gemini's per-call embedding request cap

app.use(cors());
app.use(express.json());

function getApiKey(req) {
  return req.header('x-gemini-key') || process.env.GEMINI_API_KEY || null;
}

async function embedInBatches({ apiKey, model, texts, taskType }) {
  const out = [];
  for (let i = 0; i < texts.length; i += EMBED_BATCH_SIZE) {
    if (i > 0) await new Promise((r) => setTimeout(r, 1200)); // spread batches to stay under the free-tier per-minute cap
    const batch = texts.slice(i, i + EMBED_BATCH_SIZE);
    const vectors = await batchEmbed({ apiKey, model, texts: batch, taskType });
    out.push(...vectors);
  }
  return out;
}

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 20 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const ok = /\.(txt|md|pdf)$/i.test(file.originalname) || ['text/plain', 'text/markdown', 'application/pdf'].includes(file.mimetype);
    if (!ok) return cb(new Error('Only .txt, .md, or .pdf files are supported.'));
    cb(null, true);
  },
});

app.get('/api/health', (req, res) => {
  res.json({ ok: true, service: 'documind-api', hasServerKey: Boolean(process.env.GEMINI_API_KEY), cache: store.cacheStats() });
});

app.get('/api/documents', (req, res) => {
  res.json({ documents: store.listDocuments() });
});

app.post('/api/documents', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No file uploaded. Use form field name "file".' });

    const apiKey = getApiKey(req);
    if (!apiKey) {
      return res.status(401).json({ error: 'No Gemini API key available. Paste your free key in the app settings, or set GEMINI_API_KEY in documind-api/.env. Get one at https://aistudio.google.com/apikey (no card needed).' });
    }

    let text;
    if (/\.pdf$/i.test(req.file.originalname) || req.file.mimetype === 'application/pdf') {
      const parsed = await pdfParse(req.file.buffer);
      text = parsed.text;
    } else {
      text = req.file.buffer.toString('utf-8');
    }

    const chunks = chunkText(text);
    if (!chunks.length) {
      return res.status(400).json({ error: 'Could not extract any usable text from that file.' });
    }
    if (chunks.length > 300) {
      return res.status(400).json({ error: `That document produced ${chunks.length} chunks — too large for this demo's free-tier pace. Try a shorter document (roughly under 300KB of text).` });
    }

    const embeddings = await embedInBatches({ apiKey, model: EMBED_MODEL, texts: chunks, taskType: 'RETRIEVAL_DOCUMENT' });
    const docId = store.addDocument({ name: req.file.originalname, chunkTexts: chunks, embeddings });

    res.json({ id: docId, name: req.file.originalname, chunkCount: chunks.length });
  } catch (err) {
    console.error(err);
    res.status(err.httpStatus || 500).json({ error: err?.message || 'Something went wrong processing the document.' });
  }
});

app.delete('/api/documents/:id', (req, res) => {
  const ok = store.deleteDocument(req.params.id);
  if (!ok) return res.status(404).json({ error: 'Document not found.' });
  res.json({ ok: true });
});

// Streams the answer as Server-Sent Events: "sources" -> many "delta" -> "eval" -> "done".
app.post('/api/query', rateLimit({ windowMs: 60_000, max: 20 }), async (req, res) => {
  try {
    const { question, documentIds } = req.body || {};
    if (!question || typeof question !== 'string' || !question.trim()) {
      return res.status(400).json({ error: 'Provide a non-empty "question".' });
    }

    const apiKey = getApiKey(req);
    if (!apiKey) {
      return res.status(401).json({ error: 'No Gemini API key available. Paste your free key in the app settings, or set GEMINI_API_KEY in documind-api/.env.' });
    }

    const allDocs = store.listDocuments();
    if (!allDocs.length) {
      return res.status(400).json({ error: 'Upload at least one document before asking questions.' });
    }

    const targetDocIds = Array.isArray(documentIds) && documentIds.length ? documentIds : allDocs.map((d) => d.id);
    const docIdsKey = [...targetDocIds].sort().join(',');

    const queryEmbedding = await embedOne({ apiKey, model: EMBED_MODEL, text: question, taskType: 'RETRIEVAL_QUERY' });
    const cached = store.findCachedAnswer(queryEmbedding, docIdsKey);

    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
    });
    const sendEvent = (event, data) => res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);

    if (cached) {
      sendEvent('sources', cached.sources);
      sendEvent('delta', { text: cached.answer });
      sendEvent('eval', { ...cached.evalResult, fromCache: true });
      sendEvent('done', {});
      return res.end();
    }

    const results = store.search(queryEmbedding, { docIds: targetDocIds, topK: 5 });
    if (!results.length) {
      sendEvent('sources', []);
      sendEvent('delta', { text: "I couldn't find anything relevant to that question in the selected documents." });
      sendEvent('eval', { faithfulness: null, unsupported_claims: [], notes: 'No sources were retrieved.' });
      sendEvent('done', {});
      return res.end();
    }

    const sources = results.map((r, i) => ({
      n: i + 1,
      docName: r.chunk.docName,
      chunkIndex: r.chunk.index,
      text: r.chunk.text,
      score: Math.round(r.score * 1000) / 1000,
    }));
    sendEvent('sources', sources);

    const contextBlock = sources.map((s) => `[${s.n}] (from "${s.docName}")\n${s.text}`).join('\n\n');
    const genSystemPrompt = 'You answer questions using ONLY the numbered source excerpts you are given. Cite sources inline like [1] or [2] immediately after each claim that depends on them. If the sources do not contain the answer, say so plainly instead of guessing or using outside knowledge. Be concise and direct.';
    const genUserPrompt = `Sources:\n${contextBlock}\n\nQuestion: ${question}`;

    let fullAnswer = '';
    await streamGenerate({
      apiKey,
      model: CHAT_MODEL,
      systemPrompt: genSystemPrompt,
      userPrompt: genUserPrompt,
      onDelta: (text) => { fullAnswer += text; sendEvent('delta', { text }); },
    });

    let evalResult = { faithfulness: null, unsupported_claims: [], notes: 'Eval check failed to run.' };
    try {
      evalResult = await generateJson({
        apiKey,
        model: CHAT_MODEL,
        systemPrompt: 'You are a strict fact-checker. Given numbered source excerpts and an answer that cites them, rate how well the answer is actually supported by ONLY those sources — not outside knowledge. Respond ONLY with JSON: { "faithfulness": number from 0 to 100, "unsupported_claims": string[], "notes": "one short sentence" }',
        userPrompt: `Sources:\n${contextBlock}\n\nAnswer to check:\n${fullAnswer}`,
      });
    } catch (evalErr) {
      console.warn('Faithfulness eval failed:', evalErr.message);
    }
    sendEvent('eval', evalResult);

    store.cacheAnswer({ queryEmbedding, docIdsKey, question, answer: fullAnswer, sources, evalResult });

    sendEvent('done', {});
    res.end();
  } catch (err) {
    console.error(err);
    if (!res.headersSent) {
      res.status(err.httpStatus || 500).json({ error: err?.message || 'Something went wrong.' });
    } else {
      res.write(`event: error\ndata: ${JSON.stringify({ error: err?.message || 'Something went wrong.' })}\n\n`);
      res.end();
    }
  }
});

app.use((err, req, res, next) => {
  if (err) return res.status(400).json({ error: err.message });
  next();
});

app.listen(PORT, () => {
  console.log(`documind-api listening on http://localhost:${PORT}`);
});
