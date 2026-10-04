// Retrieval and answer-quality benchmark for DocuMind.
//
//   npm run eval                 BM25 only (offline, no key needed)
//   npm run eval                 + dense and hybrid retrieval when GEMINI_API_KEY is set
//   npm run eval -- --answers    + end-to-end answers: citation accuracy, faithfulness, refusals
//
// It uses the app's own chunker and retrieval code, so it measures what the API actually does.
import 'dotenv/config';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { chunkText } from '../src/chunk.js';
import { BM25Index } from '../src/bm25.js';
import { rankChunks } from '../src/retrieve.js';
import * as gemini from '../src/gemini.js';
import { SYSTEM_PROMPT, JUDGE_PROMPT, buildContext } from '../src/app.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const API_KEY = process.env.GEMINI_API_KEY;
const EMBED_MODEL = process.env.EMBED_MODEL || 'gemini-embedding-001';
const CHAT_MODEL = process.env.CHAT_MODEL || 'gemini-3.6-flash';
const WITH_ANSWERS = process.argv.includes('--answers');
const PACE_MS = Number(process.env.EVAL_PACE_MS || 4500); // free tier: stay under ~15 requests/minute
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------------------------------------------------------------- data
const corpus = fs.readdirSync(path.join(HERE, 'corpus')).filter((f) => f.endsWith('.md')).sort();
const chunks = corpus.flatMap((name) =>
  chunkText(fs.readFileSync(path.join(HERE, 'corpus', name), 'utf-8')).map((text, index) => ({ docName: name, index, text })));
const questions = JSON.parse(fs.readFileSync(path.join(HERE, 'questions.json'), 'utf-8'));
const unanswerable = JSON.parse(fs.readFileSync(path.join(HERE, 'unanswerable.json'), 'utf-8'));

for (const q of questions) {
  if (!chunks.some((c) => c.text.includes(q.evidence))) throw new Error(`Evidence not inside any single chunk: "${q.evidence}"`);
}

// ------------------------------------------------- embeddings (cached on disk)
const cachePath = path.join(HERE, '.cache', `embeddings-${EMBED_MODEL}.json`);
const cache = fs.existsSync(cachePath) ? JSON.parse(fs.readFileSync(cachePath, 'utf-8')) : {};
const keyOf = (text, taskType) => crypto.createHash('sha1').update(`${taskType}\n${text}`).digest('hex');

async function embedAll(texts, taskType) {
  const missing = [...new Set(texts.filter((t) => !cache[keyOf(t, taskType)]))];
  for (let i = 0; i < missing.length; i += 50) {
    const batch = missing.slice(i, i + 50);
    const vectors = await gemini.batchEmbed({ apiKey: API_KEY, model: EMBED_MODEL, texts: batch, taskType });
    batch.forEach((t, j) => { cache[keyOf(t, taskType)] = vectors[j]; });
    fs.mkdirSync(path.dirname(cachePath), { recursive: true });
    fs.writeFileSync(cachePath, JSON.stringify(cache));
    if (i + 50 < missing.length) await sleep(PACE_MS);
  }
  return texts.map((t) => cache[keyOf(t, taskType)]);
}

// --------------------------------------------------------- retrieval metrics
const modes = API_KEY ? ['bm25', 'dense', 'hybrid'] : ['bm25'];
if (API_KEY) {
  const vectors = await embedAll(chunks.map((c) => c.text), 'RETRIEVAL_DOCUMENT');
  chunks.forEach((c, i) => { c.embedding = vectors[i]; });
  const qVectors = await embedAll(questions.map((q) => q.q), 'RETRIEVAL_QUERY');
  questions.forEach((q, i) => { q.embedding = qVectors[i]; });
} else {
  console.log('GEMINI_API_KEY not set: evaluating BM25 only. Set it in .env to compare dense and hybrid retrieval.\n');
}

const bm25 = new BM25Index(chunks.map((c) => c.text));
const retrieval = {};
const perQuestion = questions.map((q) => ({ q: q.q, type: q.type }));
for (const mode of modes) {
  questions.forEach((q, i) => {
    const ranked = rankChunks(chunks, { queryText: q.q, queryEmbedding: q.embedding, mode, topK: chunks.length, bm25 });
    const rank = ranked.findIndex((r) => r.chunk.text.includes(q.evidence)) + 1; // 0 = not retrieved at all
    perQuestion[i][mode] = rank;
  });
  for (const type of ['all', 'keyword', 'paraphrase']) {
    const ranks = perQuestion.filter((p) => type === 'all' || p.type === type).map((p) => p[mode]);
    const hit = (k) => ranks.filter((r) => r >= 1 && r <= k).length / ranks.length;
    retrieval[`${mode}/${type}`] = {
      mode, type, n: ranks.length,
      hit1: hit(1), hit3: hit(3), hit5: hit(5),
      mrr: ranks.reduce((s, r) => s + (r ? 1 / r : 0), 0) / ranks.length,
    };
  }
}

const pct = (x) => `${(100 * x).toFixed(0)}%`;
console.log(`Corpus: ${corpus.length} documents, ${chunks.length} chunks. Questions: ${questions.length} ` +
  `(${questions.filter((q) => q.type === 'keyword').length} keyword, ${questions.filter((q) => q.type === 'paraphrase').length} paraphrase).\n`);
console.log('| Retrieval | Questions | Hit@1 | Hit@3 | Hit@5 | MRR |');
console.log('|---|---|---:|---:|---:|---:|');
for (const r of Object.values(retrieval)) {
  console.log(`| ${r.mode} | ${r.type} (${r.n}) | ${pct(r.hit1)} | ${pct(r.hit3)} | ${pct(r.hit5)} | ${r.mrr.toFixed(3)} |`);
}

// ------------------------------------------------------- end-to-end answers
let answers = null;
if (WITH_ANSWERS) {
  if (!API_KEY) throw new Error('--answers needs GEMINI_API_KEY');
  const mode = 'hybrid';
  const unansVectors = await embedAll(unanswerable, 'RETRIEVAL_QUERY');
  const items = [
    ...questions.map((q) => ({ ...q, answerable: true })),
    ...unanswerable.map((q, i) => ({ q, embedding: unansVectors[i], answerable: false })),
  ];
  const rows = [];
  for (const [i, item] of items.entries()) {
    const sources = rankChunks(chunks, { queryText: item.q, queryEmbedding: item.embedding, mode, topK: 5, bm25 })
      .map((r, j) => ({ n: j + 1, docName: r.chunk.docName, text: r.chunk.text }));
    const context = buildContext(sources);
    let answer = '';
    await gemini.streamGenerate({ apiKey: API_KEY, model: CHAT_MODEL, systemPrompt: SYSTEM_PROMPT,
      userPrompt: `Sources:\n${context}\n\nQuestion: ${item.q}`, onDelta: (t) => { answer += t; } });
    await sleep(PACE_MS);
    let judge = { faithfulness: null };
    try {
      judge = await gemini.generateJson({ apiKey: API_KEY, model: CHAT_MODEL, systemPrompt: JUDGE_PROMPT,
        userPrompt: `Sources:\n${context}\n\nAnswer to check:\n${answer}` });
    } catch (e) { console.warn('judge failed:', e.message); }
    await sleep(PACE_MS);

    const cited = [...answer.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1]));
    const refused = /\b(do(es)? not|don't|doesn't|no information|not (mention|contain|specif|provide|state|include)|cannot find|couldn't find|isn't (mentioned|specified))/i.test(answer) && cited.length === 0;
    const row = { q: item.q, answerable: item.answerable, answer, cited, refused, faithfulness: judge.faithfulness };
    if (item.answerable) row.citesEvidence = cited.some((n) => sources[n - 1]?.text.includes(item.evidence));
    rows.push(row);
    process.stdout.write(`\r answers: ${i + 1}/${items.length}`);
  }
  console.log('\n');
  const ans = rows.filter((r) => r.answerable);
  const una = rows.filter((r) => !r.answerable);
  const faith = ans.map((r) => r.faithfulness).filter((x) => typeof x === 'number');
  answers = {
    model: CHAT_MODEL,
    answerable: ans.length,
    citesCorrectSource: ans.filter((r) => r.citesEvidence).length / ans.length,
    falseRefusals: ans.filter((r) => r.refused).length / ans.length,
    meanFaithfulness: faith.reduce((s, x) => s + x, 0) / (faith.length || 1),
    unanswerable: una.length,
    correctRefusals: una.filter((r) => r.refused).length / una.length,
    rows,
  };
  console.log(`| End-to-end (${mode} retrieval, ${CHAT_MODEL}) | |`);
  console.log('|---|---:|');
  console.log(`| Answer cites a source that contains the evidence | ${pct(answers.citesCorrectSource)} |`);
  console.log(`| Mean faithfulness (LLM judge, 0-100) | ${answers.meanFaithfulness.toFixed(1)} |`);
  console.log(`| Answerable questions wrongly refused | ${pct(answers.falseRefusals)} |`);
  console.log(`| Unanswerable questions correctly refused | ${pct(answers.correctRefusals)} (${una.length}) |`);
}

fs.writeFileSync(path.join(HERE, 'results.json'), JSON.stringify({
  date: new Date().toISOString(), embedModel: API_KEY ? EMBED_MODEL : null, chunks: chunks.length,
  retrieval: Object.values(retrieval), perQuestion, answers,
}, null, 2));
console.log('\nSaved eval/results.json');
