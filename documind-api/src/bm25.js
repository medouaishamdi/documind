// Keyword retrieval (Okapi BM25). Embeddings understand paraphrases but are weak on
// exact tokens: error codes, product names, numbers, acronyms. BM25 is the opposite.
// Hybrid retrieval (see fuse() below) keeps the strengths of both.

const STOPWORDS = new Set((
  'a an and are as at be been but by can could did do does for from had has have how i if in into is it its ' +
  'me my no not of on or our so than that the their them then there these they this to too was we were what ' +
  'when where which who whom why will with would you your ' +
  'au aux avec ce ces dans de des du elle en et eux il ils je la le les leur lui ma mais me mes moi mon ne nos ' +
  'notre nous on ou par pas pour qu que qui sa se ses son sur ta te tes toi ton tu un une vos votre vous est sont'
).split(' '));

// Lowercase, strip accents, split on anything that isn't a letter or digit, drop stopwords,
// and apply a deliberately light plural stemmer ("refunds" -> "refund", "policies" -> "policy").
export function tokenize(text) {
  return text
    .toLowerCase()
    .normalize('NFD').replace(/[̀-ͯ]/g, '')
    .split(/[^a-z0-9]+/)
    .filter((t) => t && !STOPWORDS.has(t))
    .map((t) => {
      if (t.length > 4 && t.endsWith('ies')) return `${t.slice(0, -3)}y`;
      if (t.length > 3 && t.endsWith('s') && !t.endsWith('ss')) return t.slice(0, -1);
      return t;
    });
}

export class BM25Index {
  constructor(docs, { k1 = 1.2, b = 0.75 } = {}) {
    this.k1 = k1;
    this.b = b;
    this.docs = docs.map((text) => {
      const tf = new Map();
      const tokens = tokenize(text);
      for (const t of tokens) tf.set(t, (tf.get(t) || 0) + 1);
      return { tf, length: tokens.length };
    });
    this.avgLength = this.docs.reduce((s, d) => s + d.length, 0) / (this.docs.length || 1);
    this.df = new Map();
    for (const d of this.docs) for (const t of d.tf.keys()) this.df.set(t, (this.df.get(t) || 0) + 1);
  }

  idf(term) {
    const n = this.docs.length;
    const df = this.df.get(term) || 0;
    return Math.log(1 + (n - df + 0.5) / (df + 0.5));
  }

  // Returns one score per document, in the order the documents were given.
  scores(query) {
    const terms = [...new Set(tokenize(query))];
    return this.docs.map((d) => {
      let score = 0;
      for (const t of terms) {
        const f = d.tf.get(t);
        if (!f) continue;
        const norm = f + this.k1 * (1 - this.b + (this.b * d.length) / (this.avgLength || 1));
        score += this.idf(t) * ((f * (this.k1 + 1)) / norm);
      }
      return score;
    });
  }
}

// Reciprocal Rank Fusion: score = sum over rankings of 1 / (k + rank). It only uses
// ranks, so it needs no tuning to combine scores that live on different scales
// (cosine similarity in [-1, 1] vs unbounded BM25). k = 60 is the value from the original paper.
export function fuse(rankings, { k = 60 } = {}) {
  const fused = new Map();
  for (const ranking of rankings) {
    ranking.forEach((id, rank) => fused.set(id, (fused.get(id) || 0) + 1 / (k + rank + 1)));
  }
  return [...fused.entries()].sort((a, b) => b[1] - a[1]).map(([id, score]) => ({ id, score }));
}
