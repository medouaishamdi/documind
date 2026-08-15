// Splits text into overlapping chunks, trying to break on paragraph/sentence
// boundaries near the target size rather than mid-word, so each chunk reads
// coherently on its own (important since chunks become citation units).
export function chunkText(text, { targetSize = 1200, overlap = 150 } = {}) {
  const clean = text.replace(/\r\n/g, '\n').replace(/\n{3,}/g, '\n\n').trim();
  if (!clean) return [];

  const chunks = [];
  let start = 0;

  while (start < clean.length) {
    let end = Math.min(start + targetSize, clean.length);

    if (end < clean.length) {
      // Prefer breaking at a paragraph, then sentence, then whitespace boundary.
      const windowStart = Math.max(start + targetSize * 0.5, start);
      const searchZone = clean.slice(windowStart, end + 200);
      const paraBreak = searchZone.lastIndexOf('\n\n');
      const sentenceBreak = searchZone.search(/[.!?]\s(?!.*[.!?]\s)/);
      if (paraBreak !== -1) {
        end = windowStart + paraBreak + 2;
      } else if (sentenceBreak !== -1) {
        end = windowStart + sentenceBreak + 2;
      }
    }

    const piece = clean.slice(start, end).trim();
    if (piece) chunks.push(piece);

    if (end >= clean.length) break;
    start = Math.max(end - overlap, start + 1);
  }

  return chunks;
}
