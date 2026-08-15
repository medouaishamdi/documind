// Simple in-memory sliding-window rate limiter. No external dependency (no Redis
// needed for a single-process personal app) — swap this for a Redis-backed
// limiter if this ever runs across multiple server instances.
export function rateLimit({ windowMs = 60_000, max = 20 } = {}) {
  const hits = new Map(); // ip -> timestamps[]

  return (req, res, next) => {
    const ip = req.ip || 'unknown';
    const now = Date.now();
    const windowStart = now - windowMs;

    const timestamps = (hits.get(ip) || []).filter((t) => t > windowStart);
    if (timestamps.length >= max) {
      const retryAfterMs = timestamps[0] + windowMs - now;
      res.setHeader('Retry-After', Math.ceil(retryAfterMs / 1000));
      return res.status(429).json({ error: `Too many requests. Try again in ${Math.ceil(retryAfterMs / 1000)}s.` });
    }

    timestamps.push(now);
    hits.set(ip, timestamps);
    next();
  };
}
