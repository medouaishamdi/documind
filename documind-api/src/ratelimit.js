// In-memory rate limiting. No external dependency (no Redis needed for a single-process
// app) — swap the Maps for Redis if this ever runs across several server instances.

// Sliding-window limiter per client IP.
export function rateLimit({ windowMs = 60_000, max = 20, now = Date.now } = {}) {
  const hits = new Map(); // ip -> timestamps[]
  let lastSweep = now();

  const middleware = (req, res, next) => {
    const ip = req.ip || 'unknown';
    const t = now();
    const windowStart = t - windowMs;

    // Forget clients that have been quiet for a whole window, so the map can't grow forever.
    if (t - lastSweep > windowMs) {
      for (const [key, stamps] of hits) if (stamps[stamps.length - 1] <= windowStart) hits.delete(key);
      lastSweep = t;
    }

    const timestamps = (hits.get(ip) || []).filter((s) => s > windowStart);
    if (timestamps.length >= max) {
      const retryAfter = Math.max(1, Math.ceil((timestamps[0] + windowMs - t) / 1000));
      res.setHeader('Retry-After', retryAfter);
      return res.status(429).json({ error: `Too many requests. Try again in ${retryAfter}s.` });
    }

    timestamps.push(t);
    hits.set(ip, timestamps);
    res.setHeader('X-RateLimit-Remaining', max - timestamps.length);
    next();
  };
  middleware.trackedClients = () => hits.size;
  return middleware;
}

// Daily budget for requests paid with the *server's* Gemini key. Per-IP limits alone don't
// protect a public demo: many visitors can still drain the free-tier quota together.
// Requests that bring their own key (x-gemini-key header) don't count against it.
export function dailyBudget({ max, usesServerKey, now = Date.now }) {
  let day = null;
  let used = 0;

  const middleware = (req, res, next) => {
    if (!max || !usesServerKey(req)) return next();
    const today = new Date(now()).toISOString().slice(0, 10);
    if (today !== day) { day = today; used = 0; }
    if (used >= max) {
      return res.status(429).json({
        error: 'The demo has used its free daily quota. Paste your own free Gemini key in the settings to keep going.',
      });
    }
    used += 1;
    next();
  };
  middleware.used = () => used;
  return middleware;
}
