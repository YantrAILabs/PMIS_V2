/**
 * ProMe OpenAI Proxy — Cloudflare Worker
 *
 * Purpose: let employee laptops talk to OpenAI without ever holding the master
 * key. Each laptop carries a per-employee access token; the proxy swaps it for
 * the master key, enforces endpoint + model allowlists, tracks per-user spend,
 * and logs usage.
 *
 * Required secrets (configure via `wrangler secret put`):
 *   OPENAI_MASTER_KEY   Your actual sk-... key
 *   EMPLOYEE_TOKENS     JSON map: {"<token>": {"user":"alice","cap_usd":25}, ...}
 *
 * Optional env vars (in wrangler.toml):
 *   ALLOWED_MODELS       Comma-separated. Default: "gpt-4o-mini"
 *   ALLOWED_ENDPOINTS    Comma-separated paths. Default: "/v1/chat/completions"
 *
 * Required bindings:
 *   USAGE_KV             KV namespace for per-user spend counters
 */

const DEFAULT_ALLOWED_MODELS = ["gpt-4o-mini"];
const DEFAULT_ALLOWED_ENDPOINTS = ["/v1/chat/completions"];

// gpt-4o-mini pricing as of early 2026 — update if OpenAI changes.
// USD per 1M tokens.
const MODEL_PRICING = {
  "gpt-4o-mini": { input: 0.15, output: 0.60 },
};

const OPENAI_API = "https://api.openai.com";

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === "GET" && url.pathname === "/healthz") {
      return json({ ok: true });
    }

    const allowedEndpoints = parseList(env.ALLOWED_ENDPOINTS, DEFAULT_ALLOWED_ENDPOINTS);
    if (!allowedEndpoints.includes(url.pathname)) {
      return json({ error: "endpoint_not_allowed", path: url.pathname }, 403);
    }

    const auth = request.headers.get("authorization") || "";
    const token = auth.replace(/^Bearer\s+/i, "").trim();
    if (!token) return json({ error: "missing_token" }, 401);

    let tokenMap;
    try {
      tokenMap = JSON.parse(env.EMPLOYEE_TOKENS || "{}");
    } catch {
      return json({ error: "server_misconfigured_tokens" }, 500);
    }
    const tokenInfo = tokenMap[token];
    if (!tokenInfo) return json({ error: "invalid_token" }, 401);

    const user = tokenInfo.user || "unknown";
    const capUsd = Number(tokenInfo.cap_usd ?? 25);

    let body;
    try {
      body = await request.json();
    } catch {
      return json({ error: "invalid_json_body" }, 400);
    }

    const allowedModels = parseList(env.ALLOWED_MODELS, DEFAULT_ALLOWED_MODELS);
    const model = body.model;
    if (!model || !allowedModels.includes(model)) {
      return json({ error: "model_not_allowed", model, allowed: allowedModels }, 403);
    }

    const month = new Date().toISOString().slice(0, 7); // "2026-04"
    const spendKey = `spend:${user}:${month}`;
    const currentSpend = Number((await env.USAGE_KV.get(spendKey)) || 0);
    if (currentSpend >= capUsd) {
      return json({
        error: "monthly_cap_exceeded",
        user,
        cap_usd: capUsd,
        spent_usd: round(currentSpend, 4),
      }, 429);
    }

    const upstreamStart = Date.now();
    const upstream = await fetch(`${OPENAI_API}${url.pathname}${url.search}`, {
      method: request.method,
      headers: {
        "authorization": `Bearer ${env.OPENAI_MASTER_KEY}`,
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
    });
    const upstreamMs = Date.now() - upstreamStart;

    const upstreamText = await upstream.text();
    let upstreamJson = null;
    try { upstreamJson = JSON.parse(upstreamText); } catch { /* non-json error body */ }

    if (upstream.ok && upstreamJson?.usage) {
      const cost = estimateCost(model, upstreamJson.usage);
      const newSpend = round(currentSpend + cost, 6);
      // KV writes are best-effort; eventual consistency is fine at 10 employees.
      await env.USAGE_KV.put(spendKey, String(newSpend));

      console.log(JSON.stringify({
        level: "info",
        event: "openai_call",
        user,
        model,
        input_tokens: upstreamJson.usage.prompt_tokens,
        output_tokens: upstreamJson.usage.completion_tokens,
        cost_usd: round(cost, 6),
        spend_this_month_usd: newSpend,
        upstream_ms: upstreamMs,
      }));
    } else {
      console.log(JSON.stringify({
        level: "warn",
        event: "openai_call_non2xx_or_no_usage",
        user,
        model,
        status: upstream.status,
        upstream_ms: upstreamMs,
      }));
    }

    return new Response(upstreamText, {
      status: upstream.status,
      headers: { "content-type": upstream.headers.get("content-type") || "application/json" },
    });
  },
};

function estimateCost(model, usage) {
  const p = MODEL_PRICING[model];
  if (!p) return 0;
  const inputCost = (usage.prompt_tokens || 0) / 1_000_000 * p.input;
  const outputCost = (usage.completion_tokens || 0) / 1_000_000 * p.output;
  return inputCost + outputCost;
}

function parseList(raw, fallback) {
  if (!raw) return fallback;
  return String(raw).split(",").map(s => s.trim()).filter(Boolean);
}

function round(n, digits) {
  const m = Math.pow(10, digits);
  return Math.round(n * m) / m;
}

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "content-type": "application/json" },
  });
}
