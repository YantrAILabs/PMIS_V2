# ProMe OpenAI Proxy (Cloudflare Worker)

A tiny edge proxy that lets employee laptops call OpenAI without ever holding
your master key. Each laptop carries a per-employee access token. The proxy
validates the token, enforces an endpoint + model allowlist, tracks per-user
monthly spend, and forwards to OpenAI with your master key.

## Threat model

**Prevents:**
- Employee reading `.env` and reusing the key for personal OpenAI projects — they only have a proxy token.
- Key leak blast radius — revoke one token, everyone else keeps working.
- Employee using the proxy token for non-tracker queries — model + endpoint allowlist blocks `/v1/assistants`, image gen, non-allowed models.
- Runaway spend — per-user monthly cap returns 429.

**Does NOT prevent:** a determined employee reading the proxy token out of `.env` and making arbitrary `gpt-4o-mini` chat calls via the proxy URL. This is bounded by the monthly cap and visible in logs.

## One-time setup

You need a Cloudflare account (free tier is plenty for 10 employees).

```bash
cd proxy
npm install
npx wrangler login

# 1. Create the KV namespace that tracks monthly spend
npx wrangler kv:namespace create USAGE_KV
# -> paste the returned `id` into wrangler.toml (replace REPLACE_WITH_KV_NAMESPACE_ID)

# 2. Set the master OpenAI key (never commit this)
npx wrangler secret put OPENAI_MASTER_KEY
# paste your sk-... when prompted

# 3. Generate per-employee tokens
cp tokens.example.json tokens.json
# edit tokens.json — replace placeholders with real tokens. Generate like:
#   python3 -c "import secrets; print('prome_' + secrets.token_hex(16))"

# 4. Upload token map
npx wrangler secret put EMPLOYEE_TOKENS < tokens.json

# 5. Deploy
npx wrangler deploy
# -> note the URL (e.g. https://prome-openai-proxy.<your-account>.workers.dev)
```

## Using the proxy from the tracker

On each employee laptop, set these in `productivity-tracker/.env`:

```
OPENAI_API_KEY=prome_<that-employee's-token>
OPENAI_BASE_URL=https://prome-openai-proxy.<your-account>.workers.dev/v1
```

The OpenAI Python SDK reads both env vars natively — no code changes needed.
The tracker will route all calls through the proxy.

## Daily ops

**Add a new employee:**
1. Generate a new token: `python3 -c "import secrets; print('prome_' + secrets.token_hex(16))"`
2. Add to `tokens.json` with `{ "user": "name", "cap_usd": 25 }`
3. Re-upload: `npx wrangler secret put EMPLOYEE_TOKENS < tokens.json`
4. Hand the new token to the employee (they paste it into their `.env`)

**Revoke a token:**
1. Remove the line from `tokens.json`
2. `npx wrangler secret put EMPLOYEE_TOKENS < tokens.json`
3. Takes effect globally within seconds.

**Change an employee's cap:**
Edit their `cap_usd` in `tokens.json`, re-upload.

**See what everyone is doing:**
```bash
npx wrangler tail
```
Each OpenAI call produces a JSON log line with user, model, token counts, cost, and this-month's running spend.

**Reset monthly spend (end of month / manual):**
```bash
npx wrangler kv:key delete --namespace-id=<your-id> "spend:<user>:2026-04"
```
Or just wait — new month → new key.

## Cost

- Cloudflare Worker free tier: 100k requests/day. 10 employees × ~500 tracker calls/day = 5k/day. You're fine.
- KV free tier: 100k reads/day, 1k writes/day. Covered.
- OpenAI cost: bounded by the sum of all `cap_usd` values. 10 employees × $25 = $250/mo ceiling.

## Security notes

- `tokens.json` is gitignored. **Do not commit it.**
- `OPENAI_MASTER_KEY` lives in Cloudflare secrets — not in git, not readable from worker logs.
- Employee tokens are rotated by editing `tokens.json` + re-uploading; no DB migration needed.
- HTTPS-only (Cloudflare default). Employee tokens travel in `Authorization: Bearer ...` headers.
- The worker does NOT log prompt content or responses — only metadata (user, model, tokens, cost). If you want content logging for abuse investigation, add it carefully and mind employee privacy.
