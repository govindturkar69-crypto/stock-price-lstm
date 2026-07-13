# Security Review

This project was audited against the five "Emergent Security Checks"
(Gitleaks · Bearer · ECC Production Audit · Trail of Bits · ECC Security Review).

Those prompts target web apps with backends, databases, authentication, payments
and user accounts. This project is a **local machine-learning app**: a Streamlit
front end over a PyTorch pipeline that reads **public** market data from Yahoo
Finance. It has **no database, no authentication, no payment logic, no user
accounts, and collects no personal data**, so most checks are Not Applicable by
design. The findings below are honest about that.

## 1. Secret Leak Prevention (Gitleaks) — ✅ PASS
- Full scan for API keys, tokens, passwords, DB URLs, cloud credentials:
  **none found.** yfinance requires no API key, so the app has no secrets.
- No `.env` needed. Nothing sensitive is committed.
- `.gitignore` already excludes `.venv/`, `artifacts/`, model files and caches.

## 2. Personal Data Flow Audit (Bearer) — ✅ PASS (N/A)
- The app collects **no personal data** — inputs are a stock ticker and a start
  date only. No emails, phones, passwords, addresses, or payment info.
- No third-party data sharing beyond the read-only yfinance price request.
- No cookies / localStorage of user data; nothing to hash, redact, or delete.

## 3. Pre-Deploy Production Audit (ECC) — ⚠️ PARTIAL (hardened)
- **Env vars / DB / CORS / security headers / rate limiting:** N/A — there is no
  custom web server or database; Streamlit serves the app.
- **Client-facing errors (FIXED):** raw exception strings are no longer shown in
  the UI. Users get a generic message; the detailed error is written to the
  server log only. (`app.py` fetch/train error handlers.)
- **Debug code:** training prints epoch logs to the server console only (not the
  UI, not sensitive). No test/debug/admin endpoints exist.

## 4. Deep Security Audit — Complex Logic (Trail of Bits) — ⚠️ PARTIAL (hardened)
- **Auth / IDOR / JWT / payments / webhooks / SQL:** N/A — none of these exist.
- **Input handling (FIXED):** the ticker and start-date are user-controlled and
  are passed to yfinance. They are now validated before any external call:
  - ticker must match `^[A-Za-z0-9^][A-Za-z0-9.\-^=]{0,14}$` (allowlist),
  - start date must be ISO `YYYY-MM-DD`.
  This blocks shell/SQL/path/script-style junk (`; rm -rf /`, `<script>`,
  `../../etc/passwd`, `'; DROP TABLE`) from ever reaching the data layer.
- No SQL is used anywhere (pandas/NumPy only), so SQL injection is not possible.
- No user-supplied HTML is rendered; Streamlit escapes text by default.

## 5. Attacker's Perspective Review (ECC) — ✅ LOW attack surface
- The only external input is the ticker/date, now validated (see #4).
- No authentication to bypass, no data to exfiltrate, no privileged actions.
- Model artifacts (`.pt`) are produced locally and git-ignored.

## Summary of changes applied
| Area | Change |
|------|--------|
| Input validation | Allowlist ticker + ISO date validation before any fetch |
| Error handling | Generic user-facing errors; details to server logs only |

## Honest note
No AI audit replaces professional security testing. This app handles no money and
no sensitive data, so its risk is low — but if it were extended with accounts,
a database, or an API, checks 1–5 above should be re-run in full.
