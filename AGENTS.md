# AGENTS.md

## Agent skills

### Issue tracker

Issues are tracked as GitHub issues (campagc/HelpMeeting) via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical triage roles use their default label strings. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

## Gemini model & free-tier limits

| Key | Value |
|-----|-------|
| Model constant | `gemini-2.5-flash` |
| Previous (deprecated, shut down 2026-06-01) | `gemini-2.0-flash` |
| Free-tier RPM | 10 |
| Free-tier TPM (input) | 250,000 |
| Free-tier RPD | 250 |
| Context window | 1M tokens |
| Input pricing (free tier) | Free of charge (text/image/video) |
| Output pricing (free tier) | Free of charge |
| Note | Free-tier data may be used to improve Google products |

Source: https://ai.google.dev/gemini-api/docs/pricing (confirmed 2026-06-25).
