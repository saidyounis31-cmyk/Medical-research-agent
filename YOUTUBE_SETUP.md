# YouTube AI — staging configuration and limitations

This feature is implemented on a development branch only. Do not merge or deploy publicly without authorization and abuse protection.

## Required server-only environment variables
- `YOUTUBE_API_KEY`: Google Cloud API key with **YouTube Data API v3** enabled. Restrict the key to that API and appropriate server egress restrictions.
- `OPENAI_API_KEY`: OpenAI API key used for transcript analysis.
- `OPENAI_MODEL`: optional, defaults to `gpt-4.1-mini`.

Never add secrets to the repository or HTML. Set values in Render > Service > Environment only after approval.

## Supported workflow
1. Search by topic, keywords, or video URL via `GET /api/youtube/search?q=...`.
2. Select a result and watch it in the embedded YouTube player.
3. Supply a transcript you are authorized to use (optionally with timestamps).
4. Summarize or ask questions through `POST /api/youtube/analyze`.

The **official YouTube Data API v3 does not provide unrestricted public caption text**. Captions downloading requires authorization and appropriate ownership/permissions. This version **does not automatically fetch arbitrary video transcripts or audio**. It deliberately refuses to claim otherwise.

## Pre-release blockers
- Add per-IP and per-user server-side rate limiting and request-cost quotas to AI endpoints; public anonymous OpenAI requests can incur charges.
- Use persistent, permission-aware caching if authorized transcripts are stored; avoid caching sensitive transcript material without consent.
- Run `python -m unittest -v test_youtube_service.py`, then integration tests with configured API keys in an isolated staging environment.
- Verify YouTube quota, privacy disclosures, API terms, CSP compatibility, and Render deployment behavior.
- No production deployment or merge is authorized.

## Notes
The current summary is based on a supplied transcript, not independently verified video content. Timestamp links require actual timestamp data; the model is instructed not to fabricate them.
