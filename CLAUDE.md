# ATLAS Intelligence Platform

## Core Vision
Signal convergence platform. The edge: congressional trades + insider buying + institutional flows pointing at the same ticker simultaneously, especially when legislation is in motion. That convergence is the highest-conviction signal.

**Live:** https://atlas-intelligence-wheat.vercel.app | **Repo:** https://github.com/Hsalazar023/atlas-intelligence

---

## Project Map

| What you need | Where to find it |
|---|---|
| Active todos + immediate priorities | `docs/todo.md` |
| Long-term phases + general backlog | `docs/roadmap.md` |
| Scoring tables, decay, zone logic | `docs/scoring-logic.md` |
| Brain architecture + pipeline + exports | `docs/brain.md` |
| Brain health, IC, data quality, actions | `docs/brain-status.md` (sectioned — read specific sections) |
| ALE engine, ML pipeline, signal schema | `docs/ale-engine.md` |
| Stack, file locations, JS function map | `docs/architecture.md` |
| Completed milestones (historical) | `docs/archive/completed-milestones.md` |

---

## Key Commands

```bash
python3 -m http.server 8080              # Serve locally
python scripts/fetch_data.py             # Data pipeline
python backtest/learning_engine.py --daily      # Daily collect + backfill
python backtest/learning_engine.py --analyze    # Feature analysis + ML + weight update
python backtest/learning_engine.py --summary    # Status + leaderboards
python backtest/learning_engine.py --diagnostics # Generate HTML dashboard + analysis report
python backtest/bootstrap_historical.py         # One-time historical bootstrap
cd backtest && pytest tests/ -v                 # Tests
git add . && git commit -m "msg" && git push    # Deploy
```

---

## Guardrails
- **No hardcoded trade ideas.** All signals engine-generated. Static entries labeled `[DEMO]`.
- **No keys in frontend.** API keys go to env vars / Vercel secrets.
- **Never commit** `data/`, `Skills/`, `.claude/`, `.firecrawl/` — all gitignored.

---

## Working Rules

### Token Efficiency (CRITICAL)
- **Don't re-read files you've already read** in this session. Use what's in context.
- **Don't re-read large files to find one thing.** Use Grep with line numbers first, then Read only the relevant section with offset/limit.
- **Don't read data files** (`data/*.json`, `data/*.db`, price history). Check structure with quick queries or `head` if needed.
- **Don't launch agents** without checking with the user first. Prefer working through things sequentially.
- **Keep docs concise.** Every doc line costs tokens on every session load. Remove verbose explanations, keep facts and references.
- **Don't duplicate information** across CLAUDE.md, MEMORY.md, and docs/. Single source of truth for each topic.
- **Use Grep before Read.** Find the exact line numbers you need, then read only that section.

### .gitignore + .claudeignore
Keep both files in sync. When adding new data outputs or generated files, add to **both** immediately. `data/` contents are fully ignored.

### Todos & Progress
- `docs/todo.md` = current priorities (update each session)
- `docs/roadmap.md` = long-term phases (update when priorities shift)
- Completed items → `docs/archive/completed-milestones.md`

### Don't Run — User Runs Locally
Never run data pipelines, bootstrap, ML training, or pytest. Write the code — user runs it.

### Archive Policy
Old scripts, outdated docs, completed plans → `docs/archive/` (gitignored + claudeignored).

---

## Environment
- No `sudo` — use `npm install -g --prefix ~/.npm-global`
- Vercel CLI: `~/.npm-global/bin/vercel`
- Git committer not globally configured — warning on commits is fine
