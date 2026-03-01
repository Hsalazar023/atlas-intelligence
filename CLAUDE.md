# ATLAS Intelligence Platform

## Core Vision
Signal convergence platform. The edge: congressional trades + insider buying + institutional flows pointing at the same ticker simultaneously, especially when legislation is in motion. That convergence is the highest-conviction signal.

**Live:** https://atlas-intelligence-wheat.vercel.app | **Repo:** https://github.com/Hsalazar023/atlas-intelligence

---

## Project Map — Section Index

Use `Grep` to find section headers, then `Read` with offset/limit. Never read entire docs.

### docs/todo.md
| Section | What's there |
|---|---|
| `## Current System Health` | Signal counts, IC, hit rate, score range |
| `## P0 — Immediate Fixes` | Blocking bugs and data issues |
| `## P1 — Brain Improvements` | Scoring calibration, self-check, pruning |
| `## P2 — Frontend & UX` | Card context, dashboards, demo removal |

### docs/roadmap.md
| Section | What's there |
|---|---|
| `## Phase 2 — Autonomous Brain Loop` | CI/CD, calibration, self-monitoring, features |
| `## Phase 3 — Frontend Intelligence` | Explainability, dashboards, content cleanup |
| `## Phase 4 — New Data Sources` | 13F, Congress.gov API, news, earnings, options |
| `## Phase 5 — UI/UX Redesign` | Design system, mobile, Next.js migration |
| `## Phase 6 — Notifications` | Ntfy, email digest, in-app alerts |
| `## Phase 7 — Auth & Monetization` | Clerk, Supabase, Stripe |
| `## Running Cost Targets` | Monthly cost by phase |

### docs/brain.md
| Section | What's there |
|---|---|
| `## What the Brain Is` | One-liner definition |
| `## Pipeline` | --daily and --analyze step-by-step flows |
| `## Export Files` | brain_signals.json, brain_stats.json, optimal_weights.json |
| `## Next Capabilities` | Planned features table |

### docs/brain-status.md
| Section | What's there |
|---|---|
| `## Latest Run` | Pre/post cleanup metrics, score tier counts |
| `## Data Quality` | Feature fill rates, source quality (EDGAR vs Congress) |
| `## Convergence` | Tier distribution |

### docs/ale-engine.md
| Section | What's there |
|---|---|
| `## CLI` | All --flags and one-time scripts |
| `## Database` | SQLite file, tables, diagnostics paths |
| `## ML Engine` | Models, validation, CAR, safety rail |
| `## Feature List` | 28 features by category (v4) |
| `## Data Quality` | EDGAR buy filtering, XML enrichment |
| `## Bootstrap Pipeline` | 11-step historical data load |

### docs/scoring-logic.md
| Section | What's there |
|---|---|
| `## Brain ML Score` | Scoring formula + score tiers |
| `## Convergence Tiers` | Tier 0/1/2 conditions |
| `## Frontend Heuristic Score` | Hub 1 (Congress), Hub 2 (Insider), boosts |
| `## Signal Decay` | Decay formula + half-lives |
| `## Entry Zone Logic` | Entry/target/stop formulas |

### docs/architecture.md
| Section | What's there |
|---|---|
| `## Current Stack` | All layers + status |
| `## Key Files` | Every important file + purpose |
| `## Key JS Functions` | Frontend function map |
| `## GitHub Actions` | Workflow schedules |
| `## API Keys` | Key sources + locations |

### atlas-intelligence.html (~3800 lines — always use offset/limit)
| Section | Lines | What's there |
|---|---|---|
| HTML + CSS | 1–1154 | Markup, styles, static structure |
| `ATLAS LIVE DATA ENGINE` | ~1156 | TRACKED object, globals, config |
| `DYNAMIC SCORING WEIGHTS` | ~1166 | Score weight constants |
| `TICKER → SECTOR MAP` | ~1182 | Client-side sector lookup |
| `FINNHUB LIVE PRICES` | ~1304 | fetchPrice, updatePriceStrip, updateIdeaCard |
| `MARKET INDICES` | ~1532 | SPY/QQQ/IWM/VIX proxies |
| `CONGRESSIONAL TRADES` | ~1577 | renderCongressTrades, fetchCongressTrades |
| `MARKET CONTEXT` | ~1699 | VIX + Treasury data loaders |
| `SEC EDGAR FORM 4` | ~1814 | fetchEdgarFeed, renderEdgarFeed |
| `CONVERGENCE SCORING ENGINE` | ~2206 | scoreCongressTicker, scoreEdgarTicker, computeConvergenceScore |
| `renderTopSignals` | ~2365 | Top signals table |
| `renderLiveAlerts` | ~2426 | Live alerts feed |
| `updateOverviewKPIs` | ~2493 | KPI strip computation |
| `renderSignalIdeas` | ~2590 | Trade idea cards (score bar, factors, targets) |
| `LEGISLATIVE CALENDAR` | ~2910 | BILLS array rendering |
| `SECTOR SIGNAL HEAT` | ~2957 | Sector heatmap from live data |
| `CONGRESS.GOV BILL TRACKING` | ~3053 | Congress.gov API integration |
| `NAVIGATION & UI` | ~3188 | goPage, animateBars |
| `FILTERS` | ~3214 | filterCong, filterIns, filterIdeas |
| `DYNAMIC WEIGHTS LOADER` | ~3344 | loadOptimalWeights |
| `BRAIN DATA LOADERS` | ~3373 | loadBrainSignals, loadBrainStats, loadBrainHealth |
| `INITIALIZATION` | ~3761 | Startup sequence, DOMContentLoaded |

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
- **Don't re-read large files to find one thing.** Use Grep to find the section header, then Read with offset/limit.
- **Don't read data files** (`data/*.json`, `data/*.db`, price history). Check structure with quick queries or `head` if needed.
- **Don't launch agents** without checking with the user first. Prefer working through things sequentially.
- **Keep docs concise.** Every doc line costs tokens on every session load. Remove verbose explanations, keep facts and references.
- **Don't duplicate information** across CLAUDE.md, MEMORY.md, and docs/. Single source of truth for each topic.
- **Section-first lookup:** Check the Section Index above → Grep for the `## Section` header → Read only that section with offset/limit.

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
