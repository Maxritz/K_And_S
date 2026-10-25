# learnings/

Durable, distilled knowledge from porting engagements. **This folder is committed.**

## The rule: findings vs learnings

| | `findings/` | `learnings/` |
|---|---|---|
| What | Raw run artifacts: timestamped `porting-report-*.md` from `autonomous_port`, `export_contribution` scratch | Distilled, hand-curated knowledge: what worked, what didn't, reusable patterns |
| Lifetime | Ephemeral — one per run, never committed | Durable — one file per engagement/topic, committed |
| Git | **Ignored** (see `.gitignore`) | **Tracked** |
| Purge | Purge freely at any time | Never purge; refine in place |

## Rules for agents and humans

1. **Never commit `findings/`.** It is runtime scratch; the tools recreate it on every
   run. It is git-ignored — do not remove that ignore entry, and do not `git add -f`
   files from it.
2. **One learning per engagement.** When a port finishes and something non-obvious was
   learned, write (or update) a single `learnings/<topic>.md` — distilled, vendor-honest,
   with measured facts only. A learning that repeats a report verbatim is not a learning.
3. **Purge before commit.** If `findings/` ever lands in `git status` (e.g. via
   `git add -A` before the ignore entry existed), remove it from the index:
   `git rm -r --cached findings/`.
4. **No timestamps in learning filenames.** A learning is revised in place, not
   re-created per run. Duplicated content under new names is waste (see commit
   `4f9a090`, which deleted 17 near-identical reports).
5. **Evidence labels carry over.** Learnings inherit the PASS / FAIL / NOT RUN /
   UNVALIDATED vocabulary — a claim without measured evidence is marked `UNVALIDATED`,
   not stated as fact.

## Existing learnings

- `TINY_VLLM_CPU_PORT_LEARNINGS.md` — tiny-vllm CPU port: what transfers, what drops
- `TINY_VLLM_CPU_WINDOWS_PORT_LEARNINGS.md` — tiny-vllm CPU/Windows: toolchain + BLAS reality
