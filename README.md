# LLM Eval Pipeline

## What this is

CI/CD for prompt behavior. The feature under test is a customer support email
classifier (category + one-sentence summary), but the pipeline itself is
feature-agnostic: swap `src/classifier.py` for whatever LLM feature you're
shipping and everything downstream (scoring, diffing, alerting, drift
detection) keeps working.

Every time someone edits a file in `prompts/`, GitHub Actions runs the full
golden dataset through the new prompt, diffs the results against the last
recorded run, and posts a Slack alert plus a PR comment. Critical regressions
block the merge. This exists because prompt changes are otherwise shipped
blind -- someone tweaks a system prompt to fix one bug, it silently breaks
five other cases, and nobody notices until a customer complains.

## Architecture at a glance

```
prompts/classifier_vN.yaml   <- versioned "code" (system prompt, few-shot examples, model, temp)
golden_dataset/dataset_vN.json <- hand-labeled ground truth (30 cases: 5/category + 10 adversarial edge cases)
src/classifier.py            <- the LLM feature itself (swap this for your real feature)
src/eval_runner.py           <- runs the dataset through the feature, concurrently
src/scoring.py               <- category match (binary) + summary quality (LLM-as-judge, 1-5)
src/diff.py                  <- compares current run vs. last recorded run, flags regressions
src/drift.py                 <- catches gradual degradation that no single diff would flag
src/report.py                <- renders a self-contained HTML diff report
src/slack_alert.py           <- posts a summary + report link to Slack
src/storage.py                <- SQLite run history (data/eval_history.db)
run_eval.py                  <- CLI entry point; what CI actually invokes
```

Deliberately zero external infra: SQLite + JSON files for storage, GitHub
Actions for scheduling, a webhook for alerting. Nothing to provision.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY, optionally SLACK_WEBHOOK_URL
```

Run an eval locally:

```bash
python run_eval.py --prompt-version v2
```

First run establishes a baseline (nothing to diff against, severity is
always `OK`). Every run after that diffs against the most recent prior run
stored in `data/eval_history.db`.

## Adding test cases to the golden dataset

Edit `golden_dataset/dataset_v1.json` directly, or bump the version
(`dataset_v2.json`) if you're changing the eval bar itself and want the
change tracked. Each case needs:

```json
{
  "id": "bill_006",
  "input_email": "...",
  "expected_category": "billing",
  "expected_summary": "...",
  "difficulty": "easy | medium | hard | edge",
  "notes": "why this case exists"
}
```

**Do not generate expected labels with an LLM.** The entire premise of this
system is that the ground truth is more reliable than the thing being
graded. If you generate labels with a model, you're grading the model
against itself.

The recommended way to grow the dataset over time: whenever a regression or
a customer complaint surfaces a case the classifier got wrong, add that
exact case to the golden dataset with the correct label. The dataset should
grow from real failures, not synthetic guesses about what might fail.

## Adjusting thresholds

Two knobs, both in `src/diff.py` (or pass as CLI flags):

- `WARNING_THRESHOLD` (default 3%): pass-rate delta that triggers a WARNING.
- `CRITICAL_THRESHOLD` (default 8%): pass-rate delta that triggers CRITICAL
  and fails the CI job (blocks merge).

There's also a bulk-flip override: **3 or more previously-passing cases
flipping to failing is always CRITICAL**, regardless of the percentage delta.
On a 30-case dataset, 3 flips is 10% -- big on a small dataset even if the
aggregate math looks tame. This matters more as your dataset grows past a
few hundred cases and percentage deltas start hiding small but real
regressions on a handful of important cases.

Drift detection thresholds are in `src/drift.py` (`DEFAULT_WINDOW=7`,
`DEFAULT_DRIFT_THRESHOLD=0.05`). It needs at least 14 runs of history before
it activates (7-run rolling window compared against the prior 7-run window).

## Composite pass/fail bar

A case "passes" if: no hard error, category is an exact match, AND the
LLM-judge summary score is >= 3/5. This bar lives in
`src/scoring.py::score_composite`. If your product bar changes (e.g. you
decide 3/5 summaries aren't actually good enough to ship), change it there
-- every downstream diff and alert inherits the new bar automatically.

## Running in Docker

```bash
docker build -t llm-eval-pipeline .
docker run --rm \
  -e OPENAI_API_KEY=sk-... \
  -e SLACK_WEBHOOK_URL=https://hooks.slack.com/... \
  -e PROMPT_VERSION=v2 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/reports:/app/reports \
  llm-eval-pipeline
```

Mount `data/` and `reports/` as volumes or you lose run history and reports
when the container exits.

## CI behavior

`.github/workflows/eval.yml` triggers on any PR touching `prompts/`,
`golden_dataset/`, or `src/`. It:

1. Restores eval history from cache (keyed on the base branch, so PRs diff
   against the base branch's last run, not a sibling PR's).
2. Detects which prompt version changed and runs the eval against it.
3. Uploads the HTML report as a build artifact.
4. Comments on the PR.
5. Fails the job (blocking merge) if severity is CRITICAL.

You'll need to set `OPENAI_API_KEY` and (optionally) `SLACK_WEBHOOK_URL` as
repo secrets for this to run.

## Design decision: why drift is tracked separately from per-run diffs

Per-run diffing catches sharp regressions -- someone edits a prompt and
breaks five cases immediately, and that shows up as a clear delta against
the last run. It does **not** catch gradual degradation, which is a real
failure mode with hosted models: a provider updates the weights behind a
model ID without a version bump, and your pass rate erodes half a point at
a time. No single run ever crosses `CRITICAL_THRESHOLD`, so per-run alerting
stays silent while quality quietly rots.

The fix isn't a different threshold -- it's a different question. Per-run
diffing asks "did *this* change break something." Drift detection asks "is
the rolling average worse than it was two windows ago," independent of any
single prompt or code change. That's why it runs on the full run history
after every eval, not just on the current-vs-baseline pair.

## Known limitations / honest gaps

- The golden dataset here is 30 cases. Production systems should aim for
  100+ before trusting percentage deltas on rare categories.
- LLM-as-judge scoring has its own variance; it's graded with `gpt-4o-mini`
  at temperature 0, but it's still a model judging a model. Spot-check judge
  reasoning (`CaseScore.judge_reasoning`) periodically.
- Statistical significance here is a simple magnitude + bulk-flip heuristic,
  not a proper significance test (e.g. McNemar's test for paired binary
  outcomes). That's a reasonable next iteration if this dataset grows large
  enough for it to matter.
