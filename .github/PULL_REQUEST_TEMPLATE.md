<!--
Thank you for submitting to The Clarification Challenge!

Please read CONTRIBUTING.md first and fill in every section below.
Title your pull request:  Submission: <Algorithm name> (<Team name>)

For a non-submission pull request (bug fix, docs), delete everything below
and describe your change instead. Prefix the title with "Fix:" or "Docs:".
-->

## Submission type

- [ ] **New submission** — adds `clarify/algorithms/<name>.py`
- [ ] **Update of an existing submission** — replaces our earlier file
- [ ] **Leaderboard update only** — adds/updates our row in `docs/data/leaderboard.csv` (`confirmed=false`)

## Team

| | |
| --- | --- |
| **Team name** | <!-- exactly as in the submission header --> |
| **Team members** | <!-- Name (Affiliation), Name (Affiliation), ... --> |
| **Main contact** | <!-- name and email --> |
| **License** | <!-- SPDX identifier used in the header, e.g. MIT --> |

## Algorithm

**File:** `clarify/algorithms/<name>.py`
**Class name:** `<ClassName>` <!-- the single public subclass of ClarificationAlgorithmBase -->

**Summary** (2–5 sentences: what does the algorithm do, when does it ask, how does it decide?)

<!-- ... -->

**Core mechanism** (optional, a bit more detail: pipeline stages, prompts, how `env.llm`, `env.ask_human`, and `env.exec_code` are used, stopping criteria)

<!-- ... -->

**Related work / inspiration** (optional: papers, existing systems)

<!-- ... -->

## Algorithm parameters and defaults

List every key of `DEFAULT_CONFIG` (everything read from `self.config`). The defaults below are the configuration that will be evaluated.

| Parameter | Default | Description |
| --- | --- | --- |
| `example_param` | `5` | <!-- what it controls --> |

- [ ] The defaults in this table match `DEFAULT_CONFIG` in the code.
- [ ] No parameter is read from environment variables, files, or hard-coded elsewhere.

## Evaluation settings used for the reported results

Exact `generate_responses.py` options used to produce the numbers below (fill in even if you used the defaults).

| Option | Value |
| --- | --- |
| `--language_model` | `openai/gpt-4.1-mini` |
| `--temperature` | `0.7` |
| `--clarification_model` | *(default)* |
| `--max_clarification_turns` | `1` |
| `--max_prompt_budget` | `1.0` |
| `--max_clarification_budget` | `1.0` |
| `--num_samples` | `1` |
| `--split` | `val` |
| Other options / algorithm overrides | *(none)* |

**Exact commands run:**

```bash
python generate_responses.py clarify/algorithms/<name>.py --split val --output_path ...
python evaluate_responses.py --split val --generation_path ... --output_path ...
```

## Validation results

Results of `evaluate_responses.py` on the public validation split (770 tasks).

| TDS | nDCG | Pass@1 | Clarification rate | Over-asking rate | Avg. cost / task (USD) |
| --- | --- | --- | --- | --- | --- |
| | | | | | |

**Generation / result files:** <!-- link to the *_val.jsonl and *_val_results.jsonl files (e.g. a release asset, Zenodo, or a gist). Do not commit them to the repository. -->

## Leaderboard

- [ ] No leaderboard change in this pull request (organizers will add a confirmed row after reproduction), **or**
- [ ] `docs/data/leaderboard.csv` contains exactly one added/updated row for our algorithm, generated with `evaluate_responses.py ... --submit "<Team name>"`, with `confirmed=false`
- [ ] No other rows (baselines, other teams) and no header were modified
- [ ] The `model` column matches `--language_model` above

## Submission checklist

- [ ] The pull request adds/changes only `clarify/algorithms/<name>.py` (and optionally our leaderboard row)
- [ ] The file starts with the SPDX header and the docstring contains `Team`, `Team Members`, and `Main Contact`
- [ ] Exactly one public class inherits from `ClarificationAlgorithmBase`; prototypes are prefixed with `_`
- [ ] All LLM calls, clarifications, and code execution go through `env.llm`, `env.ask_human` / `env.can_ask`, and `env.exec_code`
- [ ] Only the standard library and packages already in `pyproject.toml` are imported; `pyproject.toml` and `uv.lock` are unchanged (additional libraries were requested via a *Library request* issue: <!-- #issue or n/a -->)
- [ ] No model name is hard-coded; the algorithm runs with the default environment settings
- [ ] `python generate_responses.py clarify/algorithms/<name>.py --split val` runs without errors on a fresh `uv sync`
- [ ] We developed on the released splits only and did not tune on validation instances
- [ ] We have read [CONTRIBUTING.md](../CONTRIBUTING.md) and agree to the competition rules

## Notes for the organizers (optional)

<!-- Anything that helps reproduction: known nondeterminism, runtime, rate limits, external resources, etc. -->
