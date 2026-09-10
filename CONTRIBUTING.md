# Contributing to The Clarification Challenge

Thank you for participating! This document explains how to submit a clarification system, what a valid submission looks like, and how to (optionally) add your results to the public leaderboard.

Participation is a single pull request against this repository. Bug reports and SDK fixes are welcome as separate pull requests (see [Other contributions](#other-contributions)).

- [Key dates](#key-dates)
- [Submission overview](#submission-overview)
- [Step 1 — Fork and install](#step-1--fork-and-install)
- [Step 2 — Implement your algorithm](#step-2--implement-your-algorithm)
- [Step 3 — Declare parameters and defaults](#step-3--declare-parameters-and-defaults)
- [Step 4 — Evaluate on the validation split](#step-4--evaluate-on-the-validation-split)
- [Step 5 — Update the leaderboard (optional)](#step-5--update-the-leaderboard-optional)
- [Step 6 — Open the pull request](#step-6--open-the-pull-request)
- [What happens after you submit](#what-happens-after-you-submit)
- [Rules](#rules)
- [Other contributions](#other-contributions)

## Key dates

| Date (23:59 AoE) | Milestone |
| --- | --- |
| Sep 11, 2026 | Competition opens |
| **Oct 9, 2026** | **Registration deadline** — open a pull request (a draft is fine) to register your team |
| **Nov 6, 2026** | **Submission deadline** — final version of your pull request |
| After the deadline | Organizers evaluate eligible systems on the hidden test set |
| Nov 20, 2026 | Final results published |
| Dec 4, 2026 | ICSE 2027 Competition Track solution papers due (ranked teams only) |

The authoritative timeline is on the [competition website](https://serval-uni-lu.github.io/clarification-competition/#timeline).

## Submission overview

A submission is **one pull request** that adds **one Python file**:

```
clarify/algorithms/<your_algorithm>.py
```

The file must contain exactly one (non-underscored) class inheriting from `ClarificationAlgorithmBase`, a valid submission header, and the parameter defaults you used for evaluation. Optionally, the same pull request may add a row for your system to `docs/data/leaderboard.csv`.

The pull request description follows our [pull request template](.github/PULL_REQUEST_TEMPLATE.md), which asks for a description of the algorithm, your team, and the exact configuration used for the reported results.

Use `clarify/algorithms/demo.py` as a reference for a complete, valid submission.

## Step 1 — Fork and install

1. Fork this repository and clone your fork.
2. Install the SDK (see the [README](README.md#installation)):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv venv --python 3.13 && source .venv/bin/activate
   uv sync
   ```

3. Export the API key(s) for the LiteLLM provider(s) you use, and pull the code-execution image:

   ```bash
   docker pull ganler/evalplus
   ```

4. Create a branch for your submission, e.g. `submission/<team-name>`.

## Step 2 — Implement your algorithm

Create `clarify/algorithms/<your_algorithm>.py`. Every valid submission file:

1. **Starts with the submission header.** The SPDX lines license your code (any OSI-approved open-source license), and the module docstring identifies the algorithm and the team:

   ```python
   # SPDX-FileCopyrightText: 2026 Your Name <you@example.com>
   #
   # SPDX-License-Identifier: MIT

   """Algorithm name.

   Short description of the main mechanism of the implemented algorithm.
   This does not need to go into details, but should capture the core idea.

   Team: Your Team Name
   Team Members: Member 1, Member 2, ...
   Main Contact: main.contact@example.com
   """
   ```

   All three fields (`Team`, `Team Members`, `Main Contact`) are required. The team name is used on the leaderboard.

2. **Defines exactly one public class inheriting from `ClarificationAlgorithmBase`** (`clarify.baselines.ClarificationAlgorithmBase`) and implements `run(self, env, problem) -> str`. Helper classes that also inherit from the base class must be prefixed with an underscore (`_Prototype`), otherwise auto-discovery fails. The class name is used as the algorithm name on the leaderboard, so choose a distinctive one (not `LLMClarification`, `Okanagan`, or `ClarifyGPT`).

3. **Interacts with the environment only through the SDK.** Use `env.llm(...)` for model calls, `env.ask_human(...)` / `env.can_ask()` for clarification, and `env.exec_code(...)` for code execution. Do not call LLM providers, run subprocesses, or access the network directly; such submissions cannot be scored consistently and are disqualified.

4. **Is self-contained.** Everything must live in the single file. You may import the Python standard library and the packages already listed in `pyproject.toml` (e.g. `numpy`, `litellm`). Do not modify `pyproject.toml`, `uv.lock`, or any file under `clarify/` other than your own algorithm file. If your approach needs an additional library, see [Requesting additional libraries](#requesting-additional-libraries).

5. **Returns the implementation as a string.** The result should contain the implementation or a fenced python block containing the implementation of `problem['entry_point']`. `clarify.runtime._validate_and_parse_evalplus_result` helps you check whether a model response already contains a valid answer.

Try your algorithm interactively with `python chat.py clarify/algorithms/<your_algorithm>.py`.

## Step 3 — Declare parameters and defaults

Organizers must be able to reproduce your reported results *and* run your system on the hidden test set without guessing settings. There are two kinds of parameters.

### Algorithm parameters (yours)

Anything your algorithm can be configured with (number of samples, clustering thresholds, prompt variants, ...) is read from `self.config`, a plain dictionary that `ClarificationAlgorithmBase` receives in its constructor. Extra command-line options of `generate_responses.py` are collected into this dictionary, which lets you (and us) override parameters for experiments without touching the code:

```bash
python generate_responses.py clarify/algorithms/my_clarification.py --split val --num_candidates 8
```

Declare **all** parameters together with their defaults in a single `DEFAULT_CONFIG` class attribute and merge it with the overrides in the constructor. The defaults are the configuration that is evaluated:

```python
class MyClarification(ClarificationAlgorithmBase):
    DEFAULT_CONFIG = {
        "num_candidates": 5,  # candidate implementations sampled per round
        "max_tests": 10,  # LLM-generated tests used for clustering
    }

    def __init__(self, config=None):
        super().__init__({**self.DEFAULT_CONFIG, **(config or {})})

    def run(self, env, problem):
        k = self.config["num_candidates"]
        ...
```

Keep the number of parameters small, comment each one, and list them with their defaults in the pull request. Do not read parameters from environment variables or files; `self.config` is the only supported channel.

### Requesting additional libraries

Submissions may only import the standard library and the packages already in `pyproject.toml`. If your approach depends on another library, **open an issue** using the *Library request* template **before** submitting, stating the package, the version constraint, and why it is needed. Organizers review the request (license, maintenance, install footprint, safety of running it inside the evaluation environment) and, if approved, add it to `pyproject.toml` and `uv.lock` themselves.

Do not include dependency changes in your submission pull request; they are removed during review. Please request libraries well before the submission deadline so there is time for a decision, and expect requests received in the final days before the deadline to be declined. Until a request is approved, make sure your algorithm still runs without the library (e.g. an optional code path), otherwise it cannot be evaluated.

### Environment parameters (ours)

These are set on the command line of `generate_responses.py` and define the evaluation setting. Report the exact values you used in the pull request.

| Option | Default | Meaning |
| --- | --- | --- |
| `--language_model` | `openai/gpt-4.1-mini` | LiteLLM model string used by `env.llm` |
| `--temperature` | `0.7` | Sampling temperature for `env.llm` |
| `--clarification_model` | *(same as language model)* | Model used by the simulated human |
| `--max_clarification_turns` | `1` | Clarification budget per task (`ask_human` raises `TooManyQuestionException` beyond it) |
| `--max_prompt_budget` | `1.0` | Cost budget (USD) for `env.llm` per task |
| `--max_clarification_budget` | `1.0` | Cost budget (USD) for the simulated human per task |
| `--num_samples` | `1` | Number of samples per task |
| `--split` | — | `train` or `val` |

For the **final private-test evaluation, organizers fix the model and environment settings themselves** and run every eligible submission under the same setup. Your algorithm must therefore work with the default budget and turn settings; do not hard-code a specific model name inside your algorithm.

## Step 4  — Evaluate on the validation split (optional)

Generate responses and evaluate them on the public validation split (770 tasks):

```bash
python generate_responses.py clarify/algorithms/<your_algorithm>.py \
    --split val \
    --output_path data/<your_algorithm>_val.jsonl \
    --max_workers 8

python evaluate_responses.py \
    --split val \
    --generation_path data/<your_algorithm>_val.jsonl \
    --output_path data/<your_algorithm>_val_results.jsonl
```

`evaluate_responses.py` prints TDS, nDCG, Pass@1, clarification rate, and over-asking rate. Keep the generated `*.jsonl` files: you will attach or link them in the pull request, and organizers use them to check the leaderboard row. Do **not** commit result files to the repository.

Develop on `--split train` and use `--split val` for your reported numbers. Never evaluate on data outside the released splits, and do not tune on validation instances.

## Step 5 — Update the leaderboard (optional)

You may add your validation result to the public leaderboard in the same pull request. Run the evaluation with `--submit` set to your team name (as written in the submission header):

```bash
python evaluate_responses.py \
    --split val \
    --generation_path data/<your_algorithm>_val.jsonl \
    --output_path data/<your_algorithm>_val_results.jsonl \
    --submit "Your Team Name"
```

This appends (or replaces, for the same algorithm/model pair) one row in `docs/data/leaderboard.csv` with `confirmed=false`. Rows with `confirmed=false` are shown on the website with an **⚠ Unconfirmed** badge until an organizer has reproduced the run and flipped the flag.

Rules for leaderboard edits:

- Only the row for your own algorithm may be added or changed. Do not edit baseline rows, other teams' rows, or the header.
- Leave `confirmed` as `false`. Never pass `--trusted`; only organizers set `confirmed=true`.
- One row per algorithm/model pair. Re-running `--submit` replaces your previous row.
- The row must come from a full run over the validation split; runs with missing tasks are rejected by the script and by review.
- Rows generated with a model other than the default (`openai/gpt-4.1-mini`) are welcome (see the FAQ on the website), but the model must be stated in the `model` column exactly as passed to `--language_model`.

If you do not want a public row before verification, skip this step; organizers add a confirmed row after reproducing your run.

## Step 6 — Open the pull request

1. Push your branch and open a pull request against `main` of `serval-uni-lu/clarification-competition`.
2. Title it `Submission: <Algorithm name> (<Team name>)`.
3. Fill in every section of the [pull request template](.github/PULL_REQUEST_TEMPLATE.md): algorithm description, team, algorithm parameters and defaults, environment settings, validation results, and the leaderboard checklist.
4. Open the pull request **before the registration deadline** (a draft with a stub file is enough to register) and push your final version **before the submission deadline**. Commits after the deadline are ignored for the final evaluation.

A pull request must only contain your algorithm file and, optionally, your leaderboard row. Unrelated changes are moved to a separate pull request.

## What happens after you submit

1. **Validity check.** An organizer confirms the header, the single public class, the SDK-only interaction, and that `python generate_responses.py <file> --split val` starts without errors. Problems are raised as review comments; please respond before the submission deadline.
2. **Reproduction.** Organizers re-run your system on the validation split with the settings from your pull request. If the numbers match, the leaderboard row is marked `confirmed=true` and the pull request is merged.
3. **Final evaluation.** After the submission deadline, all merged systems are run on the hidden test set with the organizers' fixed model and setup. Private-test results are published on Nov 20, 2026.
4. **Ranking.** Systems are ranked by TDS, with nDCG as tie-breaker. Only systems that match or beat the strongest baseline receive a rank and are eligible to submit an ICSE 2027 solution paper.

## Rules

- One submission (one algorithm file) per team. To replace an earlier submission, update the same pull request rather than opening a new one.
- A person may be a member of only one team.
- Only the packages in `pyproject.toml` are available during evaluation. Additional libraries must be requested via an issue and approved by the organizers before the submission deadline.
- Your code is released under the open-source license in your SPDX header and will be published together with the competition results.
- Do not attempt to recover, memorize, or special-case hidden requirements or test cases; the simulated human is instructed not to leak them, and submissions that target the oracle rather than the task are disqualified.
- Organizers may exclude submissions that exceed the per-task budgets, fail to run, or violate these rules. Decisions are final.
- Be respectful in issues, reviews, and discussions.

## Other contributions

Bug fixes and improvements to the SDK, baselines, data loaders, or documentation are welcome as separate pull requests, clearly labeled as such in the title (e.g. `Fix: ...`). Please open an issue first for larger changes. Changes to the evaluation scripts or metrics are only accepted from organizers during the competition to keep results comparable.

Questions? Open an issue or contact the organizers listed in the [README](README.md#organizers--contact).
