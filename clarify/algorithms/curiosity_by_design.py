# SPDX-FileCopyrightText: 2026 Thibaud <lutellie@ualberta.ca>
#
# SPDX-License-Identifier: MIT

"""Curiosity by Design.

Finds what a task leaves open by differential testing, and asks about that rather than
guessing. The idea is that an underspecified requirement shows itself as a disagreement
between independent readings of the same task: if two correct-looking implementations of the
same prompt behave differently on some input, the prompt did not decide what that input
should do, and that is worth one question.

How it works, per task:

1. Sample several candidate implementations of the prompt, each written for a different
   stated user persona, so the readings differ for reasons other than sampling temperature.
2. Ask the model for test inputs, then reconcile them against the candidates' real
   signatures - an input that fits no signature is rewritten where it only makes sense as a
   single argument, and dropped otherwise.
3. Execute every candidate on every input inside one `env.exec_code` call, recording the
   return value, exception, non-termination, argument mutation and anything printed.
4. Group the candidates by the behaviour observed across those inputs. A candidate that
   crashes on an input the others handle is a bug, not a reading, and is discarded while any
   clean candidate survives; an input that every candidate rejects says more about the input
   than about the task and is excluded from the comparison.
5. If every candidate agrees, widen the search once with inputs aimed further towards the
   boundaries before concluding that the prompt is unambiguous.

When it asks: whenever a budgeted question is available - on a disagreement, the input that
splits the candidates most evenly becomes a question about the behaviour behind it, drafted
per difference, and the draft that best fits the judging criteria (one specific requirement,
answerable, not about implementation details or tests) is the one asked. When the candidates
agree, one question is still asked, drafted from the prompt alone, because a prompt that
several independent readings happen to agree on can still be underspecified.

How it decides: the answer is folded into the specification and the candidates are
regenerated from it without personas, since the point is now agreement rather than
diversity. Those are executed and grouped again, and the winner is the largest group of
candidates that fail no valid input. The final program is repaired once if it still crashes
on an input, and the unrepaired version is kept unless the repair is strictly better.

Degradation: `env.exec_code` needs Docker. Where it is unavailable, or the sandbox returns
nothing usable, `run` falls back to asking the model directly for the most critical question
about the prompt, so a result still comes back without an execution environment.

Team: Curiosity by Design
Team Members: Thibaud
Main Contact: lutellie@ualberta.ca
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import logging
import re
from typing import Any

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import (
    ClarificationEnvironment,
    LimitsExceededException,
    TooManyQuestionException,
)


def parse_fenced_code(result):
    """Extracts the single ```python block a code request asks for.

    This duplicates `clarify.runtime`'s own parser rather than importing it. That helper is
    private, and a submission that reaches into the harness for a private name breaks the
    moment the harness is refactored - at which point every task fails to import rather than
    failing gracefully. Ten lines of string splitting are cheaper than that risk.

    The message is part of the contract, not decoration: `ask_for_code` feeds it back to the
    model as the correction prompt when a response arrives unfenced, so it has to read as an
    instruction.
    """
    if "```python" not in result:
        raise ValueError("Expected the result to be enclosed in ```python and ```")

    _, result = result.split("```python", 1)

    if "```" not in result:
        raise ValueError("Expected the result to be enclosed in ```python and ```")

    result, _ = result.split("```", 1)
    return result


# One JSON record per task, for analysis on the train split. Silent unless a handler is
# attached, and nothing here opens a file: a personal runner script attaches the handler and
# then calls `generate_responses.main`. Note that `--max_workers > 1` runs each task in a
# spawned child process, so the handler has to be attached inside the worker (or the run has
# to be sequential) for the records to be captured.
LOGGER = logging.getLogger("curiosity_by_design")

# Prompts ----------------------------------------------------------------------------------

CODE_TEMPLATE = """
Write a self-contained Python script implementing `{entry_point}` that solves the following problem.

{prompt}

Resolve anything the problem leaves open with the choice you consider most natural.
Enclose your complete solution in a single block starting with ```python and ending with ```.
""".strip()

CODE_WITH_SPEC_TEMPLATE = """
Write a self-contained Python script implementing `{entry_point}` that solves the following problem.

{prompt}

### Clarifications from the author
{clarifications}

The clarifications take precedence over the problem text, including over its examples.
Enclose your complete solution in a single block starting with ```python and ending with ```.
""".strip()

INPUT_TEMPLATE = """
Below is a Python programming task whose function is called `{entry_point}`.

{prompt}

List {num_inputs} different argument tuples that `{entry_point}` could be called with.
Cover ordinary cases and edge cases: empty and one-element containers, zero, negative values,
duplicates, ties, and boundary values.

Rules:
- One call per line, as a Python tuple of the positional arguments, and nothing else.
- A single-argument call is written as a one-element tuple, e.g. `([1, 2, 3],)`.
- Use Python literals only: no variables, no function calls, no comments.
- Every input must be one the task considers valid.
""".strip()

DIFFERENCE_TEMPLATE = """
Two implementations of the same task disagree on one input. Your job is to work out which
requirement the task failed to pin down.

### Task
{prompt}

### Input
`{entry_point}{input_repr}`

### Observed behaviour
{behaviours}

### One representative implementation per behaviour
{programs}

Answer with exactly these three lines and nothing else:
DECIDED=<yes|no>
CHOICE=`<the behavioural choice that separates them, in one short phrase>`
QUESTION=`<one question to the task's author that settles the choice>`

- DECIDED=yes if the task text, as written, already determines which behaviour is correct
  (one implementation is simply wrong). Then CHOICE and QUESTION may be left empty.
- DECIDED=no if the task genuinely leaves the choice open.
- The question must be critical, impossible to guess from the task text, about exactly ONE
  fact, objective, and about the required behaviour.
- Name the concrete alternatives in the question.
- Never ask what the function should return for this specific input: that is asking for a
  test case and will not be answered.
- Write QUESTION in plain prose: no backticks, no code spans, no Markdown. The author
  mirrors the question's formatting in the reply, and a backtick there truncates the answer.
""".strip()

RANK_TEMPLATE = """
Pick the question that a busy author would answer most precisely.

### Task
{prompt}

### Candidate questions
{questions}

A good question is critical to a correct solution, cannot be guessed from the task text,
asks for exactly one fact, has a single objective answer, and is about required behaviour
rather than the implementation or its test cases.

Reply with the number of the best question and nothing else.
""".strip()

FALLBACK_TEMPLATE = """
Below is a Python programming task that may be incomplete, ambiguous or contradictory.

{prompt}

Find the single most severe problem with it: information a correct implementation needs but
the task never states, wording that admits more than one reading, a contradiction between the
description and its examples, or a requirement that looks wrong.

Then write one question to the author that resolves it. The question must be critical,
impossible to guess from the task text, about exactly ONE fact, objective, and about the
required behaviour rather than the implementation or its test cases.

If the task is genuinely unambiguous, reply with exactly NO_QUESTION.
Otherwise reply with the question alone, as one sentence, and nothing else.

Write it in plain prose: no backticks, no code spans, no Markdown. The author mirrors the
question's formatting in the reply, and a backtick there truncates the answer.
""".strip()

REPAIR_TEMPLATE = """
The implementation below is meant to solve this task.

{prompt}
{clarifications}
### Implementation
```python
{code}
```

Running it on inputs the task considers valid produced these failures:
{failures}

Those inputs were generated for this task and are expected to be ones it accepts, but that
is not guaranteed: if one of them is genuinely outside what the task describes, say so for
that input and leave the behaviour on it alone. For the rest, fix the defect while keeping
the behaviour on all other inputs exactly as it is, and do not change the intended semantics
to make a failure disappear.

Enclose your complete corrected solution in a single block starting with ```python and
ending with ```.
""".strip()

DIVERGENT_TEMPLATE = """
Below is a Python programming task whose function is called `{entry_point}`.

{prompt}

Here is one implementation of it:
```python
{candidate}
```

Read the task differently but still defensibly: find a point where the wording permits
another interpretation, and implement that reading instead. Do not introduce a bug; the
result must be a reasonable reading of the task as written.

Enclose your complete solution in a single block starting with ```python and ending with ```.
""".strip()

# User personas for sampling candidates (step 1). The persona describes the user who asks for
# the code, not a role the model should play, and is placed at the start of the user message.
# Starting points to experiment with, not a tested set; add further user descriptions here.
USER_PERSONAS = [
    "I'm a beginner who is still learning Python.",
    "I'm a senior software developer.",
]

# Sandbox harness (step 3). Runs every candidate on every input inside ONE `env.exec_code`
# call, because each call starts and tears down a Docker container. Standard library only.
# It is a plain string here, so nothing in it is subject to this file's own lint rules.
SANDBOX_SCRIPT = """
import ast, base64, contextlib, copy, io, json, signal, time

PAYLOAD = json.loads(base64.b64decode("__PAYLOAD__").decode("utf-8"))
ENTRY = PAYLOAD["entry_point"]
TIMEOUT = PAYLOAD["timeout"]
TOL = PAYLOAD["float_tol"]
BEGIN, END = "<<<CBD-BEGIN>>>", "<<<CBD-END>>>"

# `clarify/runtime.py` runs this script under a 30s subprocess timeout and the table is only
# printed at the end, so overrunning loses every row rather than the slow ones. Stop issuing
# calls in time to still print what has been collected.
DEADLINE = time.monotonic() + 20.0


def canon(value, depth=0):
    # JSON-able canonical form of an observable value. The type is part of the form, so 1,
    # 1.0 and "1.0" do not collapse; floats are rounded so noise does not split groups.
    if depth > 6:
        return ["deep", None]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", value]
    if isinstance(value, float):
        if value != value:
            return ["float", "nan"]
        if value in (float("inf"), float("-inf")):
            return ["float", str(value)]
        if not TOL:
            return ["float", value]
        # `round(value / TOL)` raises OverflowError once the quotient reaches infinity, and
        # INPUT_TEMPLATE asks for boundary values, so 1e308 is a realistic input. Compare
        # such a value unrounded rather than taking down the whole table.
        scaled = value / TOL
        if scaled != scaled or scaled in (float("inf"), float("-inf")):
            return ["float", value]
        return ["float", round(scaled) * TOL]
    if value is None:
        return ["none", None]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, (list, tuple)):
        name = "list" if isinstance(value, list) else "tuple"
        return [name, [canon(item, depth + 1) for item in value]]
    if isinstance(value, (set, frozenset)):
        return ["set", sorted(json.dumps(canon(item, depth + 1)) for item in value)]
    if isinstance(value, dict):
        items = [
            [json.dumps(canon(k, depth + 1)), json.dumps(canon(v, depth + 1))]
            for k, v in value.items()
        ]
        return ["dict", sorted(items)]
    return ["other", repr(value)]


class Alarm(Exception):
    pass


def on_alarm(signum, frame):
    raise Alarm()


signal.signal(signal.SIGALRM, on_alarm)

ARGUMENTS = []
for text in PAYLOAD["inputs"]:
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        ARGUMENTS.append(None)
        continue
    ARGUMENTS.append(parsed if isinstance(parsed, tuple) else (parsed,))

TABLE = []
for source in PAYLOAD["candidates"]:
    namespace, function, load_error = {}, None, None
    try:
        signal.setitimer(signal.ITIMER_REAL, TIMEOUT)
        exec(compile(source, "<candidate>", "exec"), namespace)
        function = namespace.get(ENTRY)
        if not callable(function):
            load_error = "EntryPointMissing"
    except BaseException as exc:
        load_error = type(exc).__name__
    finally:
        # Must be disarmed, or the timer armed for the import fires during a later call.
        signal.setitimer(signal.ITIMER_REAL, 0)

    row = []
    for arguments in ARGUMENTS:
        if load_error is not None:
            row.append({"kind": "load_error", "error": load_error})
            continue
        if arguments is None:
            row.append({"kind": "skipped"})
            continue
        if time.monotonic() > DEADLINE:
            # Out of time: leave the remaining cells unrun so the row keeps its width.
            row.append({"kind": "skipped"})
            continue

        call_arguments = copy.deepcopy(arguments)
        try:
            before = json.dumps(canon(call_arguments))
        except BaseException:
            # Unrepresentable argument: the mutation check is not worth losing the row for.
            before = None
        stream = io.StringIO()
        observation = None
        try:
            signal.setitimer(signal.ITIMER_REAL, TIMEOUT)
            with contextlib.redirect_stdout(stream):
                returned = function(*call_arguments)
            observation = {"kind": "value", "value": canon(returned)}
        except Alarm:
            observation = {"kind": "timeout"}
        except BaseException as exc:
            observation = {"kind": "error", "error": type(exc).__name__}
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)

        # Mutating the arguments in place is observable behaviour, so it belongs in the
        # signature. Printed output likewise.
        try:
            observation["mutated"] = (
                before is not None and json.dumps(canon(call_arguments)) != before
            )
        except BaseException:
            observation["mutated"] = False
        printed = stream.getvalue()
        if printed:
            observation["printed"] = printed
        row.append(observation)

    TABLE.append(row)

print(BEGIN)
print(json.dumps(TABLE))
print(END)
"""


class CuriosityByDesign(ClarificationAlgorithmBase):
    DEFAULT_CONFIG = {
        "num_candidates": 4,  # candidate programs sampled per task
        "use_personas": True,  # vary the persona per candidate; False = same prompt for all
        "num_inputs": 10,  # generated inputs used to compare the candidates
        "search_rounds": 2,  # rounds of looking for a distinguishing input before giving up
        "ask_when_agree": True,  # still ask one question when all candidates behave the same
        "max_questions": 1,  # questions asked per task (also limited by the environment)
        "generation_attempts": 3,  # retries when a response contains no usable implementation
        "max_differences": 3,  # distinguishing inputs turned into drafted questions
        "partition_inputs": True,  # separate informative inputs from malformed/rejected ones
        "input_retries": 1,  # re-ask for inputs when most do not fit the candidate signature
        "repair_attempts": 1,  # repair rounds for the final program when it fails on inputs
        "exec_timeout": 2.0,  # seconds allowed per candidate call inside the sandbox
        "float_tol": 1e-9,  # rounding applied to floats before comparing behaviour
    }

    # The base class merges DEFAULT_CONFIG with the command-line overrides and rejects
    # unknown keys, so no constructor is needed (and `super().__init__` would be rejected
    # by the PR check, which forbids every attribute access starting with an underscore).

    # Shared helpers ------------------------------------------------------------------------

    def ask_for_code(
        self, env: ClarificationEnvironment, prompt: str, entry_point: str = ""
    ) -> str | None:
        """One code request, retried up to `generation_attempts` times, parsed and compiled."""
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(max(self.config["generation_attempts"], 1)):
            response = self.ask_model(env, messages)
            try:
                code = parse_fenced_code(response)
            except ValueError as error:
                messages += [
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": str(error)},
                ]
                continue

            if self.defines_entry_point(code, entry_point):
                return code

            # Either it does not compile, or it compiles without defining the entry point.
            # Both are retryable, and saying which one keeps the retry useful.
            if attempt + 1 < max(self.config["generation_attempts"], 1):
                if entry_point and self.defines_entry_point(code, ""):
                    complaint = (
                        f"That script compiles but does not define `{entry_point}` at module "
                        f"level. Return a complete script that defines `{entry_point}`."
                    )
                else:
                    complaint = (
                        "That code is not valid Python. Return a complete, compiling script."
                    )
                messages += [
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": complaint},
                ]

        return None

    def defines_entry_point(self, code: str, entry_point: str) -> bool:
        """True when `code` compiles and (when asked) defines `entry_point` at module level."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False

        if not entry_point:
            return True

        for node in self.module_level_nodes(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == entry_point
            ):
                return True
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == entry_point:
                        return True

        return False

    def field(self, text: str, name: str) -> str:
        """Reads a `NAME=`backtick-quoted`` or `NAME=value` field out of a model reply.

        The value is read up to the LAST backtick on its line, not the first. A value that
        quotes an identifier (`QUESTION=`Should `f` return a list?`") would otherwise be cut
        off at that inner backtick, which silently turns a good question into a fragment.
        """
        quoted = re.search(rf"{name}\s*=\s*`(.+)`[ \t]*$", text, flags=re.MULTILINE)
        if quoted:
            return quoted.group(1).strip().strip("`").strip()

        # Unterminated or spanning several lines; fall back to the first closing backtick.
        quoted = re.search(rf"{name}\s*=\s*`(.+?)`", text, flags=re.DOTALL)
        if quoted:
            return quoted.group(1).strip()

        plain = re.search(rf"{name}\s*=\s*([^\n`]+)", text)
        return plain.group(1).strip() if plain else ""

    # The environment answers a refused or failed clarification with a fixed sentence rather
    # than an answer. Feeding one back would state it as the author's requirement under
    # "the clarifications take precedence over the problem text", which is worse than having
    # asked nothing, and it also skips the majority vote that would otherwise decide.
    REFUSALS = ("cannot answer", "irrelevant question", "no answer", "unable to answer")

    def usable_answer(self, answer: str) -> bool:
        if not answer:
            return False
        stripped = answer.strip().strip(".").lower()
        if any(stripped.startswith(refusal) for refusal in self.REFUSALS):
            return False
        # The clarification turn is already spent, so a terse but complete reply such as
        # "Descending order." must be kept; only an obvious fragment is worth discarding.
        # A truncated answer ends mid-sentence, which is how the backtick bug presented.
        words = answer.split()
        return len(words) >= 3 or answer.strip().endswith((".", "!", "?"))

    def plain_question(self, question: str) -> str:
        """Strips code formatting from a question before it goes to the author.

        The environment reads the author's reply with ``ANSWERS=`(.+?)` `` (clarify/env.py),
        so one backtick in the reply truncates the answer at that point. The author mirrors
        the formatting of the question, so a backtick-free question keeps the answer intact.
        The prompts ask for plain prose; this makes it true regardless of what came back.
        """
        return " ".join(question.replace("`", "").split())

    # Tracing ------------------------------------------------------------------------------
    # One record per task, so the train split can be analysed per path rather than only in
    # aggregate. `problem` carries no task id, so the record keys on a hash of the prompt and
    # on the prompt itself; both appear in the generated .jsonl and can be joined on.

    def start_trace(self, problem: dict[str, Any]) -> None:
        prompt = problem["prompt"]
        self.trace = {
            "prompt_sha1": hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12],
            "prompt": prompt.strip(),
            "entry_point": problem["entry_point"],
            "path": "no_question",
            "rounds": 0,
            "widened": False,
            "num_candidates": 0,
            "num_usable_candidates": 0,
            "personas": [],
            "num_inputs": 0,
            "sandbox": "unused",
            "group_sizes": [],
            "group_personas": [],
            "num_distinguishing_inputs": 0,
            "inputs": [],
            "failures_per_input": [],
            "inputs_all_failed": 0,
            "partition_inputs": bool(self.config["partition_inputs"]),
            "inputs_rewrapped": 0,
            "inputs_dropped_arity": 0,
            "inputs_regenerated": 0,
            "inputs_after_regeneration": 0,
            "suspect_inputs": 0,
            "discarded_unclean_search": 0,
            "grouping_fallback_search": False,
            "discarded_unclean_vote": 0,
            "grouping_fallback_vote": False,
            "chosen_failures": 0,
            "drafted_questions": [],
            "question": "",
            "answer": "",
            "vote_sizes": [],
            "final_vote_shape": "none",
            "final_failures": 0,
            "final_failures_after": 0,
            "repairs_accepted": 0,
            "repairs_rejected": 0,
            "llm_calls": 0,
            "sandbox_calls": 0,
            "prompt_cost": 0.0,
            "outcome": "ok",
            "error": "",
            "answer_rejected": False,
        }

    def trace_note(self, key: str, value: Any) -> None:
        """Records one field, tolerating a `run` that was entered without `start_trace`."""
        if getattr(self, "trace", None) is not None:
            self.trace[key] = value

    def trace_bump(self, key: str) -> None:
        if getattr(self, "trace", None) is not None:
            self.trace[key] = self.trace.get(key, 0) + 1

    def trace_add(self, key: str, amount: int) -> None:
        if getattr(self, "trace", None) is not None:
            self.trace[key] = self.trace.get(key, 0) + amount

    def ask_model(self, env: ClarificationEnvironment, messages: Any) -> str:
        """`env.llm` with a call counter, so a record says how expensive the task was."""
        self.trace_bump("llm_calls")
        return env.llm(messages)

    def run_sandbox(self, env: ClarificationEnvironment, script: str) -> str:
        """`env.exec_code` with a call counter. Each call is one Docker container."""
        self.trace_bump("sandbox_calls")
        return env.exec_code(script)

    def emit_trace(self, env: ClarificationEnvironment) -> None:
        if getattr(self, "trace", None) is None:
            return
        self.trace["prompt_cost"] = env.prompt_cost
        LOGGER.info(json.dumps(self.trace, default=str))

    # === STEP 1: sample candidate programs =================================================

    def sample_candidates(
        self, env: ClarificationEnvironment, problem: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Returns up to `num_candidates` implementations, each with the persona it was written for.

        The persona is there to make the readings differ for a reason other than sampling
        temperature. It is not the only source of diversity: most behaviour splits cut across
        persona boundaries, so `use_personas` can be turned off without the search collapsing.
        """
        base_prompt = CODE_TEMPLATE.replace("{entry_point}", problem["entry_point"]).replace(
            "{prompt}", problem["prompt"].strip()
        )

        candidates: list[dict[str, Any]] = []
        for index in range(max(self.config["num_candidates"], 1)):
            persona = ""
            prompt = base_prompt
            if self.config["use_personas"] and USER_PERSONAS:
                persona = USER_PERSONAS[index % len(USER_PERSONAS)]
                prompt = f"{persona}\n\n{base_prompt}"

            try:
                code = self.ask_for_code(env, prompt, problem["entry_point"])
            except LimitsExceededException:
                break

            if code and self.defines_entry_point(code, problem["entry_point"]):
                candidates.append({"code": code, "persona": persona})

        return candidates

    # === STEP 2: generate inputs ===========================================================

    def generate_inputs(
        self, env: ClarificationEnvironment, problem: dict[str, Any], extra_hint: str = ""
    ) -> list[str]:
        """Returns argument tuples as literal source strings, to be evaluated in the sandbox.

        One model call asking for varied and edge-case inputs. Nothing here trusts them: the
        candidates' own signatures reconcile the argument counts afterwards (`resolved_inputs`),
        and an input that every candidate rejects is excluded from the behaviour comparison
        rather than allowed to manufacture a difference.
        """
        prompt = (
            INPUT_TEMPLATE.replace("{entry_point}", problem["entry_point"])
            .replace("{prompt}", problem["prompt"].strip())
            .replace("{num_inputs}", str(max(self.config["num_inputs"], 1)))
        )
        if extra_hint:
            prompt = f"{prompt}\n\n{extra_hint}"

        try:
            response = self.ask_model(env, prompt)
        except LimitsExceededException:
            return []

        return self.parse_input_lines(response, max(self.config["num_inputs"], 1))

    def parse_input_lines(self, response: str, limit: int) -> list[str]:
        """Keeps the lines of a model reply that are self-contained Python literals."""
        inputs: list[str] = []
        for raw in response.splitlines():
            line = raw.strip().strip("`").strip()
            if not line or line.startswith("#"):
                continue

            # A bare `(1, 2)` is ambiguous between a two-argument call and a single tuple
            # argument, and the sandbox reads it as two arguments. It is not resolved here,
            # where no candidate is in scope: `resolved_inputs` reconciles every line against
            # the candidates' real signatures and rewrites or drops it there.
            try:
                ast.literal_eval(line)
            except (ValueError, SyntaxError):
                continue

            if line not in inputs:
                inputs.append(line)
            if len(inputs) >= limit:
                break

        return inputs

    # === Input partitioning ================================================================
    # `parse_input_lines` reads every line as a tuple of positional arguments, so a task
    # whose function takes ONE tuple has its inputs read as N arguments and every candidate
    # raises TypeError. Those failures are artefacts of the call, not defects in the code,
    # and charging them against the candidates poisons grouping, question selection and
    # repair alike. Both signals below are mechanical: no model call.

    def module_level_nodes(self, tree: Any) -> list[Any]:
        """Statements that run at module level, last first.

        Descends into `if`/`try`/`with`/loop bodies, which do execute on import, but never
        into a function or class body. A method or inner function named like the entry point
        is not reachable as `namespace[entry_point]`, which is how the sandbox calls it, so
        treating one as the definition yields an arity that includes `self` and rejects every
        correct input. Last first, because a later definition is the one that survives.
        """
        collected, stack = [], list(tree.body)
        while stack:
            node = stack.pop()
            collected.append(node)
            if isinstance(node, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                stack.extend(node.body)
                stack.extend(getattr(node, "orelse", []) or [])
                stack.extend(getattr(node, "finalbody", []) or [])
                for handler in getattr(node, "handlers", []) or []:
                    stack.extend(handler.body)
        return collected

    def entry_point_args(self, code: str, entry_point: str) -> Any:
        """The `ast.arguments` node of a module-level `entry_point`, or None."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None

        for node in self.module_level_nodes(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == entry_point
            ):
                return node.args
        return None

    def entry_point_arity(self, code: str, entry_point: str) -> tuple[int, int | None] | None:
        """(min, max) positional arity of `entry_point`; max is None for *args.

        None means the signature could not be read, so callers must not judge an input by
        it rather than assuming a count.
        """
        args = self.entry_point_args(code, entry_point)
        if args is None:
            return None

        # A keyword-only parameter with no default cannot be supplied positionally, so the
        # function is not callable the way the sandbox calls it. Report the arity as unknown
        # rather than a count that would license rewrapping inputs against a signature that
        # can never accept them.
        if any(default is None for default in args.kw_defaults or []):
            return None

        positional = args.posonlyargs + args.args
        required = len(positional) - len(args.defaults)
        if args.vararg is not None:
            return (required, None)
        return (required, len(positional))

    def entry_point_params(self, code: str, entry_point: str) -> list[str] | None:
        """The positional parameter names of `entry_point`, or None."""
        args = self.entry_point_args(code, entry_point)
        if args is None:
            return None
        if args.vararg is not None:
            return None  # variadic: no fixed list to state in a prompt
        return [argument.arg for argument in args.posonlyargs + args.args]

    def arity_accepts(self, arity: tuple[int, int | None] | None, count: int) -> bool:
        if arity is None:
            return True
        low, high = arity
        return count >= low and (high is None or count <= high)

    def resolve_input_arity(self, line: str, arities: list[Any]) -> str | None:
        """Fixes or rejects one input literal against the candidates' own signatures.

        Where the literal only makes sense as a single argument it is rewritten rather than
        dropped, so the input budget survives instead of shrinking.
        """
        try:
            value = ast.literal_eval(line)
        except (ValueError, SyntaxError):
            return None

        if not isinstance(value, tuple):
            value = (value,)

        known = [arity for arity in arities if arity is not None]
        if not known:
            return line
        if any(self.arity_accepts(arity, len(value)) for arity in known):
            return line
        if any(self.arity_accepts(arity, 1) for arity in known):
            rewrapped = repr((value,))
            # `repr` does not round-trip for non-finite floats (`inf`, `nan`), and the
            # sandbox parses inputs with `literal_eval`, so an unparseable rewrap would be
            # counted as a usable input while silently never running.
            try:
                ast.literal_eval(rewrapped)
            except (ValueError, SyntaxError):
                return None
            return rewrapped
        return None

    def arity_fit(
        self, inputs: list[str], candidates: list[dict[str, Any]], entry_point: str
    ) -> tuple[list[str], int, int]:
        """(reconciled inputs, number dropped, number rewrapped).

        Counts are returned rather than only recorded, because "no signature accepts this
        input" is a signal the caller has to act on, not just report.
        """
        arities = [
            self.entry_point_arity(candidate["code"], entry_point) for candidate in candidates
        ]
        resolved, rewrapped, dropped = [], 0, 0
        for line in inputs:
            fixed = self.resolve_input_arity(line, arities)
            if fixed is None:
                dropped += 1
                continue
            if fixed != line:
                rewrapped += 1
            resolved.append(fixed)
        return resolved, dropped, rewrapped

    def resolved_inputs(
        self, inputs: list[str], candidates: list[dict[str, Any]], entry_point: str
    ) -> list[str]:
        """The input list with argument counts reconciled against the candidates."""
        if not self.config["partition_inputs"]:
            return inputs

        resolved, dropped, rewrapped = self.arity_fit(inputs, candidates, entry_point)
        # Called again after widening, so these accumulate rather than overwrite.
        self.trace_add("inputs_rewrapped", rewrapped)
        self.trace_add("inputs_dropped_arity", dropped)
        return resolved or inputs

    def signature_hint(
        self, candidates: list[dict[str, Any]], entry_point: str, reason: str = ""
    ) -> str:
        """An instruction naming the signature the inputs have to match, or "" if unknown.

        `INPUT_TEMPLATE` never states the arity: it shows one example of a single-argument
        call and leaves the rest to the model, which is why inputs for a two- or
        three-argument function routinely come back with one argument.
        """
        counts: dict[tuple[str, ...], int] = {}
        for candidate in candidates:
            params = self.entry_point_params(candidate["code"], entry_point)
            if params:
                key = tuple(params)
                counts[key] = counts.get(key, 0) + 1

        if not counts:
            return ""

        # The signature the candidates agree on; a lone dissenter should not set the shape.
        params = list(max(counts.items(), key=lambda item: item[1])[0])
        placeholders = ", ".join(f"<{name}>" for name in params)
        example = f"({placeholders},)" if len(params) == 1 else f"({placeholders})"
        return (
            f"IMPORTANT: `{entry_point}` takes exactly {len(params)} positional "
            f"argument{'s' if len(params) != 1 else ''}, in this order: "
            f"{', '.join(params)}.\n"
            f"Every line must be a tuple of exactly {len(params)} value"
            f"{'s' if len(params) != 1 else ''}, shaped like `{example}`."
            + (f"\n{reason}" if reason else "")
        )

    def fit_or_regenerate(
        self,
        env: ClarificationEnvironment,
        problem: dict[str, Any],
        inputs: list[str],
        candidates: list[dict[str, Any]],
    ) -> list[str]:
        """Reconciles inputs with the candidate signatures, asking again if most do not fit.

        Rewrapping only repairs one direction: an input with too MANY arguments can be read
        as a single tuple, but one with too FEW is missing a value that cannot be invented,
        and on this benchmark that is the common direction (a two-argument function handed
        one-argument calls). Keeping the unusable inputs wastes the whole differential stage
        for that task, so the inputs are requested once more with the signature stated.
        """
        if not self.config["partition_inputs"]:
            return inputs

        resolved, dropped, rewrapped = self.arity_fit(inputs, candidates, problem["entry_point"])
        self.trace_add("inputs_rewrapped", rewrapped)
        self.trace_add("inputs_dropped_arity", dropped)

        # Tolerate a few unusable inputs; only a wholesale mismatch is worth a model call.
        if resolved and dropped <= len(inputs) // 2:
            return resolved

        best, best_dropped = resolved, dropped
        for _ in range(max(int(self.config["input_retries"]), 0)):
            # State the real reason: this path is also reached when no input parsed at all,
            # and claiming a wrong argument count would then be a false explanation.
            reason = (
                "A previous attempt returned calls with the wrong number of arguments, so "
                "none of them could be run."
                if best_dropped
                else "A previous attempt produced no usable calls at all."
            )
            hint = self.signature_hint(candidates, problem["entry_point"], reason)
            if not hint:
                break

            self.trace_bump("inputs_regenerated")
            retried, retry_dropped, retry_rewrapped = self.arity_fit(
                self.generate_inputs(env, problem, extra_hint=hint),
                candidates,
                problem["entry_point"],
            )
            self.trace_add("inputs_rewrapped", retry_rewrapped)
            self.trace_add("inputs_dropped_arity", retry_dropped)
            self.trace_note("inputs_after_regeneration", len(retried))

            # Keep a retry only when it actually fits better, and stop once it fits.
            if len(retried) > len(best):
                best, best_dropped = retried, retry_dropped
            if best and best_dropped <= len(best) // 2:
                break

        return best or inputs

    def suspect_inputs(self, outputs: list[list[dict[str, Any]]]) -> set[int]:
        """Input indices that say more about the input than about the candidates.

        A candidate that failed on every input carries no information about any particular
        one, so it gets no vote; without that, every input looks suspect on exactly the
        tasks where all candidates are broken. Among the remaining voters, an input that
        every one of them rejects is suspect: independently sampled programs rarely share a
        defect, but they do share a misreading of a malformed call.
        """
        if not self.config["partition_inputs"] or not outputs:
            return set()

        voters = [row for row in outputs if any(obs.get("kind") == "value" for obs in row)]
        if not voters:
            return set()

        width = min(len(row) for row in voters)
        return {
            index
            for index in range(width)
            if all(row[index].get("kind") in ("error", "timeout") for row in voters)
        }

    def valid_indices(self, outputs: list[list[dict[str, Any]]]) -> list[int]:
        """Input indices worth judging a candidate by.

        An input is dropped when it is suspect, and also when ANY candidate's cell for it
        was never run. The sandbox emits `skipped` both for a literal it could not parse and
        for every cell after its deadline, and a partially observed column is not evidence:
        comparing it splits two behaviourally identical programs on where the clock happened
        to stop, and the single question then gets drafted from that artefact.
        """
        if not outputs:
            return []

        width = min(len(row) for row in outputs)
        suspect = self.suspect_inputs(outputs)
        return [
            index
            for index in range(width)
            if index not in suspect
            and not any(row[index].get("kind") == "skipped" for row in outputs)
        ]

    def failures_on_valid(self, row: list[dict[str, Any]], valid: list[int]) -> int:
        return sum(1 for index in valid if row[index].get("kind") in ("error", "timeout"))

    # === STEP 3: run every candidate on every input ========================================

    def run_candidates(
        self,
        env: ClarificationEnvironment,
        problem: dict[str, Any],
        candidates: list[dict[str, Any]],
        inputs: list[str],
    ) -> list[list[dict[str, Any]]] | None:
        """Returns `outputs[c][i]` for candidate `c` on input `i`, or None when unavailable.

        One sandbox script for the whole table, since each `env.exec_code` call starts its own
        Docker container. Returns None when Docker is missing or the output is unusable, which
        makes `run` take the no-execution path.
        """
        if not candidates or not inputs:
            return None

        payload = {
            "entry_point": problem["entry_point"],
            "candidates": [candidate["code"] for candidate in candidates],
            "inputs": inputs,
            # setitimer(.., 0) disarms rather than tightens, so never pass through a zero.
            "timeout": max(float(self.config["exec_timeout"]), 0.05),
            "float_tol": float(self.config["float_tol"]),
        }
        encoded = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
        script = SANDBOX_SCRIPT.replace("__PAYLOAD__", encoded)

        try:
            output = self.run_sandbox(env, script)
        except Exception:
            # No Docker, or the sandbox could not be started.
            return None

        return self.parse_sandbox_output(output, len(candidates), len(inputs))

    def parse_sandbox_output(
        self, output: str, num_candidates: int, num_inputs: int
    ) -> list[list[dict[str, Any]]] | None:
        """Extracts the JSON table from the sandbox stdout, which may carry other noise."""
        if not output or "<<<CBD-BEGIN>>>" not in output or "<<<CBD-END>>>" not in output:
            return None

        body = output.split("<<<CBD-BEGIN>>>", 1)[1].split("<<<CBD-END>>>", 1)[0].strip()
        try:
            table = json.loads(body)
        except (ValueError, TypeError):
            return None

        if not isinstance(table, list) or len(table) != num_candidates:
            return None
        if any(not isinstance(row, list) or len(row) != num_inputs for row in table):
            return None

        return table

    # === STEP 4: group the candidates by behaviour =========================================

    def group_by_behaviour(
        self, outputs: list[list[dict[str, Any]]], stage: str = "search"
    ) -> list[list[int]]:
        """Returns groups of candidate indices whose behaviour is identical on every input.

        Candidates that never ran (no entry point, syntax error, or a failure on every single
        input) are dropped: those are bugs, not open requirements.

        A candidate that fails on any input worth judging it by is discarded: its behaviour
        is a bug, and a group built on it produces a question about a defect rather than
        about an open requirement. Suspect inputs are excluded from that test and from the
        behaviour signature, so candidates that differ only on a malformed call are no
        longer split into separate groups.

        The strict rule can empty the pool, and `run` must always return something, so it
        falls back to the old rule (ran at least once) and then to nothing.

        Discarding a crashing candidate does lose real information: "raise ValueError on an
        empty input" versus "return 0" is a genuine open requirement. It is dropped anyway
        because the final program cannot afford to raise on a test input - the harness
        compares return values, so any exception fails the task - and a difference that cannot
        be acted on is not worth spending the single question on.
        """
        if not outputs:
            return []

        valid = self.valid_indices(outputs)

        ran = [
            index
            for index, row in enumerate(outputs)
            if any(observation.get("kind") == "value" for observation in row)
        ]
        clean = [index for index in ran if not self.failures_on_valid(outputs[index], valid)]

        usable = clean or ran
        self.trace_note(f"discarded_unclean_{stage}", len(ran) - len(clean))
        self.trace_note(f"grouping_fallback_{stage}", not clean and bool(ran))

        groups: dict[str, list[int]] = {}
        for index in usable:
            signature = json.dumps(
                [self.signature(outputs[index][position]) for position in valid], sort_keys=True
            )
            groups.setdefault(signature, []).append(index)

        return sorted(groups.values(), key=len, reverse=True)

    def signature(self, observation: dict[str, Any]) -> Any:
        """The comparable part of one observation: what a caller could notice."""
        kind = observation.get("kind")
        if kind == "value":
            core: Any = ["value", observation.get("value")]
        elif kind == "error":
            core = ["error", observation.get("error")]
        else:
            core = [kind, None]

        return [core, observation.get("mutated", False), observation.get("printed", "")]

    # === STEP 5: turn the clearest difference into a question ==============================

    def pick_distinguishing_inputs(
        self, outputs: list[list[dict[str, Any]]], groups: list[list[int]]
    ) -> list[int]:
        """Returns the indices of the best distinguishing inputs, best first (may be empty)."""
        if len(groups) < 2:
            return []

        # A suspect input is excluded outright. The score below only *prefers* inputs where
        # every group returned a value, so without this filter a malformed call can still
        # win and spend the single question on an artefact.
        scored: list[tuple[float, int]] = []
        for input_index in self.valid_indices(outputs):
            classes: dict[str, int] = {}
            all_returned = True
            for group in groups:
                observation = outputs[group[0]][input_index]
                if observation.get("kind") != "value":
                    all_returned = False
                key = json.dumps(self.signature(observation), sort_keys=True)
                classes[key] = classes.get(key, 0) + len(group)

            if len(classes) < 2:
                continue

            # Prefer an even split (2-vs-2 over 3-vs-1), every group returning a value rather
            # than raising, and an earlier input (the model lists ordinary cases first).
            sizes = sorted(classes.values(), reverse=True)
            evenness = sizes[0] - sizes[-1]
            score = evenness - (2.0 if all_returned else 0.0) + input_index * 0.01
            scored.append((score, input_index))

        scored.sort()
        return [index for _, index in scored[: max(self.config["max_differences"], 1)]]

    def questions_from_differences(
        self,
        env: ClarificationEnvironment,
        problem: dict[str, Any],
        candidates: list[dict[str, Any]],
        outputs: list[list[dict[str, Any]]],
        groups: list[list[int]],
        indices: list[int],
    ) -> list[str]:
        """Drafts one question per difference, skipping differences the task already decides."""
        questions: list[str] = []

        for input_index in indices:
            # One entry per distinct behaviour AT THIS INPUT, not per group: groups that
            # agree here and differ elsewhere must not be listed as rival behaviours.
            distinct: dict[str, int] = {}
            for group in groups:
                key = json.dumps(self.signature(outputs[group[0]][input_index]), sort_keys=True)
                distinct.setdefault(key, group[0])

            if len(distinct) < 2:
                continue  # nothing actually disagrees on this input

            behaviours, programs = [], []
            for number, candidate_index in enumerate(distinct.values(), start=1):
                observation = outputs[candidate_index][input_index]
                behaviours.append(f"- Behaviour {number}: {self.describe(observation)}")
                programs.append(
                    f"#### Behaviour {number}\n```python\n{candidates[candidate_index]['code'].strip()}\n```"
                )

            prompt = (
                DIFFERENCE_TEMPLATE.replace("{prompt}", problem["prompt"].strip())
                .replace("{entry_point}", problem["entry_point"])
                .replace("{input_repr}", self.call_repr(outputs, input_index))
                .replace("{behaviours}", "\n".join(behaviours))
                .replace("{programs}", "\n\n".join(programs))
            )

            try:
                reply = self.ask_model(env, prompt)
            except LimitsExceededException:
                break

            # DECIDED=yes means the task text settles it, so one group is simply wrong and
            # there is nothing to ask about this difference.
            if self.field(reply, "DECIDED").lower().startswith("yes"):
                continue

            question = self.field(reply, "QUESTION")
            if question and question not in questions:
                questions.append(question)

        return questions

    def call_repr(self, outputs: list[list[dict[str, Any]]], input_index: int) -> str:
        """The input as it appears in a call, for the prompt. Filled in by `run`.

        Normalised to the tuple form the sandbox actually calls with: `parse_input_lines`
        accepts a bare literal such as `[1, 2, 3]`, and showing it raw renders as
        `f[1, 2, 3]` rather than a call.
        """
        if not self.current_inputs or input_index >= len(self.current_inputs):
            return "(...)"

        line = self.current_inputs[input_index]
        try:
            value = ast.literal_eval(line)
        except (ValueError, SyntaxError):
            return line
        return line if isinstance(value, tuple) else repr((value,))

    def readable(self, form: Any) -> str:
        """Turns a canonical form from the sandbox back into a readable literal.

        `canon` keeps the type beside every value so that 1, 1.0 and "1.0" do not collapse
        when behaviour is compared. That encoding must not reach a prompt: the model needs
        to see `[1, 2]`, not `[["int", 1], ["int", 2]]`.
        """
        if not isinstance(form, list) or len(form) != 2:
            return json.dumps(form)

        name, value = form
        if name == "float" and isinstance(value, str):
            return value  # the "nan" / "inf" / "-inf" sentinels canon stores as text
        if name in ("int", "bool", "none", "str", "float"):
            return repr(value)  # Python syntax: True, None, 'ab' - not true, null, "ab"
        if name in ("list", "tuple"):
            inner = ", ".join(self.readable(item) for item in value or [])
            if name == "tuple":
                return f"({inner}{',' if len(value or []) == 1 else ''})"
            return f"[{inner}]"
        if name == "set":
            return "{" + ", ".join(self.readable(json.loads(item)) for item in value or []) + "}"
        if name == "dict":
            pairs = ", ".join(
                f"{self.readable(json.loads(key))}: {self.readable(json.loads(item))}"
                for key, item in value or []
            )
            return "{" + pairs + "}"
        if name == "other":
            return str(value)
        return json.dumps(value)

    def describe(self, observation: dict[str, Any]) -> str:
        """One readable line for an observation, for use in a prompt."""
        kind = observation.get("kind")
        if kind == "value":
            form = observation.get("value") or ["?", None]
            type_name = form[0] if isinstance(form, list) and form else "?"
            detail = f"returns {self.readable(form)} of type {type_name}"
        elif kind == "error":
            detail = f"raises {observation.get('error')}"
        elif kind == "timeout":
            detail = "does not terminate"
        else:
            detail = f"did not run ({kind})"

        if observation.get("mutated"):
            detail += ", and modifies its arguments in place"
        if observation.get("printed"):
            detail += f", and prints {observation['printed'].strip()!r}"

        return detail

    def best_question(
        self, env: ClarificationEnvironment, problem: dict[str, Any], questions: list[str]
    ) -> str:
        """Returns the drafted question that best fits the judging criteria."""
        if not questions:
            return ""
        if len(questions) == 1:
            return questions[0]

        listing = "\n".join(f"{number}. {text}" for number, text in enumerate(questions, start=1))
        prompt = RANK_TEMPLATE.replace("{prompt}", problem["prompt"].strip()).replace(
            "{questions}", listing
        )

        try:
            reply = self.ask_model(env, prompt)
        except LimitsExceededException:
            return questions[0]

        match = re.search(r"\d+", reply)
        if match:
            choice = int(match.group(0)) - 1
            if 0 <= choice < len(questions):
                return questions[choice]

        return questions[0]

    # === STEP 6: widen the search, or ask without a difference =============================

    def widen_search(
        self,
        env: ClarificationEnvironment,
        problem: dict[str, Any],
        candidates: list[dict[str, Any]],
        inputs: list[str],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Returns extended (candidates, inputs) after the candidates all agreed.

        Agreement has two causes that look identical from the outside: the inputs were too
        weak to separate the candidates, or the candidates really do share one reading. Since
        nothing here can tell them apart, both are addressed at once - more edge-case inputs,
        and one candidate asked to read the task differently.
        """
        hint = (
            "The following inputs did not separate several implementations of this task, so "
            "avoid them and go further towards the boundaries:\n"
            + "\n".join(f"- {text}" for text in inputs[:10])
        )
        widened_inputs = list(inputs)
        for text in self.generate_inputs(env, problem, extra_hint=hint):
            if text not in widened_inputs:
                widened_inputs.append(text)

        widened_candidates = list(candidates)
        if candidates:
            prompt = (
                DIVERGENT_TEMPLATE.replace("{entry_point}", problem["entry_point"])
                .replace("{prompt}", problem["prompt"].strip())
                .replace("{candidate}", candidates[0]["code"].strip())
            )
            try:
                code = self.ask_for_code(env, prompt, problem["entry_point"])
            except LimitsExceededException:
                code = None

            if code and self.defines_entry_point(code, problem["entry_point"]):
                widened_candidates.append({"code": code, "persona": "divergent reading"})

        return widened_candidates, widened_inputs

    def fallback_question(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        """Returns the most critical question when no input separates the candidates.

        One question discounts that task's score by about 4%, so asking here is only worth it
        where the answer is more likely than that to change the outcome. `ask_when_agree`
        controls whether this path asks at all; it is on because a prompt that several
        independent readings agree on can still be underspecified.
        """
        prompt = FALLBACK_TEMPLATE.replace("{prompt}", problem["prompt"].strip())

        try:
            reply = self.ask_model(env, prompt).strip()
        except LimitsExceededException:
            return ""

        if not reply or "NO_QUESTION" in reply:
            return ""

        return reply.splitlines()[0].strip()

    # === STEP 7: regenerate with the answer and vote =======================================

    # === STEP 7b: repair the final program =================================================

    def failing_calls(self, row: list[dict[str, Any]], valid: list[int] | None = None) -> list[int]:
        """Input indices where one program raised or hung, restricted to `valid`.

        `INPUT_TEMPLATE` asks only for inputs the task considers valid, but that is an
        instruction to a model, not a guarantee, so the caller passes the inputs actually
        worth judging by. Without that restriction a malformed call is reported to the
        repair prompt as a defect. Observations that were skipped never ran and are not
        failures.
        """
        positions = range(len(row)) if valid is None else valid
        return [index for index in positions if row[index].get("kind") in ("error", "timeout")]

    def repair_program(
        self,
        env: ClarificationEnvironment,
        problem: dict[str, Any],
        code: str,
        row: list[dict[str, Any]],
        failures: list[int],
        clarifications: list[tuple[str, str]],
    ) -> str | None:
        """One repair attempt for a program that failed on valid inputs.

        There is no oracle here: the environment never reveals whether a program is correct,
        so the only defects that can be acted on are the ones the sandbox shows directly.
        That is deliberately narrow - a program that merely disagrees with its siblings may
        be reading the task differently, which is question material, not a bug.
        """
        rendered = "\n".join(
            f"- `{problem['entry_point']}{self.call_repr(outputs=[], input_index=index)}` "
            f"{self.describe(row[index])}"
            for index in failures
            if index < len(self.current_inputs)
        )
        if not rendered:
            return None

        spec = ""
        if clarifications:
            answers = "\n".join(
                f"- Q: {question}\n  A: {answer}" for question, answer in clarifications
            )
            spec = f"\n### Clarifications from the author\n{answers}\n"

        prompt = (
            REPAIR_TEMPLATE.replace("{prompt}", problem["prompt"].strip())
            .replace("{clarifications}", spec)
            .replace("{code}", code.strip())
            .replace("{failures}", rendered)
        )

        try:
            return self.ask_for_code(env, prompt, problem["entry_point"])
        except LimitsExceededException:
            return None

    def repaired_or_original(
        self,
        env: ClarificationEnvironment,
        problem: dict[str, Any],
        code: str,
        row: list[dict[str, Any]],
        inputs: list[str],
        clarifications: list[tuple[str, str]],
        valid: list[int] | None = None,
    ) -> str:
        """Returns a repaired program, but only when the sandbox shows it is strictly better.

        A repair is accepted only if it fails on fewer inputs and introduces no new failure,
        which is checkable without an oracle. Without that guard a repair can trade one
        defect for another and quietly lose ground.
        """
        failures = self.failing_calls(row, valid)
        self.trace_note("final_failures", len(failures))
        self.trace_note("final_failures_after", len(failures))
        if not failures:
            return code

        current_code, current_failures = code, failures
        for _ in range(max(self.config["repair_attempts"], 0)):
            candidate = self.repair_program(
                env, problem, current_code, row, current_failures, clarifications
            )
            if not candidate or not self.defines_entry_point(candidate, problem["entry_point"]):
                break

            outputs = self.run_candidates(env, problem, [{"code": candidate}], inputs)
            if not outputs:
                break

            new_failures = self.failing_calls(outputs[0], valid)
            # Strictly fewer failures, and nothing that used to work may break.
            broke = [i for i in new_failures if i not in current_failures]
            if len(new_failures) < len(current_failures) and not broke:
                current_code, current_failures = candidate, new_failures
                self.trace_bump("repairs_accepted")
                if not new_failures:
                    break
            else:
                self.trace_bump("repairs_rejected")
                break

        self.trace_note("final_failures_after", len(current_failures))
        return current_code

    def final_program(
        self,
        env: ClarificationEnvironment,
        problem: dict[str, Any],
        clarifications: list[tuple[str, str]],
        inputs: list[str],
        previous: list[dict[str, Any]],
    ) -> str:
        """Returns the implementation, chosen by majority vote over fresh candidates.

        Repeats steps 1-4 with the answers stated as part of the specification. No second
        question: the budget is spent, so remaining disagreement is settled by the vote.
        """
        rendered = "\n".join(
            f"- Q: {question}\n  A: {answer}" for question, answer in clarifications
        )
        prompt = (
            CODE_WITH_SPEC_TEMPLATE.replace("{entry_point}", problem["entry_point"])
            .replace("{prompt}", problem["prompt"].strip())
            .replace("{clarifications}", rendered)
        )

        # Sample without personas here: the specification is now explicit, so the point is
        # agreement, not diversity.
        candidates: list[dict[str, Any]] = []
        for _ in range(max(self.config["num_candidates"], 1)):
            try:
                code = self.ask_for_code(env, prompt, problem["entry_point"])
            except LimitsExceededException:
                break
            if code and self.defines_entry_point(code, problem["entry_point"]):
                candidates.append({"code": code, "persona": ""})

        if not candidates:
            return previous[0]["code"] if previous else ""

        outputs = self.run_candidates(env, problem, candidates, inputs)
        if outputs is None:
            # No sandbox: fall back to the first clarified candidate, unrepaired, because
            # there is no way to see whether it fails or whether a repair helped.
            return candidates[0]["code"]

        groups = self.group_by_behaviour(outputs, stage="vote")
        self.trace_note("vote_sizes", [len(group) for group in groups])
        self.trace_note(
            "final_vote_shape",
            "split" if len(groups) > 1 else ("unanimous" if groups else "all_failed"),
        )

        # Rank rather than discard, so nothing is lost and `run` always returns something.
        # Note what this does and does not do: `group_by_behaviour` has already dropped every
        # candidate that fails on a valid input WHENEVER at least one clean candidate exists,
        # so in the normal case every surviving group scores 0 here and the sort collapses to
        # group size - the old majority vote. The ranking only bites on the fallback path,
        # where no candidate was clean, and that is also the only path on which the repair
        # below can fire. Keep that in mind before tuning `repair_attempts`.
        valid = self.valid_indices(outputs)
        if groups:
            groups = sorted(
                groups,
                key=lambda group: (self.failures_on_valid(outputs[group[0]], valid), -len(group)),
            )
            chosen = groups[0][0]
        else:
            # Every program is broken; repair below is the only remaining lever.
            chosen = 0

        self.trace_note("chosen_failures", self.failures_on_valid(outputs[chosen], valid))

        return self.repaired_or_original(
            env,
            problem,
            candidates[chosen]["code"],
            outputs[chosen],
            inputs,
            clarifications,
            valid,
        )

    # === Entry point =======================================================================

    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        self.start_trace(problem)
        self.fallback_code = ""
        try:
            return self.solve(env, problem)
        except (LimitsExceededException, TooManyQuestionException):
            # Budget exhausted: whatever was already sampled is the best answer available.
            self.trace_note("outcome", "budget_exhausted")
            return self.fallback_code
        except Exception as error:
            # `generate_responses` turns an exception into a zero-scoring "[EXCEPTION]"
            # record, so returning a program already in hand is strictly better than
            # letting an unexpected error through.
            # The PR check rejects any access to a name starting with "_", which includes
            # `type(error).__name__`, so the class is rendered without touching it.
            self.trace_note("outcome", "error")
            self.trace_note("error", f"{type(error)}: {error}"[:200])
            return self.fallback_code
        finally:
            self.emit_trace(env)

    def solve(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        # Holds the literal source of the current inputs so that `call_repr` can show the
        # input in a prompt without threading it through every signature.
        self.current_inputs: list[str] = []

        # --- Step 1 + 2: candidates and inputs --------------------------------------------
        candidates = self.sample_candidates(env, problem)
        self.trace_note("num_candidates", len(candidates))
        self.trace_note("personas", [candidate["persona"] for candidate in candidates])
        if not candidates:
            # Nothing usable came back; a direct attempt is still better than an empty answer.
            self.trace_note("outcome", "no_candidates")
            return self.direct_solution(env, problem)

        fallback_code = candidates[0]["code"]
        self.fallback_code = fallback_code
        inputs = self.generate_inputs(env, problem)
        # Reconcile argument counts against the candidates' own signatures before anything
        # runs, so a single-tuple task does not hand every candidate a TypeError.
        inputs = self.fit_or_regenerate(env, problem, inputs, candidates)
        self.current_inputs = inputs
        self.trace_note("num_inputs", len(inputs))

        # --- Steps 3-6: find a difference, widening the search at most once ----------------
        question = ""
        groups: list[list[int]] = []
        outputs = None

        for round_index in range(max(self.config["search_rounds"], 1)):
            self.trace_note("rounds", round_index + 1)
            outputs = self.run_candidates(env, problem, candidates, inputs)
            if outputs is None:
                self.trace_note("sandbox", "unavailable")
                break  # no sandbox available; step 6b below still asks a question

            self.trace_note("sandbox", "ok")
            groups = self.group_by_behaviour(outputs)
            indices = self.pick_distinguishing_inputs(outputs, groups)

            # The group shape is the measurement that says whether differential testing is
            # doing any work on this task: one group means it found nothing to ask about.
            self.trace_note("num_usable_candidates", sum(len(group) for group in groups))
            self.trace_note("group_sizes", [len(group) for group in groups])
            self.trace_note(
                "group_personas",
                [[candidates[index]["persona"] for index in group] for group in groups],
            )
            self.trace_note("num_distinguishing_inputs", len(indices))

            # How many candidates failed on each input. An input that EVERY candidate
            # rejects says more about the input than about the candidates: independently
            # sampled programs rarely share a defect, but they do share a misreading of a
            # malformed call. `suspect_inputs` excludes those from the comparison; the counts
            # are recorded here so the size of the effect stays visible in the trace.
            width = min(len(row) for row in outputs) if outputs else 0
            per_input = [
                sum(1 for row in outputs if row[index].get("kind") in ("error", "timeout"))
                for index in range(width)
            ]
            self.trace_note("inputs", list(inputs[:width]))
            self.trace_note("failures_per_input", per_input)
            self.trace_note("inputs_all_failed", sum(1 for n in per_input if n == len(outputs)))
            self.trace_note("suspect_inputs", len(self.suspect_inputs(outputs)))

            if indices:
                drafted = self.questions_from_differences(
                    env, problem, candidates, outputs, groups, indices
                )
                self.trace_note("drafted_questions", list(drafted))
                question = self.best_question(env, problem, drafted)
                if question:
                    self.trace_note("path", "widened" if round_index else "difference")
                    break

            # Step 6: all candidates agreed (or every difference was already decided by the
            # task text). Widen once and try again.
            if round_index + 1 < max(self.config["search_rounds"], 1):
                candidates, inputs = self.widen_search(env, problem, candidates, inputs)
                inputs = self.resolved_inputs(inputs, candidates, problem["entry_point"])
                self.current_inputs = inputs
                self.trace_note("widened", True)
                self.trace_note("num_candidates", len(candidates))
                self.trace_note("num_inputs", len(inputs))

        # Step 6b: no difference found. Ask the model directly, if that is still worth it.
        if not question and self.config["ask_when_agree"]:
            question = self.fallback_question(env, problem)
            if question:
                self.trace_note("path", "fallback")

        if not question:
            self.trace_note("path", "no_question")
            return self.majority_or_fallback(outputs, groups, candidates, fallback_code)

        # --- Step 7: ask, then regenerate from the answer ---------------------------------
        clarifications: list[tuple[str, str]] = []
        asked = 0
        while question and asked < max(self.config["max_questions"], 1) and env.can_ask():
            question = self.plain_question(question)
            try:
                answer = env.ask_human(question)
            except TooManyQuestionException:
                break

            asked += 1
            self.trace_note("question", question)
            self.trace_note("answer", answer)
            if self.usable_answer(answer):
                clarifications.append((question, answer))
            else:
                self.trace_note("answer_rejected", True)

            # One question per task: the evaluation setting fixes the clarification budget at
            # a single turn, so there is no second difference to fold in.
            question = ""

        if not clarifications:
            self.trace_note("outcome", "asked_but_no_answer")
            return self.majority_or_fallback(outputs, groups, candidates, fallback_code)

        try:
            final = self.final_program(env, problem, clarifications, inputs, candidates)
        except LimitsExceededException:
            final = ""

        if not final:
            self.trace_note("outcome", "final_generation_failed")
        return final or fallback_code

    def majority_or_fallback(
        self,
        outputs: list[list[dict[str, Any]]] | None,
        groups: list[list[int]],
        candidates: list[dict[str, Any]],
        fallback_code: str,
    ) -> str:
        """Returns a program from the largest behaviour group, or the first candidate."""
        if outputs is not None and groups:
            return candidates[groups[0][0]]["code"]
        return fallback_code

    def direct_solution(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        """Last resort: one unconditioned attempt, so a task never returns nothing."""
        prompt = CODE_TEMPLATE.replace("{entry_point}", problem["entry_point"]).replace(
            "{prompt}", problem["prompt"].strip()
        )
        try:
            response = self.ask_model(env, prompt)
        except LimitsExceededException:
            return ""

        try:
            return parse_fenced_code(response)
        except ValueError:
            # Prose is not a solution: `evaluate_responses` wraps the text in a python
            # fence and compiles it, so returning it guarantees a syntax error and a
            # misleading record. An empty result scores the same and says what happened.
            self.trace_note("outcome", "direct_solution_unparseable")
            return ""
