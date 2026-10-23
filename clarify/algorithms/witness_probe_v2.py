# SPDX-FileCopyrightText: 2026 zhangchuangnankai <zhangchuang@lenovo.com>
#
# SPDX-License-Identifier: MIT

"""WitnessProbe v4.1 — fork analysis, witness execution, answer-priority regeneration.

Mechanism:
1. Generate a first-guess draft of the task.
2. Ask the model to locate a single critical fork: one missing or vague fact
   whose two plausible readings change what the function returns for ordinary
   valid inputs. If none exists, the draft stands and no question is asked.
3. Build a witness draft committed to the other reading and run both drafts
   in the sandbox on a few probe inputs. Ask the user the fork question
   (assumption-framed) unless the witness evidence shows the readings agree.
4. When an answer is received, regenerate the implementation with the answer
   taking precedence over the task text, and pick the best candidate by
   executing self-written asserts derived from the task description.

Team: zhangchuangnankai
Team Members: Zhang Chuang (Lenovo)
Main Contact: zhangchuang@lenovo.com
"""

from __future__ import annotations

import json
import re
from typing import Any

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import (
    ClarificationEnvironment,
    LimitsExceededException,
    TooManyQuestionException,
)

CODE_PROMPT = """\
Generate Python code directly (Markdown) to solve the coding problem implementing `{entry_point}`.

{prompt}

Name the function exactly `{entry_point}`. Enclose your solution in ```python and ```.
"""

RETRY_CODE_PROMPT = """\
Answer the previous request now. Name the function exactly `{entry_point}` and return only the complete implementation enclosed in ```python and ```.
"""

SIDE_PROMPT = """\
Implement `{entry_point}` as a self-contained Python module.

{prompt}

The original description is ambiguous. Your job is to implement the reading of the task that DIFFERS from the draft shown below at the single point of ambiguity described after it. Keep the same function name and signature; only the ambiguous behavior changes.

Ambiguous point: {fork_note}

Reference draft (the other reading):
```python
{draft}
```

Enclose your solution in ```python and ```.
"""

ANALYSIS_PROMPT = """\
You are reviewing a coding task and one candidate implementation before finalizing.

<problem>
{prompt}
</problem>

Candidate implementation:
```python
{candidate}
```

Look for a SINGLE critical fork: one place where the description is missing information, is vague, or contradicts itself (text vs examples, text vs signature) such that two plausible readings change what `{entry_point}` returns or accepts for ordinary, valid inputs. Common forks: which quantity to compute or return, which elements to keep or drop, the meaning of a parameter, tie-break or boundary conventions, output format, the center or reference point of a formula. Ignore cosmetic details, style, and behavior on invalid inputs.

If you find exactly one critical fork, reply with EXACTLY ONE line starting with FORK, followed by a JSON object with these keys:
- why: one short sentence on why the other reading is plausible (this forces you to commit to a real alternative).
- question: one short question about exactly one fact, answerable with a single value or choice, phrased as an assumption the user can confirm or correct, e.g. "I will assume X. Is that right, or should it be Y?" where X is what the candidate does and Y is the other reading. Plain text, no backticks, no markdown, at most two sentences.
- probes: a string containing a python code block that defines INPUTS, a list of 3 to 5 small concrete argument tuples of valid inputs (typical and boundary cases, no invalid types) on which the two readings are likely to differ.

If no single critical fork exists, reply with exactly: NO_FORK
"""

ANALYSIS_RETRY_PROMPT = """\
You are reviewing a coding task. Describe exactly what the implemented function must DO.

<problem>
{prompt}
</problem>

The description may be vague or incomplete in one place that changes the output for ordinary valid inputs (for example: which value is returned, which items are kept, the meaning of a parameter, a boundary or tie-break convention, an output format). Find the single most important such place and reply with EXACTLY ONE line starting with FORK, followed by a JSON object with keys:
- why: one short sentence on why the description admits another reading.
- question: one short question about exactly one fact, answerable with a single value or choice, phrased as an assumption the user can confirm or correct, e.g. "I will assume X. Is that right, or should it be Y?". Plain text, no backticks, no markdown, at most two sentences.
- probes: a string containing a python code block that defines INPUTS, a list of 3 to 5 small concrete argument tuples of valid inputs on which the readings are likely to differ.

Only reply NO_FORK if the task is genuinely fully specified.
"""

FINAL_PROMPT = """\
Implement `{entry_point}` as a self-contained Python module for the task below.

<problem>
{prompt}
</problem>
{clarification}
Name the function exactly `{entry_point}`. Where the user's answer conflicts with the task text or its examples, follow the answer; everything the answer does not mention follows the task text. Enclose your solution in ```python and ```.
"""

TESTS_PROMPT = """\
Task for `{entry_point}`:

<problem>
{prompt}
</problem>
{clarification}
Write up to {num_tests} assert statements that check `{entry_point}` on concrete inputs whose expected output follows unambiguously from the task text, its examples, or the clarification above. Do not invent behavior that is not stated; skip a case rather than guess.
Return only a Python code block with one `assert {entry_point}(...) == ...` statement per line.
"""

INTERFACE_QUESTION = "Which arguments does {entry_point} take, and in what order?"

HARNESS = r"""
import copy
import json
import signal

BUILTIN_TYPES = {}
try:
    import builtins as bt
    for bn in dir(bt):
        bo = vars(bt)[bn]
        try:
            bcls = type(bo())
        except BaseException:
            continue
        if issubclass(bcls, BaseException):
            BUILTIN_TYPES[repr(bcls)] = bn
except BaseException:
    pass

def on_alarm(signum, frame):
    raise TimeoutError("step timeout")

signal.signal(signal.SIGALRM, on_alarm)

CANDIDATES = json.loads(CANDIDATES_JSON)
ENTRY = ENTRY_JSON
MODE = MODE_JSON
PAYLOAD = json.loads(PAYLOAD_JSON)

def type_name(exc):
    try:
        return BUILTIN_TYPES.get(repr(type(exc)), "exception")
    except BaseException:
        return "exception"

def load_code(code):
    ns = {}
    try:
        import builtins
        ns["__builtins__"] = builtins
    except BaseException:
        pass
    ns["__name__"] = "candidate_module"
    signal.alarm(5)
    try:
        exec(code, ns)
    finally:
        signal.alarm(0)
    return ns

rows = []
inputs = []
if MODE == "probe":
    ins = {}
    try:
        signal.alarm(5)
        exec(PAYLOAD, ins)
        signal.alarm(0)
        inputs = list(ins.get("INPUTS", []))[:12]
    except BaseException:
        signal.alarm(0)
        inputs = []

for code in CANDIDATES:
    row = []
    try:
        ns = load_code(code)
        fn = ns[ENTRY]
    except BaseException:
        rows.append(None)
        continue
    if MODE == "probe":
        for item in inputs:
            args = item if isinstance(item, tuple) else (item,)
            try:
                signal.alarm(2)
                out = fn(*copy.deepcopy(args))
                signal.alarm(0)
                row.append(repr(out)[:100])
            except BaseException as exc:
                signal.alarm(0)
                row.append("raises " + type_name(exc))
    else:
        for stmt in PAYLOAD:
            try:
                signal.alarm(2)
                exec(stmt, ns)
                signal.alarm(0)
                row.append(1)
            except BaseException:
                signal.alarm(0)
                row.append(0)
    rows.append(row)

print("@@HARNESS@@" + json.dumps({"inputs": [repr(i)[:80] for i in inputs], "rows": rows}))
"""


def extract_code(response: str) -> str | None:
    if not isinstance(response, str) or "```python" not in response:
        return None
    _, rest = response.split("```python", 1)
    if "```" not in rest:
        return rest.strip() or None
    return rest.split("```", 1)[0].strip() or None


def call_llm(env: ClarificationEnvironment, content: str) -> str:
    last = ""
    for _ in range(2):
        try:
            return env.llm(content)
        except LimitsExceededException:
            raise
        except Exception as exc:
            last = "[ERROR] " + str(exc)[:120]
    return last


def generate_code(env: ClarificationEnvironment, prompt: str, entry_point: str) -> str:
    response = call_llm(env, CODE_PROMPT.format(prompt=prompt, entry_point=entry_point))
    code = extract_code(response)
    if code:
        return code
    response = call_llm(env, RETRY_CODE_PROMPT.format(entry_point=entry_point))
    code = extract_code(response)
    if code:
        return code
    return response.strip()


def first_json(text: str) -> dict:
    if not isinstance(text, str):
        return {}
    decoder = json.JSONDecoder()
    for match in list(re.finditer(r"\{", text))[:8]:
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except (ValueError, RecursionError):
            continue
        if isinstance(value, dict):
            return value
    return {}


def parse_fork(text: str) -> tuple[str, str, str]:
    """Return (why, question, probe_block) or ("", "", "")."""
    if not isinstance(text, str) or not text.strip():
        return "", "", ""
    if "NO_FORK" in text.upper():
        return "", "", ""
    if "FORK" not in text.upper():
        return "", "", ""
    obj = first_json(text)
    why = str(obj.get("why") or "").strip()
    question = str(obj.get("question") or "").strip()
    probes = str(obj.get("probes") or "").strip()
    if not question or not probes:
        return "", "", ""
    return why, question, probes


def clean_question(question: str) -> str:
    question = question.replace("`", "").replace("\n", " ").strip()
    question = re.sub(r"\s+", " ", question)
    if len(question) > 400:
        return ""
    return question


def harness_script(candidates: list[str], entry: str, mode: str, payload: Any) -> str:
    return (
        "CANDIDATES_JSON = " + repr(json.dumps(candidates)) + "\n"
        "ENTRY_JSON = " + repr(entry) + "\n"
        "MODE_JSON = " + repr(mode) + "\n"
        "PAYLOAD_JSON = " + repr(json.dumps(payload)) + "\n" + HARNESS
    )


def run_harness(env: ClarificationEnvironment, candidates: list[str], entry: str, mode: str, payload: Any) -> dict:
    try:
        out = env.exec_code(harness_script(candidates, entry, mode, payload))
    except Exception:
        return {}
    idx = out.rfind("@@HARNESS@@")
    if idx < 0:
        return {}
    try:
        return json.loads(out[idx + len("@@HARNESS@@") :].strip().splitlines()[0])
    except (ValueError, IndexError):
        return {}


class WitnessProbe(ClarificationAlgorithmBase):
    DEFAULT_CONFIG: dict[str, Any] = {
        "num_final": 1,
        "num_tests": 6,
        "run_final_tests": True,
        "use_interface_question": True,
    }

    def cfg(self, key: str) -> Any:
        return self.config.get(key, self.DEFAULT_CONFIG[key])

    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        entry = problem["entry_point"]
        prompt = problem["prompt"]
        question: str | None = None
        answer: str | None = None

        # Stage 1: seed draft.
        base = generate_code(env, prompt, entry)

        # Stage 2: locate a single critical fork (if any).
        if env.can_ask():
            question = self.fork_question(env, prompt, entry, base)
            if question:
                try:
                    answer = env.ask_human(question)
                except TooManyQuestionException:
                    answer = None
                if answer is None or not answer.strip() or answer.strip() in (
                    "Cannot answer the given question.",
                    "Irrelevant question.",
                ):
                    question, answer = None, None

        # Stage 3: answer-priority regeneration + selection.
        if question is None or answer is None:
            return base
        clarification = (
            "\nThe user was asked: " + question + "\nAnswer: " + answer + "\n"
        )
        candidates = [base]
        for _ in range(max(1, int(self.cfg("num_final")))):
            code = extract_code(
                call_llm(
                    env,
                    FINAL_PROMPT.format(prompt=prompt, entry_point=entry, clarification=clarification),
                )
            )
            if code:
                candidates.append(code)
        if len(candidates) == 1 or not self.cfg("run_final_tests"):
            return candidates[0]
        return self.select(env, prompt, entry, candidates, clarification)

    def fork_question(self, env: ClarificationEnvironment, prompt: str, entry: str, base: str) -> str | None:
        if self.cfg("use_interface_question") and not re.search(r"def\s+\w+\s*\(", prompt):
            return INTERFACE_QUESTION.replace("{entry_point}", entry)
        raw = call_llm(
            env, ANALYSIS_PROMPT.format(prompt=prompt, entry_point=entry, candidate=base[:3500])
        )
        why, question, probes = parse_fork(raw)
        if not question:
            raw = call_llm(env, ANALYSIS_RETRY_PROMPT.format(prompt=prompt, entry_point=entry))
            why, question, probes = parse_fork(raw)
        question = clean_question(question)
        if not question:
            return None
        # Stage 2b: witness — does the other reading actually change behavior?
        fork_note = why or ("The candidate implements one reading; implement the other plausible reading of the ambiguous point: " + question)
        witness = extract_code(
            call_llm(
                env,
                SIDE_PROMPT.format(prompt=prompt, entry_point=entry, fork_note=fork_note, draft=base[:3000]),
            )
        )
        if not witness:
            return question
        result = run_harness(env, [base, witness], entry, "probe", probes)
        rows = [r for r in result.get("rows", []) if r is not None]
        n_inputs = len(result.get("inputs", []))
        if not n_inputs or len(rows) < 2:
            # witness evidence unreliable — the fork question still stands
            return question
        both_run = all(len(row) == n_inputs for row in rows)
        if not both_run:
            return question
        for i in range(n_inputs):
            if rows[0][i] != rows[1][i]:
                return question
        # both readings agree on every probe: not a real fork
        return None

    def select(self, env, prompt: str, entry: str, codes: list[str], clarification: str) -> str:
        resp = call_llm(
            env,
            TESTS_PROMPT.format(
                prompt=prompt,
                entry_point=entry,
                clarification=clarification,
                num_tests=int(self.cfg("num_tests")),
            ),
        )
        block = extract_code(resp) or ""
        asserts = [ln.strip() for ln in block.splitlines() if ln.strip().startswith("assert ")]
        if not asserts:
            return codes[0]
        result = run_harness(env, codes, entry, "asserts", asserts)
        rows = result.get("rows", [])
        if len(rows) != len(codes):
            return codes[0]
        scores = [sum(r) if r is not None else -1 for r in rows]
        best = max(scores)
        tied = [i for i, score in enumerate(scores) if score == best]
        row_of = {i: tuple(rows[i]) if rows[i] is not None else () for i in tied}
        top = max(tied, key=lambda i: sum(1 for j in tied if row_of[j] == row_of[i]))
        return codes[top]
