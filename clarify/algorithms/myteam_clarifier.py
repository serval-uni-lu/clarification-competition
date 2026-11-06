# SPDX-FileCopyrightText: 2026 Yunhao Zhang <yunhaozhang900@gmail.com>
# SPDX-FileCopyrightText: 2026 Semyon Zarutskiy <zarutskiysy@gmail.com>
#
# SPDX-License-Identifier: MIT

"""C1-EvidenceCheck-Public-v1 (DistilledClarifier).

Generate an initial implementation, ask one evidence-grounded clarification,
then regenerate the function using the task and the answer. Four ambiguity
categories guide question selection; the final prompt checks the answer's
scope, unchanged requirements, consistency, and unresolved assumptions.

Team: Hitsz Hago
Team Members: Yunhao Zhang, Jiahao Zhang, Haoman Li, Haojia Xiang, Lianghao Xia, Liqiang Nie, Erwei Yin
Main Contact: yunhaozhang900@gmail.com

The upstream MIT attribution is retained in the bundled source below.
"""

from __future__ import annotations

# Source: clarify/algorithms/zarutskiy_clarifier.py
# Historical source documentation (not submission metadata):
# ZarutskiyClarifier.
#
# Always asks exactly one clarifying question, then regenerates the function from
# the task plus the answer, with the answer overriding conflicting task text.
# An analysis prompt picks the one missing, vague or contradictory requirement
# that changes the output and asks one concrete, atomic question about it,
# phrased as an assumption the user can confirm or correct ("I will assume X.
# Is that right, or should it be Y?", where X is what the first draft does).
# A deterministic sanitizer aliases the required entry point when a draft renamed
# the function and drops __main__ blocks and top-level prints; questions are sent
# without backticks, because the simulated user's answer parser stops at the
# first backtick. Optional switches (off by default) ask a fixed call-interface
# question on prompts without a signature, add disagreement-driven questions
# (candidates run on probe inputs in one exec_code call), and select among
# several samples by execution.
#
# Prompts for the gated question and the regeneration baseline are adapted from
# the Okanagan baseline (Wu and Fard, TOSEM 2025); the disagreement idea follows
# ClarifyGPT (Mu et al., FSE 2024). Fresh regeneration with the answer taking
# precedence follows GatedClarification (STIL-ETS); AST hygiene follows
# ContractFirstClarifier (D4vidHuang) and GatedClarification; the backtick
# gotcha and single-exec batching follow the CuriosityByDesign notes
# (thibolu-uofa). Omega (OmegaTeam) asks interface questions concurrently.
#
# Team: zarutskiysy
# Team Members: Semyon Zarutskiy, Veronica Veselova
# Main Contact: zarutskiysy@gmail.com
# SPDX-FileCopyrightText: 2026 Semyon Zarutskiy <zarutskiysy@gmail.com>
#
# SPDX-License-Identifier: MIT
import ast
import json
import re
from typing import Any

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import (
    ClarificationEnvironment,
    LimitsExceededException,
    TooManyQuestionException,
)

CODE_PROMPT = """
Generate Python code directly (Markdown) to solve the coding problem implementing `{entry_point}`.

{prompt}

{name_rule}Enclose your solution in ```python and ```.
""".strip()

# Okanagan's question prompt (baseline wording), used for question_style="okanagan".
OKANAGAN_QUESTION_PROMPT = """
Given the programming problem and a generated candidate, ask ONE clarifying question if the requirements in the given problem description are incomplete, inconsistent or ambiguous
for solving the problem correctly and passing the tests.
If no need to ask clarifying questions, return strictly ’NO_QUESTIONS’ only. Otherwise, return the clarifying questions.

### Problem:

{prompt}

### Candidate

{candidate}
""".strip()

# Same wording without the opt-out, used for ask_policy="always" with question_style="okanagan".
OKANAGAN_ALWAYS_PROMPT = """
Given the programming problem and a generated candidate, ask ONE clarifying question about the requirement in the given problem description that is most likely incomplete, inconsistent or ambiguous
for solving the problem correctly and passing the tests.
Return only the clarifying question.

### Problem:

{prompt}

### Candidate

{candidate}
""".strip()

TARGETED_QUESTION_PROMPT = """
A user asked you to implement `{entry_point}`:

<problem>
{prompt}
</problem>

Your first draft:
```python
{candidate}
```

Before you finalize the code you may ask the user exactly ONE clarifying question; the user knows the intended behavior, and there is no second chance.

Use the four ambiguity categories as an evidence checklist for finding one unresolved requirement. Begin with the task text: identify the exact wording that permits different readings, or the precise fact that is absent. Lexical: determine which meaning of a word changes the required behavior. Syntactic: determine the referent or scope of a condition, modifier, or operation. Semantic: state the distinct input-to-output rules that the same statement could express. Vagueness: name the missing value, boundary, convention, or input condition needed to implement the task. Check each interpretation against the signature, examples, and other explicit statements; discard interpretations those sources already resolve. Keep a candidate only when resolving it can change accepted inputs or required outputs. Treat contradictions as disagreements between explicit sources and identify the disputed behavior. Do not create ambiguity from implementation preferences or from assumptions made only by the draft. Select one supported uncertainty; the categories are diagnostic aids, not separate questions.

Locate the single unresolved requirement that most affects what `{entry_point}` must accept or return for ordinary inputs. In ANALYSIS, identify the exact ambiguous wording or conflicting statements, or name the missing fact. For ambiguity, state two contextually plausible interpretations and a concrete difference in their required outputs or accepted inputs; use a tiny ordinary input when helpful. Check the description, signature, and examples before treating an interpretation as unresolved. For a contradiction, identify the disagreeing sources and the behavior that depends on which is intended. When a value or rule is missing, identify exactly what is unspecified; do not invent arbitrary alternatives. Treat the draft as an unconfirmed interpretation, not evidence of user intent, and prefer the uncertainty whose answer is most likely to correct its behavior. Do not confuse an implementation bug or a stylistic choice with a requirement ambiguity. Ask about exactly that one fact, making QUESTION understandable on its own without the ANALYSIS; use concrete alternatives only when supported by the task.

Answer in exactly this format:
ANALYSIS: <2-4 sentences: the competing readings and how their outputs differ>
QUESTION: <one short question about exactly one fact, answerable with a single value or choice. When it helps, name the concrete alternatives or a tiny example input, e.g. "Should f([3, 1, 3]) return 2 or 3?">

{interface_hint}Rules for the question: ask about required behavior (a value, a rule, a return type, an input format, an edge-case convention), never about implementation details or test cases; do not bundle several questions; do not ask to confirm something the description already states clearly. Write it in plain text: no Markdown, no backticks.
""".strip()

INTERFACE_QUESTION = "Which arguments does {entry_point} take, and in what order?"

ASSUME_FRAMING_TEXT = 'QUESTION: <one short question about exactly one fact, phrased as an assumption for the user to confirm or correct: "I will assume X. Is that right, or should it be Y?", where X is what your draft does at that point and Y is the most plausible other reading. Name concrete values or a tiny example input when it helps, e.g. "I will assume f([3, 1, 3]) returns 2. Is that right, or should it return 3?">'

INTERFACE_HINT = """The description gives no signature, so the user's tests will call `{entry_point}` with arguments whose number, order and types you have to guess. If your draft's parameters are not clearly determined by the description, ask about them (e.g. "Does the function take the list and the target value as two separate arguments, in that order?").

"""

MULTI_QUESTION_PROMPT = """
A user asked you to implement `{entry_point}`:

<problem>
{prompt}
</problem>

Your first draft:
```python
{candidate}
```

Before you finalize the code you may send the user ONE message with clarifying questions; there is no second chance.
List the points where the description is missing information, is vague, or contradicts itself in a way that changes what `{entry_point}` must return for ordinary inputs, most important first.

Answer in exactly this format:
ANALYSIS: <2-4 sentences>
QUESTIONS:
1. <short question about one fact>
2. <short question about one fact>
3. <short question about one fact>

Ask at most {max_questions} questions, each about required behavior (a value, a rule, a return type, an input format, an edge-case convention), never about implementation details or test cases. Write them in plain text: no Markdown, no backticks.
""".strip()

PROBE_PROMPT = """
Problem for `{entry_point}`:

{prompt}

Write {num_probes} small, valid test inputs for `{entry_point}` that would expose differences between plausible readings of this description: the examples given in the description (if any), typical cases, and boundary cases (empty input, ties, duplicates, zero, boundaries) where readings could differ. Use only inputs the description allows; skip invalid types or values that the function would reject.
Return only a Python code block that defines a list named INPUTS; each element is a tuple with the positional arguments of one call, e.g.
```python
INPUTS = [
    ([1, 2, 3], 2),
    ([], 0),
]
```
""".strip()

DISAGREEMENT_QUESTION_PROMPT = """
A user asked you to implement `{entry_point}`:

<problem>
{prompt}
</problem>

You wrote several candidate implementations. They disagree on these inputs, which means the description admits different readings:

{table}

Before you finalize the code you may ask the user exactly ONE clarifying question; there is no second chance.
Pick the disagreement that matters most for ordinary, valid inputs (ignore differences in error handling for invalid inputs) and ask about the underlying requirement in one short, concrete question about exactly one fact. You may cite one of the inputs above with its alternative outputs (e.g. "Should f([3, 1, 3]) return 2 or 3?").

Answer in exactly this format:
ANALYSIS: <1-3 sentences: which readings the outputs correspond to>
QUESTION: <the question>

Never ask about implementation details or test cases, and do not bundle several questions. Write the question in plain text: no Markdown, no backticks.
""".strip()

# Okanagan's regeneration prompt (baseline wording), used for integrate="okanagan".
OKANAGAN_REGEN_PROMPT = """
{prompt}
{clarification}

Given the above conversations, generate Python code directly (Markdown) to solve the coding problem implementing `{entry_point}`:
"""

FINAL_PROMPT = """
Implement `{entry_point}` for the task below.

<problem>
{prompt}
</problem>

You asked the user a clarifying question and got an answer:
Question: {question}
Answer: {answer}

The answer states the user's intent: where it conflicts with the task text or its examples, follow the answer; everything the answer does not mention follows the task text. Keep the function name `{entry_point}`.
Enclose your solution in ```python and ```.
""".strip()

REWRITE_PROMPT = """
Task description for `{entry_point}`:

<problem>
{prompt}
</problem>

Clarification from the user:
Question: {question}
Answer: {answer}

Rewrite the task description into one complete and consistent specification that incorporates the user's answer. Where the answer conflicts with the original text or its examples, the answer wins; keep everything else, including the function name `{entry_point}`, its parameters, and the examples that are consistent with the answer (fix or drop the others). Output only the rewritten specification.
""".strip()

FROM_SPEC_PROMPT = """
Implement `{entry_point}` according to this specification:

<specification>
{spec}
</specification>

For reference, the user's original wording was:
<original>
{prompt}
</original>

Follow the specification where the two differ. Enclose your solution in ```python and ```.
""".strip()

TESTS_PROMPT = """
Task for `{entry_point}`:

<problem>
{prompt}
</problem>
{clarification}
Write up to {num_tests} assert statements that check `{entry_point}` on concrete inputs whose expected output follows unambiguously from the task text, its examples, or the clarification above. Do not invent behavior that is not stated; skip a case rather than guess.
Return only a Python code block with one `assert {entry_point}(...) == ...` statement per line.
""".strip()

# Runs several candidates on several inputs inside ONE exec_code call and prints JSON.
HARNESS = r"""
import copy, json, signal

def _alarm(signum, frame):
    raise TimeoutError("probe timeout")

signal.signal(signal.SIGALRM, _alarm)
CANDIDATES = json.loads(CANDIDATES_JSON)
ENTRY = ENTRY_JSON
MODE = MODE_JSON
PAYLOAD = json.loads(PAYLOAD_JSON)

def _load(code):
    ns = {"__name__": "candidate_module"}
    signal.alarm(5)
    try:
        exec(code, ns)
    finally:
        signal.alarm(0)
    return ns

rows = []
inputs = None
if MODE == "probe":
    ins = {}
    try:
        signal.alarm(5)
        exec(PAYLOAD, ins)
        signal.alarm(0)
        inputs = list(ins["INPUTS"])[:20]
    except BaseException as e:
        signal.alarm(0)
        inputs = []
for code in CANDIDATES:
    row = []
    try:
        ns = _load(code)
        fn = ns[ENTRY]
    except BaseException as e:
        rows.append(None)
        continue
    if MODE == "probe":
        for inp in inputs:
            args = inp if isinstance(inp, tuple) else (inp,)
            try:
                signal.alarm(2)
                out = fn(*copy.deepcopy(args))
                signal.alarm(0)
                row.append(repr(out)[:120])
            except BaseException as e:
                signal.alarm(0)
                row.append("raises " + type(e).__name__)
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
print("@@HARNESS@@" + json.dumps({"inputs": [repr(i)[:120] for i in (inputs or [])], "rows": rows}))
"""


def extract_code(response: str) -> str | None:
    """Code inside the first ```python ... ``` block (same rule as the SDK's result parser)."""
    if "```python" not in response:
        return None
    _, rest = response.split("```python", 1)
    if "```" not in rest:
        return None
    return rest.split("```", 1)[0]


def escape_closing_tags(text: str) -> str:
    """Escape rich-markup closing tags such as `[/-]` (e.g. inside a regex character class):
    the harness prints every prompt with rich, which raises on an unmatched closing tag."""
    return re.sub(r"(\\*)(\[/[^\[]*?\])", lambda m: m.group(1) * 2 + "\\" + m.group(2), text)


def call_llm(env: ClarificationEnvironment, messages: list[dict[str, str]] | str) -> str:
    """env.llm with up to two retries on errors other than the budget limit; retries escape
    rich-markup closing tags in the last message, the only text the harness prints."""
    attempts = 3
    for attempt in range(attempts):
        try:
            return env.llm(messages)
        except LimitsExceededException:
            raise
        except Exception:
            if attempt == attempts - 1:
                raise
            if isinstance(messages, str):
                messages = escape_closing_tags(messages)
            else:
                last = dict(messages[-1])
                last["content"] = escape_closing_tags(last["content"])
                messages = messages[:-1] + [last]
    raise AssertionError("unreachable")


def as_code(text: str) -> str:
    """Code from a response that may hold an unclosed ```python fence (e.g. a truncated answer)."""
    if "```python" in text:
        text = text.split("```python", 1)[1]
        if "```" in text:
            text = text.split("```", 1)[0]
    return text


def after_tag(text: str, tag: str) -> str:
    idx = text.rfind(tag)
    if idx < 0:
        return ""
    return text[idx + len(tag) :].strip()


def norm_name(name: str) -> str:
    return name.lower().replace("_", "")


def sanitize_code(code: str, entry: str, prompt: str) -> str:
    """Drop `if __name__ == "__main__"` blocks and top-level print/input/main calls, and alias
    `entry` to the implemented function when the draft used another name."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    lines = code.splitlines()
    drop = []
    for node in tree.body:
        if isinstance(node, ast.If):
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
            ):
                drop.append((node.lineno, node.end_lineno))
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Name) and func.id in ("print", "input", "main"):
                drop.append((node.lineno, node.end_lineno))
    for start, end in sorted(drop, reverse=True):
        del lines[start - 1 : end]
    code = "\n".join(lines) + "\n"
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    defined = set()
    funcs = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
            funcs.append(node)
        elif isinstance(node, ast.ClassDef):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined.add(target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                defined.add((alias.asname or alias.name).split(".")[0])
    if entry in defined or not funcs:
        return code
    names = [f.name for f in funcs]
    chosen = None
    for name in names:
        if norm_name(name) == norm_name(entry):
            chosen = name
    if chosen is None:
        prompt_defs = set(re.findall(r"def\s+(\w+)\s*\(", prompt))
        for name in reversed(names):
            if name in prompt_defs:
                chosen = name
                break
    if chosen is None:
        called = set()
        for f in funcs:
            for sub in ast.walk(f):
                if isinstance(sub, ast.Name):
                    called.add(sub.id)
        roots = [f.name for f in funcs if f.name not in called - {f.name}]
        chosen = roots[-1] if roots else names[-1]
    return code + f"\n\n{entry} = {chosen}\n"


def plain_question(text: str | None) -> str | None:
    if not text:
        return None
    text = text.replace("`", "").strip()
    return text or None


def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
    return text


class _ZarutskiyClarifier(ClarificationAlgorithmBase):
    DEFAULT_CONFIG: dict[str, Any] = {
        "ask_policy": "always",  # "gate": the question prompt may opt out (Okanagan); "always": always ask
        "question_style": "targeted",  # "okanagan" | "targeted" | "disagreement" | "multi"
        "integrate": "final",  # "okanagan" (baseline regen prompt) | "final" | "rewrite"
        "num_candidates": 3,  # candidates sampled for disagreement probing
        "num_probes": 10,  # probe inputs requested for disagreement probing
        "max_questions": 3,  # cap for question_style="multi"
        "num_final": 1,  # final samples; > 1 enables execution-based selection
        "num_tests": 8,  # self-written asserts for execution-based selection
        "generation_attempts": 3,  # retries when a response has no ```python block
        "sanitize": True,  # entry-point alias + drop __main__ blocks / top-level prints
        "plain_questions": True,  # strip backticks (the simulator's ANSWERS regex stops at one)
        "exact_name": True,  # ask for the exact entry-point name in code prompts
        "interface_hint": False,  # signature-less prompts: point the question at the call interface
        "interface_question": False,  # signature-less prompts: ask which arguments the function takes
        "assume_framing": True,  # P05: phrase the question as an assumption to confirm or correct
    }

    def cfg(self, key: str) -> Any:
        return self.config.get(key, self.DEFAULT_CONFIG[key])

    # -- generation helpers -------------------------------------------------

    def generate_code(self, env: ClarificationEnvironment, content: str) -> tuple[str, str]:
        """Returns (code or "", last raw response)."""
        messages = [{"role": "user", "content": content}]
        response = ""
        for _ in range(max(1, int(self.cfg("generation_attempts")))):
            response = call_llm(env, messages)
            code = extract_code(response)
            if code is not None and code.strip():
                return code, response
            messages = messages + [
                {"role": "assistant", "content": response},
                {
                    "role": "user",
                    "content": "Please answer with the complete implementation enclosed in ```python and ```.",
                },
            ]
        return "", response

    def code_prompt(self, problem) -> str:
        rule = "Name the function exactly `{entry_point}`. " if self.cfg("exact_name") else ""
        return (
            CODE_PROMPT.replace("{name_rule}", rule)
            .replace("{prompt}", problem["prompt"])
            .replace("{entry_point}", problem["entry_point"])
        )

    def seed_candidate(self, env, problem) -> str:
        content = self.code_prompt(problem)
        code, raw = self.generate_code(env, content)
        return code or as_code(raw)

    def run_harness(self, env, candidates: list[str], entry: str, mode: str, payload: Any) -> dict:
        script = (
            "CANDIDATES_JSON = "
            + repr(json.dumps(candidates))
            + "\nENTRY_JSON = "
            + repr(entry)
            + "\nMODE_JSON = "
            + repr(mode)
            + "\nPAYLOAD_JSON = "
            + repr(json.dumps(payload))
            + "\n"
            + HARNESS
        )
        try:
            out = env.exec_code(script)
        except Exception:
            return {}
        idx = out.rfind("@@HARNESS@@")
        if idx < 0:
            return {}
        try:
            return json.loads(out[idx + len("@@HARNESS@@") :].strip().splitlines()[0])
        except (ValueError, IndexError):
            return {}

    # -- question styles ------------------------------------------------------

    def question_okanagan(self, env, problem, candidate) -> str | None:
        template = (
            OKANAGAN_QUESTION_PROMPT if self.cfg("ask_policy") == "gate" else OKANAGAN_ALWAYS_PROMPT
        )
        response = call_llm(
            env,
            template.replace("{prompt}", problem["prompt"]).replace("{candidate}", candidate),
        )
        if "```" in response:
            _, response = response.split("```", 1)
        if "```" in response:
            response, _ = response.split("```", 1)
        if self.cfg("ask_policy") == "gate" and "NO_QUESTIONS" in response:
            return None
        return response.strip() or None

    def question_targeted(self, env, problem, candidate) -> str | None:
        hint = ""
        if self.cfg("interface_hint") and not re.search(r"def\s+\w+\s*\(", problem["prompt"]):
            hint = INTERFACE_HINT.replace("{entry_point}", problem["entry_point"])
        template = TARGETED_QUESTION_PROMPT
        if self.cfg("assume_framing"):
            template = template.replace(
                'QUESTION: <one short question about exactly one fact, answerable with a single value or choice. When it helps, name the concrete alternatives or a tiny example input, e.g. "Should f([3, 1, 3]) return 2 or 3?">',
                ASSUME_FRAMING_TEXT,
            )
        response = call_llm(
            env,
            template.replace("{interface_hint}", hint)
            .replace("{prompt}", problem["prompt"])
            .replace("{entry_point}", problem["entry_point"])
            .replace("{candidate}", candidate),
        )
        question = strip_fences(after_tag(response, "QUESTION:"))
        if not question:
            question = response.strip().splitlines()[-1] if response.strip() else ""
        return question or None

    def question_multi(self, env, problem, candidate) -> str | None:
        response = call_llm(
            env,
            MULTI_QUESTION_PROMPT.replace("{prompt}", problem["prompt"])
            .replace("{entry_point}", problem["entry_point"])
            .replace("{candidate}", candidate)
            .replace("{max_questions}", str(self.cfg("max_questions"))),
        )
        block = after_tag(response, "QUESTIONS:")
        lines = [ln.strip() for ln in block.splitlines() if re.match(r"^\s*\d+[.)]\s+\S", ln)]
        lines = lines[: max(1, int(self.cfg("max_questions")))]
        return "\n".join(lines) if lines else None

    def question_disagreement(self, env, problem, candidate) -> tuple[str | None, dict]:
        entry = problem["entry_point"]
        base = self.code_prompt(problem)
        candidates = [candidate]
        for _ in range(max(0, int(self.cfg("num_candidates")) - 1)):
            code, _ = self.generate_code(env, base)
            if code:
                candidates.append(code)
        probe_resp = call_llm(
            env,
            PROBE_PROMPT.replace("{prompt}", problem["prompt"])
            .replace("{entry_point}", entry)
            .replace("{num_probes}", str(self.cfg("num_probes"))),
        )
        probe_code = extract_code(probe_resp) or ""
        info = {"num_candidates": len(candidates), "disagreements": 0}
        if not probe_code or len(candidates) < 2:
            return None, info
        result = self.run_harness(env, candidates, entry, "probe", probe_code)
        rows = [r for r in result.get("rows", []) if r is not None]
        inputs = result.get("inputs", [])
        if len(rows) < 2 or not inputs:
            return None, info
        valid_lines, error_lines = [], []
        for j, inp in enumerate(inputs):
            outs = [r[j] for r in rows if j < len(r)]
            distinct = sorted(set(outs))
            if len(distinct) > 1:
                info["disagreements"] += 1
                call = entry + (inp if inp.startswith("(") else "(" + inp + ")")
                line = f"- {call}: " + " vs. ".join(distinct)
                if any(o.startswith("raises ") for o in outs):
                    error_lines.append(line)
                else:
                    valid_lines.append(line)
        # disagreements on valid inputs first: tests exercise valid inputs, not error handling
        table_lines = (valid_lines + error_lines)[:5]
        if not table_lines:
            return None, info
        response = call_llm(
            env,
            DISAGREEMENT_QUESTION_PROMPT.replace("{prompt}", problem["prompt"])
            .replace("{entry_point}", entry)
            .replace("{table}", "\n".join(table_lines)),
        )
        question = strip_fences(after_tag(response, "QUESTION:"))
        return (question or None), info

    # -- integration ------------------------------------------------------------

    def final_code(self, env, problem, question: str | None, answer: str | None) -> list[str]:
        entry = problem["entry_point"]
        n = max(1, int(self.cfg("num_final")))
        if question is None:
            content = self.code_prompt(problem)
        elif self.cfg("integrate") == "okanagan":
            clar = f"Questions #1:\n{question}\nAnswers:\n{answer}\n"
            content = (
                OKANAGAN_REGEN_PROMPT.replace("{prompt}", problem["prompt"])
                .replace("{clarification}", clar)
                .replace("{entry_point}", entry)
            )
        elif self.cfg("integrate") == "rewrite":
            spec = call_llm(
                env,
                REWRITE_PROMPT.replace("{prompt}", problem["prompt"])
                .replace("{entry_point}", entry)
                .replace("{question}", question)
                .replace("{answer}", answer or ""),
            ).strip()
            content = (
                FROM_SPEC_PROMPT.replace("{spec}", spec)
                .replace("{prompt}", problem["prompt"])
                .replace("{entry_point}", entry)
            )
        else:
            content = (
                FINAL_PROMPT.replace("{prompt}", problem["prompt"])
                .replace("{entry_point}", entry)
                .replace("{question}", question)
                .replace("{answer}", answer or "")
            )
        codes = []
        last_raw = ""
        for _ in range(n):
            code, raw = self.generate_code(env, content)
            last_raw = raw
            if code:
                codes.append(code)
        return codes or [as_code(last_raw)]

    def select(self, env, problem, codes: list[str], question, answer) -> str:
        if len(codes) == 1:
            return codes[0]
        entry = problem["entry_point"]
        clar = ""
        if question:
            clar = f"\nClarification from the user:\nQuestion: {question}\nAnswer: {answer}\n"
        resp = call_llm(
            env,
            TESTS_PROMPT.replace("{prompt}", problem["prompt"])
            .replace("{entry_point}", entry)
            .replace("{clarification}", clar)
            .replace("{num_tests}", str(self.cfg("num_tests"))),
        )
        block = extract_code(resp) or ""
        asserts = [ln.strip() for ln in block.splitlines() if ln.strip().startswith("assert ")]
        if not asserts:
            return codes[0]
        result = self.run_harness(env, codes, entry, "asserts", asserts)
        rows = result.get("rows", [])
        if len(rows) != len(codes):
            return codes[0]
        scores = [sum(r) if r is not None else -1 for r in rows]
        best = max(scores)
        tied = [i for i, sc in enumerate(scores) if sc == best]
        # tie-break by agreement: prefer the pass/fail pattern shared by most candidates
        patterns = [tuple(rows[i]) if rows[i] is not None else () for i in tied]
        counts = {pat: patterns.count(pat) for pat in patterns}
        top = max(tied, key=lambda i: counts[tuple(rows[i]) if rows[i] is not None else ()])
        return codes[top]

    # -- main ---------------------------------------------------------------------

    def clean(self, code: str, problem: dict[str, Any]) -> str:
        if not self.cfg("sanitize"):
            return code
        return sanitize_code(code, problem["entry_point"], problem["prompt"])

    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        candidate = self.clean(self.seed_candidate(env, problem), problem)
        try:
            return self.refine(env, problem, candidate)
        except Exception:
            # any failure after the seed (budget, harness console error) keeps the seed draft
            return candidate

    def refine(self, env: ClarificationEnvironment, problem: dict[str, Any], candidate: str) -> str:
        question, answer = None, None
        if env.can_ask():
            style = self.cfg("question_style")
            if self.cfg("interface_question") and not re.search(
                r"def\s+\w+\s*\(", problem["prompt"]
            ):
                # the call interface is the first thing a signature-less draft gets wrong
                style = "interface"
            if style == "interface":
                question = INTERFACE_QUESTION.replace("{entry_point}", problem["entry_point"])
            elif style == "okanagan":
                question = self.question_okanagan(env, problem, candidate)
            elif style == "multi":
                question = self.question_multi(env, problem, candidate)
            elif style == "disagreement":
                question, probe_info = self.question_disagreement(env, problem, candidate)
                if question is None and self.cfg("ask_policy") == "always":
                    question = self.question_targeted(env, problem, candidate)
            else:
                question = self.question_targeted(env, problem, candidate)
            if self.cfg("plain_questions"):
                question = plain_question(question)
            if question is not None:
                try:
                    answer = env.ask_human(question)
                except TooManyQuestionException:
                    question, answer = None, None
        if question is None and int(self.cfg("num_final")) <= 1:
            return candidate
        codes = [self.clean(c, problem) for c in self.final_code(env, problem, question, answer)]
        return self.select(env, problem, codes, question, answer)


# Source: clarify/algorithms/contrast_question_clarifier.py
# Historical source documentation (not submission metadata):
# Training prototype: one behavioral question with a distinguishing example.
#
# The final prompt is frozen to the original CoT experiment. Only the question
# prompt differs between FrozenCoTClarifier and ContrastQuestionClarifier.


FROZEN_COT_FINAL_PROMPT = """
Implement `{entry_point}` for the task below.

<problem>
{prompt}
</problem>

You asked the user a clarifying question and got an answer:
Question: {question}
Answer: {answer}

The answer states the user's intent: where it conflicts with the task text or its
examples, follow the answer; everything the answer does not mention follows the task text.

Reason through the problem step by step before writing any code:

1. What the answer settles: in one sentence, state the fact or rule the answer
   establishes, and name any part of the original text it overrides or refines.
2. Complete requirement: restate the full, de-conflicted requirement in one or two
   sentences.
3. Edge cases: list every boundary or special case (empty input, zero, negative,
   ties, duplicates, boundary values) and state the exact correct behavior for each,
   consulting the answer wherever it applies.
4. Implementation: write the code that satisfies the requirement and all edge cases.

Keep the function name exactly `{entry_point}`. Enclose your solution in ```python and ```.
""".strip()


CONTRAST_QUESTION_PROMPT = """
A user asked you to implement {entry_point}:

<problem>
{prompt}
</problem>

Your first draft (a fallible guess, not a specification):
<draft>
{candidate}
</draft>

You may ask exactly ONE short clarification about ONE behavioral requirement.
Choose the unresolved fact whose answer would change the implementation most.

First locate the actual missing rule or contradiction. Check the basic operation,
return meaning/structure, and any explicitly missing parameter or membership rule
before considering peripheral cases. Do not spend the question on a detail that
leaves the main operation unspecified. A fact already stated consistently in the
description is not missing. When two statements or an example conflict, ask which
behavior is intended rather than silently selecting one. Ignore implementation
restrictions and unusual invalid inputs when an ordinary-input rule is unresolved.

Construct two plausible interpretations of that ONE fact and ONE small valid
input for which they give different outputs. Work out both outputs carefully.
Use the example to ask about the general rule, not to request a hidden test.
If a rule is completely omitted (such as what one output field means or which
values belong to a named category), the two options are hypotheses; explicitly
allow the user to supply the intended rule instead. Do not fabricate a complete
table, ask for several independent facts, or ask for code or an algorithm.
If the ambiguity concerns the accepted call format, illustrate the competing
call forms instead of making up an input outside the domain.

Return exactly:
ANALYSIS: <2-4 sentences identifying the unresolved fact, interpretations A and B,
and why resolving this fact changes the draft.>
WITNESS: <one input/call; output under A; different output under B>
QUESTION: <one concise question about that rule, illustrated by the witness.
Do not imply the options are exhaustive if the rule is absent.>

Only the QUESTION is sent to the user. Plain text only, no Markdown or backticks.
Do not bundle questions or ask the user to provide a full specification.
""".strip()


def render(template, values):
    """Substitute once so task text containing placeholders is preserved."""
    return re.sub(
        r"\{(entry_point|prompt|candidate|question|answer)\}",
        lambda match: values[match[1]],
        template,
    )


class _FrozenCoTClarifier(_ZarutskiyClarifier):
    def final_code(self, env, problem, question, answer):
        content = render(
            FROZEN_COT_FINAL_PROMPT,
            {
                **problem,
                "question": question or "",
                "answer": answer or "",
            },
        )
        code, raw = self.generate_code(env, content)
        return [code or as_code(raw)]


class _ContrastQuestionClarifier(_FrozenCoTClarifier):
    def question_targeted(self, env, problem, candidate):
        response = call_llm(
            env,
            render(
                CONTRAST_QUESTION_PROMPT,
                {
                    **problem,
                    "candidate": candidate,
                },
            ),
        )
        self.question_analysis = response
        question = strip_fences(after_tag(response, "QUESTION:"))
        if not question:
            # Use the same deterministic malformed-response fallback as the baseline.
            question = response.strip().splitlines()[-1] if response.strip() else ""
        return question or None


# Source: clarify/algorithms/distilled_clarifier.py
# Historical source documentation (not submission metadata):
# SDK entry point for frozen distilled v1; observed validation range 543–555/770.
#
# The prompt is embedded so generation does not require private runs/ artifacts.
# No model weights are trained by this prompt-distillation experiment.

DISTILLED_FINAL_PROMPT = r"""
Implement `{entry_point}` using the task and its clarification below.

Task:
<problem>
{prompt}
</problem>
Clarification question: {question}
User answer: {answer}

Use the answer as evidence of user intent. An explicit correction overrides the
conflicting part of the task or its examples. Preserve requirements the answer
does not change. A question's proposed assumption is not a confirmed requirement
unless the answer actually confirms it. An evasive answer adds no missing facts.

Before the implementation, write a short requirements check, at most six bullets:
- Contract: identify the input interface, output form, and required behavior.
  Distinguish explicit requirements from necessary implementation assumptions.
- Answer scope: state exactly what was settled or corrected; do not extend a
  narrow answer to other undecided requirements.
- Consistency: check the proposed behavior against examples supplied in this
  task that the answer has not overridden. Do not claim to have executed code.
- Gaps: identify only unresolved details that can change the result. If any
  remain, choose a minimal, consistent assumption and label it as an assumption.
  Do not claim that the user confirmed it or that hidden tests require it.

Use the following brief demonstrations of these checks. They are illustrations
of the checking method, not rules or implementations for the current task.

Demonstration 1 — a correction has a specific scope.
Task excerpt: "Return list of all prefixes from shortest to longest";
example: all_prefixes('abc') -> ['abc', 'ab', 'a'].
Question: "I will assume all_prefixes returns prefixes from longest to shortest as shown in the example ['abc', 'ab', 'a']. Is that right, or should it return from shortest to longest as the description states?"
Answer: "The function should return prefixes from shortest to longest, as stated in the description."
Requirements check:
- The answer settles ordering and overrides the conflicting example order.
- The input remains a string; the result remains a list of prefixes. Neither
  the data type nor the definition of a prefix was changed by the answer.
- For 'abc', the clarified ordering is ['a', 'ab', 'abc']; this is a derived
  illustration, not a claim that the original example or a test passed.

Demonstration 2 — a partial answer leaves a real gap.
Task excerpt: numerical_letter_grade(grades) receives GPAs and returns
"a list of percentage or letter grades using the following table", but no
conversion table is supplied.
Question: "I will assume the function returns "100%" only for a GPA of 4.0 and letter grades for all other GPAs. Is that right, or should the function return percentage grades for other GPAs as well?"
Answer: "The function should return letter grades for all GPAs, including 4.0, and not percentage grades."
Requirements check:
- Letter-grade output is confirmed; the GPA-to-letter conversion thresholds
  are still absent. The answer does not supply or confirm a grading table.
- If a conversion rule must be chosen to return an implementation, identify
  it as an assumption. Do not present a familiar grading scheme as supplied fact.

Demonstration 3 — use the supplied examples to constrain the implementation.
Task excerpt: return factors in ascending order, each repeated as often as it
occurs in the decomposition; their product equals the input.
Examples: decompose(8) -> [2, 2, 2]; decompose(25) -> [5, 5];
decompose(70) -> [2, 5, 7].
Question: "I will assume the input to factorize is always a positive integer greater than 1. Is that right, or should the function handle 1, 0, or negative integers differently?"
Answer: "The function should only handle positive integers greater than 1 as input; other values do not need to be handled."
Requirements check:
- The answer restricts the input domain; it does not replace the factorization
  requirement. Repeated factors are required by the text and examples.
- A distinct-divisors list would contradict the example for 8. Extracting a
  divisor only once would also miss repeated factors; preserve multiplicity.
- Return the required entry-point name even if the task uses a placeholder.

Now apply the checks to the current task. Spend effort on decisions that affect
its output, rather than inventing behavior for inputs excluded by its contract.
Then provide one complete implementation, named exactly `{entry_point}`, in one
```python code block. Use no other code blocks. Do not return only an analysis
or ask another question. Do not include demonstrations in the implementation.
""".strip()


class DistilledClarifier(_FrozenCoTClarifier):
    def final_code(self, env, problem, question, answer):
        content = render(
            DISTILLED_FINAL_PROMPT,
            {
                **problem,
                "question": question or "",
                "answer": answer or "",
            },
        )
        code, raw = self.generate_code(env, content)
        return [code or as_code(raw)]
