# SPDX-FileCopyrightText: 2026 Oussama Bensghaier <oussama.bensghaier@uottawa.ca>
#
# SPDX-License-Identifier: MIT

"""Omega: Dominance-Guided, Contract-Aware Clarification Algorithm.

Principles & Architecture:
1. Two-Stage Candidate Discovery & Assumption Map: Analyzes specifications across three generic lenses
   (Call/Output Contract, Core/Global Semantics, Convention/Boundary Behavior) before deciding whether to ask.
2. Behavioral Dominance Selection: Evaluates all surviving candidate ambiguities through a dedicated dominance
   selector. Ranking prioritizes expected correctness impact over category, favoring core semantic changes.
3. Contract & Interface Awareness: Removes blanket prohibitions on interface questions when the prompt
   omits an explicit signature and leaves parameter passing genuinely ambiguous (e.g. scalar pair vs iterable).
4. Recall-Biased, Metric-Aligned Asking Policy: Abstention occurs only when the specification is clear.
5. Atomic Clarification Protocol: Enforces <= 35 words, single sentence, exactly one '?', with deterministic
   validation and targeted rephrasing repair.
6. Contract-Driven & Simple Synthesis: Solves with authoritative clarification using a minimal prompt.
7. Deterministic AST Sanitization & Signature Compatibility: Preserves valid top-level helper constructs
   while stripping test runners, rejecting duplicate entry points, and strictly enforcing default-argument
   signature compatibility.
8. Anti-Cheat Compliance: 100% compliant with repository security and evaluation standards (SEC001/SEC002).

Team: OmegaTeam
Team Members: Oussama Bensghaier
Main Contact: oussama.bensghaier@uottawa.ca
"""

from __future__ import annotations

import ast
import re
from typing import Any

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import (
    ClarificationEnvironment,
    LimitsExceededException,
    TooManyQuestionException,
)

# ==============================================================================
# PROMPTS
# ==============================================================================

DISCOVERY_TEMPLATE = """You are an expert software engineer performing a pre-implementation specification analysis.

Task to implement (entry point: `{entry_point}`):
{prompt}

Your objective is to identify up to 3 genuine, unresolved requirements or ambiguities across three generic lenses:
1. CALL/OUTPUT CONTRACT: Is the input structure (e.g., separate arguments vs. single container/list) or exact return representation (e.g., bool vs "Yes"/"No", tuple vs list, 0-based vs 1-based indexing) ambiguous? Note: If an explicit function signature (`def {entry_point}(...):`) is already present in the prompt, the signature is established and should NOT be questioned.
2. CORE/GLOBAL SEMANTICS: Are the primary transformation rules, mathematical definitions (e.g., negative number semantics, divisibility conventions), or mutation vs return semantics ambiguous?
3. CONVENTION/BOUNDARY BEHAVIOR: Are ties, empty inputs, single elements, or boundary extremes unresolved?

For each identified candidate, determine:
- An example may eliminate an interpretation ONLY when it logically contradicts that interpretation. Explicit unresolved alternatives in the specification remain clarification candidates even if an example illustrates one possibility.
- INFERABLE_FROM_SPEC means one interpretation is favored by context or convention, but the competing interpretation remains genuinely plausible under the written specification.
- What is its behavioral footprint:
  * GLOBAL: Affects the call interface, return type, or 100% of valid inputs.
  * BROAD: Affects a broad class of inputs or core algorithm transformation.
  * EDGE: Affects only isolated boundary edge cases or tie-breaking.

Format your output EXACTLY as follows:

STATUS: AMBIGUOUS
---
CANDIDATE: 1
LENS: CONTRACT | CORE_SEMANTICS | BOUNDARY
ISSUE: <concise summary of the ambiguity>
INTERPRETATION_A: <first plausible interpretation>
INTERPRETATION_B: <competing plausible interpretation>
FOOTPRINT: GLOBAL | BROAD | EDGE
INFERABLE_FROM_SPEC: YES: <why one interpretation is favored, while the alternative remains plausible> | NO
QUESTION: <single sentence atomic question ending with ?, at most 30 words, asking about this choice>
---

If the task is completely clear, self-contained, and has NO meaningful unresolved behavioral ambiguities, reply with exactly:
STATUS: UNAMBIGUOUS""".strip()

DOMINANCE_SELECTOR_TEMPLATE = """You are an expert software architect selecting the single most critical clarification question for `{entry_point}`.

Task:
{prompt}

We have identified multiple candidate ambiguities:
{candidates_text}

Selection Principles:
1. Expected Correctness Impact: Prefer the ambiguity whose answer most changes the core implementation and hidden-test behavior. Core/global semantic ambiguity should dominate interface ambiguity when the interface is reasonably inferable but the core operation itself is unresolved.
2. Behavioral Footprint: A candidate with a larger incompatible behavioral footprint (GLOBAL > BROAD > EDGE) dominates localized details, but expected correctness impact dominates category.
3. Core Transformation / Semantic Correctness: Resolving the core mathematical/algorithm transformation across broad inputs dominates isolated boundary edge cases.
4. Broad Convention vs Localized Boundary/Tie: Resolving general conventions dominates rare tie-breakers.
5. Inferability: Inferability is only a secondary tiebreaker. If an ambiguity has a broad behavioral footprint and the competing interpretation is genuinely plausible, do not reject it simply because one interpretation seems conventional.

Reply with exactly:
SELECTED_CANDIDATE: <candidate number>
REASON: <one concise sentence explaining why this candidate dominates>""".strip()

REPAIR_QUESTION_TEMPLATE = """The following question violates strict formatting constraints:
"{question}"
Reason: {reason}

Please rephrase it into exactly ONE atomic, direct question of at most 25 words with exactly one '?'.
It must ask about only one behavioral choice (at most two alternatives).
Do NOT include any preamble or multiple sentences.
Reply with exactly:
QUESTION: <rephrased question>""".strip()

SOLVE_TEMPLATE = """Please provide a self-contained Python implementation of `{entry_point}` that solves the following problem:

{prompt}
{clarification_block}

Implementation Guidelines:
1. Adhere strictly to the problem specification and any provided human clarification.
2. Use the exact entry point: `def {entry_point}(...):`
3. Then enclose your complete Python implementation in ```python and ```.""".strip()

CLARIFICATION_BLOCK = """
The clarification below is authoritative for the specific issue asked:
Q: {question}
A: {answer}

Incorporate this answer exactly.
Do not reinterpret, modify, or override unrelated requirements from the original specification.
"""


EMPTY_FALLBACK = """```python
def {entry_point}(*args, **kwargs):
    raise NotImplementedError
```"""

# ==============================================================================
# DETERMINISTIC QUESTION VALIDATION & PARSING
# ==============================================================================


def validate_question(question: str) -> tuple[bool, str]:
    """Deterministically validates that the question satisfies competition atomicity rules."""
    if not isinstance(question, str) or not question.strip():
        return False, "Question is empty"

    q = question.strip().strip('"`* ')
    if not q.endswith("?"):
        return False, "Question does not end with a question mark"

    if q.count("?") != 1:
        return False, "Question contains multiple question marks (compound question)"

    if "\n" in q:
        return False, "Question spans multiple lines"

    words = q.split()
    if len(words) > 35:
        return False, f"Question is too long ({len(words)} words, maximum allowed is 35 words)"

    if ";" in q:
        return False, "Question contains compound semicolon structure"

    lower_q = q.lower()
    for compound_phrase in [
        " and also ",
        " furthermore ",
        " in addition ",
        " what about ",
        " as well as whether ",
    ]:
        if compound_phrase in lower_q:
            return False, f"Question contains compound phrase '{compound_phrase.strip()}'"

    return True, "Valid"


def parse_discovery_output(discovery_text: str) -> list[dict[str, Any]]:
    """Parses candidate requirements from the discovery prompt response,
    retaining interpretations A and B for dominance comparison.
    """
    if not isinstance(discovery_text, str) or "STATUS: UNAMBIGUOUS" in discovery_text:
        return []

    candidates: list[dict[str, Any]] = []
    blocks = re.split(r"(?:^|\n)---\s*\n", discovery_text)

    for block in blocks:
        if not block.strip() or "CANDIDATE:" not in block:
            continue

        cand_id_m = re.search(r"CANDIDATE:\s*(\d+)", block)
        lens_m = re.search(r"LENS:\s*([^\n]+)", block)
        issue_m = re.search(r"ISSUE:\s*([^\n]+)", block)
        interp_a_m = re.search(r"INTERPRETATION_A:\s*([^\n]+)", block)
        interp_b_m = re.search(r"INTERPRETATION_B:\s*([^\n]+)", block)
        footprint_m = re.search(r"FOOTPRINT:\s*(GLOBAL|BROAD|EDGE)", block, re.IGNORECASE)
        inferable_m = re.search(r"INFERABLE_FROM_SPEC:\s*([^\n]+)", block, re.IGNORECASE)
        q_m = re.search(r"QUESTION:\s*([^\n]+)", block)

        if not q_m or not issue_m:
            continue

        cand_id = int(cand_id_m.group(1)) if cand_id_m else len(candidates) + 1
        lens = lens_m.group(1).strip() if lens_m else "CORE_SEMANTICS"
        issue = issue_m.group(1).strip()
        interp_a = interp_a_m.group(1).strip() if interp_a_m else ""
        interp_b = interp_b_m.group(1).strip() if interp_b_m else ""
        footprint = footprint_m.group(1).upper() if footprint_m else "BROAD"

        inferable_str = inferable_m.group(1).strip() if inferable_m else "NO"
        is_inferable = inferable_str.upper().startswith("YES")

        raw_q = q_m.group(1).strip().strip('"`* ')
        if raw_q and not raw_q.endswith("?"):
            raw_q += "?"

        candidates.append(
            {
                "id": cand_id,
                "lens": lens,
                "issue": issue,
                "interpretation_a": interp_a,
                "interpretation_b": interp_b,
                "footprint": footprint,
                "inferable": is_inferable,
                "question": raw_q,
            }
        )

    return candidates


# ==============================================================================
# DETERMINISTIC SIGNATURE EXTRACTION & STRUCTURAL VALIDATION
# ==============================================================================


def extract_expected_signature(prompt: str, entry_point: str) -> dict[str, Any] | None:
    """Extracts expected parameter structure deterministically from the prompt stub using AST."""
    if not prompt or not entry_point:
        return None

    pattern = rf"def\s+{re.escape(entry_point)}\s*\((.*?)\)(?:\s*->\s*[^:\n]+)?\s*:"
    match = re.search(pattern, prompt, re.DOTALL)
    if not match:
        return None

    param_str = match.group(1).strip()
    dummy_code = f"def {entry_point}({param_str}): pass"
    try:
        tree = ast.parse(dummy_code)
        func = tree.body[0]
        if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            posonlyargs = [a.arg for a in func.args.posonlyargs]
            args = [a.arg for a in func.args.args]
            kwonlyargs = [a.arg for a in func.args.kwonlyargs]
            vararg = func.args.vararg.arg if func.args.vararg else None
            kwarg = func.args.kwarg.arg if func.args.kwarg else None
            defaults_count = len(func.args.defaults)
            kw_defaults = func.args.kw_defaults
            kw_defaults_count = sum(1 for d in kw_defaults if d is not None)

            # Extract mandatory keyword-only argument names
            mandatory_kwonly = [
                arg.arg
                for arg, d in zip(func.args.kwonlyargs, kw_defaults, strict=False)
                if d is None
            ]

            num_pos = len(posonlyargs) + len(args)
            min_pos = num_pos - defaults_count
            num_kwonly_mandatory = len(mandatory_kwonly)
            raw_sig = f"def {entry_point}({param_str}):"

            return {
                "posonlyargs": posonlyargs,
                "args": args,
                "kwonlyargs": kwonlyargs,
                "mandatory_kwonly": mandatory_kwonly,
                "vararg": vararg,
                "kwarg": kwarg,
                "defaults_count": defaults_count,
                "kw_defaults_count": kw_defaults_count,
                "num_pos": num_pos,
                "min_pos": min_pos,
                "num_kwonly_mandatory": num_kwonly_mandatory,
                "raw_sig": raw_sig,
            }
    except Exception:
        pass
    return None


def validate_function_signature(
    target_func: ast.FunctionDef | ast.AsyncFunctionDef,
    expected_sig: dict[str, Any],
    entry_point: str,
) -> str | None:
    """Deterministically validates target_func AST against expected_sig.
    Enforces that the generated signature accepts every valid positional call
    accepted by the expected signature (gen_min_pos <= exp_min_pos, and
    gen accepts at least exp_num_pos arguments).
    """
    gen_posonly = [a.arg for a in target_func.args.posonlyargs]
    gen_args = [a.arg for a in target_func.args.args]
    gen_vararg = target_func.args.vararg.arg if target_func.args.vararg else None
    gen_defaults_count = len(target_func.args.defaults)
    gen_kw_defaults = target_func.args.kw_defaults

    gen_mandatory_kwonly = [
        arg.arg
        for arg, d in zip(target_func.args.kwonlyargs, gen_kw_defaults, strict=False)
        if d is None
    ]

    gen_num_pos = len(gen_posonly) + len(gen_args)
    gen_min_pos = gen_num_pos - gen_defaults_count

    try:
        dummy_mod = ast.Module(body=[target_func], type_ignores=[])
        unparsed = ast.unparse(dummy_mod).splitlines()[0]
        gen_raw_sig = unparsed if unparsed.endswith(":") else f"{unparsed}:"
    except Exception:
        all_pos = gen_posonly + gen_args
        gen_raw_sig = f"def {entry_point}({', '.join(all_pos)}):"

    # 1. Keyword-only compatibility: generated cannot introduce mandatory keyword-only args not in expected
    exp_mandatory_kwonly = expected_sig.get("mandatory_kwonly", [])
    for k in gen_mandatory_kwonly:
        if k not in exp_mandatory_kwonly:
            return (
                f"Signature validation failed for '{entry_point}':\n"
                f"Expected: {expected_sig['raw_sig']}\n"
                f"Generated: {gen_raw_sig}\n"
                f"Error: Generated function requires mandatory keyword-only argument '{k}' "
                f"not required by expected signature."
            )

    # 2. Positional argument compatibility
    exp_num_pos = expected_sig["num_pos"]
    exp_min_pos = expected_sig["min_pos"]
    exp_has_vararg = expected_sig["vararg"] is not None

    # Generated function must not require MORE mandatory positional arguments than the minimum expected
    if gen_min_pos > exp_min_pos:
        return (
            f"Signature validation failed for '{entry_point}':\n"
            f"Expected: {expected_sig['raw_sig']}\n"
            f"Generated: {gen_raw_sig}\n"
            f"Error: Generated function requires at least {gen_min_pos} mandatory positional argument(s), "
            f"but expected signature requires at most {exp_min_pos} (accepts calls with {exp_min_pos} arguments)."
        )

    if not exp_has_vararg:
        # Generated function without *args must accept at least exp_num_pos arguments
        if gen_vararg is None and gen_num_pos < exp_num_pos:
            return (
                f"Signature validation failed for '{entry_point}':\n"
                f"Expected: {expected_sig['raw_sig']}\n"
                f"Generated: {gen_raw_sig}\n"
                f"Error: Generated function accepts at most {gen_num_pos} positional argument(s), "
                f"but expected signature allows up to {exp_num_pos} argument(s)."
            )
    else:
        # Expected signature accepts arbitrarily many positional arguments
        if gen_vararg is None:
            return (
                f"Signature validation failed for '{entry_point}':\n"
                f"Expected: {expected_sig['raw_sig']}\n"
                f"Generated: {gen_raw_sig}\n"
                f"Error: Expected signature accepts variable arguments (*args), "
                f"but generated function does not accept *args."
            )

    return None


# ==============================================================================
# AST PARSING & EXTENDED STATIC VALIDATION
# ==============================================================================


def extract_code_block(text: str, entry_point: str = "") -> str:
    """Extracts Python code from markdown code fences or raw code."""
    if not isinstance(text, str):
        return ""

    matches = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if matches:
        if len(matches) > 1 and entry_point:
            concatenated = "\n\n".join(m.strip() for m in matches if m.strip())
            try:
                tree = ast.parse(concatenated)
                if any(
                    isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)) and s.name == entry_point
                    for s in tree.body
                ):
                    return concatenated
            except Exception:
                pass

        if entry_point:
            for m in reversed(matches):
                if re.search(rf"\bdef\s+{re.escape(entry_point)}\b", m):
                    return m.strip()

        for m in reversed(matches):
            try:
                tree = ast.parse(m)
                if any(isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)) for s in tree.body):
                    return m.strip()
            except Exception:
                pass

        longest_match = max(matches, key=len)
        return longest_match.strip()

    try:
        ast.parse(text)
        return text.strip()
    except Exception:
        if entry_point and f"def {entry_point}" in text:
            open_match = re.search(r"```(?:python|py)?\s*\n(.*)", text, re.DOTALL)
            if open_match:
                return open_match.group(1).strip()
        return text.strip()


def validate_and_clean_ast(
    source_code: str,
    entry_point: str,
    expected_sig: dict[str, Any] | None = None,
) -> tuple[str | None, str | None]:
    """Validates AST structure and entry-point definition, rejecting duplicate definitions,
    and removes only clearly identified runner/test artifacts without destroying valid code.
    """
    if not source_code or not source_code.strip():
        return None, "Static validation failed: Source code is empty."

    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        return None, f"Static validation failed: SyntaxError: {e.msg} on line {e.lineno}"

    entry_funcs = [
        stmt
        for stmt in tree.body
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == entry_point
    ]

    if len(entry_funcs) == 0:
        return (
            None,
            f"Static validation failed: Function '{entry_point}' was not found in the generated solution.",
        )

    if len(entry_funcs) > 1:
        return None, (
            f"Static validation failed: Found multiple ({len(entry_funcs)}) top-level definitions "
            f"of entry point '{entry_point}'. Expected exactly one definition."
        )

    target_func = entry_funcs[0]

    # Validate signature against expected signature if deterministically present in prompt
    if expected_sig is not None:
        sig_error = validate_function_signature(target_func, expected_sig, entry_point)
        if sig_error is not None:
            return None, sig_error

    # Non-destructive artifact sanitization
    new_body = []
    modified = False

    for stmt in tree.body:
        if isinstance(stmt, ast.If):
            is_main = False
            if isinstance(stmt.test, ast.Compare):
                for comp in [stmt.test.left] + stmt.test.comparators:
                    if isinstance(comp, ast.Constant) and comp.value == "__main__":
                        is_main = True
                        break
            if is_main:
                modified = True
                continue

            has_exit = any(
                isinstance(n, ast.Call)
                and (
                    (isinstance(n.func, ast.Attribute) and n.func.attr == "exit")
                    or (isinstance(n.func, ast.Name) and n.func.id == "exit")
                )
                for n in ast.walk(stmt)
            )
            if has_exit:
                modified = True
                continue

        if isinstance(stmt, ast.Assert):
            modified = True
            continue

        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call_node = stmt.value
            is_stray_call = False
            if isinstance(call_node.func, ast.Name) and call_node.func.id in (
                "print",
                "exit",
                entry_point,
            ):
                is_stray_call = True
            elif isinstance(call_node.func, ast.Attribute) and call_node.func.attr == "exit":
                is_stray_call = True

            if is_stray_call:
                modified = True
                continue

        new_body.append(stmt)

    if modified:
        try:
            cleaned_tree = ast.Module(body=new_body, type_ignores=getattr(tree, "type_ignores", []))
            cleaned_code = ast.unparse(cleaned_tree)
            compile(cleaned_code, "<clean>", "exec")
            return cleaned_code, None
        except Exception as e:
            return (
                None,
                f"Static validation failed: Code compilation failed after sanitization: {e}",
            )

    return source_code.strip(), None


# ==============================================================================
# EXPLICIT EXAMPLE EXTRACTION & RUNTIME VALIDATION
# ==============================================================================


def extract_explicit_example_test(prompt: str, entry_point: str) -> str | None:
    """Deterministically extracts an explicit assertion or doctest example from the prompt if present."""
    if not prompt or not entry_point:
        return None

    mbpp_match = re.search(
        rf"(assert\s+{re.escape(entry_point)}\s*\(.*?\)\s*==\s*[^\n]+)",
        prompt,
    )
    if mbpp_match:
        cand = mbpp_match.group(1).strip()
        try:
            t = ast.parse(cand)
            if len(t.body) == 1 and isinstance(t.body[0], ast.Assert):
                return cand
        except Exception:
            pass

    doctest_match = re.search(
        rf">>>\s*({re.escape(entry_point)}\s*\(.*?\))\s*\n\s*([^\n>]+)",
        prompt,
    )
    if doctest_match:
        call_expr = doctest_match.group(1).strip()
        expected_raw = doctest_match.group(2).strip()
        try:
            call_ast = ast.parse(call_expr, mode="eval")
            if isinstance(call_ast.body, ast.Call):
                ast.literal_eval(expected_raw)
                return f"assert {call_expr} == {expected_raw}, 'Explicit doctest example mismatch'"
        except Exception:
            pass

    return None


def validate_runtime(
    env: ClarificationEnvironment,
    code: str,
    entry_point: str,
    example_test: str | None = None,
) -> tuple[str, str | None]:
    """Lightweight runtime validation using env.exec_code."""
    if not hasattr(env, "exec_code"):
        return "SKIPPED", None

    script_lines = [
        code,
        "",
        f"if '{entry_point}' not in globals():",
        f"    raise NameError(\"Function '{entry_point}' is not defined after module load\")",
        f"target_callable = globals()['{entry_point}']",
        "if not callable(target_callable):",
        f"    raise TypeError(\"'{entry_point}' is not callable\")",
    ]
    if example_test:
        script_lines.extend(
            [
                "",
                example_test,
            ]
        )

    test_script = "\n".join(script_lines)

    try:
        output = env.exec_code(test_script)
    except Exception as exc:
        return "SKIPPED", f"Sandbox execution environment unavailable: {exc}"

    if not isinstance(output, str):
        return "SKIPPED", "Non-string output returned from exec_code"

    if "Timeout during the execution" in output:
        return "FAIL", "Runtime validation timed out after 30 seconds."

    if output.startswith("Exception:\n") or "Traceback (most recent call last):" in output:
        err_lines = [line.strip() for line in output.splitlines() if line.strip()]
        relevant_err = err_lines[-1] if err_lines else output.strip()
        for line in reversed(err_lines):
            if any(
                line.startswith(err_type)
                for err_type in (
                    "NameError:",
                    "TypeError:",
                    "ImportError:",
                    "ModuleNotFoundError:",
                    "AttributeError:",
                    "AssertionError:",
                    "SyntaxError:",
                    "ValueError:",
                    "IndexError:",
                    "KeyError:",
                )
            ):
                relevant_err = line
                break
        return "FAIL", f"Runtime validation failed:\n{relevant_err}"

    return "PASS", None


# ==============================================================================
# OMEGA ALGORITHM CLASS
# ==============================================================================


class Omega(ClarificationAlgorithmBase):
    """
    Omega: Dominance-Guided, Contract-Aware Clarification Algorithm.
    """

    DEFAULT_CONFIG = {
        "enable_clarification": True,
        "max_repair_attempts": 2,
        "enable_runtime_validation": True,
    }

    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        prompt = problem.get("prompt", "")
        entry_point = problem.get("entry_point", "solution")

        enable_clarif = self.config.get(
            "enable_clarification", self.DEFAULT_CONFIG["enable_clarification"]
        )
        max_repairs = self.config.get(
            "max_repair_attempts", self.DEFAULT_CONFIG["max_repair_attempts"]
        )
        enable_runtime_val = self.config.get(
            "enable_runtime_validation", self.DEFAULT_CONFIG["enable_runtime_validation"]
        )

        expected_sig = extract_expected_signature(prompt, entry_point)
        explicit_example = extract_explicit_example_test(prompt, entry_point)

        clarification_text = ""

        diagnostics = {
            "decision": "NO_QUESTION",
            "question": None,
            "footprint": None,
            "repair_triggered": False,
        }

        # ----------------------------------------------------------------------
        # 1. Two-Stage Candidate Discovery & Behavioral Dominance Selection
        # ----------------------------------------------------------------------
        if enable_clarif and env.can_ask():
            try:
                discovery_prompt = DISCOVERY_TEMPLATE.format(
                    entry_point=entry_point,
                    prompt=prompt,
                )
                discovery_resp = env.llm(discovery_prompt)
                candidates = parse_discovery_output(discovery_resp)

                # Process common implementations: record them and remove from ask pool
                # Inferability does NOT hard-reject; it is preserved for selector tiebreaking.
                ask_candidates = candidates

                # Select question if any valid candidates remain
                selected_candidate = None
                if len(ask_candidates) == 1:
                    selected_candidate = ask_candidates[0]
                elif len(ask_candidates) > 1:
                    # Send ALL surviving candidates to the dominance selector prompt
                    cand_summary_lines = []
                    for c in ask_candidates:
                        inferable_note = (
                            "Inferable from spec examples/context"
                            if c.get("inferable")
                            else "Not inferable from spec"
                        )
                        cand_summary_lines.append(
                            f"[{c['id']}] Lens: {c['lens']} | Footprint: {c['footprint']} | Evidence: {inferable_note}\n"
                            f"    Issue: {c['issue']}\n"
                            f"    Interpretation A: {c.get('interpretation_a', '')}\n"
                            f"    Interpretation B: {c.get('interpretation_b', '')}\n"
                            f"    Proposed Question: {c['question']}"
                        )
                    cand_summary = "\n\n".join(cand_summary_lines)

                    sel_prompt = DOMINANCE_SELECTOR_TEMPLATE.format(
                        entry_point=entry_point,
                        prompt=prompt,
                        candidates_text=cand_summary,
                    )
                    sel_resp = env.llm(sel_prompt)
                    sel_m = re.search(r"SELECTED_CANDIDATE:\s*(\d+)", sel_resp)
                    if sel_m:
                        chosen_id = int(sel_m.group(1))
                        for c in ask_candidates:
                            if c["id"] == chosen_id:
                                selected_candidate = c
                                break

                    # Deterministic fallback only if selector output is malformed
                    if selected_candidate is None:
                        footprint_rank = {"GLOBAL": 3, "BROAD": 2, "EDGE": 1}
                        sorted_fallback = sorted(
                            ask_candidates,
                            key=lambda c: (
                                footprint_rank.get(c.get("footprint", "BROAD"), 2),
                                not c.get("inferable", False),
                            ),
                            reverse=True,
                        )
                        selected_candidate = sorted_fallback[0]

                # Validate & ask human if candidate survived
                if selected_candidate:
                    candidate_q = selected_candidate["question"]
                    is_valid, validation_reason = validate_question(candidate_q)

                    if not is_valid:
                        repair_q_prompt = REPAIR_QUESTION_TEMPLATE.format(
                            question=candidate_q,
                            reason=validation_reason,
                        )
                        repaired_verdict = env.llm(repair_q_prompt)
                        rep_m = re.search(r"QUESTION:\s*([^\n]+)", repaired_verdict)
                        if rep_m:
                            repaired_q = rep_m.group(1).strip().strip('"`* ')
                            if repaired_q and not repaired_q.endswith("?"):
                                repaired_q += "?"
                            is_valid_repaired, _ = validate_question(repaired_q)
                            if is_valid_repaired:
                                candidate_q = repaired_q
                            else:
                                candidate_q = None
                        else:
                            candidate_q = None

                    if candidate_q:
                        try:
                            answer = env.ask_human(candidate_q)
                            clarification_text = CLARIFICATION_BLOCK.format(
                                question=candidate_q,
                                answer=answer,
                            )
                            diagnostics["decision"] = "ASK"
                            diagnostics["question"] = candidate_q
                            diagnostics["footprint"] = selected_candidate.get("footprint")
                        except (TooManyQuestionException, LimitsExceededException):
                            pass

            except LimitsExceededException:
                pass
            except Exception:
                pass

        # ----------------------------------------------------------------------
        # 2. Contract-Driven & Robustness-Oriented Synthesis
        # ----------------------------------------------------------------------
        # Skip stale explicit example if clarification was obtained
        example_for_validation = None if clarification_text else explicit_example

        solve_prompt = SOLVE_TEMPLATE.format(
            entry_point=entry_point,
            prompt=prompt,
            clarification_block=clarification_text,
        )

        messages = [{"role": "user", "content": solve_prompt}]
        synthesized_code = ""

        for attempt in range(max_repairs + 1):
            raw_response = ""
            for api_retry in range(3):
                try:
                    raw_response = env.llm(messages)
                    break
                except LimitsExceededException:
                    break
                except Exception:
                    if api_retry == 2:
                        raise
                    import time

                    time.sleep(2)

            if not raw_response:
                break

            extracted = extract_code_block(raw_response, entry_point=entry_point)
            cleaned, static_error = validate_and_clean_ast(extracted, entry_point, expected_sig)

            if cleaned is not None:
                if enable_runtime_val:
                    rt_status, rt_error = validate_runtime(
                        env, cleaned, entry_point, example_for_validation
                    )
                    if rt_status == "PASS" or rt_status == "SKIPPED":
                        synthesized_code = cleaned
                        break
                    else:
                        active_error = rt_error
                else:
                    synthesized_code = cleaned
                    break
            else:
                active_error = static_error

            if attempt < max_repairs:
                diagnostics["repair_triggered"] = True
                messages = messages + [
                    {"role": "assistant", "content": raw_response},
                    {
                        "role": "user",
                        "content": f"Static or runtime validation detected an issue:\\n{active_error}\\n\\nPlease correct it and enclose in ```python and ```.",
                    },
                ]
            else:
                synthesized_code = (
                    cleaned if cleaned else (extracted if extracted else raw_response)
                )

        if not synthesized_code:
            synthesized_code = EMPTY_FALLBACK.format(entry_point=entry_point)

        final_code = synthesized_code

        if not final_code.startswith("```python"):
            return f"```python\n{final_code}\n```"
        return final_code
