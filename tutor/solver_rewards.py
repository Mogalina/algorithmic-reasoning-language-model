"""
RLHF Reward signal for the Solver model (code-generation).

The solver is a fine-tuned model that produces ground-truth Python solutions
to coding problems.  The reward is binary:

    +1  — code compiles (parses without syntax errors)
    -1  — code does NOT compile (syntax error)
"""

from __future__ import annotations

import ast
import re


# ---------------------------------------------------------------------------
# Extractor — pull code from LLM output
# ---------------------------------------------------------------------------

def extract_code(text: str) -> str:
    """
    Extract Python code from LLM output.

    If the text contains a fenced code block (```python ... ``` or ``` ... ```),
    the content of the first block is returned.  Otherwise the full text is
    returned as-is.
    """
    pattern = r"```(?:python|py)?\s*\n([\s\S]*?)```"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return text


# ---------------------------------------------------------------------------
# Compilation check
# ---------------------------------------------------------------------------

def _compiles(code: str) -> tuple[bool, str | None]:
    """
    Check whether *code* is syntactically valid Python.

    Returns ``(True, None)`` on success or ``(False, error_message)`` on
    failure.
    """
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


# ──────────────────────── reward function ─────────────────────────────────

def compute_solver_reward(code: str) -> dict:
    """
    Compute the RLHF reward for a solver-generated Python solution.

    Args:
        code: The generated code (raw string or LLM output with fences).

    Returns::

        {
            "reward": +1 | -1,
            "compiled": bool,
            "error": str | None,
        }
    """
    if not code or not code.strip():
        return {"reward": -1, "compiled": False, "error": "Empty code output."}

    code = extract_code(code)

    compiled, error = _compiles(code)

    return {
        "reward": +1 if compiled else -1,
        "compiled": compiled,
        "error": error,
    }
