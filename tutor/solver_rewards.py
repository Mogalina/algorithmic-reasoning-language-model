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
    """
    if not text:
        return ""

    original_text = text
    pattern = r"```(?:python|py)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, text)
    if match:
        text = match.group(1).strip()
    else:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]
            text = re.sub(r"^(?:python|py)\s*", "", text, flags=re.IGNORECASE)
            text = text.strip()
        else:
            text = text.replace("```python", "").replace("```py", "").replace("```", "").strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:python|py)?\s*", "", text, flags=re.IGNORECASE)
    if text.endswith("```"):
        text = text[:-3]
        
    return text.strip()


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
        if not code:
            return False, "Empty code."
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


# ──────────────────────── reward function ─────────────────────────────────

def compute_solver_reward(code: str) -> dict:
    """
    Compute the RLHF reward for a solver-generated Python solution.
    """
    if not code or not code.strip():
        return {"reward": -1, "compiled": False, "error": "Empty code output."}

    original_code = code
    code = extract_code(code)

    compiled, error = _compiles(code)
    if not compiled:
        print(f"[SolverReward] Compilation failed for code:\n{code[:100]}...")

    return {
        "reward": +1 if compiled else -1,
        "compiled": compiled,
        "error": error,
    }
