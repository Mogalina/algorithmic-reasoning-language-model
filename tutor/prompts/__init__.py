"""
Prompt loader for the tutor module.

All LLM prompts live as .txt files in this directory.  Static prompts are
loaded once at import time; templates with {placeholders} are rendered on
demand via ``render_prompt(name, **kwargs)``.
"""

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent

TEMPLATES: dict[str, str] = {
    "socratic_system":       "socratic_system.txt",
    "problem_context":       "problem_context.txt",
    "ground_truth":          "ground_truth.txt",
    "optimizer_meta":        "optimizer_meta.txt",
    "failure_analysis":      "failure_analysis.txt",
    "solver_system":         "solver_system.txt",
    "solver_user":           "solver_user.txt",
    "solver_local":          "solver_local.txt",
    "few_shot_preamble":     "few_shot_preamble.txt",
    "eval_socratic_quality": "eval_socratic_quality.txt",
    "eval_code_correctness": "eval_code_correctness.txt",
}

_cache: dict[str, str] = {}


def _load_raw(name: str) -> str:
    """Read a prompt file from disk (cached after first load)."""
    if name in _cache:
        return _cache[name]

    filename = TEMPLATES.get(name)
    if filename is None:
        raise KeyError(f"Unknown prompt: {name!r}. Available: {list(TEMPLATES)}")

    path = _PROMPTS_DIR / filename
    text = path.read_text(encoding="utf-8")
    _cache[name] = text
    return text


def load_prompt(name: str) -> str:
    """Return the raw prompt text (no variable substitution)."""
    return _load_raw(name)


def render_prompt(name: str, **kwargs) -> str:
    """Load a prompt template and fill ``{placeholders}`` with *kwargs*."""
    template = _load_raw(name)
    return template.format(**kwargs)


def save_prompt(name: str, content: str) -> None:
    """Overwrite a prompt file on disk (used by the optimizer)."""
    filename = TEMPLATES.get(name)
    if filename is None:
        raise KeyError(f"Unknown prompt: {name!r}")

    path = _PROMPTS_DIR / filename
    path.write_text(content, encoding="utf-8")
    _cache[name] = content
