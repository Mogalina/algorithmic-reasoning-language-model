"""
Evaluate the Socratic tutor and meta-prompting loop.

Tests:
  1. (OFFLINE) Prompt loading & rendering: all templates load, render, and
     round-trip through save without corruption.
  2. (OFFLINE) Meta-prompting invariants: the optimizer prompt template
     includes both current_prompt and failure_log placeholders.
  3. (ONLINE) Socratic quality on synthetic conversations: the tutor should
     guide without revealing solutions, scored by the GEval evaluator.
  4. (ONLINE) Meta-prompting self-improvement: after a deliberately poor
     prompt, the optimizer should produce a measurably different one.

Tests 1-2 are OFFLINE (no API calls).
Tests 3-4 require OPENROUTER_API_KEY in .env.

Usage:
    python scripts/eval_tutor.py              # tests 1-2 (offline)
    python scripts/eval_tutor.py --online     # tests 1-4 (needs API keys)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tutor.prompts import load_prompt, render_prompt, save_prompt

SEPARATOR = "=" * 70

# All registered template names (from tutor/prompts/__init__.py)
REQUIRED_TEMPLATES = [
    "socratic_system",
    "problem_context",
    "ground_truth",
    "solver_system",
    "solver_user",
    "solver_local",
    "eval_socratic_quality",
    "eval_code_correctness",
    "optimizer_meta",
    "failure_analysis",
    "few_shot_preamble",
]

SYNTHETIC_CONVERSATIONS = [
    {
        "problem": (
            "Given an array of integers nums and an integer target, return "
            "indices of the two numbers such that they add up to target. You "
            "may assume that each input would have exactly one solution."
        ),
        "solution": (
            "def twoSum(nums, target):\n"
            "    seen = {}\n"
            "    for i, n in enumerate(nums):\n"
            "        if target - n in seen:\n"
            "            return [seen[target - n], i]\n"
            "        seen[n] = i"
        ),
        "conversation": [
            {"role": "user", "content": "I have no idea where to start with this problem."},
        ],
        "label": "Two Sum — cold start",
    },
    {
        "problem": (
            "Given the head of a singly linked list, reverse the list, "
            "and return the reversed list."
        ),
        "solution": (
            "def reverseList(head):\n"
            "    prev = None\n"
            "    curr = head\n"
            "    while curr:\n"
            "        nxt = curr.next\n"
            "        curr.next = prev\n"
            "        prev = curr\n"
            "        curr = nxt\n"
            "    return prev"
        ),
        "conversation": [
            {"role": "user", "content": "I think I need to change the pointers somehow?"},
        ],
        "label": "Reverse Linked List — partial understanding",
    },
    {
        "problem": (
            "Given a string s containing just the characters '(', ')', '{', "
            "'}', '[' and ']', determine if the input string is valid."
        ),
        "solution": (
            "def isValid(s):\n"
            "    stack = []\n"
            "    mapping = {')': '(', '}': '{', ']': '['}\n"
            "    for c in s:\n"
            "        if c in mapping:\n"
            "            top = stack.pop() if stack else '#'\n"
            "            if mapping[c] != top: return False\n"
            "        else:\n"
            "            stack.append(c)\n"
            "    return not stack"
        ),
        "conversation": [
            {"role": "user", "content": "I was thinking of using a stack. Is that the right approach?"},
        ],
        "label": "Valid Parentheses — correct intuition",
    },
]


def eval_prompt_loading():
    """All registered templates should load without errors."""
    print(f"\n{SEPARATOR}")
    print("TEST 1: Prompt Loading & Rendering")
    print(SEPARATOR)

    passed = 0
    for name in REQUIRED_TEMPLATES:
        try:
            text = load_prompt(name)
            ok = len(text.strip()) > 0
        except Exception as e:
            ok = False
            text = f"ERROR: {e}"

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        length = len(text) if ok else 0
        print(f"  [{status}] {name:30s}  ({length} chars)")

    print(f"  Result: {passed}/{len(REQUIRED_TEMPLATES)} templates loaded")

    # Round-trip test: save and reload the socratic_system prompt
    print("\n  Round-trip save/reload test:")
    original = load_prompt("socratic_system")
    save_prompt("socratic_system", original)
    reloaded = load_prompt("socratic_system")
    match = original == reloaded
    print(f"    Save + reload identical: {match}  [{'PASS' if match else 'FAIL'}]")


def eval_meta_prompt_structure():
    """The optimizer meta-prompt should accept current_prompt and failure_log."""
    print(f"\n{SEPARATOR}")
    print("TEST 2: Meta-Prompting Template Structure")
    print(SEPARATOR)

    template = load_prompt("optimizer_meta")
    has_current = "{current_prompt}" in template or "current_prompt" in template
    has_failure = "{failure_log}" in template or "failure_log" in template
    print(f"  Template length:          {len(template)} chars")
    print(f"  Contains current_prompt:  {has_current}  [{'PASS' if has_current else 'FAIL'}]")
    print(f"  Contains failure_log:     {has_failure}  [{'PASS' if has_failure else 'FAIL'}]")

    rendered = render_prompt(
        "optimizer_meta",
        current_prompt="You are a test tutor.",
        failure_log="Score 0.2: response revealed the solution.",
    )
    rendered_ok = "test tutor" in rendered and "Score 0.2" in rendered
    print(f"  Renders with placeholders: {rendered_ok}  [{'PASS' if rendered_ok else 'FAIL'}]")

    # Check failure_analysis template too
    failure_tmpl = render_prompt(
        "failure_analysis",
        score=0.3,
        reason="Gave away the solution",
        response_text="Here is the code: def f(): pass",
    )
    failure_ok = "0.3" in failure_tmpl and "Gave away" in failure_tmpl
    print(f"  failure_analysis renders:  {failure_ok}  [{'PASS' if failure_ok else 'FAIL'}]")


def eval_socratic_quality():
    """Score tutor responses on synthetic conversations using the GEval evaluator."""
    print(f"\n{SEPARATOR}")
    print("TEST 3: Socratic Quality (ONLINE — GEval via OpenRouter)")
    print(SEPARATOR)

    try:
        from tutor.chat import SocraticTutor
        from tutor.evaluator import evaluator
    except Exception as e:
        print(f"  SKIP: Could not import tutor components: {e}")
        return

    tutor = SocraticTutor()
    scores = []

    for case in SYNTHETIC_CONVERSATIONS:
        print(f"\n  Scenario: {case['label']}")

        # Override the solution cache so we don't call the solver
        tutor.solution_cache[case["problem"]] = case["solution"]

        response = tutor.reply(case["conversation"], problem_description=case["problem"])
        print(f"    Tutor response: {response[:120]}...")

        score, reason = evaluator.evaluate(
            case["conversation"][-1]["content"],
            response,
            case["solution"],
        )
        scores.append(score)
        print(f"    Score: {score:.2f}  Reason: {reason[:100]}")

        # Check non-disclosure: response should not contain the solution code
        solution_lines = [l.strip() for l in case["solution"].split("\n") if l.strip()]
        leaked = any(line in response for line in solution_lines if len(line) > 15)
        print(f"    Non-disclosure: {'FAIL (leaked code)' if leaked else 'PASS'}")

    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"\n  Average Socratic score: {avg_score:.2f}")
    print(f"  {'PASS' if avg_score >= 0.6 else 'WARN'}: "
          f"{'Good' if avg_score >= 0.6 else 'Low'} average Socratic quality")


def eval_meta_prompting_improvement():
    """The optimizer should produce a different (improved) prompt from a failure log."""
    print(f"\n{SEPARATOR}")
    print("TEST 4: Meta-Prompting Self-Improvement (ONLINE)")
    print(SEPARATOR)

    try:
        from tutor.optimizer import optimizer
    except Exception as e:
        print(f"  SKIP: Could not import optimizer: {e}")
        return

    current_prompt = load_prompt("socratic_system")
    failure_log = (
        "Score: 0.2\n"
        "Reason: The tutor revealed the complete solution code to the student "
        "instead of guiding them with hints. The response contained a working "
        "implementation of twoSum that the student could copy-paste directly.\n"
        "Response: Here you go: def twoSum(nums, target): seen = {}; "
        "for i, n in enumerate(nums): if target - n in seen: "
        "return [seen[target-n], i]; seen[n] = i"
    )

    print("  Sending failure log to optimizer...")
    new_prompt = optimizer.optimize(current_prompt, failure_log)

    changed = new_prompt != current_prompt
    longer = len(new_prompt) > 0
    print(f"  Original prompt length:  {len(current_prompt)} chars")
    print(f"  Optimized prompt length: {len(new_prompt)} chars")
    print(f"  Prompt changed:          {changed}  [{'PASS' if changed else 'FAIL'}]")
    print(f"  Non-empty output:        {longer}  [{'PASS' if longer else 'FAIL'}]")

    if changed:
        # Show a short diff preview
        print(f"  Preview (first 200 chars of new prompt):")
        print(f"    {new_prompt[:200]}...")


def main():
    print(SEPARATOR)
    print("EVALUATION: Socratic Tutor & Meta-Prompting")
    print(SEPARATOR)

    eval_prompt_loading()
    eval_meta_prompt_structure()

    if "--online" in sys.argv:
        eval_socratic_quality()
        eval_meta_prompting_improvement()
    else:
        print(f"\n{SEPARATOR}")
        print("TESTS 3-4: SKIPPED (run with --online to enable)")
        print(SEPARATOR)

    print(f"\n{SEPARATOR}")
    print("TUTOR EVALUATION COMPLETE")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
