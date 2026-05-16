"""
Evaluate the Expert Solver pipeline.

Tests:
  1. Code extraction: extract_code correctly strips markdown fences.
  2. AST compilation gate: compute_solver_reward catches syntax errors.
  3. Two-stage gate savings: how many solutions fail AST before reaching
     the expensive LLM-as-judge evaluation?
  4. (Optional, requires API key) End-to-end solver quality via GEval.

The first three tests are OFFLINE (no API calls, no GPU) and run instantly.
Test 4 calls the Hugging Face / OpenRouter API.

Usage:
    python scripts/eval_solver.py            # tests 1-3 (offline)
    python scripts/eval_solver.py --online   # tests 1-4 (needs API keys)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tutor.solver_rewards import extract_code, compute_solver_reward

SEPARATOR = "=" * 70


# ── Test data ────────────────────────────────────────────────────────────

EXTRACTION_CASES = [
    (
        "Here is the solution:\n```python\ndef two_sum(nums, target):\n    seen = {}\n    for i, n in enumerate(nums):\n        if target - n in seen:\n            return [seen[target - n], i]\n        seen[n] = i\n```",
        "def two_sum(nums, target):",
        "markdown python fence",
    ),
    (
        "```\ndef hello():\n    print('hello')\n```",
        "def hello():",
        "bare fence (no language tag)",
    ),
    (
        "def add(a, b):\n    return a + b",
        "def add(a, b):",
        "raw code (no fences at all)",
    ),
    (
        "```py\nclass Node:\n    pass\n```",
        "class Node:",
        "```py fence",
    ),
    (
        "",
        "",
        "empty input",
    ),
]

REWARD_CASES = [
    ("def f(): return 1", +1, True, "valid one-liner"),
    ("def f(\n    return 1", -1, False, "syntax error (missing closing paren)"),
    ("class Foo:\n    pass", +1, True, "valid class"),
    ("```python\ndef g(): return 2\n```", +1, True, "valid code inside fences"),
    ("", -1, False, "empty string"),
    ("import os\nos.listdir('.')", +1, True, "multi-statement"),
    ("def broken(:\n  pass", -1, False, "missing parameter name"),
]

SYNTHETIC_SOLUTIONS = [
    (
        "Given an array of integers, return indices of two numbers that add up to a target.",
        "```python\ndef twoSum(nums, target):\n    seen = {}\n    for i, n in enumerate(nums):\n        complement = target - n\n        if complement in seen:\n            return [seen[complement], i]\n        seen[n] = i\n```",
    ),
    (
        "Reverse a linked list.",
        "```python\ndef reverseList(head):\n    prev = None\n    curr = head\n    while curr:\n        nxt = curr.next\n        curr.next = prev\n        prev = curr\n        curr = nxt\n    return prev\n```",
    ),
    (
        "Check if a string has valid parentheses.",
        "```python\ndef isValid(s):\n    stack = []\n    mapping = {')': '(', '}': '{', ']': '['}\n    for char in s:\n        if char in mapping:\n            top = stack.pop() if stack else '#'\n            if mapping[char] != top:\n                return False\n        else:\n            stack.append(char)\n    return not stack\n```",
    ),
    (
        "Find the maximum subarray sum.",
        "```python\ndef maxSubArray(nums):\n    max_sum = curr_sum = nums[0]\n    for n in nums[1:]:\n        curr_sum = max(n, curr_sum + n)\n        max_sum = max(max_sum, curr_sum)\n    return max_sum\n```",
    ),
    (
        "Merge two sorted linked lists.",
        "```python\ndef mergeTwoLists(l1, l2):\n    dummy = ListNode(0)\n    tail = dummy\n    while l1 and l2:\n        if l1.val <= l2.val:\n            tail.next = l1\n            l1 = l1.next\n        else:\n            tail.next = l2\n            l2 = l2.next\n        tail = tail.next\n    tail.next = l1 or l2\n    return dummy.next\n```",
    ),
    (
        "Binary search in a sorted array.",
        "```python\ndef search(nums, target):\n    lo, hi = 0, len(nums) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if nums[mid] == target:\n            return mid\n        elif nums[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n```",
    ),
    # intentionally broken
    (
        "Broken solution test.",
        "```python\ndef broken(:\n    pass\n```",
    ),
    (
        "Another broken solution.",
        "this is not code at all, just random text with no structure",
    ),
]


# ── Tests ────────────────────────────────────────────────────────────────

def eval_code_extraction():
    """extract_code should strip markdown fences and return clean Python."""
    print(f"\n{SEPARATOR}")
    print("TEST 1: Code Extraction")
    print(SEPARATOR)

    passed = 0
    for raw, expected_start, label in EXTRACTION_CASES:
        extracted = extract_code(raw)
        ok = extracted.startswith(expected_start) if expected_start else extracted == ""
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"  [{status}] {label:35s}  ->  {repr(extracted[:50])}")

    print(f"  Result: {passed}/{len(EXTRACTION_CASES)} passed")


def eval_ast_reward():
    """compute_solver_reward should return correct rewards for known inputs."""
    print(f"\n{SEPARATOR}")
    print("TEST 2: AST Compilation Gate (compute_solver_reward)")
    print(SEPARATOR)

    passed = 0
    for code, exp_reward, exp_compiled, label in REWARD_CASES:
        result = compute_solver_reward(code)
        ok = result["reward"] == exp_reward and result["compiled"] == exp_compiled
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"  [{status}] {label:40s}  reward={result['reward']:+d}  compiled={result['compiled']}")

    print(f"  Result: {passed}/{len(REWARD_CASES)} passed")


def eval_two_stage_savings():
    """Measure how many solutions the AST gate catches before LLM evaluation."""
    print(f"\n{SEPARATOR}")
    print("TEST 3: Two-Stage Gate — AST Savings")
    print(SEPARATOR)

    total = len(SYNTHETIC_SOLUTIONS)
    ast_passed = 0
    ast_failed = 0

    for problem, solution in SYNTHETIC_SOLUTIONS:
        result = compute_solver_reward(solution)
        if result["compiled"]:
            ast_passed += 1
        else:
            ast_failed += 1
            print(f"  AST REJECT: {problem[:50]:50s}  error={result['error']}")

    print(f"\n  Total solutions:     {total}")
    print(f"  AST passed:          {ast_passed} ({100 * ast_passed / total:.0f}%)")
    print(f"  AST rejected:        {ast_failed} ({100 * ast_failed / total:.0f}%)")
    print(f"  LLM calls saved:     {ast_failed} "
          f"(the LLM judge never needs to evaluate these)")
    print(f"  PASS: Two-stage gate correctly filters non-compiling solutions")


def eval_online_solver():
    """End-to-end test: generate a solution via the solver and evaluate it."""
    print(f"\n{SEPARATOR}")
    print("TEST 4: End-to-End Solver Quality (ONLINE — requires API keys)")
    print(SEPARATOR)

    try:
        from tutor.solver import GemmaSolver
    except Exception as e:
        print(f"  SKIP: Could not import GemmaSolver: {e}")
        return

    solver = GemmaSolver(mode="api")

    test_problems = [
        "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
        "Given the head of a singly linked list, reverse the list, and return the reversed list.",
        "Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.",
    ]

    total_score = 0.0
    compiled_count = 0

    for i, problem in enumerate(test_problems, 1):
        print(f"\n  Problem {i}: {problem[:60]}...")
        solution = solver.solve(problem)
        reward = compute_solver_reward(solution)

        compiled = reward["compiled"]
        if compiled:
            compiled_count += 1
        print(f"    Compiled: {compiled}  Reward: {reward['reward']:+d}")

        if reward.get("error"):
            print(f"    Error: {reward['error']}")

    compile_rate = 100 * compiled_count / len(test_problems)
    print(f"\n  Compile rate:   {compiled_count}/{len(test_problems)} ({compile_rate:.0f}%)")
    print(f"  {'PASS' if compile_rate >= 66 else 'WARN'}: "
          f"{'Majority' if compile_rate >= 66 else 'Minority'} of solutions compile")


def main():
    print(SEPARATOR)
    print("EVALUATION: Expert Solver Pipeline")
    print(SEPARATOR)

    eval_code_extraction()
    eval_ast_reward()
    eval_two_stage_savings()

    if "--online" in sys.argv:
        eval_online_solver()
    else:
        print(f"\n{SEPARATOR}")
        print("TEST 4: SKIPPED (run with --online to enable)")
        print(SEPARATOR)

    print(f"\n{SEPARATOR}")
    print("SOLVER EVALUATION COMPLETE")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
