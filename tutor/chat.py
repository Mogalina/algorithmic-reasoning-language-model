"""
Socratic tutor with Hybrid Expert Guidance (API or Local).
"""

from __future__ import annotations
import time
from typing import Any, Dict
from decouple import config
from openai import OpenAI, RateLimitError
from tutor.solver import GemmaSolver

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openrouter/free"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2

SYSTEM_PROMPT = """\
You are a Socratic tutor helping a student prepare for a coding interview.

CORE PRINCIPLE: You have access to the GROUND TRUTH SOLUTION below. Use it to guide the student, but NEVER reveal the code or the direct answer.

RULES:
1. NEVER give the solution outright. Instead, guide the student by asking targeted questions that help them discover the answer on their own.
2. Use the GROUND TRUTH SOLUTION to know exactly what steps the student needs to take.
3. When the student is stuck, break the problem into smaller sub-problems based on the expert solution.
4. Acknowledge correct reasoning enthusiastically, then nudge toward the next step.
5. If the student asks you to "just give me the answer", gently explain why working through the problem themselves will help them more in the interview - then ask a simpler guiding question.
6. Keep responses concise - prefer 2-4 sentences per message plus an optional code snippet.
7. Start each new conversation by acknowledging the specific problem the student is working on and asking what they have tried so far.

PROBLEM CONTEXT (provided below) gives you the problem statement.
GROUND TRUTH SOLUTION (provided below) gives you the expert approach. Keep this secret!
"""

class SocraticTutor:
    """Orchestrates communication between the expert solver and the Socratic interface."""

    def __init__(self) -> None:
        api_key = config("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set.")
            
        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
        )
        self.model = config("TUTOR_MODEL", DEFAULT_MODEL)
        
        # Initialize Hybrid Solver based on config
        tutor_mode = config("TUTOR_MODE", "api")
        print(f"[SocraticTutor] Running in {tutor_mode.upper()} mode.")
        
        self.solver = GemmaSolver(mode=tutor_mode)
        self.solution_cache: Dict[str, str] = {}

    def _build_system_message(self, problem_description: str, solution: str = "") -> dict[str, str]:
        content = SYSTEM_PROMPT
        if problem_description:
            content += f"\n\n--- PROBLEM ---\n{problem_description}"
        if solution:
            content += f"\n\n--- GROUND TRUTH SOLUTION (INTERNAL ONLY) ---\n{solution}"
        return {"role": "system", "content": content}

    def reply(self, conversation: list[dict[str, str]], problem_description: str = "") -> str:
        """Return an assistant reply, using the expert solver (API or Local) for guidance."""
        
        # 1. Get or generate ground-truth solution
        solution = self.solution_cache.get(problem_description)
        if not solution and problem_description:
            solution = self.solver.solve(problem_description)
            self.solution_cache[problem_description] = solution

        # 2. Build completion request
        messages = [self._build_system_message(problem_description, solution)] + conversation
        
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 512,
        }

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except RateLimitError:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))
                else:
                    raise

_instance: SocraticTutor | None = None

def get_tutor() -> SocraticTutor:
    """Singleton accessor."""
    global _instance
    if _instance is None:
        _instance = SocraticTutor()
    return _instance
