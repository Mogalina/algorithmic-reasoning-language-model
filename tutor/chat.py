"""
Socratic tutor with Hybrid Expert Guidance (API or Local).
"""

from __future__ import annotations
import time
from typing import Any, Dict
from decouple import config
from openai import OpenAI, RateLimitError
import os
from tutor.solver import GemmaSolver
from tutor.evaluator import evaluator
from tutor.optimizer import optimizer

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openrouter/free"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2

PROMPT_FILE = os.path.join(os.path.dirname(__file__), "system_prompt.txt")

def load_system_prompt() -> str:
    with open(PROMPT_FILE, "r") as f:
        return f.read()

def save_system_prompt(content: str):
    with open(PROMPT_FILE, "w") as f:
        f.write(content)

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
        content = load_system_prompt()
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

        response_text = ""
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(**kwargs)
                response_text = response.choices[0].message.content or ""
                break
            except RateLimitError:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))
                else:
                    raise

        # 3. Dynamic Prompt Evaluation & Optimization
        if response_text and solution:
            user_input = conversation[-1]["content"] if conversation else "Start of conversation"
            score, reason = evaluator.evaluate(user_input, response_text, solution)
            
            if score < 0.4: # Threshold for failure
                print(f"[SocraticTutor] Low score detected ({score}). Reason: {reason}. Optimizing prompt...")
                failure_analysis = f"Score: {score}\nReason: {reason}\nAssistant Response: {response_text}"
                current_prompt = load_system_prompt()
                new_prompt = optimizer.optimize(current_prompt, failure_analysis)
                
                if new_prompt and new_prompt != current_prompt:
                    save_system_prompt(new_prompt)
                    print("[SocraticTutor] Prompt updated and saved.")
        
        return response_text

_instance: SocraticTutor | None = None

def get_tutor() -> SocraticTutor:
    """Singleton accessor."""
    global _instance
    if _instance is None:
        _instance = SocraticTutor()
    return _instance
