"""
Socratic tutor powered by an LLM via OpenRouter.

Uses the OpenAI Python SDK (OpenRouter is API-compatible) so that
streaming, tool-calling, and model swapping work out of the box.

Environment variables
---------------------
OPENROUTER_API_KEY : str   - required
TUTOR_MODEL        : str   - optional, defaults to a cheap model
"""

from __future__ import annotations

import time
from typing import Any

from decouple import config
from openai import OpenAI, RateLimitError

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DEFAULT_MODEL = "openrouter/free"

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2

SYSTEM_PROMPT = """\
You are a Socratic tutor helping a student prepare for a coding interview.

RULES:
1. NEVER give the solution outright.  Instead, guide the student by asking targeted questions that help them discover the answer on their own.
2. When the student is stuck, break the problem into smaller sub-problems and ask about each one.
3. Acknowledge correct reasoning enthusiastically, then nudge toward the next step.
4. If the student asks you to "just give me the answer", gently explain why working through the problem themselves will help them more in the interview - then ask a simpler guiding question.
5. You may use brief code snippets to illustrate a concept, but never write a full solution.
6. Keep responses concise - prefer 2-4 sentences per message plus an optional code snippet.
7. Start each new conversation by acknowledging the specific problem the student is working on and asking what they have tried so far.

PROBLEM CONTEXT (provided below) gives you the problem statement. Use it to tailor your questions.\
"""

# Placeholder: when tool-calling is added, define tools here and pass them
# to the chat completion request via the `tools` parameter.
TOOLS: list[dict[str, Any]] = []


class SocraticTutor:
    """Lightweight wrapper around a chat-completions call."""

    def __init__(self) -> None:
        api_key = config("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY environment variable is not set. "
                "Get a free key at https://openrouter.ai/keys"
            )
        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
        )
        self.model = config("TUTOR_MODEL", DEFAULT_MODEL)

    def _build_system_message(self, problem_description: str) -> dict[str, str]:
        content = SYSTEM_PROMPT
        if problem_description:
            content += f"\n\n--- PROBLEM ---\n{problem_description}"
        return {"role": "system", "content": content}

    def reply(self, conversation: list[dict[str, str]], problem_description: str = "",) -> str:
        """Return an assistant reply for the given conversation history.

        Parameters
        ----------
        conversation : list[dict]
            List of ``{"role": "user"|"assistant", "content": "..."}`` dicts
            representing the chat so far.
        problem_description : str
            The full problem statement to inject into the system prompt.
        """
        messages = [self._build_system_message(problem_description)] + conversation

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 512,
        }
        if TOOLS:
            kwargs["tools"] = TOOLS

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
    """Singleton accessor - avoids re-creating the client on every request."""
    global _instance
    if _instance is None:
        _instance = SocraticTutor()
    return _instance
