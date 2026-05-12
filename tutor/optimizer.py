from decouple import config
from openai import OpenAI
from tutor.prompts import render_prompt

class TutorOptimizer:
    def __init__(self):
        self.api_key = config("OPENROUTER_API_KEY")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        # Using a more capable model for prompt engineering
        self.model = "meta-llama/llama-3.1-70b-instruct"

    def optimize(self, current_prompt: str, failure_log: str) -> str:
        """
        Takes the current prompt and a description of why it failed, 
        and returns an improved system prompt.
        """
        meta_prompt = render_prompt(
            "optimizer_meta",
            current_prompt=current_prompt,
            failure_log=failure_log,
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": meta_prompt}],
                temperature=0.3,
                max_tokens=2048
            )
            new_prompt = response.choices[0].message.content or ""
            return new_prompt.strip() if new_prompt else current_prompt
        except Exception as e:
            print(f"[Optimizer] Error: {e}")
            return current_prompt

optimizer = TutorOptimizer()
