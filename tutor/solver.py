import os
from typing import Optional
from decouple import config
from huggingface_hub import InferenceClient

class GemmaSolver:
    """
    Hybrid Expert Solver.
    - API Mode: Uses Hugging Face Inference API (Zero disk/RAM footprint).
    - Local Mode: Uses local fine-tuned weights (Requires space/GPU).
    """
    def __init__(self, mode: str = "api"):
        self.mode = mode.lower()
        self.hf_token = config("HF_TOKEN", None)
        self.model_id = config("GEMMA_BASE_MODEL", "google/gemma-2-2b-it")
        
        self.client = None
        self.local_model = None
        self.tokenizer = None
        
        if self.mode == "api":
            if not self.hf_token:
                print("[GemmaSolver] Warning: HF_TOKEN not found in .env. API calls will likely fail.")
            self.client = InferenceClient(model=self.model_id, token=self.hf_token)
        else:
            print("[GemmaSolver] Initializing in LOCAL mode...")

    def _load_local_model(self):
        """Internal helper to load local weights only when first used."""
        if self.local_model is not None:
            return
            
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        adapter_path = config("GEMMA_ADAPTER_PATH", "./gemma2-dsa-qlora copy")

        print(f"[GemmaSolver] Loading local weights: {adapter_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=self.hf_token)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            token=self.hf_token
        )
        self.local_model = PeftModel.from_pretrained(base_model, adapter_path)
        self.local_model.eval()
        print("[GemmaSolver] Local weights ready.")

    def solve(self, problem_description: str) -> str:
        """Entry point for getting the expert solution."""
        if self.mode == "api":
            return self._solve_via_hf_api(problem_description)
        else:
            return self._solve_locally(problem_description)

    def _solve_via_hf_api(self, problem_description: str) -> str:
        """Calls the Hugging Face Inference API using chat completion."""
        system_prompt = "You are an expert competitive programmer. Solve the problem in Python 3. Return ONLY the code inside triple backticks. Do not include any other conversational text."
        user_prompt = f"Problem:\n{problem_description}\n\nSolution:"
        
        try:
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1024,
                temperature=0.1,
                stop=["\n\n"]
            )
            print(f"[GemmaSolver] Problem solution: {response.choices[0].message.content}")
            return response.choices[0].message.content or "No solution generated."
        except Exception as e:
            print(f"[GemmaSolver] HF API Error: {e}")
            return "Expert solution unavailable via HF API."

    def _solve_locally(self, problem_description: str) -> str:
        try:
            self._load_local_model()
            import torch
            prompt = f"You are an expert competitive programmer. Solve the following problem in Python 3:\n\n{problem_description}\n\nSolution:\n"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.local_model.device)
            
            with torch.no_grad():
                outputs = self.local_model.generate(**inputs, max_new_tokens=1024)
            
            return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        except Exception as e:
            print(f"[GemmaSolver] Local Error: {e}")
            return "Expert solution unavailable locally."
