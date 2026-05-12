from decouple import config
from huggingface_hub import InferenceClient
from tutor.solver_evaluator import solver_evaluator
from tutor.prompts import load_prompt, render_prompt
from rag.retriever import get_retriever

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
        self._retriever = None  # Lazy-loaded on first solve
        
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

    def _get_retriever(self):
        """Lazy-load the RAG retriever to avoid startup cost."""
        if self._retriever is None:
            try:
                self._retriever = get_retriever()
            except Exception as e:
                print(f"[GemmaSolver] Warning: Could not load RAG retriever: {e}")
        return self._retriever

    def _build_few_shot_context(self, problem_description: str) -> str:
        """
        Retrieve similar solved problems from the ChromaDB solutions index
        and format them as few-shot examples to enrich the solver prompt.
        """
        retriever = self._get_retriever()
        if retriever is None:
            return ""

        try:
            results = retriever.retrieve_solutions(problem_description, n_results=2)
            documents = results.get("documents", [[]])[0]
            if not documents:
                return ""

            examples = []
            for idx, solution_code in enumerate(documents, start=1):
                examples.append(f"--- Reference Solution {idx} ---\n{solution_code.strip()}")

            return load_prompt("few_shot_preamble") + "\n\n".join(examples)
        except Exception as e:
            print(f"[GemmaSolver] RAG retrieval failed (non-fatal): {e}")
            return ""

    def solve(self, problem_description: str, max_retries: int = 2) -> str:
        """Entry point for getting the expert solution with self-evaluation."""
        best_solution = ""
        best_score = -1.0

        # Build few-shot context once for all retry attempts
        few_shot_context = self._build_few_shot_context(problem_description)
        if few_shot_context:
            print("[GemmaSolver] RAG: injecting similar solutions into prompt.")
        
        for attempt in range(max_retries + 1):
            if self.mode == "api":
                solution = self._solve_via_hf_api(problem_description, few_shot_context)
            else:
                solution = self._solve_locally(problem_description, few_shot_context)
            
            # Evaluate the solution
            score, reason = solver_evaluator.evaluate(problem_description, solution)
            print(f"[GemmaSolver] Attempt {attempt+1} score: {score}. Reason: {reason}")
            
            if score >= 0.8: # Threshold for "Expert" quality
                return solution
            
            if score > best_score:
                best_score = score
                best_solution = solution
                
            if attempt < max_retries:
                print(f"[GemmaSolver] Score {score} too low. Retrying...")

        print(f"[GemmaSolver] Warning: Could not reach target score. Returning best solution (score: {best_score})")
        return best_solution

    def _solve_via_hf_api(self, problem_description: str, few_shot_context: str = "") -> str:
        """Calls the Hugging Face Inference API using chat completion."""
        system_prompt = load_prompt("solver_system").strip()
        user_prompt = render_prompt("solver_user", problem_description=problem_description, few_shot_context=few_shot_context)
        
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

    def _solve_locally(self, problem_description: str, few_shot_context: str = "") -> str:
        try:
            self._load_local_model()
            import torch
            prompt = render_prompt("solver_local", problem_description=problem_description, few_shot_context=few_shot_context)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.local_model.device)
            
            with torch.no_grad():
                outputs = self.local_model.generate(**inputs, max_new_tokens=1024)
            
            return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        except Exception as e:
            print(f"[GemmaSolver] Local Error: {e}")
            return "Expert solution unavailable locally."
