from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models.base_model import DeepEvalBaseLLM
from decouple import config
from tutor.prompts import load_prompt

class OpenRouterModel(DeepEvalBaseLLM):
    def __init__(self, model_name):
        self.model_name = model_name

    def load_model(self):
        from openai import OpenAI
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config("OPENROUTER_API_KEY")
        )

    def generate(self, prompt: str) -> str:
        client = self.load_model()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0 # Use 0 for more deterministic evaluation
        )
        res = response.choices[0].message.content or ""
        return res.strip()

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name

class TutorEvaluator:
    def __init__(self):
        model = OpenRouterModel(model_name=config("EVALUATOR_MODEL", "meta-llama/Llama-3.1-8B-Instruct")) 
        
        self.socratic_quality_metric = GEval(
            name="Socratic Quality",
            criteria=load_prompt("eval_socratic_quality"),
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
            model=model,
            threshold=0.7
        )

    def evaluate(self, user_input: str, assistant_response: str, ground_truth_solution: str) -> tuple[float, str]:
        test_case = LLMTestCase(
            input=user_input,
            actual_output=assistant_response,
            retrieval_context=[ground_truth_solution]
        )
        try:
            self.socratic_quality_metric.measure(test_case)
            score = self.socratic_quality_metric.score
            reason = self.socratic_quality_metric.reason
            print(f"[Evaluator] Score: {score}")
            print(f"[Evaluator] Reason: {reason}")
            return score, reason
        except Exception as e:
            print(f"[Evaluator] Error during evaluation: {e}")
            return 1.0, "" 

evaluator = TutorEvaluator()
