from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models.base_model import DeepEvalBaseLLM
from decouple import config

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
            criteria="""
            Evaluate the assistant's response based on these four pillars:
            1. NON-DISCLOSURE: Does it avoid giving the solution or code? (Critical failure if it gives code).
            2. RELEVANCE: Is the response directly related to the current problem and the user's last input?
            3. HELPFULNESS: Does it provide a meaningful nudge or question that helps the student progress? Avoid "I don't know" or irrelevant comments.
            4. PEDAGOGICAL BALANCE: Does it acknowledge correct steps? It shouldn't just ask questions; it should confirm if the student is on the right track.
            
            Score 0 if it reveals code.
            Score 1 if it is a perfect Socratic nudge.
            Score 0.5 if it is irrelevant or unhelpful.
            """,
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
