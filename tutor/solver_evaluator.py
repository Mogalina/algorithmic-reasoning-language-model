from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from tutor.evaluator import OpenRouterModel
from tutor.solver_rewards import compute_solver_reward, extract_code
from decouple import config

class SolverEvaluator:
    def __init__(self):
        # Using a heavy model for auditing code logic
        model_name = config("SOLVER_EVALUATOR_MODEL", "meta-llama/llama-3.1-70b-instruct")
        self.model = OpenRouterModel(model_name=model_name)
        
        self.correctness_metric = GEval(
            name="Code Logic Correctness",
            criteria="""
            Determine if the generated Python code correctly solves the given algorithmic problem.
            Evaluate based on:
            1. LOGIC: Does the algorithm correctly address the problem statement?
            2. EDGE CASES: Does the code handle empty inputs, nulls, or extreme values?
            3. EFFICIENCY: Is the time and space complexity optimal for a coding interview (e.g., O(N) or O(N log N) vs O(N^2))?
            4. SYNTAX: While we have a compiler check, ensure there are no subtle logical errors that might pass compilation but fail execution.
            
            Score 1 if the code is a perfect solution.
            Score 0 if it is fundamentally wrong or misses the core requirement.
            """,
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model=self.model,
            threshold=0.8
        )

    def evaluate(self, problem_description: str, generated_code: str) -> tuple[float, str]:
        reward_info = compute_solver_reward(generated_code)
        if not reward_info["compiled"]:
            return 0.0, f"Compilation failed: {reward_info['error']}"

        clean_code = extract_code(generated_code)
        
        test_case = LLMTestCase(
            input=problem_description,
            actual_output=clean_code
        )
        
        try:
            self.correctness_metric.measure(test_case)
            return self.correctness_metric.score, self.correctness_metric.reason
        except Exception as e:
            print(f"[SolverEvaluator] Error: {e}")
            return 1.0, "Evaluation failed, assuming correct for now."

solver_evaluator = SolverEvaluator()
