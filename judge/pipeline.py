import os
import json
import re
import time
from google import genai
from judge.rubric import RUBRICS, POINTWISE_JUDGE_TEMPLATE, PAIRWISE_JUDGE_TEMPLATE
from rag.config import GEMINI_API_KEY, LLM_MODEL

def parse_structured_verdict(raw_response: str, expected_type: str = "pointwise") -> dict:
    """
    Robustly parses the JSON verdict from the judge's response.
    Handles code blocks, extra text, and malformed JSON.
    """
    clean_text = raw_response.strip()
    
    # Strip markdown json code blocks if present
    if clean_text.startswith("```"):
        # Remove first line (e.g. ```json)
        lines = clean_text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        clean_text = "\n".join(lines).strip()

    # Find the JSON block starting with { and ending with }
    json_match = re.search(r'(\{.*\})', clean_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
        try:
            return json.loads(json_str)
        except Exception as e:
            # Try to fix basic JSON formatting errors (e.g. trailing commas, smart quotes)
            # Replace smart quotes
            json_str_fixed = json_str.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
            # Remove trailing commas before closing braces
            json_str_fixed = re.sub(r',\s*\}', '}', json_str_fixed)
            json_str_fixed = re.sub(r',\s*\]', ']', json_str_fixed)
            try:
                return json.loads(json_str_fixed)
            except Exception as e_fixed:
                print(f"JSON Parsing failed on fixed string: {e_fixed}")
    
    # If all parsing fails, perform regex extraction based on expected type
    print("WARNING: Falling back to regex extraction for judge verdict.")
    
    if expected_type == "pointwise":
        # Extract score
        score_match = re.search(r'"score"\s*:\s*(\d+)', clean_text)
        if not score_match:
            score_match = re.search(r'score\s*is\s*(\d+)', clean_text, re.IGNORECASE)
            
        score = int(score_match.group(1)) if score_match else 3
        
        # Extract rationale
        rat_match = re.search(r'"rationale"\s*:\s*"(.*?)"', clean_text, re.DOTALL)
        if not rat_match:
            rat_match = re.search(r'rationale\s*:\s*(.*)', clean_text, re.IGNORECASE)
            
        rationale = rat_match.group(1).strip() if rat_match else clean_text[:200]
        
        return {"score": score, "rationale": rationale}
    
    else: # pairwise
        winner_match = re.search(r'"winner"\s*:\s*"(A|B|TIE)"', clean_text, re.IGNORECASE)
        if not winner_match:
            winner_match = re.search(r'winner\s*is\s*(A|B|TIE)', clean_text, re.IGNORECASE)
            
        winner = winner_match.group(1).upper() if winner_match else "TIE"
        
        rat_match = re.search(r'"overall_rationale"\s*:\s*"(.*?)"', clean_text, re.DOTALL)
        rationale = rat_match.group(1).strip() if rat_match else clean_text[:200]
        
        return {
            "winner": winner,
            "overall_rationale": rationale,
            "reason_correctness": "Extracted via fallback",
            "reason_completeness": "Extracted via fallback",
            "reason_faithfulness": "Extracted via fallback"
        }

class LLMJudge:
    def __init__(self, judge_model=None, judge_provider="gemini"):
        self.provider = judge_provider
        self.model = judge_model or LLM_MODEL
        self.client = None
        self.calls_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        # Load API key
        if self.provider == "gemini":
            if not GEMINI_API_KEY:
                print("WARNING: GEMINI_API_KEY not found. Operating LLMJudge in offline/mock mode.")
                self.provider = "mock"
            else:
                try:
                    self.client = genai.Client(api_key=GEMINI_API_KEY)
                except Exception as e:
                    print(f"Error initializing judge: {e}. Falling back to mock.")
                    self.provider = "mock"

    def _call_llm(self, prompt: str) -> str:
        """Helper to invoke LLM and track token usage."""
        self.calls_count += 1
        
        if self.provider == "gemini" and self.client:
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt
                )
                if response.usage_metadata:
                    self.total_prompt_tokens += response.usage_metadata.prompt_token_count
                    self.total_completion_tokens += response.usage_metadata.candidates_token_count
                return response.text
            except Exception as e:
                print(f"Judge API call failed: {e}. Falling back to mock response.")
                # Fallback to mock behavior
        
        # Mock LLM Judge response generator (for offline runs/tests)
        time.sleep(0.05)  # Simulate API delay
        self.total_prompt_tokens += len(prompt.split())
        
        # Check if pairwise or pointwise
        if "winner" in prompt:
            # Pairwise mock response
            # Heuristic: A is better if it doesn't contain "bad word" or based on text length
            winner = "A"
            if "response_a" in prompt and "response_b" in prompt:
                # If response_a has "wrong" or "error", make B win
                if "wrong" in prompt.lower() or "error" in prompt.lower():
                    winner = "B"
            mock_res = {
                "winner": winner,
                "reason_correctness": "Response B is factually accurate.",
                "reason_completeness": "Response B covers all aspects.",
                "reason_faithfulness": "No hallucinations found in winning response.",
                "overall_rationale": "Mock judge selected candidate based on mock heuristic."
            }
            res_str = json.dumps(mock_res)
            self.total_completion_tokens += len(res_str.split())
            return res_str
        else:
            # Pointwise mock response
            score = 4
            if "model_output" in prompt:
                # Look for indicators of low quality
                if "wrong" in prompt.lower() or "error" in prompt.lower() or "not sure" in prompt.lower():
                    score = 2
            mock_res = {
                "score": score,
                "rationale": "Mock judge rated output based on keyword heuristics."
            }
            res_str = json.dumps(mock_res)
            self.total_completion_tokens += len(res_str.split())
            return res_str

    def evaluate_pointwise(self, query: str, context: str, expected_output: str, model_output: str, criterion: str) -> dict:
        """Runs pointwise scoring for a single criterion."""
        if criterion not in RUBRICS:
            raise ValueError(f"Criterion '{criterion}' not defined in rubrics.")
            
        rubric_details = RUBRICS[criterion]
        prompt = POINTWISE_JUDGE_TEMPLATE.format(
            query=query,
            context=context,
            expected_output=expected_output or "N/A",
            model_output=model_output,
            criterion=criterion.capitalize(),
            criterion_desc=rubric_details["description"],
            anchor_1=rubric_details["scale"]["1"],
            anchor_2=rubric_details["scale"]["2"],
            anchor_3=rubric_details["scale"]["3"],
            anchor_4=rubric_details["scale"]["4"],
            anchor_5=rubric_details["scale"]["5"]
        )

        raw_verdict = self._call_llm(prompt)
        verdict = parse_structured_verdict(raw_verdict, expected_type="pointwise")
        
        # Keep raw prompt and response logs for auditability
        verdict["raw_prompt"] = prompt
        verdict["raw_response"] = raw_verdict
        return verdict

    def evaluate_pairwise(self, query: str, context: str, expected_output: str, response_a: str, response_b: str) -> dict:
        """Runs pairwise A-vs-B comparison."""
        prompt = PAIRWISE_JUDGE_TEMPLATE.format(
            query=query,
            context=context,
            expected_output=expected_output or "N/A",
            response_a=response_a,
            response_b=response_b
        )

        raw_verdict = self._call_llm(prompt)
        verdict = parse_structured_verdict(raw_verdict, expected_type="pairwise")
        
        verdict["raw_prompt"] = prompt
        verdict["raw_response"] = raw_verdict
        return verdict
