import numpy as np
from judge.pipeline import LLMJudge

def run_mitigated_pairwise(judge: LLMJudge, query: str, context: str, expected_output: str, 
                           response_a: str, response_b: str) -> dict:
    """
    Runs pairwise evaluation in both orders (A-vs-B and B-vs-A) to detect and mitigate position bias.
    Maps decisions back and computes a debiased consensus winner.
    """
    # Run 1: A in Pos 1, B in Pos 2
    verdict1 = judge.evaluate_pairwise(query, context, expected_output, response_a, response_b)
    w1 = verdict1["winner"]  # "A", "B", or "TIE"

    # Run 2: B in Pos 1, A in Pos 2
    verdict2 = judge.evaluate_pairwise(query, context, expected_output, response_b, response_a)
    w2 = verdict2["winner"]  # "A", "B", or "TIE"

    # Map Run 2 winner back to original candidates (from Run 2 perspective, A is B, B is A)
    w2_mapped = "TIE"
    if w2 == "A":
        w2_mapped = "B"
    elif w2 == "B":
        w2_mapped = "A"

    # Check for position bias / flips
    # If the judge chose Position 1 in both runs (w1 == 'A' and w2 == 'A'),
    # or Position 2 in both runs (w1 == 'B' and w2 == 'B'), it's a position-biased flip.
    is_flip = False
    if (w1 == "A" and w2 == "A") or (w1 == "B" and w2 == "B"):
        is_flip = True
    elif w1 != w2_mapped:
        is_flip = True

    # Mitigation decision:
    # Consensus: only declare a winner if both runs agree. Otherwise, TIE.
    consensus_winner = "TIE"
    if w1 == w2_mapped:
        consensus_winner = w1
    
    return {
        "run1_winner": w1,
        "run2_winner": w2,
        "run2_mapped_winner": w2_mapped,
        "consensus_winner": consensus_winner,
        "is_position_biased": is_flip,
        "run1_raw": verdict1,
        "run2_raw": verdict2
    }

def measure_position_bias(judge: LLMJudge, test_suite: list[dict]) -> dict:
    """
    Evaluates a test suite of pairwise comparisons.
    Computes the flip rate (how often order swap leads to inconsistent results).
    """
    flips = 0
    total = len(test_suite)
    
    run1_winners = []
    consensus_winners = []
    
    for case in test_suite:
        res = run_mitigated_pairwise(
            judge,
            case["query"],
            case.get("context", ""),
            case.get("expected_output", ""),
            case["response_a"],
            case["response_b"]
        )
        if res["is_position_biased"]:
            flips += 1
            
        run1_winners.append(res["run1_winner"])
        consensus_winners.append(res["consensus_winner"])
        
    flip_rate = flips / total if total > 0 else 0.0
    
    return {
        "total_cases": total,
        "flip_rate": flip_rate,
        "run1_distribution": {
            "A": run1_winners.count("A"),
            "B": run1_winners.count("B"),
            "TIE": run1_winners.count("TIE")
        },
        "consensus_distribution": {
            "A": consensus_winners.count("A"),
            "B": consensus_winners.count("B"),
            "TIE": consensus_winners.count("TIE")
        }
    }
