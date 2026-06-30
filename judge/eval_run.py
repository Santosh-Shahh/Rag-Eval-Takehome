import os
import json
import argparse
import time
from pathlib import Path

from judge.pipeline import LLMJudge
from judge.bias import measure_position_bias, run_mitigated_pairwise
from judge.validation import run_adversarial_probe, run_test_retest

def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge Evaluation Runner")
    parser.add_argument("--suite", type=str, default="data/test_suite.json", help="Path to the JSON test suite file.")
    parser.add_argument("--judge-model", type=str, default=None, help="Gemini model to use as the judge.")
    parser.add_argument("--runs", type=int, default=2, help="Number of test-retest runs for validation.")
    
    args = parser.parse_args()
    
    suite_path = Path(args.suite)
    if not suite_path.exists():
        print(f"Test suite file {suite_path} does not exist. Run tests/generate_test_suite.py first.")
        return
        
    with open(suite_path, "r") as f:
        test_cases = json.load(f)
        
    print(f"Loaded {len(test_cases)} test cases from {suite_path.name}")
    
    # Initialize Judge
    # Default is gemini-1.5-flash
    judge = LLMJudge(judge_model=args.judge_model)
    print(f"Judge initialized with provider: {judge.provider}, model: {judge.model}")
    
    # 1. Run A/B Comparison and Position Bias Check
    print("\nRunning A/B pairwise evaluation with position-swapping...")
    start_time = time.time()
    
    results = []
    flips = 0
    
    for idx, case in enumerate(test_cases):
        q = case["query"]
        ctx = case.get("context", "")
        exp = case.get("expected_output", "")
        resp_a = case["response_a"]
        resp_b = case["response_b"]
        
        # Position swap evaluation
        res = run_mitigated_pairwise(judge, q, ctx, exp, resp_a, resp_b)
        
        results.append({
            "id": case["id"],
            "query": q,
            "response_a_name": "Config A (Verbose)",
            "response_b_name": "Config B (Terse)",
            "run1_winner": res["run1_winner"],
            "run2_winner": res["run2_winner"],
            "run2_mapped_winner": res["run2_mapped_winner"],
            "consensus_winner": res["consensus_winner"],
            "is_position_biased": res["is_position_biased"],
            "run1_rationale": res["run1_raw"]["overall_rationale"],
            "run2_rationale": res["run2_raw"]["overall_rationale"]
        })
        
        if res["is_position_biased"]:
            flips += 1
            
        print(f"Case [{idx+1}/{len(test_cases)}]: Run1 Winner={res['run1_winner']} | Run2 Winner={res['run2_winner']} | Mapped={res['run2_mapped_winner']} | Consensus={res['consensus_winner']}")
        
    elapsed = time.time() - start_time
    flip_rate = flips / len(test_cases) if test_cases else 0.0
    
    # Calculate distributions
    run1_wins_a = sum(1 for r in results if r["run1_winner"] == "A")
    run1_wins_b = sum(1 for r in results if r["run1_winner"] == "B")
    run1_ties = sum(1 for r in results if r["run1_winner"] == "TIE")
    
    consensus_wins_a = sum(1 for r in results if r["consensus_winner"] == "A")
    consensus_wins_b = sum(1 for r in results if r["consensus_winner"] == "B")
    consensus_ties = sum(1 for r in results if r["consensus_winner"] == "TIE")
    
    # Declare Winner
    winner_name = "TIE"
    if consensus_wins_a > consensus_wins_b:
        winner_name = "Config A (Verbose)"
    elif consensus_wins_b > consensus_wins_a:
        winner_name = "Config B (Terse)"
        
    report = {
        "summary": {
            "total_cases": len(test_cases),
            "duration_seconds": round(elapsed, 2),
            "judge_calls": judge.calls_count,
            "total_tokens": judge.total_prompt_tokens + judge.total_completion_tokens,
            "prompt_tokens": judge.total_prompt_tokens,
            "completion_tokens": judge.total_completion_tokens,
            "position_flip_rate_before_mitigation": flip_rate
        },
        "win_rates": {
            "before_mitigation": {
                "config_a_win_rate": run1_wins_a / len(test_cases),
                "config_b_win_rate": run1_wins_b / len(test_cases),
                "tie_rate": run1_ties / len(test_cases)
            },
            "after_mitigation": {
                "config_a_win_rate": consensus_wins_a / len(test_cases),
                "config_b_win_rate": consensus_wins_b / len(test_cases),
                "tie_rate": consensus_ties / len(test_cases)
            }
        },
        "declared_winner": winner_name
    }
    
    # 2. Run Validation - Adversarial Probe
    print("\nRunning Adversarial Probe Validation...")
    adv_probe = run_adversarial_probe(judge)
    print(f"Adversarial Probe Completed: Winner={adv_probe['winner_declared']} | Fooled={adv_probe['fooled']}")
    
    # 3. Run Validation - Test-Retest Consistency
    print("\nRunning Test-Retest Consistency check...")
    test_retest = run_test_retest(judge, test_cases, runs=args.runs)
    print(f"Test-Retest Completed: Flip Rate={test_retest['test_retest_flip_rate']:.4f}")
    
    # Save Report and Audit trail
    out_dir = suite_path.parent
    
    # Save audit trail (reproducibility / audit log)
    audit_trail = {
        "results": results,
        "adversarial_validation": adv_probe,
        "consistency_validation": test_retest,
        "token_accounting": {
            "judge_calls": judge.calls_count,
            "total_tokens": judge.total_prompt_tokens + judge.total_completion_tokens,
            "prompt_tokens": judge.total_prompt_tokens,
            "completion_tokens": judge.total_completion_tokens,
        }
    }
    with open(out_dir / "judge_audit.json", "w") as f:
        json.dump(audit_trail, f, indent=2)
        
    with open(out_dir / "judge_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print("\n=== JUDGE EVALUATION SUMMARY REPORT ===")
    print(f"Total Cases: {report['summary']['total_cases']}")
    print(f"Position Flip Rate: {report['summary']['position_flip_rate_before_mitigation']:.2%}")
    print("-" * 35)
    print("Before Position-Swap Mitigation (Run 1 Only):")
    print(f"  Config A Win Rate: {report['win_rates']['before_mitigation']['config_a_win_rate']:.2%}")
    print(f"  Config B Win Rate: {report['win_rates']['before_mitigation']['config_b_win_rate']:.2%}")
    print(f"  Tie Rate:          {report['win_rates']['before_mitigation']['tie_rate']:.2%}")
    print("After Position-Swap Mitigation (Consensus):")
    print(f"  Config A Win Rate: {report['win_rates']['after_mitigation']['config_a_win_rate']:.2%}")
    print(f"  Config B Win Rate: {report['win_rates']['after_mitigation']['config_b_win_rate']:.2%}")
    print(f"  Tie Rate:          {report['win_rates']['after_mitigation']['tie_rate']:.2%}")
    print("-" * 35)
    print(f"Declared Winner:   {report['declared_winner']}")
    print(f"Judge Validation:")
    print(f"  Adversarial Probe Fooled? {adv_probe['fooled']} (Winner: {adv_probe['winner_declared']})")
    print(f"  Test-Retest Flip Rate:    {test_retest['test_retest_flip_rate']:.2%}")
    print("=======================================\n")
    
if __name__ == "__main__":
    main()
