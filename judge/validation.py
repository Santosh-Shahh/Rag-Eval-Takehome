import time
from judge.pipeline import LLMJudge
from judge.bias import run_mitigated_pairwise

def run_adversarial_probe(judge: LLMJudge) -> dict:
    """
    Validates the judge using an adversarial probe:
    Verbose-but-Wrong response vs Terse-but-Correct response.
    Checks if the judge is fooled by verbosity/style.
    """
    query = "What is the default replication factor for AetherDB?"
    context = "AetherDB default replication factor is 3. Raft consensus protocol is used to replicate writes."
    expected_output = "The default replication factor is 3."

    # Terse-but-Correct response
    terse_correct = "The default replication factor is 3."
    
    # Verbose-but-Wrong response (authoritative tone, verbose, but states incorrect replication factor 5)
    verbose_wrong = (
        "AetherDB is an advanced distributed database designed for extreme reliability and high availability. "
        "To achieve partition tolerance, AetherDB replicates data dynamically across nodes. According to the "
        "cluster deployment guidelines, the default replication factor is configured to 5, which ensures "
        "replicated partitions are distributed across multiple availability zones and nodes for enterprise "
        "hardening, though it might increase disk storage usage slightly."
    )

    # Run mitigated pairwise
    res = run_mitigated_pairwise(
        judge,
        query,
        context,
        expected_output,
        verbose_wrong,  # Response A
        terse_correct   # Response B
    )

    winner = res["consensus_winner"]
    # If the judge chose Response A (verbose_wrong), it was fooled.
    # If it chose Response B (terse_correct) or declared a TIE, it was NOT fooled.
    fooled = "Y" if winner == "A" else "N"
    
    return {
        "query": query,
        "verbose_wrong_answer": verbose_wrong,
        "terse_correct_answer": terse_correct,
        "winner_declared": winner,
        "fooled": fooled,
        "rationale": res["run1_raw"]["overall_rationale"]
    }

def run_test_retest(judge: LLMJudge, test_suite: list[dict], runs: int = 2) -> dict:
    """
    Validates judge consistency by running the same test cases multiple times
    and measuring how often the verdict changes (flip rate).
    """
    if not test_suite:
        return {"test_retest_flip_rate": 0.0}

    flips = 0
    total = len(test_suite)

    for case in test_suite:
        verdicts = []
        for _ in range(runs):
            # Run pointwise or pairwise
            # Let's run pointwise correctness scoring
            v = judge.evaluate_pointwise(
                case["query"],
                case.get("context", ""),
                case.get("expected_output", ""),
                case["response_a"],  # evaluate response_a
                "correctness"
            )
            verdicts.append(v["score"])
            # Small sleep to avoid rate limits
            time.sleep(0.02)
        
        # Check if any scores differ
        if len(set(verdicts)) > 1:
            flips += 1

    flip_rate = flips / total if total > 0 else 0.0
    return {
        "runs_per_case": runs,
        "total_cases": total,
        "consistent_cases": total - flips,
        "inconsistent_cases": flips,
        "test_retest_flip_rate": flip_rate
    }
