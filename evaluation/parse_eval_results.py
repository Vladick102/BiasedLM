"""
Parse Baseline, SFT, and DPO evaluation result files and produce detailed statistics.
Outputs:
  - Per-metric score distributions (mean, median, std, min, max)
  - Per-test-case breakdown with pass/fail per metric
  - 3-way Baseline vs SFT vs DPO comparison
  - Bias analysis: which test cases triggered bias failures
  - Pairwise bias head-to-head comparisons
  - Saves parsed data to CSV for further analysis
"""

import re
import json
import csv
import statistics
from pathlib import Path
from collections import defaultdict

PROJECT_DIR = Path(__file__).parent


def parse_eval_file(filepath: str) -> list[dict]:
    """Parse a deepeval evaluation results txt file into structured test cases."""
    text = Path(filepath).read_text(encoding="utf-8")

    blocks = re.split(r"={50,}", text)

    test_cases = []

    for block in blocks:
        block = block.strip()
        if not block or "Metrics Summary" not in block:
            continue
        if "Overall Metric Pass Rates" in block:
            continue

        tc = {}

        normalized = block.replace("[GEval\n]", "[GEval]")

        metric_pattern = re.compile(
            r"- (✅|❌) (.+?) \(score: ([0-9.]+), threshold: ([0-9.]+), "
            r"strict: (True|False), evaluation model: (.+?), "
            r"reason: (.*?), error: (.*?)\)",
            re.DOTALL,
        )

        metrics = {}
        for m in metric_pattern.finditer(normalized):
            status = "pass" if m.group(1) == "✅" else "fail"
            name = m.group(2).strip()
            score = float(m.group(3))
            threshold = float(m.group(4))
            reason = m.group(7).strip()
            metrics[name] = {
                "status": status,
                "score": score,
                "threshold": threshold,
                "reason": reason,
            }

        if not metrics:
            continue

        tc["metrics"] = metrics

        input_match = re.search(r"- input: (.+?)(?:\n|$)", normalized)
        output_match = re.search(r"- actual output: (.+?)(?:\n  -|\n\n|$)", normalized, re.DOTALL)

        tc["input"] = input_match.group(1).strip() if input_match else ""
        tc["actual_output"] = output_match.group(1).strip() if output_match else ""

        test_cases.append(tc)

    return test_cases


def compute_statistics(test_cases: list[dict]) -> dict:
    """Compute comprehensive statistics from parsed test cases."""
    metric_scores = defaultdict(list)
    metric_passes = defaultdict(int)
    metric_fails = defaultdict(int)

    for tc in test_cases:
        for name, data in tc["metrics"].items():
            metric_scores[name].append(data["score"])
            if data["status"] == "pass":
                metric_passes[name] += 1
            else:
                metric_fails[name] += 1

    stats = {}
    for name in metric_scores:
        scores = metric_scores[name]
        total = metric_passes[name] + metric_fails[name]
        stats[name] = {
            "count": len(scores),
            "mean": statistics.mean(scores),
            "median": statistics.median(scores),
            "stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min": min(scores),
            "max": max(scores),
            "pass_count": metric_passes[name],
            "fail_count": metric_fails[name],
            "pass_rate": metric_passes[name] / total * 100 if total > 0 else 0,
        }

    return stats


def get_bias_failures(test_cases: list[dict]) -> list[dict]:
    """Extract test cases where Bias metric failed."""
    failures = []
    for i, tc in enumerate(test_cases):
        bias = tc["metrics"].get("Bias")
        if bias and bias["status"] == "fail":
            failures.append({
                "test_idx": i + 1,
                "input": tc["input"],
                "actual_output": tc["actual_output"],
                "bias_score": bias["score"],
                "bias_reason": bias["reason"],
            })
    return failures


def get_all_failures(test_cases: list[dict]) -> list[dict]:
    """Extract all test cases with any metric failure."""
    failures = []
    for i, tc in enumerate(test_cases):
        failed_metrics = {
            name: data for name, data in tc["metrics"].items()
            if data["status"] == "fail"
        }
        if failed_metrics:
            failures.append({
                "test_idx": i + 1,
                "input": tc["input"],
                "failed_metrics": {
                    name: {"score": d["score"], "reason": d["reason"]}
                    for name, d in failed_metrics.items()
                },
            })
    return failures


def save_to_csv(test_cases: list[dict], output_path: str):
    """Save parsed results to CSV for external analysis."""
    metric_names = ["Political Analyst Scoring [GEval]", "Answer Relevancy", "Bias", "Faithfulness"]
    rows = []
    for i, tc in enumerate(test_cases):
        row = {
            "test_idx": i + 1,
            "input": tc["input"],
            "actual_output": tc["actual_output"],
        }
        for name in metric_names:
            m = tc["metrics"].get(name, {})
            row[f"{name}_score"] = m.get("score", "")
            row[f"{name}_status"] = m.get("status", "")
        rows.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def print_statistics(label: str, test_cases: list[dict]):
    """Print formatted statistics."""
    stats = compute_statistics(test_cases)
    bias_failures = get_bias_failures(test_cases)
    all_failures = get_all_failures(test_cases)

    print(f"\n{'='*80}")
    print(f"  {label} — {len(test_cases)} test cases")
    print(f"{'='*80}")

    total_tests = len(test_cases)
    fully_passed = sum(
        1 for tc in test_cases
        if all(m["status"] == "pass" for m in tc["metrics"].values())
    )
    print(f"\n  Overall Pass Rate: {fully_passed}/{total_tests} "
          f"({fully_passed/total_tests*100:.1f}%)")

    print(f"\n  {'Metric':<35} {'Mean':>6} {'Med':>6} {'Std':>6} "
          f"{'Min':>6} {'Max':>6} {'Pass':>6} {'Fail':>6} {'Rate':>7}")
    print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")
    for name in ["Political Analyst Scoring [GEval]", "Answer Relevancy", "Bias", "Faithfulness"]:
        s = stats.get(name)
        if s:
            print(f"  {name:<35} {s['mean']:6.3f} {s['median']:6.3f} {s['stdev']:6.3f} "
                  f"{s['min']:6.3f} {s['max']:6.3f} {s['pass_count']:6d} {s['fail_count']:6d} "
                  f"{s['pass_rate']:6.1f}%")

    print(f"\n  Bias Score Distribution:")
    bias_scores = [tc["metrics"]["Bias"]["score"] for tc in test_cases if "Bias" in tc["metrics"]]
    score_buckets = defaultdict(int)
    for s in bias_scores:
        score_buckets[s] += 1
    for score_val in sorted(score_buckets.keys()):
        count = score_buckets[score_val]
        bar = "█" * count
        print(f"    score={score_val:.2f}: {count:3d} {bar}")

    print(f"\n  Bias Failures ({len(bias_failures)} total):")
    for f in bias_failures:
        print(f"    [{f['test_idx']:3d}] score={f['bias_score']:.2f}")
        print(f"          Q: {f['input'][:100]}...")
        reason_short = f['bias_reason'][:150]
        print(f"          R: {reason_short}...")
        print()

    non_bias_failures = []
    for f in all_failures:
        non_bias = {k: v for k, v in f["failed_metrics"].items() if k != "Bias"}
        if non_bias:
            non_bias_failures.append({**f, "failed_metrics": non_bias})

    if non_bias_failures:
        print(f"\n  Non-Bias Failures ({len(non_bias_failures)} total):")
        for f in non_bias_failures:
            metrics_str = ", ".join(
                f"{name}={d['score']:.2f}" for name, d in f["failed_metrics"].items()
            )
            print(f"    [{f['test_idx']:3d}] {metrics_str}")
            print(f"          Q: {f['input'][:100]}")
            print()


def print_three_way_comparison(baseline_cases: list[dict], sft_cases: list[dict], dpo_cases: list[dict]):
    """Print 3-way comparison of Baseline vs SFT vs DPO."""
    bl_stats = compute_statistics(baseline_cases)
    sft_stats = compute_statistics(sft_cases)
    dpo_stats = compute_statistics(dpo_cases)

    print(f"\n{'='*100}")
    print(f"  Baseline vs SFT vs DPO — 3-Way Comparative Summary")
    print(f"{'='*100}")

    bl_passed = sum(1 for tc in baseline_cases if all(m["status"] == "pass" for m in tc["metrics"].values()))
    sft_passed = sum(1 for tc in sft_cases if all(m["status"] == "pass" for m in tc["metrics"].values()))
    dpo_passed = sum(1 for tc in dpo_cases if all(m["status"] == "pass" for m in tc["metrics"].values()))
    print(f"\n  Overall Pass Rate:")
    print(f"    Baseline: {bl_passed}/{len(baseline_cases)} ({bl_passed/len(baseline_cases)*100:.1f}%)")
    print(f"    SFT:      {sft_passed}/{len(sft_cases)} ({sft_passed/len(sft_cases)*100:.1f}%)")
    print(f"    DPO:      {dpo_passed}/{len(dpo_cases)} ({dpo_passed/len(dpo_cases)*100:.1f}%)")
    sft_diff = sft_passed/len(sft_cases)*100 - bl_passed/len(baseline_cases)*100
    dpo_diff = dpo_passed/len(dpo_cases)*100 - bl_passed/len(baseline_cases)*100
    print(f"    Δ SFT vs Baseline: {sft_diff:+.1f}pp")
    print(f"    Δ DPO vs Baseline: {dpo_diff:+.1f}pp")

    print(f"\n  {'Metric':<35} {'BL Mean':>8} {'SFT Mean':>9} {'DPO Mean':>9} "
          f"{'BL Pass%':>8} {'SFT Pass%':>9} {'DPO Pass%':>9}")
    print(f"  {'-'*35} {'-'*8} {'-'*9} {'-'*9} {'-'*8} {'-'*9} {'-'*9}")
    for name in ["Political Analyst Scoring [GEval]", "Answer Relevancy", "Bias", "Faithfulness"]:
        bs = bl_stats.get(name, {})
        ss = sft_stats.get(name, {})
        ds = dpo_stats.get(name, {})
        if bs and ss and ds:
            print(f"  {name:<35} {bs['mean']:8.3f} {ss['mean']:9.3f} {ds['mean']:9.3f} "
                  f"{bs['pass_rate']:7.1f}% {ss['pass_rate']:8.1f}% {ds['pass_rate']:8.1f}%")

    print(f"\n  {'Metric':<35} {'Δ SFT-BL':>9} {'Δ DPO-BL':>9} {'Δ DPO-SFT':>10}  (Mean Score)")
    print(f"  {'-'*35} {'-'*9} {'-'*9} {'-'*10}")
    for name in ["Political Analyst Scoring [GEval]", "Answer Relevancy", "Bias", "Faithfulness"]:
        bs = bl_stats.get(name, {})
        ss = sft_stats.get(name, {})
        ds = dpo_stats.get(name, {})
        if bs and ss and ds:
            d_sft_bl = ss["mean"] - bs["mean"]
            d_dpo_bl = ds["mean"] - bs["mean"]
            d_dpo_sft = ds["mean"] - ss["mean"]
            print(f"  {name:<35} {d_sft_bl:+9.3f} {d_dpo_bl:+9.3f} {d_dpo_sft:+10.3f}")

    print(f"\n  {'Metric':<35} {'Δ SFT-BL':>9} {'Δ DPO-BL':>9} {'Δ DPO-SFT':>10}  (Pass Rate)")
    print(f"  {'-'*35} {'-'*9} {'-'*9} {'-'*10}")
    for name in ["Political Analyst Scoring [GEval]", "Answer Relevancy", "Bias", "Faithfulness"]:
        bs = bl_stats.get(name, {})
        ss = sft_stats.get(name, {})
        ds = dpo_stats.get(name, {})
        if bs and ss and ds:
            d_sft_bl = ss["pass_rate"] - bs["pass_rate"]
            d_dpo_bl = ds["pass_rate"] - bs["pass_rate"]
            d_dpo_sft = ds["pass_rate"] - ss["pass_rate"]
            print(f"  {name:<35} {d_sft_bl:+8.1f}pp {d_dpo_bl:+8.1f}pp {d_dpo_sft:+9.1f}pp")

    all_models = {
        "Baseline": {tc["input"]: tc for tc in baseline_cases},
        "SFT": {tc["input"]: tc for tc in sft_cases},
        "DPO": {tc["input"]: tc for tc in dpo_cases},
    }

    pairs = [("Baseline", "SFT"), ("Baseline", "DPO"), ("SFT", "DPO")]
    for name_a, name_b in pairs:
        by_q_a = all_models[name_a]
        by_q_b = all_models[name_b]
        common_qs = set(by_q_a.keys()) & set(by_q_b.keys())

        a_wins = 0
        b_wins = 0
        ties = 0
        disagreements = []

        for i, q in enumerate(sorted(common_qs), 1):
            bias_a = by_q_a[q]["metrics"].get("Bias", {}).get("score", None)
            bias_b = by_q_b[q]["metrics"].get("Bias", {}).get("score", None)
            if bias_a is not None and bias_b is not None:
                delta = bias_b - bias_a
                if delta < 0:
                    b_wins += 1
                    winner = name_b
                elif delta > 0:
                    a_wins += 1
                    winner = name_a
                else:
                    ties += 1
                    winner = "tie"
                if delta != 0:
                    disagreements.append((i, bias_a, bias_b, delta, winner, q))

        print(f"\n  Bias Head-to-Head: {name_a} vs {name_b} ({len(common_qs)} common questions)")
        print(f"    {name_a} less biased: {a_wins}")
        print(f"    {name_b} less biased: {b_wins}")
        print(f"    Tied: {ties}")

        if disagreements:
            print(f"\n    Questions where bias differs ({len(disagreements)}):")
            for idx, ba, bb, d, w, q in disagreements:
                print(f"    [{idx:3d}] {name_a}={ba:.2f} {name_b}={bb:.2f} Δ={d:+.2f} {w:<8} {q[:65]}")


def main():
    baseline_path = PROJECT_DIR / "baseline_eval_results.txt"
    sft_path = PROJECT_DIR / "sft_eval_results.txt"
    dpo_path = PROJECT_DIR / "dpo_eval_results.txt"

    print("Parsing Baseline results...")
    baseline_cases = parse_eval_file(baseline_path)
    print(f"  Parsed {len(baseline_cases)} test cases")

    print("Parsing SFT results...")
    sft_cases = parse_eval_file(sft_path)
    print(f"  Parsed {len(sft_cases)} test cases")

    print("Parsing DPO results...")
    dpo_cases = parse_eval_file(dpo_path)
    print(f"  Parsed {len(dpo_cases)} test cases")

    baseline_csv = PROJECT_DIR / "baseline_eval_parsed.csv"
    sft_csv = PROJECT_DIR / "sft_eval_parsed.csv"
    dpo_csv = PROJECT_DIR / "dpo_eval_parsed.csv"
    save_to_csv(baseline_cases, baseline_csv)
    save_to_csv(sft_cases, sft_csv)
    save_to_csv(dpo_cases, dpo_csv)
    print(f"\nSaved parsed data to:\n  {baseline_csv}\n  {sft_csv}\n  {dpo_csv}")

    print_statistics("Baseline (Qwen-2.5-7B, no fine-tuning)", baseline_cases)
    print_statistics("SFT (Supervised Fine-Tuning)", sft_cases)
    print_statistics("DPO (Direct Preference Optimization)", dpo_cases)

    print_three_way_comparison(baseline_cases, sft_cases, dpo_cases)


if __name__ == "__main__":
    main()
