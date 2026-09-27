import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.run_eval import run_eval  # reuse your existing function

N_RUNS = 3

def run_multiple(n=N_RUNS):
    all_summaries = []

    for run_num in range(1, n + 1):
        print("\n" + "#" * 60)
        print(f"# RUN {run_num}/{n}")
        print("#" * 60)

        run_eval()  # this already saves eval/results.json each time

        with open("eval/results.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        summary = data["summary"]
        summary["run"] = run_num
        all_summaries.append(summary)

        # save a copy per run so you don't overwrite results between runs
        with open(f"eval/results_run{run_num}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # aggregate
    avg_naive = sum(s["naive_pct"] for s in all_summaries) / n
    avg_agentic = sum(s["agentic_pct"] for s in all_summaries) / n
    avg_delta = sum(s["delta_pct"] for s in all_summaries) / n
    avg_mh_naive = sum(s["multihop_naive_pct"] for s in all_summaries) / n
    avg_mh_agentic = sum(s["multihop_agentic_pct"] for s in all_summaries) / n

    print("\n" + "=" * 60)
    print(f"FINAL AGGREGATE OVER {n} RUNS")
    print("=" * 60)
    for s in all_summaries:
        print(f"Run {s['run']}: naive={s['naive_pct']}%  agentic={s['agentic_pct']}%  delta={s['delta_pct']:+.1f}%")

    print(f"\nAverage naive:   {avg_naive:.1f}%")
    print(f"Average agentic: {avg_agentic:.1f}%")
    print(f"Average delta:   {avg_delta:+.1f} percentage points")
    print(f"\nAverage multi-hop naive:   {avg_mh_naive:.1f}%")
    print(f"Average multi-hop agentic: {avg_mh_agentic:.1f}%")

    with open("eval/results_aggregate.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_runs": n,
            "runs": all_summaries,
            "average_naive_pct": round(avg_naive, 1),
            "average_agentic_pct": round(avg_agentic, 1),
            "average_delta_pct": round(avg_delta, 1),
            "average_multihop_naive_pct": round(avg_mh_naive, 1),
            "average_multihop_agentic_pct": round(avg_mh_agentic, 1),
        }, f, indent=2)

    print("\nSaved aggregate to eval/results_aggregate.json")


if __name__ == "__main__":
    run_multiple(N_RUNS)