import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.naive_rag import naive_rag_answer
from src.agentic_rag import agentic_rag_answer
from src.llm import ask_llm

QUESTIONS_PATH = "eval/questions.json"
RESULTS_PATH = "eval/results.json"


def load_questions():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def judge_answer(question, ground_truth, model_answer):
    prompt = f"""You are evaluating an AI's answer for factual correctness against a ground truth.

Question: {question}
Ground truth: {ground_truth}
Model's answer: {model_answer}

Does the model's answer correctly and substantially capture the ground truth (partial credit for honest "not specified" caveats that match the ground truth's caveats)? Reply with ONLY a single digit: "1" if correct/acceptable, "0" if incorrect or missing the key fact."""
    result = ask_llm(prompt).strip()
    # defensive parsing in case model adds extra text
    for ch in result:
        if ch in "01":
            return int(ch)
    return 0


def run_eval():
    questions = load_questions()
    results = []

    naive_correct = 0
    agentic_correct = 0
    naive_correct_multihop = 0
    agentic_correct_multihop = 0
    multihop_count = 0

    for i, q in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] {q['id']} ({q['type']}): {q['question']}")

        naive_result = naive_rag_answer(q["question"],k=4)
        naive_score = judge_answer(q["question"], q["ground_truth"], naive_result["answer"])

        agentic_result = agentic_rag_answer(q["question"], max_rounds=3, verbose=False)
        agentic_score = judge_answer(q["question"], q["ground_truth"], agentic_result["answer"])

        print(f"  naive={naive_score}  agentic={agentic_score}")

        naive_correct += naive_score
        agentic_correct += agentic_score

        if q["type"] == "multi-hop":
            multihop_count += 1
            naive_correct_multihop += naive_score
            agentic_correct_multihop += agentic_score

        results.append({
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "ground_truth": q["ground_truth"],
            "naive_answer": naive_result["answer"],
            "naive_score": naive_score,
            "agentic_answer": agentic_result["answer"],
            "agentic_score": agentic_score,
            "agentic_rounds": agentic_result["rounds_used"],
            "agentic_retrieval_log": agentic_result["retrieval_log"],
        })

    total = len(questions)
    naive_pct = naive_correct / total * 100
    agentic_pct = agentic_correct / total * 100
    delta = agentic_pct - naive_pct

    multihop_naive_pct = (naive_correct_multihop / multihop_count * 100) if multihop_count else 0
    multihop_agentic_pct = (agentic_correct_multihop / multihop_count * 100) if multihop_count else 0

    summary = {
        "total_questions": total,
        "naive_score": naive_correct,
        "agentic_score": agentic_correct,
        "naive_pct": round(naive_pct, 1),
        "agentic_pct": round(agentic_pct, 1),
        "delta_pct": round(delta, 1),
        "multihop_count": multihop_count,
        "multihop_naive_pct": round(multihop_naive_pct, 1),
        "multihop_agentic_pct": round(multihop_agentic_pct, 1),
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": results}, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Naive RAG:   {naive_correct}/{total} ({naive_pct:.1f}%)")
    print(f"Agentic RAG: {agentic_correct}/{total} ({agentic_pct:.1f}%)")
    print(f"Delta:       {delta:+.1f} percentage points")
    print(f"\nMulti-hop only ({multihop_count} questions):")
    print(f"  Naive:   {multihop_naive_pct:.1f}%")
    print(f"  Agentic: {multihop_agentic_pct:.1f}%")
    print(f"\nFull results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run_eval()