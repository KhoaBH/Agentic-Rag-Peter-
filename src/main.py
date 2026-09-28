
from dotenv import load_dotenv

load_dotenv()  # load .env before the src modules create their clients

from src.agentic_rag import agentic_rag_answer  # adjust module name if yours differs
from src.naive_rag import naive_rag_answer

HELP = """Commands:
  /mode agentic   use the agentic pipeline (default)
  /mode naive     use the naive pipeline
  /mode both      run both and compare
  /help           show this help
  /quit           exit
"""


def run_agentic(question):
    result = agentic_rag_answer(question, verbose=True)
    print("\n=== AGENTIC ANSWER ===")
    print(result["answer"])
    print(f"\n(rounds used: {result['rounds_used']}, retrievals: {len(result['retrieval_log'])})")
    for entry in result["retrieval_log"]:
        print(f"  round {entry['round']}: '{entry['query']}' -> {entry['sources']}")


def run_naive(question):
    result = naive_rag_answer(question, k=4)
    answer = result["answer"] if isinstance(result, dict) else result
    print("\n=== NAIVE ANSWER ===")
    print(answer)
    if isinstance(result, dict) and "sources" in result:
        print(f"\n(sources: {result['sources']})")


def main():
    mode = "agentic"
    print("RAG assistant. Type a question, or /help for commands.")

    while True:
        try:
            question = input(f"\n[{mode}] Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question:
            continue
        if question.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Bye.")
            break
        if question.lower() == "/help":
            print(HELP)
            continue
        if question.lower().startswith("/mode"):
            parts = question.split()
            if len(parts) == 2 and parts[1] in ("agentic", "naive", "both"):
                mode = parts[1]
                print(f"Mode set to {mode}.")
            else:
                print("Usage: /mode agentic | naive | both")
            continue

        try:
            if mode in ("naive", "both"):
                run_naive(question)
            if mode in ("agentic", "both"):
                run_agentic(question)
        except Exception as e:
            print(f"\nError: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()