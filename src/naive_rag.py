from src.retrieve import retrieve
from src.llm import ask_llm

def naive_rag_answer(question, k=4):
    # 1. Retrieve once, blindly
    results = retrieve(question, k=k)

    # 2. Stuff into context
    context = "\n\n".join(
        f"[Source: {r['source']}]\n{r['text']}" for r in results
    )

    # 3. Ask the LLM using only that context
    prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""

    answer = ask_llm(prompt)
    return {
        "question": question,
        "answer": answer,
        "retrieved_sources": [r["source"] for r in results],
    }

if __name__ == "__main__":
    result = naive_rag_answer("Who developed backpropagation and where did they work?")
    print("Question:", result["question"])
    print("Sources used:", result["retrieved_sources"])
    print("\nAnswer:\n", result["answer"])