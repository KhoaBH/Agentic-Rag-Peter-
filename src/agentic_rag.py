import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from src.retrieve import retrieve

load_dotenv()

client = OpenAI(
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

LLM_DEPLOYMENT = os.getenv("AZURE_LLM_DEPLOYMENT")

tools = [
    {
        "type": "function",
        "name": "retrieve_docs",
        "description": "Search the knowledge base for relevant text chunks. Call this whenever you need more information to answer the question. You may call it multiple times with different queries if the question requires combining facts from different topics.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"],
            "additionalProperties": False
        },
        "strict": True
    }
]

SYSTEM_PROMPT = """You are a research assistant answering questions using a retrieval tool over a small knowledge base about neural networks and their history.

Rules:
- Use the retrieve_docs tool to search for information before answering.
- You may call it multiple times with different, refined queries if the question needs facts from multiple topics (multi-hop).
- IMPORTANT: You have a MAXIMUM of 3 retrieval calls. After that, you MUST answer using whatever you found.
- If after 2-3 searches you still can't find a specific detail (e.g. an exact workplace), answer with what you DID find, and explicitly state which part you couldn't confirm from the knowledge base. Do not keep retrying indefinitely.
- Only answer based on retrieved content. Be concise."""

def agentic_rag_answer(question, max_rounds=5, verbose=True):
    input_items = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    retrieval_log = []

    for round_num in range(max_rounds):
        response = client.responses.create(
        model=LLM_DEPLOYMENT,
        input=input_items,
        tools=tools,
    )

        function_calls = [item for item in response.output if item.type == "function_call"]

        if verbose:
            print(f"\n--- Round {round_num+1} ---")
            print(f"Function calls this round: {len(function_calls)}")

        if not function_calls:
            return {
                "question": question,
                "answer": response.output_text,
                "rounds_used": round_num + 1,
                "retrieval_log": retrieval_log,
            }

        for fc in function_calls:
            input_items.append({
                "type": "function_call",
                "call_id": fc.call_id,
                "name": fc.name,
                "arguments": fc.arguments,
            })

        for fc in function_calls:
            args = json.loads(fc.arguments)
            query = args["query"]
            results = retrieve(query, k=4)
            retrieval_log.append({"round": round_num + 1, "query": query, "sources": [r["source"] for r in results]})
            context = "\n\n".join(f"[{r['source']}]: {r['text']}" for r in results)

            if verbose:
                print(f"  Round {round_num+1} | query: '{query}' -> sources: {[r['source'] for r in results]}")

            input_items.append({
                "type": "function_call_output",
                "call_id": fc.call_id,
                "output": context,
            })

    # GUARANTEED FINAL ANSWER: force synthesis using everything gathered so far, no tools at all
    input_items.append({
        "role": "user",
        "content": "You have used up your retrieval budget. Based on everything retrieved so far, give your best answer. If you are not certain about a specific detail, say so explicitly rather than guessing."
    })
    final_response = client.responses.create(
        model=LLM_DEPLOYMENT,
        input=input_items,
        # no tools param at all this time -> physically cannot call a tool
    )

    return {
        "question": question,
        "answer": final_response.output_text,
        "rounds_used": max_rounds,
        "retrieval_log": retrieval_log,
    }

if __name__ == "__main__":
    result = agentic_rag_answer("Who developed backpropagation and where did they work?")
    print("\n=== FINAL ===")
    print("Question:", result["question"])
    print("Rounds used:", result["rounds_used"])
    print("Retrieval log:", json.dumps(result["retrieval_log"], indent=2))
    print("\nAnswer:\n", result["answer"])