from huggingface_hub import InferenceClient

REWRITE_PROMPT = """Rewrite the user's question to be specific and unambiguous, \
using context clues about what kind of document is being searched (a research paper). \
Return ONLY the rewritten question, nothing else.

Question: {question}
Rewritten:"""

MULTI_QUERY_PROMPT = """Generate {n} different search queries that would help answer \
the user's question, each focusing on a different angle or phrasing. \
Return ONLY the queries, one per line, no numbering.

Question: {question}
Queries:"""


def rewrite_query(client: InferenceClient, question: str) -> str:
    response = client.chat_completion(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[{"role": "user", "content": REWRITE_PROMPT.format(question=question)}],
        max_tokens=128,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def generate_multi_queries(client: InferenceClient, question: str, n: int = 4) -> list[str]:
    response = client.chat_completion(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[{"role": "user", "content": MULTI_QUERY_PROMPT.format(question=question, n=n)}],
        max_tokens=256,
        temperature=0.3,
    )
    lines = response.choices[0].message.content.strip().split("\n")
    queries = [line.strip("- ").strip() for line in lines if line.strip()]
    return [question] + queries  # keep original as one of the variants