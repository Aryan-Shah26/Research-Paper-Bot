import re
from huggingface_hub import InferenceClient

VERIFY_PROMPT = """Does the CONTEXT support the CLAIM? Answer only "yes" or "no".

CONTEXT: {context}
CLAIM: {claim}
Answer:"""

CITATION_PATTERN = re.compile(r"\[([^\[\]]+?)\s*-\s*Page\s*(\d+)\]")


def extract_cited_claims(answer: str) -> list[dict]:
    """
    Splits answer into sentences and pairs each with any inline
    citation e.g. "[paper.pdf - Page 4]" found in that sentence.
    """
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    claims = []
    for sentence in sentences:
        match = CITATION_PATTERN.search(sentence)
        if match:
            claims.append({
                "text": sentence,
                "source": match.group(1).strip(),
                "page": int(match.group(2)),
            })
    return claims


def verify_citations(client: InferenceClient, claims: list[dict], retrieved_docs: list) -> list[dict]:
    """
    For each claim, finds the matching source doc(s) and asks the LLM
    whether the claim is actually supported. Returns claims annotated
    with "verified": bool. Claims with no matching doc are marked False.
    """
    results = []
    for claim in claims:
        matching_docs = [
            doc for doc in retrieved_docs
            if doc.metadata.get("source") == claim["source"] and doc.metadata.get("page") == claim["page"]
        ]
        if not matching_docs:
            results.append({**claim, "verified": False})
            continue

        context = "\n\n".join(doc.page_content for doc in matching_docs)
        response = client.chat_completion(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=[{"role": "user", "content": VERIFY_PROMPT.format(context=context, claim=claim["text"])}],
            max_tokens=5,
            temperature=0.0,
        )
        verdict = response.choices[0].message.content.strip().lower().startswith("yes")
        results.append({**claim, "verified": verdict})
    return results


def filter_hallucinated_citations(answer: str, verified_claims: list[dict]) -> str:
    """
    Strips the citation tag (but keeps the sentence) for any claim that
    failed verification, so the answer no longer implies false support.
    """
    for claim in verified_claims:
        if not claim["verified"]:
            tag = f"[{claim['source']} - Page {claim['page']}]"
            answer = answer.replace(tag, "[unverified]")
    return answer