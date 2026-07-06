from src.retrieval.retriever import build_chroma_retriever, load_chroma_retriever, rerank
from src.retrieval.bm25_retriever import build_bm25_retriever, load_bm25_retriever, add_bm25_retriever
from src.retrieval.fusion import reciprocal_rank_fusion


def build_hybrid_index(chunks: list[dict], top_k: int = 5):
    """
    Builds both the dense (Chroma) and sparse (BM25) indexes from the
    same chunks. Returns everything callers need to persist/reuse.
    """
    dense_retriever, vectorstore = build_chroma_retriever(chunks, top_k=top_k)
    bm25_retriever = build_bm25_retriever(chunks, top_k=top_k)
    return dense_retriever, bm25_retriever, vectorstore


def load_hybrid_index(top_k: int = 5):
    """
    Loads both indexes from disk. bm25_retriever is None if nothing's
    been ingested yet.
    """
    dense_retriever, vectorstore = load_chroma_retriever(top_k=top_k)
    bm25_retriever = load_bm25_retriever(top_k=top_k)
    return dense_retriever, bm25_retriever, vectorstore


def add_to_hybrid_index(new_chunks: list[dict], vectorstore, top_k: int = 5):
    """
    Adds new chunks to both indexes. Chroma supports incremental add;
    BM25 rebuilds from the merged chunk set. Returns the refreshed
    bm25_retriever (dense vectorstore is mutated in place).
    """
    from src.retrieval.retriever import add_chroma_retriever
    add_chroma_retriever(new_chunks, vectorstore)
    return add_bm25_retriever(new_chunks, top_k=top_k)


def hybrid_search(query: str, dense_retriever, bm25_retriever, top_k: int = 5) -> list:
    """
    Runs dense + sparse retrieval, fuses via RRF, then reranks with the
    cross-encoder. This is the single call sites should use.
    """
    dense_docs = dense_retriever.invoke(query)
    sparse_docs = bm25_retriever.invoke(query) if bm25_retriever else []

    fused = reciprocal_rank_fusion(dense_docs, sparse_docs)
    return rerank(query, fused, top_k=top_k)


def multi_query_hybrid_search(queries: list[str], dense_retriever, bm25_retriever, top_k: int = 5) -> list:
    """
    Runs the dense+sparse+RRF stage per query variant (e.g. from
    generate_multi_queries), fuses all result lists via a second RRF
    pass, then reranks once against the original question (queries[0]).
    """
    per_query_fused = []
    for q in queries:
        dense_docs = dense_retriever.invoke(q)
        sparse_docs = bm25_retriever.invoke(q) if bm25_retriever else []
        per_query_fused.append(reciprocal_rank_fusion(dense_docs, sparse_docs))

    fused = reciprocal_rank_fusion(*per_query_fused)
    return rerank(queries[0], fused, top_k=top_k)


def multi_paper_search(query: str, dense_retriever, bm25_retriever, sources: list[str], top_k_per_source: int = 3) -> list:
    """
    For cross-paper questions ("compare X across all papers"). Runs
    hybrid_search once per source filename and concatenates results,
    so every paper gets guaranteed representation instead of the top-k
    across all papers being dominated by one paper's chunks.

    BM25Retriever has no native metadata filter, so sparse results are
    filtered post-hoc by source before fusion.
    """
    all_docs = []
    for source in sources:
        dense_docs = dense_retriever.vectorstore.similarity_search(
            query, k=top_k_per_source * 4, filter={"source": source}
        ) if hasattr(dense_retriever, "vectorstore") else dense_retriever.invoke(query)

        sparse_raw = bm25_retriever.invoke(query) if bm25_retriever else []
        sparse_docs = [d for d in sparse_raw if d.metadata.get("source") == source]

        fused = reciprocal_rank_fusion(dense_docs, sparse_docs)
        all_docs.extend(rerank(query, fused, top_k=top_k_per_source))

    return all_docs