from app.retrieval.dense import search
from app.llm_clients import co, generate

def retrieve_and_rerank(query, num_candidates=10, top_k=3):
    candidates_df = search(query, number_of_results=num_candidates)
    candidate_docs = candidates_df['texts'].tolist()
    reranked = co.rerank(
        query=query,
        documents=candidate_docs,
        top_n=top_k,
        return_documents=True
    )
    return [hit.document.text for hit in reranked.results]

def rag_answer(query, num_candidates=10, top_k=3, max_new_tokens=250):
    top_chunks = retrieve_and_rerank(query, num_candidates, top_k)
    context = "\n".join(top_chunks)

    rag_instruction = (
        "Answer the question using only the information in the context below. "
        "If the answer isn't in the context, say you don't know — don't make anything up.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}"
    )
    messages = [{"role": "user", "content": rag_instruction}]
    answer = generate(messages, max_new_tokens=max_new_tokens)
    return answer.content, top_chunks