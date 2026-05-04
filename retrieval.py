from langchain_chroma import Chroma

from config import CHROMA_DIR, COLLECTION_NAME, UNIXCODER_MODEL_NAME, TOP_K
from embeddings_unixcoder import UniXcoderEmbeddings


def load_vectorstore():
    embedding_fn = UniXcoderEmbeddings(model_name=UNIXCODER_MODEL_NAME)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        persist_directory=str(CHROMA_DIR),
    )
    return vectorstore


def retrieve_similar_code(query_code: str, k: int = TOP_K):
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search_with_relevance_scores(query_code, k=k)
    return results


if __name__ == "__main__":
    test_query = "cursor.execute(f\"SELECT * FROM users WHERE username = '{username}'\")"
    results = retrieve_similar_code(test_query, k=3)

    for i, (doc, score) in enumerate(results, start=1):
        print(f"\n=== Result {i} | score={score:.4f} ===")
        print(doc.metadata)
        print(doc.page_content[:500])