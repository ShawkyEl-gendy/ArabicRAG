import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from prompts import build_enhanced_rag_prompt
from evaluation import build_cross_encoder_reranker, build_rerank_retriever,build_ground_truth,build_chunk_lookup,build_sparse_retriever,build_sparse_index , build_dense_retriever, build_weighted_hybrid_retriever,build_chunk_embeddings,load_embedding_model
from sentence_transformers import SentenceTransformer
from prepare_data import load_dataset
from chunking import create_chunk_dataframe
from ner import add_ner_to_text, generate_ner, load_ner_model
from dense_retrieval import load_chroma_client, create_collection
from sparse_retrieval import build_sparse_index
from config import (
    DATASET_PATH,
    CHUNK_STRATEGY,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_SIZE,
    SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL,
    DB_PATH,
    COLLECTION_NAME,
    USE_NER,
    MODEL_NAME
)

from ollama import Client

client = Client(
    host="http://localhost:11434"
)

def build_context(
    reranked_chunks,
    df,
    chunk_lookup,
    ner_model=None,
    use_ner=USE_NER,
):
    """
    Build the context from retrieved chunks.

    Parameters
    ----------
    reranked_chunks : list[(chunk_id, score)]
    df : pandas.DataFrame
    chunk_lookup : dict
    ner_model : optional
        Loaded NER model.
    use_ner : bool
        Whether to prepend NER metadata to each chunk.

    Returns
    -------
    str
    """

    id_to_index = chunk_lookup["chunk_id_to_index"]

    contexts = []

    for i, (chunk_id, _) in enumerate(reranked_chunks, start=1):

        idx = id_to_index.get(chunk_id)

        if idx is None:
            print(f"Warning: chunk '{chunk_id}' not found.")
            continue

        chunk_text = df.iloc[idx]["chunk_text"]

        if use_ner:

            ner = generate_ner(
                text=chunk_text,
                ner_model=ner_model,
            )

            chunk_text = add_ner_to_text(
                text=chunk_text,
                ner=ner,
                use_ner=True,
            )

        contexts.append(
            f"[Chunk {i}]\n{chunk_text}"
        )

    return "\n\n".join(contexts)


def generate_answer(
    question,
    context,
    model=MODEL_NAME,
):

    prompt = build_enhanced_rag_prompt(question, context)
    

    llm = ChatOllama(
        model=model,
        base_url="http://localhost:11434",
        temperature=0,
        num_ctx=4096,
        # format="json", # Uncomment ONLY if your prompt explicitly asks for a specific JSON schema
    )

    response = llm.invoke([
        HumanMessage(content=prompt)
    ])

    return response.content




if __name__ == "__main__":

    # =====================================================
    # Load and prepare dataset
    # =====================================================

    df = load_dataset(DATASET_PATH)

    df = create_chunk_dataframe(df)

    # =====================================================
    # Build lookup tables
    # =====================================================

    embedding_model = load_embedding_model()

    chunk_embeddings = build_chunk_embeddings(
        df,
        embedding_model,
    )

    ground_truth = build_ground_truth(
        df=df,
        embedding_model=embedding_model,
        chunk_embeddings=chunk_embeddings,
        similarity_threshold=0.90,
        min_relevant_chunks=1,
    )

    chunk_lookup = build_chunk_lookup(df)

    # =====================================================
    # Sparse retrieval
    # =====================================================

    _, chunk_vectors, query_vectors = build_sparse_index(df)

    sparse_retrieve = build_sparse_retriever(
        chunk_vectors=chunk_vectors,
        query_vectors=query_vectors,
        chunk_lookup=chunk_lookup,
    )

    # =====================================================
    # Dense retrieval
    # =====================================================

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cuda",
    )

    client = load_chroma_client(DB_PATH)

    collection = create_collection(
        client=client,
        collection_name=COLLECTION_NAME,
    )

    dense_retrieve = build_dense_retriever(
        collection=collection,
        embedding_model=embedding_model,
    )

    # =====================================================
    # Hybrid Retrieval weighted sum
    # =====================================================

    weighted_retrieve = build_weighted_hybrid_retriever(
        dense_retrieve=dense_retrieve,
        sparse_retrieve=sparse_retrieve,
        alpha=0.5,
        retrieve_k=50,
    )


    # =====================================================
    # Cross-Encoder Reranker
    # =====================================================

    reranker = build_cross_encoder_reranker()

    retrieve = build_rerank_retriever(
        base_retriever=weighted_retrieve,
        reranker=reranker,
        chunk_lookup=chunk_lookup,
        retrieve_k=50,
        final_k=3,
    )

    # =====================================================
    # Generation Loop
    # =====================================================

    
    ner_model = load_ner_model()
    total_queries = len(ground_truth)
    for model in ["aya:8b","llama3:8b","qwen2.5:3b","command-r7b-arabic"]:
        results = []
        print(model ,"Model loading....")
        
        for i, (query_id, info) in enumerate(ground_truth.items(), start=1):


            print(f"[{i}/{total_queries}] Processing Query {query_id}")

            question = info["question"]

            # ---------------------------------------------
            # Retrieve Top Chunks
            # ---------------------------------------------

            reranked_chunks = retrieve(
                question,
                query_id,
            )

            # ---------------------------------------------
            # Build Context
            # ---------------------------------------------


            context = build_context(
                reranked_chunks,
                df,
                chunk_lookup,
                ner_model=ner_model,
                use_ner=USE_NER
            )


            # ---------------------------------------------
            # Generate Answer
            # ---------------------------------------------

            response = generate_answer(
                question,
                context,
                model=model,
            )

            # ---------------------------------------------
            # Save Result
            # ---------------------------------------------

            results.append(
                {
                    "retrieved_context": context,
                    "question": question,
                    "reference": info["answer"],
                    "response": response,
        
                }
            )

        # =====================================================
        # Save Results
        # =====================================================

        results_df = pd.DataFrame(results)

        file_name = f"results/MRC/enhanced_rag/{model}_naive_rag_results.csv".replace(":", "-")
        results_df.to_csv(
        file_name,
        index=False,
        encoding="utf-8-sig",
        )
        print("\nFinished!")
        print(f"Generated {len(results_df)} answers.")