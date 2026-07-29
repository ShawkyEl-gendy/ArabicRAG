import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from prompts import build_rag_prompt
from sentence_transformers import SentenceTransformer
from prepare_data import load_dataset
from chunking import create_chunk_dataframe
from evaluation import build_chunk_lookup, build_ground_truth, build_dense_retriever,build_chunk_embeddings,load_embedding_model
from dense_retrieval import load_chroma_client, create_collection
from config import (
    DATASET_PATH,
    EMBEDDING_MODEL,
    DB_PATH,
    COLLECTION_NAME,
   
)

from ollama import Client

client = Client(
    host="http://localhost:11434"
)

def build_context(
    reranked_chunks,
    df,
    chunk_lookup,
):
    id_to_index = chunk_lookup["chunk_id_to_index"]

    contexts = []

    for i, (chunk_id, _) in enumerate(reranked_chunks, start=1):

        idx = id_to_index.get(chunk_id)

        if idx is None:
            print(f"Warning: chunk '{chunk_id}' not found.")
            continue

        contexts.append(
            f"[Chunk {i}]\n{df.iloc[idx]['chunk_text']}"
        )

    return "\n\n".join(contexts)


def generate_answer(
    question,
    context,
    model,
):

    prompt = build_rag_prompt(question, context)
    

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
        retrieve_k=3,
    )



    # =====================================================
    # Generation Loop
    # =====================================================
    

    total_queries = len(ground_truth)

    for model in ["aya:8b","llama3:8b","qwen2.5:3b","command-r7b-arabic"]:
        results = []
        print(model ,"Model loading....")
                
        

        for i, (query_id, info) in enumerate(
            ground_truth.items(),
            start=1,
            ):
       

            print(f"[{i}/{total_queries}] Processing Query {query_id}")

            question = info["question"]

            # ---------------------------------------------
            # Dense Retrieval
            # ---------------------------------------------

            retrieved_chunks = dense_retrieve(
                question,
                query_id,
            )
            
            # ---------------------------------------------
            # Build Context
            # ---------------------------------------------

            context = build_context(
                retrieved_chunks,
                df,
                chunk_lookup,
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

        file_name = f"results/MRC/naive_rag/{model}_naive_rag_results.csv".replace(":", "-")
        results_df.to_csv(
        file_name,
        index=False,
        encoding="utf-8-sig",
        )
        print("\nFinished!")
        print(f"Generated {len(results_df)} answers.")