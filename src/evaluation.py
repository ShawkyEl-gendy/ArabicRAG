import numpy as np
from collections import defaultdict
from ranx import Qrels, Run, evaluate
from FlagEmbedding import FlagReranker
from sentence_transformers import SentenceTransformer
from dense_retrieval import load_embedding_model,index_chunks, create_embeddings
from prepare_data import load_dataset
from chunking import create_chunk_dataframe
from dense_retrieval import load_chroma_client, create_collection
from sparse_retrieval import build_sparse_index, sparse_similarity
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
    
)





def cosine_similarity(a, b):
    """
    Compute cosine similarity between two normalized embeddings.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    )





def build_chunk_embeddings(df, embedding_model):
    """
    Compute embeddings for every chunk once.

    Returns
    -------
    dict
        {
            "doc_0_0": embedding,
            "doc_0_1": embedding,
            ...
        }
    """

    texts = df["chunk_text"].tolist()
    chunk_ids = df["chunk_id"].tolist()

    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True,
    )

    return {
        chunk_id: embedding
        for chunk_id, embedding in zip(chunk_ids, embeddings)
    }



def cosine_similarity(a, b):
    """
    Compute cosine similarity.
    """
    return np.dot(a, b) / (
        np.linalg.norm(a) *
        np.linalg.norm(b) +
        1e-12
    )


def token_overlap(answer, chunk):
    """
    Compute token overlap between the answer and a chunk.

    Score = (# common tokens) / (# answer tokens)

    Returns
    -------
    float
        Value in [0, 1].
    """

    answer_tokens = set(answer.split())
    chunk_tokens = set(chunk.split())

    if len(answer_tokens) == 0:
        return 0.0

    return len(answer_tokens & chunk_tokens) / len(answer_tokens)



def build_ground_truth(
    df,
    embedding_model,
    chunk_embeddings,
    similarity_threshold=0.80,
    min_relevant_chunks=2,
):
    """
    Build hybrid ground truth using:

        1. Embedding similarity.
        2. Token overlap.

    Strategy
    --------
    1. Keep all chunks whose embedding similarity >= threshold.
    2. Add ONLY the chunk with the highest token overlap.
    3. If no chunk is selected, keep the top-N chunks by similarity.
    """

    # -----------------------------------------------------
    # document_id -> chunks
    # -----------------------------------------------------

    document_chunks = defaultdict(list)

    for row in df.itertuples(index=False):

        document_chunks[row.document_id].append(
            {
                "chunk_id": row.chunk_id,
                "chunk_text": row.chunk_text,
            }
        )

    # -----------------------------------------------------
    # Unique queries
    # -----------------------------------------------------

    queries = (
        df[
            [
                "query_id",
                "question",
                "answer",
                "document_id",
            ]
        ]
        .drop_duplicates("query_id")
        .sort_values("query_id")
    )

    ground_truth = {}

    # -----------------------------------------------------
    # Build qrels
    # -----------------------------------------------------

    for row in queries.itertuples(index=False):

        answer_embedding = embedding_model.encode(row.answer)

        candidates = []

        for chunk in document_chunks[row.document_id]:

            similarity = cosine_similarity(
                answer_embedding,
                chunk_embeddings[chunk["chunk_id"]],
            )

            overlap = token_overlap(
                row.answer,
                chunk["chunk_text"],
            )

            candidates.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "similarity": similarity,
                    "overlap": overlap,
                }
            )

        # -------------------------------------------------
        # Select by embedding similarity
        # -------------------------------------------------

        relevant_chunks = {

            c["chunk_id"]

            for c in candidates

            if c["similarity"] >= similarity_threshold

        }

        # -------------------------------------------------
        # Add the best overlap chunk
        # -------------------------------------------------

        best_overlap = max(
            candidates,
            key=lambda x: x["overlap"],
        )

        relevant_chunks.add(
            best_overlap["chunk_id"]
        )

        # -------------------------------------------------
        # Fallback
        # -------------------------------------------------

        if len(relevant_chunks) == 0:

            candidates.sort(
                key=lambda x: x["similarity"],
                reverse=True,
            )

            relevant_chunks = {

                c["chunk_id"]

                for c in candidates[:min_relevant_chunks]

            }

        # -------------------------------------------------

        ground_truth[row.query_id] = {

            "question": row.question,

            "answer": row.answer,

            "relevant_chunks": relevant_chunks,

        }

    return ground_truth




def show_relevant_chunks(df, ground_truth, query_id):
    """
    Display the question, answer, and the text of all relevant chunks.

    Parameters
    ----------
    df : pd.DataFrame

    ground_truth : dict

    query_id : int
    """

    gt = ground_truth[query_id]

    print("=" * 100)
    print(f"Query ID : {query_id}")
    print(f"Question : {gt['question']}")
    print(f"Answer   : {gt['answer']}")
    print("=" * 100)

    relevant_chunks = gt["relevant_chunks"]

    rows = (
        df[df["chunk_id"].isin(relevant_chunks)]
        .sort_values("chunk_id")
    )

    for row in rows.itertuples(index=False):

        print(f"\nChunk ID : {row.chunk_id}")
        print("-" * 100)
        print(row.chunk_text)






def build_chunk_lookup(df):
    """
    Build lookup tables for chunk indexing.

    Parameters
    ----------
    df : pandas.DataFrame

    Required columns
    ----------------
    chunk_id

    Returns
    -------
    dict

    {
        "index_to_chunk_id": list,
        "chunk_id_to_index": dict,
        chunk_id_to_text: dict
    }


    Like:

    chunk_lookup = {
        "index_to_chunk_id": [
            "doc_0_0",
            "doc_0_1",
            "doc_0_2",
            "doc_0_3",
            "doc_0_4",
            "doc_1_0",
            ...
        ],

        "chunk_id_to_index": {
            "doc_0_0": 0,
            "doc_0_1": 1,
            "doc_0_2": 2,
            "doc_0_3": 3,
            "doc_0_4": 4,
            "doc_1_0": 5,
            ...
        }

        "chunk_id_to_text": {
            "doc_0_0": "Chunk text for doc_0_0",
            "doc_0_1": "Chunk text for doc_0_1",
            "doc_0_2": "Chunk text for doc_0_2",
            "doc_0_3": "Chunk text for doc_0_3",
            "doc_0_4": "Chunk text for doc_0_4",
            "doc_1_0": "Chunk text for doc_1_0",
            ...
        }
    }




    Notes
    -----
    - index_to_chunk_id[i] -> chunk_id
    - chunk_id_to_index[chunk_id] -> row index
    - chunk_id_to_text[chunk_id] -> chunk_text
    """

    index_to_chunk_id = df["chunk_id"].tolist()

    chunk_id_to_index = {
        chunk_id: idx
        for idx, chunk_id in enumerate(index_to_chunk_id)
    }

    chunk_id_to_text = {
    row.chunk_id: row.chunk_text
    for row in df.itertuples(index=False)
    }

    return {
        "index_to_chunk_id": index_to_chunk_id,
        "chunk_id_to_index": chunk_id_to_index,
        "chunk_id_to_text": chunk_id_to_text,
    }




def evaluate_retrieval(
    ground_truth,
    retrieve_fn,
):
    """
    Evaluate any retrieval method using ranx.

    Parameters
    ----------
    ground_truth : dict
        Output of build_ground_truth().

    retrieve_fn : callable

        Signature
        ---------
        retrieve_fn(question, query_id)

        Returns
        -------
        [
            (chunk_id, score),
            ...
        ]

    Returns
    -------
    dict
        recall@5
        recall@10
        mrr
        ndcg@10
    """

    # ----------------------------------------
    # Build Qrels
    # ----------------------------------------

    qrels = {}

    for query_id, info in ground_truth.items():

        qrels[f"q_{query_id}"] = {

            chunk_id: 1

            for chunk_id in info["relevant_chunks"]

        }

    qrels = Qrels(qrels)


    # ----------------------------------------
    # Build Run
    # ----------------------------------------

    run = {}

    for query_id, info in ground_truth.items():

        retrieved = retrieve_fn(
            info["question"],
            query_id
        )[:50]

        run[f"q_{query_id}"] = {

            chunk_id: float(score)

            for chunk_id, score in retrieved

        }

    run = Run(run)


    # ----------------------------------------
    # Evaluate
    # ----------------------------------------

    metrics = evaluate(
        qrels=qrels,
        run=run,
        metrics=[
            "recall@5",
            "recall@10",
            "mrr",
            "ndcg@10",
        ],
    )

    return metrics




def build_dense_retriever(
    collection,
    embedding_model,
    retrieve_k=50,
):
    """
    Build a dense retriever.

    Parameters
    ----------
    collection : chromadb.Collection

    embedding_model : SentenceTransformer

    retrieve_k : int
        Number of chunks to retrieve from Chroma.

    Returns
    -------
    retrieve(question, query_id)

    Returns
    -------
    List[(chunk_id, score)]

    Example
    -------
    [
        ("doc_0_1", 0.91),
        ("doc_5_0", 0.82),
        ("doc_0_2", 0.80),
    ]
    """

    def retrieve(question, query_id=None):

        # Encode the query
        query_embedding = embedding_model.encode(
            "query: " + question,
            convert_to_numpy=True,
        )

        # Search Chroma
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=retrieve_k,
            include=[
                "metadatas",
                "distances",
            ],
        )

        retrieved = []

        for metadata, distance in zip(
            results["metadatas"][0],
            results["distances"][0],
        ):

            chunk_id = metadata["chunk_id"]

            # Convert cosine distance to similarity
            score = 1.0 - distance

            retrieved.append(
                (
                    chunk_id,
                    float(score),
                )
            )

        return retrieved
    
    return retrieve




def build_sparse_retriever(
    chunk_vectors,
    query_vectors,
    chunk_lookup,
):
    """
    Build sparse retriever using BGE-M3 lexical vectors.
    """

    index_to_chunk_id = chunk_lookup["index_to_chunk_id"]

    def retrieve(question, query_id):

        if query_id not in query_vectors:

            raise KeyError(
                f"query_id={query_id} was not encoded."
            )

        query_vector = query_vectors[query_id]

        scores = np.array([
            sparse_similarity(
                query_vector,
                chunk_vector,
            )
            for chunk_vector in chunk_vectors
        ])

        ranking = np.argsort(scores)[::-1]

        retrieved = [

            (
                index_to_chunk_id[idx],
                float(scores[idx]),
            )

            for idx in ranking

        ]

        return retrieved

    return retrieve



############################################################################

def normalize_scores(results):
    """
    Min-Max normalize retrieval scores.

    Parameters
    ----------
    results : list[(chunk_id, score)]

    Returns
    -------
    list[(chunk_id, normalized_score)]
    """

    if len(results) == 0:
        return results

    scores = [score for _, score in results]

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [(cid, 1.0) for cid, _ in results]

    return [
        (
            cid,
            (score - min_score) / (max_score - min_score)
        )
        for cid, score in results
    ]


def build_weighted_hybrid_retriever(
    dense_retrieve,
    sparse_retrieve,
    alpha=0.5,
    retrieve_k=50,
):
    """
    Weighted score fusion.

    HybridScore =
        alpha * dense +
        (1-alpha) * sparse
    """

    def retrieve(question, query_id):

        dense_results = normalize_scores(
            dense_retrieve(question, query_id)
        )

        sparse_results = normalize_scores(
            sparse_retrieve(question, query_id)
        )

        scores = {}

        # -----------------------
        # Dense
        # -----------------------

        for chunk_id, score in dense_results:

            scores[chunk_id] = (
                scores.get(chunk_id, 0.0)
                + alpha * score
            )

        # -----------------------
        # Sparse
        # -----------------------

        for chunk_id, score in sparse_results:

            scores[chunk_id] = (
                scores.get(chunk_id, 0.0)
                + (1 - alpha) * score
            )

        ranking = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return ranking[:retrieve_k]

    return retrieve



def build_rrf_retriever(
    dense_retrieve,
    sparse_retrieve,
    retrieve_k=50,
    rrf_k=60,
):
    """
    Reciprocal Rank Fusion (RRF).

    Parameters
    ----------
    dense_retrieve : callable

    sparse_retrieve : callable

    retrieve_k : int
        Number of final chunks to return.

    rrf_k : int
        RRF constant.
        Literature commonly uses 60.

    Returns
    -------
    retrieve(question, query_id)
    """

    def retrieve(question, query_id):

        dense_results = dense_retrieve(
            question,
            query_id,
        )[:retrieve_k]

        sparse_results = sparse_retrieve(
            question,
            query_id,
        )[:retrieve_k]

        scores = {}

        # -----------------------------
        # Dense ranking
        # -----------------------------

        for rank, (chunk_id, _) in enumerate(
            dense_results,
            start=1,
        ):

            scores[chunk_id] = (
                scores.get(chunk_id, 0.0)
                + 1.0 / (rrf_k + rank)
            )

        # -----------------------------
        # Sparse ranking
        # -----------------------------

        for rank, (chunk_id, _) in enumerate(
            sparse_results,
            start=1,
        ):

            scores[chunk_id] = (
                scores.get(chunk_id, 0.0)
                + 1.0 / (rrf_k + rank)
            )

        ranking = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return ranking[:retrieve_k]

    return retrieve




def build_cross_encoder_reranker(
    model_name="BAAI/bge-reranker-v2-m3",
    use_fp16=True,
):
    """
    Build a BGE Cross-Encoder reranker.

    Parameters
    ----------
    model_name : str

    use_fp16 : bool

    Returns
    -------
    FlagReranker
    """

    print("Loading Cross-Encoder Reranker...")

    reranker = FlagReranker(
        model_name,
        use_fp16=use_fp16,
    )

    return reranker



def build_rerank_retriever(
    base_retriever,
    reranker,
    chunk_lookup,
    retrieve_k=50,
    final_k=10,
):
    """
    Build a retrieval pipeline with Cross-Encoder reranking.

    Pipeline
    --------
    Base Retriever
            │
            ▼
       Top retrieve_k
            │
            ▼
      Cross-Encoder
            │
            ▼
        Top final_k

    Parameters
    ----------
    base_retriever : callable
        Dense, Sparse, Weighted or RRF retriever.

    reranker : FlagReranker

    chunk_lookup : dict

    retrieve_k : int
        Number of retrieved chunks before reranking.

    final_k : int
        Number of reranked chunks returned.

    Returns
    -------
    retrieve(question, query_id)

    Returns
    -------
    [
        (chunk_id, rerank_score),
        ...
    ]
    """

    chunk_id_to_text = chunk_lookup["chunk_id_to_text"]

    def retrieve(question, query_id):

        # ------------------------------------
        # Retrieve candidate chunks
        # ------------------------------------

        retrieved = base_retriever(
            question,
            query_id,
        )[:retrieve_k]

        if len(retrieved) == 0:
            return []

        # ------------------------------------
        # Build query-document pairs
        # ------------------------------------

        pairs = []
        chunk_ids = []

        for chunk_id, _ in retrieved:

            chunk_text = chunk_id_to_text.get(chunk_id)

            if chunk_text is None:
                continue

            chunk_ids.append(chunk_id)

            pairs.append(
                [question, chunk_text]
            )

        if len(pairs) == 0:
            return []

        # ------------------------------------
        # Cross-Encoder scores
        # ------------------------------------

        scores = reranker.compute_score(
            pairs
        )

        reranked = sorted(
            zip(chunk_ids, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return reranked[:final_k]

    return retrieve




def build_rerank_retriever(
    base_retriever,
    reranker,
    chunk_lookup,
    retrieve_k=50,
    final_k=10,
):
    """
    Build a retrieval pipeline with Cross-Encoder reranking.

    Pipeline
    --------
    Base Retriever
            │
            ▼
       Top retrieve_k
            │
            ▼
      Cross-Encoder
            │
            ▼
        Top final_k

    Parameters
    ----------
    base_retriever : callable
        Dense, Sparse, Weighted or RRF retriever.

    reranker : FlagReranker

    chunk_lookup : dict

    retrieve_k : int
        Number of retrieved chunks before reranking.

    final_k : int
        Number of reranked chunks returned.

    Returns
    -------
    retrieve(question, query_id)

    Returns
    -------
    [
        (chunk_id, rerank_score),
        ...
    ]
    """

    chunk_id_to_text = chunk_lookup["chunk_id_to_text"]



    def retrieve(question, query_id):

        # ------------------------------------
        # Retrieve candidate chunks
        # ------------------------------------

        retrieved = base_retriever(
            question,
            query_id,
        )[:retrieve_k]

        if len(retrieved) == 0:
            return []

        # ------------------------------------
        # Build query-document pairs
        # ------------------------------------

        pairs = []
        chunk_ids = []

        for chunk_id, _ in retrieved:
            

            chunk_text = chunk_id_to_text.get(chunk_id)

            if chunk_text is None:
                continue

            chunk_ids.append(chunk_id)

            pairs.append(
                [question, chunk_text]
            )

        if len(pairs) == 0:
            return []

        # ------------------------------------
        # Cross-Encoder scores
        # ------------------------------------

        scores = reranker.compute_score(
            pairs
        )

        reranked = sorted(
            zip(chunk_ids, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return reranked[:final_k]
    

    return retrieve



######################################################################################




if __name__ == "__main__":
    

    # Load dataset
    df = load_dataset(DATASET_PATH)
    # Create chunks
    df = create_chunk_dataframe(df)
    

################################## evaluate sparse retrieval ################################################

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

    show_relevant_chunks(df, ground_truth, 0)
    show_relevant_chunks(df, ground_truth, 9)
    show_relevant_chunks(df, ground_truth, 156)
    show_relevant_chunks(df, ground_truth, 200)
    show_relevant_chunks(df, ground_truth, 336)
    show_relevant_chunks(df, ground_truth, 550)
    show_relevant_chunks(df, ground_truth, 856)
    show_relevant_chunks(df, ground_truth, 992)
    show_relevant_chunks(df, ground_truth,1000)
    chunk_lookup = build_chunk_lookup(df)
    print("Chunk Lookup:", chunk_lookup["index_to_chunk_id"][0])
    print("Chunk Lookup:", chunk_lookup["chunk_id_to_text"]["doc_0_0"])
    print("Chunk Lookup:", chunk_lookup["chunk_id_to_index"]["doc_0_0"])
    model, chunk_vectors, query_vectors = build_sparse_index(df)

        

    sparse = build_sparse_retriever(
        chunk_vectors,
        query_vectors,
        chunk_lookup,
    )
 
 

    sparse_metrics = evaluate_retrieval(
        ground_truth,
        sparse,
    )

    
    
    print("Sparse Metrics:", sparse_metrics)




################################### evaluate dense retrieval ################################################

         
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cuda"      # or "cpu"
    )

    client = load_chroma_client(DB_PATH)
    collection = create_collection(
        client=client,
        collection_name=COLLECTION_NAME,
    )


    dense = build_dense_retriever(
        collection,
        embedding_model
    )

    dense_metrics = evaluate_retrieval(
        ground_truth,
        dense,
    )
    

    print("Dense Metrics:", dense_metrics)

################################### hybrid retrieval with weighted fusion ################################################


    weighted_retrieve = build_weighted_hybrid_retriever(
        dense_retrieve=dense,
        sparse_retrieve=sparse,
        alpha=0.5,
        retrieve_k=50,
    )


    hybrid_weighted_metrics = evaluate_retrieval(
        ground_truth,
        weighted_retrieve,
    )
    

    print("Hybrid Metrics using Weighted Fusion:", hybrid_weighted_metrics)

################################### hybrid retrieval with reciprocal ranking fusion ################################################




    rrf_retrieve = build_rrf_retriever(
        dense_retrieve=dense,
        sparse_retrieve=sparse,
        retrieve_k=50,
        rrf_k=60,
    )


    hybrid_rrf_metrics = evaluate_retrieval(
        ground_truth,
        rrf_retrieve,
    )


    print("Hybrid Metrics using RRF:", hybrid_rrf_metrics)




################################### Weighted Fusion with cross-encoder reranking ################################################



    Weighted_with_reranker = build_cross_encoder_reranker()

    Weighted_withrerank_retrieve = build_rerank_retriever(
        base_retriever=weighted_retrieve,
        reranker=Weighted_with_reranker,
        chunk_lookup=chunk_lookup,
        retrieve_k=50,
        final_k=10,
    )


    Weighted_withrerank_metrics = evaluate_retrieval(
    ground_truth,
    Weighted_withrerank_retrieve,
    )


    print("Weighted Fusion With Cross-Encoder Rerank Metrics:", Weighted_withrerank_metrics)






