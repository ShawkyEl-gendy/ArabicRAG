import numpy as np
from FlagEmbedding import BGEM3FlagModel
from prepare_data import load_dataset, remove_duplicate_documents, clean_dataset
from chunking import create_chunk_dataframe





def sparse_similarity(query_vec, doc_vec):
    """
    Compute sparse dot-product similarity between
    two BGE-M3 lexical weight dictionaries.
    """
    score = 0.0

    # iterate over the smaller dictionary
    if len(query_vec) > len(doc_vec):
        query_vec, doc_vec = doc_vec, query_vec

    for token, weight in query_vec.items():
        if token in doc_vec:
            score += weight * doc_vec[token]

    return score




def build_sparse_index(
    df,
    model_name="BAAI/bge-m3",
    batch_size=32,
):
    """
    Build BGE-M3 sparse representations for chunks and queries.

    Returns
    -------
    model

    chunk_vectors : list
        chunk_vectors[i] corresponds to df.iloc[i]

    query_vectors : dict
        {
            query_id: sparse_vector
        }
    """

    print("Loading BGE-M3...")

    model = BGEM3FlagModel(
        model_name,
        use_fp16=True,
    )

    # --------------------------------------------------
    # Encode chunks
    # --------------------------------------------------

    print("Encoding chunks...")

    chunk_output = model.encode(
        df["chunk_text"].tolist(),
        batch_size=batch_size,
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    chunk_vectors = chunk_output["lexical_weights"]

    # --------------------------------------------------
    # Encode unique queries
    # --------------------------------------------------

    queries = (
        df[
            [
                "query_id",
                "question",
            ]
        ]
        .drop_duplicates("query_id")
        .sort_values("query_id")
    )

    query_output = model.encode(
        queries["question"].tolist(),
        batch_size=batch_size,
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    # --------------------------------------------------
    # Build lookup
    # --------------------------------------------------

    query_vectors = {

        query_id: vector

        for query_id, vector in zip(
            queries["query_id"],
            query_output["lexical_weights"],
        )

    }

    print(f"Encoded {len(query_vectors)} unique queries.")

    return model, chunk_vectors, query_vectors



