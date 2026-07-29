import pandas as pd
from tqdm import tqdm
from chunking import create_chunk_dataframe
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from prepare_data import load_dataset
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


def load_embedding_model():

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    return model

def create_embeddings(
    texts,
    model
):

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    return embeddings


def load_chroma_client(db_path):
    """
    Create or load a persistent Chroma database.
    """

    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(
            anonymized_telemetry=False
        )
    )

    return client

def create_collection(
    client,
    collection_name
):
    """
    Create (or load) a Chroma collection.
    """

    collection = client.get_or_create_collection(
        name=collection_name
    )

    return collection






def index_chunks(
    chunk_df,
    embeddings,
    collection,
):
    """
    Index pre-computed embeddings into ChromaDB.

    Parameters
    ----------
    chunk_df : pandas.DataFrame
        DataFrame containing chunk information.

    embeddings : list or np.ndarray
        Pre-computed embeddings corresponding to each row in chunk_df.

    collection : chromadb.Collection
        ChromaDB collection.

    use_metadata : bool, default=False
        False:
            Store only the evaluation metadata.

        True:
            Store evaluation metadata together with additional metadata
            (NER, keywords, summaries, etc.).
    """

    for row, embedding in tqdm(
        zip(chunk_df.itertuples(index=False), embeddings),
        total=len(chunk_df),
        desc="Indexing",
    ):

        metadata = {
            "document_id": row.document_id,
            "chunk_id": row.chunk_id,
            "question": row.question,
            "answer": row.answer,
        }


        collection.add(
            ids=[str(row.chunk_id)],
            documents=[row.chunk_text],      # Store original chunk
            embeddings=[embedding.tolist()],
            metadatas=[metadata],
        )



if __name__ == "__main__":

    # Load dataset
    df = load_dataset(DATASET_PATH)
    print("Number of documents loaded:", len(df))

    # Create chunks
    df = create_chunk_dataframe(df)
    print("Number of chunks created:", len(df))


    # Load embedding model
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cuda"      # or "cpu"
    )


    # Generate embeddings
    embeddings = create_embeddings(
        texts=df["chunk_text"],
        model=embedding_model,
    )

    # Load Chroma
    client = load_chroma_client(DB_PATH)

    # remove old collection if it exists
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        print(f"Collection {COLLECTION_NAME} does not exist. Creating a new one.")
        pass

    # Get or create collection
    collection = create_collection(
        client=client,
        collection_name=COLLECTION_NAME,
    )

    # Index chunks
    index_chunks(
        chunk_df=df,
        embeddings=embeddings,
        collection=collection,
        
    )


    print("\nVector database created successfully!")
    print(f"Number of vectors: {collection.count()}")
