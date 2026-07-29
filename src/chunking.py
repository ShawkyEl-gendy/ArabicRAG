import re
import numpy as np
import pandas as pd
from prepare_data import load_dataset
from sentence_transformers import SentenceTransformer
from   langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from config import (
    DATASET_PATH,
    CHUNK_STRATEGY,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_SIZE,
    SIMILARITY_THRESHOLD,

)



def document_chunking(text):
    """
    Return entire document as one chunk.
    """
    return [text]


def fixed_chunking(
    text,
    chunk_size=500,
    overlap=50,
    min_chunk_size=75,
):
    """
    Fixed-size overlapping chunks.

    Parameters
    ----------
    text : str
        Input document.

    chunk_size : int
        Maximum chunk size.

    overlap : int
        Overlap between consecutive chunks.

    min_chunk_size : int
        Merge the last chunk with the previous one if it is
        smaller than this size.

    Returns
    -------
    list[str]
    """

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end])

        start += chunk_size - overlap

    # Merge the final small chunk
    if (
        len(chunks) > 1
        and len(chunks[-1]) < min_chunk_size
    ):
        chunks[-2] += chunks[-1]
        chunks.pop()

    return chunks



def recursive_chunking(
    text,
    chunk_size=500,
    overlap=50,
    min_chunk_size=75,
):
    """
    Recursive character chunking.

    Parameters
    ----------
    text : str
        Input document.

    chunk_size : int
        Maximum chunk size.

    overlap : int
        Overlap between chunks.

    min_chunk_size : int
        Merge chunks smaller than this with the previous chunk.

    Returns
    -------
    list[str]
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=[
            "\n\n",
            "\n",
            ".",
            " ",
            "",
        ],
    )

    chunks = splitter.split_text(text)

    if not chunks:
        return []

    merged_chunks = [chunks[0]]

    for chunk in chunks[1:]:

        if len(chunk.strip()) < min_chunk_size:
            merged_chunks[-1] += " " + chunk
        else:
            merged_chunks.append(chunk)

    return merged_chunks


SPLIT_PATTERN = re.compile(r"[.!؟؛:\n]+")
MODEL = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    local_files_only=True,
)




def semantic_chunking(
    text,
    similarity_threshold=0.65,
    chunk_size=500,
    min_chunk_size=75,
):
    """
    Split text into semantically coherent chunks while ensuring that
    no final chunk is smaller than min_chunk_size.
    """

    if not text or not text.strip():
        return []

    sentences = [
        s.strip()
        for s in SPLIT_PATTERN.split(text)
        if s.strip()
    ]

    if len(sentences) == 1:
        return [text] if len(text) >= min_chunk_size else []

    embeddings = MODEL.encode(
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    chunks = []

    current_sentences = []
    current_length = 0
    embedding_sum = None
    sentence_count = 0

    for sentence, embedding in zip(sentences, embeddings):

        sentence_length = len(sentence)

        if not current_sentences:
            current_sentences.append(sentence)
            current_length = sentence_length
            embedding_sum = embedding.copy()
            sentence_count = 1
            continue

        centroid = embedding_sum / sentence_count
        centroid /= np.linalg.norm(centroid) + 1e-8

        similarity = float(np.dot(centroid, embedding))

        should_split = (
            (similarity < similarity_threshold and current_length >= min_chunk_size)
            or current_length >= chunk_size
        )

        if should_split:
            chunks.append(" ".join(current_sentences))

            current_sentences = [sentence]
            current_length = sentence_length
            embedding_sum = embedding.copy()
            sentence_count = 1
        else:
            current_sentences.append(sentence)
            current_length += sentence_length + 1
            embedding_sum += embedding
            sentence_count += 1

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    # --------------------------------------------------
    # Post-processing: merge small chunks with previous
    # --------------------------------------------------
    if not chunks:
        return []

    merged_chunks = []

    for chunk in chunks:

        if len(chunk) < min_chunk_size and merged_chunks:
            merged_chunks[-1] += " " + chunk
        else:
            merged_chunks.append(chunk)

    return merged_chunks




#########################################################################


def chunk_text(
    text,
    strategy=CHUNK_STRATEGY,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP,
    min_chunk_size=MIN_CHUNK_SIZE,
    similarity_threshold=SIMILARITY_THRESHOLD,
    
):
   
    if strategy == "document":
        return document_chunking(text)

    elif strategy == "fixed":
        return fixed_chunking(
            text,
            chunk_size,
            overlap,
            min_chunk_size,
        )

    elif strategy == "recursive":
        return recursive_chunking(
            text,
            chunk_size,
            overlap,
            min_chunk_size
        )

    elif strategy == "semantic":
        return semantic_chunking(
            text,
            similarity_threshold,
            chunk_size,
            min_chunk_size,
        ) 


    else:
        raise ValueError(
            f"Unknown strategy: {strategy}"
        )



def create_chunk_dataframe(
    df,
    strategy=CHUNK_STRATEGY,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP,
    min_chunk_size=MIN_CHUNK_SIZE,
    similarity_threshold=SIMILARITY_THRESHOLD,
    
):
    """
    Convert documents into chunks.
    """

    rows = []
    print(f"Chunking documents using {strategy} strategy...")
    for _, row in tqdm(df.iterrows(), total=len(df)):

        chunks = chunk_text(
            text=row["context"],
            strategy=strategy,
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_size=min_chunk_size,
            similarity_threshold=similarity_threshold,
            
        )
        for chunk_idx, chunk in enumerate(chunks):

            rows.append({
                "chunk_id":
                    f"{row['document_id']}_{chunk_idx}",

                "document_id":
                    row["document_id"],

                "question":
                    row["question"],

                "query_id":
                    row["query_id"],

                "answer":
                    row["answer"],

                "chunk_text":
                    chunk
            })

    return pd.DataFrame(rows)



if __name__ == "__main__":
    
    df = load_dataset(DATASET_PATH).head(10)
    print("Number of documents loaded:", len(df))

    # Create chunks
    df = create_chunk_dataframe(df)
    print("Number of chunks created:", len(df))
    print(df)
    file_name = f"{CHUNK_STRATEGY}_test.csv"
    df.to_csv(
    file_name,
    index=False,
    encoding="utf-8-sig",
)
