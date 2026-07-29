import pandas as pd
import re
from preprocess_arabic_data_class import ArabicPreprocessor
from nltk.corpus import stopwords
arabic_stopwords = stopwords.words('arabic')


def load_dataset(file_path):
    """
    Load dataset from JSON file into a DataFrame.
    """

    df = pd.read_json(
    file_path,
    lines=True
    )

    return df




preprocessor = ArabicPreprocessor(
    # Remove web noise
    remove_html_markup=True,
    replace_urls_emails_mentions=True,

    # Arabic normalization
    strip_tashkeel=True,
    strip_tatweel=True,
    strip_extended_arabic_and_quranic_symbols=True,
    standarize_non_traditional_ar_chars=True,

    # Numbers
    map_hindi_numbers_to_arabic=True,
    seperate_nums_and_words=True,

    # Language preservation
    keep_latten=True,
    keep_all_non_arabic_letters=False,

    # Emojis and noise
    keep_emojis=False,
    remove_non_digit_repetition=True,

    # Punctuation handling
    seperate_nums_from_pucks=False,
    seperate_words_from_pucks=True,
)


def clean_text(text):
    text = preprocessor.preprocess(text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_dataset(df):

    # Clean text columns
    df["context"] = df["context"].apply(clean_text)
    df["question"] = df["question"].apply(clean_text)
    df["answer"] = df["answer"].apply(clean_text)

    return df






def remove_duplicate_documents(df):
    """
    Remove duplicate contexts and create document IDs.
    """

    df = (
        df
        .drop_duplicates(subset=["context"])
        .reset_index(drop=True)
        .copy()
    )
    
    return df


def add_document_and_query_ids(df):
    """
      Add unique document and query IDs to the DataFrame.
    """

    
    df["document_id"] = [
        f"doc_{i}"
        for i in range(len(df))
    ]

    df["query_id"] = [
        int(i)
        for i in range(len(df))
    ]
    
    return df



if __name__ == "__main__":

    df = load_dataset("./data/ALRAGE-processed.jsonl")
    print("shape of original df:", df.shape)
    df = remove_duplicate_documents(df)
    print("shape after removing duplicates:", df.shape)

    df = add_document_and_query_ids(df)

    df = clean_dataset(df)

    df.to_json(
        "./data/ALRAGE-filtered.jsonl",
        orient="records",
        lines=True,
        force_ascii=False
    )

    df = load_dataset("./data/ALRAGE-filtered.jsonl")
    print("Filtered Documents head:", df.head())
    print("Columns:", df.columns.tolist())


    





